from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import chat_service
from app.services.llm.resilience import (
    LLMAllProvidersFailedError,
    LLMAuthenticationError,
    LLMServiceError,
)

router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        200: {
            "description": "JSON ChatResponse (if stream=false) or text/event-stream SSE chunks (if stream=true)"
        },
        422: {"description": "Validation Error"},
        502: {"description": "LLM Provider Authentication Error"},
        503: {"description": "All LLM Providers Unavailable"},
    },
)
async def chat_endpoint(payload: ChatRequest):
    """
    Unified chat endpoint supporting both standard JSON and real-time SSE streaming.
    - Set `"stream": false` (default) for a standard JSON response.
    - Set `"stream": true` for real-time Server-Sent Events (SSE) token streaming.
    """
    if payload.stream:
        return StreamingResponse(
            chat_service.stream_response(payload),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        reply, provider, model = await chat_service.get_response(payload)
        return ChatResponse(response=reply, provider=provider, model=model)
    except LLMAllProvidersFailedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"All configured LLM providers failed: {exc}",
        ) from exc
    except LLMAuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM authentication error: {exc}",
        ) from exc
    except LLMServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM service error: {exc}",
        ) from exc

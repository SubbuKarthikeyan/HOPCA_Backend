import json
import logging
from typing import AsyncGenerator, Dict, List, Tuple

from app.schemas.chat import ChatRequest, StreamChunk
from app.services.llm.llm_service import llm_service
from app.services.llm.resilience import LLMServiceError

logger = logging.getLogger("app.services.chat")


class ChatService:
    """High-level chat coordinator formatting messages and invoking the resilient LLM engine."""

    def format_messages(self, request: ChatRequest) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        if request.history:
            for turn in request.history:
                messages.append({"role": turn.role, "content": turn.content})

        messages.append({"role": "user", "content": request.message})
        return messages

    async def get_response(self, request: ChatRequest) -> Tuple[str, str, str]:
        """Process chat request and return (response_text, provider_name, model_name)."""
        messages = self.format_messages(request)
        return await llm_service.generate(
            messages=messages,
            preferred_provider=request.preferred_provider,
            model=request.model,
            temperature=request.temperature or 0.7,
        )

    async def stream_response(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """
        Process chat request as Server-Sent Events (SSE) format string:
        data: {"chunk": "...", "provider": "...", "model": "...", "is_final": false}\n\n
        """
        messages = self.format_messages(request)
        last_provider = "unknown"
        last_model = "unknown"

        try:
            async for chunk_text, provider_name, model_name in llm_service.stream_generate(
                messages=messages,
                preferred_provider=request.preferred_provider,
                model=request.model,
                temperature=request.temperature or 0.7,
            ):
                last_provider = provider_name
                last_model = model_name
                event_data = StreamChunk(
                    chunk=chunk_text,
                    provider=provider_name,
                    model=model_name,
                    is_final=False,
                )
                yield f"data: {event_data.model_dump_json()}\n\n"

            # Final completion event
            final_event = StreamChunk(
                chunk="",
                provider=last_provider,
                model=last_model,
                is_final=True,
            )
            yield f"data: {final_event.model_dump_json()}\n\n"
        except LLMServiceError as exc:
            logger.error(f"Error during stream generation: {exc}")
            error_event = StreamChunk(
                chunk="",
                provider=last_provider,
                model=last_model,
                is_final=True,
                error=str(exc),
            )
            yield f"data: {error_event.model_dump_json()}\n\n"


chat_service = ChatService()

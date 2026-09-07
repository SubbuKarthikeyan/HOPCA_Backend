import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import chat
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app.main")

app = FastAPI(
    title="Hospital Operations & Patient Coordination Agent",
    version="0.1.0",
    description="Phase 1: FastAPI + Multi-Provider LLM Engine (Groq, Gemini, Mistral, DeepSeek) with Retry, Fallback & Streaming",
)

# Allow Cross-Origin Requests for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])


@app.get("/health", tags=["health"])
def health_check():
    """Liveness probe verifying API health and configured providers."""
    return {
        "status": "healthy",
        "app_env": settings.app_env,
        "configured_providers": [
            p for p in ["groq", "gemini", "mistral", "deepseek"]
            if getattr(settings, f"{p}_api_key", None)
        ],
    }


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled server error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again."},
    )

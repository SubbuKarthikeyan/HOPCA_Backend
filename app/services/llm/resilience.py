import asyncio
import logging
import random
from typing import Any, Callable, Coroutine, Optional, TypeVar
import httpx

logger = logging.getLogger("app.services.llm.resilience")

T = TypeVar("T")


class LLMServiceError(Exception):
    """Base exception for all LLM service failures."""
    pass


class LLMConnectionError(LLMServiceError):
    """Raised when connection to LLM host fails."""
    pass


class LLMRateLimitError(LLMServiceError):
    """Raised when provider rate limit is exceeded."""
    pass


class LLMAuthenticationError(LLMServiceError):
    """Raised when provider authentication fails (e.g. invalid API key)."""
    pass


class LLMAllProvidersFailedError(LLMServiceError):
    """Raised when all candidate providers in fallback chain have failed."""
    pass


def is_retryable_error(exc: Exception) -> bool:
    """Determine whether an error is transient and safe to retry."""
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        # 429 (Rate Limit) and 5xx (Server Errors) are retryable
        if exc.response.status_code == 429 or 500 <= exc.response.status_code < 600:
            return True
    if isinstance(exc, (LLMConnectionError, LLMRateLimitError)):
        return True
    return False


async def retry_async(
    func: Callable[..., Coroutine[Any, Any, T]],
    max_retries: int = 2,
    backoff_factor: float = 1.5,
    provider_name: str = "provider",
    *args: Any,
    **kwargs: Any,
) -> T:
    """
    Executes an async function with exponential backoff and jitter for transient errors.
    """
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as exc:
            last_error = exc
            if attempt < max_retries and is_retryable_error(exc):
                # Calculate exponential delay with small random jitter
                delay = (backoff_factor ** attempt) + random.uniform(0.1, 0.5)
                logger.warning(
                    f"[{provider_name}] Attempt {attempt + 1}/{max_retries + 1} failed with {exc.__class__.__name__}: {exc}. "
                    f"Retrying in {delay:.2f}s..."
                )
                await asyncio.sleep(delay)
            else:
                logger.error(f"[{provider_name}] Final attempt {attempt + 1} failed: {exc}")
                break

    if last_error is not None:
        raise last_error
    raise LLMServiceError(f"[{provider_name}] Unknown failure during execution.")

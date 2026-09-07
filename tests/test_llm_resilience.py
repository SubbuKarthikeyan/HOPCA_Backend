import pytest
from unittest.mock import AsyncMock, MagicMock
import httpx

from app.services.llm.llm_service import LLMService
from app.services.llm.providers import BaseLLMProvider
from app.services.llm.resilience import (
    LLMAllProvidersFailedError,
    LLMConnectionError,
    LLMRateLimitError,
    retry_async,
)


class MockProvider(BaseLLMProvider):
    def __init__(self, name: str, available: bool = True, default_model: str = "test-model"):
        self.name = name
        self._available = available
        self._default_model = default_model
        self.generate_mock = AsyncMock()
        self.stream_mock = MagicMock()

    def is_available(self) -> bool:
        return self._available

    def get_default_model(self) -> str:
        return self._default_model

    async def generate(self, messages, model=None, temperature=0.7, **kwargs):
        return await self.generate_mock(messages, model=model, temperature=temperature, **kwargs)

    async def stream_generate(self, messages, model=None, temperature=0.7, **kwargs):
        async for chunk in self.stream_mock(messages, model=model, temperature=temperature, **kwargs):
            yield chunk


@pytest.mark.asyncio
async def test_retry_async_transient_failure_then_success():
    call_count = 0

    async def flaky_operation():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise LLMConnectionError("Network glitch")
        return "Success on retry"

    result = await retry_async(
        flaky_operation,
        max_retries=2,
        backoff_factor=0.01,
        provider_name="test_provider",
    )
    assert result == "Success on retry"
    assert call_count == 2


@pytest.mark.asyncio
async def test_fallback_when_primary_fails():
    p1 = MockProvider("groq", available=True)
    p1.generate_mock.side_effect = LLMRateLimitError("Rate limit exceeded")

    p2 = MockProvider("gemini", available=True)
    p2.generate_mock.return_value = "Gemini fallback response"

    service = LLMService(providers={"groq": p1, "gemini": p2})

    response, provider_used, model_used = await service.generate(
        messages=[{"role": "user", "content": "hi"}],
        preferred_provider="groq",
    )

    assert response == "Gemini fallback response"
    assert provider_used == "gemini"


@pytest.mark.asyncio
async def test_all_providers_fail_raises_error():
    p1 = MockProvider("groq", available=True)
    p1.generate_mock.side_effect = LLMRateLimitError("Rate limit exceeded")

    p2 = MockProvider("gemini", available=True)
    p2.generate_mock.side_effect = LLMConnectionError("Cannot connect")

    service = LLMService(providers={"groq": p1, "gemini": p2})

    with pytest.raises(LLMAllProvidersFailedError):
        await service.generate(
            messages=[{"role": "user", "content": "hi"}],
            preferred_provider="groq",
        )


@pytest.mark.asyncio
async def test_streaming_fallback_on_initial_failure():
    p1 = MockProvider("groq", available=True)

    async def failing_stream(*args, **kwargs):
        raise LLMConnectionError("Groq stream connection failed")
        yield "never"

    p1.stream_generate = failing_stream

    p2 = MockProvider("gemini", available=True)

    async def successful_stream(*args, **kwargs):
        yield "Gemini"
        yield " stream"
        yield " response"

    p2.stream_generate = successful_stream

    service = LLMService(providers={"groq": p1, "gemini": p2})

    chunks = []
    async for chunk, prov, model in service.stream_generate(
        messages=[{"role": "user", "content": "hi"}],
        preferred_provider="groq",
    ):
        chunks.append(chunk)

    assert "".join(chunks) == "Gemini stream response"

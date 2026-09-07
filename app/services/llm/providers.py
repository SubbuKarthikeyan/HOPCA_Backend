import json
import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional
import httpx

from app.core.config import settings
from app.services.llm.resilience import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMServiceError,
)

logger = logging.getLogger("app.services.llm.providers")


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    name: str

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is configured and available for use."""
        pass

    @abstractmethod
    def get_default_model(self) -> str:
        """Return default model name for this provider."""
        pass

    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """Generate a complete text response."""
        pass

    @abstractmethod
    async def stream_generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens chunk by chunk."""
        pass

    def _handle_http_error(self, exc: httpx.HTTPStatusError) -> None:
        status = exc.response.status_code
        err_msg = f"{self.name} returned HTTP {status}: {exc.response.text}"
        if status in (401, 403):
            raise LLMAuthenticationError(err_msg) from exc
        if status == 429:
            raise LLMRateLimitError(err_msg) from exc
        if 500 <= status < 600:
            raise LLMConnectionError(err_msg) from exc
        raise LLMServiceError(err_msg) from exc


class OpenAICompatibleProvider(BaseLLMProvider):
    """Generic provider for OpenAI-compatible REST APIs (Groq, Mistral, DeepSeek)."""

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: Optional[str],
        default_model: str,
        timeout: float = 45.0,
    ):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_model = default_model
        self.timeout = timeout

    def is_available(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def get_default_model(self) -> str:
        return self.default_model

    def _get_headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise LLMAuthenticationError(f"Missing API key for provider '{self.name}'.")
        return {
            "Authorization": f"Bearer {self.api_key.strip()}",
            "Content-Type": "application/json",
        }

    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        target_model = model or self.default_model
        payload = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        url = f"{self.base_url}/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, json=payload, headers=self._get_headers())
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return content.strip() if content else ""
        except httpx.ConnectError as exc:
            raise LLMConnectionError(f"Could not connect to {self.name}: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise LLMConnectionError(f"Request to {self.name} timed out: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            self._handle_http_error(exc)
        except Exception as exc:
            raise LLMServiceError(f"Unexpected error in {self.name}: {exc}") from exc

    async def stream_generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        target_model = model or self.default_model
        payload = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        url = f"{self.base_url}/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", url, json=payload, headers=self._get_headers()) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        line_str = line.strip()
                        if line_str.startswith("data: "):
                            data_str = line_str[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk_json = json.loads(data_str)
                                choices = chunk_json.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                continue
        except httpx.ConnectError as exc:
            raise LLMConnectionError(f"Could not connect to {self.name}: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise LLMConnectionError(f"Stream request to {self.name} timed out: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            self._handle_http_error(exc)
        except Exception as exc:
            raise LLMServiceError(f"Unexpected stream error in {self.name}: {exc}") from exc


class GeminiNativeProvider(BaseLLMProvider):
    """Google Gemini Provider using Google AI REST API."""

    def __init__(
        self,
        api_key: Optional[str],
        default_model: str = "gemini-1.5-flash",
        timeout: float = 45.0,
    ):
        self.name = "gemini"
        self.api_key = api_key
        self.default_model = default_model
        self.timeout = timeout
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    def is_available(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def get_default_model(self) -> str:
        return self.default_model

    def _format_messages_to_contents(self, messages: List[Dict[str, str]]) -> tuple[Optional[str], List[Dict[str, Any]]]:
        system_instruction: Optional[str] = None
        contents: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_instruction = content
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})
            else:
                contents.append({"role": "user", "parts": [{"text": content}]})
        return system_instruction, contents

    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        if not self.api_key:
            raise LLMAuthenticationError("Missing Gemini API key.")
        target_model = model or self.default_model
        system_instruction, contents = self._format_messages_to_contents(messages)
        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": temperature},
        }
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        url = f"{self.base_url}/models/{target_model}:generateContent?key={self.api_key.strip()}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return "".join(p.get("text", "") for p in parts).strip()
                return ""
        except httpx.ConnectError as exc:
            raise LLMConnectionError(f"Could not connect to Gemini: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise LLMConnectionError(f"Request to Gemini timed out: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            self._handle_http_error(exc)
        except Exception as exc:
            raise LLMServiceError(f"Unexpected error in Gemini: {exc}") from exc

    async def stream_generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        if not self.api_key:
            raise LLMAuthenticationError("Missing Gemini API key.")
        target_model = model or self.default_model
        system_instruction, contents = self._format_messages_to_contents(messages)
        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": temperature},
        }
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        url = f"{self.base_url}/models/{target_model}:streamGenerateContent?alt=sse&key={self.api_key.strip()}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", url, json=payload) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        line_str = line.strip()
                        if line_str.startswith("data: "):
                            data_str = line_str[6:].strip()
                            try:
                                chunk_json = json.loads(data_str)
                                candidates = chunk_json.get("candidates", [])
                                if candidates:
                                    parts = candidates[0].get("content", {}).get("parts", [])
                                    for p in parts:
                                        text = p.get("text", "")
                                        if text:
                                            yield text
                            except json.JSONDecodeError:
                                continue
        except httpx.ConnectError as exc:
            raise LLMConnectionError(f"Could not connect to Gemini: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise LLMConnectionError(f"Stream request to Gemini timed out: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            self._handle_http_error(exc)
        except Exception as exc:
            raise LLMServiceError(f"Unexpected stream error in Gemini: {exc}") from exc


def create_providers_registry() -> Dict[str, BaseLLMProvider]:
    """Instantiate and register all configured LLM provider adapters."""
    return {
        "groq": OpenAICompatibleProvider(
            name="groq",
            base_url="https://api.groq.com/openai/v1",
            api_key=settings.groq_api_key,
            default_model=settings.groq_model,
            timeout=settings.llm_timeout_seconds,
        ),
        "gemini": GeminiNativeProvider(
            api_key=settings.gemini_api_key,
            default_model=settings.gemini_model,
            timeout=settings.llm_timeout_seconds,
        ),
        "mistral": OpenAICompatibleProvider(
            name="mistral",
            base_url="https://api.mistral.ai/v1",
            api_key=settings.mistral_api_key,
            default_model=settings.mistral_model,
            timeout=settings.llm_timeout_seconds,
        ),
        "deepseek": OpenAICompatibleProvider(
            name="deepseek",
            base_url="https://api.deepseek.com",
            api_key=settings.deepseek_api_key,
            default_model=settings.deepseek_model,
            timeout=settings.llm_timeout_seconds,
        ),
    }

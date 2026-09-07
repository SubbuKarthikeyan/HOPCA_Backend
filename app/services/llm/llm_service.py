import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from app.core.config import settings
from app.services.llm.providers import BaseLLMProvider, create_providers_registry
from app.services.llm.resilience import (
    LLMAllProvidersFailedError,
    LLMServiceError,
    retry_async,
)

logger = logging.getLogger("app.services.llm.service")


class LLMService:
    """Orchestrates multi-provider LLM calls with automatic retry and seamless fallback."""

    def __init__(self, providers: Optional[Dict[str, BaseLLMProvider]] = None):
        self.providers = providers if providers is not None else create_providers_registry()

    def get_candidate_providers(self, preferred: Optional[str] = None) -> List[str]:
        """Determine the ordered list of provider names to attempt."""
        candidates: List[str] = []
        # If a preferred provider is specified and available, try it first
        if preferred and preferred in self.providers and self.providers[preferred].is_available():
            candidates.append(preferred)

        # Append remaining providers from configured priority order
        for p_name in settings.llm_provider_order:
            if p_name in self.providers and p_name not in candidates:
                if self.providers[p_name].is_available():
                    candidates.append(p_name)

        # Fallback to any remaining registered available providers
        for p_name, provider in self.providers.items():
            if p_name not in candidates and provider.is_available():
                candidates.append(p_name)

        return candidates

    async def generate(
        self,
        messages: List[Dict[str, str]],
        preferred_provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Tuple[str, str, str]:
        """
        Generate a complete response using the first succeeding provider in the fallback chain.
        Returns: (response_text, provider_name, model_name)
        """
        candidates = self.get_candidate_providers(preferred_provider)
        if not candidates:
            raise LLMAllProvidersFailedError(
                "No LLM providers are configured or available. Please check your API keys in .env."
            )

        errors_encountered: List[str] = []

        for provider_name in candidates:
            provider = self.providers[provider_name]
            target_model = model if (model and preferred_provider == provider_name) else provider.get_default_model()
            logger.info(f"Attempting LLM generation with provider '{provider_name}' (model: '{target_model}')")

            try:
                # Wrap with retry logic for transient errors
                response_text = await retry_async(
                    provider.generate,
                    max_retries=settings.llm_max_retries,
                    backoff_factor=settings.llm_backoff_factor,
                    provider_name=provider_name,
                    messages=messages,
                    model=target_model,
                    temperature=temperature,
                    **kwargs,
                )
                logger.info(f"Successfully generated response via provider '{provider_name}'")
                return response_text, provider_name, target_model
            except Exception as exc:
                err_summary = f"{provider_name} ({exc.__class__.__name__}: {exc})"
                logger.warning(f"Provider '{provider_name}' failed after retries: {exc}. Falling back to next provider...")
                errors_encountered.append(err_summary)

        # If loop completed without returning, all providers failed
        all_errs = "; ".join(errors_encountered)
        raise LLMAllProvidersFailedError(f"All LLM providers in fallback chain failed: {all_errs}")

    async def stream_generate(
        self,
        messages: List[Dict[str, str]],
        preferred_provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncGenerator[Tuple[str, str, str], None]:
        """
        Stream tokens chunk-by-chunk using fallback logic.
        Yields tuples of: (chunk_text, provider_name, model_name)
        """
        candidates = self.get_candidate_providers(preferred_provider)
        if not candidates:
            raise LLMAllProvidersFailedError(
                "No LLM providers are configured or available. Please check your API keys in .env."
            )

        errors_encountered: List[str] = []

        for provider_name in candidates:
            provider = self.providers[provider_name]
            target_model = model if (model and preferred_provider == provider_name) else provider.get_default_model()
            logger.info(f"Attempting LLM stream generation with provider '{provider_name}' (model: '{target_model}')")

            yielded_any = False
            try:
                async for chunk in provider.stream_generate(
                    messages=messages,
                    model=target_model,
                    temperature=temperature,
                    **kwargs,
                ):
                    yielded_any = True
                    yield chunk, provider_name, target_model

                if yielded_any:
                    logger.info(f"Successfully finished streaming from provider '{provider_name}'")
                    return
            except Exception as exc:
                err_summary = f"{provider_name} ({exc.__class__.__name__}: {exc})"
                logger.warning(f"Streaming provider '{provider_name}' failed: {exc}")
                errors_encountered.append(err_summary)

                # If tokens were already yielded to client, we cannot cleanly fallback to another provider mid-stream
                if yielded_any:
                    raise LLMServiceError(
                        f"Stream interrupted after output started on provider '{provider_name}': {exc}"
                    ) from exc

        # If no provider yielded anything
        all_errs = "; ".join(errors_encountered)
        raise LLMAllProvidersFailedError(f"All LLM providers in stream fallback chain failed: {all_errs}")


llm_service = LLMService()

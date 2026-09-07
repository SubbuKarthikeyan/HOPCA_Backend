from typing import Dict, List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    log_level: str = "INFO"

    # API Keys
    groq_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    mistral_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None

    # Configurable Models per Provider
    groq_model: str = "qwen/qwen3.8-27b"
    gemini_model: str = "gemini-1.5-flash"
    mistral_model: str = "mistral-small-latest"
    deepseek_model: str = "deepseek-chat"

    # Fallback and Resilience Configuration
    llm_provider_order: List[str] = ["groq", "gemini", "mistral", "deepseek"]
    llm_timeout_seconds: float = 45.0
    llm_max_retries: int = 2
    llm_backoff_factor: float = 1.5

    def get_model_for_provider(self, provider_name: str) -> str:
        """Get the configured model name for a specific provider."""
        mapping = {
            "groq": self.groq_model,
            "gemini": self.gemini_model,
            "mistral": self.mistral_model,
            "deepseek": self.deepseek_model,
        }
        return mapping.get(provider_name.lower(), self.groq_model)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


settings = Settings()

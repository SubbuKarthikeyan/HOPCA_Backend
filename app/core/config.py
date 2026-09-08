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
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_model: str = "gemini-1.5-flash"
    mistral_model: str = "mistral-small-latest"
    deepseek_model: str = "deepseek-chat"

    # Fallback and Resilience Configuration
    llm_provider_order: List[str] = ["groq", "gemini", "mistral", "deepseek"]
    llm_timeout_seconds: float = 45.0
    llm_max_retries: int = 2
    llm_backoff_factor: float = 1.5

    # ── RAG Pipeline Configuration ──────────────────────────────────────────
    # Knowledge Base
    knowledge_base_dir: str = "../knowledge_base"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 150
    chunking_strategy: str = "semantic"

    # Embeddings (Gemini only)
    embedding_model: str = "gemini-embedding-001"
    embedding_dimension: int = 768
    embedding_version: str = "1.0"
    embedding_batch_size: int = 50

    # Data directory for interim storage (hash registry, etc.)
    data_dir: str = "./data"

    # ── MongoDB Foundation & Storage ─────────────────────────────────────────
    mongo_url: Optional[str] = None
    mongo_db_name: str = "hopca_database"
    mongo_timeout_ms: int = 5000

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


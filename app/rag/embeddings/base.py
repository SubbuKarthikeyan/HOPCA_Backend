"""Base interface for embedding providers."""
from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingProvider(ABC):
    """Abstract base class for text embedding model providers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the embedding model."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Version identifier of the embedding model/configuration."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a batch of text strings."""
        pass

    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """Generate an embedding vector for a single query text."""
        pass

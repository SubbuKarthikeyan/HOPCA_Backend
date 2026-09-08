"""Tests for Embedding Service and Providers (Scenarios 10, 16)."""
from typing import List
import pytest

from app.rag.embeddings.base import BaseEmbeddingProvider
from app.rag.embeddings.embedding_service import EmbeddingService
from app.rag.models import Chunk


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Mock embedding provider for deterministic, fast offline testing."""

    def __init__(self, dimension: int = 768, fail_dim: bool = False):
        self._dim = dimension
        self._fail_dim = fail_dim

    @property
    def model_name(self) -> str:
        return "mock-embedding-model"

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def version(self) -> str:
        return "1.0"

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if self._fail_dim:
            # Return incorrect dimension
            return [[0.1] * (self._dim - 10) for _ in texts]
        return [[float(i) / 100.0] * self._dim for i in range(len(texts))]

    async def embed_query(self, text: str) -> List[float]:
        if self._fail_dim:
            return [0.1] * (self._dim - 10)
        return [0.5] * self._dim


@pytest.mark.asyncio
async def test_scenario_10_embedding_generation():
    """Scenario 10: Generate embeddings for chunks."""
    provider = MockEmbeddingProvider(dimension=768)
    service = EmbeddingService(provider=provider, batch_size=2)

    chunks = [
        Chunk(
            chunk_id="doc1_chunk_0",
            document_id="doc1",
            domain="hospital",
            document_type="guideline",
            title="Hospital",
            chunk_index=0,
            content="Some text about hospital",
            content_hash="abc123",
        ),
        Chunk(
            chunk_id="doc1_chunk_1",
            document_id="doc1",
            domain="hospital",
            document_type="guideline",
            title="Hospital",
            chunk_index=1,
            content="More text about hospital",
            content_hash="def456",
        ),
    ]

    embedded = await service.embed_chunks(chunks)
    assert len(embedded) == 2
    assert len(embedded[0].embedding) == 768
    assert embedded[0].embedding_model == "mock-embedding-model"
    assert embedded[0].embedding_dimension == 768


@pytest.mark.asyncio
async def test_scenario_16_invalid_embedding_dimension_failure():
    """Scenario 16: Invalid embedding dimension -> raises ValueError."""
    provider = MockEmbeddingProvider(dimension=768, fail_dim=True)
    service = EmbeddingService(provider=provider, batch_size=2)

    chunks = [
        Chunk(
            chunk_id="doc1_chunk_0",
            document_id="doc1",
            domain="hospital",
            document_type="guideline",
            title="Hospital",
            chunk_index=0,
            content="Some text",
            content_hash="abc123",
        )
    ]

    with pytest.raises(ValueError, match="dimension mismatch"):
        await service.embed_chunks(chunks)

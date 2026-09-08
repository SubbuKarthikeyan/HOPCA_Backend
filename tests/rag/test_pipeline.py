"""Tests for End-to-End Ingestion Pipeline (Scenario 17)."""
from pathlib import Path
import pytest

from app.rag.embeddings.base import BaseEmbeddingProvider
from app.rag.ingestion.pipeline import IngestionPipeline


class MockPipelineEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, dimension: int = 768):
        self._dim = dimension

    @property
    def model_name(self) -> str:
        return "mock-pipeline-embedder"

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def version(self) -> str:
        return "1.0"

    async def embed_batch(self, texts):
        return [[0.1] * self._dim for _ in texts]

    async def embed_query(self, text):
        return [0.1] * self._dim


@pytest.fixture
def actual_kb_path():
    return Path(__file__).resolve().parent.parent.parent.parent / "knowledge_base"


@pytest.mark.asyncio
async def test_scenario_17_full_pipeline_end_to_end(actual_kb_path, tmp_path):
    """Scenario 17: Full pipeline end-to-end execution, idempotency, and reporting."""
    registry_file = tmp_path / "test_registry.json"
    mock_provider = MockPipelineEmbeddingProvider(dimension=768)

    pipeline = IngestionPipeline(
        kb_path=actual_kb_path,
        registry_path=registry_file,
        embedding_provider=mock_provider,
        use_mongo=False,
    )

    # First run: should scan all files and embed all chunks
    stats1, embedded1 = await pipeline.run()

    assert stats1.status == "SUCCESS"
    assert stats1.documents_scanned >= 12
    assert stats1.documents_valid >= 12
    assert stats1.chunks_total > 0
    assert stats1.new_chunks == stats1.chunks_total
    assert stats1.embeddings_generated == stats1.chunks_total
    assert stats1.embeddings_reused == 0
    assert len(embedded1) == stats1.chunks_total

    # Second run: should recognize everything as unchanged and reuse embeddings
    pipeline2 = IngestionPipeline(
        kb_path=actual_kb_path,
        registry_path=registry_file,
        embedding_provider=mock_provider,
        use_mongo=False,
    )
    stats2, embedded2 = await pipeline2.run()

    assert stats2.status == "SUCCESS"
    assert stats2.documents_scanned >= 12
    assert stats2.chunks_total == stats1.chunks_total
    assert stats2.new_chunks == 0
    assert stats2.changed_chunks == 0
    assert stats2.unchanged_chunks == stats1.chunks_total
    assert stats2.embeddings_generated == 0
    assert stats2.embeddings_reused == stats1.chunks_total
    assert len(embedded2) == stats1.chunks_total

"""Automated tests for HOPCA MongoDB Foundation & RAG Storage (Scenarios 1-11)."""
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
import pytest
import pytest_asyncio
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.core.config import settings
from app.db.init_db import REQUIRED_COLLECTIONS, init_mongodb
from app.db.repositories.chunk_repo import ChunkRepository
from app.db.repositories.document_repo import DocumentRepository
from app.db.repositories.metadata_repo import MetadataRepository
from app.db.repositories.run_repo import IngestionRunRepository
from app.db.session import MongoDBManager
from app.rag.embeddings.base import BaseEmbeddingProvider
from app.rag.ingestion.pipeline import IngestionPipeline


class MockTestEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic mock embedding provider."""

    def __init__(self, dimension: int = 768):
        self._dim = dimension

    @property
    def model_name(self) -> str:
        return "mock-mongo-embedder"

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def version(self) -> str:
        return "1.0"

    async def embed_batch(self, texts):
        return [[float((i + 1) * 0.001)] * self._dim for i in range(len(texts))]

    async def embed_query(self, text):
        return [0.05] * self._dim


@pytest_asyncio.fixture
async def test_db():
    """Function-scoped fixture for test database connected to MongoDB Atlas."""
    if not settings.mongo_url:
        pytest.skip("MONGO_URL not configured")

    db = MongoDBManager.get_database("hopca_test_database")
    yield db


@pytest.fixture(scope="module")
def sample_kb_dir(tmp_path_factory):
    """Create a mini valid knowledge base in a temporary directory shared across module scenarios."""
    kb_dir = tmp_path_factory.mktemp("test_kb")

    # Doc 1: Doctors
    doc1 = kb_dir / "doctors.md"
    doc1.write_text(
        """---
document_id: test_doctors
title: Test Hospital Doctors
domain: doctors
document_type: directory
version: "1.0"
source: HOPCA
status: active
synthetic_data: true
---
## Doctor: DOC-101
Dr. Alice Green - Cardiologist
Available: Mon, Wed, Fri

---

## Doctor: DOC-102
Dr. Bob White - Neurologist
Available: Tue, Thu
""",
        encoding="utf-8",
    )

    # Doc 2: Hospital Info
    doc2 = kb_dir / "hospital.md"
    doc2.write_text(
        """---
document_id: test_hospital
title: Test Hospital Overview
domain: hospital
document_type: guideline
version: "1.0"
source: HOPCA
status: active
synthetic_data: true
---
## HOSP-001: Overview
HOPCA General Hospital is an acute care facility located in Metro City.
""",
        encoding="utf-8",
    )

    return kb_dir


# ──────────────────────────────────────────────────────────────────────────────
# Test Scenarios
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_1_mongodb_connection(test_db):
    """Scenario 1: MongoDB connection and ping."""
    ping_ok = await MongoDBManager.ping(test_db.client)
    assert ping_ok is True


@pytest.mark.asyncio
async def test_2_database_initialization(test_db):
    """Scenario 2: Database initialization runs idempotently."""
    created_indexes = await init_mongodb(test_db)
    assert len(created_indexes) == 4
    for col in REQUIRED_COLLECTIONS:
        assert col in created_indexes


@pytest.mark.asyncio
async def test_3_collection_creation(test_db):
    """Scenario 3: Verify all 4 collections exist in database."""
    collections = await test_db.list_collection_names()
    for col in REQUIRED_COLLECTIONS:
        assert col in collections


@pytest.mark.asyncio
async def test_4_index_creation(test_db):
    """Scenario 4: Verify unique and required indexes."""
    doc_indexes = [idx["name"] async for idx in test_db["rag_documents"].list_indexes()]
    chunk_indexes = [idx["name"] async for idx in test_db["rag_chunks"].list_indexes()]
    run_indexes = [idx["name"] async for idx in test_db["ingestion_runs"].list_indexes()]

    assert "idx_document_id_unique" in doc_indexes
    assert "idx_chunk_id_unique" in chunk_indexes
    assert "idx_run_id_unique" in run_indexes


@pytest.mark.asyncio
async def test_5_initial_ingestion(test_db, sample_kb_dir):
    """Scenario 5: Initial ingestion stores documents, chunks, runs, and metadata in MongoDB."""
    for col in REQUIRED_COLLECTIONS:
        try:
            await test_db[col].drop()
        except Exception:
            pass

    mock_provider = MockTestEmbeddingProvider(dimension=768)
    pipeline = IngestionPipeline(
        kb_path=sample_kb_dir,
        embedding_provider=mock_provider,
        db=test_db,
        use_mongo=True,
    )

    stats, embedded_chunks = await pipeline.run()

    assert stats.status == "SUCCESS"
    assert stats.documents_scanned == 2
    assert stats.chunks_total == 3  # 2 in doctors + 1 in hospital
    assert stats.new_chunks == 3
    assert stats.embeddings_generated == 3
    assert stats.embeddings_reused == 0

    # Verify MongoDB persistence
    doc_repo = DocumentRepository(test_db)
    chunk_repo = ChunkRepository(test_db)
    run_repo = IngestionRunRepository(test_db)
    meta_repo = MetadataRepository(test_db)

    assert await doc_repo.count_documents() == 2
    assert await chunk_repo.count_chunks(status="active") == 3

    latest_run = await run_repo.get_latest_run()
    assert latest_run is not None
    assert latest_run["status"] == "SUCCESS"
    assert latest_run["embeddings_generated"] == 3

    meta = await meta_repo.get_system_metadata()
    assert meta is not None
    assert meta["embedding_model"] == "mock-mongo-embedder"
    assert meta["embedding_dimension"] == 768


@pytest.mark.asyncio
async def test_6_repeated_unchanged_ingestion(test_db, sample_kb_dir):
    """Scenario 6: Repeated unchanged ingestion reuses embeddings from MongoDB."""
    mock_provider = MockTestEmbeddingProvider(dimension=768)
    pipeline = IngestionPipeline(
        kb_path=sample_kb_dir,
        embedding_provider=mock_provider,
        db=test_db,
        use_mongo=True,
    )

    stats, embedded_chunks = await pipeline.run()

    assert stats.status == "SUCCESS"
    assert stats.chunks_total == 3
    assert stats.new_chunks == 0
    assert stats.changed_chunks == 0
    assert stats.unchanged_chunks == 3
    assert stats.embeddings_generated == 0
    assert stats.embeddings_reused == 3


@pytest.mark.asyncio
async def test_7_changed_document(test_db, sample_kb_dir):
    """Scenario 7: Modifying one document updates only affected data in MongoDB."""
    # Modify hospital.md
    doc2 = sample_kb_dir / "hospital.md"
    doc2.write_text(
        """---
document_id: test_hospital
title: Test Hospital Overview
domain: hospital
document_type: guideline
version: "1.1"
source: HOPCA
status: active
synthetic_data: true
---
## HOSP-001: Overview
HOPCA General Hospital is an acute care facility located in Metro City with an expanded 500-bed capacity.
""",
        encoding="utf-8",
    )

    mock_provider = MockTestEmbeddingProvider(dimension=768)
    pipeline = IngestionPipeline(
        kb_path=sample_kb_dir,
        embedding_provider=mock_provider,
        db=test_db,
        use_mongo=True,
    )

    stats, _ = await pipeline.run()

    assert stats.status == "SUCCESS"
    assert stats.documents_changed == 1
    assert stats.changed_chunks == 1
    assert stats.unchanged_chunks == 2
    assert stats.embeddings_generated == 1
    assert stats.embeddings_reused == 2

    # Verify updated content in MongoDB
    chunk_repo = ChunkRepository(test_db)
    hosp_chunks = await chunk_repo.get_chunks_by_document("test_hospital")
    assert len(hosp_chunks) == 1
    assert "expanded 500-bed capacity" in hosp_chunks[0]["content"]


@pytest.mark.asyncio
async def test_8_new_document(test_db, sample_kb_dir):
    """Scenario 8: Adding a new document inserts only new data in MongoDB."""
    doc3 = sample_kb_dir / "services.md"
    doc3.write_text(
        """---
document_id: test_services
title: Test Hospital Services
domain: services
document_type: directory
version: "1.0"
source: HOPCA
status: active
synthetic_data: true
---
## Service: SERV-101
Emergency Trauma Unit available 24/7.
""",
        encoding="utf-8",
    )

    mock_provider = MockTestEmbeddingProvider(dimension=768)
    pipeline = IngestionPipeline(
        kb_path=sample_kb_dir,
        embedding_provider=mock_provider,
        db=test_db,
        use_mongo=True,
    )

    stats, _ = await pipeline.run()

    assert stats.status == "SUCCESS"
    assert stats.new_chunks == 1
    assert stats.unchanged_chunks == 3
    assert stats.embeddings_generated == 1
    assert stats.embeddings_reused == 3

    chunk_repo = ChunkRepository(test_db)
    assert await chunk_repo.count_chunks(status="active") == 4


@pytest.mark.asyncio
async def test_9_removed_document(test_db, sample_kb_dir):
    """Scenario 9: Removing a document marks its chunks as removed in MongoDB."""
    # Delete services.md
    doc3 = sample_kb_dir / "services.md"
    if doc3.exists():
        doc3.unlink()

    mock_provider = MockTestEmbeddingProvider(dimension=768)
    pipeline = IngestionPipeline(
        kb_path=sample_kb_dir,
        embedding_provider=mock_provider,
        db=test_db,
        use_mongo=True,
    )

    stats, _ = await pipeline.run()

    assert stats.status == "SUCCESS"
    assert stats.removed_chunks == 1

    chunk_repo = ChunkRepository(test_db)
    assert await chunk_repo.count_chunks(status="active") == 3
    assert await chunk_repo.count_chunks(status="removed") == 1


@pytest.mark.asyncio
async def test_10_duplicate_prevention(test_db, sample_kb_dir):
    """Scenario 10: Upserting existing chunks prevents duplicate records."""
    chunk_repo = ChunkRepository(test_db)
    initial_count = await chunk_repo.count_chunks(status="active")

    # Run ingestion again
    mock_provider = MockTestEmbeddingProvider(dimension=768)
    pipeline = IngestionPipeline(
        kb_path=sample_kb_dir,
        embedding_provider=mock_provider,
        db=test_db,
        use_mongo=True,
    )
    await pipeline.run()

    final_count = await chunk_repo.count_chunks(status="active")
    assert initial_count == final_count == 3


@pytest.mark.asyncio
async def test_11_stored_embedding_metadata(test_db):
    """Scenario 11: Verify vector dimensions, model name, and UTC timestamps."""
    chunk_repo = ChunkRepository(test_db)
    chunks = await chunk_repo.get_all_active_chunks()

    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk["embedding"]) == 768
        assert chunk["embedding_model"] == "mock-mongo-embedder"
        assert chunk["embedding_dimension"] == 768
        assert chunk["embedding_version"] == "1.0"
        assert isinstance(chunk["created_at"], datetime)
        assert isinstance(chunk["updated_at"], datetime)

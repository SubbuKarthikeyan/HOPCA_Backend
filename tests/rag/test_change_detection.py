"""Tests for Change Detection and Hash Registry (Scenarios 11, 12, 13, 14, 15)."""
from app.rag.ingestion.change_detector import ChangeDetector
from app.rag.models import Chunk, EmbeddedChunk


def test_scenario_11_unchanged_chunks_reused(tmp_path):
    """Scenario 11: Second run with unchanged content -> chunks recognized as unchanged and cached embeddings reused."""
    registry_file = tmp_path / "registry.json"
    detector = ChangeDetector(registry_path=registry_file, embedding_dimension=768)

    c1 = Chunk(
        chunk_id="doc1_chunk_0",
        document_id="doc1",
        domain="hospital",
        document_type="guideline",
        title="Hospital",
        chunk_index=0,
        content="Content 1",
        content_hash="hash1",
    )

    # First run
    new_chunks, changed_chunks, unchanged_embedded, removed = detector.detect_changes([c1])
    assert len(new_chunks) == 1

    # Simulate saving after embedding
    embedded_c1 = EmbeddedChunk(
        chunk_id="doc1_chunk_0",
        document_id="doc1",
        domain="hospital",
        document_type="guideline",
        title="Hospital",
        chunk_index=0,
        content="Content 1",
        content_hash="hash1",
        embedding=[0.1] * 768,
        embedding_model="text-embedding-004",
        embedding_dimension=768,
        embedding_version="1.0",
    )
    detector.update_registry([embedded_c1])

    # Second run with same chunk
    new_chunks2, changed_chunks2, unchanged_embedded2, removed2 = detector.detect_changes([c1])
    assert len(new_chunks2) == 0
    assert len(changed_chunks2) == 0
    assert len(unchanged_embedded2) == 1
    assert len(removed2) == 0
    assert unchanged_embedded2[0].chunk_id == "doc1_chunk_0"


def test_scenario_12_modify_one_doc_targeted_reembed(tmp_path):
    """Scenario 12: Modify one doc -> only affected chunk is marked as changed."""
    registry_file = tmp_path / "registry.json"
    detector = ChangeDetector(registry_path=registry_file, embedding_dimension=768)

    # Pre-populate registry with doc1 and doc2
    emb1 = EmbeddedChunk(
        chunk_id="doc1_chunk_0",
        document_id="doc1",
        domain="hospital",
        document_type="guideline",
        title="Hospital",
        chunk_index=0,
        content="Doc1 Content",
        content_hash="hash_doc1",
        embedding=[0.1] * 768,
        embedding_model="text-embedding-004",
        embedding_dimension=768,
        embedding_version="1.0",
    )
    emb2 = EmbeddedChunk(
        chunk_id="doc2_chunk_0",
        document_id="doc2",
        domain="doctors",
        document_type="directory",
        title="Doctors",
        chunk_index=0,
        content="Doc2 Content",
        content_hash="hash_doc2",
        embedding=[0.2] * 768,
        embedding_model="text-embedding-004",
        embedding_dimension=768,
        embedding_version="1.0",
    )
    detector.update_registry([emb1, emb2])

    # Modify doc2 content
    c1_same = Chunk(
        chunk_id="doc1_chunk_0",
        document_id="doc1",
        domain="hospital",
        document_type="guideline",
        title="Hospital",
        chunk_index=0,
        content="Doc1 Content",
        content_hash="hash_doc1",
    )
    c2_modified = Chunk(
        chunk_id="doc2_chunk_0",
        document_id="doc2",
        domain="doctors",
        document_type="directory",
        title="Doctors",
        chunk_index=0,
        content="Doc2 Modified Content",
        content_hash="hash_doc2_MODIFIED",
    )

    new_chunks, changed_chunks, unchanged_embedded, removed = detector.detect_changes([c1_same, c2_modified])
    assert len(new_chunks) == 0
    assert len(changed_chunks) == 1
    assert changed_chunks[0].chunk_id == "doc2_chunk_0"
    assert len(unchanged_embedded) == 1
    assert unchanged_embedded[0].chunk_id == "doc1_chunk_0"


def test_scenario_13_add_new_doc_only_new_embedded(tmp_path):
    """Scenario 13: Add new doc -> only new chunk is marked as NEW."""
    registry_file = tmp_path / "registry.json"
    detector = ChangeDetector(registry_path=registry_file, embedding_dimension=768)

    emb1 = EmbeddedChunk(
        chunk_id="doc1_chunk_0",
        document_id="doc1",
        domain="hospital",
        document_type="guideline",
        title="Hospital",
        chunk_index=0,
        content="Doc1 Content",
        content_hash="hash_doc1",
        embedding=[0.1] * 768,
        embedding_model="text-embedding-004",
        embedding_dimension=768,
        embedding_version="1.0",
    )
    detector.update_registry([emb1])

    c1_same = Chunk(
        chunk_id="doc1_chunk_0",
        document_id="doc1",
        domain="hospital",
        document_type="guideline",
        title="Hospital",
        chunk_index=0,
        content="Doc1 Content",
        content_hash="hash_doc1",
    )
    c3_new = Chunk(
        chunk_id="doc3_chunk_0",
        document_id="doc3",
        domain="faq",
        document_type="faq",
        title="FAQ",
        chunk_index=0,
        content="Doc3 Brand New FAQ",
        content_hash="hash_doc3",
    )

    new_chunks, changed_chunks, unchanged_embedded, removed = detector.detect_changes([c1_same, c3_new])
    assert len(new_chunks) == 1
    assert new_chunks[0].chunk_id == "doc3_chunk_0"
    assert len(changed_chunks) == 0
    assert len(unchanged_embedded) == 1


def test_scenario_14_deleted_source_detected_as_removed(tmp_path):
    """Scenario 14: Deleted source -> stale chunk detected as REMOVED."""
    registry_file = tmp_path / "registry.json"
    detector = ChangeDetector(registry_path=registry_file, embedding_dimension=768)

    emb1 = EmbeddedChunk(
        chunk_id="doc1_chunk_0",
        document_id="doc1",
        domain="hospital",
        document_type="guideline",
        title="Hospital",
        chunk_index=0,
        content="Doc1 Content",
        content_hash="hash_doc1",
        embedding=[0.1] * 768,
        embedding_model="text-embedding-004",
        embedding_dimension=768,
        embedding_version="1.0",
    )
    emb2 = EmbeddedChunk(
        chunk_id="doc2_chunk_0",
        document_id="doc2",
        domain="doctors",
        document_type="directory",
        title="Doctors",
        chunk_index=0,
        content="Doc2 Content",
        content_hash="hash_doc2",
        embedding=[0.2] * 768,
        embedding_model="text-embedding-004",
        embedding_dimension=768,
        embedding_version="1.0",
    )
    detector.update_registry([emb1, emb2])

    # Now doc2 was deleted, only doc1 is present
    c1 = Chunk(
        chunk_id="doc1_chunk_0",
        document_id="doc1",
        domain="hospital",
        document_type="guideline",
        title="Hospital",
        chunk_index=0,
        content="Doc1 Content",
        content_hash="hash_doc1",
    )

    new_chunks, changed_chunks, unchanged_embedded, removed = detector.detect_changes([c1])
    assert len(removed) == 1
    assert "doc2_chunk_0" in removed


def test_scenario_15_model_or_dimension_change_triggers_full_reembed(tmp_path):
    """Scenario 15: Model, dimension, or version change -> re-embed all chunks."""
    registry_file = tmp_path / "registry.json"
    detector = ChangeDetector(
        registry_path=registry_file,
        embedding_model="text-embedding-004",
        embedding_dimension=768,
        embedding_version="1.0",
    )

    emb1 = EmbeddedChunk(
        chunk_id="doc1_chunk_0",
        document_id="doc1",
        domain="hospital",
        document_type="guideline",
        title="Hospital",
        chunk_index=0,
        content="Doc1 Content",
        content_hash="hash_doc1",
        embedding=[0.1] * 768,
        embedding_model="text-embedding-004",
        embedding_dimension=768,
        embedding_version="1.0",
    )
    detector.update_registry([emb1])

    # Now instantiate a new detector with updated model
    detector_v2 = ChangeDetector(
        registry_path=registry_file,
        embedding_model="text-embedding-005",
        embedding_dimension=768,
        embedding_version="2.0",
    )

    c1 = Chunk(
        chunk_id="doc1_chunk_0",
        document_id="doc1",
        domain="hospital",
        document_type="guideline",
        title="Hospital",
        chunk_index=0,
        content="Doc1 Content",
        content_hash="hash_doc1",
    )

    new_chunks, changed_chunks, unchanged_embedded, removed = detector_v2.detect_changes([c1])
    # Should treat existing chunk as changed to force re-embedding under new model
    assert len(changed_chunks) == 1
    assert len(unchanged_embedded) == 0

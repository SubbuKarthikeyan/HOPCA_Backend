"""Tests for Semantic Chunker (Scenario 7)."""
from app.rag.chunkers.semantic_chunker import SemanticChunker
from app.rag.models import KBDocument


def test_scenario_7_chunking_valid_chunks():
    """Scenario 7: Chunking produces valid chunks with all required metadata populated."""
    doc = KBDocument(
        document_id="doctors",
        title="HOPCA General Hospital Doctors",
        domain="doctors",
        document_type="directory",
        version="1.0",
        source="HOPCA",
        status="active",
        synthetic_data=True,
        file_path="/fake/path/doctors.md",
        raw_content="""## Doctor: DOC-001
Name: Dr. John Smith
Specialty: Cardiology
Available: Mon-Fri

---

## Doctor: DOC-002
Name: Dr. Sarah Connor
Specialty: Neurology
Available: Tue-Thu
""",
    )

    chunker = SemanticChunker(chunk_size=1000, chunk_overlap=150)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 2

    c1 = chunks[0]
    assert c1.document_id == "doctors"
    assert c1.entity_id == "DOC-001"
    assert c1.domain == "doctors"
    assert c1.document_type == "directory"
    assert c1.title == "HOPCA General Hospital Doctors"
    assert c1.chunk_index == 0
    assert "Dr. John Smith" in c1.content
    assert len(c1.content_hash) == 64

    c2 = chunks[1]
    assert c2.document_id == "doctors"
    assert c2.entity_id == "DOC-002"
    assert c2.chunk_index == 1
    assert "Dr. Sarah Connor" in c2.content
    assert len(c2.content_hash) == 64

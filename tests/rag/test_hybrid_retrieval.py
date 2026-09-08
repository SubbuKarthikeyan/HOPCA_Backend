"""Unit and integration tests for Hybrid Retrieval (RRF), Context Construction, and RAG Service."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.rag.context import ContextBuilder
from app.rag.models import RAGContextResult, RAGSearchResult, RetrievedChunk
from app.rag.rag_service import RAGService
from app.rag.retriever import HybridRetriever


# ─────────────────────────────────────────────────────────────────────────────
# 1. RRF Mathematics & Scoring Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_rrf_score_calculation():
    """Verify that RRF score computes accurately using reciprocal rank formula."""
    retriever = HybridRetriever(
        chunk_repo=MagicMock(),
        embedding_service=MagicMock(),
        k_constant=60,
        dense_weight=0.6,
        sparse_weight=0.4,
    )

    # When item is #1 in dense and #1 in sparse:
    # rrf = 0.6 / (60 + 1) + 0.4 / (60 + 1) = 1.0 / 61 ≈ 0.0163934
    score_both_first = retriever.compute_rrf_score(dense_rank=1, sparse_rank=1)
    assert pytest.approx(score_both_first, rel=1e-5) == (0.6 / 61 + 0.4 / 61)

    # When item is only in dense rank #1:
    score_dense_only = retriever.compute_rrf_score(dense_rank=1, sparse_rank=None)
    assert pytest.approx(score_dense_only, rel=1e-5) == (0.6 / 61)

    # When item is only in sparse rank #1:
    score_sparse_only = retriever.compute_rrf_score(dense_rank=None, sparse_rank=1)
    assert pytest.approx(score_sparse_only, rel=1e-5) == (0.4 / 61)

    # Both present should strictly exceed single modality
    assert score_both_first > score_dense_only
    assert score_both_first > score_sparse_only


# ─────────────────────────────────────────────────────────────────────────────
# 2. Hybrid Search Mocking & Rank Fusion Tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_hybrid_search_fusion_and_ranking():
    """Test that items present in both semantic and keyword results get highest rank."""
    mock_repo = MagicMock()
    mock_embedder = MagicMock()

    mock_embedder.embed_query = AsyncMock(return_value=[0.1] * 768)

    # Dense results: docA is #1, docB is #2
    mock_repo.search_vectors_cosine = AsyncMock(return_value=[
        {
            "chunk_id": "chunk_docA_1",
            "document_id": "docA",
            "title": "ICU Visiting Policy",
            "entity_id": "ICU-01",
            "domain": "policies",
            "content": "Visiting hours in ICU are 10 AM to 6 PM.",
            "score": 0.92,
        },
        {
            "chunk_id": "chunk_docB_1",
            "document_id": "docB",
            "title": "General Ward Policy",
            "entity_id": "WARD-01",
            "domain": "policies",
            "content": "General ward visiting is open.",
            "score": 0.81,
        },
    ])

    # Sparse text results: docC is #1, docA is #2
    mock_repo.search_text = AsyncMock(return_value=[
        {
            "chunk_id": "chunk_docC_1",
            "document_id": "docC",
            "title": "Emergency Admission",
            "entity_id": "EMERG-01",
            "domain": "emergency",
            "content": "Emergency admissions are open 24/7.",
            "score": 4.5,
        },
        {
            "chunk_id": "chunk_docA_1",
            "document_id": "docA",
            "title": "ICU Visiting Policy",
            "entity_id": "ICU-01",
            "domain": "policies",
            "content": "Visiting hours in ICU are 10 AM to 6 PM.",
            "score": 3.2,
        },
    ])

    retriever = HybridRetriever(
        chunk_repo=mock_repo,
        embedding_service=mock_embedder,
        k_constant=60,
        dense_weight=0.6,
        sparse_weight=0.4,
    )

    result: RAGSearchResult = await retriever.search(
        query="What are the ICU visiting hours?",
        top_k=5,
    )

    assert result.total_retrieved == 3
    # docA is in both (dense #1 + sparse #2) -> must be ranked #1
    assert result.chunks[0].chunk_id == "chunk_docA_1"
    assert result.chunks[0].dense_rank == 1
    assert result.chunks[0].sparse_rank == 2
    assert result.chunks[0].rrf_score > result.chunks[1].rrf_score


@pytest.mark.asyncio
async def test_hybrid_search_empty_query():
    """Test that empty queries return empty result safely."""
    retriever = HybridRetriever(chunk_repo=MagicMock(), embedding_service=MagicMock())
    result = await retriever.search(query="")
    assert result.total_retrieved == 0
    assert len(result.chunks) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Context Builder & Token Bounding Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_context_builder_with_chunks():
    """Test building structured, citation-attributed context from retrieved chunks."""
    builder = ContextBuilder(max_context_chars=1000)

    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            document_id="doc_cardio",
            entity_id="DOC-CARDIO-01",
            domain="doctors",
            title="Dr. Sarah Jenkins Cardiology",
            content="Dr. Sarah Jenkins specializes in pediatric cardiology. Office room 402.",
            score=0.9,
            rrf_score=0.015,
        ),
        RetrievedChunk(
            chunk_id="c2",
            document_id="doc_visiting",
            entity_id="POL-VISIT",
            domain="policies",
            title="Hospital Visiting Guidelines",
            content="General visiting hours are 9:00 AM to 8:00 PM daily.",
            score=0.85,
            rrf_score=0.012,
        ),
    ]

    result: RAGContextResult = builder.build_context(
        query="Tell me about Dr. Sarah Jenkins",
        chunks=chunks,
    )

    assert result.has_context is True
    assert result.chunks_used == 2
    assert len(result.sources) == 2
    assert "Dr. Sarah Jenkins Cardiology" in result.formatted_context
    assert "Hospital Visiting Guidelines" in result.formatted_context
    assert "[Source: Dr. Sarah Jenkins Cardiology (Entity: DOC-CARDIO-01) | Domain: Doctors]" in result.formatted_context


def test_context_builder_empty_chunks():
    """Test that zero chunks yield a controlled fallback message without hallucination."""
    builder = ContextBuilder()
    result = builder.build_context(query="Random query", chunks=[])
    assert result.has_context is False
    assert result.chunks_used == 0
    assert "No relevant hospital operational guidelines" in result.formatted_context


def test_context_builder_char_limit():
    """Test that context builder respects character limit bounding."""
    builder = ContextBuilder(max_context_chars=200)

    long_chunks = [
        RetrievedChunk(
            chunk_id=f"c_{i}",
            document_id=f"doc_{i}",
            domain="general",
            title=f"Doc Title {i}",
            content="A" * 150,
            rrf_score=0.1 - (i * 0.01),
        )
        for i in range(5)
    ]

    result = builder.build_context(query="test", chunks=long_chunks)
    assert result.has_context is True
    # Should only include 1 block before exceeding 200 chars
    assert result.chunks_used == 1


# ─────────────────────────────────────────────────────────────────────────────
# 4. End-to-End RAG Service Tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rag_service_query_and_prompt_formatting():
    """Test complete query flow and prompt generation in RAGService."""
    mock_retriever = MagicMock()
    mock_retriever.search = AsyncMock(return_value=RAGSearchResult(
        query="Emergency parking rules",
        chunks=[
            RetrievedChunk(
                chunk_id="chunk_park_1",
                document_id="doc_parking",
                entity_id="PARK-01",
                domain="hospital",
                title="Hospital Parking and Access",
                content="Emergency department parking is free for the first 3 hours.",
                rrf_score=0.016,
            )
        ],
        total_retrieved=1,
    ))

    service = RAGService(retriever=mock_retriever, context_builder=ContextBuilder())

    context_result = await service.query_knowledge_base("Emergency parking rules")
    assert context_result.has_context is True
    assert "Emergency department parking is free" in context_result.formatted_context

    messages = service.build_llm_messages(
        query="Emergency parking rules",
        context_result=context_result,
        conversation_history=[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi! How can I help you today?"}],
    )

    assert len(messages) == 4
    assert messages[0]["role"] == "system"
    assert "--- HOSPITAL KNOWLEDGE CONTEXT ---" in messages[0]["content"]
    assert "Emergency department parking is free" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "Hello"}
    assert messages[2] == {"role": "assistant", "content": "Hi! How can I help you today?"}
    assert messages[3] == {"role": "user", "content": "Emergency parking rules"}

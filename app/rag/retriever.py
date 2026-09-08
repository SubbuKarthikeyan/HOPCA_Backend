"""Hybrid Retriever for RAG pipeline combining Dense Vector Search and Sparse Text Search using Reciprocal Rank Fusion (RRF)."""
import logging
from typing import Dict, List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.db.repositories.chunk_repo import ChunkRepository
from app.rag.embeddings.embedding_service import EmbeddingService
from app.rag.models import RAGSearchResult, RetrievedChunk

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid search retriever merging dense vector similarity and sparse keyword search via RRF."""

    def __init__(
        self,
        chunk_repo: Optional[ChunkRepository] = None,
        embedding_service: Optional[EmbeddingService] = None,
        db: Optional[AsyncIOMotorDatabase] = None,
        k_constant: int = 60,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ):
        self.chunk_repo = chunk_repo if chunk_repo is not None else ChunkRepository(db=db)
        self.embedding_service = embedding_service if embedding_service is not None else EmbeddingService()
        self.k_constant = k_constant
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def compute_rrf_score(
        self,
        dense_rank: Optional[int],
        sparse_rank: Optional[int],
    ) -> float:
        """Compute Reciprocal Rank Fusion (RRF) score from dense and sparse 1-based ranks."""
        score = 0.0
        if dense_rank is not None and dense_rank > 0:
            score += self.dense_weight / (self.k_constant + dense_rank)
        if sparse_rank is not None and sparse_rank > 0:
            score += self.sparse_weight / (self.k_constant + sparse_rank)
        return score

    async def search(
        self,
        query: str,
        top_k: int = 5,
        domain_filter: Optional[str] = None,
        score_threshold: float = 0.0,
    ) -> RAGSearchResult:
        """Perform hybrid search over indexed knowledge chunks.
        
        Args:
            query: User's search query string.
            top_k: Maximum number of top chunks to return.
            domain_filter: Optional domain string to filter chunks (e.g. 'emergency', 'policies').
            score_threshold: Minimum RRF score required for a chunk to be included.

        Returns:
            RAGSearchResult with ranked, deduplicated RetrievedChunk items.
        """
        if not query or not query.strip():
            return RAGSearchResult(query="", chunks=[], total_retrieved=0)

        cleaned_query = query.strip()
        fetch_limit = max(top_k * 3, 20)

        # 1. Dense Semantic Vector Search
        dense_chunks: List[Dict] = []
        try:
            query_embedding = await self.embedding_service.embed_query(cleaned_query)
            dense_chunks = await self.chunk_repo.search_vectors_cosine(
                query_embedding=query_embedding,
                limit=fetch_limit,
                domain=domain_filter,
            )
        except Exception as exc:
            logger.warning("Dense vector search encountered an issue: %s", exc)

        # 2. Sparse Lexical / Keyword Text Search
        sparse_chunks: List[Dict] = []
        try:
            sparse_chunks = await self.chunk_repo.search_text(
                query=cleaned_query,
                limit=fetch_limit,
                domain=domain_filter,
            )
        except Exception as exc:
            logger.warning("Sparse text search encountered an issue: %s", exc)

        # 3. Reciprocal Rank Fusion (RRF)
        dense_rank_map: Dict[str, int] = {
            doc["chunk_id"]: idx + 1 for idx, doc in enumerate(dense_chunks)
        }
        sparse_rank_map: Dict[str, int] = {
            doc["chunk_id"]: idx + 1 for idx, doc in enumerate(sparse_chunks)
        }

        # Collect unique chunk documents
        all_chunk_docs: Dict[str, Dict] = {}
        for doc in dense_chunks:
            all_chunk_docs[doc["chunk_id"]] = doc
        for doc in sparse_chunks:
            if doc["chunk_id"] not in all_chunk_docs:
                all_chunk_docs[doc["chunk_id"]] = doc

        # Score all candidates
        scored_candidates: List[RetrievedChunk] = []
        for chunk_id, doc in all_chunk_docs.items():
            d_rank = dense_rank_map.get(chunk_id)
            s_rank = sparse_rank_map.get(chunk_id)
            rrf = self.compute_rrf_score(d_rank, s_rank)

            if rrf >= score_threshold:
                chunk_obj = RetrievedChunk(
                    chunk_id=doc.get("chunk_id", chunk_id),
                    document_id=doc.get("document_id", ""),
                    entity_id=doc.get("entity_id"),
                    domain=doc.get("domain", "general"),
                    title=doc.get("title", ""),
                    content=doc.get("content", ""),
                    chunk_index=doc.get("chunk_index", 0),
                    score=doc.get("score", 0.0),
                    dense_rank=d_rank,
                    sparse_rank=s_rank,
                    rrf_score=rrf,
                    metadata=doc.get("metadata", {}),
                )
                scored_candidates.append(chunk_obj)

        # Sort by RRF score descending
        scored_candidates.sort(key=lambda c: c.rrf_score, reverse=True)
        top_results = scored_candidates[:top_k]

        return RAGSearchResult(
            query=cleaned_query,
            chunks=top_results,
            total_retrieved=len(top_results),
            dense_count=len(dense_chunks),
            sparse_count=len(sparse_chunks),
        )

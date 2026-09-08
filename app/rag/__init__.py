"""RAG (Retrieval-Augmented Generation) pipeline package for HOPCA."""
from app.rag.context import ContextBuilder
from app.rag.ingestion.pipeline import IngestionPipeline
from app.rag.models import (
    Chunk,
    EmbeddedChunk,
    KBDocument,
    RAGContextResult,
    RAGSearchResult,
    RetrievedChunk,
)
from app.rag.rag_service import RAGService, rag_service
from app.rag.retriever import HybridRetriever

__all__ = [
    "HybridRetriever",
    "ContextBuilder",
    "RAGService",
    "rag_service",
    "IngestionPipeline",
    "Chunk",
    "EmbeddedChunk",
    "KBDocument",
    "RetrievedChunk",
    "RAGSearchResult",
    "RAGContextResult",
]

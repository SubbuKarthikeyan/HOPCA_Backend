"""Embeddings package for vector embedding generation."""
from app.rag.embeddings.base import BaseEmbeddingProvider
from app.rag.embeddings.embedding_service import EmbeddingService
from app.rag.embeddings.gemini_embedder import GeminiEmbeddingProvider

__all__ = ["BaseEmbeddingProvider", "GeminiEmbeddingProvider", "EmbeddingService"]

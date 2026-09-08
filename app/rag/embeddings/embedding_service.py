"""Embedding service orchestrator for batching, retry, and model metadata tracking."""
import logging
from typing import List, Optional

from app.core.config import settings
from app.rag.embeddings.base import BaseEmbeddingProvider
from app.rag.embeddings.gemini_embedder import GeminiEmbeddingProvider
from app.rag.models import Chunk, EmbeddedChunk

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Orchestrates embedding generation across chunk collections with batching and validation."""

    def __init__(
        self,
        provider: Optional[BaseEmbeddingProvider] = None,
        batch_size: int = 20,
    ):
        if provider is not None:
            self.provider = provider
        else:
            self.provider = GeminiEmbeddingProvider(
                api_key=settings.gemini_api_key,
                model_name=settings.embedding_model,
                dimension=settings.embedding_dimension,
                version=settings.embedding_version,
            )
        self.batch_size = batch_size

    @property
    def model_name(self) -> str:
        return self.provider.model_name

    @property
    def dimension(self) -> int:
        return self.provider.dimension

    @property
    def version(self) -> str:
        return self.provider.version

    async def embed_chunks(self, chunks: List[Chunk]) -> List[EmbeddedChunk]:
        """Embed a list of chunks in batches and return EmbeddedChunk objects."""
        if not chunks:
            return []

        embedded_chunks: List[EmbeddedChunk] = []
        total_chunks = len(chunks)

        for i in range(0, total_chunks, self.batch_size):
            batch = chunks[i : i + self.batch_size]
            texts = [c.content for c in batch]

            logger.info("Generating embeddings for batch %d-%d of %d chunks", i + 1, min(i + len(batch), total_chunks), total_chunks)
            vectors = await self.provider.embed_batch(texts)

            for chunk, vec in zip(batch, vectors):
                if len(vec) != self.dimension:
                    raise ValueError(
                        f"Vector dimension mismatch for chunk {chunk.chunk_id}: expected {self.dimension}, got {len(vec)}"
                    )

                embedded_chunk = EmbeddedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    entity_id=chunk.entity_id,
                    domain=chunk.domain,
                    document_type=chunk.document_type,
                    title=chunk.title,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    content_hash=chunk.content_hash,
                    embedding=vec,
                    embedding_model=self.model_name,
                    embedding_dimension=self.dimension,
                    embedding_version=self.version,
                )
                embedded_chunks.append(embedded_chunk)

        return embedded_chunks

    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding vector for search query."""
        vec = await self.provider.embed_query(query)
        if len(vec) != self.dimension:
            raise ValueError(
                f"Query vector dimension mismatch: expected {self.dimension}, got {len(vec)}"
            )
        return vec

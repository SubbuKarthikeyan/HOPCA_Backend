"""Deterministic context construction and token-budget bounding for LLM input."""
import logging
from typing import Dict, List, Optional

from app.rag.models import RAGContextResult, RetrievedChunk

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Formats retrieved knowledge chunks into deterministic, source-attributed LLM context."""

    def __init__(
        self,
        max_context_chars: int = 4000,
        no_context_fallback: str = "No relevant hospital operational guidelines or documents were found for this query.",
    ):
        self.max_context_chars = max_context_chars
        self.no_context_fallback = no_context_fallback

    def build_context(
        self,
        query: str,
        chunks: List[RetrievedChunk],
    ) -> RAGContextResult:
        """Construct structured, source-attributed context from retrieved chunks.
        
        Args:
            query: User's original search query.
            chunks: Ranked retrieved chunks from HybridRetriever.

        Returns:
            RAGContextResult containing formatted markdown context and sources.
        """
        if not chunks:
            return RAGContextResult(
                query=query,
                formatted_context=self.no_context_fallback,
                sources=[],
                chunks_used=0,
                has_context=False,
            )

        formatted_blocks: List[str] = []
        sources: List[Dict[str, str]] = []
        seen_chunk_hashes = set()
        current_char_count = 0

        for chunk in chunks:
            # Simple content fingerprint for deduplication
            content_cleaned = chunk.content.strip()
            if not content_cleaned:
                continue

            content_key = f"{chunk.document_id}_{chunk.chunk_index}"
            if content_key in seen_chunk_hashes:
                continue
            seen_chunk_hashes.add(content_key)

            # Build source attribution header
            source_label = chunk.title or chunk.document_id
            entity_info = f" (Entity: {chunk.entity_id})" if chunk.entity_id else ""
            header = f"--- [Source: {source_label}{entity_info} | Domain: {chunk.domain.capitalize()}] ---"
            block = f"{header}\n{content_cleaned}\n"

            # Check character budget
            if current_char_count + len(block) > self.max_context_chars and formatted_blocks:
                logger.info(
                    "Context char limit reached (%d/%d chars). Stopping context inclusion.",
                    current_char_count,
                    self.max_context_chars,
                )
                break

            formatted_blocks.append(block)
            current_char_count += len(block)
            sources.append({
                "document_id": chunk.document_id,
                "title": chunk.title,
                "entity_id": chunk.entity_id or "",
                "domain": chunk.domain,
                "chunk_id": chunk.chunk_id,
            })

        if not formatted_blocks:
            return RAGContextResult(
                query=query,
                formatted_context=self.no_context_fallback,
                sources=[],
                chunks_used=0,
                has_context=False,
            )

        full_context = "\n".join(formatted_blocks).strip()
        return RAGContextResult(
            query=query,
            formatted_context=full_context,
            sources=sources,
            chunks_used=len(formatted_blocks),
            has_context=True,
        )

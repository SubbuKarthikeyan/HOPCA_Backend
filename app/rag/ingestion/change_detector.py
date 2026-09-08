"""Change detection and hash registry management for incremental RAG updates."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from app.rag.models import Chunk, ChunkChangeStatus, EmbeddedChunk

logger = logging.getLogger(__name__)


class ChangeDetector:
    """Detects NEW, CHANGED, UNCHANGED, and REMOVED chunks using SHA-256 hash comparisons."""

    def __init__(
        self,
        registry_path: Optional[str | Path] = None,
        embedding_model: str = "text-embedding-004",
        embedding_dimension: int = 768,
        embedding_version: str = "1.0",
    ):
        self.registry_path = Path(registry_path).resolve() if registry_path else None
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self.embedding_version = embedding_version
        self._registry_data: Dict = {}
        if self.registry_path is not None:
            self.load_registry()

    def populate_from_mongo(
        self,
        chunk_records: List[Dict],
        system_metadata: Optional[Dict] = None,
    ) -> None:
        """Populate change detector state directly from MongoDB collections."""
        chunks_map: Dict[str, Dict] = {}
        for doc in chunk_records:
            cid = doc.get("chunk_id")
            if not cid:
                continue
            chunks_map[cid] = {
                "document_id": doc.get("document_id"),
                "entity_id": doc.get("entity_id"),
                "domain": doc.get("domain"),
                "content_hash": doc.get("content_hash"),
                "embedding": doc.get("embedding"),
            }

        meta = system_metadata or {}
        self._registry_data = {
            "embedding_model": meta.get("embedding_model", self.embedding_model),
            "embedding_dimension": meta.get("embedding_dimension", self.embedding_dimension),
            "embedding_version": meta.get("embedding_version", self.embedding_version),
            "chunks": chunks_map,
        }

    def load_registry(self) -> None:
        """Load hash registry from disk if it exists."""
        if self.registry_path and self.registry_path.exists():
            try:
                content = self.registry_path.read_text(encoding="utf-8")
                self._registry_data = json.loads(content)
            except Exception as e:
                logger.warning("Failed to load hash registry from %s, starting fresh: %s", self.registry_path, e)
                self._registry_data = {}
        else:
            self._registry_data = {}

    def save_registry(self) -> None:
        """Persist hash registry to disk if registry_path is configured."""
        if self.registry_path is not None:
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            self.registry_path.write_text(json.dumps(self._registry_data, indent=2), encoding="utf-8")

    def detect_changes(
        self, current_chunks: List[Chunk]
    ) -> Tuple[
        List[Chunk],  # new_chunks
        List[Chunk],  # changed_chunks
        List[EmbeddedChunk],  # unchanged_embedded_chunks (reused from cache)
        List[str],  # removed_chunk_ids
    ]:
        """Classify chunks into NEW, CHANGED, UNCHANGED, and detect REMOVED chunk IDs."""
        stored_model = self._registry_data.get("embedding_model")
        stored_dim = self._registry_data.get("embedding_dimension")
        stored_ver = self._registry_data.get("embedding_version")
        chunks_map = self._registry_data.get("chunks", {})

        # If embedding configuration changed, invalidate all existing embeddings
        model_changed = (
            stored_model is not None
            and (
                stored_model != self.embedding_model
                or stored_dim != self.embedding_dimension
                or stored_ver != self.embedding_version
            )
        )

        new_chunks: List[Chunk] = []
        changed_chunks: List[Chunk] = []
        unchanged_embedded: List[EmbeddedChunk] = []

        current_chunk_ids: Set[str] = set()

        for chunk in current_chunks:
            current_chunk_ids.add(chunk.chunk_id)

            if model_changed:
                changed_chunks.append(chunk)
                continue

            if chunk.chunk_id not in chunks_map:
                new_chunks.append(chunk)
            else:
                entry = chunks_map[chunk.chunk_id]
                stored_hash = entry.get("content_hash")

                if stored_hash != chunk.content_hash:
                    changed_chunks.append(chunk)
                else:
                    # Unchanged chunk - reconstruct EmbeddedChunk from cached vector if present
                    cached_vec = entry.get("embedding")
                    if cached_vec and len(cached_vec) == self.embedding_dimension:
                        unchanged_embedded.append(
                            EmbeddedChunk(
                                chunk_id=chunk.chunk_id,
                                document_id=chunk.document_id,
                                entity_id=chunk.entity_id,
                                domain=chunk.domain,
                                document_type=chunk.document_type,
                                title=chunk.title,
                                chunk_index=chunk.chunk_index,
                                content=chunk.content,
                                content_hash=chunk.content_hash,
                                embedding=cached_vec,
                                embedding_model=self.embedding_model,
                                embedding_dimension=self.embedding_dimension,
                                embedding_version=self.embedding_version,
                            )
                        )
                    else:
                        # If cached embedding is missing or corrupted, treat as changed
                        changed_chunks.append(chunk)

        # Detect removed chunk IDs
        removed_chunk_ids: List[str] = [
            cid for cid in chunks_map if cid not in current_chunk_ids
        ]

        return new_chunks, changed_chunks, unchanged_embedded, removed_chunk_ids

    def update_registry(
        self,
        all_embedded_chunks: List[EmbeddedChunk],
        removed_chunk_ids: Optional[List[str]] = None,
    ) -> None:
        """Update and save registry with newly embedded and existing chunks."""
        chunks_map: Dict[str, Dict] = {}

        for chunk in all_embedded_chunks:
            chunks_map[chunk.chunk_id] = {
                "document_id": chunk.document_id,
                "entity_id": chunk.entity_id,
                "domain": chunk.domain,
                "content_hash": chunk.content_hash,
                "embedding": chunk.embedding,
            }

        self._registry_data = {
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "embedding_version": self.embedding_version,
            "chunks": chunks_map,
        }
        self.save_registry()

"""Complete Ingestion Pipeline orchestrating Validation -> Loading -> Cleaning -> Chunking -> Hashing -> Change Detection -> Embedding -> MongoDB Persistence."""
import logging
import os
import uuid
from pathlib import Path
from typing import List, Optional, Tuple, Union
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.config import settings
from app.db.init_db import init_mongodb
from app.db.repositories.chunk_repo import ChunkRepository
from app.db.repositories.document_repo import DocumentRepository
from app.db.repositories.metadata_repo import MetadataRepository
from app.db.repositories.run_repo import IngestionRunRepository
from app.db.session import MongoDBManager
from app.rag.chunkers.semantic_chunker import SemanticChunker
from app.rag.cleaners.markdown_cleaner import MarkdownCleaner
from app.rag.embeddings.base import BaseEmbeddingProvider
from app.rag.embeddings.embedding_service import EmbeddingService
from app.rag.hashing.content_hasher import ContentHasher
from app.rag.ingestion.change_detector import ChangeDetector
from app.rag.loaders.markdown_loader import MarkdownLoader
from app.rag.models import EmbeddedChunk, IngestionStats, ValidationSeverity
from app.rag.validators.validator import KnowledgeBaseValidator

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Orchestrates the entire RAG knowledge preparation and MongoDB storage pipeline."""

    def __init__(
        self,
        kb_path: Optional[Union[str, Path]] = None,
        registry_path: Optional[Union[str, Path]] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        strict_validation: bool = False,
        db: Optional[AsyncIOMotorDatabase] = None,
        use_mongo: bool = True,
    ):
        self.kb_path = Path(kb_path or settings.knowledge_base_dir).resolve()
        data_dir = Path(settings.data_dir).resolve()
        self.registry_path = Path(registry_path or (data_dir / "hash_registry.json")).resolve() if registry_path else (data_dir / "hash_registry.json")

        self.validator = KnowledgeBaseValidator(strict=strict_validation)
        self.loader = MarkdownLoader()
        self.cleaner = MarkdownCleaner()
        self.chunker = SemanticChunker(
            chunk_size=chunk_size or settings.chunk_size,
            chunk_overlap=chunk_overlap or settings.chunk_overlap,
            strategy=settings.chunking_strategy,
        )
        self.embedding_service = EmbeddingService(
            provider=embedding_provider,
            batch_size=settings.embedding_batch_size,
        )
        self.change_detector = ChangeDetector(
            registry_path=self.registry_path,
            embedding_model=self.embedding_service.model_name,
            embedding_dimension=self.embedding_service.dimension,
            embedding_version=self.embedding_service.version,
        )

        self.use_mongo = use_mongo and bool(settings.mongo_url or db is not None)
        self.db = db
        self.doc_repo: Optional[DocumentRepository] = None
        self.chunk_repo: Optional[ChunkRepository] = None
        self.run_repo: Optional[IngestionRunRepository] = None
        self.meta_repo: Optional[MetadataRepository] = None

    async def _setup_mongo(self) -> None:
        """Initialize MongoDB repositories and database schema."""
        if not self.use_mongo:
            return

        try:
            if self.db is None:
                self.db = MongoDBManager.get_database()
            await init_mongodb(self.db)
            self.doc_repo = DocumentRepository(self.db)
            self.chunk_repo = ChunkRepository(self.db)
            self.run_repo = IngestionRunRepository(self.db)
            self.meta_repo = MetadataRepository(self.db)
        except Exception as e:
            logger.warning("MongoDB initialization failed, continuing in memory/file mode: %s", e)
            self.use_mongo = False

    async def run(self) -> Tuple[IngestionStats, List[EmbeddedChunk]]:
        """Run the complete ingestion pipeline with MongoDB persistence."""
        stats = IngestionStats()
        run_id = f"run_{uuid.uuid4().hex[:12]}"

        # Initialize MongoDB
        await self._setup_mongo()

        if self.use_mongo and self.run_repo:
            try:
                await self.run_repo.start_run(run_id=run_id)
            except Exception as e:
                logger.warning("Failed to record run start in MongoDB: %s", e)

        # Step 1: Validate Knowledge Base
        logger.info("Step 1: Validating Knowledge Base at %s", self.kb_path)
        val_result = self.validator.validate(self.kb_path)

        stats.documents_scanned = val_result.summary.total_files
        stats.documents_valid = val_result.summary.valid_files

        if not val_result.valid:
            stats.status = "FAILED"
            for issue in val_result.issues:
                if issue.severity == ValidationSeverity.ERROR:
                    stats.errors.append(f"[{issue.code.value}] {Path(issue.file).name}: {issue.message}")
                elif issue.severity == ValidationSeverity.WARNING:
                    stats.warnings.append(f"[{issue.code.value}] {Path(issue.file).name}: {issue.message}")
            logger.error("Knowledge base validation failed with %d errors", len(stats.errors))
            if self.use_mongo and self.run_repo:
                await self.run_repo.complete_run(run_id=run_id, stats=stats)
            return stats, []

        for issue in val_result.issues:
            if issue.severity == ValidationSeverity.WARNING:
                stats.warnings.append(f"[{issue.code.value}] {Path(issue.file).name}: {issue.message}")

        # Step 2: Load Documents
        logger.info("Step 2: Loading markdown documents")
        documents = self.loader.load_all(self.kb_path)

        # Step 3: Clean Documents
        logger.info("Step 3: Normalizing and cleaning markdown content")
        for doc in documents:
            doc.cleaned_content = self.cleaner.clean(doc.raw_content)

        # Step 4: Chunk Documents & Compute Content Hashes
        logger.info("Step 4: Performing semantic chunking and hash computation")
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)

        stats.chunks_total = len(all_chunks)

        # Step 5: Detect Changes (Load from MongoDB if available)
        if self.use_mongo and self.chunk_repo and self.meta_repo:
            try:
                active_chunks = await self.chunk_repo.get_all_active_chunks()
                system_meta = await self.meta_repo.get_system_metadata()
                self.change_detector.populate_from_mongo(active_chunks, system_meta)
            except Exception as e:
                logger.warning("Failed to fetch state from MongoDB, falling back to local detector: %s", e)

        logger.info("Step 5: Detecting changes against registry/database")
        new_chunks, changed_chunks, unchanged_embedded, removed_chunk_ids = (
            self.change_detector.detect_changes(all_chunks)
        )

        stats.new_chunks = len(new_chunks)
        stats.changed_chunks = len(changed_chunks)
        stats.unchanged_chunks = len(unchanged_embedded)
        stats.removed_chunks = len(removed_chunk_ids)

        chunks_to_embed = new_chunks + changed_chunks

        # Track documents that had changes
        changed_doc_ids = {c.document_id for c in chunks_to_embed}
        stats.documents_changed = len(changed_doc_ids)

        # Step 6: Generate Embeddings for New and Changed Chunks
        logger.info(
            "Step 6: Generating embeddings for %d chunks (%d reused)",
            len(chunks_to_embed),
            len(unchanged_embedded),
        )

        newly_embedded: List[EmbeddedChunk] = []
        if chunks_to_embed:
            try:
                newly_embedded = await self.embedding_service.embed_chunks(chunks_to_embed)
                stats.embeddings_generated = len(newly_embedded)
            except Exception as e:
                stats.status = "FAILED"
                stats.errors.append(f"Embedding generation failed: {str(e)}")
                logger.error("Embedding generation failed: %s", e)
                if self.use_mongo and self.run_repo:
                    await self.run_repo.complete_run(run_id=run_id, stats=stats)
                return stats, []
        else:
            stats.embeddings_generated = 0

        stats.embeddings_reused = len(unchanged_embedded)

        # Combine all embedded chunks
        all_embedded = newly_embedded + unchanged_embedded

        # Step 7: Persist to MongoDB
        if self.use_mongo and self.doc_repo and self.chunk_repo and self.meta_repo:
            try:
                # 7a. Upsert Documents
                for doc in documents:
                    doc_hash = ContentHasher.compute_hash(doc.cleaned_content or doc.raw_content)
                    await self.doc_repo.upsert_document(doc, doc_hash)

                # 7b. Upsert Chunks
                if newly_embedded:
                    await self.chunk_repo.upsert_chunks(newly_embedded)

                # 7c. Mark Removed Chunks
                if removed_chunk_ids:
                    await self.chunk_repo.mark_chunks_removed(removed_chunk_ids)

                # 7d. Update System Metadata
                await self.meta_repo.update_system_metadata(
                    pipeline_version="1.0",
                    embedding_model=self.embedding_service.model_name,
                    embedding_dimension=self.embedding_service.dimension,
                    chunking_strategy=settings.chunking_strategy,
                    chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap,
                )

                # 7e. Clean up interim JSON file if it exists
                if self.registry_path and self.registry_path.exists():
                    try:
                        self.registry_path.unlink()
                        logger.info("Removed interim JSON hash registry: %s", self.registry_path)
                    except Exception as e:
                        logger.debug("Could not remove interim JSON registry: %s", e)

            except Exception as e:
                logger.error("MongoDB persistence failed: %s", e)
                stats.status = "FAILED"
                stats.errors.append(f"MongoDB persistence failed: {str(e)}")
                if self.run_repo:
                    await self.run_repo.complete_run(run_id=run_id, stats=stats)
                return stats, []

        # Step 8: Update in-memory registry & complete run
        self.change_detector.update_registry(all_embedded, removed_chunk_ids)

        stats.status = "SUCCESS"
        if self.use_mongo and self.run_repo:
            await self.run_repo.complete_run(run_id=run_id, stats=stats)

        logger.info("Ingestion pipeline completed successfully: %d total chunks stored/tracked", len(all_embedded))
        return stats, all_embedded

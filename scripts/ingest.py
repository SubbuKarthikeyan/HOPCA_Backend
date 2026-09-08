"""CLI script to run knowledge base ingestion pipeline and populate MongoDB."""
import asyncio
import logging
import sys
from pathlib import Path

# Add backend directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings
from app.rag.ingestion.pipeline import IngestionPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("scripts.ingest")


async def main():
    logger.info("Starting HOPCA Knowledge Base Ingestion Pipeline...")
    pipeline = IngestionPipeline()
    stats, embedded_chunks = await pipeline.run()

    logger.info("Ingestion completed with status: %s", stats.status)
    logger.info("Documents Scanned: %d (Valid: %d)", stats.documents_scanned, stats.documents_valid)
    logger.info("Total Chunks: %d (New: %d, Changed: %d, Unchanged: %d, Removed: %d)",
                stats.chunks_total, stats.new_chunks, stats.changed_chunks, stats.unchanged_chunks, stats.removed_chunks)
    logger.info("Embeddings Generated: %d (Reused: %d)", stats.embeddings_generated, stats.embeddings_reused)

    if stats.errors:
        logger.error("Encountered %d errors during ingestion:", len(stats.errors))
        for err in stats.errors:
            logger.error(" - %s", err)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

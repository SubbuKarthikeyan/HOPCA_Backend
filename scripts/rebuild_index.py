"""CLI script to initialize or rebuild MongoDB indexes for HOPCA RAG."""
import asyncio
import logging
import sys
from pathlib import Path

# Add backend directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.init_db import init_mongodb
from app.db.session import MongoDBManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("scripts.rebuild_index")


async def main():
    logger.info("Connecting to MongoDB and verifying collections & search indexes...")
    try:
        db = MongoDBManager.get_database()
        indexes = await init_mongodb(db)
        logger.info("Successfully configured indexes across collections:")
        for col, idx_list in indexes.items():
            logger.info(" - %s: %s", col, idx_list)
        logger.info("MongoDB index verification complete.")
    except Exception as exc:
        logger.error("Failed to verify/rebuild MongoDB indexes: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

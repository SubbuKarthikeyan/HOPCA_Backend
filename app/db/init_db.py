"""Idempotent database and index initialization for HOPCA MongoDB collections."""
import logging
from typing import Dict, List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
import pymongo

from app.db.session import MongoDBManager

logger = logging.getLogger(__name__)

REQUIRED_COLLECTIONS = [
    "rag_documents",
    "rag_chunks",
    "ingestion_runs",
    "system_metadata",
]


async def init_mongodb(
    db: Optional[AsyncIOMotorDatabase] = None,
) -> Dict[str, List[str]]:
    """Initialize MongoDB collections and indexes idempotently.
    
    Ensures required collections exist and creates necessary unique and search indexes.
    Never drops existing data or collections.
    
    Returns:
        Dict mapping collection name to list of active index names.
    """
    database = db if db is not None else MongoDBManager.get_database()
    logger.info("Initializing MongoDB foundation for database: '%s'", database.name)

    existing_collections = await database.list_collection_names()

    # 1. Ensure collections exist
    for col_name in REQUIRED_COLLECTIONS:
        if col_name not in existing_collections:
            logger.info("Creating collection: '%s'", col_name)
            await database.create_collection(col_name)

    created_indexes: Dict[str, List[str]] = {}

    # 2. Indexes for 'rag_documents'
    doc_col = database["rag_documents"]
    await doc_col.create_index([("document_id", pymongo.ASCENDING)], unique=True, name="idx_document_id_unique")
    await doc_col.create_index([("domain", pymongo.ASCENDING)], name="idx_document_domain")
    await doc_col.create_index([("status", pymongo.ASCENDING)], name="idx_document_status")
    created_indexes["rag_documents"] = [idx["name"] async for idx in doc_col.list_indexes()]

    # 3. Indexes for 'rag_chunks'
    chunk_col = database["rag_chunks"]
    await chunk_col.create_index([("chunk_id", pymongo.ASCENDING)], unique=True, name="idx_chunk_id_unique")
    await chunk_col.create_index([("document_id", pymongo.ASCENDING)], name="idx_chunk_document_id")
    await chunk_col.create_index([("entity_id", pymongo.ASCENDING)], name="idx_chunk_entity_id")
    await chunk_col.create_index([("domain", pymongo.ASCENDING)], name="idx_chunk_domain")
    await chunk_col.create_index([("status", pymongo.ASCENDING)], name="idx_chunk_status")
    await chunk_col.create_index(
        [("document_id", pymongo.ASCENDING), ("chunk_index", pymongo.ASCENDING)],
        name="idx_chunk_doc_index",
    )
    # Compound text index for lexical / keyword hybrid search
    try:
        await chunk_col.create_index(
            [("title", pymongo.TEXT), ("entity_id", pymongo.TEXT), ("content", pymongo.TEXT)],
            name="idx_chunk_text_search",
            weights={"title": 3, "entity_id": 2, "content": 1},
            default_language="english",
        )
    except Exception as exc:
        logger.warning("Could not create text index 'idx_chunk_text_search' (may already exist with different params): %s", exc)
    created_indexes["rag_chunks"] = [idx["name"] async for idx in chunk_col.list_indexes()]

    # 4. Indexes for 'ingestion_runs'
    run_col = database["ingestion_runs"]
    await run_col.create_index([("run_id", pymongo.ASCENDING)], unique=True, name="idx_run_id_unique")
    await run_col.create_index([("started_at", pymongo.DESCENDING)], name="idx_run_started_at")
    await run_col.create_index([("status", pymongo.ASCENDING)], name="idx_run_status")
    created_indexes["ingestion_runs"] = [idx["name"] async for idx in run_col.list_indexes()]

    # 5. Indexes for 'system_metadata'
    meta_col = database["system_metadata"]
    await meta_col.create_index([("schema_version", pymongo.ASCENDING)], unique=True, name="idx_meta_schema_unique")
    created_indexes["system_metadata"] = [idx["name"] async for idx in meta_col.list_indexes()]

    logger.info("MongoDB foundation and indexes successfully verified.")
    return created_indexes

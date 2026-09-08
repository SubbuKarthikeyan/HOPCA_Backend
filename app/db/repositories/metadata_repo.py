"""Metadata repository for MongoDB system_metadata collection."""
from datetime import datetime, timezone
from typing import Dict, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
import pymongo

from app.core.config import settings
from app.db.session import MongoDBManager
from app.rag.models import MongoSystemMetadataRecord, utc_now


class MetadataRepository:
    """Repository for managing system configuration, active models, and schema version metadata."""

    def __init__(self, db: Optional[AsyncIOMotorDatabase] = None):
        self.db = db if db is not None else MongoDBManager.get_database()
        self.collection = self.db["system_metadata"]

    async def get_system_metadata(self, schema_version: str = "1.0") -> Optional[Dict]:
        """Fetch system metadata record."""
        return await self.collection.find_one({"schema_version": schema_version})

    async def update_system_metadata(
        self,
        pipeline_version: str = "1.0",
        embedding_model: Optional[str] = None,
        embedding_dimension: Optional[int] = None,
        chunking_strategy: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        last_successful_ingestion: Optional[datetime] = None,
        schema_version: str = "1.0",
    ) -> Dict:
        """Update or insert system configuration metadata."""
        now = utc_now()
        existing = await self.collection.find_one({"schema_version": schema_version})
        created_at = existing.get("created_at", now) if existing else now

        record = MongoSystemMetadataRecord(
            pipeline_version=pipeline_version,
            embedding_model=embedding_model or settings.embedding_model,
            embedding_dimension=embedding_dimension or settings.embedding_dimension,
            chunking_strategy=chunking_strategy or settings.chunking_strategy,
            chunk_size=chunk_size or settings.chunk_size,
            chunk_overlap=chunk_overlap or settings.chunk_overlap,
            last_successful_ingestion=last_successful_ingestion or now,
            schema_version=schema_version,
            created_at=created_at,
            updated_at=now,
        )

        data = record.model_dump()
        await self.collection.update_one(
            {"schema_version": schema_version},
            {"$set": data},
            upsert=True,
        )
        return data

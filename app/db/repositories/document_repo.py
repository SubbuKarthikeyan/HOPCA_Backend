"""Document repository for MongoDB rag_documents collection."""
from datetime import datetime, timezone
from typing import Dict, List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
import pymongo

from app.db.session import MongoDBManager
from app.rag.models import KBDocument, MongoDocumentRecord, utc_now


class DocumentRepository:
    """Repository for CRUD and upsert operations on rag_documents collection."""

    def __init__(self, db: Optional[AsyncIOMotorDatabase] = None):
        self.db = db if db is not None else MongoDBManager.get_database()
        self.collection = self.db["rag_documents"]

    async def upsert_document(self, doc: KBDocument, content_hash: str) -> None:
        """Insert or update a KBDocument record by document_id."""
        now = utc_now()
        existing = await self.collection.find_one({"document_id": doc.document_id})
        created_at = existing.get("created_at", now) if existing else now

        record = MongoDocumentRecord(
            document_id=doc.document_id,
            title=doc.title,
            domain=doc.domain,
            document_type=doc.document_type,
            source=doc.source,
            version=doc.version,
            content_hash=content_hash,
            status=doc.status,
            synthetic_data=doc.synthetic_data,
            file_path=doc.file_path,
            created_at=created_at,
            updated_at=now,
        )

        await self.collection.update_one(
            {"document_id": doc.document_id},
            {"$set": record.model_dump()},
            upsert=True,
        )

    async def get_document(self, document_id: str) -> Optional[Dict]:
        """Fetch a single document record by document_id."""
        return await self.collection.find_one({"document_id": document_id})

    async def list_active_documents(self) -> List[Dict]:
        """Fetch all documents with status='active'."""
        cursor = self.collection.find({"status": "active"})
        return await cursor.to_list(length=1000)

    async def mark_documents_removed(self, document_ids: List[str]) -> int:
        """Mark documents as removed when deleted from source."""
        if not document_ids:
            return 0
        res = await self.collection.update_many(
            {"document_id": {"$in": document_ids}},
            {"$set": {"status": "removed", "updated_at": utc_now()}},
        )
        return res.modified_count

    async def count_documents(self, status: Optional[str] = None) -> int:
        """Count total documents optionally filtered by status."""
        query = {"status": status} if status else {}
        return await self.collection.count_documents(query)

"""Ingestion run repository for MongoDB ingestion_runs collection."""
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
import pymongo

from app.db.session import MongoDBManager
from app.rag.models import IngestionStats, MongoIngestionRunRecord, utc_now


class IngestionRunRepository:
    """Repository for recording and querying pipeline execution runs."""

    def __init__(self, db: Optional[AsyncIOMotorDatabase] = None):
        self.db = db if db is not None else MongoDBManager.get_database()
        self.collection = self.db["ingestion_runs"]

    async def start_run(self, run_id: Optional[str] = None, documents_scanned: int = 0) -> str:
        """Create a new ingestion run record with status='RUNNING'."""
        rid = run_id or f"run_{uuid.uuid4().hex[:12]}"
        record = MongoIngestionRunRecord(
            run_id=rid,
            started_at=utc_now(),
            documents_scanned=documents_scanned,
            status="RUNNING",
        )
        await self.collection.insert_one(record.model_dump())
        return rid

    async def complete_run(self, run_id: str, stats: IngestionStats) -> None:
        """Update existing run record with final metrics, errors, and completed_at timestamp."""
        now = utc_now()
        update_data = {
            "completed_at": now,
            "documents_scanned": stats.documents_scanned,
            "documents_changed": stats.documents_changed,
            "chunks_scanned": stats.chunks_total,
            "new_chunks": stats.new_chunks,
            "changed_chunks": stats.changed_chunks,
            "unchanged_chunks": stats.unchanged_chunks,
            "removed_chunks": stats.removed_chunks,
            "embeddings_generated": stats.embeddings_generated,
            "embeddings_reused": stats.embeddings_reused,
            "status": stats.status,
            "errors": stats.errors,
        }
        await self.collection.update_one(
            {"run_id": run_id},
            {"$set": update_data},
            upsert=True,
        )

    async def get_run(self, run_id: str) -> Optional[Dict]:
        """Fetch a specific ingestion run by run_id."""
        return await self.collection.find_one({"run_id": run_id})

    async def get_latest_run(self) -> Optional[Dict]:
        """Fetch the most recent ingestion run."""
        return await self.collection.find_one(sort=[("started_at", pymongo.DESCENDING)])

    async def list_runs(self, limit: int = 10) -> List[Dict]:
        """List past ingestion runs ordered by start time descending."""
        cursor = self.collection.find().sort("started_at", pymongo.DESCENDING).limit(limit)
        return await cursor.to_list(length=limit)

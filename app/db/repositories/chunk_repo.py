"""Chunk repository for MongoDB rag_chunks collection."""
from datetime import datetime, timezone
from typing import Dict, List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
import pymongo

from app.db.session import MongoDBManager
from app.rag.models import EmbeddedChunk, MongoChunkRecord, utc_now


class ChunkRepository:
    """Repository for CRUD, bulk upsert, and querying operations on rag_chunks collection."""

    def __init__(self, db: Optional[AsyncIOMotorDatabase] = None):
        self.db = db if db is not None else MongoDBManager.get_database()
        self.collection = self.db["rag_chunks"]

    async def upsert_chunks(self, chunks: List[EmbeddedChunk]) -> int:
        """Bulk upsert embedded chunks into rag_chunks collection."""
        if not chunks:
            return 0

        now = utc_now()
        operations = []

        # Find existing chunks to preserve created_at
        chunk_ids = [c.chunk_id for c in chunks]
        existing_docs = {}
        cursor = self.collection.find({"chunk_id": {"$in": chunk_ids}}, {"chunk_id": 1, "created_at": 1})
        async for doc in cursor:
            existing_docs[doc["chunk_id"]] = doc.get("created_at", now)

        for c in chunks:
            created_at = existing_docs.get(c.chunk_id, now)
            record = MongoChunkRecord(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                entity_id=c.entity_id,
                domain=c.domain,
                document_type=c.document_type,
                title=c.title,
                content=c.content,
                content_hash=c.content_hash,
                chunk_index=c.chunk_index,
                embedding=c.embedding,
                embedding_model=c.embedding_model,
                embedding_dimension=c.embedding_dimension,
                embedding_version=c.embedding_version,
                metadata={
                    "entity_id": c.entity_id,
                    "domain": c.domain,
                    "document_id": c.document_id,
                },
                status="active",
                created_at=created_at,
                updated_at=now,
            )

            operations.append(
                pymongo.UpdateOne(
                    {"chunk_id": c.chunk_id},
                    {"$set": record.model_dump()},
                    upsert=True,
                )
            )

        if operations:
            res = await self.collection.bulk_write(operations, ordered=False)
            return res.upserted_count + res.modified_count
        return 0

    async def get_chunk(self, chunk_id: str) -> Optional[Dict]:
        """Fetch a single chunk by chunk_id."""
        return await self.collection.find_one({"chunk_id": chunk_id})

    async def get_chunks_by_document(self, document_id: str) -> List[Dict]:
        """Fetch all active chunks for a document."""
        cursor = self.collection.find({"document_id": document_id, "status": "active"}).sort("chunk_index", pymongo.ASCENDING)
        return await cursor.to_list(length=1000)

    async def get_all_active_chunks(self) -> List[Dict]:
        """Fetch all active chunks with content_hash and embedding for change detection caching."""
        cursor = self.collection.find({"status": "active"})
        return await cursor.to_list(length=10000)

    async def mark_chunks_removed(self, chunk_ids: List[str]) -> int:
        """Mark chunks as removed."""
        if not chunk_ids:
            return 0
        res = await self.collection.update_many(
            {"chunk_id": {"$in": chunk_ids}},
            {"$set": {"status": "removed", "updated_at": utc_now()}},
        )
        return res.modified_count

    async def count_chunks(self, status: Optional[str] = None) -> int:
        """Count total chunks optionally filtered by status."""
        query = {"status": status} if status else {}
        return await self.collection.count_documents(query)

    async def search_text(
        self,
        query: str,
        limit: int = 20,
        domain: Optional[str] = None,
    ) -> List[Dict]:
        """Perform sparse / lexical search using MongoDB text index.
        
        Falls back to regex search if text index is not ready or query fails.
        """
        if not query or not query.strip():
            return []

        match_filter: Dict = {"status": "active"}
        if domain:
            match_filter["domain"] = domain

        # 1. Try MongoDB native $text search
        try:
            match_filter["$text"] = {"$search": query.strip()}
            cursor = self.collection.find(
                match_filter,
                {"score": {"$meta": "textScore"}},
            ).sort([("score", {"$meta": "textScore"})]).limit(limit)

            results = await cursor.to_list(length=limit)
            if results:
                return results
        except Exception:
            # Fallback to regex matching
            pass

        # 2. Fallback: Case-insensitive regex matching across active chunks
        regex_filter: Dict = {"status": "active"}
        if domain:
            regex_filter["domain"] = domain

        tokens = [t.strip() for t in query.split() if len(t.strip()) > 2]
        if not tokens:
            tokens = [query.strip()]

        or_clauses = []
        for token in tokens:
            pattern = {"$regex": token, "$options": "i"}
            or_clauses.extend([
                {"title": pattern},
                {"entity_id": pattern},
                {"content": pattern},
            ])

        regex_filter["$or"] = or_clauses
        cursor = self.collection.find(regex_filter).limit(limit)
        return await cursor.to_list(length=limit)

    async def search_vector_atlas(
        self,
        query_embedding: List[float],
        limit: int = 20,
        domain: Optional[str] = None,
        index_name: str = "vector_index",
    ) -> List[Dict]:
        """Perform dense vector search using MongoDB Atlas $vectorSearch aggregation stage."""
        pipeline: List[Dict] = [
            {
                "$vectorSearch": {
                    "index": index_name,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": limit * 10,
                    "limit": limit,
                    "filter": {"domain": {"$eq": domain}} if domain else {},
                }
            },
            {
                "$project": {
                    "score": {"$meta": "vectorSearchScore"},
                    "chunk_id": 1,
                    "document_id": 1,
                    "entity_id": 1,
                    "domain": 1,
                    "title": 1,
                    "content": 1,
                    "metadata": 1,
                    "chunk_index": 1,
                    "status": 1,
                }
            }
        ]
        cursor = self.collection.aggregate(pipeline)
        return await cursor.to_list(length=limit)

    async def search_vectors_cosine(
        self,
        query_embedding: List[float],
        limit: int = 20,
        domain: Optional[str] = None,
    ) -> List[Dict]:
        """Compute cosine similarity across active MongoDB chunks in-memory.
        
        Guarantees local vector search works without requiring Atlas Vector Search setup.
        """
        if not query_embedding:
            return []

        query = {"status": "active"}
        if domain:
            query["domain"] = domain

        cursor = self.collection.find(
            query,
            {
                "chunk_id": 1,
                "document_id": 1,
                "entity_id": 1,
                "domain": 1,
                "title": 1,
                "content": 1,
                "metadata": 1,
                "chunk_index": 1,
                "embedding": 1,
                "status": 1,
            },
        )
        chunks = await cursor.to_list(length=5000)
        if not chunks:
            return []

        # Cosine similarity calculation: dot(u, v) / (norm(u) * norm(v))
        import math
        q_norm = math.sqrt(sum(x * x for x in query_embedding))
        if q_norm == 0:
            return []

        scored = []
        for c in chunks:
            c_emb = c.get("embedding")
            if not c_emb or len(c_emb) != len(query_embedding):
                continue
            dot = sum(a * b for a, b in zip(query_embedding, c_emb))
            c_norm = math.sqrt(sum(y * y for y in c_emb))
            score = dot / (q_norm * c_norm) if c_norm > 0 else 0.0
            scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_chunks = []
        for score, c in scored[:limit]:
            c_dict = dict(c)
            c_dict["score"] = score
            top_chunks.append(c_dict)

        return top_chunks

"""MongoDB connection session manager with sanitized logging and lifecycle management."""
import asyncio
import logging
import re
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.core.config import settings

logger = logging.getLogger(__name__)


def sanitize_mongo_url(url: Optional[str]) -> str:
    """Mask credentials in MongoDB connection string for safe logging."""
    if not url:
        return "<empty>"
    # Mask password or user:password in mongodb URI
    sanitized = re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", url)
    return sanitized


class MongoDBManager:
    """Manages AsyncIOMotorClient lifecycle and database access."""

    _client: Optional[AsyncIOMotorClient] = None
    _loop = None

    @classmethod
    def get_client(cls, uri: Optional[str] = None) -> AsyncIOMotorClient:
        """Get or create singleton AsyncIOMotorClient bound to current event loop."""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if cls._client is None or uri is not None or (current_loop is not None and cls._loop != current_loop):
            conn_uri = uri or settings.mongo_url
            if not conn_uri:
                raise ValueError("MONGO_URL environment variable is not configured.")

            if cls._client is not None and cls._loop != current_loop:
                try:
                    cls._client.close()
                except Exception:
                    pass

            logger.info("Initializing MongoDB client connection to: %s", sanitize_mongo_url(conn_uri))
            client = AsyncIOMotorClient(
                conn_uri,
                serverSelectionTimeoutMS=settings.mongo_timeout_ms,
                connectTimeoutMS=settings.mongo_timeout_ms,
            )
            if uri is None:
                cls._client = client
                cls._loop = current_loop
                return cls._client
            return client

        return cls._client

    @classmethod
    def get_database(
        cls, db_name: Optional[str] = None, client: Optional[AsyncIOMotorClient] = None
    ) -> AsyncIOMotorDatabase:
        """Get database instance by name (defaults to 'hopca_database')."""
        active_client = client or cls.get_client()
        name = db_name or settings.mongo_db_name
        return active_client[name]

    @classmethod
    async def ping(cls, client: Optional[AsyncIOMotorClient] = None) -> bool:
        """Ping MongoDB deployment to verify active connection."""
        active_client = client or cls.get_client()
        try:
            res = await active_client.admin.command("ping")
            return bool(res.get("ok") == 1)
        except Exception as e:
            logger.error("MongoDB ping failed: %s", e)
            return False

    @classmethod
    def close(cls) -> None:
        """Close global MongoDB client connection."""
        if cls._client is not None:
            logger.info("Closing MongoDB client connection.")
            cls._client.close()
            cls._client = None


def get_db_client() -> AsyncIOMotorClient:
    """Convenience getter for singleton Motor client."""
    return MongoDBManager.get_client()


def get_database(db_name: Optional[str] = None) -> AsyncIOMotorDatabase:
    """Convenience getter for Motor database."""
    return MongoDBManager.get_database(db_name=db_name)

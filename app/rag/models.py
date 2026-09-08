"""Pydantic models and data structures for the HOPCA RAG pipeline."""
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ValidationSeverity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


class ValidationCode(str, Enum):
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    INVALID_UTF8 = "INVALID_UTF8"
    EMPTY_FILE = "EMPTY_FILE"
    MISSING_FRONTMATTER = "MISSING_FRONTMATTER"
    INVALID_FRONTMATTER = "INVALID_FRONTMATTER"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    INVALID_DOMAIN = "INVALID_DOMAIN"
    DUPLICATE_DOCUMENT_ID = "DUPLICATE_DOCUMENT_ID"
    DUPLICATE_ENTITY_ID = "DUPLICATE_ENTITY_ID"
    INVALID_ENTITY_FORMAT = "INVALID_ENTITY_FORMAT"
    UNRESOLVED_CROSS_REFERENCE = "UNRESOLVED_CROSS_REFERENCE"
    DYNAMIC_CONTENT_DETECTED = "DYNAMIC_CONTENT_DETECTED"
    MISSING_SYNTHETIC_FLAG = "MISSING_SYNTHETIC_FLAG"
    EMPTY_SECTION = "EMPTY_SECTION"
    EXTREMELY_SHORT_DOC = "EXTREMELY_SHORT_DOC"
    GENERAL_VALIDATION_ERROR = "GENERAL_VALIDATION_ERROR"


class ValidationIssue(BaseModel):
    severity: ValidationSeverity
    file: str
    code: ValidationCode
    message: str
    line_number: Optional[int] = None
    context: Optional[str] = None


class ValidationSummary(BaseModel):
    total_files: int = 0
    valid_files: int = 0
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    total_entities: int = 0


class ValidationResult(BaseModel):
    valid: bool
    summary: ValidationSummary
    issues: List[ValidationIssue] = Field(default_factory=list)


class KBDocument(BaseModel):
    document_id: str
    title: str
    domain: str
    document_type: str = "guideline"
    version: str = "1.0"
    source: str = "HOPCA General Hospital"
    status: str = "active"
    synthetic_data: bool = True
    file_path: str
    raw_content: str
    cleaned_content: Optional[str] = None


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    entity_id: Optional[str] = None
    domain: str
    document_type: str
    title: str
    chunk_index: int
    content: str
    content_hash: str


class EmbeddedChunk(Chunk):
    embedding: List[float]
    embedding_model: str
    embedding_dimension: int
    embedding_version: str


class ChunkChangeStatus(str, Enum):
    NEW = "NEW"
    CHANGED = "CHANGED"
    UNCHANGED = "UNCHANGED"
    REMOVED = "REMOVED"


class IngestionStats(BaseModel):
    documents_scanned: int = 0
    documents_valid: int = 0
    documents_changed: int = 0
    chunks_total: int = 0
    new_chunks: int = 0
    changed_chunks: int = 0
    unchanged_chunks: int = 0
    removed_chunks: int = 0
    embeddings_generated: int = 0
    embeddings_reused: int = 0
    status: str = "SUCCESS"
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


def utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


class MongoDocumentRecord(BaseModel):
    document_id: str
    title: str
    domain: str
    document_type: str = "guideline"
    source: str = "HOPCA General Hospital"
    version: str = "1.0"
    content_hash: str
    status: str = "active"
    synthetic_data: bool = True
    file_path: str
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class MongoChunkRecord(BaseModel):
    chunk_id: str
    document_id: str
    entity_id: Optional[str] = None
    domain: str
    document_type: str = "guideline"
    title: str
    content: str
    content_hash: str
    chunk_index: int
    embedding: List[float] = Field(default_factory=list)
    embedding_model: str = "text-embedding-004"
    embedding_dimension: int = 768
    embedding_version: str = "1.0"
    metadata: Dict = Field(default_factory=dict)
    status: str = "active"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class MongoIngestionRunRecord(BaseModel):
    run_id: str
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: Optional[datetime] = None
    documents_scanned: int = 0
    documents_changed: int = 0
    chunks_scanned: int = 0
    new_chunks: int = 0
    changed_chunks: int = 0
    unchanged_chunks: int = 0
    removed_chunks: int = 0
    embeddings_generated: int = 0
    embeddings_reused: int = 0
    status: str = "RUNNING"
    errors: List[str] = Field(default_factory=list)


class MongoSystemMetadataRecord(BaseModel):
    pipeline_version: str = "1.0"
    embedding_model: str = "text-embedding-004"
    embedding_dimension: int = 768
    chunking_strategy: str = "semantic"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    last_successful_ingestion: Optional[datetime] = None
    schema_version: str = "1.0"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    entity_id: Optional[str] = None
    domain: str
    title: str
    content: str
    chunk_index: int = 0
    score: float = 0.0
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    rrf_score: float = 0.0
    metadata: Dict = Field(default_factory=dict)


class RAGSearchResult(BaseModel):
    query: str
    chunks: List[RetrievedChunk] = Field(default_factory=list)
    total_retrieved: int = 0
    dense_count: int = 0
    sparse_count: int = 0


class RAGContextResult(BaseModel):
    query: str
    formatted_context: str
    sources: List[Dict[str, str]] = Field(default_factory=list)
    chunks_used: int = 0
    has_context: bool = True

"""Individual validation rules for HOPCA knowledge base verification."""
import os
from pathlib import Path
import re
from typing import Dict, List, Optional, Set, Tuple

from app.rag.models import (
    KBDocument,
    ValidationCode,
    ValidationIssue,
    ValidationSeverity,
)

ALLOWED_DOMAINS: Set[str] = {
    "hospital",
    "departments",
    "doctors",
    "staff",
    "services",
    "appointments",
    "admission",
    "discharge",
    "billing",
    "policies",
    "emergency",
    "faq",
}

REQUIRED_FRONTMATTER_FIELDS = [
    ("document_id", str),
    ("title", str),
    ("domain", str),
    ("document_type", str),
    ("version", str),
    ("source", str),
    ("status", str),
    ("synthetic_data", bool),
]

DYNAMIC_CONTENT_PATTERNS = [
    r"EXEC\s+sp_",
    r"INSERT\s+INTO\s+",
    r"UPDATE\s+\w+\s+SET",
    r"DELETE\s+FROM\s+",
    r"live\s+websocket\s+feed",
    r"real-time\s+telemetry\s+stream",
]


def validate_file_discovery(kb_path: Path) -> Tuple[List[Path], List[ValidationIssue]]:
    """Scan directory, check for existence, non-markdown files, and return valid markdown file paths."""
    issues: List[ValidationIssue] = []
    if not kb_path.exists():
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                file=str(kb_path),
                code=ValidationCode.FILE_NOT_FOUND,
                message=f"Knowledge base directory does not exist: {kb_path}",
            )
        )
        return [], issues

    md_files: List[Path] = []
    for root, _, files in os.walk(kb_path):
        for file in files:
            file_path = Path(root) / file
            if file.lower() == "readme.md":
                continue
            if not file.endswith(".md"):
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        file=str(file_path),
                        code=ValidationCode.GENERAL_VALIDATION_ERROR,
                        message=f"Non-markdown file found in knowledge base: {file}",
                    )
                )
            else:
                md_files.append(file_path)

    if not md_files and kb_path.exists():
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                file=str(kb_path),
                code=ValidationCode.EMPTY_FILE,
                message="No markdown files found in knowledge base directory.",
            )
        )

    return sorted(md_files), issues


def validate_markdown(file_path: Path) -> Tuple[Optional[str], List[ValidationIssue]]:
    """Validate UTF-8 encoding and check that the file is not empty."""
    issues: List[ValidationIssue] = []
    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                file=str(file_path),
                code=ValidationCode.INVALID_UTF8,
                message=f"File is not valid UTF-8: {e}",
            )
        )
        return None, issues
    except Exception as e:
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                file=str(file_path),
                code=ValidationCode.GENERAL_VALIDATION_ERROR,
                message=f"Failed to read file: {e}",
            )
        )
        return None, issues

    if not content.strip():
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                file=str(file_path),
                code=ValidationCode.EMPTY_FILE,
                message=f"Document file is completely empty: {file_path.name}",
            )
        )
        return content, issues

    return content, issues


def validate_frontmatter(metadata: Dict, file_path: Path) -> List[ValidationIssue]:
    """Validate presence, types, and values of required front-matter fields."""
    issues: List[ValidationIssue] = []

    if not metadata:
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                file=str(file_path),
                code=ValidationCode.MISSING_FRONTMATTER,
                message="Missing YAML front-matter block (---) at top of document.",
            )
        )
        return issues

    for field_name, expected_type in REQUIRED_FRONTMATTER_FIELDS:
        if field_name not in metadata:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    file=str(file_path),
                    code=ValidationCode.MISSING_REQUIRED_FIELD,
                    message=f"Missing required front-matter field: '{field_name}'",
                )
            )
        else:
            val = metadata[field_name]
            if not isinstance(val, expected_type):
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        file=str(file_path),
                        code=ValidationCode.INVALID_FRONTMATTER,
                        message=f"Front-matter field '{field_name}' expected {expected_type.__name__}, got {type(val).__name__}",
                    )
                )

    domain = metadata.get("domain")
    if domain and domain not in ALLOWED_DOMAINS:
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                file=str(file_path),
                code=ValidationCode.INVALID_DOMAIN,
                message=f"Invalid domain '{domain}'. Allowed domains: {sorted(list(ALLOWED_DOMAINS))}",
            )
        )

    status = metadata.get("status")
    if status and status not in {"active", "draft", "archived"}:
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                file=str(file_path),
                code=ValidationCode.INVALID_FRONTMATTER,
                message=f"Unknown status value: '{status}' (recommended: 'active', 'draft', 'archived')",
            )
        )

    return issues


def validate_document_ids(documents: List[KBDocument]) -> List[ValidationIssue]:
    """Ensure document_id values are unique across all documents."""
    issues: List[ValidationIssue] = []
    seen: Dict[str, str] = {}  # doc_id -> file_path

    for doc in documents:
        doc_id = doc.document_id
        if doc_id in seen:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    file=doc.file_path,
                    code=ValidationCode.DUPLICATE_DOCUMENT_ID,
                    message=f"Duplicate document_id '{doc_id}' found in '{doc.file_path}' (already defined in '{seen[doc_id]}')",
                )
            )
        else:
            seen[doc_id] = doc.file_path

    return issues


def extract_entities_from_doc(doc: KBDocument) -> List[Tuple[str, int]]:
    """Extract (entity_id, line_number) from a document's content."""
    entities: List[Tuple[str, int]] = []
    lines = doc.raw_content.split("\n")

    for line_idx, line in enumerate(lines, start=1):
        if line.startswith("## "):
            heading = line.lstrip("#").strip()
            # Match Entity: ID or Doctor: DOC-001
            m1 = re.search(r"(?:Doctor|Department|Staff|Service|Entity):\s*([A-Za-z0-9_-]+)", heading, re.IGNORECASE)
            if m1:
                entities.append((m1.group(1).strip(), line_idx))
                continue

            # Match HOSP-001: ... or FAQ-001
            m2 = re.search(r"^([A-Z]{3,5}-\d{3,}(?:-[A-Z0-9]+)?)(?::|\s|$)", heading)
            if m2:
                entities.append((m2.group(1).strip(), line_idx))
                continue

            m3 = re.search(r"\b([A-Z]{2,6}-(?:[A-Z0-9]+-)?\d{3,})\b", heading)
            if m3:
                entities.append((m3.group(1).strip(), line_idx))

    return entities


def validate_entity_ids(documents: List[KBDocument]) -> Tuple[Dict[str, str], List[ValidationIssue]]:
    """Validate entity ID uniqueness across documents and return entity_registry {entity_id: file_path}."""
    issues: List[ValidationIssue] = []
    registry: Dict[str, str] = {}

    for doc in documents:
        doc_entities = extract_entities_from_doc(doc)
        for entity_id, line_no in doc_entities:
            if entity_id in registry:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        file=doc.file_path,
                        code=ValidationCode.DUPLICATE_ENTITY_ID,
                        message=f"Duplicate entity_id '{entity_id}' detected (also in '{registry[entity_id]}')",
                        line_number=line_no,
                        context=f"Entity ID: {entity_id}",
                    )
                )
            else:
                registry[entity_id] = doc.file_path

    return registry, issues


def validate_cross_references(
    documents: List[KBDocument], entity_registry: Dict[str, str]
) -> List[ValidationIssue]:
    """Check entity cross-references (e.g. references to DOC-xxx, DEPT-xxx) against known entities."""
    issues: List[ValidationIssue] = []
    # Pattern to find potential entity IDs mentioned in markdown body
    ref_pattern = re.compile(r"\b([A-Z]{3,5}-(?:[A-Z0-9]+-)?\d{3,})\b")

    for doc in documents:
        lines = doc.raw_content.split("\n")
        for line_idx, line in enumerate(lines, start=1):
            # Skip headings where entities are defined
            if line.startswith("## "):
                continue
            matches = ref_pattern.findall(line)
            for ref in matches:
                # If ref is not in entity registry and not a standard code (e.g. ICD-10, CPT)
                if ref not in entity_registry and not ref.startswith("ICD") and not ref.startswith("CPT"):
                    issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            file=doc.file_path,
                            code=ValidationCode.UNRESOLVED_CROSS_REFERENCE,
                            message=f"Potential unresolvable entity reference '{ref}'",
                            line_number=line_idx,
                            context=line.strip()[:100],
                        )
                    )

    return issues


def validate_static_boundary(content: str, file_path: str) -> List[ValidationIssue]:
    """Detect dynamic runtime claims, SQL queries, or unauthorized dynamic operations."""
    issues: List[ValidationIssue] = []
    for pattern in DYNAMIC_CONTENT_PATTERNS:
        matches = list(re.finditer(pattern, content, re.IGNORECASE))
        for m in matches:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    file=file_path,
                    code=ValidationCode.DYNAMIC_CONTENT_DETECTED,
                    message=f"Potential dynamic operation syntax detected: '{m.group(0)}'",
                    context=m.group(0),
                )
            )
    return issues


def validate_synthetic_metadata(metadata: Dict, content: str, file_path: str) -> List[ValidationIssue]:
    """Ensure synthetic_data flag is true for synthetic knowledge base documents."""
    issues: List[ValidationIssue] = []
    if not metadata.get("synthetic_data", True):
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                file=file_path,
                code=ValidationCode.MISSING_SYNTHETIC_FLAG,
                message="synthetic_data should be explicitly set to true for HOPCA synthetic dataset",
            )
        )
    return issues


def validate_content_quality(content: str, file_path: str) -> List[ValidationIssue]:
    """Check for empty headings, empty sections, or extremely short documents."""
    issues: List[ValidationIssue] = []
    if len(content.strip()) < 50:
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                file=file_path,
                code=ValidationCode.EXTREMELY_SHORT_DOC,
                message=f"Document content is extremely short (< 50 characters): {len(content.strip())} chars",
            )
        )

    # Empty heading check
    lines = content.split("\n")
    for line_idx, line in enumerate(lines, start=1):
        if re.match(r"^#{1,6}\s*$", line):
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    file=file_path,
                    code=ValidationCode.EMPTY_SECTION,
                    message="Empty heading line detected",
                    line_number=line_idx,
                )
            )

    return issues

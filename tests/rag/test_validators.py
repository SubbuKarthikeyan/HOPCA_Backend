"""Tests for Knowledge Base Validator and Validation Rules (Scenarios 1, 2, 3, 4, 5)."""
from pathlib import Path
import pytest

from app.rag.models import ValidationCode, ValidationSeverity
from app.rag.validators.validator import KnowledgeBaseValidator


@pytest.fixture
def actual_kb_path():
    return Path(__file__).resolve().parent.parent.parent.parent / "knowledge_base"


def test_scenario_1_valid_kb(actual_kb_path):
    """Scenario 1: Valid KB -> PASS"""
    validator = KnowledgeBaseValidator()
    result = validator.validate(actual_kb_path)
    assert result.valid is True
    assert result.summary.total_files >= 12
    assert result.summary.error_count == 0


def test_scenario_2_invalid_metadata(tmp_path):
    """Scenario 2: Invalid metadata -> FAIL (Missing required field or wrong domain)"""
    bad_doc = tmp_path / "bad.md"
    bad_doc.write_text(
        """---
document_id: bad_doc
title: Bad Doc
domain: non_existent_domain
---
## Doctor: DOC-999
Valid body content with enough length to pass quality checks.
""",
        encoding="utf-8",
    )

    validator = KnowledgeBaseValidator()
    result = validator.validate(tmp_path)
    assert result.valid is False
    assert any(i.code == ValidationCode.INVALID_DOMAIN for i in result.issues)
    assert any(i.code == ValidationCode.MISSING_REQUIRED_FIELD for i in result.issues)


def test_scenario_3_duplicate_document_id(tmp_path):
    """Scenario 3: Duplicate document ID -> FAIL"""
    doc1 = tmp_path / "doc1.md"
    doc1.write_text(
        """---
document_id: duplicate_id
title: Doc One
domain: hospital
document_type: guideline
version: "1.0"
source: HOPCA
status: active
synthetic_data: true
---
## HOSP-001: General Overview
Some valid content for doc1 to pass length requirements.
""",
        encoding="utf-8",
    )
    doc2 = tmp_path / "doc2.md"
    doc2.write_text(
        """---
document_id: duplicate_id
title: Doc Two
domain: hospital
document_type: guideline
version: "1.0"
source: HOPCA
status: active
synthetic_data: true
---
## HOSP-002: Another Overview
Some valid content for doc2 to pass length requirements.
""",
        encoding="utf-8",
    )

    validator = KnowledgeBaseValidator()
    result = validator.validate(tmp_path)
    assert result.valid is False
    assert any(i.code == ValidationCode.DUPLICATE_DOCUMENT_ID for i in result.issues)


def test_scenario_4_empty_document(tmp_path):
    """Scenario 4: Empty document -> FAIL"""
    empty_doc = tmp_path / "empty.md"
    empty_doc.write_text("", encoding="utf-8")

    validator = KnowledgeBaseValidator()
    result = validator.validate(tmp_path)
    assert result.valid is False
    assert any(i.code == ValidationCode.EMPTY_FILE for i in result.issues)


def test_scenario_5_duplicate_entity_id(tmp_path):
    """Scenario 5: Duplicate entity ID -> detected"""
    doc1 = tmp_path / "doc1.md"
    doc1.write_text(
        """---
document_id: doc_a
title: Doc A
domain: doctors
document_type: directory
version: "1.0"
source: HOPCA
status: active
synthetic_data: true
---
## Doctor: DOC-001
Dr. Alice Smith is a senior cardiologist.
""",
        encoding="utf-8",
    )
    doc2 = tmp_path / "doc2.md"
    doc2.write_text(
        """---
document_id: doc_b
title: Doc B
domain: doctors
document_type: directory
version: "1.0"
source: HOPCA
status: active
synthetic_data: true
---
## Doctor: DOC-001
Dr. Bob Jones with colliding ID.
""",
        encoding="utf-8",
    )

    validator = KnowledgeBaseValidator()
    result = validator.validate(tmp_path)
    assert result.valid is False
    assert any(i.code == ValidationCode.DUPLICATE_ENTITY_ID for i in result.issues)

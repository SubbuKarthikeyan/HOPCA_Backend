"""Knowledge base validator orchestrator for HOPCA."""
from pathlib import Path
from typing import List, Optional, Union

from app.rag.loaders.markdown_loader import MarkdownLoader
from app.rag.models import (
    KBDocument,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    ValidationSummary,
)
from app.rag.validators.rules import (
    validate_content_quality,
    validate_cross_references,
    validate_document_ids,
    validate_entity_ids,
    validate_file_discovery,
    validate_frontmatter,
    validate_markdown,
    validate_static_boundary,
    validate_synthetic_metadata,
)


class KnowledgeBaseValidator:
    """Orchestrates comprehensive validation across all knowledge base documents."""

    def __init__(self, strict: bool = False):
        self.strict = strict

    def validate(self, kb_path: Union[str, Path]) -> ValidationResult:
        """Execute full validation suite on the specified knowledge base directory.
        
        This method is strictly read-only and produces no side-effects.
        """
        path = Path(kb_path).resolve()
        issues: List[ValidationIssue] = []

        # 1. Discover files
        files, discovery_issues = validate_file_discovery(path)
        issues.extend(discovery_issues)

        if not files:
            summary = ValidationSummary(
                total_files=0,
                valid_files=0,
                error_count=sum(1 for i in issues if i.severity == ValidationSeverity.ERROR),
                warning_count=sum(1 for i in issues if i.severity == ValidationSeverity.WARNING),
                info_count=sum(1 for i in issues if i.severity == ValidationSeverity.INFO),
            )
            return ValidationResult(valid=False, summary=summary, issues=issues)

        documents: List[KBDocument] = []
        valid_files_count = 0

        # 2. File-by-file validation
        for file_path in files:
            file_had_error = False

            # Check markdown & UTF-8
            raw_text, md_issues = validate_markdown(file_path)
            issues.extend(md_issues)
            if any(i.severity == ValidationSeverity.ERROR for i in md_issues) or raw_text is None:
                file_had_error = True
                continue

            # Parse front-matter
            metadata, body = MarkdownLoader.parse_frontmatter(raw_text)
            fm_issues = validate_frontmatter(metadata, file_path)
            issues.extend(fm_issues)
            if any(i.severity == ValidationSeverity.ERROR for i in fm_issues):
                file_had_error = True

            # Content quality & boundaries
            cq_issues = validate_content_quality(body, str(file_path))
            issues.extend(cq_issues)
            if any(i.severity == ValidationSeverity.ERROR for i in cq_issues):
                file_had_error = True

            sb_issues = validate_static_boundary(body, str(file_path))
            issues.extend(sb_issues)

            sm_issues = validate_synthetic_metadata(metadata, body, str(file_path))
            issues.extend(sm_issues)

            # Build KBDocument
            doc_id = str(metadata.get("document_id") or file_path.stem)
            title = str(metadata.get("title") or doc_id.replace("_", " ").title())
            domain = str(metadata.get("domain") or file_path.parent.name)
            doc_type = str(metadata.get("document_type") or "guideline")
            version = str(metadata.get("version") or "1.0")
            source = str(metadata.get("source") or "HOPCA General Hospital")
            status = str(metadata.get("status") or "active")
            synthetic_data = bool(metadata.get("synthetic_data", True))

            doc = KBDocument(
                document_id=doc_id,
                title=title,
                domain=domain,
                document_type=doc_type,
                version=version,
                source=source,
                status=status,
                synthetic_data=synthetic_data,
                file_path=str(file_path),
                raw_content=body,
            )
            documents.append(doc)

            if not file_had_error:
                valid_files_count += 1

        # 3. Cross-document validations
        doc_id_issues = validate_document_ids(documents)
        issues.extend(doc_id_issues)

        entity_registry, entity_id_issues = validate_entity_ids(documents)
        issues.extend(entity_id_issues)

        cross_ref_issues = validate_cross_references(documents, entity_registry)
        issues.extend(cross_ref_issues)

        error_count = sum(1 for i in issues if i.severity == ValidationSeverity.ERROR)
        warning_count = sum(1 for i in issues if i.severity == ValidationSeverity.WARNING)
        info_count = sum(1 for i in issues if i.severity == ValidationSeverity.INFO)

        is_valid = error_count == 0
        if self.strict and warning_count > 0:
            is_valid = False

        summary = ValidationSummary(
            total_files=len(files),
            valid_files=valid_files_count if error_count == 0 else max(0, len(files) - error_count),
            error_count=error_count,
            warning_count=warning_count,
            info_count=info_count,
            total_entities=len(entity_registry),
        )

        return ValidationResult(valid=is_valid, summary=summary, issues=issues)

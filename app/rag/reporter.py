"""Reporting utilities for HOPCA RAG pipeline execution."""
import json
from app.rag.models import IngestionStats


class Reporter:
    """Formats ingestion pipeline statistics into human-readable tables or structured JSON."""

    @staticmethod
    def format_human_report(stats: IngestionStats) -> str:
        status_symbol = "PASSED [OK]" if stats.status == "SUCCESS" else "FAILED [ERROR]"
        lines = [
            "=" * 70,
            " HOPCA RAG INGESTION PIPELINE EXECUTION REPORT",
            "=" * 70,
            f"Overall Status       : {status_symbol}",
            f"Documents Scanned    : {stats.documents_scanned}",
            f"Documents Valid      : {stats.documents_valid}",
            f"Documents Changed    : {stats.documents_changed}",
            "-" * 70,
            f"Total Chunks Processed : {stats.chunks_total}",
            f"  - New Chunks       : {stats.new_chunks}",
            f"  - Changed Chunks   : {stats.changed_chunks}",
            f"  - Unchanged Chunks : {stats.unchanged_chunks}",
            f"  - Removed Chunks   : {stats.removed_chunks}",
            "-" * 70,
            f"Embeddings Generated : {stats.embeddings_generated}",
            f"Embeddings Reused    : {stats.embeddings_reused}",
            "=" * 70,
        ]

        if stats.errors:
            lines.append("\nErrors:")
            for err in stats.errors:
                lines.append(f"  [ERROR] {err}")

        if stats.warnings:
            lines.append("\nWarnings:")
            for warn in stats.warnings:
                lines.append(f"  [WARN]  {warn}")

        return "\n".join(lines)

    @staticmethod
    def format_json_report(stats: IngestionStats) -> str:
        return json.dumps(stats.model_dump(), indent=2)

"""CLI entry point for knowledge base validation."""
import argparse
import json
import sys
from pathlib import Path

from app.core.config import settings
from app.rag.validators.validator import KnowledgeBaseValidator


def main():
    parser = argparse.ArgumentParser(description="Validate HOPCA Knowledge Base documents and structure.")
    parser.add_argument(
        "--path",
        type=str,
        default=settings.knowledge_base_dir,
        help="Path to knowledge base directory (defaults to config setting)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output validation results as structured JSON",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors",
    )

    args = parser.parse_args()
    kb_path = Path(args.path)

    validator = KnowledgeBaseValidator(strict=args.strict)
    result = validator.validate(kb_path)

    if sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    if args.json:
        print(json.dumps(result.model_dump(), indent=2))
    else:
        print("=" * 70)
        print(" HOPCA KNOWLEDGE BASE VALIDATION REPORT")
        print("=" * 70)
        print(f"Status        : {'PASSED [OK]' if result.valid else 'FAILED [ERROR]'}")
        print(f"Total Files   : {result.summary.total_files}")
        print(f"Valid Files   : {result.summary.valid_files}")
        print(f"Total Entities: {result.summary.total_entities}")
        print(f"Errors        : {result.summary.error_count}")
        print(f"Warnings      : {result.summary.warning_count}")
        print(f"Info Messages : {result.summary.info_count}")
        print("-" * 70)

        if result.issues:
            print("\nIssues Found:")
            for issue in result.issues:
                prefix = f"[{issue.severity.value}]"
                line_info = f" (line {issue.line_number})" if issue.line_number else ""
                print(f"  {prefix:<10} {Path(issue.file).name}{line_info}: {issue.message}")
        else:
            print("\nNo issues detected. Knowledge base is valid.")
        print("=" * 70)

    sys.exit(0 if result.valid else 1)


if __name__ == "__main__":
    main()

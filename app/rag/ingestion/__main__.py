"""CLI entry point for HOPCA RAG data ingestion pipeline."""
import argparse
import asyncio
import sys
from pathlib import Path

from app.core.config import settings
from app.rag.ingestion.pipeline import IngestionPipeline
from app.rag.reporter import Reporter


async def _main():
    parser = argparse.ArgumentParser(description="Run HOPCA RAG Data Preparation Pipeline.")
    parser.add_argument(
        "--path",
        type=str,
        default=settings.knowledge_base_dir,
        help="Path to knowledge base directory (defaults to config setting)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output ingestion stats as JSON",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat validation warnings as errors",
    )

    args = parser.parse_args()
    kb_path = Path(args.path)

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    pipeline = IngestionPipeline(kb_path=kb_path, strict_validation=args.strict)
    stats, _ = await pipeline.run()

    if args.json:
        print(Reporter.format_json_report(stats))
    else:
        print(Reporter.format_human_report(stats))

    sys.exit(0 if stats.status == "SUCCESS" else 1)


def main():
    asyncio.run(_main())


if __name__ == "__main__":
    main()

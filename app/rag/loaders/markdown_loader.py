"""Markdown file loader and front-matter parser for HOPCA knowledge base."""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml

from app.rag.models import KBDocument


class MarkdownLoader:
    """Discovers and parses markdown files with YAML front-matter."""

    @staticmethod
    def parse_frontmatter(content: str) -> Tuple[Dict, str]:
        """Extract YAML front-matter dictionary and body content from markdown string.
        
        Returns:
            Tuple of (metadata_dict, body_content_str)
        """
        content_stripped = content.strip()
        if not content_stripped.startswith("---"):
            return {}, content

        # Find the second '---' delimiter
        parts = content_stripped.split("---", 2)
        if len(parts) < 3:
            return {}, content

        yaml_str = parts[1]
        body = parts[2].lstrip("\r\n")

        try:
            metadata = yaml.safe_load(yaml_str) or {}
            if not isinstance(metadata, dict):
                metadata = {}
            return metadata, body
        except yaml.YAMLError:
            return {}, content

    @classmethod
    def discover_files(cls, kb_path: str | Path) -> List[Path]:
        """Find all relevant markdown files in the knowledge base directory."""
        path = Path(kb_path).resolve()
        if not path.exists() or not path.is_dir():
            return []

        md_files: List[Path] = []
        for root, _, files in os.walk(path):
            for file in sorted(files):
                if file.endswith(".md") and file.lower() != "readme.md":
                    md_files.append(Path(root) / file)
        return sorted(md_files)

    @classmethod
    def load_document(cls, file_path: str | Path) -> KBDocument:
        """Load and parse a single markdown file into a KBDocument."""
        p = Path(file_path).resolve()
        content = p.read_text(encoding="utf-8")
        metadata, body = cls.parse_frontmatter(content)

        doc_id = str(metadata.get("document_id") or p.stem)
        title = str(metadata.get("title") or doc_id.replace("_", " ").title())
        domain = str(metadata.get("domain") or p.parent.name)
        document_type = str(metadata.get("document_type") or "guideline")
        version = str(metadata.get("version") or "1.0")
        source = str(metadata.get("source") or "HOPCA General Hospital")
        status = str(metadata.get("status") or "active")
        synthetic_data = bool(metadata.get("synthetic_data", True))

        return KBDocument(
            document_id=doc_id,
            title=title,
            domain=domain,
            document_type=document_type,
            version=version,
            source=source,
            status=status,
            synthetic_data=synthetic_data,
            file_path=str(p),
            raw_content=body,
        )

    @classmethod
    def load_all(cls, kb_path: str | Path) -> List[KBDocument]:
        """Discover and load all markdown documents in the knowledge base."""
        files = cls.discover_files(kb_path)
        documents: List[KBDocument] = []
        for file in files:
            doc = cls.load_document(file)
            documents.append(doc)
        return documents

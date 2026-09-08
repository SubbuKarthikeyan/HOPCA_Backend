"""Semantic and section-aware markdown chunker for HOPCA knowledge base."""
import re
from typing import List, Optional, Tuple

from app.rag.hashing.content_hasher import ContentHasher
from app.rag.models import Chunk, KBDocument


class SemanticChunker:
    """Chunks markdown documents by entity/section boundaries with size constraints and overlap."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        strategy: str = "semantic",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy

    @staticmethod
    def extract_entity_id(heading_text: str) -> Optional[str]:
        """Extract entity ID from a markdown heading line (e.g. ## Doctor: DOC-001, ## HOSP-001: ...)."""
        heading = heading_text.lstrip("#").strip()

        # Pattern 1: Domain-prefixed (e.g., 'Doctor: DOC-001', 'Department: DEPT-CARD-001', 'Staff: STAFF-001')
        match1 = re.search(r"(?:Doctor|Department|Staff|Service|Entity):\s*([A-Za-z0-9_-]+)", heading, re.IGNORECASE)
        if match1:
            return match1.group(1).strip()

        # Pattern 2: Standard hospital entity codes (e.g., 'HOSP-001:', 'FAQ-001:', 'ADM-001:', 'DOC-001')
        match2 = re.search(r"^([A-Z]{3,5}-\d{3,}(?:-[A-Z0-9]+)?)(?::|\s|$)", heading)
        if match2:
            return match2.group(1).strip()

        # Pattern 3: Generic prefix like 'FAQ-001' or 'DEPT-001'
        match3 = re.search(r"\b([A-Z]{2,6}-(?:[A-Z0-9]+-)?\d{3,})\b", heading)
        if match3:
            return match3.group(1).strip()

        return None

    def _split_text_with_overlap(self, text: str, max_size: int, overlap: int) -> List[str]:
        """Split a long text block into overlapping chunks at paragraph, sentence, or word boundaries."""
        if len(text) <= max_size:
            return [text]

        # Try splitting by subheadings or double newlines first
        paragraphs = text.split("\n\n")
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_len = 0

        for p in paragraphs:
            p_len = len(p) + 2
            if current_len + p_len > max_size and current_chunk:
                combined = "\n\n".join(current_chunk)
                chunks.append(combined)

                # Keep overlap from the end of the combined text
                overlap_text = combined[-overlap:] if overlap > 0 else ""
                current_chunk = [overlap_text, p] if overlap_text else [p]
                current_len = len(overlap_text) + p_len
            else:
                current_chunk.append(p)
                current_len += p_len

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        # Fallback if any single paragraph is still larger than max_size
        final_chunks: List[str] = []
        for c in chunks:
            if len(c) <= max_size:
                final_chunks.append(c)
            else:
                # Split by words
                words = c.split(" ")
                sub_chunk: List[str] = []
                sub_len = 0
                for w in words:
                    if sub_len + len(w) + 1 > max_size and sub_chunk:
                        sub_text = " ".join(sub_chunk)
                        final_chunks.append(sub_text)
                        overlap_words = sub_chunk[-max(1, overlap // 10):] if overlap > 0 else []
                        sub_chunk = overlap_words + [w]
                        sub_len = sum(len(x) + 1 for x in sub_chunk)
                    else:
                        sub_chunk.append(w)
                        sub_len += len(w) + 1
                if sub_chunk:
                    final_chunks.append(" ".join(sub_chunk))

        return final_chunks

    def chunk_document(self, document: KBDocument) -> List[Chunk]:
        """Split a KBDocument into structured Chunk objects."""
        content = document.cleaned_content if document.cleaned_content is not None else document.raw_content
        if not content or not content.strip():
            return []

        # Split content into sections demarcated by '## '
        # We look for lines starting with '## '
        lines = content.split("\n")
        sections: List[Tuple[Optional[str], Optional[str], str]] = []  # (heading, entity_id, text)

        current_heading: Optional[str] = None
        current_entity_id: Optional[str] = None
        current_lines: List[str] = []

        for line in lines:
            if line.startswith("## "):
                if current_lines or current_heading is not None:
                    section_text = "\n".join(current_lines).strip()
                    if section_text:
                        sections.append((current_heading, current_entity_id, section_text))
                current_heading = line
                current_entity_id = self.extract_entity_id(line)
                current_lines = [line]
            else:
                current_lines.append(line)

        if current_lines:
            section_text = "\n".join(current_lines).strip()
            if section_text:
                sections.append((current_heading, current_entity_id, section_text))

        # If no '## ' headings were found, treat entire document as a single section
        if not sections:
            sections.append((None, None, content.strip()))

        chunks: List[Chunk] = []
        global_chunk_idx = 0

        for heading, entity_id, section_text in sections:
            # Check if section needs further splitting
            if len(section_text) > self.chunk_size:
                sub_texts = self._split_text_with_overlap(section_text, self.chunk_size, self.chunk_overlap)
            else:
                sub_texts = [section_text]

            for sub_text in sub_texts:
                sub_text = sub_text.strip()
                if not sub_text:
                    continue

                entity_suffix = f"_{entity_id}" if entity_id else ""
                chunk_id = f"{document.document_id}{entity_suffix}_chunk_{global_chunk_idx}"
                content_hash = ContentHasher.compute_hash(sub_text)

                chunk = Chunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    entity_id=entity_id,
                    domain=document.domain,
                    document_type=document.document_type,
                    title=document.title,
                    chunk_index=global_chunk_idx,
                    content=sub_text,
                    content_hash=content_hash,
                )
                chunks.append(chunk)
                global_chunk_idx += 1

        return chunks

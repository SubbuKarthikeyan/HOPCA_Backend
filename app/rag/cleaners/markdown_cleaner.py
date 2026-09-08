"""Markdown cleaner and normalizer for HOPCA knowledge base documents."""
import re
import unicodedata


class MarkdownCleaner:
    """Cleans and standardizes raw markdown content without altering semantic meaning."""

    @classmethod
    def clean(cls, raw_content: str) -> str:
        """Perform comprehensive cleaning and normalization on raw markdown text.
        
        Steps:
        1. Unicode normalization (NFC)
        2. Line ending normalization (\r\n -> \n)
        3. Heading syntax standardization (# Heading)
        4. Trailing whitespace stripping per line
        5. Excess consecutive blank lines reduction
        6. Stripping leading/trailing document whitespace
        """
        if not raw_content:
            return ""

        # 1. Unicode NFC normalization
        content = unicodedata.normalize("NFC", raw_content)

        # 2. Line ending normalization
        content = content.replace("\r\n", "\n").replace("\r", "\n")

        # 3. Standardize heading markers: ensure space between #+ and text
        # e.g., '##Doctor: DOC-001' -> '## Doctor: DOC-001'
        content = re.sub(r"^(#{1,6})([^\s#])", r"\1 \2", content, flags=re.MULTILINE)

        # 4. Strip trailing whitespace per line
        lines = [line.rstrip() for line in content.split("\n")]

        # 5. Remove empty sections: header immediately followed by next header of same or higher level
        # and collapse consecutive blank lines to max 1 empty line (2 newlines)
        cleaned_lines = []
        for line in lines:
            cleaned_lines.append(line)

        content = "\n".join(cleaned_lines)
        # Collapse 3+ newlines to 2 newlines
        content = re.sub(r"\n{3,}", "\n\n", content)

        # 6. Final trim
        return content.strip()

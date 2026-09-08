"""Deterministic content hashing using SHA-256."""
import hashlib


class ContentHasher:
    """Computes deterministic SHA-256 hashes of string content."""

    @classmethod
    def compute_hash(cls, content: str) -> str:
        """Compute SHA-256 hex digest for given text string."""
        if content is None:
            content = ""
        # Ensure utf-8 encoded byte representation
        encoded = content.encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def hash(cls, content: str) -> str:
        """Alias for compute_hash."""
        return cls.compute_hash(content)

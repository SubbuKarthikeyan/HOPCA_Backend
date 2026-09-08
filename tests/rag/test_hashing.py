"""Tests for Content Hasher (Scenarios 8 & 9)."""
from app.rag.hashing.content_hasher import ContentHasher


def test_scenario_8_same_content_same_hash():
    """Scenario 8: Same content produces identical hash."""
    text1 = "HOPCA General Hospital Cardiology Department DOC-001"
    text2 = "HOPCA General Hospital Cardiology Department DOC-001"

    hash1 = ContentHasher.compute_hash(text1)
    hash2 = ContentHasher.compute_hash(text2)

    assert hash1 == hash2
    assert len(hash1) == 64


def test_scenario_9_new_content_new_hash():
    """Scenario 9: New/modified content produces different hash."""
    text1 = "HOPCA General Hospital Cardiology Department DOC-001"
    text2 = "HOPCA General Hospital Neurology Department DOC-001"

    hash1 = ContentHasher.compute_hash(text1)
    hash2 = ContentHasher.compute_hash(text2)

    assert hash1 != hash2

"""Tests for Markdown Cleaner (Scenario 6)."""
from app.rag.cleaners.markdown_cleaner import MarkdownCleaner


def test_scenario_6_cleaning_deterministic():
    """Scenario 6: Cleaning deterministic — same input produces same output across multiple runs."""
    raw_input = "##Heading 1\r\n\r\n\r\nSome text with trailing whitespace   \r\n\r\n\r\n\r\n## Heading 2\nBody text."

    cleaned_1 = MarkdownCleaner.clean(raw_input)
    cleaned_2 = MarkdownCleaner.clean(raw_input)

    assert cleaned_1 == cleaned_2
    assert "## Heading 1" in cleaned_1
    assert "   \n" not in cleaned_1
    assert "\r" not in cleaned_1
    assert "\n\n\n" not in cleaned_1


def test_cleaning_unicode_normalization():
    """Test unicode NFC normalization."""
    # Decomposed vs precomposed unicode characters
    decomposed = "e\u0301"  # e + acute accent
    precomposed = "\u00e9"  # é

    cleaned_decomp = MarkdownCleaner.clean(f"## {decomposed}")
    cleaned_precomp = MarkdownCleaner.clean(f"## {precomposed}")

    assert cleaned_decomp == cleaned_precomp

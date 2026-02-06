"""
Unit tests for the preprocess module.
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'spam_detector', 'src'))

from preprocess import clean_text


class TestCleanText:
    """Test cases for clean_text function."""

    def test_clean_text_basic(self):
        """Test basic cleaning functionality."""
        text = "Hello World"
        result = clean_text(text)
        assert isinstance(result, str)
        assert result.lower() == result  # Should be lowercase

    def test_clean_text_empty_string(self):
        """Test with empty string."""
        result = clean_text("")
        assert result == ""

    def test_clean_text_special_characters(self):
        """Test removal of special characters."""
        text = "Hello @#$% World! 123"
        result = clean_text(text)
        # Should contain no special characters except spaces
        assert "@" not in result
        assert "#" not in result
        assert "$" not in result
        assert "%" not in result

    def test_clean_text_urls(self):
        """Test URL removal."""
        text = "Check this http://example.com and www.test.com"
        result = clean_text(text)
        assert "http" not in result
        assert "www" not in result

    def test_clean_text_emails(self):
        """Test email removal."""
        text = "Contact me at user@example.com"
        result = clean_text(text)
        # Email should be removed
        assert "@" not in result

    def test_clean_text_whitespace(self):
        """Test whitespace normalization."""
        text = "Hello    World\n\nTest"
        result = clean_text(text)
        # Multiple spaces should be reduced to single space
        assert "    " not in result

    def test_clean_text_case_insensitive(self):
        """Test that cleaning is case-insensitive."""
        text1 = "HELLO WORLD"
        text2 = "hello world"
        result1 = clean_text(text1)
        result2 = clean_text(text2)
        # Content should be the same
        assert result1 == result2

    def test_clean_text_non_string_input(self):
        """Test handling of non-string input."""
        # Should convert to string
        result = clean_text(123)
        assert isinstance(result, str)

    def test_clean_text_removes_stopwords(self):
        """Test that common stopwords are removed."""
        text = "the quick brown fox"
        result = clean_text(text)
        # 'the' is a stopword, should be removed
        # 'quick', 'brown', 'fox' should remain
        assert "quick" in result or "brown" in result or "fox" in result

    def test_clean_text_spam_example(self):
        """Test with a real spam example."""
        spam_text = "Congratulations! You've won $1000. Click here: http://spam.example"
        result = clean_text(spam_text)
        assert isinstance(result, str)
        assert len(result) > 0
        assert "congratulation" in result or "won" in result

    def test_clean_text_legitimate_email(self):
        """Test with a legitimate email example."""
        legit_text = "Meeting rescheduled to tomorrow at 2 PM"
        result = clean_text(legit_text)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_clean_text_preserves_numbers(self):
        """Test that numbers are preserved."""
        text = "2024 year with numbers 123"
        result = clean_text(text)
        # Numbers should be preserved
        assert any(c.isdigit() for c in result)


class TestCleanTextEdgeCases:
    """Test edge cases for clean_text."""

    def test_only_special_characters(self):
        """Test text with only special characters."""
        text = "@#$%^&*()"
        result = clean_text(text)
        # Should return mostly empty
        assert len(result) == 0 or result.isspace()

    def test_only_urls(self):
        """Test text with only URLs."""
        text = "http://example.com www.test.com https://another.org"
        result = clean_text(text)
        # URLs should be removed
        assert "http" not in result
        assert "www" not in result

    def test_single_character(self):
        """Test with single character."""
        result = clean_text("a")
        assert isinstance(result, str)

    def test_very_long_text(self):
        """Test with very long text."""
        text = "word " * 10000
        result = clean_text(text)
        assert isinstance(result, str)

    def test_unicode_characters(self):
        """Test with unicode characters."""
        text = "Hello 你好 مرحبا"
        result = clean_text(text)
        assert isinstance(result, str)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

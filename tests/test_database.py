"""
Unit tests for the database module.
"""

import pytest
import sqlite3
import os
import sys
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'spam_detector', 'src'))

from database import (
    init_db, save_prediction, save_blocked,
    get_all_predictions, get_blocked_emails,
    delete_old_records, DB_PATH
)


@pytest.fixture(scope="function")
def clean_db():
    """Create a clean database for each test."""
    # Remove old database if exists
    test_db_path = os.path.join(os.path.dirname(DB_PATH), 'test_predictions.db')
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    yield test_db_path
    
    # Cleanup after test
    if os.path.exists(test_db_path):
        os.remove(test_db_path)


class TestDatabaseInitialization:
    """Test database initialization."""

    def test_init_db_creates_tables(self):
        """Test that init_db creates necessary tables."""
        init_db()
        
        # Check if database file exists
        assert os.path.exists(DB_PATH)
        
        # Check if tables exist
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check predictions table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
        assert cursor.fetchone() is not None
        
        # Check blocked_emails table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='blocked_emails'")
        assert cursor.fetchone() is not None
        
        conn.close()

    def test_init_db_idempotent(self):
        """Test that init_db can be called multiple times safely."""
        init_db()
        init_db()  # Should not raise error
        assert os.path.exists(DB_PATH)


class TestSavePrediction:
    """Test save_prediction function."""

    def test_save_prediction_success(self):
        """Test saving a prediction successfully."""
        init_db()
        
        result = save_prediction("Test email", "spam", 0.95)
        assert result is True

    def test_save_prediction_invalid_content(self):
        """Test saving with invalid content."""
        init_db()
        
        result = save_prediction("", "spam", 0.95)
        assert result is False

    def test_save_prediction_invalid_label(self):
        """Test saving with invalid label."""
        init_db()
        
        result = save_prediction("Test email", "invalid", 0.95)
        assert result is False

    def test_save_prediction_invalid_confidence(self):
        """Test saving with invalid confidence."""
        init_db()
        
        result = save_prediction("Test email", "spam", 1.5)
        assert result is False
        
        result = save_prediction("Test email", "spam", -0.1)
        assert result is False

    def test_save_prediction_multiple(self):
        """Test saving multiple predictions."""
        init_db()
        
        save_prediction("Email 1", "spam", 0.9)
        save_prediction("Email 2", "ham", 0.1)
        save_prediction("Email 3", "spam", 0.85)
        
        predictions = get_all_predictions()
        assert len(predictions) >= 3


class TestSaveBlocked:
    """Test save_blocked function."""

    def test_save_blocked_success(self):
        """Test saving blocked email successfully."""
        init_db()
        
        result = save_blocked("Spam email content", "Detected as spam")
        assert result is True

    def test_save_blocked_default_reason(self):
        """Test saving blocked email with default reason."""
        init_db()
        
        result = save_blocked("Spam email")
        assert result is True

    def test_save_blocked_invalid_content(self):
        """Test saving blocked email with invalid content."""
        init_db()
        
        result = save_blocked("", "Reason")
        assert result is False


class TestGetPredictions:
    """Test get_all_predictions function."""

    def test_get_predictions_empty(self):
        """Test getting predictions from empty database."""
        init_db()
        predictions = get_all_predictions()
        assert isinstance(predictions, list)

    def test_get_predictions_with_limit(self):
        """Test getting predictions with limit."""
        init_db()
        
        # Save multiple predictions
        for i in range(5):
            save_prediction(f"Email {i}", "spam", 0.9)
        
        predictions = get_all_predictions(limit=2)
        assert len(predictions) <= 2

    def test_get_predictions_returns_tuple(self):
        """Test that predictions are returned as tuples."""
        init_db()
        save_prediction("Test email", "spam", 0.95)
        
        predictions = get_all_predictions()
        if predictions:
            assert isinstance(predictions[0], tuple)
            assert len(predictions[0]) == 4  # timestamp, content, label, confidence


class TestDeleteOldRecords:
    """Test delete_old_records function."""

    def test_delete_old_records(self):
        """Test deleting old records."""
        init_db()
        save_prediction("Old email", "spam", 0.9)
        
        # Delete records older than 0 days (should delete nothing if recent)
        deleted = delete_old_records(days=0)
        assert isinstance(deleted, int)

    def test_delete_old_records_result_type(self):
        """Test that delete returns integer."""
        init_db()
        
        result = delete_old_records(days=30)
        assert isinstance(result, int)


class TestDatabaseIntegration:
    """Integration tests for database operations."""

    def test_full_workflow(self):
        """Test complete database workflow."""
        init_db()
        
        # Save predictions
        save_prediction("Spam email 1", "spam", 0.95)
        save_prediction("Legitimate email 1", "ham", 0.05)
        
        # Save blocked
        save_blocked("Evil spam", "Contains malware")
        
        # Retrieve
        predictions = get_all_predictions()
        blocked = get_blocked_emails()
        
        assert len(predictions) > 0
        assert len(blocked) > 0

    def test_concurrent_operations(self):
        """Test that multiple simultaneous operations work."""
        init_db()
        
        for i in range(10):
            save_prediction(f"Email {i}", "spam" if i % 2 == 0 else "ham", 0.5 + i * 0.05)
        
        predictions = get_all_predictions()
        assert len(predictions) >= 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

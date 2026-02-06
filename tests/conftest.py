"""
Pytest configuration and shared fixtures.
"""

import pytest
import sys
import os
import sqlite3
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'spam_detector', 'src'))


@pytest.fixture(scope="session")
def test_db_path():
    """Provide testing database path."""
    return os.path.join(
        os.path.dirname(__file__),
        '..',
        'spam_detector',
        'data',
        'test_predictions.db'
    )


@pytest.fixture(autouse=True)
def reset_modules():
    """Reset imported modules between tests."""
    yield
    # Cleanup code here if needed


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


@pytest.fixture
def sample_spam_emails():
    """Provide sample spam emails for testing."""
    return [
        "FREE MONEY!!! Click here now!!!",
        "CONGRATULATIONS! You won $1000000",
        "Urgent: Verify your account immediately",
        "Nigerian Prince needs your help",
        "MAKE MONEY FROM HOME - NO WORK REQUIRED",
    ]


@pytest.fixture
def sample_legitimate_emails():
    """Provide sample legitimate emails."""
    return [
        "Meeting scheduled for tomorrow at 2 PM",
        "Here's the project report you requested",
        "Coffee break at 3 PM in the office",
        "Your package has been delivered",
        "Team lunch is postponed to next week",
    ]


@pytest.fixture
def sample_spam_words():
    """Provide sample spam indicator words."""
    return [
        "free", "money", "urgent", "click", "congratulations",
        "limited", "offer", "act", "now", "guarantee",
        "winner", "prize", "claim", "verify", "update"
    ]

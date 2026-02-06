"""
Integration tests for the FastAPI application.
"""

import pytest
import sys
import os
from fastapi.testclient import TestClient
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'spam_detector', 'src'))

from api import app
from database import init_db

# Create test client
client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_database():
    """Setup database before running tests."""
    init_db()
    yield


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self):
        """Test that health endpoint works."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data


class TestAuthentication:
    """Test authentication endpoints."""

    def test_register_user(self):
        """Test user registration."""
        response = client.post(
            "/register",
            json={
                "username": "testuser",
                "password": "testpass123"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "message" in data

    def test_register_duplicate_user(self):
        """Test registering duplicate username."""
        # Register first user
        client.post(
            "/register",
            json={
                "username": "duplicateuser",
                "password": "pass123"
            }
        )
        
        # Try to register same username
        response = client.post(
            "/register",
            json={
                "username": "duplicateuser",
                "password": "pass456"
            }
        )
        assert response.status_code == 409

    def test_register_empty_credentials(self):
        """Test registration with empty credentials."""
        response = client.post(
            "/register",
            json={
                "username": "",
                "password": ""
            }
        )
        assert response.status_code == 422  # Validation error

    def test_login_success(self):
        """Test successful login."""
        # Register user
        client.post(
            "/register",
            json={
                "username": "loginuser",
                "password": "loginpass123"
            }
        )
        
        # Login
        response = client.post(
            "/login",
            json={
                "username": "loginuser",
                "password": "loginpass123"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_invalid_credentials(self):
        """Test login with invalid credentials."""
        response = client.post(
            "/login",
            json={
                "username": "nonexistent",
                "password": "wrongpass"
            }
        )
        assert response.status_code == 401

    def test_login_wrong_password(self):
        """Test login with wrong password."""
        # Register user
        client.post(
            "/register",
            json={
                "username": "wrongpassuser",
                "password": "correctpass"
            }
        )
        
        # Try to login with wrong password
        response = client.post(
            "/login",
            json={
                "username": "wrongpassuser",
                "password": "wrongpass"
            }
        )
        assert response.status_code == 401


class TestPredictionEndpoints:
    """Test prediction endpoints."""

    @pytest.fixture
    def auth_token(self):
        """Get authentication token for tests."""
        client.post(
            "/register",
            json={
                "username": "predictionuser",
                "password": "predpass123"
            }
        )
        
        response = client.post(
            "/login",
            json={
                "username": "predictionuser",
                "password": "predpass123"
            }
        )
        return response.json()["access_token"]

    def test_predict_without_auth(self):
        """Test prediction without authentication."""
        response = client.post(
            "/predict",
            json={"text": "Test email"}
        )
        assert response.status_code == 403  # Forbidden

    def test_predict_success(self, auth_token):
        """Test successful prediction."""
        response = client.post(
            "/predict",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={"text": "This is a legitimate email"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "label" in data
        assert data["label"] in ["spam", "ham"]

    def test_predict_spam_detection(self, auth_token):
        """Test that obvious spam is detected."""
        spam_text = "FREE MONEY NOW!!! Click here to win $1000"
        response = client.post(
            "/predict",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={"text": spam_text}
        )
        assert response.status_code == 200
        data = response.json()
        assert "label" in data

    def test_predict_empty_text(self, auth_token):
        """Test prediction with empty text."""
        response = client.post(
            "/predict",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={"text": ""}
        )
        assert response.status_code == 422  # Validation error

    def test_batch_predict_success(self, auth_token):
        """Test successful batch prediction."""
        response = client.post(
            "/batch_predict",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "texts": [
                    "Email 1",
                    "Email 2",
                    "Email 3"
                ]
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total" in data
        assert "successful" in data
        assert data["total"] == 3

    def test_batch_predict_too_many(self, auth_token):
        """Test batch predict with too many emails."""
        texts = ["Email"] * 101  # More than max of 100
        response = client.post(
            "/batch_predict",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={"texts": texts}
        )
        assert response.status_code == 422

    def test_batch_predict_empty_list(self, auth_token):
        """Test batch predict with empty list."""
        response = client.post(
            "/batch_predict",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={"texts": []}
        )
        assert response.status_code == 422


class TestHistoryEndpoints:
    """Test history retrieval endpoints."""

    @pytest.fixture
    def auth_token_with_history(self):
        """Setup user with prediction history."""
        client.post(
            "/register",
            json={
                "username": "historyuser",
                "password": "historypass123"
            }
        )
        
        response = client.post(
            "/login",
            json={
                "username": "historyuser",
                "password": "historypass123"
            }
        )
        token = response.json()["access_token"]
        
        # Create some predictions
        for i in range(3):
            client.post(
                "/predict",
                headers={"Authorization": f"Bearer {token}"},
                json={"text": f"Test email {i}"}
            )
        
        return token

    def test_get_history_without_auth(self):
        """Test getting history without authentication."""
        response = client.get("/history")
        assert response.status_code == 403

    def test_get_history_success(self, auth_token_with_history):
        """Test successful history retrieval."""
        response = client.get(
            "/history",
            headers={"Authorization": f"Bearer {auth_token_with_history}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "history" in data
        assert "count" in data

    def test_get_history_with_limit(self, auth_token_with_history):
        """Test history retrieval with limit."""
        response = client.get(
            "/history?limit=2",
            headers={"Authorization": f"Bearer {auth_token_with_history}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["count"] <= 2

    def test_get_blocked_without_auth(self):
        """Test getting blocked emails without authentication."""
        response = client.get("/blocked")
        assert response.status_code == 403

    def test_get_blocked_success(self, auth_token_with_history):
        """Test getting blocked emails."""
        response = client.get(
            "/blocked",
            headers={"Authorization": f"Bearer {auth_token_with_history}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "blocked" in data
        assert "count" in data


class TestBlockEndpoint:
    """Test block email endpoint."""

    @pytest.fixture
    def auth_token(self):
        """Get authentication token."""
        client.post(
            "/register",
            json={
                "username": "blockuser",
                "password": "blockpass123"
            }
        )
        
        response = client.post(
            "/login",
            json={
                "username": "blockuser",
                "password": "blockpass123"
            }
        )
        return response.json()["access_token"]

    def test_block_without_auth(self):
        """Test block without authentication."""
        response = client.post(
            "/block",
            json={"text": "Email to block"}
        )
        assert response.status_code == 403

    def test_block_email(self, auth_token):
        """Test blocking an email."""
        response = client.post(
            "/block",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={"text": "SPAM EMAIL!!!"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "prediction" in data

    def test_block_legitimate_email(self, auth_token):
        """Test blocking a legitimate email (not spam)."""
        response = client.post(
            "/block",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={"text": "This is a normal email"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "Email not blocked" in data["message"]


class TestErrorHandling:
    """Test error handling."""

    def test_invalid_token(self):
        """Test with invalid token."""
        response = client.post(
            "/predict",
            headers={"Authorization": "Bearer invalid_token"},
            json={"text": "Test email"}
        )
        assert response.status_code == 401

    def test_missing_required_field(self):
        """Test with missing required field."""
        response = client.post(
            "/register",
            json={"username": "testuser"}  # Missing password
        )
        assert response.status_code == 422


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

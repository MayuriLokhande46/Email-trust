# Testing Guide

This directory contains comprehensive unit and integration tests for the Spam Detection project.

## Test Structure

```
tests/
├── conftest.py                 # Pytest configuration and shared fixtures
├── test_preprocess.py         # Unit tests for text preprocessing
├── test_database.py           # Unit tests for database operations
├── test_api.py                # Integration tests for API endpoints
└── README.md                  # This file
```

## Prerequisites

Install testing dependencies:

```bash
pip install pytest pytest-cov

# Or using the requirements.txt with test dependencies
pip install -r requirements.txt
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run tests with verbose output
```bash
pytest -v
```

### Run specific test file
```bash
pytest tests/test_preprocess.py
```

### Run specific test class
```bash
pytest tests/test_preprocess.py::TestCleanText
```

### Run specific test function
```bash
pytest tests/test_preprocess.py::TestCleanText::test_clean_text_basic
```

### Run tests with coverage report
```bash
pytest --cov=src --cov-report=html
```

This generates an HTML coverage report in `htmlcov/index.html`

### Run only unit tests
```bash
pytest -m unit
```

### Run only integration tests
```bash
pytest -m integration
```

### Run tests excluding slow ones
```bash
pytest -m "not slow"
```

## Test Categories

### 1. Unit Tests (test_preprocess.py)
Tests for text preprocessing functionality.

**Coverage:**
- Basic text cleaning
- URL removal
- Email removal
- Special character removal
- Whitespace normalization
- Case handling
- Stopword removal
- Edge cases (empty strings, very long text, unicode)

**Example:**
```bash
pytest tests/test_preprocess.py -v
```

### 2. Database Tests (test_database.py)
Tests for database operations.

**Coverage:**
- Database initialization
- Prediction saving
- Blocked email saving
- Record retrieval with limits
- Old record deletion
- Data validation
- Error handling

**Example:**
```bash
pytest tests/test_database.py -v
```

### 3. API Integration Tests (test_api.py)
Tests for FastAPI endpoints with authentication.

**Coverage:**
- User registration
- User login
- JWT token generation
- Single email prediction
- Batch email prediction
- History retrieval
- Blocked emails retrieval
- Email blocking
- Error handling
- Authentication enforcement

**Example:**
```bash
pytest tests/test_api.py -v
```

## Test Examples

### Testing Text Cleaning
```python
def test_clean_text_removes_urls(self):
    """Test URL removal."""
    text = "Check this http://example.com"
    result = clean_text(text)
    assert "http" not in result
```

### Testing Database Operations
```python
def test_save_prediction_success(self):
    """Test saving a prediction successfully."""
    init_db()
    result = save_prediction("Test email", "spam", 0.95)
    assert result is True
```

### Testing API Endpoints
```python
def test_predict_success(self, auth_token):
    """Test successful prediction."""
    response = client.post(
        "/predict",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={"text": "This is a legitimate email"}
    )
    assert response.status_code == 200
```

## Fixtures

### Built-in Fixtures (conftest.py)

- `test_db_path`: Path to test database
- `sample_spam_emails`: List of sample spam emails
- `sample_legitimate_emails`: List of sample legitimate emails
- `sample_spam_words`: List of spam indicator words
- `auth_token`: Authentication token for API tests
- `clean_db`: Clean database for each test

## Coverage Goals

Target coverage metrics:

| Module | Target |
|--------|--------|
| preprocess.py | >90% |
| database.py | >85% |
| predict.py | >80% |
| api.py | >75% |
| Overall | >80% |

## Continuous Integration

These tests can be integrated into CI/CD pipelines:

```bash
# For GitHub Actions
pytest --cov=src --cov-report=xml --junitxml=test-results.xml
```

## Troubleshooting

### Tests fail with module not found
Make sure you're running pytest from the project root:
```bash
cd /path/to/email-spam-detection
pytest
```

### Database locked error
SQLite can have locking issues. Ensure:
- No other process is accessing the database
- Tests run serially (not in parallel)
- Clean up database files between test runs

### Token expiration in API tests
JWT tokens expire after 60 minutes. If tests take too long:
- Run them in smaller batches
- Increase `ACCESS_TOKEN_EXPIRE_MINUTES` in security.py
- Use fixtures to refresh tokens

## Best Practices

1. **Test Isolation**: Each test should be independent
2. **Clear Names**: Use descriptive test names
3. **Assertions**: Use specific assertions with clear error messages
4. **Fixtures**: Reuse fixtures for common setup
5. **Mocking**: Mock external dependencies when necessary
6. **Edge Cases**: Test boundary conditions and error cases

## Adding New Tests

When adding new features:

1. Create test file: `test_module_name.py`
2. Define test classes for related tests
3. Use clear test function names: `test_<action>_<condition>_<expected_result>`
4. Document test purpose with docstring
5. Use fixtures for common setup
6. Add to CI/CD pipeline

Example:
```python
def test_predict_with_special_characters_succeeds(self, auth_token):
    """Test that prediction handles special characters correctly."""
    response = client.post(
        "/predict",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={"text": "Hello @#$% World!"}
    )
    assert response.status_code == 200
```

## Performance Testing

For performance-critical tests, use the `@pytest.mark.slow` decorator:

```python
@pytest.mark.slow
def test_large_batch_prediction(self, auth_token):
    """Test batch prediction with large dataset."""
    texts = ["Email"] * 1000
    # ... test code
```

Run performance tests separately:
```bash
pytest -m slow
```

## Debugging Tests

### Run with detailed output
```bash
pytest -v -s
```

### Run with print statements
```bash
pytest -v -s --capture=no
```

### Run with debugger (pdb)
```bash
pytest --pdb
```

### Print variable values
```python
import pytest
def test_something():
    result = predict("test")
    print(result)  # Will show with -s flag
    assert result is not None
```

## Test Maintenance

Regular maintenance tasks:

1. **Review failing tests** - Fix broken tests immediately
2. **Update fixtures** - Keep test data up-to-date
3. **Remove obsolete tests** - Clean up deprecated functionality tests
4. **Optimize slow tests** - Improve test performance
5. **Expand coverage** - Add tests for new code paths

---

**Last Updated:** February 6, 2024

For questions or issues, refer to the main project README.

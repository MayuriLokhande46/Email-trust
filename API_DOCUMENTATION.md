# Spam Detector API - Complete Documentation

## Overview
The Spam Detector API is a Flask/FastAPI-based REST API for classifying emails as spam or legitimate (ham) using machine learning. The API uses JWT (JSON Web Tokens) for authentication and provides comprehensive endpoints for predictions, history management, and user authentication.

**Base URL:** `http://localhost:8000` (default)

**Version:** 1.0.0

---

## Table of Contents
1. [Authentication](#authentication)
2. [Endpoints](#endpoints)
3. [Request/Response Models](#requestresponse-models)
4. [Error Handling](#error-handling)
5. [Examples](#examples)
6. [Rate Limiting](#rate-limiting)
7. [Security](#security)

---

## Authentication

### Overview
The API uses **JWT (JSON Web Tokens)** for secure authentication. All endpoints except `/register`, `/login`, and `/` require a valid JWT token.

### Getting Started

#### 1. Register a New User
First, register a new account:

```bash
curl -X POST "http://localhost:8000/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "your_username",
    "password": "your_password"
  }'
```

**Response:**
```json
{
  "message": "User registered successfully"
}
```

#### 2. Login and Get Access Token
After registration, login to receive a JWT token:

```bash
curl -X POST "http://localhost:8000/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "your_username",
    "password": "your_password"
  }'
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

#### 3. Using the Token
Include the token in the `Authorization` header for all protected endpoints:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Authorization: Bearer <your_access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Email content to analyze..."
  }'
```

---

## Endpoints

### 1. Health Check

**Endpoint:** `GET /`

**Description:** Check API health and get available endpoints.

**Authentication:** Not required

**Request:**
```bash
curl -X GET "http://localhost:8000/"
```

**Response:**
```json
{
  "message": "Spam Detector API is running",
  "version": "1.0.0",
  "endpoints": [
    "/register",
    "/login",
    "/predict",
    "/batch_predict",
    "/history",
    "/blocked"
  ]
}
```

---

### 2. Register User

**Endpoint:** `POST /register`

**Description:** Create a new user account.

**Authentication:** Not required

**Request Body:**
```json
{
  "username": "string (1-255 characters)",
  "password": "string (1+ characters)"
}
```

**Response (201 Created):**
```json
{
  "message": "User registered successfully"
}
```

**Error Responses:**
- `409 Conflict`: Username already exists
- `400 Bad Request`: Invalid input data
- `500 Internal Server Error`: Server error during registration

---

### 3. Login

**Endpoint:** `POST /login`

**Description:** Authenticate user and receive JWT token.

**Authentication:** Not required

**Request Body:**
```json
{
  "username": "string",
  "password": "string"
}
```

**Response (200 OK):**
```json
{
  "access_token": "string (JWT token)",
  "token_type": "bearer"
}
```

**Error Responses:**
- `401 Unauthorized`: Invalid username or password
- `400 Bad Request`: Invalid input data
- `500 Internal Server Error`: Server error during login

**Token Expiration:** 60 minutes (configurable via `ACCESS_TOKEN_EXPIRE_MINUTES`)

---

### 4. Single Email Prediction

**Endpoint:** `POST /predict`

**Description:** Analyze a single email to determine if it's spam or legitimate.

**Authentication:** Required (JWT Bearer token)

**Request Body:**
```json
{
  "text": "string (1-50,000 characters)"
}
```

**Response (200 OK):**
```json
{
  "label": "spam or ham",
  "prob_spam": 0.95,
  "confidence": 0.95,
  "spam_words": [
    "congratulations",
    "money",
    "click"
  ]
}
```

**Fields Explanation:**
- `label`: Classification result ("spam" or "ham")
- `prob_spam`: Probability score (0.0-1.0) of being spam
- `confidence`: Model confidence in the prediction
- `spam_words`: List of words matching known spam indicators

**Error Responses:**
- `400 Bad Request`: Invalid input (empty text, etc.)
- `401 Unauthorized`: Invalid or expired token
- `500 Internal Server Error`: Prediction failed

**Example:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Congratulations! You won $1000. Click here to claim: www.scam.com"
  }'
```

---

### 5. Batch Email Prediction

**Endpoint:** `POST /batch_predict`

**Description:** Analyze multiple emails in a single request.

**Authentication:** Required (JWT Bearer token)

**Request Body:**
```json
{
  "texts": [
    "string",
    "string"
  ]
}
```

**Constraints:**
- Minimum 1 email, maximum 100 emails per request
- Each email must be non-empty

**Response (200 OK):**
```json
{
  "results": [
    {
      "label": "spam",
      "prob_spam": 0.92,
      "confidence": 0.92,
      "spam_words": ["money", "urgent"]
    },
    {
      "label": "ham",
      "prob_spam": 0.05,
      "confidence": 0.95,
      "spam_words": []
    }
  ],
  "total": 2,
  "successful": 2
}
```

**Fields Explanation:**
- `results`: Array of prediction results for each email
- `total`: Total emails submitted
- `successful`: Number of successfully predicted emails

**Error Responses:**
- `400 Bad Request`: Invalid batch format or constraints violated
- `401 Unauthorized`: Invalid or expired token
- `500 Internal Server Error`: Batch prediction failed

**Example:**
```bash
curl -X POST "http://localhost:8000/batch_predict" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "URGENT: Your account needs verification!",
      "Meeting scheduled for tomorrow at 2 PM",
      "FREE MONEY - No work required!"
    ]
  }'
```

---

### 6. Prediction History

**Endpoint:** `GET /history`

**Description:** Retrieve past prediction records.

**Authentication:** Required (JWT Bearer token)

**Query Parameters:**
- `limit` (optional, integer): Maximum records to return
  - Default: 100
  - Maximum: 1000
  - Minimum: 1

**Response (200 OK):**
```json
{
  "history": [
    {
      "timestamp": "2024-02-06 10:30:45",
      "email_content": "Sample email text (truncated to 100 chars)...",
      "prediction": "spam",
      "confidence": "92.50%"
    },
    {
      "timestamp": "2024-02-06 10:25:12",
      "email_content": "Another email content...",
      "prediction": "ham",
      "confidence": "87.30%"
    }
  ],
  "count": 2
}
```

**Fields Explanation:**
- `timestamp`: When the prediction was made
- `email_content`: Original email text (first 100 chars + "..." if longer)
- `prediction`: Classification result
- `confidence`: Confidence score as percentage

**Error Responses:**
- `401 Unauthorized`: Invalid or expired token
- `500 Internal Server Error`: Failed to retrieve history

**Example:**
```bash
curl -X GET "http://localhost:8000/history?limit=50" \
  -H "Authorization: Bearer <token>"
```

---

### 7. Blocked Emails

**Endpoint:** `GET /blocked`

**Description:** Retrieve records of emails that were blocked as spam.

**Authentication:** Required (JWT Bearer token)

**Query Parameters:**
- `limit` (optional, integer): Maximum records to return
  - Default: 100
  - Maximum: 1000

**Response (200 OK):**
```json
{
  "blocked": [
    {
      "timestamp": "2024-02-06 11:00:00",
      "email_content": "SPAM EMAIL CONTENT...",
      "reason": "Detected as spam"
    }
  ],
  "count": 1
}
```

**Error Responses:**
- `401 Unauthorized`: Invalid or expired token
- `500 Internal Server Error`: Failed to retrieve blocked emails

**Example:**
```bash
curl -X GET "http://localhost:8000/blocked?limit=25" \
  -H "Authorization: Bearer <token>"
```

---

### 8. Block Email

**Endpoint:** `POST /block`

**Description:** Predict and block an email if detected as spam.

**Authentication:** Required (JWT Bearer token)

**Request Body:**
```json
{
  "text": "string (1-50,000 characters)"
}
```

**Response (200 OK):**
```json
{
  "message": "Email blocked",
  "prediction": {
    "label": "spam",
    "prob_spam": 0.95,
    "confidence": 0.95,
    "spam_words": ["click", "money"]
  }
}
```

or if not spam:

```json
{
  "message": "Email not blocked (not spam)",
  "prediction": {
    "label": "ham",
    "prob_spam": 0.15,
    "confidence": 0.85,
    "spam_words": []
  }
}
```

**Error Responses:**
- `400 Bad Request`: Invalid input
- `401 Unauthorized`: Invalid or expired token
- `500 Internal Server Error`: Block operation failed

**Example:**
```bash
curl -X POST "http://localhost:8000/block" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "CLICK HERE FOR FREE MONEY!!!!"
  }'
```

---

## Request/Response Models

### EmailInput
```json
{
  "text": "string (1-50,000 characters)"
}
```

### BatchEmailInput
```json
{
  "texts": [
    "string",
    "string",
    "..."
  ]
}
```

### PredictionResult
```json
{
  "label": "spam | ham",
  "prob_spam": 0.0-1.0,
  "confidence": 0.0-1.0,
  "spam_words": ["word1", "word2"]
}
```

### TokenResponse
```json
{
  "access_token": "string",
  "token_type": "bearer"
}
```

---

## Error Handling

### Standard Error Response Format

All errors return a JSON object with an error message:

```json
{
  "error": "Error description"
}
```

### HTTP Status Codes

| Code | Meaning | Common Cause |
|------|---------|--------------|
| 200 | OK | Successful request |
| 201 | Created | User successfully created |
| 400 | Bad Request | Invalid input data |
| 401 | Unauthorized | Missing or invalid token |
| 409 | Conflict | Duplicate username |
| 500 | Internal Server Error | Server-side error |

### Common Error Messages

```json
{
  "error": "Invalid token"
}
```

```json
{
  "error": "Email text cannot be empty or whitespace only"
}
```

```json
{
  "error": "Username already exists"
}
```

```json
{
  "error": "Prediction failed"
}
```

---

## Examples

### Complete Workflow Example

#### 1. Register and Login

```bash
# Register
curl -X POST "http://localhost:8000/register" \
  -H "Content-Type: application/json" \
  -d '{"username": "john_doe", "password": "secure_password123"}'

# Response
# {"message": "User registered successfully"}

# Login
curl -X POST "http://localhost:8000/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "john_doe", "password": "secure_password123"}'

# Response
# {"access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...", "token_type": "bearer"}
```

#### 2. Analyze Single Email

```bash
TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

curl -X POST "http://localhost:8000/predict" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Subject: Congratulations!\nYou have won $1000! Click here: www.spam-site.com"
  }'

# Response
# {
#   "label": "spam",
#   "prob_spam": 0.92,
#   "confidence": 0.92,
#   "spam_words": ["congratulations", "won", "money"]
# }
```

#### 3. Analyze Multiple Emails

```bash
curl -X POST "http://localhost:8000/batch_predict" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Meeting rescheduled to tomorrow",
      "CLICK HERE FOR FREE MONEY!!!",
      "Project update ready for review"
    ]
  }'
```

#### 4. Check Prediction History

```bash
curl -X GET "http://localhost:8000/history?limit=10" \
  -H "Authorization: Bearer $TOKEN"
```

---

## Rate Limiting

Currently, the API does not implement rate limiting. For production deployments, consider:
- Adding rate limiting middleware (e.g., Slowapi)
- Implementing request quotas per user
- Using external services like API Gateway

---

## Security

### Best Practices

1. **Token Management**
   - Store tokens securely (not in plain text)
   - Use HTTPS for all API calls
   - Regenerate tokens periodically
   - Never expose tokens in logs

2. **Password Security**
   - Passwords are hashed using SHA-256
   - Always use strong, unique passwords
   - Never reuse passwords across services

3. **API Security**
   - All sensitive endpoints require authentication
   - Input validation on all endpoints
   - SQL injection protection via parameterized queries
   - CORS enabled for cross-origin requests

4. **Environment Variables**
   - Store `SECRET_KEY` in `.env` file (not in code)
   - Use different keys for development and production
   - Rotate SECRET_KEY periodically

### Setting Environment Variables

Create a `.env` file in the project root:

```env
SECRET_KEY=your-very-secure-secret-key-change-this
DATABASE_URL=sqlite:///./predictions.db
ACCESS_TOKEN_EXPIRE_MINUTES=60
```

---

## Testing with cURL

### Test Suite

```bash
#!/bin/bash

BASE_URL="http://localhost:8000"

# 1. Register user
echo "1. Registering user..."
curl -X POST "$BASE_URL/register" \
  -H "Content-Type: application/json" \
  -d '{"username": "testuser", "password": "testpass123"}'

echo -e "\n\n2. Logging in..."
TOKEN=$(curl -s -X POST "$BASE_URL/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "testuser", "password": "testpass123"}' | jq -r '.access_token')

echo "Token: $TOKEN"

echo -e "\n\n3. Single prediction..."
curl -X POST "$BASE_URL/predict" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text": "FREE MONEY - NO WORK NEEDED!"}'

echo -e "\n\n4. Batch prediction..."
curl -X POST "$BASE_URL/batch_predict" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Normal email here", "SPAM EMAIL!", "Another normal one"]}'

echo -e "\n\n5. Get history..."
curl -X GET "$BASE_URL/history?limit=5" \
  -H "Authorization: Bearer $TOKEN"

echo -e "\n\nTests completed!"
```

---

## Advanced Features

### Email Header Parsing & Validation

The API includes advanced email header parsing and validation capabilities that analyze:

#### Authentication Headers (SPF, DKIM, DMARC)
- **SPF (Sender Policy Framework):** Validates if the email sender is authorized
- **DKIM (DomainKeys Identified Mail):** Checks email signature authenticity
- **DMARC (Domain-based Message Authentication, Reporting & Conformance):** Ensures alignment of authentication methods

#### Phishing Detection
Analyzes headers for phishing indicators including:
- Urgent language in subject lines
- Generic greetings (Dear User, Valued Customer)
- Domain mismatches between From and Reply-To
- IP-based sender domains
- Suspicious sender patterns

#### Header Features Included in Predictions

When using the `/predict` endpoint with advanced features enabled, responses include:

```json
{
  "label": "spam",
  "prob_spam": 0.92,
  "header_analysis": {
    "sender": {
      "email": "attacker@suspicious.tk",
      "domain": "suspicious.tk"
    },
    "subject": "URGENT!!! Verify Your Account NOW!!!",
    "authentication": {
      "dkim": {
        "valid": false,
        "reason": "Invalid DKIM format"
      },
      "spf": {
        "valid": false,
        "reason": "SPF validation failed (spoofed)"
      },
      "dmarc": {
        "valid": false,
        "reason": "DMARC validation failed"
      },
      "overall_score": 0.0,
      "authenticated": false
    },
    "phishing_analysis": {
      "phishing_score": 0.85,
      "risk_level": "high",
      "indicators": [
        "Phishing keyword in subject: \"urgent\"",
        "Generic greeting detected",
        "High urgency markers"
      ]
    },
    "overall_risk_score": 0.87,
    "is_suspicious": true
  },
  "advanced_spam_score": 0.78
}
```

#### How Headers Improve Detection

Header analysis contributes to the final spam score with the following weights:
- **Header Authentication (25%):** SPF/DKIM/DMARC validation + phishing indicators
- **Header Features (30%):** Subject line spam keywords, urgency markers
- **Sender Reputation (25%):** Domain reputation checks
- **URL Analysis (20%):** Shortened URLs, IP-based URLs, suspicious patterns

#### Example: Phishing Email Detection

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "From: support@bank-fake.com\nTo: user@example.com\nSubject: URGENT!!! Verify Your Account Now!!!\n\nDear Valued Customer,\nYour account has been suspended. Click here immediately to verify."
  }'
```

**Response Shows:**
- Failed authentication checks (SPF/DKIM/DMARC failure)
- High phishing score (0.8+)
- Urgency keywords detected
- Domain mismatch indicators
- Final prediction: SPAM with 92% confidence

---

## Troubleshooting

### Common Issues

**Issue:** `401 Unauthorized - Invalid token`
- **Solution:** Ensure token is valid and not expired. Get a new token via login endpoint.

**Issue:** `400 Bad Request - Email text cannot be empty`
- **Solution:** Provide non-empty email text in the request.

**Issue:** `409 Conflict - Username already exists`
- **Solution:** Use a different username for registration.

**Issue:** Connection refused (127.0.0.1:8000)
- **Solution:** Ensure API server is running: `python -m uvicorn src.api:app --reload`

---

## Support

For issues or questions, please refer to the project README or contact the development team.

---

**Last Updated:** February 6, 2024  
**API Version:** 1.0.0

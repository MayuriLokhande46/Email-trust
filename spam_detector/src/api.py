"""
FastAPI application for Spam Detection with JWT authentication.
"""

import logging
import os
from typing import List, Dict, Any
from fastapi import (
    FastAPI, HTTPException, Depends, status,
    Request, Response
)
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import time
import sys

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    import spam_detector.src.predict as predict_module
    predict = predict_module.predict
    import spam_detector.src.database as db_module
    save_prediction = db_module.save_prediction
    get_all_predictions = db_module.get_all_predictions
    save_blocked = db_module.save_blocked
    get_blocked_emails = db_module.get_blocked_emails
    import spam_detector.src.security as sec_module
    create_access_token = sec_module.create_access_token
    get_current_user = sec_module.get_current_user
    import spam_detector.src.authentication as auth
except ImportError:
    import predict as predict_module
    predict = predict_module.predict
    import database as db_module
    save_prediction = db_module.save_prediction
    get_all_predictions = db_module.get_all_predictions
    save_blocked = db_module.save_blocked
    get_blocked_emails = db_module.get_blocked_emails
    import security as sec_module
    create_access_token = sec_module.create_access_token
    get_current_user = sec_module.get_current_user
    import authentication as auth

# Set up logging
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "backend.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# FastAPI application
app = FastAPI(
    title="Spam Detector API",
    description="Email spam detection using Machine Learning with JWT authentication",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize user database
auth.create_table()


# ==================== Request/Response Models ====================

class EmailInput(BaseModel):
    """Model for single email prediction request."""
    text: str = Field(..., min_length=1, max_length=50000, description="Email text to analyze")
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v or v.isspace():
            raise ValueError('Email text cannot be empty or whitespace only')
        return v


class BatchEmailInput(BaseModel):
    """Model for batch email prediction requests."""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of email texts")
    
    @validator('texts')
    def texts_valid(cls, v):
        if not v:
            raise ValueError('At least one email text is required')
        for text in v:
            if not text or text.isspace():
                raise ValueError('Email texts cannot be empty or whitespace only')
        return v


class PredictionResult(BaseModel):
    """Model for prediction results."""
    label: str = Field(..., description="'spam' or 'ham'")
    prob_spam: float = Field(None, ge=0, le=1, description="Probability of being spam")
    confidence: float = Field(None, ge=0, le=1, description="Model confidence score")
    spam_words: List[str] = Field(default=[], description="Detected spam indicator words")
    language: str = Field(default='en', description="Detected language of the email")
    url_analysis: Dict[str, Any] = Field(default={}, description="URL analysis results")
    header_analysis: Dict[str, Any] = Field(default={}, description="Header analysis results")
    header_features: Dict[str, Any] = Field(default={}, description="Legacy header features")
    advanced_spam_score: float = Field(default=0.0, description="Composite spam score")


class BatchPredictionResult(BaseModel):
    """Model for batch prediction results."""
    results: List[Dict[str, Any]] = Field(..., description="List of prediction results")
    total: int = Field(..., description="Total emails processed")
    successful: int = Field(..., description="Successfully predicted emails")


class LoginRequest(BaseModel):
    """Model for login requests."""
    username: str = Field(..., min_length=1, description="Username")
    password: str = Field(..., min_length=1, description="Password")


class TokenResponse(BaseModel):
    """Model for token responses."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")


class HistoryItem(BaseModel):
    """Model for prediction history items."""
    timestamp: str
    email_content: str
    prediction: str
    confidence: str


class BlockedItem(BaseModel):
    """Model for blocked email items."""
    timestamp: str
    email_content: str
    reason: str


# ==================== Authentication Endpoints ====================

@app.post(
    "/register",
    response_model=Dict[str, str],
    summary="Register a new user",
    tags=["Authentication"]
)
def register(request: LoginRequest):
    """
    Register a new user account.
    
    Returns:
        - success: Registration successful message
        - error: Error message if registration fails
    """
    try:
        if auth.add_user(request.username, request.password):
            logger.info(f"User registered: {request.username}")
            return {"message": "User registered successfully"}
        else:
            logger.warning(f"Registration failed - username exists: {request.username}")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Username already exists"
            )
    except HTTPException:
        raise
    except Exception as e:
        error_detail = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Registration error: {error_detail}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {error_detail}"
        )


@app.post(
    "/login",
    response_model=TokenResponse,
    summary="Login and get access token",
    tags=["Authentication"]
)
def login(request: LoginRequest):
    """
    Authenticate user and return JWT access token.
    
    Returns:
        - access_token: JWT token for subsequent requests
        - token_type: "bearer"
    """
    try:
        user = auth.get_user(request.username)
        
        if not user or not auth.check_password(user[2], request.password):
            logger.warning(f"Login failed for user: {request.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
        
        # Create JWT token
        access_token = create_access_token({"sub": request.username})
        logger.info(f"User logged in: {request.username}")
        
        return TokenResponse(access_token=access_token)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


# ==================== Prediction Endpoints ====================

@app.get(
    "/",
    summary="Health check",
    tags=["Health"]
)
def home():
    """
    Health check endpoint.
    
    Returns:
        - message: API status message
    """
    return {
        "message": "Spam Detector API is running",
        "version": "1.0.0",
        "endpoints": ["/register", "/login", "/predict", "/batch_predict", "/history", "/blocked"]
    }


@app.post(
    "/predict",
    response_model=PredictionResult,
    summary="Predict single email",
    tags=["Prediction"]
)
def get_prediction(
    input_data: EmailInput,
    current_user: str = Depends(get_current_user)
):
    """
    Analyze a single email to predict if it's spam or ham.
    
    **Authentication:** Required (JWT Bearer token)
    
    Args:
        - text: Email text to analyze
        
    Returns:
        - label: 'spam' or 'ham'
        - prob_spam: Probability of being spam
        - confidence: Model confidence score
        - spam_words: Detected spam indicator words
    """
    try:
        logger.info(f"Prediction requested by {current_user}")
        result = predict(input_data.text)
        
        # Save to database
        label = result['label']
        prob_spam = result.get('prob_spam', 0)
        confidence = prob_spam if label == 'spam' else 1 - prob_spam
        
        save_prediction(input_data.text, label, confidence)
        logger.info(f"Prediction saved: {label} ({confidence:.2f})")
        
        return PredictionResult(**result)
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed"
        )


@app.post(
    "/batch_predict",
    response_model=BatchPredictionResult,
    summary="Predict multiple emails",
    tags=["Prediction"]
)
def get_batch_prediction(
    input_data: BatchEmailInput,
    current_user: str = Depends(get_current_user)
):
    """
    Analyze multiple emails in batch.
    
    **Authentication:** Required (JWT Bearer token)
    
    Args:
        - texts: List of email texts (max 100)
        
    Returns:
        - results: List of prediction results
        - total: Total emails processed
        - successful: Successfully predicted emails
    """
    try:
        logger.info(f"Batch prediction requested by {current_user} for {len(input_data.texts)} emails")
        
        results = []
        successful = 0
        
        for i, text in enumerate(input_data.texts):
            try:
                result = predict(text)
                
                label = result['label']
                prob_spam = result.get('prob_spam', 0)
                confidence = prob_spam if label == 'spam' else 1 - prob_spam
                
                # Save to database
                save_prediction(text, label, confidence)
                
                results.append(result)
                successful += 1
                
            except Exception as e:
                logger.error(f"Failed to predict email {i+1}: {str(e)}")
                results.append({
                    "error": str(e),
                    "index": i
                })
        
        logger.info(f"Batch prediction completed: {successful}/{len(input_data.texts)} successful")
        
        return BatchPredictionResult(
            results=results,
            total=len(input_data.texts),
            successful=successful
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch prediction failed"
        )


# ==================== History Endpoints ====================

@app.get(
    "/history",
    summary="Get prediction history",
    tags=["History"]
)
def get_history(
    limit: int = 100,
    current_user: str = Depends(get_current_user)
):
    """
    Retrieve prediction history.
    
    **Authentication:** Required (JWT Bearer token)
    
    Args:
        - limit: Maximum number of records to return (default: 100, max: 1000)
        
    Returns:
        - history: List of past predictions
        - count: Number of records returned
    """
    try:
        # Validate limit
        limit = min(int(limit), 1000)
        if limit < 1:
            limit = 100
        
        logger.info(f"History requested by {current_user} (limit: {limit})")
        records = get_all_predictions(limit=limit)
        
        history = [
            {
                "timestamp": record[0],
                "email_content": record[1][:100] + "..." if len(record[1]) > 100 else record[1],
                "prediction": record[2],
                "confidence": f"{record[3]*100:.2f}%"
            }
            for record in records
        ]
        
        return {
            "history": history,
            "count": len(history)
        }
        
    except Exception as e:
        logger.error(f"History retrieval error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve history"
        )


@app.get(
    "/blocked",
    summary="Get blocked emails",
    tags=["History"]
)
def get_blocked(
    limit: int = 100,
    current_user: str = Depends(get_current_user)
):
    """
    Retrieve blocked email records.
    
    **Authentication:** Required (JWT Bearer token)
    
    Args:
        - limit: Maximum number of records to return (default: 100, max: 1000)
        
    Returns:
        - blocked: List of blocked emails
        - count: Number of blocked emails
    """
    try:
        # Validate limit
        limit = min(int(limit), 1000)
        if limit < 1:
            limit = 100
        
        logger.info(f"Blocked emails requested by {current_user}")
        records = get_blocked_emails(limit=limit)
        
        blocked = [
            {
                "timestamp": record[0],
                "email_content": record[1][:100] + "..." if len(record[1]) > 100 else record[1],
                "reason": record[2]
            }
            for record in records
        ]
        
        return {
            "blocked": blocked,
            "count": len(blocked)
        }
        
    except Exception as e:
        logger.error(f"Blocked emails retrieval error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve blocked emails"
        )


@app.post(
    "/block",
    summary="Block spam email",
    tags=["Prediction"]
)
def block_email(
    input_data: EmailInput,
    current_user: str = Depends(get_current_user)
):
    """
    Predict and block spam emails.
    
    **Authentication:** Required (JWT Bearer token)
    
    Args:
        - text: Email text to analyze
        
    Returns:
        - message: Block status
        - prediction: Prediction result
    """
    try:
        logger.info(f"Block request from {current_user}")
        result = predict(input_data.text)
        
        if result['label'] == 'spam':
            save_blocked(input_data.text, "Detected as spam")
            logger.info(f"Email blocked by {current_user}")
            return {
                "message": "Email blocked",
                "prediction": result
            }
        else:
            return {
                "message": "Email not blocked (not spam)",
                "prediction": result
            }
            
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Block error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Block operation failed"
        )


# ==================== Error Handlers ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    logger.error(f"HTTP Exception: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler."""
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "An unexpected error occurred"},
    )


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
Run the FastAPI backend server.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("spam_detector.src.api:app", host="0.0.0.0", port=8000, reload=True)

"""
Run the Streamlit frontend.
"""
import sys
import os
import subprocess

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

if __name__ == "__main__":
    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "spam_detector/src/app.py"
    ])

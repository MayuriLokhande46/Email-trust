import streamlit as st
import subprocess
import sys
import os
import time
import nltk

# --- NLTK Data Download ---
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        nltk.download('punkt_tab')
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")

# --- Backend Startup Logic ---
@st.cache_resource
def start_backend():
    print("🚀 Starting Backend API...")
    env = os.environ.copy()
    root_dir = os.path.dirname(os.path.abspath(__file__))
    env["PYTHONPATH"] = root_dir
    
    # Start the backend server
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "spam_detector.src.api:app", "--host", "0.0.0.0", "--port", "8000"],
        env=env,
        cwd=root_dir
    )
    
    time.sleep(5)
    return process

# Ensure backend and NLTK are ready only once
download_nltk_data()
start_backend()

# Add path for internal imports
sys.path.append(os.path.join(os.getcwd(), "spam_detector", "src"))

try:
    from spam_detector.src.app import login_page, spam_detector_app
    
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
        
    if st.session_state['authenticated']:
        spam_detector_app()
    else:
        login_page()
except Exception as e:
    st.error(f"Error starting application: {e}")

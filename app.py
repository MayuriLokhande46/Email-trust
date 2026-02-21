import streamlit as st
import subprocess
import sys
import os
import time
import nltk

# Set page config at the very beginning
st.set_page_config(page_title="Email Spam Detection", page_icon="📧", layout="centered")

# --- NLTK Data Download ---
@st.cache_resource
def download_nltk_data():
    for res in ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']:
        try:
            nltk.download(res, quiet=True)
        except Exception:
            pass

# --- Backend Startup Logic ---
@st.cache_resource(show_spinner=False)
def start_backend():
    env = os.environ.copy()
    root_dir = os.path.dirname(os.path.abspath(__file__))
    env["PYTHONPATH"] = root_dir
    
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "spam_detector.src.api:app", "--host", "0.0.0.0", "--port", "8000"],
            env=env,
            cwd=root_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(7)
        return process
    except Exception as e:
        return None

# Execution Flow
download_nltk_data()
start_backend()

# Path configuration
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)
src_dir = os.path.join(root_dir, "spam_detector", "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

try:
    from spam_detector.src.app import login_page, spam_detector_app, init_session_state
    
    # Initialize session state
    init_session_state()
    
    if st.session_state.get('authenticated', False):
        spam_detector_app()
    else:
        login_page()

except Exception as e:
    st.error("Something went wrong while starting the app.")
    st.exception(e)

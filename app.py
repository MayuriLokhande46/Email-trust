import streamlit as st
import subprocess
import sys
import os
import time

# Set page config at the very beginning
st.set_page_config(page_title="Email Spam Detector", page_icon="📧")

try:
    import nltk
    # --- NLTK Data Download ---
    @st.cache_resource
    def download_nltk_data():
        for res in ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']:
            try:
                nltk.download(res, quiet=True)
            except Exception as e:
                st.warning(f"Note: Could not download {res}: {e}")

    # --- Backend Startup Logic ---
    @st.cache_resource
    def start_backend():
        st.write("🚀 Initializing Backend services...")
        env = os.environ.copy()
        root_dir = os.path.dirname(os.path.abspath(__file__))
        env["PYTHONPATH"] = root_dir
        
        try:
            # Start the backend server
            process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "spam_detector.src.api:app", "--host", "0.0.0.0", "--port", "8000"],
                env=env,
                cwd=root_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            time.sleep(5)
            return process
        except Exception as e:
            st.error(f"Backend failed to start: {e}")
            return None

    # Step 1: NLTK
    download_nltk_data()
    
    # Step 2: Backend
    start_backend()

    # Step 3: Imports
    # Add root folder and src folder to path
    root_dir = os.path.dirname(os.path.abspath(__file__))
    if root_dir not in sys.path:
        sys.path.append(root_dir)
    
    src_dir = os.path.join(root_dir, "spam_detector", "src")
    if src_dir not in sys.path:
        sys.path.append(src_dir)

    # Now try to load the main app
    from spam_detector.src.app import login_page, spam_detector_app
    
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
        
    if st.session_state['authenticated']:
        spam_detector_app()
    else:
        login_page()

except Exception as e:
    st.error("Critical error during application startup.")
    st.exception(e)
    st.info("Check Streamlit Logs for more details.")

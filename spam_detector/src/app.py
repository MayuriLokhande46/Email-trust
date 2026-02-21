import streamlit as st
import pandas as pd
import io
import re
import requests
import os
import time
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import sys

# Ensure project root is in path for absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    import spam_detector.src.predict as predict_module
    predict = predict_module.predict
    import spam_detector.src.authentication as auth_direct
    import spam_detector.src.security as sec_direct
except ImportError:
    import predict as predict_module
    predict = predict_module.predict
    import authentication as auth_direct
    import security as sec_direct

# Initialize the users table directly (independent of backend)
try:
    auth_direct.create_table()
except Exception:
    pass

# --- Configuration ---
API_URL = os.getenv("API_URL", "http://localhost:8000")

# --- Initial Setup ---
# Note: set_page_config is set in root app.py to avoid conflicts

# Custom CSS for Premium Look
st.markdown("""
<style>
    /* Main Background & Centering */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #ffffff;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    .auth-container {
        width: 100%;
        max-width: 500px;
        margin: auto;
        padding: 40px;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        text-align: center;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700 !important;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        padding: 10px 25px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(79, 172, 254, 0.4) !important;
    }
    
    /* Inputs */
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border-radius: 10px !important;
    }
    
    /* Metric Card */
    [data-testid="stMetricValue"] {
        color: #00f2fe !important;
        font-size: 2rem !important;
    }
    
    /* Highlight Spam */
    mark {
        background-color: #ff4b4b;
        color: white;
        padding: 2px 4px;
        border-radius: 4px;
    }

    /* Link Safety Classes */
    .link-safe { color: #00ff00; font-weight: bold; }
    .link-danger { color: #ff4b4b; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Initialize session state for authentication
def init_session_state():
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    if 'username' not in st.session_state:
        st.session_state['username'] = ''
    if 'token' not in st.session_state:
        st.session_state['token'] = ''
    if 'auth_mode' not in st.session_state:
        st.session_state['auth_mode'] = 'Signup'

# --- Core App Functions ---
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def make_api_request(method, endpoint, data=None, params=None):
    headers = {}
    if st.session_state['token']:
        headers['Authorization'] = f"Bearer {st.session_state['token']}"
    
    url = f"{API_URL}{endpoint}"
    try:
        if method == 'GET':
            response = requests.get(url, headers=headers, params=params)
        elif method == 'POST':
            response = requests.post(url, headers=headers, json=data)
        
        if response.status_code in [200, 201]:
            return response.json()
        elif response.status_code == 401:
            st.error("Session expired. Please login again.")
            st.session_state['authenticated'] = False
            st.session_state['token'] = ''
            st.rerun()
        else:
            try:
                err_msg = response.json().get('error', response.json().get('detail', 'Unknown error'))
                st.error(f"API Error ({endpoint}): {err_msg}")
            except:
                st.error(f"API Error ({endpoint}): {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None

# --- Main Spam Detector App ---
def spam_detector_app():
    init_session_state()
    # Sidebar for extra info
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state['username']}")
        st.info("Role: Authorized User")
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state['authenticated'] = False
            st.session_state['username'] = ''
            st.session_state['token'] = ''
            st.rerun()

    st.title("Email Spam Detection")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 🔍 Analyze Single Email")
        user_input = st.text_area("Email Content:", height=300, placeholder="Paste email body here...")
        analyze_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

    with col2:
        st.markdown("### 📊 Analysis Results")
        if analyze_button and user_input.strip():
            # No spinner as requested: "ai thinking aisa show nhi hona chhaiye"
            result = make_api_request('POST', '/predict', data={"text": user_input})
            
            if result:
                label = result['label']
                prob_spam = result.get('prob_spam', 0)
                spam_words = result.get('spam_words', [])
                url_analysis = result.get('url_analysis', {})

                # Result Status
                if label == 'spam':
                    st.error("🚨 **SPAM DETECTED**", icon="🚨")
                    st.metric("Spam Confidence", f"{prob_spam*100:.1f}%")
                    st.progress(float(prob_spam))
                    
                    # Show heuristic explanations if any
                    explanations = result.get('explanation', [])
                    if explanations:
                        st.info("**Heuristic Analysis Signals:**\n" + "\n".join([f"- {e}" for e in explanations]))
                else:
                    prob_ham = 1 - prob_spam
                    st.success("✅ **SAFE EMAIL**", icon="✅")
                    st.metric("Safe Confidence", f"{prob_ham*100:.1f}%")
                    st.progress(float(prob_ham))

                # Display detailed signals
                with st.expander("📊 Detailed Spam Signal Analysis"):
                    col1, col2 = st.columns(2)
                    with col1:
                        adv_score = result.get('advanced_spam_score', 0)
                        st.write("**Advanced Risk Score:**")
                        st.write(f"{adv_score:.2f} / 1.0")
                        if adv_score > 0.4:
                            st.warning("⚠️ High structural risk detected")
                    
                    with col2:
                        st.write("**Language Detected:**")
                        st.code(result.get('language', 'en').upper())
                
                # Show full response for verification
                with st.expander("Details"):
                    st.json(result)
                
                # Link Verification Section
                st.markdown("---")
                st.markdown("### 🔗 Link Verification")
                if url_analysis and url_analysis.get('urls'):
                    urls = url_analysis.get('urls', [])
                    st.write(f"Found {len(urls)} links in the email.")
                    
                    for url in urls:
                        is_suspicious = False
                        reasons = []
                        
                        # Use flags from enhanced url_analyzer
                        url_lower = url.lower()
                        if any(s in url_lower for s in ['bit.ly', 'tinyurl', 't.co', 'goo.gl', 'ow.ly']):
                            is_suspicious = True
                            reasons.append("Shortened URL detected")
                        if re.match(r'https?://\d+\.\d+\.\d+\.\d+', url):
                            is_suspicious = True
                            reasons.append("IP-based URL detected")
                        
                        # New flags
                        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.pw', '.win', '.bid', '.loan']
                        if any(tld in url_lower for tld in suspicious_tlds):
                            is_suspicious = True
                            reasons.append(f"Suspicious TLD ({url_lower.split('.')[-1]})")
                        
                        if len(url) > 100:
                            is_suspicious = True
                            reasons.append("Highly obfuscated long URL")
                            
                        spam_keywords = ['verify', 'confirm', 'update', 'claim', 'login', 'secure', 'account', 'banking', 'prize', 'gift']
                        if any(k in url_lower for k in spam_keywords):
                            is_suspicious = True
                            reasons.append("Suspicious keyword in URL")
                            
                        if is_suspicious:
                            st.markdown(f"❌ `{url}` : <span class='link-danger'>Suspicious / Harmful</span>", unsafe_allow_html=True)
                            for r in reasons:
                                st.caption(f"Reason: {r}")
                        else:
                            st.markdown(f"✅ `{url}` : <span class='link-safe'>Safe</span>", unsafe_allow_html=True)
                else:
                    st.info("No links found in this email.")
                
                # Highlight Spam Words
                if spam_words and label == 'spam':
                    st.markdown("---")
                    st.markdown("**Highlighted Spam Signals:**")
                    highlighted_text = user_input
                    for word in spam_words[:15]: # Show more words
                        highlighted_text = re.sub(r'\b' + re.escape(word) + r'\b', f'<mark>{word}</mark>', highlighted_text, flags=re.IGNORECASE)
                    st.markdown(f'<div style="background: rgba(0,0,0,0.2); padding: 10px; border-radius: 5px; color: white;">{highlighted_text}</div>', unsafe_allow_html=True)
        else:
            st.write("Results will appear here after analysis.")

    st.markdown("---")

    # --- Bulk Analysis Feature ---
    st.header("📂 Bulk Email Analysis")
    uploaded_files = st.file_uploader("Upload CSV or TXT files for batch analysis", type=['csv', 'txt'], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                emails = []
                
                if uploaded_file.name.endswith('.csv'):
                    df_up = pd.read_csv(stringio)
                    email_col = next((c for c in df_up.columns if any(k in c.lower() for k in ['email', 'text'])), None)
                    if email_col:
                        emails = df_up[email_col].dropna().tolist()
                    else:
                        st.error(f"Could not find email/text column in {uploaded_file.name}")
                elif uploaded_file.name.endswith('.txt'):
                    emails = [line.strip() for line in stringio.readlines() if line.strip()]

                if emails:
                    if st.button(f"Analyze {len(emails)} Emails from {uploaded_file.name}"):
                        batch_emails = emails[:100]
                        response = make_api_request('POST', '/batch_predict', data={"texts": batch_emails})
                        if response:
                            results = response.get('results', [])
                            final_data = []
                            for i, res in enumerate(results):
                                final_data.append({
                                    'Email': batch_emails[i][:50] + "...",
                                    'Status': res.get('label', 'Error').upper(),
                                    'Spam Score': f"{res.get('prob_spam', 0)*100:.1f}%"
                                })
                            st.table(pd.DataFrame(final_data))

            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")

    st.markdown("---")

    # --- Prediction History ---
    st.header("🕒 Prediction History")
    show_history = st.toggle("Show Full History")
    if show_history:
        history_data = make_api_request('GET', '/history', params={"limit": 50})
        if history_data and 'history' in history_data:
            records = history_data['history']
            if records:
                # Use actual keys from API: timestamp, email_content, prediction, confidence
                df_hist = pd.DataFrame(records)
                df_hist.columns = ['Time', 'Email Preview', 'Result', 'Confidence']
                st.dataframe(df_hist, use_container_width=True)
            else:
                st.info("No history found.")

# --- Authentication Pages ---
def login_page():
    init_session_state()

    # Ensure auth_mode is set
    if 'auth_mode' not in st.session_state:
        st.session_state['auth_mode'] = 'Signup'

    st.markdown(
        "<h1 style='text-align:center; background: linear-gradient(90deg,#00f2fe,#4facfe); "
        "-webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:20px;'>"
        "📧 Email Trust</h1>",
        unsafe_allow_html=True
    )

    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📝 Sign Up", use_container_width=True,
                     type="primary" if st.session_state['auth_mode'] == 'Signup' else "secondary"):
            st.session_state['auth_mode'] = 'Signup'
            st.rerun()
    with col2:
        if st.button("🔑 Login", use_container_width=True,
                     type="primary" if st.session_state['auth_mode'] == 'Login' else "secondary"):
            st.session_state['auth_mode'] = 'Login'
            st.rerun()

    st.markdown("---")

    # ---- SIGN UP PAGE ----
    if st.session_state['auth_mode'] == 'Signup':
        st.subheader("✨ Create Account")
        with st.form("signup_form", clear_on_submit=True):
            new_user = st.text_input("Username")
            new_pass = st.text_input("Password", type="password")
            conf_pass = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("Join Now 🚀", use_container_width=True, type="primary")

        if submitted:
            if new_pass != conf_pass:
                st.error("❌ Passwords don't match")
            elif not new_user or not new_user.strip():
                st.warning("⚠️ Username is required")
            else:
                try:
                    success = auth_direct.add_user(new_user.strip(), new_pass)
                    if success:
                        st.success("🎉 Account created! Taking you to Login...")
                        time.sleep(1.5)
                        st.session_state['auth_mode'] = 'Login'  # Auto switch to login
                        st.rerun()
                    else:
                        st.error("⚠️ Username already exists. Try a different one.")
                except Exception as e:
                    st.error(f"Registration error: {str(e)}")

    # ---- LOGIN PAGE ----
    else:
        st.subheader("👋 Welcome Back")
        with st.form("login_form", clear_on_submit=True):
            user = st.text_input("Username")
            pw = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In 🔑", use_container_width=True, type="primary")

        if submitted:
            try:
                db_user = auth_direct.get_user(user.strip())
                if db_user and auth_direct.check_password(db_user[2], pw):
                    try:
                        token = sec_direct.create_access_token({"sub": user.strip()})
                    except Exception:
                        token = 'local-auth'
                    st.session_state['authenticated'] = True
                    st.session_state['username'] = user.strip()
                    st.session_state['token'] = token
                    st.success("✅ Login successful! Opening Dashboard...")
                    time.sleep(1)
                    st.rerun()  # Auto-opens dashboard
                else:
                    st.error("❌ Invalid username or password.")
            except Exception as e:
                st.error(f"Login error: {str(e)}")


# --- Main Flow ---
if __name__ == "__main__":
    init_session_state()
    if st.session_state['authenticated']:
        spam_detector_app()
    else:
        login_page()

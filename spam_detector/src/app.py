import streamlit as st
import nltk
import time
import pandas as pd
import io
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from preprocess import clean_text
from predict import predict
from database import init_db, save_prediction, get_all_predictions
import authentication as auth

# --- Initial Setup ---
st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="wide")

# Initialize databases
# For prediction history
init_db() 
# For user authentication
user_db_conn = auth.create_connection()
auth.create_table(user_db_conn)

# Initialize session state for authentication
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ''

# --- NLTK Resource Download ---
# Moved to a function to be more organized
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
download_nltk_data()


# --- Core App Functions ---
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color="black", colormap='Reds', max_words=100).generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

# --- Main Spam Detector App ---
def spam_detector_app():
    # Sidebar for extra info
    with st.sidebar:
        st.title("ℹ️ Info")
        with st.expander("ℹ️ About this App"):
            st.write("This Machine Learning model analyzes emails to detect if they are Spam or Safe (Ham).")
        st.markdown("---")

    st.title("📧 Email Spam Detector")
    st.markdown("Paste your email text below and click the **Analyze** button.")

    user_input = st.text_area("Email Content:", height=250, placeholder="Paste text here...")
    analyze_button = st.button("🔍 Analyze Email", type="primary")

    st.markdown("---")

    # --- Result Display ---
    if analyze_button:
        if user_input.strip():
            with st.spinner("Analyzing..."):
                time.sleep(0.5)  # Visual effect
                result = predict(user_input)
                label = result['label']
                prob_spam = result.get('prob_spam', 0)
                spam_words = result.get('spam_words', [])

            result_container = st.container(border=True)
            with result_container:
                if label == 'spam':
                    st.error("🚨 **SPAM DETECTED!**", icon="🚨")
                    st.metric(label="Confidence", value=f"{prob_spam*100:.2f}%")
                    st.progress(float(prob_spam))
                    save_prediction(user_input, "spam", prob_spam)

                    if spam_words:
                        st.markdown("---")
                        st.subheader("Highlight Spam Words:")
                        highlighted_text = user_input
                        for word in spam_words:
                            highlighted_text = re.sub(r'\b' + re.escape(word) + r'\b', f'<mark>{word}</mark>', highlighted_text, flags=re.IGNORECASE)
                        st.markdown(highlighted_text, unsafe_allow_html=True)
                else:
                    prob_ham = 1 - prob_spam
                    st.success("✅ **HAM (Safe Email)**", icon="✅")
                    st.metric(label="Confidence", value=f"{prob_ham*100:.2f}%")
                    st.progress(float(prob_ham))
                    save_prediction(user_input, "ham", prob_ham)
        else:
            st.warning("👈 Please enter some text to analyze.", icon="⚠️")

    st.markdown("---")

    # --- Bulk Analysis Feature ---
    st.header("📧 Bulk Email Analysis")

    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'txt'])

    if uploaded_file is not None:
        try:
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            emails = []
            
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(stringio)
                email_col = None
                for col in df.columns:
                    if 'email' in col.lower() or 'text' in col.lower():
                        email_col = col
                        break
                if email_col:
                    emails = df[email_col].dropna().tolist()
                    st.info(f"Found {len(emails)} emails in column '{email_col}'.")
                else:
                    st.error("Could not find a column with 'email' or 'text' in the CSV.")

            elif uploaded_file.name.endswith('.txt'):
                emails = [line.strip() for line in stringio.readlines() if line.strip()]
                st.info(f"Found {len(emails)} emails in the text file.")

            if emails:
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, email_text in enumerate(emails):
                    prediction = predict(email_text)
                    results.append({
                        'Email': email_text,
                        'Prediction': prediction['label'],
                        'Confidence (Spam)': f"{prediction.get('prob_spam', 0)*100:.2f}%"
                    })
                    progress_percentage = (i + 1) / len(emails)
                    progress_bar.progress(progress_percentage)
                    status_text.text(f"Analyzing email {i+1}/{len(emails)}")

                progress_bar.empty()
                status_text.empty()

                results_df = pd.DataFrame(results)
                st.subheader("Analysis Results")
                st.dataframe(results_df)

                csv_data = convert_df_to_csv(results_df)

                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_data,
                    file_name='spam_analysis_results.csv',
                    mime='text/csv',
                )

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

    st.markdown("---")

    # --- Prediction History ---
    st.header("📧 Prediction History")

    show_history = st.toggle("Show Full History")

    if show_history:
        history_records = get_all_predictions()
        if history_records:
            history_df = pd.DataFrame(history_records, columns=['Timestamp', 'Email Content', 'Prediction', 'Confidence'])
            history_df['Confidence'] = history_df['Confidence'].apply(lambda x: f"{x*100:.2f}%")
            st.dataframe(history_df, use_container_width=True)
        else:
            st.info("No prediction history found.")

# --- Authentication Pages ---
def login_page():
    st.title("Welcome to the Spam Detector")
    st.write("Please sign up or log in to continue.")

    tabs = st.tabs(["Sign Up", "Login"])

    # Sign Up Tab
    with tabs[0]:
        with st.form("signup_form"):
            st.subheader("Create a New Account")
            new_username = st.text_input("Choose a Username", key="signup_username")
            new_password = st.text_input("Choose a Password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm_password")
            signup_button = st.form_submit_button("Sign Up")

            if signup_button:
                if not new_username or not new_password:
                    st.warning("Username and password cannot be empty.")
                elif new_password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    if auth.add_user(user_db_conn, new_username, new_password):
                        st.success("Account created successfully! Please go to the Login tab to log in.")
                    else:
                        st.error("Username already exists. Please choose another one.")

    # Login Tab
    with tabs[1]:
        with st.form("login_form"):
            st.subheader("Login")
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submit_button = st.form_submit_button("Login")

            if submit_button:
                user = auth.get_user(user_db_conn, username)
                if user and auth.check_password(user[2], password):
                    st.session_state['authenticated'] = True
                    st.session_state['username'] = username
                    st.rerun() 
                else:
                    st.error("Invalid username or password.")


# --- Main Control Flow ---
if st.session_state['authenticated']:
    with st.sidebar:
        st.success(f"Logged in as **{st.session_state['username']}**")
        if st.button("Logout"):
            st.session_state['authenticated'] = False
            st.session_state['username'] = ''
            st.rerun() # Rerun to show the login page
    spam_detector_app()
else:
    login_page()


import sqlite3
import hashlib
import os
from contextlib import contextmanager
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Use /tmp for writable storage on Streamlit Cloud (read-only filesystem)
# Falls back to local data/ directory for local dev
def _get_db_path():
    tmp_path = '/tmp/users.db'
    local_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    local_path = os.path.join(local_dir, 'users.db')
    # On cloud, /tmp is always writable
    try:
        os.makedirs('/tmp', exist_ok=True)
        test_file = '/tmp/.write_test'
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        logger.info(f'Using /tmp database path')
        return tmp_path
    except Exception:
        os.makedirs(local_dir, exist_ok=True)
        logger.info(f'Using local database path: {local_path}')
        return local_path

DB_PATH = _get_db_path()

@contextmanager
def get_db_connection():
    """Context manager for authentication database connections."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        yield conn
    except Exception as e:
        logger.error(f"Authentication DB Connection Error: {e}")
        raise
    finally:
        if conn:
            conn.close()

def create_table():
    """Initializes the users table."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    password TEXT NOT NULL
                )
            """)
            conn.commit()
            logger.info("Users table verified/created.")
    except Exception as e:
        logger.error(f"Failed to create users table: {e}")

def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_password(hashed_password, user_password):
    return hashed_password == hashlib.sha256(str.encode(user_password)).hexdigest()

def add_user(username, password):
    """Adds a new user to the database."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                         (username, hash_password(password)))
            conn.commit()
            return True
    except sqlite3.IntegrityError:
        logger.warning(f"Registration failed: Username '{username}' already exists.")
        return False
    except Exception as e:
        logger.error(f"Error adding user: {e}")
        raise

def get_user(username):
    """Retrieves a user by username."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            return cursor.fetchone()
    except Exception as e:
        logger.error(f"Error getting user: {e}")
        return None

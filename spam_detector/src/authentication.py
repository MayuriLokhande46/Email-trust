import sqlite3
import hashlib
import os
from contextlib import contextmanager
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Use absolute path for users.db
ROOT = os.path.dirname(os.path.dirname(__file__))
DB_PATH = os.path.join(ROOT, 'data', 'users.db')

# Ensure directory exists
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

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

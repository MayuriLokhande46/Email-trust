import sqlite3
import hashlib
import os

# Use absolute path for users.db
ROOT = os.path.dirname(os.path.dirname(__file__))
DB_PATH = os.path.join(ROOT, 'data', 'users.db')

def create_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

def create_table(conn):
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL
            )
        """)
        conn.commit()
    except Exception as e:
        print(e)


def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_password(hashed_password, user_password):
    return hashed_password == hashlib.sha256(str.encode(user_password)).hexdigest()

def add_user(conn, username, password):
    try:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def get_user(conn, username):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    return cursor.fetchone()

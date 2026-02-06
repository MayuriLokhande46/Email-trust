import sqlite3
import os
import logging
from datetime import datetime
from contextlib import contextmanager

# Set up logging
logger = logging.getLogger(__name__)

# Define the path for the database in the data directory
ROOT = os.path.dirname(os.path.dirname(__file__))
DB_PATH = os.path.join(ROOT, 'data', 'predictions.db')
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# Connection timeout to handle concurrent access
DB_TIMEOUT = 10.0


@contextmanager
def get_db_connection():
    """Context manager for database connections with proper error handling."""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
        conn.row_factory = sqlite3.Row
        yield conn
    except sqlite3.OperationalError as e:
        logger.error(f'Database connection error: {str(e)}')
        raise
    finally:
        try:
            conn.close()
        except Exception as e:
            logger.error(f'Error closing database connection: {str(e)}')


def init_db():
    """Initializes the database and creates tables if they don't exist."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Create predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    email_content TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create blocked_emails table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS blocked_emails (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    email_content TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create index for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_predictions_timestamp 
                ON predictions(timestamp DESC)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_blocked_timestamp 
                ON blocked_emails(timestamp DESC)
            ''')
            
            conn.commit()
            logger.info('Database initialized successfully')
            
    except Exception as e:
        logger.error(f'Failed to initialize database: {str(e)}')
        raise


def save_prediction(content: str, prediction: str, confidence: float) -> bool:
    """
    Saves a prediction record to the database.
    
    Args:
        content (str): Email content
        prediction (str): 'spam' or 'ham'
        confidence (float): Confidence score (0.0 to 1.0)
        
    Returns:
        bool: True if saved successfully, False otherwise
    """
    try:
        if not isinstance(content, str) or not content.strip():
            raise ValueError('Content must be a non-empty string')
        if prediction not in ['spam', 'ham']:
            raise ValueError('Prediction must be "spam" or "ham"')
        if not (0.0 <= confidence <= 1.0):
            raise ValueError('Confidence must be between 0.0 and 1.0')
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions (timestamp, email_content, prediction, confidence)
                VALUES (?, ?, ?, ?)
            ''', (datetime.now(), content, prediction, confidence))
            conn.commit()
            logger.info(f'Prediction saved: {prediction} (confidence: {confidence:.2f})')
            return True
            
    except ValueError as e:
        logger.warning(f'Invalid prediction data: {str(e)}')
        return False
    except Exception as e:
        logger.error(f'Failed to save prediction: {str(e)}')
        return False


def save_blocked(content: str, reason: str = "Detected as spam") -> bool:
    """
    Saves a blocked email record to the database.
    
    Args:
        content (str): Email content
        reason (str): Reason for blocking
        
    Returns:
        bool: True if saved successfully, False otherwise
    """
    try:
        if not isinstance(content, str) or not content.strip():
            raise ValueError('Content must be a non-empty string')
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO blocked_emails (timestamp, email_content, reason)
                VALUES (?, ?, ?)
            ''', (datetime.now(), content, reason))
            conn.commit()
            logger.info(f'Email blocked: {reason}')
            return True
            
    except ValueError as e:
        logger.warning(f'Invalid blocked email data: {str(e)}')
        return False
    except Exception as e:
        logger.error(f'Failed to save blocked email: {str(e)}')
        return False


def get_all_predictions(limit: int = None):
    """
    Retrieves all prediction records from the database.
    
    Args:
        limit (int, optional): Maximum number of records to return
        
    Returns:
        list: List of prediction records
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            query = 'SELECT timestamp, email_content, prediction, confidence FROM predictions ORDER BY timestamp DESC'
            if limit:
                query += f' LIMIT {limit}'
            
            cursor.execute(query)
            records = cursor.fetchall()
            logger.info(f'Retrieved {len(records)} prediction records')
            return records
            
    except Exception as e:
        logger.error(f'Failed to retrieve predictions: {str(e)}')
        return []


def get_blocked_emails(limit: int = None):
    """
    Retrieves all blocked email records from the database.
    
    Args:
        limit (int, optional): Maximum number of records to return
        
    Returns:
        list: List of blocked email records
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            query = 'SELECT timestamp, email_content, reason FROM blocked_emails ORDER BY timestamp DESC'
            if limit:
                query += f' LIMIT {limit}'
            
            cursor.execute(query)
            records = cursor.fetchall()
            logger.info(f'Retrieved {len(records)} blocked email records')
            return records
            
    except Exception as e:
        logger.error(f'Failed to retrieve blocked emails: {str(e)}')
        return []


def delete_old_records(days: int = 30) -> int:
    """
    Deletes records older than specified days.
    
    Args:
        days (int): Delete records older than this many days
        
    Returns:
        int: Number of records deleted
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Delete old predictions
            cursor.execute('''
                DELETE FROM predictions 
                WHERE datetime(timestamp) < datetime('now', ? || ' days')
            ''', (f'-{days}',))
            
            deleted_predictions = cursor.rowcount
            
            # Delete old blocked emails
            cursor.execute('''
                DELETE FROM blocked_emails 
                WHERE datetime(timestamp) < datetime('now', ? || ' days')
            ''', (f'-{days}',))
            
            deleted_blocked = cursor.rowcount
            conn.commit()
            
            total_deleted = deleted_predictions + deleted_blocked
            logger.info(f'Deleted {total_deleted} old records ({deleted_predictions} predictions, {deleted_blocked} blocked)')
            return total_deleted
            
    except Exception as e:
        logger.error(f'Failed to delete old records: {str(e)}')
        return 0


if __name__ == '__main__':
    # Example usage
    init_db()
    print("Database initialized...")
    
    # Test saving predictions
    save_prediction("This is a test email", "ham", 0.98)
    save_prediction("URGENT: Win money now! Click here!", "spam", 0.95)
    
    print("\nAll predictions:")
    for pred in get_all_predictions():
        print(pred)

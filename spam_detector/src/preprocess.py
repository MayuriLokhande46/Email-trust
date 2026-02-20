import re
import nltk
import logging
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Set up logging
logger = logging.getLogger(__name__)

# Ensure nltk resources are available
try:
    _stopwords = set(stopwords.words('english'))
except Exception as e:
    logger.warning(f'Failed to load stopwords: {str(e)}. Using empty set.')
    _stopwords = set()

try:
    lemmatizer = WordNetLemmatizer()
except Exception as e:
    logger.error(f'Failed to initialize lemmatizer: {str(e)}')
    raise


def clean_text(text: str) -> str:
    """
    Clean and preprocess email text.
    
    Steps:
    - Convert to lowercase
    - Remove URLs
    - Remove special characters
    - Tokenize and lemmatize
    - Remove stopwords
    
    Args:
        text (str): Raw email text
        
    Returns:
        str: Cleaned and preprocessed text
        
    Raises:
        ValueError: If input is not a string
    """
    try:
        if not isinstance(text, str):
            text = str(text)
        
        if not text:
            return ''
        
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+', ' ', text)
        text = re.sub(r'www\S+', ' ', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', ' ', text)
        
        # Remove special characters (keep alphanumeric, space, $, and !)
        text = re.sub(r'[^a-z0-9\s$!]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        try:
            tokens = nltk.word_tokenize(text)
        except Exception as e:
            logger.warning(f'NLTK tokenization failed: {str(e)}. Using split() instead.')
            tokens = text.split()
        
        # Filter and lemmatize
        processed_tokens = []
        for token in tokens:
            if not token:
                continue
            
            # Keep single-letter alphabetic tokens, tokens with $ or !, or non-stopword tokens
            is_special = '$' in token or '!' in token
            if (len(token) == 1 and token.isalpha()) or is_special or (token not in _stopwords):
                try:
                    lemmatized = lemmatizer.lemmatize(token)
                    processed_tokens.append(lemmatized)
                except Exception as e:
                    logger.debug(f'Lemmatization failed for token "{token}": {str(e)}')
                    processed_tokens.append(token)
        
        result = ' '.join(processed_tokens)
        return result
        
    except Exception as e:
        logger.error(f'Error in clean_text: {str(e)}')
        # Fallback: return lowercase version without special chars
        return re.sub(r'[^a-z0-9\s]', ' ', str(text).lower()).strip()


if __name__ == '__main__':
    test_cases = [
        "Congratulations! You've won $1000. Click here: http://spam.example",
        "Meeting tomorrow at 3 PM",
        "Special offer: FREE MONEY! Visit www.spam.com",
        "user@email.com sent you a message",
    ]
    
    for test in test_cases:
        print(f"Original: {test}")
        print(f"Cleaned:  {clean_text(test)}")
        print()

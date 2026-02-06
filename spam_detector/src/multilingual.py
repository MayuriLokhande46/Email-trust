"""
Multi-language support for spam detection.

Supports: English, Hindi, Urdu, Spanish, French, German
- Auto-detect language
- Language-specific preprocessing
- Multi-language feature extraction
"""

import logging
import re
from typing import Tuple, Dict, Any
from collections import Counter

logger = logging.getLogger(__name__)

# Try to import language detection libraries
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from langdetect import detect, detect_langs, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False


class LanguageDetector:
    """Detect language of email text."""

    # Supported languages
    SUPPORTED_LANGUAGES = {
        'en': 'English',
        'hi': 'Hindi',
        'ur': 'Urdu',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'pt': 'Portuguese',
        'it': 'Italian',
        'ru': 'Russian',
        'ja': 'Japanese',
        'zh': 'Chinese'
    }

    # Language-specific spam keywords
    SPAM_KEYWORDS = {
        'en': {
            'free', 'money', 'urgent', 'congratulations', 'winner',
            'click', 'limited', 'offer', 'act now', 'verify', 'confirm'
        },
        'hi': {
            'आज़ाद', 'पैसा', 'जीतना', 'खरीदें', 'तुरंत', 'क्लिक',
            'सीमित', 'समय', 'जल्दी', 'पुरस्कार', 'निःशुल्क'
        },
        'ur': {
            'مفت', 'رقم', 'جیتنا', 'جلدی', 'کلک', 'محدود',
            'فوری', 'انعام', 'تصدیق', 'تازہ ترین'
        },
        'es': {
            'gratis', 'dinero', 'urgente', 'click', 'ganar',
            'felicitaciones', 'limitado', 'oferta', 'verificar'
        },
        'fr': {
            'gratuit', 'argent', 'gagnant', 'urgent', 'cliquer',
            'félicitations', 'limité', 'offre', 'vérifier'
        },
        'de': {
            'kostenlos', 'geld', 'gewinner', 'dringend', 'klicken',
            'beglückwünschung', 'begrenzt', 'angebot', 'überprüfen'
        }
    }

    def __init__(self):
        """Initialize language detector."""
        self.detected_language = None
        self.confidence = 0.0

    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect the language of the text.

        Args:
            text (str): Text to analyze

        Returns:
            Tuple of (language_code, confidence)
        """
        try:
            if not text or len(text) < 10:
                return 'en', 0.5  # Default to English if text too short

            # Try langdetect first (more reliable)
            if LANGDETECT_AVAILABLE:
                try:
                    detected = detect(text)
                    probabilities = detect_langs(text)
                    confidence = max([p.prob for p in probabilities])
                    
                    self.detected_language = detected
                    self.confidence = confidence
                    logger.info(f'Language detected: {detected} (confidence: {confidence:.2f})')
                    return detected, confidence
                except LangDetectException:
                    pass

            # Fallback to TextBlob
            if TEXTBLOB_AVAILABLE:
                blob = TextBlob(text)
                detected = blob.detect_language()
                self.detected_language = detected
                self.confidence = 0.7  # TextBlob doesn't return confidence
                logger.info(f'Language detected (TextBlob): {detected}')
                return detected, self.confidence

            # Fallback: detect by common words
            return self._detect_by_keywords(text)

        except Exception as e:
            logger.warning(f'Language detection error: {str(e)}. Defaulting to English.')
            return 'en', 0.0

    def _detect_by_keywords(self, text: str) -> Tuple[str, float]:
        """
        Fallback: detect language by keyword matching.

        Args:
            text (str): Text to analyze

        Returns:
            Tuple of (language_code, confidence)
        """
        text_lower = text.lower()
        language_scores = {}

        for lang_code, keywords in self.SPAM_KEYWORDS.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            language_scores[lang_code] = matches

        if language_scores:
            detected_lang = max(language_scores, key=language_scores.get)
            confidence = language_scores[detected_lang] / len(self.SPAM_KEYWORDS[detected_lang])
            return detected_lang, confidence

        return 'en', 0.0  # Default to English

    def get_supported_languages(self) -> Dict[str, str]:
        """Get list of supported languages."""
        return self.SUPPORTED_LANGUAGES.copy()


class MultiLanguagePreprocessor:
    """Language-specific text preprocessing."""

    # Stopwords by language
    STOPWORDS = {
        'en': {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were'
        },
        'hi': {
            'और', 'या', 'लेकिन', 'में', 'पर', 'को', 'से', 'यह',
            'वह', 'है', 'हैं', 'था', 'थे', 'किया'
        },
        'ur': {
            'اور', 'یا', 'لیکن', 'میں', 'پر', 'کو', 'سے', 'یہ',
            'وہ', 'ہے', 'ہیں', 'تھا', 'تھے'
        },
        'es': {
            'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'ser',
            'se', 'no', 'haber', 'por', 'con', 'su'
        },
        'fr': {
            'le', 'de', 'a', 'et', 'que', 'en', 'un', 'qu', 'dans',
            'pour', 'pas', 'vous', 'ne', 'sur'
        },
        'de': {
            'der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das',
            'mit', 'sich', 'des', 'auf', 'für', 'ist'
        }
    }

    def __init__(self):
        """Initialize preprocessor."""
        self.language = 'en'
        self.detector = LanguageDetector()

    def preprocess_multilingual(self, text: str) -> Tuple[str, str]:
        """
        Preprocess text with language detection.

        Args:
            text (str): Text to preprocess

        Returns:
            Tuple of (cleaned_text, language_code)
        """
        # Detect language
        lang_code, confidence = self.detector.detect_language(text)
        self.language = lang_code

        logger.info(f'Processing {lang_code} text (confidence: {confidence:.2f})')

        # Preprocess based on language
        if lang_code == 'hi':
            cleaned = self._preprocess_hindi(text)
        elif lang_code == 'ur':
            cleaned = self._preprocess_urdu(text)
        elif lang_code == 'es':
            cleaned = self._preprocess_spanish(text)
        elif lang_code == 'fr':
            cleaned = self._preprocess_french(text)
        elif lang_code == 'de':
            cleaned = self._preprocess_german(text)
        else:
            cleaned = self._preprocess_english(text)

        return cleaned, lang_code

    def _preprocess_english(self, text: str) -> str:
        """Preprocess English text."""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', ' ', text)
        # Remove emails
        text = re.sub(r'\S+@\S+', ' ', text)
        # Remove special characters
        text = re.sub(r'[^a-z0-9\s]', ' ', text.lower())
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove stopwords
        words = text.split()
        words = [w for w in words if w not in self.STOPWORDS.get('en', set())]
        return ' '.join(words)

    def _preprocess_hindi(self, text: str) -> str:
        """Preprocess Hindi text."""
        # Remove URLs and emails
        text = re.sub(r'http\S+|www\S+|\S+@\S+', ' ', text)
        # Remove extra spaces (preserve Devanagari script)
        text = re.sub(r'[^\u0900-\u097F\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove stopwords
        words = text.split()
        words = [w for w in words if w not in self.STOPWORDS.get('hi', set())]
        return ' '.join(words)

    def _preprocess_urdu(self, text: str) -> str:
        """Preprocess Urdu text."""
        # Remove URLs and emails
        text = re.sub(r'http\S+|www\S+|\S+@\S+', ' ', text)
        # Remove extra spaces (preserve Urdu script)
        text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove stopwords
        words = text.split()
        words = [w for w in words if w not in self.STOPWORDS.get('ur', set())]
        return ' '.join(words)

    def _preprocess_spanish(self, text: str) -> str:
        """Preprocess Spanish text."""
        text = re.sub(r'http\S+|www\S+|\S+@\S+', ' ', text)
        text = re.sub(r'[^a-záéíóúñ0-9\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        words = text.split()
        words = [w for w in words if w not in self.STOPWORDS.get('es', set())]
        return ' '.join(words)

    def _preprocess_french(self, text: str) -> str:
        """Preprocess French text."""
        text = re.sub(r'http\S+|www\S+|\S+@\S+', ' ', text)
        text = re.sub(r'[^a-zàâäæçéèêëïîôùûüœ0-9\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        words = text.split()
        words = [w for w in words if w not in self.STOPWORDS.get('fr', set())]
        return ' '.join(words)

    def _preprocess_german(self, text: str) -> str:
        """Preprocess German text."""
        text = re.sub(r'http\S+|www\S+|\S+@\S+', ' ', text)
        text = re.sub(r'[^a-zäöüß0-9\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        words = text.split()
        words = [w for w in words if w not in self.STOPWORDS.get('de', set())]
        return ' '.join(words)

    def get_spam_keywords_for_language(self, lang_code: str) -> set:
        """Get spam keywords for a specific language."""
        return self.SPAM_KEYWORDS.get(lang_code, self.SPAM_KEYWORDS['en']).copy()


class LanguageAwareSpamDetector:
    """Wrapper for language-aware spam detection."""

    def __init__(self):
        """Initialize detector."""
        self.detector = LanguageDetector()
        self.preprocessor = MultiLanguagePreprocessor()

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze text with language support.

        Args:
            text (str): Email text

        Returns:
            Dict with language analysis
        """
        try:
            # Detect language
            lang_code, confidence = self.detector.detect_language(text)

            # Preprocess
            cleaned, detected_lang = self.preprocessor.preprocess_multilingual(text)

            # Get spam keywords for language
            spam_keywords = self.preprocessor.get_spam_keywords_for_language(lang_code)

            # Check for spam keywords
            text_lower = text.lower()
            found_keywords = [kw for kw in spam_keywords if kw in text_lower]

            return {
                'language': lang_code,
                'language_name': self.detector.SUPPORTED_LANGUAGES.get(lang_code, 'Unknown'),
                'confidence': float(confidence),
                'cleaned_text': cleaned,
                'spam_keywords_found': found_keywords,
                'keyword_count': len(found_keywords),
                'is_supported': lang_code in self.detector.SUPPORTED_LANGUAGES
            }

        except Exception as e:
            logger.error(f'Language analysis error: {str(e)}')
            return {
                'language': 'en',
                'error': str(e),
                'is_supported': True
            }


if __name__ == '__main__':
    # Test multi-language support
    test_texts = {
        'English': "Congratulations! You won $1000. Click here now!",
        'Hindi': "बधाई हो! आप 1000 जीते हैं। अभी क्लिक करें!",
        'Urdu': "مبارک ہو! آپ 1000 روپے جیتے ہیں۔ ابھی کلک کریں!",
        'Spanish': "¡Felicidades! Ganaste $1000. ¡Haz clic aquí ahora!",
        'French': "Félicitations! Vous avez gagné 1000 $. Cliquez maintenant!",
        'German': "Glückwunsch! Sie haben 1000 $ gewonnen. Klicken Sie jetzt!"
    }

    detector = LanguageAwareSpamDetector()

    print("=" * 60)
    print("Multi-Language Spam Detection Test")
    print("=" * 60)

    for lang_name, text in test_texts.items():
        result = detector.analyze(text)
        print(f"\n{lang_name}:")
        print(f"  Detected: {result['language_name']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Spam Keywords Found: {result['keyword_count']}")
        print(f"  Keywords: {', '.join(result['spam_keywords_found'][:3])}")

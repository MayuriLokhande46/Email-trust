from deep_translator import GoogleTranslator
from deep_translator.detection import single_detection

def detect_language(text):
    """
    Detects the language of a given text.

    Args:
        text (str): The text to analyze.

    Returns:
        str: The detected language code (e.g., 'en', 'fr').
    """
    try:
        return single_detection(text, api_key=None)
    except:
        return 'en'  # Default to English on error

def translate_to_english(text, src_lang):
    """
    Translates text to English if it's not already in English.

    Args:
        text (str): The text to translate.
        src_lang (str): The source language code.

    Returns:
        str: The translated text (in English).
    """
    if src_lang == 'en':
        return text
    
    try:
        return GoogleTranslator(source=src_lang, target='en').translate(text)
    except:
        return text  # Return original text on error
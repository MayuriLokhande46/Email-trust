import re
import os
import joblib
import json
import logging
from typing import Dict, Any
try:
    from spam_detector.src.preprocess import clean_text
    from spam_detector.src.features import load_vectorizer
    from spam_detector.src.feature_engineering import (
        EmailHeaderExtractor, SenderReputationChecker, EmailUrlExtractor
    )
    from spam_detector.src.email_headers import CompleteHeaderAnalyzer
    from spam_detector.src.multilingual import detect_language, translate_to_english
except ImportError:
    from preprocess import clean_text
    from features import load_vectorizer
    from feature_engineering import (
        EmailHeaderExtractor, SenderReputationChecker, EmailUrlExtractor
    )
    from email_headers import CompleteHeaderAnalyzer
    from multilingual import detect_language, translate_to_english

# Set up logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(ROOT, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'spam_model.joblib')
VECT_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')
SPAM_WORDS_PATH = os.path.join(MODEL_DIR, 'spam_words.json')

# Load model and vectorizer with error handling
model = None
vect = None
spam_words_list = set()

# Initialize feature engineering tools
header_extractor = EmailHeaderExtractor()
reputation_checker = SenderReputationChecker()
url_analyzer = EmailUrlExtractor()
header_analyzer = CompleteHeaderAnalyzer()  # New: Email header parsing and validation

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f'Model not found at {MODEL_PATH}. Run src/train.py first.')
    if not os.path.exists(VECT_PATH):
        raise FileNotFoundError(f'Vectorizer not found at {VECT_PATH}. Run src/train.py first.')
    
    model = joblib.load(MODEL_PATH)
    logger.info(f'Model loaded successfully from {MODEL_PATH}')
    
    vect = load_vectorizer(VECT_PATH)
    logger.info(f'Vectorizer loaded successfully from {VECT_PATH}')
    
    # Load spam words if the file exists
    if os.path.exists(SPAM_WORDS_PATH):
        with open(SPAM_WORDS_PATH, 'r', encoding='utf-8') as f:
            spam_words_list = set(json.load(f))
        logger.info(f'Loaded {len(spam_words_list)} spam words from {SPAM_WORDS_PATH}')
    else:
        logger.warning(f'Spam words file not found at {SPAM_WORDS_PATH}')
        
except Exception as e:
    logger.error(f'Failed to load model/vectorizer: {str(e)}')
    raise SystemExit(f'Critical error: {str(e)}')


def predict(text: str, use_advanced_features: bool = True) -> Dict[str, Any]:
    """
    Predict if the given text is spam or ham.
    """
    # 1. Initialize ALL variables at the top of function scope for safety
    reputation = {}
    advanced_spam_score = 0.0
    header_features = {}
    url_analysis = {}
    header_analysis = {'status': 'failed'}
    heuristic_spam = False
    heuristic_reasons = []
    translated_text = text
    detected_lang = 'en'
    found_spam_words = []
    
    try:
        if not isinstance(text, str):
            raise ValueError('Input must be a string')
        
        if not text or text.isspace():
            raise ValueError('Input text cannot be empty')
        
        if model is None or vect is None:
            raise RuntimeError('Model or vectorizer not loaded. Check initialization logs.')
        
        # Language detection and translation
        try:
            detected_lang = detect_language(text)
            if detected_lang != 'en':
                translated_text = translate_to_english(text, source_lang=detected_lang)
                logger.info(f'Translated {detected_lang} content for analysis')
        except Exception as e:
            logger.warning(f'Language handling error: {str(e)}')

        # 2. Heuristic-based direct spam detection (Raw text check)
        raw_text_lower = text.lower()
        
        # Broad Phishing & Call-to-action patterns
        if re.search(r'click.*here.*to.*(verify|confirm|re-activate|update|authenticate)', raw_text_lower) or \
           re.search(r'verify.*your.*(account|identity|email|details)', raw_text_lower) or \
           re.search(r'security.*(alert|update).*account', raw_text_lower) or \
           re.search(r'unauthorized.*(access|activity)', raw_text_lower) or \
           re.search(r'login.*to.*(restore|secure|verify)', raw_text_lower):
            heuristic_spam = True
            heuristic_reasons.append("Phishing pattern detected")
        
        # Broad Financial/Lottery patterns
        if re.search(r'\$\d+.*(million|thousand|cash|bonus|prize|reward)', raw_text_lower) or \
           re.search(r'winner.*of.*(lottery|draw|sweepstakes|prize)', raw_text_lower) or \
           re.search(r'claim.*your.*(cash|reward|prizes|bonus)', raw_text_lower) or \
           re.search(r'receive.*\$[0-9]+', raw_text_lower):
            heuristic_spam = True
            heuristic_reasons.append("Financial/Lottery pattern detected")

        # Broad Urgency & Account Status
        if re.search(r'(urgent|important|action).*required', raw_text_lower) or \
           re.search(r'account.*(suspended|locked|blocked|temporary|frozen)', raw_text_lower) or \
           re.search(r'final.*notice|last.*warning|expires.*in.*\d+.*hours', raw_text_lower) or \
           re.search(r'immediate.*action.*needed', raw_text_lower):
            heuristic_spam = True
            heuristic_reasons.append("Urgency/Account status manipulation")
            
        # 3. Clean and prepare for ML model
        clean = clean_text(translated_text)
        found_spam_words = [word for word in clean.split() if word in spam_words_list]

        # Prepare default result_dict
        result_dict: Dict[str, Any] = {
            'label': 'ham',
            'prob_spam': 0.0,
            'confidence': 1.0,
            'language': detected_lang,
            'explanation': [],
            'spam_words': found_spam_words
        }

        # Run ML model on cleaned text
        clean = clean_text(translated_text)
        
        # Identify which spam words are in the text for highlighting
        found_spam_words = [word for word in clean.split() if word in spam_words_list]
        
        # Prepare a default result_dict
        result_dict: Dict[str, Any] = {
            'label': 'ham',
            'prob_spam': 0.0,
            'confidence': 1.0,
            'language': detected_lang,
            'explanation': [],
            'spam_words': found_spam_words
        }
        
        prob_spam = 0.5
        if vect is not None and model is not None and clean:
            try:
                # Transform and predict
                X = vect.transform([clean])
                if hasattr(model, 'predict_proba'):
                    prob_spam = float(model.predict_proba(X)[0][1])
                else:
                    pred = model.predict(X)[0]
                    prob_spam = 0.9 if pred in [1, 'spam'] else 0.1
                
                # Update result_dict with model results
                result_dict['prob_spam'] = prob_spam
                result_dict['label'] = 'spam' if prob_spam > 0.5 else 'ham'
                result_dict['confidence'] = prob_spam if prob_spam > 0.5 else 1 - prob_spam
            except Exception as e:
                logger.error(f"Model prediction failed: {str(e)}")
        
        # Add advanced features if enabled
        advanced_spam_score = 0.0
        if use_advanced_features:
            reputation = {}
            header_features = {}
            url_analysis = {}
            header_analysis = {'status': 'failed'}
            
            try:
                # Analyze complete email headers (RFC 5322 compliant)
                header_analysis = header_analyzer.analyze_complete(text)
                
                if header_analysis.get('status') == 'success':
                    result_dict['header_analysis'] = {
                        'sender': header_analysis.get('sender'),
                        'recipient': header_analysis.get('recipient'),
                        'subject': header_analysis.get('subject'),
                        'authentication': header_analysis.get('authentication'),
                        'phishing_analysis': header_analysis.get('phishing_analysis'),
                        'overall_risk_score': header_analysis.get('overall_risk_score'),
                        'is_suspicious': header_analysis.get('is_suspicious')
                    }
                
                # Legacy header extraction
                headers = header_extractor.extract_from_text(text)
                header_features = header_extractor.extract_features(text)
                result_dict['headers'] = headers
                result_dict['header_features'] = header_features
                
                # Check sender reputation
                sender = headers.get('from', '')
                if sender:
                    reputation = reputation_checker.check_sender_reputation(sender)
                    result_dict['sender_reputation'] = reputation
                
                # Analyze URLs
                url_analysis = url_analyzer.analyze_urls(text)
                result_dict['url_analysis'] = url_analysis
                
                # Calculate advanced spam score (weighted combination)
                advanced_spam_score = _calculate_advanced_spam_score(
                    header_features,
                    reputation,
                    url_analysis,
                    header_analysis if (header_analysis and header_analysis.get('status') == 'success') else {}
                )
                result_dict['advanced_spam_score'] = advanced_spam_score
            except Exception as e:
                logger.warning(f'Error computing advanced features: {str(e)}')
                pass
        
        # Override Logic (Always evaluate heuristics)
        is_spam_override = (advanced_spam_score >= 0.32) or heuristic_spam
        
        if is_spam_override:
            logger.info(f'Override triggered (score: {advanced_spam_score:.2f}, heuristic: {heuristic_spam})')
            result_dict['label'] = 'spam'
            
            # Boost confidence
            current_prob = float(result_dict.get('prob_spam', 0.5))
            if heuristic_spam:
                result_dict['prob_spam'] = min(0.99, max(0.92, current_prob + 0.7))
            else:
                result_dict['prob_spam'] = min(0.98, max(0.80, current_prob + 0.5))
            
            result_dict['confidence'] = result_dict['prob_spam']
            
            if heuristic_reasons:
                if not isinstance(result_dict.get('explanation'), list):
                    result_dict['explanation'] = []
                result_dict['explanation'].extend(heuristic_reasons)
        
        return result_dict
            
    except ValueError as e:
        logger.error(f'Invalid input: {str(e)}')
        raise
    except Exception as e:
        logger.error(f'Prediction error: {str(e)}')
        raise RuntimeError(f'Prediction failed: {str(e)}')


def _calculate_advanced_spam_score(
    header_features: Dict[str, float],
    reputation: Dict[str, Any],
    url_analysis: Dict[str, Any],
    header_analysis: Dict[str, Any] = None
) -> float:
    """
    Calculate composite spam score from multiple feature sources.
    
    Args:
        header_features: Features extracted from email headers
        reputation: Sender reputation information
        url_analysis: URL analysis results
        header_analysis: Complete header analysis (RFC 5322 validation)
        
    Returns:
        Composite spam score (0.0-1.0)
    """
    scores = []
    
    # Header analysis (RFC 5322 validation)
    if header_analysis and header_analysis.get('status') == 'success':
        auth_score = header_analysis.get('authentication', {}).get('overall_score', 0)
        phishing_score = header_analysis.get('phishing_analysis', {}).get('phishing_score', 0)
        header_auth_score = (1.0 - auth_score) * 0.5 + phishing_score * 0.5
        scores.append(header_auth_score)
    
    # Header features
    if header_features:
        header_score = (
            header_features.get('subject_spam_score', 0) * 0.4 +
            header_features.get('sender_spam_score', 0) * 0.3 +
            header_features.get('subject_urgency_score', 0) * 0.2 +
            header_features.get('suspicious_sender_score', 0) * 0.1
        )
        scores.append(header_score)
    
    # Sender reputation
    if reputation:
        rep_score = 1.0 - reputation.get('reputation_score', 0.5)
        scores.append(rep_score)
    
    # URL analysis (Heavily weighted)
    if url_analysis:
        url_score = 0.0
        if url_analysis.get('has_shortened_url'): url_score += 0.4
        if url_analysis.get('has_ip_url'): url_score += 0.5
        if url_analysis.get('has_suspicious_tld'): url_score += 0.4
        if url_analysis.get('has_many_urls'): url_score += 0.3
        if url_analysis.get('has_long_url'): url_score += 0.2
            
        susp_count = url_analysis.get('suspicious_url_count', 0)
        if susp_count > 0:
            url_score += min(susp_count * 0.15, 0.6)
            
        scores.append(min(url_score, 1.0))
    
    if not scores:
        return 0.0
        
    # Logic: Prioritize the HIGHEST risk signal found rather than averaging
    # This prevents 'safe' signals from diluting a single very dangerous signal
    max_signal = max(scores)
    avg_signal = sum(scores) / len(scores)
    
    # Use 70% of the strongest signal + 30% of the average
    final_score = (0.7 * max_signal) + (0.3 * avg_signal)
    return min(final_score, 1.0)


def predict_with_explanation(text: str) -> Dict[str, Any]:
    """
    Predict and provide detailed explanation for the prediction.
    
    Args:
        text (str): Email text to classify
        
    Returns:
        Dict with prediction and detailed explanation
    """
    try:
        result = predict(text, use_advanced_features=True)
        
        explanation = {
            'prediction': result.get('label'),
            'confidence': result.get('confidence', result.get('prob_spam', 0)),
            'reasoning': [],
            'detected_signals': []
        }
        
        # Add reasoning based on spam words
        if result.get('spam_words'):
            explanation['reasoning'].append(
                f"Detected {len(result['spam_words'])} known spam indicator words"
            )
            explanation['detected_signals'].extend(result['spam_words'])
        
        # Add reasoning based on header analysis
        header_analysis = result.get('header_analysis', {})
        if header_analysis:
            overall_risk = header_analysis.get('overall_risk_score', 0)
            if overall_risk > 0.6:
                explanation['reasoning'].append(
                    f"Email header analysis shows high risk (score: {overall_risk:.2f})"
                )
            
            auth_results = header_analysis.get('authentication', {})
            if auth_results:
                if not auth_results.get('authenticated', False):
                    explanation['reasoning'].append(
                        "Email failed authentication checks (SPF/DKIM/DMARC)"
                    )
                    explanation['detected_signals'].append('Failed authentication')
            
            phishing = header_analysis.get('phishing_analysis', {})
            if phishing:
                phishing_score = phishing.get('phishing_score', 0)
                if phishing_score > 0.6:
                    explanation['reasoning'].append(
                        f"Phishing indicators detected (score: {phishing_score:.2f})"
                    )
                    if phishing.get('indicators'):
                        explanation['detected_signals'].extend(phishing['indicators'][:3])
        
        # Add reasoning based on header features
        header_features = result.get('header_features', {})
        if header_features.get('subject_spam_score', 0) > 0.5:
            explanation['reasoning'].append(
                "Subject line contains suspicious keywords"
            )
        
        if header_features.get('subject_urgency_score', 0) > 0.6:
            explanation['reasoning'].append(
                "Subject line has high urgency indicators"
            )
        
        # Add reasoning based on sender
        reputation = result.get('sender_reputation', {})
        risk_level = reputation.get('risk_level', '')
        if risk_level == 'high':
            explanation['reasoning'].append(
                f"Sender has high spam risk: {reputation.get('reason', 'Unknown')}"
            )
        
        # Add reasoning based on URLs
        url_analysis = result.get('url_analysis', {})
        if url_analysis.get('has_shortened_url'):
            explanation['reasoning'].append("Email contains shortened URLs")
        if url_analysis.get('has_ip_url'):
            explanation['reasoning'].append("Email contains IP-based URLs")
        
        explanation['advanced_spam_score'] = result.get('advanced_spam_score')
        
        return explanation
        
    except Exception as e:
        logger.error(f'Error in predict_with_explanation: {str(e)}')
        raise


if __name__ == '__main__':
    import sys
    try:
        txt = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else input('Enter text to classify:\n')
        
        # Show prediction with explanation
        if '-explain' in sys.argv:
            result = predict_with_explanation(txt)
            print(json.dumps(result, indent=2))
        else:
            result = predict(txt)
            print(json.dumps(result, indent=2))
    except Exception as e:
        print(f'Error: {str(e)}', file=sys.stderr)
        sys.exit(1)


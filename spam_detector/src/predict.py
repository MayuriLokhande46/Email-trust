import os
import joblib
import json
import logging
from typing import Dict, Any
from preprocess import clean_text
from features import load_vectorizer
from feature_engineering import (
    EmailHeaderExtractor, SenderReputationChecker, EmailUrlExtractor
)
from email_headers import CompleteHeaderAnalyzer

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
    
    Args:
        text (str): The email text to classify
        use_advanced_features (bool): Whether to use advanced feature engineering
        
    Returns:
        Dict with keys: label, prob_spam (if available), spam_words, 
                       header_features (if enabled), urls (if enabled)
        
    Raises:
        ValueError: If text is empty or invalid
        RuntimeError: If model is not loaded
    """
    try:
        if not isinstance(text, str):
            raise ValueError('Input must be a string')
        
        if not text or text.isspace():
            raise ValueError('Input text cannot be empty')
        
        if model is None or vect is None:
            raise RuntimeError('Model or vectorizer not loaded. Check initialization logs.')
        
        # Clean and preprocess text
        clean = clean_text(text)
        
        if not clean or clean.isspace():
            logger.warning('Text became empty after cleaning')
            return {
                'label': 'ham',
                'prob_spam': 0.0,
                'spam_words': [],
                'warning': 'Text became empty after preprocessing'
            }
        
        # Find which spam words are in the text
        found_spam_words = [word for word in clean.split() if word in spam_words_list]
        
        # Vectorize and predict
        X = vect.transform([clean])
        
        result_dict = {}
        
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(X)[0][1]
            label = 'spam' if prob >= 0.5 else 'ham'
            result_dict = {
                'label': label,
                'prob_spam': float(prob),
                'spam_words': found_spam_words,
                'confidence': float(prob) if label == 'spam' else float(1 - prob)
            }
        else:
            prediction = int(model.predict(X)[0])
            label = 'spam' if prediction == 1 else 'ham'
            result_dict = {
                'label': label,
                'spam_words': found_spam_words
            }
        
        # Add advanced features if enabled
        if use_advanced_features:
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
                
                # Extract and analyze headers (legacy method)
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
                
                # Compute advanced spam score (weighted combination)
                advanced_spam_score = _calculate_advanced_spam_score(
                    header_features,
                    reputation,
                    url_analysis,
                    header_analysis if header_analysis.get('status') == 'success' else None
                )
                result_dict['advanced_spam_score'] = advanced_spam_score
                
                # If advanced features suggest high spam, adjust prediction
                if use_advanced_features and advanced_spam_score > 0.7:
                    if result_dict.get('prob_spam', 0.5) < 0.7:
                        logger.info(f'Advanced features boosted spam prediction')
                        result_dict['label'] = 'spam'
                        result_dict['prob_spam'] = min(0.75, result_dict.get('prob_spam', 0.5) + 0.2)
                        
            except Exception as e:
                logger.warning(f'Error computing advanced features: {str(e)}')
                # Continue prediction without advanced features
                pass
        
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
    score = 0.0
    weight_sum = 0.0
    
    # Header analysis contribution (25% weight) - RFC 5322 validation
    if header_analysis and header_analysis.get('status') == 'success':
        auth_score = header_analysis.get('authentication', {}).get('overall_score', 0)
        phishing_score = header_analysis.get('phishing_analysis', {}).get('phishing_score', 0)
        # High phishing/auth failure = high spam score
        header_auth_score = (1.0 - auth_score) * 0.5 + phishing_score * 0.5
        score += header_auth_score * 0.25
        weight_sum += 0.25
    
    # Header features contribution (30% weight)
    if header_features:
        header_score = (
            header_features.get('subject_spam_score', 0) * 0.4 +
            header_features.get('sender_spam_score', 0) * 0.3 +
            header_features.get('subject_urgency_score', 0) * 0.2 +
            header_features.get('suspicious_sender_score', 0) * 0.1
        )
        score += header_score * 0.3
        weight_sum += 0.3
    
    # Sender reputation contribution (25% weight)
    if reputation:
        rep_score = 1.0 - reputation.get('reputation_score', 0.5)
        score += rep_score * 0.25
        weight_sum += 0.25
    
    # URL analysis contribution (20% weight)
    if url_analysis:
        url_score = 0.0
        if url_analysis.get('has_shortened_url'):
            url_score += 0.3
        if url_analysis.get('has_ip_url'):
            url_score += 0.4
        if url_analysis.get('suspicious_url_count', 0) > 0:
            url_score += 0.2
        score += url_score * 0.2
        weight_sum += 0.2
    
    # Normalize score
    if weight_sum > 0:
        return min(score / weight_sum, 1.0)
    return 0.0


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


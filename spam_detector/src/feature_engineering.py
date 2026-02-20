"""
Advanced feature engineering for email spam detection.

Extracts additional signals from email content such as:
- Email headers (Subject, From, To)
- Sender reputation indicators
- Email structure features
- Content-based signals
"""

import re
import logging
from typing import Dict, List, Tuple, Any
from collections import Counter

logger = logging.getLogger(__name__)


class EmailHeaderExtractor:
    """Extract and analyze email headers."""

    # Common spam subject keywords
    SPAM_SUBJECT_KEYWORDS = {
        'urgent', 'act', 'now', 'free', 'money', 'click', 'congratulations',
        'winner', 'prize', 'claim', 'verify', 'confirm', 'limited', 'offer',
        'expire', 'expired', 'exclusive', 'alert', 'urgent action', 'important',
        'update', 're-activate', 're-confirm', 'guaranteed', 'risk free',
        'loan', 'debt', 'cash', 'reward', 'gift', 'bonus', 'crypto', 'bitcoin',
        'investment', 'profit', 'bank', 'account', 'security', 'suspended'
    }

    # Common spam sender patterns
    SPAM_SENDER_PATTERNS = [
        r'.*@[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+',  # IP-based addresses
        r'.*noreply.*',  # noreply addresses
        r'.*no-reply.*',
        r'.*automated.*',
        r'.*support.*@.*\.ru$',  # Russian domains often used for spam
        r'.*support.*@.*\.cn$',  # Chinese domains
    ]

    def __init__(self):
        """Initialize header extractor."""
        self.spam_sender_patterns = [re.compile(pattern, re.IGNORECASE) 
                                     for pattern in self.SPAM_SENDER_PATTERNS]

    def extract_from_text(self, email_text: str) -> Dict[str, Any]:
        """
        Extract headers from email text (if present in plain text format).

        Args:
            email_text (str): Email content

        Returns:
            Dict with extracted header information
        """
        try:
            # First try standard parser patterns
            headers = {
                'subject': self._extract_subject(email_text),
                'from': self._extract_from(email_text),
                'to': self._extract_to(email_text),
                'reply_to': self._extract_reply_to(email_text),
                'date': self._extract_date(email_text),
            }
            
            # If all are empty, the email might be just the body or improperly formatted
            # Attempt a more aggressive global search
            if not any(headers.values()):
                logger.debug("Standard header extraction failed, attempting fallback regex")
                headers['subject'] = self._fallback_extract(email_text, r'(?:subject|re|fwd):\s*(.+)')
                headers['from'] = self._fallback_extract(email_text, r'(?:from|sender|sent by):\s*(.+)')
                
            return headers
        except Exception as e:
            logger.warning(f'Error extracting headers: {str(e)}')
            return {}

    def _fallback_extract(self, text: str, pattern: str) -> str:
        """Global search fallback for headers embedded in body."""
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _extract_subject(self, text: str) -> str:
        """Extract Subject field."""
        match = re.search(r'Subject:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _extract_from(self, text: str) -> str:
        """Extract From field."""
        match = re.search(r'From:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        if match:
            email = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
                            match.group(1))
            return email.group(0) if email else match.group(1).strip()
        return ""

    def _extract_to(self, text: str) -> str:
        """Extract To field."""
        match = re.search(r'To:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _extract_reply_to(self, text: str) -> str:
        """Extract Reply-To field."""
        match = re.search(r'Reply-To:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _extract_date(self, text: str) -> str:
        """Extract Date field."""
        match = re.search(r'Date:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def extract_features(self, email_text: str) -> Dict[str, float]:
        """
        Extract spam probability features from email headers.

        Args:
            email_text (str): Email content

        Returns:
            Dict with feature scores (0.0-1.0)
        """
        headers = self.extract_from_text(email_text)
        features = {}

        # Subject line analysis
        features['subject_spam_score'] = self._analyze_subject(headers.get('subject', ''))
        features['subject_urgency_score'] = self._calculate_urgency(headers.get('subject', ''))
        features['subject_caps_score'] = self._calculate_caps_ratio(headers.get('subject', ''))

        # Sender analysis
        features['sender_spam_score'] = self._analyze_sender(headers.get('from', ''))
        features['suspicious_sender_score'] = self._check_suspicious_sender(headers.get('from', ''))

        # Reply-To analysis
        features['reply_to_mismatch_score'] = self._check_reply_to_mismatch(
            headers.get('from', ''),
            headers.get('reply_to', '')
        )

        return features

    def _analyze_subject(self, subject: str) -> float:
        """
        Analyze subject line for spam keywords.

        Returns:
            Score between 0 and 1
        """
        if not subject:
            return 0.1

        subject_lower = subject.lower()
        spam_keyword_count = sum(1 for keyword in self.SPAM_SUBJECT_KEYWORDS 
                                if keyword in subject_lower)

        # Normalize score
        max_keywords = 3
        score = min(spam_keyword_count / max_keywords, 1.0)

        return score

    def _calculate_urgency(self, subject: str) -> float:
        """
        Calculate urgency indicators in subject.

        Urgency indicators:
        - Exclamation marks
        - "URGENT", "ACTION REQUIRED", etc.
        - Multiple capitals
        """
        if not subject:
            return 0.0

        urgency_score = 0.0

        # Count exclamation marks (max 2)
        urgency_score += min(subject.count('!') / 2, 0.3)

        # Check for urgency keywords
        urgency_keywords = ['urgent', 'action required', 'immediate', 'asap']
        if any(keyword in subject.lower() for keyword in urgency_keywords):
            urgency_score += 0.4

        # Check for multiple capitals
        caps_count = sum(1 for c in subject if c.isupper())
        if len(subject) > 0 and caps_count / len(subject) > 0.5:
            urgency_score += 0.3

        return min(urgency_score, 1.0)

    def _calculate_caps_ratio(self, text: str) -> float:
        """
        Calculate ratio of capital letters.

        High caps ratio often indicates SPAM.
        """
        if not text:
            return 0.0

        letters = [c for c in text if c.isalpha()]
        if not letters:
            return 0.0

        caps_count = sum(1 for c in letters if c.isupper())
        ratio = caps_count / len(letters)

        # Score: >50% caps = high spam score
        return min(ratio, 1.0)

    def _analyze_sender(self, sender: str) -> float:
        """
        Analyze sender email address for spam signals.

        Spam indicators:
        - Generic names (no-reply, support, etc.)
        - IP-based addresses
        - Suspicious domains
        """
        if not sender:
            return 0.5  # Unknown sender

        sender_lower = sender.lower()
        spam_score = 0.0

        # Check for generic/automated sender names
        generic_names = ['noreply', 'no-reply', 'automated', 'support', 'admin']
        if any(name in sender_lower for name in generic_names):
            spam_score += 0.4

        # Check for IP-based addresses
        if re.search(r'@\d+\.\d+\.\d+\.\d+', sender):
            spam_score += 0.5

        # Check for suspicious patterns
        for pattern in self.spam_sender_patterns:
            if pattern.match(sender):
                spam_score += 0.3
                break

        return min(spam_score, 1.0)

    def _check_suspicious_sender(self, sender: str) -> float:
        """
        Check for suspicious sender characteristics.

        Returns:
            Suspicion score 0-1
        """
        if not sender:
            return 0.0

        suspicion_score = 0.0

        # Check if sender name doesn't match domain
        match = re.search(r'(.+?)@(.+)', sender)
        if match:
            name_part, domain = match.groups()

            # Check if sender name is random/obfuscated
            if len(name_part) > 20 or ('.' in name_part and name_part.count('.') > 2):
                suspicion_score += 0.3

            # Check for suspicious domains
            suspicious_extensions = ['.ru', '.cn', '.kr', '.ua', '.pk']
            if any(domain.endswith(ext) for ext in suspicious_extensions):
                suspicion_score += 0.2

        return min(suspicion_score, 1.0)

    def _check_reply_to_mismatch(self, from_addr: str, reply_to: str) -> float:
        """
        Check if Reply-To doesn't match From address.

        This is a common phishing/spam tactic.
        """
        if not reply_to:
            return 0.0

        if not from_addr:
            return 0.5  # Can't verify

        # Extract domains
        from_domain = re.search(r'@(.+)$', from_addr)
        reply_domain = re.search(r'@(.+)$', reply_to)

        if from_domain and reply_domain:
            if from_domain.group(1).lower() != reply_domain.group(1).lower():
                return 0.8  # High suspicion

        return 0.0


class SenderReputationChecker:
    """Check sender reputation (basic implementation)."""

    # Known spam sender domains (in production, use external API)
    KNOWN_SPAM_DOMAINS = {
        '.tk', '.ml', '.ga', '.cf',  # Free domains
        'tempmail.', 'throwaway',  # Temporary mail services
        'spam', 'fake', 'test'  # Obvious spam indicators
    }

    # Known legitimate domains
    LEGITIMATE_DOMAINS = {
        'gmail.com', 'outlook.com', 'yahoo.com', 'hotmail.com',
        'linkedin.com', 'twitter.com', 'facebook.com', 'apple.com',
        'microsoft.com', 'google.com', 'amazon.com'
    }

    def check_sender_reputation(self, sender_email: str) -> Dict[str, Any]:
        """
        Check sender reputation.

        Args:
            sender_email (str): Email address to check

        Returns:
            Dict with reputation information
        """
        try:
            domain = re.search(r'@(.+)$', sender_email)
            if not domain:
                return {'reputation_score': 0.5, 'reason': 'Invalid email'}

            domain_name = domain.group(1).lower()

            # Check if domain is in legitimate list
            if domain_name in self.LEGITIMATE_DOMAINS:
                return {
                    'reputation_score': 0.95,
                    'reason': 'Known legitimate domain',
                    'risk_level': 'low'
                }

            # Check for spam indicators
            for spam_indicator in self.KNOWN_SPAM_DOMAINS:
                if spam_indicator in domain_name:
                    return {
                        'reputation_score': 0.2,
                        'reason': f'Matches spam indicator: {spam_indicator}',
                        'risk_level': 'high'
                    }

            # Default: unknown domain
            return {
                'reputation_score': 0.5,
                'reason': 'Unknown domain',
                'risk_level': 'medium'
            }

        except Exception as e:
            logger.warning(f'Error checking sender reputation: {str(e)}')
            return {'reputation_score': 0.5, 'reason': 'Error checking reputation'}


class EmailUrlExtractor:
    """Extract and analyze URLs in email content."""

    def extract_urls(self, text: str) -> List[str]:
        """
        Extract all URLs from email text.

        Args:
            text (str): Email content

        Returns:
            List of URLs found
        """
        url_pattern = r'https?://[^\s\)>\]"]+'
        urls = re.findall(url_pattern, text, re.IGNORECASE)
        return urls

    def analyze_urls(self, text: str) -> Dict[str, Any]:
        """
        Analyze URLs in email for spam signals.

        Spam indicators:
        - Shortened URLs
        - Suspicious TLDs
        - Multiple URLs
        - IP-based URLs
        - Long URLs
        - Suspicious keywords
        """
        urls = self.extract_urls(text)
        analysis = {
            'url_count': len(urls),
            'has_shortened_url': False,
            'has_ip_url': False,
            'has_suspicious_tld': False,
            'has_many_urls': len(urls) > 3,
            'has_long_url': False,
            'suspicious_url_count': 0,
            'urls': urls
        }

        suspicious_count = 0
        has_shortened = False
        has_ip = False
        has_susp_tld = False
        has_long = False

        shortened_domains = ['bit.ly', 'tinyurl', 'short.link', 'goo.gl', 't.co', 'ow.ly']
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.pw', '.win', '.bid', '.loan']
        
        for url in urls:
            url_lower = url.lower()
            
            if any(domain in url_lower for domain in shortened_domains):
                has_shortened = True
                suspicious_count += 1
            
            if re.search(r'https?://\d+\.\d+\.\d+\.\d+', url):
                has_ip = True
                suspicious_count += 1
            
            if any(tld in url_lower for tld in suspicious_tlds):
                has_susp_tld = True
                suspicious_count += 1

            if len(url) > 100:
                has_long = True
                suspicious_count += 1
            
            spam_keywords = ['verify', 'confirm', 'update', 'claim', 'login', 'secure', 'account', 'banking', 'prize', 'gift']
            if any(keyword in url_lower for keyword in spam_keywords):
                suspicious_count += 1

        analysis = {
            'url_count': len(urls),
            'has_shortened_url': has_shortened,
            'has_ip_url': has_ip,
            'has_suspicious_tld': has_susp_tld,
            'has_many_urls': len(urls) > 3,
            'has_long_url': has_long,
            'suspicious_url_count': suspicious_count,
            'urls': urls
        }
        return analysis


if __name__ == '__main__':
    # Test examples
    extractor = EmailHeaderExtractor()
    reputation = SenderReputationChecker()
    url_analyzer = EmailUrlExtractor()

    # Test email
    test_email = """
    Subject: URGENT!!! Claim Your FREE $1000 NOW!!!
    From: noreply@suspicious.tk
    To: user@gmail.com
    
    CONGRATULATIONS! You have been selected to receive $1000!
    Click here NOW: http://bit.ly/scam123
    """

    print("=" * 50)
    print("Feature Extraction Test")
    print("=" * 50)

    # Extract headers
    headers = extractor.extract_from_text(test_email)
    print(f"\nHeaders: {headers}")

    # Extract features
    features = extractor.extract_features(test_email)
    print(f"\nHeader Features: {features}")

    # Check sender reputation
    sender = headers.get('from', '')
    rep = reputation.check_sender_reputation(sender)
    print(f"\nSender Reputation: {rep}")

    # Analyze URLs
    url_analysis = url_analyzer.analyze_urls(test_email)
    print(f"\nURL Analysis: {url_analysis}")

    print("\n" + "=" * 50)
    print("All features indicate SPAM")
    print("=" * 50)

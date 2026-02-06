"""
Advanced Email Header Parsing and Analysis.

Extracts and validates:
- Email headers (From, To, Subject, Date, etc.)
- SPF/DKIM/DMARC headers
- Email authentication verification
- Header metadata analysis
- Phishing indicators
"""

import logging
import re
import email
from typing import Dict, Any, List, Optional, Tuple
from email.parser import HeaderParser
from datetime import datetime

logger = logging.getLogger(__name__)


class EmailHeaderParser:
    """Parse and extract email headers."""

    # Standard email headers
    STANDARD_HEADERS = {
        'From', 'To', 'Subject', 'Date', 'Reply-To',
        'Cc', 'Bcc', 'Message-ID', 'Content-Type',
        'Content-Transfer-Encoding', 'MIME-Version',
        'Received', 'Return-Path', 'Sender'
    }

    # Authentication headers
    AUTH_HEADERS = {
        'Authentication-Results',
        'DKIM-Signature',
        'ARC-Authentication-Results',
        'SPF-Record',
    }

    def __init__(self):
        """Initialize header parser."""
        self.headers = {}
        self.raw_headers = {}

    def parse_headers(self, email_text: str) -> Dict[str, Any]:
        """
        Parse email headers from raw email text.

        Args:
            email_text (str): Raw email text with headers

        Returns:
            Dict with parsed headers
        """
        try:
            # Split headers from body (headers end with blank line)
            parts = email_text.split('\n\n', 1)
            header_section = parts[0] if parts else email_text

            # Parse with email library
            parser = HeaderParser()
            message = parser.parsestr(header_section)

            # Extract standard headers
            self.headers = {}
            for header_name in self.STANDARD_HEADERS:
                value = message.get(header_name, '')
                if value:
                    self.headers[header_name] = value.strip()

            # Extract custom/authentication headers
            for key, value in message.items():
                if key not in self.STANDARD_HEADERS:
                    self.headers[key] = value.strip()

            self.raw_headers = dict(message.items())
            logger.info(f'Extracted {len(self.headers)} headers')

            return self.headers

        except Exception as e:
            logger.warning(f'Error parsing headers: {str(e)}')
            return {}

    def extract_email_address(self, email_field: str) -> Optional[str]:
        """
        Extract email address from header field.

        Handles formats like:
        - user@example.com
        - "User Name" <user@example.com>
        - User Name <user@example.com>

        Args:
            email_field (str): Email header field value

        Returns:
            Extract email address or None
        """
        if not email_field:
            return None

        # Try regex pattern
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        match = re.search(email_pattern, email_field)

        return match.group(0) if match else None

    def extract_domain(self, email_address: str) -> Optional[str]:
        """Extract domain from email address."""
        if not email_address or '@' not in email_address:
            return None

        return email_address.split('@')[1].lower()

    def get_all_headers(self) -> Dict[str, str]:
        """Get all parsed headers."""
        return self.headers.copy()

    def get_header(self, header_name: str) -> Optional[str]:
        """Get specific header value."""
        return self.headers.get(header_name)


class EmailAuthenticationValidator:
    """Validate email authentication headers (SPF, DKIM, DMARC)."""

    # Known legitimate domains for verification
    TRUSTED_DOMAINS = {
        'gmail.com', 'outlook.com', 'yahoo.com', 'hotmail.com',
        'microsoft.com', 'google.com', 'apple.com', 'amazon.com'
    }

    def __init__(self):
        """Initialize validator."""
        self.dkim_valid = False
        self.spf_valid = False
        self.dmarc_valid = False

    def validate_dkim(self, dkim_signature: str) -> Dict[str, Any]:
        """
        Validate DKIM signature header.

        Args:
            dkim_signature (str): DKIM-Signature header value

        Returns:
            Dict with validation results
        """
        try:
            if not dkim_signature:
                return {
                    'valid': False,
                    'reason': 'No DKIM signature found'
                }

            # Check for required DKIM fields
            required_fields = ['v=1', 'd=', 'a=']
            has_required = all(field in dkim_signature for field in required_fields)

            # Extract domain
            domain_match = re.search(r'd=([^;]+)', dkim_signature)
            domain = domain_match.group(1).strip() if domain_match else None

            result = {
                'valid': has_required,
                'complete': True,
                'domain': domain,
                'algorithm': self._extract_dkim_algo(dkim_signature),
                'reason': 'Valid DKIM signature' if has_required else 'Invalid DKIM format'
            }

            self.dkim_valid = has_required
            return result

        except Exception as e:
            logger.error(f'DKIM validation error: {str(e)}')
            return {
                'valid': False,
                'reason': str(e)
            }

    def _extract_dkim_algo(self, dkim_signature: str) -> Optional[str]:
        """Extract DKIM algorithm."""
        match = re.search(r'a=([^;]+)', dkim_signature)
        return match.group(1).strip() if match else None

    def validate_spf(self, headers: Dict[str, str]) -> Dict[str, Any]:
        """
        Validate SPF record (basic check).

        SPF helps prevent email spoofing by specifying which IPs
        are authorized to send emails for a domain.

        Args:
            headers (Dict): Email headers

        Returns:
            Dict with SPF validation
        """
        try:
            received = headers.get('Received', '')
            auth_results = headers.get('Authentication-Results', '')

            # Check for SPF pass
            spf_pass = False
            spf_reason = 'No SPF information'

            if 'spf=pass' in auth_results.lower():
                spf_pass = True
                spf_reason = 'SPF validation passed'
            elif 'spf=fail' in auth_results.lower():
                spf_reason = 'SPF validation failed (spoofed)'
            elif 'spf=' in auth_results.lower():
                spf_match = re.search(r'spf=(\w+)', auth_results.lower())
                if spf_match:
                    spf_reason = f'SPF status: {spf_match.group(1)}'

            result = {
                'valid': spf_pass,
                'reason': spf_reason,
                'suspicious': 'spf=fail' in auth_results.lower() or 'spf=' not in auth_results.lower()
            }

            self.spf_valid = spf_pass
            return result

        except Exception as e:
            logger.error(f'SPF validation error: {str(e)}')
            return {
                'valid': False,
                'reason': str(e),
                'suspicious': True
            }

    def validate_dmarc(self, headers: Dict[str, str]) -> Dict[str, Any]:
        """
        Validate DMARC record (basic check).

        DMARC (Domain-based Message Authentication, Reporting & Conformance)
        combines SPF and DKIM for authentication.

        Args:
            headers (Dict): Email headers

        Returns:
            Dict with DMARC validation
        """
        try:
            auth_results = headers.get('Authentication-Results', '')
            dmarc_pass = False
            dmarc_reason = 'No DMARC information'

            if 'dmarc=pass' in auth_results.lower():
                dmarc_pass = True
                dmarc_reason = 'DMARC validation passed'
            elif 'dmarc=fail' in auth_results.lower():
                dmarc_reason = 'DMARC validation failed'
            elif 'dmarc=' in auth_results.lower():
                dmarc_match = re.search(r'dmarc=(\w+)', auth_results.lower())
                if dmarc_match:
                    dmarc_reason = f'DMARC status: {dmarc_match.group(1)}'

            result = {
                'valid': dmarc_pass,
                'reason': dmarc_reason,
                'suspicious': 'dmarc=fail' in auth_results.lower() or 'dmarc=' not in auth_results.lower()
            }

            self.dmarc_valid = dmarc_pass
            return result

        except Exception as e:
            logger.error(f'DMARC validation error: {str(e)}')
            return {
                'valid': False,
                'reason': str(e),
                'suspicious': True
            }

    def validate_all(self, headers: Dict[str, str]) -> Dict[str, Any]:
        """Validate all authentication methods."""
        dkim_sig = headers.get('DKIM-Signature', '')
        dkim_result = self.validate_dkim(dkim_sig)
        spf_result = self.validate_spf(headers)
        dmarc_result = self.validate_dmarc(headers)

        # Calculate overall authentication score
        auth_pass_count = sum([
            dkim_result.get('valid', False),
            spf_result.get('valid', False),
            dmarc_result.get('valid', False)
        ])

        return {
            'dkim': dkim_result,
            'spf': spf_result,
            'dmarc': dmarc_result,
            'overall_score': auth_pass_count / 3,  # 0-1 score
            'authenticated': auth_pass_count >= 2  # Pass if 2+ methods valid
        }


class HeaderPhishingDetector:
    """Detect phishing indicators in email headers."""

    PHISHING_INDICATORS = [
        'urgent',
        'verify',
        'confirm',
        'update account',
        'click here',
        'act now',
        'authenticate',
        'reactivate',
        'suspended',
        'limited time',
        'immediate action'
    ]

    def __init__(self):
        """Initialize detector."""
        self.phishing_score = 0.0

    def analyze_headers(self, headers: Dict[str, str]) -> Dict[str, Any]:
        """
        Analyze headers for phishing indicators.

        Args:
            headers (Dict): Email headers

        Returns:
            Dict with phishing analysis
        """
        try:
            indicators = []
            score = 0.0

            # 1. Check Subject line
            subject = headers.get('Subject', '').lower()
            for indicator in self.PHISHING_INDICATORS:
                if indicator in subject:
                    indicators.append(f'Phishing keyword in subject: "{indicator}"')
                    score += 0.15

            # 2. Check From-To mismatch
            from_email = headers.get('From', '')
            from_domain = self._extract_domain(from_email)
            reply_to = headers.get('Reply-To', '')
            reply_domain = self._extract_domain(reply_to)

            if from_domain and reply_domain and from_domain != reply_domain:
                indicators.append(f'Mismatch: From domain ({from_domain}) != Reply-To ({reply_domain})')
                score += 0.25

            # 3. Check for suspicious sender
            if self._is_suspicious_sender(from_email):
                indicators.append('Suspicious sender pattern detected')
                score += 0.20

            # 4. Check for generic greetings
            generic_greetings = ['dear user', 'dear customer', 'valued customer']
            if any(greeting in subject.lower() for greeting in generic_greetings):
                indicators.append('Generic greeting detected (phishing indicator)')
                score += 0.15

            # 5. Check for urgency markers in subject
            urgency_words = ['urgent', 'immediate', 'action required', 'asap']
            urgency_count = sum(1 for word in urgency_words if word in subject)
            if urgency_count > 0:
                indicators.append(f'High urgency markers in subject ({urgency_count})')
                score += 0.15

            self.phishing_score = min(score, 1.0)

            return {
                'phishing_score': self.phishing_score,
                'is_phishing': self.phishing_score > 0.6,
                'risk_level': self._calculate_risk_level(self.phishing_score),
                'indicators': indicators
            }

        except Exception as e:
            logger.error(f'Phishing detection error: {str(e)}')
            return {
                'phishing_score': 0.0,
                'error': str(e)
            }

    def _extract_domain(self, email_field: str) -> Optional[str]:
        """Extract domain from email field."""
        match = re.search(r'@([a-zA-Z0-9.-]+)', email_field)
        return match.group(1).lower() if match else None

    def _is_suspicious_sender(self, from_email: str) -> bool:
        """Check if sender looks suspicious."""
        suspicious_patterns = [
            r'.*@\d+\.\d+\.\d+\.\d+',  # IP-based
            r'.*noreply.*',             # auto-generated
            r'.*no-reply.*',
            r'.*support.*@.*\.tk',      # free domain
            r'.*admin.*@.*\.ml',
        ]

        return any(
            re.match(pattern, from_email, re.IGNORECASE)
            for pattern in suspicious_patterns
        )

    def _calculate_risk_level(self, score: float) -> str:
        """Calculate risk level from score."""
        if score < 0.3:
            return 'low'
        elif score < 0.6:
            return 'medium'
        else:
            return 'high'


class CompleteHeaderAnalyzer:
    """Complete email header analysis."""

    def __init__(self):
        """Initialize analyzer."""
        self.parser = EmailHeaderParser()
        self.auth_validator = EmailAuthenticationValidator()
        self.phishing_detector = HeaderPhishingDetector()

    def analyze_complete(self, email_text: str) -> Dict[str, Any]:
        """
        Complete analysis of email headers.

        Args:
            email_text (str): Raw email text

        Returns:
            Comprehensive header analysis
        """
        try:
            # Parse headers
            headers = self.parser.parse_headers(email_text)

            if not headers:
                return {
                    'status': 'error',
                    'message': 'Could not parse headers'
                }

            # Extract key information
            from_email = self.parser.extract_email_address(headers.get('From', ''))
            to_email = self.parser.extract_email_address(headers.get('To', ''))
            from_domain = self.parser.extract_domain(from_email) if from_email else None

            # Validate authentication
            auth_results = self.auth_validator.validate_all(headers)

            # Detect phishing
            phishing_results = self.phishing_detector.analyze_headers(headers)

            # Calculate overall risk score
            overall_risk = self._calculate_overall_risk(auth_results, phishing_results)

            return {
                'status': 'success',
                'headers_parsed': len(headers),
                'sender': {
                    'email': from_email,
                    'domain': from_domain
                },
                'recipient': {
                    'email': to_email
                },
                'subject': headers.get('Subject', ''),
                'date': headers.get('Date', ''),
                'authentication': auth_results,
                'phishing_analysis': phishing_results,
                'overall_risk_score': overall_risk,
                'is_suspicious': overall_risk > 0.6,
                'headers': headers
            }

        except Exception as e:
            logger.error(f'Complete analysis error: {str(e)}')
            return {
                'status': 'error',
                'message': str(e)
            }

    def _calculate_overall_risk(self, auth_results: Dict, phishing_results: Dict) -> float:
        """Calculate overall risk score from multiple indicators."""
        auth_score = 1.0 - auth_results.get('overall_score', 0)  # No auth = high risk
        phishing_score = phishing_results.get('phishing_score', 0)

        # Weighted average
        overall = (auth_score * 0.4) + (phishing_score * 0.6)
        return min(overall, 1.0)


if __name__ == '__main__':
    # Test email header analysis
    test_email = """From: support@bank-example.tk <fake.support@suspicious.ru>
To: user@gmail.com
Subject: URGENT!!! Verify Your Account NOW!!!
Date: Wed, 6 Feb 2024 10:30:00 +0000
Reply-To: verify@different-domain.com
Authentication-Results: dmarc=fail spf=fail dkim=fail
DKIM-Signature: v=1; a=rsa-sha256; d=bank-example.tk; s=default

Dear Valued Customer,

URGENT ACTION REQUIRED!
Your account has been suspended. Click here to verify your account immediately.
"""

    analyzer = CompleteHeaderAnalyzer()
    result = analyzer.analyze_complete(test_email)

    print("=" * 60)
    print("Email Header Analysis Report")
    print("=" * 60)
    print(f"\nSender: {result['sender']['email']}")
    print(f"Domain: {result['sender']['domain']}")
    print(f"Subject: {result['subject']}")
    print(f"\nAuthentication Results:")
    print(f"  DKIM: {result['authentication']['dkim']['reason']}")
    print(f"  SPF: {result['authentication']['spf']['reason']}")
    print(f"  DMARC: {result['authentication']['dmarc']['reason']}")
    print(f"\nPhishing Analysis:")
    print(f"  Risk Score: {result['phishing_analysis']['phishing_score']:.2f}")
    print(f"  Risk Level: {result['phishing_analysis']['risk_level']}")
    print(f"  Indicators: {len(result['phishing_analysis']['indicators'])}")
    for indicator in result['phishing_analysis']['indicators']:
        print(f"    - {indicator}")
    print(f"\nOverall Risk Score: {result['overall_risk_score']:.2f}")
    print(f"Is Suspicious: {result['is_suspicious']}")
    print("=" * 60)

# Email Header Parsing & Analysis Guide

## Overview

The Spam Detector now includes advanced email header parsing and analysis capabilities that detect phishing, authentication failures, and other header-based spam indicators. This document explains the feature in detail.

---

## Table of Contents

1. [What is Email Header Analysis?](#what-is-email-header-analysis)
2. [Key Components](#key-components)
3. [Authentication Validation](#authentication-validation)
4. [Phishing Detection](#phishing-detection)
5. [Integration with Predictions](#integration-with-predictions)
6. [Usage Examples](#usage-examples)
7. [Troubleshooting](#troubleshooting)

---

## What is Email Header Analysis?

Email headers contain metadata about the message including:
- **Envelope Information:** From, To, Subject, Date
- **Routing Information:** Received headers showing email path
- **Authentication Records:** SPF, DKIM, DMARC validation
- **Content Information:** MIME type, encoding

Advanced header analysis detects:
- **Authentication Failures:** SPF fails, DKIM missing, DMARC problems
- **Phishing Indicators:** Urgent language, generic greetings, domain mismatches
- **Spoofing Attempts:** IP-based senders, suspicious domains
- **Suspicious Patterns:** Free domain registrations, no-reply addresses

---

## Key Components

### 1. EmailHeaderParser

Extracts standard and custom headers from raw email text.

```python
from email_headers import EmailHeaderParser

parser = EmailHeaderParser()
headers = parser.parse_headers(raw_email_text)

# Extract specific information
from_email = parser.extract_email_address(headers.get('From'))
domain = parser.extract_domain(from_email)
```

**Extracted Headers:**
- Standard: From, To, Subject, Date, Reply-To, Cc, Bcc, Message-ID
- Custom: X-* headers, List-* headers, etc.

### 2. EmailAuthenticationValidator

Validates SPF, DKIM, and DMARC records.

```python
from email_headers import EmailAuthenticationValidator

validator = EmailAuthenticationValidator()

# Validate individual methods
dkim_result = validator.validate_dkim(dkim_signature)
spf_result = validator.validate_spf(headers)
dmarc_result = validator.validate_dmarc(headers)

# Validate all methods
full_auth = validator.validate_all(headers)
# Returns: {dkim, spf, dmarc, overall_score (0-1), authenticated}
```

### 3. HeaderPhishingDetector

Detects phishing indicators in headers.

```python
from email_headers import HeaderPhishingDetector

detector = HeaderPhishingDetector()
phishing_results = detector.analyze_headers(headers)

# Returns: {
#   phishing_score: 0.85,
#   is_phishing: True,
#   risk_level: "high",
#   indicators: [list of detected issues]
# }
```

**Detects:**
- Urgent language: "urgent", "verify", "confirm", "act now"
- Generic greetings: "Dear User", "Valued Customer"
- Domain mismatches: From ≠ Reply-To
- Suspicious senders: IP addresses, free domains, generic addresses
- Urgency markers in subject

### 4. CompleteHeaderAnalyzer

Master class combining all analysis methods.

```python
from email_headers import CompleteHeaderAnalyzer

analyzer = CompleteHeaderAnalyzer()
complete_result = analyzer.analyze_complete(raw_email_text)

# Returns comprehensive analysis with:
# - Parsing results
# - Authentication validation
# - Phishing analysis
# - Overall risk score
# - Is suspicious flag
```

---

## Authentication Validation

### SPF (Sender Policy Framework)

SPF allows domain owners to specify which mail servers can send emails for their domain.

**What it checks:**
- Is the sending IP authorized by the domain's SPF record?
- ✅ Pass = Server is authorized
- ❌ Fail = Likely spoofed email
- ⚠️ Neutral/SoftFail = Ambiguous

**Example failure:**
```
From: attacker@legitimate-bank.com (but using attacker's mail server)
Authentication-Results: spf=fail
→ Indicates spoofed email (email spoofing attempt)
```

---

### DKIM (DomainKeys Identified Mail)

DKIM adds a cryptographic signature to emails so recipients can verify the sender's identity.

**What it checks:**
- Is the email signed with the domain's private key?
- ✅ Valid = Email was sent by the domain owner
- ❌ Invalid = Email modified after signing
- ❌ Missing = No signature (could be spoofed)

**Example valid DKIM:**
```
DKIM-Signature: v=1; a=rsa-sha256; d=gmail.com; s=20210112;
→ Gmail-signed email (trusted)
```

---

### DMARC (Domain-based Message Authentication, Reporting & Conformance)

DMARC combines SPF and DKIM with a domain policy.

**What it checks:**
- Did the email pass SPF OR DKIM?
- Does the From domain match the SPF/DKIM domain?
- ✅ Pass = Sender is authenticated
- ❌ Fail = Could be phishing

**DMARC Policies:**
- **None:** No action (monitoring only)
- **Quarantine:** Move to spam folder if failed
- **Reject:** Block email if failed (most secure)

---

## Phishing Detection

### Indicators

The system detects multiple phishing indicators:

#### 1. Subject Line Indicators
```
❌ "URGENT!!! Verify Your Account Now!!!"
❌ "IMMEDIATE ACTION REQUIRED"
❌ "Click Here to Confirm Your Identity"
✅ "February Invoice #12345"
✅ "Meeting Schedule Updated"
```

#### 2. Generic Greetings
```
❌ "Dear User"
❌ "Valued Customer"
⚠️ Legitimate senders usually use your name
```

#### 3. Domain Mismatches
```
From: support@bank.com
Reply-To: support@different-domain.tk
→ Spammers often use different reply-to domain
```

#### 4. Suspicious Sender Patterns
```
❌ 192.168.1.1 (IP address as sender)
❌ user@192.168.1.1.tk (IP-based domain)
❌ no-reply@suspicious.tk (auto-generated)
❌ user@bank-clone.ml (free domain)
```

### Risk Levels

- **Low (0.0-0.3):** Normal email
- **Medium (0.3-0.6):** Some suspicious indicators
- **High (0.6-1.0):** Strong phishing indicators

### Example Phishing Email

```
From: support@bank-example.tk
To: customer@gmail.com
Subject: URGENT!!! Verify Your Account NOW!!!
Reply-To: verify@different-domain.com

Dear Valued Customer,

Your account has been suspended due to suspicious activity.
CLICK HERE IMMEDIATELY to verify your identity.
```

**Analysis:**
- ❌ SPF: FAIL (not authorized)
- ❌ DKIM: MISSING (no signature)
- ❌ DMARC: FAIL (failed authentication)
- ❌ Subject: "URGENT!!!", "NOW!!!" (urgency)
- ❌ Greeting: "Valued Customer" (generic)
- ❌ Domain: .tk (free domain)
- ❌ Reply-To: Different domain
- ❌ Reason to Click: "IMMEDIATELY" (urgency)

**Overall Risk Score: 0.85 (HIGH)**

---

## Integration with Predictions

### How It Affects Spam Score

The `/predict` endpoint now includes header analysis in the spam scoring:

```python
# Weighted contributions to spam score:
overall_score = (
    header_auth_score * 0.25 +      # SPF/DKIM/DMARC failures
    header_features_score * 0.30 +  # Subject/sender analysis
    reputation_score * 0.25 +       # Domain reputation
    url_analysis_score * 0.20       # URL analysis
)
```

### Response Format

```json
{
  "label": "spam",
  "prob_spam": 0.92,
  "header_analysis": {
    "sender": {
      "email": "attacker@suspicious.tk",
      "domain": "suspicious.tk"
    },
    "authentication": {
      "dkim": {"valid": false, "reason": "..."},
      "spf": {"valid": false, "reason": "..."},
      "dmarc": {"valid": false, "reason": "..."},
      "overall_score": 0.0,
      "authenticated": false
    },
    "phishing_analysis": {
      "phishing_score": 0.85,
      "risk_level": "high",
      "indicators": ["Urgency marker", "Generic greeting", ...],
    },
    "overall_risk_score": 0.87,
    "is_suspicious": true
  },
  "advanced_spam_score": 0.78
}
```

---

## Usage Examples

### Example 1: Legitimate Email

```python
from email_headers import CompleteHeaderAnalyzer

legitimate_email = """From: noreply@github.com
To: user@example.com
Subject: GitHub notification
Authentication-Results: dmarc=pass spf=pass dkim=pass
DKIM-Signature: v=1; a=rsa-sha256; d=github.com; s=...

Hello,
You have a new notification on GitHub.
"""

analyzer = CompleteHeaderAnalyzer()
result = analyzer.analyze_complete(legitimate_email)

print(f"Risk Score: {result['overall_risk_score']}")  # Low (0.1-0.2)
print(f"Suspicious: {result['is_suspicious']}")       # False
print(f"Authenticated: {result['authentication']['authenticated']}")  # True
```

**Output:**
```
Risk Score: 0.15
Suspicious: False
Authenticated: True
```

---

### Example 2: Phishing Email

```python
phishing_email = """From: support@paypa1.com
To: customer@gmail.com
Subject: URGENT - Verify Your PayPal Account Now!!!
Reply-To: security@different-domain.tk
Authentication-Results: dmarc=fail spf=fail dkim=fail

Dear Valued Customer,

Your PayPal account has been limited. 
CLICK HERE IMMEDIATELY to restore access.
"""

result = analyzer.analyze_complete(phishing_email)

print(f"Risk Score: {result['overall_risk_score']}")  # High (0.8-0.95)
print(f"Phishing Score: {result['phishing_analysis']['phishing_score']}")  # 0.85+
```

**Output:**
```
Risk Score: 0.89
Phishing Score: 0.85
Risk Level: high
Indicators: [
  "Phishing keyword in subject: 'urgent'",
  "Mismatch: From domain (paypa1.com) != Reply-To (different-domain.tk)",
  "Generic greeting detected",
  "High urgency markers"
]
```

---

### Example 3: Using in API

```bash
TOKEN="your_jwt_token_here"

# Send email for analysis
curl -X POST "http://localhost:8000/predict" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "From: attacker@suspicious.tk\nTo: you@gmail.com\nSubject: URGENT!!!\nACTION REQUIRED NOW!!!"
  }'

# Response includes full header analysis
```

---

## Troubleshooting

### Issue: Headers Not Being Parsed

**Cause:** Email text doesn't contain proper headers.

**Solution:** Ensure email includes standard header format with blank line separator:
```
From: sender@example.com
To: recipient@example.com
Subject: Test

Body content here
```

### Issue: All Authentication Fields Empty

**Cause:** No authentication headers in the email.

**Solution:** This is normal for internal emails or forwarded messages. The system will flag as suspicious which is correct behavior.

### Issue: High Phishing Score on Legitimate Email

**Cause:** Email may have legitimate urgency keywords or informal greeting.

**Solution:** Check context:
- Is sender's domain legitimate?
- Are authentication checks passing?
- Is there excessive urgency language?
- Combine with other spam signals

### Issue: Domain Mismatch False Positive

**Cause:** Legitimate emails may have different Reply-To domain.

**Solution:** This alone isn't a spam indicator. Consider:
- Is sender domain trusted?
- Do authentication checks pass?
- Is Reply-To a known support domain?

---

## Best Practices

### Using Header Analysis Effectively

1. **Don't rely solely on headers:** Combine with content analysis
2. **Check authentication:** Pass rates indicate sender legitimacy
3. **Context matters:** Marketing emails may have urgency but be legitimate
4. **Monitor trends:** Track suspicious senders and domains
5. **User feedback:** Let users report false positives/negatives

### For Email Providers

1. **Publish SPF/DKIM/DMARC:** Protects your domain
2. **Enforce authentication:** Reject unauthenticated emails
3. **Monitor Reply-To:** Flag mismatches
4. **Implement DMARC quarantine:** Catch failed authentications

---

## References

- [RFC 5322 - Email Format](https://tools.ietf.org/html/rfc5322)
- [SPF Specification (RFC 7208)](https://tools.ietf.org/html/rfc7208)
- [DKIM Specification (RFC 6376)](https://tools.ietf.org/html/rfc6376)
- [DMARC Specification (RFC 7489)](https://tools.ietf.org/html/rfc7489)

---

**Last Updated:** February 2024  
**Version:** 1.0


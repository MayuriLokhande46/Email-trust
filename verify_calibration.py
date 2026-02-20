from spam_detector.src.predict import predict
import json

test_cases = [
    {
        "name": "Suspended Account (Real Sample)",
        "text": "Subject: Dear user, your account is suspended. \npermanent suspension.\nClick the link below to claim now.",
        "expected": "spam"
    },
    {
        "name": "Body-only Phishing",
        "text": "Verify your account immediately: http://bank-secure.tk/login",
        "expected": "spam"
    },
    {
        "name": "Lottery Prize Scam",
        "text": "Congratulations! You won $10,000 cash bonus! Click here to claim your reward now.",
        "expected": "spam"
    },
    {
        "name": "Account Suspension (No Headers)",
        "text": "Important notice: Your account has been suspended due to unauthorized activity. Please login to restore access.",
        "expected": "spam"
    },
    {
        "name": "Legitimate Meeting Request",
        "text": "Hey everyone, let's meet tomorrow at 10 AM to discuss the project progress. See you there!",
        "expected": "ham"
    }
]

def run_tests():
    print(f"{'Test Case':<35} | {'Result':<10} | {'Confidence':<10} | {'Reason'}")
    print("-" * 100)
    for case in test_cases:
        result = predict(case['text'])
        label = result['label']
        prob = result.get('prob_spam', 0)
        reasons = ", ".join(result.get('explanation', []))
        
        status = "PASS" if label == case['expected'] else "FAIL"
        print(f"{case['name']:<35} | {label:<10} | {prob:.2f} | {reasons}")

if __name__ == "__main__":
    run_tests()

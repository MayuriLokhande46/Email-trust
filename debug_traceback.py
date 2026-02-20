import traceback
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from spam_detector.src.predict import predict

def debug_predict():
    text = "Subject: Dear user, your account is suspended. \npermanent suspension.\nClick the link below to claim now."
    print(f"Testing text: {text[:50]}...")
    try:
        result = predict(text)
        print("Prediction successful!")
        print(f"Result: {result}")
    except Exception:
        print("\n--- TRACEBACK ---")
        traceback.print_exc()
        print("--- END TRACEBACK ---\n")

if __name__ == "__main__":
    debug_predict()

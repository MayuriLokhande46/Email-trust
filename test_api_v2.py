import requests
import json

def test_api():
    base_url = "http://localhost:8000"
    
    # 1. Register/Login
    username = "testuser_" + str(int(time.time()))
    password = "testpassword"
    
    print(f"Testing with user: {username}")
    
    try:
        # Register
        reg = requests.post(f"{base_url}/register", json={"username": username, "password": password})
        print(f"Register status: {reg.status_code}")
        
        # Login
        log = requests.post(f"{base_url}/login", json={"username": username, "password": password})
        print(f"Login status: {log.status_code}")
        token = log.json().get('access_token')
        
        if not token:
            print("Failed to get token")
            return

        headers = {"Authorization": f"Bearer {token}"}
        
        # 2. Predict
        test_email = "CLAIM YOUR PRIZE NOW! Click here: http://bit.ly/fake-prize"
        pred = requests.post(f"{base_url}/predict", json={"text": test_email}, headers=headers)
        print(f"Predict status: {pred.status_code}")
        print("Result:")
        print(json.dumps(pred.json(), indent=2))
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import time
    test_api()

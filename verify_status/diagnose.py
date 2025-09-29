import requests
import sys

def diagnose_literature_search():
    BASE_URL = "http://localhost:5000"
    
    print("Diagnosing Literature Search Issue")
    print("=" * 50)
    
    # Step 1: Check if server is running
    try:
        health_response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Health check: {health_response.status_code}")
        print(f"   Response: {health_response.json()}")
    except Exception as e:
        print(f"Cannot connect to server: {e}")
        print("Make sure to run: python app.py")
        return
    
    # Step 2: Get authentication token
    login_data = {"username": "demo_provider", "password": "demo123"}
    try:
        login_response = requests.post(f"{BASE_URL}/auth/login", json=login_data)
        if login_response.status_code == 200:
            token = login_response.json()['access_token']
            print("Authentication successful")
        else:
            print(f"Authentication failed: {login_response.status_code}")
            print(f"   Response: {login_response.text}")
            return
    except Exception as e:
        print(f"Authentication error: {e}")
        return
    
    # Step 3: Test literature search with detailed debugging
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "specialty": "Cardiology",
        "keywords": ["heart failure", "statins"],
        "patient_conditions": ["hypertension"],
        "max_results": 5
    }
    
    print(f"\nTesting literature search...")
    print(f"   Endpoint: {BASE_URL}/literature/search")
    print(f"   Payload: {payload}")
    
    try:
        response = requests.post(f"{BASE_URL}/literature/search", json=payload, headers=headers, timeout=10)
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("Literature search successful!")
            print(f"   Source: {data.get('source')}")
            print(f"   Studies found: {len(data.get('studies', []))}")
            for study in data.get('studies', [])[:2]:
                print(f"   - {study.get('title')}")
        else:
            print(f"Literature search failed")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"Request failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnose_literature_search()

import requests
import json

BASE_URL = "http://localhost:5000"

def test_literature_links():
    """Test specifically to see study links in output"""
    print("=== Testing Literature Search Links ===")
    
    # Get token first
    login_data = {
        "username": "demo_provider",
        "password": "demo123"
    }
    
    response = requests.post(f"{BASE_URL}/auth/login", json=login_data)
    if response.status_code != 200:
        print("Login failed")
        return
    
    token = response.json()['access_token']
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test payload that should work
    payload = {
        "specialty": "Cardiology",
        "keywords": ["heart failure", "treatment"],
        "patient_conditions": ["hypertension"],
        "enable_ai_analysis": True,
        "ai_model": "llama-3.1-8b-instant",
        "max_results": 5
    }
    
    print("Sending literature search request...")
    response = requests.post(f"{BASE_URL}/literature/search", json=payload, headers=headers, timeout=30)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("\n=== FULL RESPONSE ===")
        print(json.dumps(data, indent=2))
        
        print("\n=== STUDIES ANALYSIS ===")
        studies = data.get('data', {}).get('studies', [])
        print(f"Number of studies found: {len(studies)}")
        
        if studies:
            print("\n=== STUDY LINKS ===")
            for i, study in enumerate(studies):
                print(f"\nStudy {i+1}:")
                print(f"Title: {study.get('title', 'No title')}")
                print(f"Journal: {study.get('journal', 'Unknown')}")
                print(f"URL: {study.get('url', 'No URL')}")
                print(f"Display URL: {study.get('display_url', 'No display URL')}")
                print(f"Full URL: {study.get('full_url', 'No full URL')}")
                print(f"PubMed ID: {study.get('pubmed_id', 'No ID')}")
        else:
            print("No studies returned! This is the problem.")
            
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    test_literature_links()

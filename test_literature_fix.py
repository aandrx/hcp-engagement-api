#!/usr/bin/env python3
"""
Quick test script to verify literature search fix
"""
import requests
import json
import time

BASE_URL = "http://localhost:5000"

def test_literature_search():
    """Test literature search endpoint with proper error handling"""
    print("Testing Literature Search Fix...")
    
    # First, get authentication token
    login_data = {
        "username": "demo_provider",
        "password": "demo123"
    }
    
    try:
        # Login
        print("1. Attempting login...")
        response = requests.post(f"{BASE_URL}/auth/login", json=login_data, timeout=10)
        
        if response.status_code != 200:
            print(f"Login failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
        token = response.json()['access_token']
        print("Login successful")
        
        # Test literature search
        print("2. Testing literature search...")
        payload = {
            "specialty": "Cardiology",
            "keywords": ["heart failure", "statins"],
            "patient_conditions": ["hypertension"],
            "max_results": 5
        }
        
        headers = {"Authorization": f"Bearer {token}"}
        
        response = requests.post(
            f"{BASE_URL}/literature/search", 
            json=payload, 
            headers=headers, 
            timeout=15
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("Literature search successful!")
            data = response.json()
            print(f"Source: {data.get('source', 'Unknown')}")
            studies = data.get('studies', [])
            print(f"Found {len(studies)} studies")
            
            # Print first study details for verification
            if studies:
                first_study = studies[0]
                print(f"\nFirst study:")
                print(f"  Title: {first_study.get('title', 'N/A')}")
                print(f"  Journal: {first_study.get('journal', 'N/A')}")
                print(f"  Date: {first_study.get('publication_date', 'N/A')}")
                print(f"  Authors: {first_study.get('authors', [])}")
                print(f"  Source: {first_study.get('source', 'N/A')}")
            
            return True
        else:
            print(f"Literature search failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("Cannot connect to API server. Is it running on localhost:5000?")
        print("   Start the server with: python app.py")
        return False
    except requests.exceptions.Timeout:
        print("Request timed out. Server might be overloaded.")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_literature_search()
    if success:
        print("\nLiterature search is working correctly!")
    else:
        print("\nLiterature search still has issues.")

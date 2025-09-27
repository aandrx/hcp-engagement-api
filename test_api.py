import requests
import json
import time

BASE_URL = "http://localhost:5000"

def test_real_literature_search():
    """Test the real literature search endpoint"""
    print("=== Testing Real Literature Search (PubMed) ===")
    
    payload = {
        "specialty": "Cardiology",
        "keywords": ["heart failure", "statins"],
        "patient_conditions": ["hypertension"],
        "max_results": 5
    }
    
    try:
        response = requests.post(f"{BASE_URL}/literature/search-real", json=payload, timeout=15)
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Data Source: {data.get('source', 'Unknown')}")
        print(f"Found {len(data.get('studies', []))} studies")
        for study in data.get('studies', [])[:3]:  # Show first 3
            print(f" - {study.get('title', 'No title')}")
            print(f"   Journal: {study.get('journal', 'Unknown')}")
            print(f"   URL: {study.get('url', 'No URL')}")
    except Exception as e:
        print(f"Error: {e}")

def test_clinical_trials():
    """Test clinical trials search"""
    print("\n=== Testing Clinical Trials Search ===")
    
    payload = {
        "condition": "diabetes",
        "intervention": "metformin"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/literature/clinical-trials", json=payload, timeout=15)
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Found {len(data.get('trials', []))} trials")
        for trial in data.get('trials', [])[:2]:
            print(f" - {trial.get('title', 'No title')}")
            print(f"   Status: {trial.get('status', 'Unknown')}")
    except Exception as e:
        print(f"Error: {e}")

def test_real_billing_analysis():
    """Test real billing analysis with ICD-10/CPT codes"""
    print("\n=== Testing Real Billing Analysis ===")
    
    payload = {
        "patient_id": "pat_123",
        "emr_data": {
            "conditions": ["hypertension", "diabetes"]
        },
        "proposed_treatments": ["medication management", "lab tests"]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/insurance/billing-analysis-real", json=payload)
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print("Billing Codes:")
        for code in data.get('billing_codes', []):
            print(f" - {code}")
        print(f"Estimated Cost: ${data.get('estimated_cost', 0)}")
        print(f"Coding System: {data.get('coding_system', 'Unknown')}")
    except Exception as e:
        print(f"Error: {e}")

def test_health_endpoint():
    """Test health endpoint"""
    print("\n=== Testing Health Endpoint ===")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Status: {data.get('status')}")
        print("Available Services:")
        for service, sources in data.get('available_services', {}).items():
            print(f" - {service}: {sources}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    time.sleep(2)
    
    test_health_endpoint()
    test_real_literature_search()
    test_clinical_trials()
    test_real_billing_analysis()

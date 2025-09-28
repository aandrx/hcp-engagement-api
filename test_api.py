import requests
import json
import time
import sys

BASE_URL = "http://localhost:5000"

def get_auth_token():
    """Get authentication token for testing"""
    print("=== Getting Authentication Token ===")
    
    login_data = {
        "username": "demo_provider",
        "password": "demo123"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/auth/login", json=login_data)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            token = response.json()['access_token']
            print("✅ Login successful")
            return token
        else:
            print(f"❌ Login failed: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error during login: {e}")
        return None

def test_health_endpoint():
    """Test health endpoint"""
    print("\n=== Testing Health Endpoint ===")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Status: {data.get('status')}")
        print(f"Version: {data.get('version')}")
        print("Services Status:")
        for service, status in data.get('services', {}).items():
            print(f" - {service}: {status}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_literature_search(token):
    """Test literature search endpoint - FIXED to match original API"""
    print("\n=== Testing Literature Search ===")
    
    payload = {
        "specialty": "Cardiology",
        "keywords": ["heart failure", "statins"],
        "patient_conditions": ["hypertension"],
        "max_results": 5
    }
    
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        print(f"Payload: {json.dumps(payload, indent=2)}")
        response = requests.post(f"{BASE_URL}/literature/search", json=payload, headers=headers, timeout=15)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Search successful")
            print(f"Data Source: {data.get('source', 'Unknown')}")
            print(f"Found {len(data.get('studies', []))} studies")
            for i, study in enumerate(data.get('studies', [])):
                print(f" {i+1}. {study.get('title', 'No title')}")
                print(f"    Journal: {study.get('journal', 'Unknown')}")
                print(f"    Date: {study.get('publication_date', 'Unknown')}")
                print(f"    Source: {study.get('source', 'Unknown')}")
                print()
            return True
        else:
            print(f"❌ Search failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_risk_prediction(token):
    """Test risk prediction endpoint"""
    print("\n=== Testing Risk Prediction ===")
    
    payload = {
        "patient_data": {
            "age": 65,
            "systolic_bp": 150,
            "glucose": 130,
            "cholesterol": 260,
            "bmi": 32,
            "smoking": 1
        },
        "model_type": "risk"
    }
    
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.post(f"{BASE_URL}/analytics/predict-risk", json=payload, headers=headers)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("✅ Risk prediction successful")
            print(f"Risk Score: {data.get('risk_score')}")
            print(f"Risk Level: {data.get('risk_level')}")
            print(f"Risk Factors: {', '.join(data.get('risk_factors', []))}")
            print(f"Method: {data.get('method')}")
            return True
        else:
            print(f"❌ Prediction failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_cost_prediction(token):
    """Test cost prediction endpoint"""
    print("\n=== Testing Cost Prediction ===")
    
    payload = {
        "patient_data": {
            "age": 65,
            "systolic_bp": 150,
            "proposed_treatments": ["medication", "lab", "consultation"]
        },
        "model_type": "cost"
    }
    
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.post(f"{BASE_URL}/analytics/predict-cost", json=payload, headers=headers)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("✅ Cost prediction successful")
            print(f"Estimated Cost: ${data.get('estimated_cost')}")
            print(f"Cost Efficiency: {data.get('cost_efficiency')}")
            breakdown = data.get('cost_breakdown', {})
            print("Cost Breakdown:")
            for item, cost in breakdown.items():
                print(f" - {item}: ${cost}")
            return True
        else:
            print(f"❌ Prediction failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_population_analytics(token):
    """Test population analytics endpoint"""
    print("\n=== Testing Population Analytics ===")
    
    payload = {
        "patients": [
            {"age": 45, "systolic_bp": 120, "glucose": 100},
            {"age": 65, "systolic_bp": 150, "glucose": 130},
            {"age": 35, "systolic_bp": 110, "glucose": 90},
            {"age": 55, "systolic_bp": 140, "glucose": 115}
        ]
    }
    
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.post(f"{BASE_URL}/analytics/population-trends", json=payload, headers=headers)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("✅ Population analytics successful")
            print(f"Average Age: {data.get('average_age')}")
            print("Risk Distribution:")
            for risk_level, count in data.get('risk_distribution', {}).items():
                print(f" - {risk_level}: {count}")
            return True
        else:
            print(f"❌ Analytics failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("🚀 Starting Open HCP API Tests")
    print("=" * 60)
    
    # First test health endpoint (no auth required)
    if not test_health_endpoint():
        print("❌ Health check failed - API might not be running")
        print("💡 Make sure to run: python app.py")
        return False
    
    # Get authentication token
    token = get_auth_token()
    if not token:
        print("❌ Authentication failed - cannot proceed with protected tests")
        return False
    
    # Run protected endpoint tests
    tests = [
        ("Literature Search", test_literature_search),
        ("Risk Prediction", test_risk_prediction),
        ("Cost Prediction", test_cost_prediction),
        ("Population Analytics", test_population_analytics)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*40}")
        print(f"🧪 TEST: {test_name}")
        print(f"{'='*40}")
        result = test_func(token)
        results.append(result)
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {len(tests) + 1}")  # +1 for health check
    print(f"Passed: {sum(results) + 1}")  # +1 for health check
    print(f"Failed: {len(tests) - sum(results)}")
    
    if all(results):
        print("✅ All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    # Wait a moment for the server to start if running together
    time.sleep(3)
    success = run_all_tests()
    sys.exit(0 if success else 1)

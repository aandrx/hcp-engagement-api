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
            print("Login successful")
            return token
        else:
            print(f"Login failed: {response.text}")
            return None
    except Exception as e:
        print(f"Error during login: {e}")
        return None

def test_health_endpoint():
    """Test health endpoint with Groq status"""
    print("\n=== Testing Health Endpoint ===")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Status: {data.get('status')}")
        print(f"Version: {data.get('version')}")
        
        groq_info = data.get('groq_integration', {})
        print(f"Groq Available: {groq_info.get('available', False)}")
        print(f"Available Models: {', '.join(groq_info.get('models_available', []))}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_groq_literature_search(token):
    """Test literature search with Groq AI analysis"""
    print("\n=== Testing Groq AI-Enhanced Literature Search ===")
    
    payload = {
        "specialty": "Cardiology",
        "keywords": ["heart failure", "statins", "mortality reduction"],
        "patient_conditions": ["hypertension", "diabetes", "hyperlipidemia"],
        "max_results": 5,
        "enable_ai_analysis": True,
        "ai_model": "llama-3.1-8b-instant"
    }
    
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        print(f"Payload: {json.dumps(payload, indent=2)}")
        response = requests.post(f"{BASE_URL}/literature/search", json=payload, headers=headers, timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("Groq AI-enhanced search successful!")
            
            # Show AI capabilities
            ai_caps = data.get('ai_capabilities', {})
            print(f"Groq Available: {ai_caps.get('groq_available', False)}")
            print(f"Model Used: {ai_caps.get('model_used', 'N/A')}")
            
            # Show AI analysis results
            ai_analysis = data.get('ai_analysis')
            if ai_analysis:
                print("\nGROQ AI ANALYSIS RESULTS:")
                print(f"Summary: {ai_analysis.get('summary', 'No summary')}")
                print(f"Confidence Score: {ai_analysis.get('confidence_score', 'N/A')}")
                print(f"Model Used: {ai_analysis.get('model_used', 'Unknown')}")
                
                print("\nKEY FINDINGS:")
                for i, finding in enumerate(ai_analysis.get('key_findings', [])[:3]):
                    print(f" {i+1}. {finding}")
                
                print("\nCLINICAL IMPLICATIONS:")
                for i, implication in enumerate(ai_analysis.get('clinical_implications', [])[:2]):
                    print(f" {i+1}. {implication}")
                
                if 'limitations' in ai_analysis:
                    print("\nLIMITATIONS:")
                    for i, limitation in enumerate(ai_analysis.get('limitations', [])[:2]):
                        print(f" {i+1}. {limitation}")
            else:
                print("AI analysis not available or disabled")
            
            # Show studies
            print(f"\nSTUDIES FOUND ({len(data.get('studies', []))}):")
            for i, study in enumerate(data.get('studies', [])[:2]):
                print(f" {i+1}. {study.get('title', 'No title')}")
                print(f"    Journal: {study.get('journal', 'Unknown')}")
                print(f"    Date: {study.get('publication_date', 'Unknown')}")
                print(f"    Relevance: {study.get('relevance_score', 'N/A')}")
                print()
            
            return True
        else:
            print(f"Search failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_groq_models(token):
    """Test different Groq models"""
    print("\n=== Testing Different Groq Models ===")
    
    # Use only currently available models
    models_to_test = ['llama-3.1-8b-instant', 'mixtral-8x7b-32768', 'gemma2-9b-it', 'llama-3.1-70b-versatile']
    
    for model in models_to_test:
        print(f"\nTesting model: {model}")
        
        payload = {
            "specialty": "Oncology",
            "keywords": ["immunotherapy", "cancer treatment"],
            "patient_conditions": ["lung cancer"],
            "enable_ai_analysis": True,
            "ai_model": model,
            "max_results": 2
        }
        
        headers = {"Authorization": f"Bearer {token}"}
        
        try:
            response = requests.post(f"{BASE_URL}/literature/search", json=payload, headers=headers, timeout=25)
            if response.status_code == 200:
                data = response.json()
                ai_analysis = data.get('ai_analysis')
                if ai_analysis:
                    print(f"{model}: Success (Confidence: {ai_analysis.get('confidence_score', 'N/A')})")
                else:
                    print(f"{model}: No AI analysis returned")
            else:
                print(f"{model}: Failed - {response.status_code}")
            
            time.sleep(2)  # Rate limiting
                
        except Exception as e:
            print(f"{model}: Error - {e}")

def test_groq_direct_analysis(token):
    """Test direct Groq AI analysis endpoint"""
    print("\n=== Testing Direct Groq Analysis ===")
    
    payload = {
        "text": "Heart failure patients with hypertension often benefit from ACE inhibitors and beta-blockers. Recent studies show combination therapy can reduce mortality by up to 30%.",
        "analysis_type": "clinical_implications",
        "model": "llama-3.1-8b-instant",  # Use correct model name
        "context": "Cardiology patient with heart failure and hypertension"
    }
    
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.post(f"{BASE_URL}/ai/analyze", json=payload, headers=headers, timeout=20)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("Direct Groq analysis successful!")
            print(f"Analysis Type: {data.get('analysis_type')}")
            print(f"Model Used: {data.get('model_used')}")
            print(f"Analysis Result: {data.get('analysis', 'No result')[:200]}...")
            return True
        else:
            print(f"Analysis failed: {response.text}")
            return False
    except Exception as e:
        print(f"Error: {e}")
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
            print("Risk prediction successful")
            print(f"Risk Score: {data.get('risk_score')}")
            print(f"Risk Level: {data.get('risk_level')}")
            print(f"Risk Factors: {', '.join(data.get('risk_factors', []))}")
            return True
        else:
            print(f"Prediction failed: {response.text}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def run_groq_demo():
    """Run Groq AI-enhanced demonstration"""
    print("Starting Groq-Powered HCP API Demo")
    print("=" * 60)
    
    # Health check
    if not test_health_endpoint():
        print("Health check failed")
        return False
    
    # Get token
    token = get_auth_token()
    if not token:
        print("Authentication failed")
        return False
    
    # Run Groq-enhanced tests
    tests = [
        ("Groq Literature Search", test_groq_literature_search),
        ("Direct Groq Analysis", test_groq_direct_analysis),
        ("Risk Prediction", test_risk_prediction),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"TEST: {test_name}")
        print(f"{'='*50}")
        result = test_func(token)
        results.append(result)
        time.sleep(3)  # Longer pause for Groq API calls
    
    # Test different models
    test_different_groq_models(token)
    
    # Summary
    print("\n" + "=" * 60)
    print("GROQ AI DEMO SUMMARY")
    print("=" * 60)
    print(f"Tests Completed: {len(tests)}")
    print(f"Successful: {sum(results)}")
    print(f"Failed: {len(tests) - sum(results)}")
    
    if sum(results) >= 2:  # At least 2 successful
        print("Groq AI demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("• Ultra-fast Groq AI inference")
        print("• Multiple model support (Llama 3, Mixtral)")
        print("• Intelligent literature relevance analysis")
        print("• Clinical context understanding")
        print("• Direct AI analysis endpoint")
    else:
        print("Some tests failed")
    
    return sum(results) >= 2

if __name__ == "__main__":
    time.sleep(3)
    success = run_groq_demo()
    sys.exit(0 if success else 1)

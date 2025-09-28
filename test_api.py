import requests
import json
import time
import sys
import textwrap
from typing import Dict, Any

# formatting functions
def format_compact_output(data: Dict[str, Any]) -> str:
    """Format compact output for better readability print(f"Analysis Type: {analysis_data.get('analysis_type', 'N/A')}")
                print(f"Model Used: {metadata.get('model_used', 'N/A')}")
                
                analysis_result = analysis_data.get('analysis', 'No analysis content')r better readability"""
    output = []
    
    if 'status' in data:
        status_icon = "PASS" if data['status'] == 'success' else "FAIL"
        output.append(f"{status_icon} Status: {data['status']}")
    
    if 'data' in data:
        if 'studies' in data['data']:
            output.append(f"Studies Found: {len(data['data']['studies'])}")
        
        if 'ai_analysis' in data['data'] and data['data']['ai_analysis']:
            analysis = data['data']['ai_analysis']
            output.append(f"AI Confidence: {analysis.get('confidence_score', 'N/A')}")
            if 'summary' in analysis:
                summary = textwrap.fill(analysis['summary'], width=80)
                output.append(f"Summary: {summary}")
    
    if 'metadata' in data:
        if 'next_steps' in data['metadata']:
            output.append("Next Steps:")
            for step in data['metadata']['next_steps'][:2]:
                output.append(f"   • {step}")
    
    return "\n".join(output)

def print_formatted_response(response_data: Dict[str, Any], title: str = ""):
    """Print formatted API response"""
    if title:
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")
    
    print(format_compact_output(response_data))

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
        # Test both formats
        for format_type in ['compact', 'detailed']:
            print(f"\nTesting {format_type.upper()} format:")
            response = requests.post(
                f"{BASE_URL}/literature/search?format={format_type}", 
                json=payload, 
                headers=headers, 
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print_formatted_response(data, f"Groq AI Search ({format_type.upper()})")
                
                if format_type == 'detailed':
                    # Show additional details in detailed mode
                    if data.get('data', {}).get('ai_analysis'):
                        analysis = data['data']['ai_analysis']
                        print("\nKey Findings:")
                        for i, finding in enumerate(analysis.get('key_findings', [])[:3]):
                            print(f" {i+1}. {finding}")
            
            else:
                print(f"Search failed: {response.status_code}")
                return False
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
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
        "model": "llama-3.1-8b-instant",
        "context": "Cardiology patient with heart failure and hypertension"
    }
    
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.post(f"{BASE_URL}/ai/analyze", json=payload, headers=headers, timeout=20)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("Direct Groq analysis successful!")
            
            # Debug: Print the full response to see actual structure
            print(f"\nFull API Response:")
            print(json.dumps(data, indent=2))
            
            # Check the new response structure
            if data.get('status') == 'success':
                analysis_data = data.get('data', {})
                metadata = data.get('metadata', {})
                
                print(f"\nAnalysis Type: {analysis_data.get('analysis_type', 'N/A')}")
                print(f"Model Used: {metadata.get('model_used', 'N/A')}")
                
                analysis_result = analysis_data.get('analysis', 'No analysis content')
                print(f"\nFull Analysis Result:")
                print("-" * 50)
                print(analysis_result)
                print("-" * 50)
                
                if 'next_steps' in metadata:
                    print(f"\nNext Steps:")
                    for step in metadata.get('next_steps', []):
                        print(f"   • {step}")
            else:
                print(f"API returned error status: {data.get('message', 'Unknown error')}")
                return False
            
            return True
        else:
            print(f"Analysis failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"Error: {e}")
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

def test_population_analysis(token):
    """Test population health trends analysis"""
    print("\n=== Testing Population Health Analysis ===")
    
    # Sample patient data for population analysis - FIXED STRUCTURE
    patient_data_list = [
        {
            "age": 65,
            "systolic_bp": 150,
            "glucose": 130,
            "cholesterol": 260,
            "bmi": 32,
            "smoking": 1,
            "conditions": ["hypertension", "diabetes"]
        },
        {
            "age": 58,
            "systolic_bp": 142,
            "glucose": 125,
            "cholesterol": 240,
            "bmi": 28,
            "smoking": 0,
            "conditions": ["hypertension"]
        },
        {
            "age": 72,
            "systolic_bp": 160,
            "glucose": 140,
            "cholesterol": 280,
            "bmi": 35,
            "smoking": 1,
            "conditions": ["hypertension", "diabetes", "hyperlipidemia"]
        },
        {
            "age": 45,
            "systolic_bp": 130,
            "glucose": 95,
            "cholesterol": 200,
            "bmi": 25,
            "smoking": 0,
            "conditions": []
        }
    ]
    
    payload = {
        "patients": patient_data_list
    }
    
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.post(f"{BASE_URL}/analytics/population-trends", json=payload, headers=headers)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("Population analysis successful!")
            
            # Debug: Print the full response to see what's actually returned
            print(f"\nFULL RESPONSE:")
            print(json.dumps(data, indent=2))
            
            # Check if we have the expected data structure
            if 'data' in data:
                trends = data['data']
            else:
                trends = data  # Fallback to direct response
                
            print(f"\nPOPULATION HEALTH INSIGHTS:")
            print(f"Population Size: {trends.get('population_size', 'N/A')}")
            print(f"Average Age: {trends.get('average_age', 'N/A')}")
            print(f"Average Risk Score: {trends.get('average_risk_score', 'N/A')}")
            
            risk_dist = trends.get('risk_distribution', {})
            if risk_dist:
                print(f"Risk Distribution: {risk_dist}")
                total_patients = sum(risk_dist.values())
                for risk_level, count in risk_dist.items():
                    percentage = (count / total_patients) * 100 if total_patients > 0 else 0
                    print(f"  {risk_level.upper()}: {count} patients ({percentage:.1f}%)")
            else:
                print("Risk Distribution: No data")
            
            # Show age groups
            age_groups = trends.get('age_groups', {})
            if age_groups:
                print(f"\nAGE GROUPS:")
                for group, count in age_groups.items():
                    print(f"  {group.replace('_', ' ').title()}: {count} patients")
            
            # Show risk factors prevalence
            risk_factors = trends.get('risk_factors_prevalence', {})
            if risk_factors:
                print(f"\nRISK FACTORS PREVALENCE:")
                for factor, stats in risk_factors.items():
                    print(f"  {factor}: {stats.get('count', 0)} patients ({stats.get('prevalence', 0)}%)")
            
            return True
        else:
            print(f"Population analysis failed: {response.text}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_groq_demo():
    """Run Groq AI-enhanced demonstration"""
    print("Starting Groq-Powered HCP API Demo")
    print("=" * 60)
    
    test_results = {}
    
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
        ("Population Analysis", test_population_analysis),  # Added population analysis
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"TEST: {test_name}")
        print(f"{'='*50}")
        result = test_func(token)
        test_results[test_name] = {
            'status': 'success' if result else 'failed',
            'timestamp': time.time()
        }
        time.sleep(2)
    
    # Test different models
    test_different_groq_models(token)
    
    # Enhanced summary
    print("\n" + "=" * 60)
    print("GROQ AI DEMO SUMMARY")
    print("=" * 60)
    
    successful_tests = sum(1 for result in test_results.values() if result['status'] == 'success')
    total_tests = len(test_results)
    
    print(f"Tests Completed: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {total_tests - successful_tests}")
    print(f"Success Rate: {(successful_tests/total_tests)*100:.1f}%")
    
    print("\nFeatures Demonstrated:")
    features = [
        "• Ultra-fast Groq AI inference",
        "• Multiple response formats (compact/detailed)",
        "• Structured clinical insights", 
        "• Actionable next steps",
        "• Multi-model analysis support",
        "• Population health analytics"  # Added population analytics
    ]
    for feature in features:
        print(feature)
    
    return successful_tests >= 2

if __name__ == "__main__":
    time.sleep(3)
    success = run_groq_demo()
    sys.exit(0 if success else 1)

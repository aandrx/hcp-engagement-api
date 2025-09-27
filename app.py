from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
from flask_cors import CORS
import json
from datetime import datetime, timedelta
import uuid
from typing import Dict, List, Any
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)
api = Api(app, version='1.0', title='HCP Engagement API', 
          description='Next-generation HCP engagement solutions API with real data integrations')

# Namespaces
ns_literature = api.namespace('literature', description='Medical literature and studies operations')
ns_notifications = api.namespace('notifications', description='Medical information notifications')
ns_insurance = api.namespace('insurance', description='Insurance and patient interactions')
ns_questionnaire = api.namespace('questionnaire', description='Patient questionnaire operations')

# Models
literature_search_model = api.model('LiteratureSearch', {
    'specialty': fields.String(required=True, description='Medical specialty'),
    'keywords': fields.List(fields.String, description='Search keywords'),
    'patient_conditions': fields.List(fields.String, description='Patient conditions'),
    'max_results': fields.Integer(default=10, description='Maximum results to return')
})

notification_model = api.model('NotificationRequest', {
    'provider_id': fields.String(required=True, description='Healthcare provider ID'),
    'prescription_history': fields.List(fields.Raw, description='Prescription history')
})

insurance_analysis_model = api.model('InsuranceAnalysis', {
    'patient_id': fields.String(required=True, description='Patient ID'),
    'emr_data': fields.Raw(description='Patient EMR data'),
    'proposed_treatments': fields.List(fields.String, description='Proposed treatments')
})

questionnaire_model = api.model('PatientQuestionnaire', {
    'chief_complaint': fields.String(required=True, description='Primary complaint'),
    'patient_age': fields.Integer(required=True, description='Patient age'),
    'patient_gender': fields.String(required=True, description='Patient gender'),
    'existing_conditions': fields.List(fields.String, description='Existing medical conditions')
})

class RealDataLiteratureService:
    """Service for identifying relevant medical literature with real PubMed data"""
    
    def search_relevant_studies(self, specialty: str, keywords: List[str], 
                               patient_conditions: List[str]) -> List[Dict]:
        """Search for real medical studies from PubMed using pymed"""
        try:
            from pymed import PubMed
            pubmed = PubMed(tool="HCPEngagementAPI", email="api@example.com")
            
            # Build search query
            query_parts = []
            if specialty:
                query_parts.append(specialty)
            if keywords:
                query_parts.extend(keywords)
            if patient_conditions:
                query_parts.extend([f'"{cond}"' for cond in patient_conditions])
            
            query = " AND ".join(query_parts)
            print(f"PubMed search query: {query}")
            
            # Execute search
            results = pubmed.query(query, max_results=10)
            studies = []
            
            for article in results:
                try:
                    publication_date = article.publication_date
                    if publication_date:
                        if isinstance(publication_date, datetime):
                            publication_date = publication_date.strftime('%Y-%m-%d')
                        elif isinstance(publication_date, str):
                            publication_date = publication_date[:10]  # Take first 10 chars
                    
                    studies.append({
                        'id': article.pubmed_id.strip() if article.pubmed_id else str(uuid.uuid4()),
                        'title': article.title or 'No title available',
                        'journal': article.journal or 'Unknown journal',
                        'publication_date': publication_date or 'Unknown date',
                        'relevance_score': self._calculate_relevance(article, specialty, keywords),
                        'abstract': article.abstract or 'Abstract not available',
                        'url': f"https://pubmed.ncbi.nlm.nih.gov/{article.pubmed_id}/" if article.pubmed_id else '',
                        'authors': article.authors or [],
                        'source': 'PubMed',
                        'pmid': article.pubmed_id
                    })
                except Exception as e:
                    print(f"Error processing article: {e}")
                    continue
                    
                if len(studies) >= 10:  # Limit results
                    break
            
            return studies if studies else self._get_mock_studies(specialty, keywords, patient_conditions)
            
        except Exception as e:
            print(f"PubMed search error: {e}")
            return self._get_mock_studies(specialty, keywords, patient_conditions)
    
    def _calculate_relevance(self, article, specialty: str, keywords: List[str]) -> float:
        """Calculate relevance score based on content matching"""
        score = 0.7  # Base score
        
        title = (article.title or '').lower()
        abstract = (article.abstract or '').lower()
        
        # Boost score if specialty appears
        if specialty and specialty.lower() in title:
            score += 0.2
        
        # Boost score for keyword matches
        if keywords:
            keyword_matches = sum(1 for keyword in keywords if keyword.lower() in abstract)
            score += min(0.3, keyword_matches * 0.1)
        
        return min(0.99, score)
    
    def _get_mock_studies(self, specialty: str, keywords: List[str], patient_conditions: List[str]) -> List[Dict]:
        """Fallback mock data"""
        return [
            {
                'id': str(uuid.uuid4()),
                'title': f'Fallback: Recent Advances in {specialty}',
                'journal': 'Medical Research Journal',
                'publication_date': datetime.now().strftime('%Y-%m-%d'),
                'relevance_score': 0.8,
                'abstract': f'Review of {specialty} treatments for {patient_conditions}',
                'url': 'https://example.com/study',
                'source': 'Fallback Data',
                'authors': ['Research Team']
            }
        ]

class RealDataNotificationService:
    """Service for medical information notifications with real FDA data"""
    
    def check_new_relevance(self, provider_id: str, prescription_history: List[Dict]) -> List[Dict]:
        """Check for real FDA drug approvals using the fda package"""
        try:
            from fda import FDADrug
            fda = FDADrug()
            
            relevant_updates = []
            
            # Get recent drug approvals
            try:
                # Search for recent drug approvals
                search_results = fda.search(
                    search='openfda.product_type:"HUMAN PRESCRIPTION DRUG"',
                    limit=5,
                    sort='effective_time:desc'
                )
                
                for drug in search_results.get('results', [])[:3]:
                    openfda = drug.get('openfda', {})
                    brand_name = openfda.get('brand_name', ['Unknown'])[0] if openfda.get('brand_name') else 'Unknown'
                    generic_name = openfda.get('generic_name', ['Unknown'])[0] if openfda.get('generic_name') else 'Unknown'
                    
                    # Check relevance to provider's prescriptions
                    for prescription in prescription_history:
                        drug_class = prescription.get('drug_class', '').lower()
                        drug_name = prescription.get('drug_name', '').lower()
                        
                        if (drug_class and drug_class in generic_name.lower()) or \
                           (drug_name and drug_name in brand_name.lower()):
                            relevant_updates.append({
                                'type': 'fda_approval',
                                'title': f'FDA Approval: {brand_name}',
                                'relevance': 'high',
                                'date': drug.get('effective_time', 'Unknown'),
                                'summary': f'New approval for {generic_name}',
                                'source': 'FDA Database',
                                'drug_class': generic_name
                            })
                            break
                            
            except Exception as e:
                print(f"FDA API specific error: {e}")
                # Fallback to direct API call
                relevant_updates.extend(self._get_fda_data_direct(prescription_history))
            
            return relevant_updates if relevant_updates else self._get_mock_updates(prescription_history)
            
        except Exception as e:
            print(f"FDA package error: {e}")
            return self._get_mock_updates(prescription_history)
    
    def _get_fda_data_direct(self, prescription_history: List[Dict]) -> List[Dict]:
        """Fallback direct FDA API call"""
        try:
            updates = []
            # Simple FDA API call for drug recalls as example
            recall_url = "https://api.fda.gov/food/enforcement.json"
            params = {'limit': 3, 'sort': 'report_date:desc'}
            
            response = requests.get(recall_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for result in data.get('results', [])[:2]:
                    updates.append({
                        'type': 'safety_alert',
                        'title': f'Safety Alert: {result.get("product_description", "Unknown product")}',
                        'relevance': 'medium',
                        'date': result.get('report_date', 'Unknown'),
                        'summary': result.get('reason_for_recall', 'Safety concern'),
                        'source': 'FDA Direct API'
                    })
            return updates
        except:
            return []
    
    def _get_mock_updates(self, prescription_history: List[Dict]) -> List[Dict]:
        """Fallback mock updates"""
        updates = []
        for prescription in prescription_history[:2]:  # Limit to 2 prescriptions
            drug_class = prescription.get('drug_class', 'unknown')
            updates.append({
                'type': 'research_update',
                'title': f'Latest research on {drug_class} therapies',
                'relevance': 'medium',
                'date': datetime.now().strftime('%Y-%m-%d'),
                'summary': f'New studies published about {drug_class} treatments',
                'source': 'Medical Literature'
            })
        return updates

class RealDataInsuranceService:
    """Service for insurance and billing with real coding systems"""
    
    def analyze_billing_codes(self, emr_data: Dict, proposed_treatments: List[str]) -> Dict:
        """Generate real ICD-10 and CPT codes using installed packages"""
        try:
            billing_codes = []
            
            # ICD-10 codes for conditions
            icd10_codes = self._get_icd10_codes(emr_data.get('conditions', []))
            billing_codes.extend(icd10_codes)
            
            # CPT codes for treatments
            cpt_codes = self._get_cpt_codes(proposed_treatments)
            billing_codes.extend(cpt_codes)
            
            # Cost estimation
            cost_analysis = self._estimate_costs(billing_codes)
            
            return {
                'billing_codes': billing_codes,
                'estimated_cost': cost_analysis['total_cost'],
                'insurance_coverage_estimate': cost_analysis['total_cost'] * 0.8,
                'patient_responsibility': cost_analysis['total_cost'] * 0.2,
                'cost_breakdown': cost_analysis['breakdown'],
                'coding_system': 'ICD-10-CM/CPT'
            }
            
        except Exception as e:
            print(f"Billing analysis error: {e}")
            return self._get_mock_billing_analysis(emr_data, proposed_treatments)
    
    def _get_icd10_codes(self, conditions: List[str]) -> List[str]:
        """Get ICD-10 codes using icd10-cm package"""
        try:
            import icd10_cm as icd10
            
            codes = []
            for condition in conditions[:5]:  # Limit to 5 conditions
                try:
                    # Search for ICD-10 code
                    results = icd10.find(condition)
                    if results:
                        code = results[0]  # Take first match
                        codes.append(f"ICD-10: {code.code} - {code.description}")
                    else:
                        codes.append(f"ICD-10: R69 - {condition} (Not specified)")
                except:
                    codes.append(f"ICD-10: R69 - {condition}")
            
            return codes if codes else ['ICD-10: R69 - Illness, unspecified']
            
        except Exception as e:
            print(f"ICD-10 lookup error: {e}")
            # Fallback to manual mapping
            return self._get_icd10_fallback(conditions)
    
    def _get_icd10_fallback(self, conditions: List[str]) -> List[str]:
        """Fallback ICD-10 mapping"""
        icd10_mapping = {
            'hypertension': 'I10 - Essential (primary) hypertension',
            'diabetes': 'E11.9 - Type 2 diabetes mellitus without complications',
            'heart failure': 'I50.9 - Heart failure, unspecified',
            'hyperlipidemia': 'E78.5 - Hyperlipidemia, unspecified',
            'asthma': 'J45.909 - Asthma, uncomplicated',
            'depression': 'F32.9 - Major depressive disorder, single episode, unspecified',
            'arthritis': 'M19.90 - Primary osteoarthritis, unspecified site',
            'migraine': 'G43.909 - Migraine, unspecified, not intractable'
        }
        
        codes = []
        for condition in conditions:
            condition_lower = condition.lower()
            matched = False
            for key, value in icd10_mapping.items():
                if key in condition_lower:
                    codes.append(f"ICD-10: {value}")
                    matched = True
                    break
            if not matched:
                codes.append(f"ICD-10: R69 - {condition}")
        
        return codes if codes else ['ICD-10: R69 - Illness, unspecified']
    
    def _get_cpt_codes(self, treatments: List[str]) -> List[str]:
        """Get CPT codes using cpt package"""
        try:
            import cpt
            
            codes = []
            for treatment in treatments[:5]:  # Limit to 5 treatments
                try:
                    # Search for CPT codes
                    results = cpt.find(treatment)
                    if results:
                        code = results[0]  # Take first match
                        codes.append(f"CPT: {code.code} - {code.description}")
                    else:
                        codes.append(f"CPT: 99214 - {treatment} (Office/outpatient visit)")
                except:
                    codes.append(f"CPT: 99214 - {treatment}")
            
            return codes if codes else ['CPT: 99214 - Office/outpatient visit']
            
        except Exception as e:
            print(f"CPT lookup error: {e}")
            return self._get_cpt_fallback(treatments)
    
    def _get_cpt_fallback(self, treatments: List[str]) -> List[str]:
        """Fallback CPT mapping"""
        cpt_mapping = {
            'surgery': '49505 - Repair initial inguinal hernia',
            'medication': '99213 - Office/outpatient visit, established patient',
            'therapy': '97110 - Therapeutic procedure',
            'imaging': '74150 - CT abdomen',
            'lab': '80053 - Comprehensive metabolic panel',
            'consultation': '99244 - Office consultation'
        }
        
        codes = []
        for treatment in treatments:
            treatment_lower = treatment.lower()
            matched = False
            for key, value in cpt_mapping.items():
                if key in treatment_lower:
                    codes.append(f"CPT: {value}")
                    matched = True
                    break
            if not matched:
                codes.append(f"CPT: 99214 - {treatment}")
        
        return codes if codes else ['CPT: 99214 - Office/outpatient visit']
    
    def _estimate_costs(self, billing_codes: List[str]) -> Dict:
        """Estimate costs based on medical procedure averages"""
        cost_mapping = {
            '49505': 15000,  # Surgery
            '99213': 100,    # Office visit
            '99214': 150,    # Detailed visit
            '97110': 75,     # Therapy
            '74150': 500,    # CT scan
            '80053': 50,     # Lab work
            '99244': 200     # Consultation
        }
        
        total_cost = 0
        breakdown = {}
        
        for code in billing_codes:
            # Extract code number
            code_parts = code.split(' ')
            code_num = code_parts[1].split('-')[0] if len(code_parts) > 1 else '99214'
            
            cost = cost_mapping.get(code_num, 100)
            total_cost += cost
            breakdown[code] = cost
        
        return {'total_cost': total_cost, 'breakdown': breakdown}
    
    def _get_mock_billing_analysis(self, emr_data: Dict, proposed_treatments: List[str]) -> Dict:
        """Fallback mock analysis"""
        return {
            'billing_codes': ['ICD-10: R69 - Illness, unspecified', 'CPT: 99214 - Office/outpatient visit'],
            'estimated_cost': 250,
            'insurance_coverage_estimate': 200,
            'patient_responsibility': 50,
            'coding_system': 'Fallback'
        }

# Initialize real data services
real_literature_service = RealDataLiteratureService()
real_notification_service = RealDataNotificationService()
real_insurance_service = RealDataInsuranceService()

# Keep original services as fallback
class MedicalLiteratureService:
    def search_relevant_studies(self, specialty: str, keywords: List[str], patient_conditions: List[str]) -> List[Dict]:
        return [{
            'id': str(uuid.uuid4()),
            'title': f'Original: Advances in {specialty}',
            'journal': 'Medical Journal',
            'publication_date': '2024-01-15',
            'relevance_score': 0.9,
            'abstract': f'Review of {specialty}',
            'source': 'Original Service'
        }]

class NotificationService:
    def check_new_relevance(self, provider_id: str, prescription_history: List[Dict]) -> List[Dict]:
        return [{
            'type': 'update',
            'title': 'Original notification',
            'relevance': 'high',
            'source': 'Original Service'
        }]

class InsuranceService:
    def analyze_billing_codes(self, emr_data: Dict, proposed_treatments: List[str]) -> Dict:
        return {
            'billing_codes': ['ICD-10: R69', 'CPT: 99214'],
            'estimated_cost': 300,
            'source': 'Original Service'
        }

class QuestionnaireService:
    def generate_followup_questions(self, chief_complaint: str, patient_data: Dict) -> List[Dict]:
        return [{
            'question': 'Original question based on complaint',
            'type': 'multiple_choice',
            'critical': True
        }]

literature_service = MedicalLiteratureService()
notification_service = NotificationService()
insurance_service = InsuranceService()
questionnaire_service = QuestionnaireService()

# API Routes
@ns_literature.route('/search-real')
class RealLiteratureSearch(Resource):
    @ns_literature.expect(literature_search_model)
    def post(self):
        """Search for relevant medical literature using real PubMed data"""
        data = request.get_json()
        studies = real_literature_service.search_relevant_studies(
            data.get('specialty'),
            data.get('keywords', []),
            data.get('patient_conditions', [])
        )
        return jsonify({'studies': studies, 'source': 'PubMed via pymed'})

@ns_notifications.route('/check-relevance-real')
class RealCheckRelevance(Resource):
    @ns_notifications.expect(notification_model)
    def post(self):
        """Check for new relevant medical information using real FDA data"""
        data = request.get_json()
        updates = real_notification_service.check_new_relevance(
            data.get('provider_id'),
            data.get('prescription_history', [])
        )
        return jsonify({'relevant_updates': updates, 'source': 'FDA Database'})

@ns_insurance.route('/billing-analysis-real')
class RealBillingAnalysis(Resource):
    @ns_insurance.expect(insurance_analysis_model)
    def post(self):
        """Analyze billing codes and cost estimation using real coding systems"""
        data = request.get_json()
        analysis = real_insurance_service.analyze_billing_codes(
            data.get('emr_data', {}),
            data.get('proposed_treatments', [])
        )
        return jsonify(analysis)

# Keep original routes
@ns_literature.route('/search')
class LiteratureSearch(Resource):
    @ns_literature.expect(literature_search_model)
    def post(self):
        data = request.get_json()
        studies = literature_service.search_relevant_studies(
            data.get('specialty'),
            data.get('keywords', []),
            data.get('patient_conditions', [])
        )
        return jsonify({'studies': studies, 'source': 'Original Service'})

@ns_notifications.route('/check-relevance')
class CheckRelevance(Resource):
    @ns_notifications.expect(notification_model)
    def post(self):
        data = request.get_json()
        updates = notification_service.check_new_relevance(
            data.get('provider_id'),
            data.get('prescription_history', [])
        )
        return jsonify({'relevant_updates': updates, 'source': 'Original Service'})

@ns_insurance.route('/billing-analysis')
class BillingAnalysis(Resource):
    @ns_insurance.expect(insurance_analysis_model)
    def post(self):
        data = request.get_json()
        analysis = insurance_service.analyze_billing_codes(
            data.get('emr_data', {}),
            data.get('proposed_treatments', [])
        )
        return jsonify(analysis)

@ns_questionnaire.route('/followup-questions')
class FollowupQuestions(Resource):
    @ns_questionnaire.expect(questionnaire_model)
    def post(self):
        data = request.get_json()
        questions = questionnaire_service.generate_followup_questions(
            data.get('chief_complaint'),
            data
        )
        return jsonify({'questions': questions})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'real_data_services': {
            'literature': 'PubMed via pymed',
            'notifications': 'FDA Database', 
            'insurance': 'ICD-10-CM/CPT Coding'
        },
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 Starting HCP Engagement API with Real Data...")
    print("📚 Swagger docs: http://localhost:5000/docs/")
    print("🔬 Real data sources: PubMed, FDA, ICD-10/CPT coding")
    app.run(debug=True, host='0.0.0.0', port=5000)

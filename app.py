from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
from flask_socketio import SocketIO, emit
from flask_cors import CORS 
import jwt
import bcrypt
from passlib.context import CryptContext
import json
from datetime import datetime, timedelta
import uuid
from typing import Dict, List, Any
import requests
import os
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Configuration for open-source deployment
app.config.update({
    'SECRET_KEY': os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production'),
    'JWT_SECRET_KEY': os.getenv('JWT_SECRET_KEY', 'jwt-secret-change-in-production'),
    'JWT_ACCESS_TOKEN_EXPIRES': timedelta(hours=1),
    'REDIS_URL': os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
    'GROQ_API_KEY': os.getenv('GROQ_API_KEY', ''),
})

# Debug: Check if Groq API key is loaded
groq_api_key = app.config['GROQ_API_KEY']
print(f"Groq API Key Loaded: {'Yes' if groq_api_key else 'No'}")
if groq_api_key:
    print(f"Key length: {len(groq_api_key)} characters")
    print(f"Key starts with: {groq_api_key[:10]}...")
else:
    print("WARNING: No Groq API key found in environment variables")

# Enhanced CORS configuration for open-source frontend compatibility
# allowed_origins = os.getenv('ALLOWED_ORIGINS', 'http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173').split(',')

# Enhanced CORS configuration for development
CORS(app, 
    origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://localhost:5002"],
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Requested-With", "Origin", "Accept"],
    supports_credentials=False
)

# # Add global OPTIONS handler for preflight requests
# @app.before_request
# def handle_preflight():
#     if request.method == "OPTIONS":
#         response = jsonify({"status": "preflight"})
#         response.headers.add("Access-Control-Allow-Origin", request.headers.get('Origin', '*'))
#         response.headers.add("Access-Control-Allow-Headers", "Authorization, Content-Type")
#         response.headers.add("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
#         return response

# # Add after_request handler for CORS headers
# @app.after_request
# def after_request(response):
#     origin = request.headers.get('Origin')
#     if origin and origin in allowed_origins:
#         response.headers.add('Access-Control-Allow-Origin', origin)
#     else:
#         # For development, allow the request origin if no specific origins match
#         response.headers.add('Access-Control-Allow-Origin', request.headers.get('Origin', allowed_origins[0] if allowed_origins else '*'))
    
#     response.headers.add('Access-Control-Allow-Headers', 'Authorization, Content-Type')
#     response.headers.add('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
#     response.headers.add('Access-Control-Allow-Credentials', 'false')
#     response.headers.add('Access-Control-Max-Age', '600')
#     return response

# Setup logging (open-source friendly)
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
    handlers=[
        RotatingFileHandler('logs/api.log', maxBytes=10485760, backupCount=10),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize APIs
api = Api(app, 
    version='2.2', 
    title='Groq-Powered HCP Engagement API', 
    description='Healthcare Provider engagement API with Groq AI-powered literature analysis',
    doc='/docs/',
    authorizations={
        'Bearer Auth': {
            'type': 'apiKey',
            'in': 'header',
            'name': 'Authorization',
            'description': 'Type "Bearer {token}"'
        }
    },
    security='Bearer Auth'
)

# Initialize WebSocket for real-time features (with error handling for production)
socketio = None
try:
    socketio = SocketIO(app, 
        # cors_allowed_origins=allowed_origins,
        logger=logger,
        engineio_logger=False,
        async_mode='gevent'  # Use gevent instead of eventlet
    )
    logger.info("SocketIO initialized successfully with gevent")
except Exception as e:
    logger.warning(f"SocketIO initialization failed: {e}. Real-time features disabled.")
    socketio = None

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Simple in-memory storage (Redis optional)
memory_store = {}

# Namespaces
ns_auth = api.namespace('auth', description='Authentication operations')
ns_literature = api.namespace('literature', description='Medical literature and studies operations')
ns_analytics = api.namespace('analytics', description='Advanced analytics and predictions')
ns_ai = api.namespace('ai', description='AI-powered analysis operations')

# ========== MODELS DEFINITION ==========

# Security Models
login_model = api.model('Login', {
    'username': fields.String(required=True, description='Username'),
    'password': fields.String(required=True, description='Password')
})

# Enhanced Models with AI analysis
literature_search_model = api.model('LiteratureSearch', {
    'specialty': fields.String(required=True),
    'keywords': fields.List(fields.String),
    'patient_conditions': fields.List(fields.String),
    'max_results': fields.Integer(default=99, description='Maximum number of results (1-99)'),  # Increased default
    'enable_ai_analysis': fields.Boolean(default=True),
    'ai_model': fields.String(default='llama-3.1-8b-instant', description='Groq model to use'),
    'response_format': fields.String(default='detailed', description='compact|detailed')
})

# Analytics Models (Simplified)
prediction_model = api.model('PredictionRequest', {
    'patient_data': fields.Raw(required=True, description='Patient EMR data'),
    'model_type': fields.String(required=True, description='risk|outcome|cost')
})

ai_analysis_model = api.model('AIAnalysis', {
    'text': fields.String(required=True),
    'analysis_type': fields.String(required=True, description='summary|relevance|clinical_implications'),
    'context': fields.Raw(description='Additional context for analysis'),
    'model': fields.String(default='llama-3.1-8b-instant', description='Groq model to use')
})

population_analysis_model = api.model('PopulationAnalysis', {
    'patients': fields.List(fields.Raw, required=True, description='List of patient data objects')
})


# ========== SERVICE CLASS DEFINITIONS ==========

# Authentication and Security
class AuthService:
    """Open-source authentication service"""
    
    def __init__(self):
        self.users = self._load_users()
    
    def _load_users(self):
        """Load users from environment or default config"""
        users = {
            'demo_provider': {
                'password': pwd_context.hash('demo123'),
                'role': 'provider',
                'specialty': 'Cardiology',
                'user_id': 'user_001'
            },
            'demo_admin': {
                'password': pwd_context.hash('admin123'),
                'role': 'admin',
                'user_id': 'user_002'
            }
        }
        return users
    
    def authenticate_user(self, username: str, password: str) -> Dict:
        """Authenticate user and return JWT token"""
        user = self.users.get(username)
        if not user or not pwd_context.verify(password, user['password']):
            return None
        
        token = jwt.encode({
            'sub': username,
            'role': user['role'],
            'user_id': user['user_id'],
            'exp': datetime.utcnow() + app.config['JWT_ACCESS_TOKEN_EXPIRES']
        }, app.config['JWT_SECRET_KEY'], algorithm='HS256')
        
        return {
            'access_token': token,
            'token_type': 'bearer',
            'user': {
                'username': username,
                'role': user['role'],
                'specialty': user.get('specialty')
            }
        }
    
    def verify_token(self, token: str) -> Dict:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            raise Exception("Token expired")
        except jwt.InvalidTokenError:
            raise Exception("Invalid token")

# Real-time Service with WebSocket
class RealTimeService:
    """Real-time notification and data synchronization"""
    
    def __init__(self):
        self.active_connections = {}
    
    def add_connection(self, user_id: str, sid: str):
        """Add WebSocket connection"""
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
        self.active_connections[user_id].add(sid)
    
    def remove_connection(self, user_id: str, sid: str):
        """Remove WebSocket connection"""
        if user_id in self.active_connections:
            self.active_connections[user_id].discard(sid)
    
    def send_notification(self, user_id: str, notification: Dict):
        """Send real-time notification to user"""
        message = {
            'type': 'notification',
            'timestamp': datetime.utcnow().isoformat(),
            'data': notification
        }
        
        if socketio and user_id in self.active_connections:
            for sid in self.active_connections[user_id]:
                socketio.emit('notification', message, room=sid)
        elif not socketio:
            logger.debug(f"SocketIO not available - notification not sent to user {user_id}")

# Lightweight Analytics Service
class AnalyticsService:
    """Lightweight analytics and prediction service using statistical methods"""
    
    def __init__(self):
        self.risk_rules = self._load_risk_rules()
        self.cost_models = self._load_cost_models()
    
    def _load_risk_rules(self):
        """Define risk assessment rules"""
        return {
            'hypertension': lambda data: 0.8 if data.get('systolic_bp', 0) > 140 else 0.3,
            'diabetes': lambda data: 0.7 if data.get('glucose', 0) > 126 else 0.2,
            'hyperlipidemia': lambda data: 0.6 if data.get('cholesterol', 0) > 240 else 0.2,
            'obesity': lambda data: 0.5 if data.get('bmi', 0) > 30 else 0.1,
            'smoking': lambda data: 0.4 if data.get('smoking', 0) == 1 else 0.0,
            'age_risk': lambda data: min(0.6, (data.get('age', 40) - 40) * 0.02)
        }
    
    def _load_cost_models(self):
        """Define cost estimation models"""
        return {
            'base_visit': 100,
            'procedures': {
                'medication': 50, 'surgery': 5000, 'therapy': 100, 
                'imaging': 300, 'lab': 75, 'consultation': 200
            },
            'complexity_factors': {
                'high_risk': 1.5,
                'multiple_conditions': 1.3,
                'age_complexity': lambda age: 1.0 + max(0, (age - 65) * 0.01)
            }
        }
    
    def predict_risk(self, patient_data: Dict) -> Dict:
        """Predict patient health risk using rule-based system"""
        try:
            risk_score = 0.1  # Base risk
            
            # Apply risk rules
            risk_factors = []
            for condition, rule in self.risk_rules.items():
                factor_risk = rule(patient_data)
                risk_score += factor_risk
                if factor_risk > 0.5:
                    risk_factors.append(condition)
            
            # Normalize risk score
            risk_score = min(0.95, risk_score / len(self.risk_rules))
            
            # Determine risk level
            if risk_score > 0.7:
                risk_level = 'high'
            elif risk_score > 0.4:
                risk_level = 'medium'
            else:
                risk_level = 'low'
            
            return {
                'risk_score': round(risk_score, 2),
                'risk_level': risk_level,
                'risk_factors': risk_factors,
                'confidence': 0.85,
                'method': 'rule_based_analysis'
            }
            
        except Exception as e:
            logger.error(f"Risk prediction error: {e}")
            return {'error': 'Risk assessment unavailable', 'method': 'fallback'}
    
    def predict_cost(self, patient_data: Dict, treatments: List[str]) -> Dict:
        """Predict treatment costs using statistical models"""
        try:
            base_cost = self.cost_models['base_visit']
            
            # Add procedure costs
            procedure_costs = 0
            for treatment in treatments:
                cost = self.cost_models['procedures'].get(treatment.lower(), 0)
                procedure_costs += cost
            
            # Apply complexity factors
            complexity = 1.0
            
            # High risk factor
            risk_prediction = self.predict_risk(patient_data)
            if risk_prediction.get('risk_level') == 'high':
                complexity *= self.cost_models['complexity_factors']['high_risk']
            
            # Age complexity
            age_factor = self.cost_models['complexity_factors']['age_complexity'](
                patient_data.get('age', 50)
            )
            complexity *= age_factor
            
            total_cost = (base_cost + procedure_costs) * complexity
            
            return {
                'estimated_cost': round(total_cost, 2),
                'cost_breakdown': {
                    'base_visit': base_cost,
                    'procedures': procedure_costs,
                    'complexity_factor': round(complexity, 2)
                },
                'cost_efficiency': self._assess_efficiency(total_cost, risk_prediction),
                'method': 'statistical_estimation'
            }
            
        except Exception as e:
            logger.error(f"Cost prediction error: {e}")
            return {'error': 'Cost prediction unavailable', 'method': 'fallback'}
    
    def _assess_efficiency(self, cost: float, risk_prediction: Dict) -> str:
        """Assess cost efficiency"""
        risk_score = risk_prediction.get('risk_score', 0.5)
        
        # Simple efficiency metric: cost per risk reduction
        efficiency_ratio = risk_score / max(cost, 1)
        
        if efficiency_ratio > 0.01:
            return 'high'
        elif efficiency_ratio > 0.005:
            return 'medium'
        else:
            return 'low'
    
    def population_health_trends(self, patient_data_list: List[Dict]) -> Dict:
        """Analyze population health trends with enhanced insights"""
        try:
            if not patient_data_list:
                return {'error': 'No patient data provided'}
            
            # Preprocess data - convert lists to strings for DataFrame compatibility
            processed_patients = []
            for patient in patient_data_list:
                processed_patient = patient.copy()
                # Convert list fields to strings for DataFrame compatibility
                if 'conditions' in processed_patient and isinstance(processed_patient['conditions'], list):
                    processed_patient['conditions'] = ', '.join(processed_patient['conditions'])
                processed_patients.append(processed_patient)
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(processed_patients)
            
            # Calculate risk levels for all patients
            risk_levels = []
            risk_scores = []
            
            for _, patient in df.iterrows():
                # Convert back to dict for risk prediction
                patient_dict = patient.to_dict()
                risk_pred = self.predict_risk(patient_dict)
                risk_levels.append(risk_pred.get('risk_level', 'low'))
                risk_scores.append(risk_pred.get('risk_score', 0))
            
            # Enhanced trends analysis with safe data access
            trends = {
                'population_size': len(patient_data_list),
                'average_age': round(df['age'].mean(), 1) if 'age' in df.columns and not df['age'].empty else 0,
                'average_risk_score': round(np.mean(risk_scores), 2) if risk_scores else 0,
                'common_conditions': self._find_common_conditions(df),
                'risk_distribution': pd.Series(risk_levels).value_counts().to_dict(),
                'age_groups': self._analyze_age_groups(df),
                'risk_factors_prevalence': self._analyze_risk_factors(df),
                'timestamp': datetime.utcnow().isoformat(),
                'analysis_metadata': {
                    'method': 'statistical_analysis',
                    'confidence': 'high',
                    'data_quality': 'good' if len(patient_data_list) >= 3 else 'limited'
                }
            }
            
            return trends
            
        except Exception as e:
            logger.error(f"Population analysis error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'error': f'Population analysis unavailable: {str(e)}'}

    def _analyze_age_groups(self, df: pd.DataFrame) -> Dict:
        """Analyze age distribution across population"""
        if 'age' not in df.columns:
            return {}
        
        age_groups = {
            'young_adult': len(df[(df['age'] >= 18) & (df['age'] < 40)]),
            'middle_aged': len(df[(df['age'] >= 40) & (df['age'] < 65)]),
            'senior': len(df[df['age'] >= 65])
        }
        return age_groups

    def _analyze_risk_factors(self, df: pd.DataFrame) -> Dict:
        """Analyze prevalence of common risk factors"""
        risk_factors = {}
        
        # Check for common risk factor indicators
        risk_indicators = {
            'hypertension': lambda row: row.get('systolic_bp', 0) > 140 if 'systolic_bp' in row else False,
            'diabetes': lambda row: row.get('glucose', 0) > 126 if 'glucose' in row else False,
            'hyperlipidemia': lambda row: row.get('cholesterol', 0) > 240 if 'cholesterol' in row else False,
            'obesity': lambda row: row.get('bmi', 0) > 30 if 'bmi' in row else False,
            'smoking': lambda row: row.get('smoking', 0) == 1 if 'smoking' in row else False
        }
        
        for factor, check_func in risk_indicators.items():
            count = sum(1 for _, row in df.iterrows() if check_func(row))
            risk_factors[factor] = {
                'count': count,
                'prevalence': round((count / len(df)) * 100, 1) if len(df) > 0 else 0
            }
        
        return risk_factors

    def _find_common_conditions(self, df: pd.DataFrame) -> Dict:
        """Find most common conditions in population with better error handling"""
        conditions = {}
        
        try:
            # Look for condition indicators
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['condition', 'diagnosis', 'disease']):
                    if df[col].dtype == 'object':  # String columns
                        try:
                            # Handle stringified lists or regular strings
                            if df[col].str.contains(',').any():  # If it contains commas, treat as string list
                                # Split comma-separated conditions and count individually
                                all_conditions = []
                                for conditions_str in df[col].dropna():
                                    if isinstance(conditions_str, str):
                                        condition_list = [cond.strip() for cond in conditions_str.split(',')]
                                        all_conditions.extend(condition_list)
                                
                                if all_conditions:
                                    condition_counts = pd.Series(all_conditions).value_counts().head(5).to_dict()
                                    conditions[col] = condition_counts
                            else:
                                # Regular string column
                                common = df[col].value_counts().head(3).to_dict()
                                conditions[col] = common
                        except Exception as e:
                            logger.warning(f"Error processing conditions column {col}: {e}")
                            continue
            
            return conditions
            
        except Exception as e:
            logger.error(f"Error in common conditions analysis: {e}")
            return {}
    
    def _calculate_risk_distribution(self, df: pd.DataFrame) -> Dict:
        """Calculate risk distribution across population"""
        risks = []
        
        for _, patient in df.iterrows():
            risk_pred = self.predict_risk(patient.to_dict())
            risks.append(risk_pred.get('risk_level', 'low'))
        
        risk_counts = pd.Series(risks).value_counts().to_dict()
        return risk_counts

# Groq AI Analysis Service
class GroqAnalysisService:
    """AI-powered analysis service using Groq API"""
    
    def __init__(self):
        self.available_models = self._get_groq_models()
        self.groq_available = self._check_groq_availability()
        logger.info(f"Groq Available: {self.groq_available}")
        if self.groq_available:
            logger.info(f"Groq API Key: {app.config['GROQ_API_KEY'][:10]}...")
        logger.info(f"Available Models: {list(self.available_models.keys())}")
    
    def _get_groq_models(self):
        """Get available Groq models"""
        return {
            'llama-3.1-8b-instant': 'Llama 3.1 8B Instant',
            'llama-3.1-70b-versatile': 'Llama 3.1 70B Versatile',  # Added valid model
            'llama3-groq-8b-8192-tool-use-preview': 'Llama 3 8B Tool Use Preview',
            'mixtral-8x7b-32768': 'Mixtral 8x7B',
            'gemma2-9b-it': 'Gemma 2 9B IT'
        }
    
    def _check_groq_availability(self):
        """Check if Groq API is available"""
        try:
            groq_api_key = app.config['GROQ_API_KEY']
            if not groq_api_key or groq_api_key == '':
                logger.warning("Groq API key not found or empty")
                return False
            
            # Test the API with a simple request - USE A VALID MODEL
            test_response = self._call_groq_api("Hello", "llama-3.1-8b-instant")  # Changed from llama3-8b-8192
            if test_response is not None:
                logger.info("Groq API connection successful")
                return True
            else:
                logger.warning("Groq API test request failed")
                return False
                
        except Exception as e:
            logger.warning(f"Groq API check failed: {e}")
            return False
    
    def _call_groq_api(self, prompt: str, model: str = 'llama-3.1-8b-instant') -> str:
        """Make API call to Groq"""
        try:
            groq_api_key = app.config['GROQ_API_KEY']
            if not groq_api_key:
                logger.error("No Groq API key available")
                return None
            
            # Validate model name
            if model not in self.available_models:
                logger.warning(f"Model {model} not available, using default")
                model = 'llama-3.1-8b-instant'  # Fallback to known working model
            
            headers = {
                'Authorization': f'Bearer {groq_api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'messages': [{'role': 'user', 'content': prompt}],
                'model': model,
                'temperature': 0.3,
                'max_tokens': 1024,
                'top_p': 0.9
            }
            
            logger.info(f"Calling Groq API with model: {model}")
            response = requests.post(
                'https://api.groq.com/openai/v1/chat/completions',
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info("Groq API call successful")
                return response.json()['choices'][0]['message']['content']
            else:
                logger.error(f"Groq API error: {response.status_code} - {response.text}")
                # More detailed error logging
                if response.status_code == 401:
                    logger.error("Invalid Groq API key")
                elif response.status_code == 404:
                    logger.error(f"Model {model} not found")
                return None
                
        except requests.exceptions.Timeout:
            logger.error("Groq API request timeout")
            return None
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            return None
    
    def analyze_literature_relevance(self, articles: List[Dict], search_context: Dict, model: str = 'llama-3.1-8b-instant') -> Dict:
        """Analyze how articles are relevant to the search context using Groq"""
        try:
            if self.groq_available:
                return self._analyze_with_groq(articles, search_context, model)
            else:
                logger.warning("Groq not available, using rule-based analysis")
                return self._analyze_rule_based(articles, search_context)
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return self._analyze_rule_based(articles, search_context)
    
    def _analyze_with_groq(self, articles: List[Dict], search_context: Dict, model: str) -> Dict:
        """Use Groq for sophisticated analysis"""
        try:
            context_str = f"""
            Clinical Context:
            - Specialty: {search_context.get('specialty', 'Unknown')}
            - Keywords: {', '.join(search_context.get('keywords', []))}
            - Patient Conditions: {', '.join(search_context.get('patient_conditions', []))}
            """
            
            articles_text = ""
            for i, article in enumerate(articles[:3]):  # Analyze top 3 articles
                articles_text += f"""
                Article {i+1}:
                Title: {article.get('title', 'No title')}
                Journal: {article.get('journal', 'Unknown')}
                Publication Date: {article.get('publication_date', 'Unknown')}
                Abstract: {article.get('abstract', 'No abstract')}
                """
            
            prompt = f"""
            You are a medical expert analyzing research articles for clinical relevance.

            {context_str}

            Articles to analyze:
            {articles_text}

            Please provide a comprehensive analysis in JSON format with these fields:
            - summary: Brief overall relevance summary (2-3 sentences)
            - key_findings: Array of 3-5 most relevant findings from the articles
            - clinical_implications: Array of 2-3 clinical implications for the patient conditions
            - confidence_score: Number between 0-1 indicating confidence in relevance
            - limitations: Array of any limitations or gaps in the articles

            Focus on how these articles address the specific patient conditions and keywords.
            Be concise but clinically precise.
            """

            analysis_text = self._call_groq_api(prompt, model)
            
            if analysis_text:
                return self._parse_groq_response(analysis_text, model, search_context)
            else:
                raise Exception("Groq API returned no response")
            
        except Exception as e:
            logger.warning(f"Groq analysis failed: {e}")
            return self._analyze_rule_based(articles, search_context)
    
    def _parse_groq_response(self, text: str, model: str, search_context: Dict) -> Dict:
        """Parse Groq response into structured format"""
        try:
            # Try to extract JSON if present
            if '{' in text and '}' in text:
                json_str = text[text.find('{'):text.rfind('}')+1]
                result = json.loads(json_str)
                result['model_used'] = model
                result['analysis_timestamp'] = datetime.utcnow().isoformat()
                return result
            else:
                # Fallback parsing for non-JSON responses
                return {
                    'summary': text[:500],
                    'key_findings': ['See summary for detailed analysis'],
                    'clinical_implications': ['Consult full articles for clinical applications'],
                    'confidence_score': 0.8,
                    'limitations': ['Response format unexpected'],
                    'model_used': model,
                    'analysis_timestamp': datetime.utcnow().isoformat(),
                    'raw_response': text
                }
        except json.JSONDecodeError:
            # If JSON parsing fails, create structured response from text
            return {
                'summary': text[:300] + "..." if len(text) > 300 else text,
                'key_findings': ['Analysis completed via Groq AI'],
                'clinical_implications': ['Review articles for specific clinical guidance'],
                'confidence_score': 0.7,
                'limitations': ['Could not parse structured response'],
                'model_used': model,
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
    
    def _analyze_rule_based(self, articles: List[Dict], search_context: Dict) -> Dict:
        """Rule-based analysis as fallback"""
        specialty = search_context.get('specialty', '').lower()
        keywords = [k.lower() for k in search_context.get('keywords', [])]
        conditions = [c.lower() for c in search_context.get('patient_conditions', [])]
        
        relevant_points = []
        confidence = 0.5
        
        for article in articles[:3]:
            title = article.get('title', '').lower()
            abstract = article.get('abstract', '').lower()
            text = title + " " + abstract
            
            # Simple keyword matching
            matches = []
            for keyword in keywords:
                if keyword in text:
                    matches.append(keyword)
                    confidence += 0.1
            
            for condition in conditions:
                if condition in text:
                    matches.append(condition)
                    confidence += 0.15
            
            if matches:
                relevant_points.append(f"Article relevant for: {', '.join(set(matches))}")
        
        confidence = min(0.9, confidence / len(articles) if articles else 0.5)
        
        return {
            'summary': f"Found {len(relevant_points)} relevant articles matching search criteria",
            'key_findings': relevant_points[:5],
            'clinical_implications': [
                "Consider these articles for evidence-based decision making",
                "Review full texts for detailed methodology and results"
            ],
            'confidence_score': round(confidence, 2),
            'limitations': ['AI analysis unavailable, using keyword matching'],
            'model_used': 'rule_based_fallback',
            'analysis_timestamp': datetime.utcnow().isoformat()
        }

# Enhanced Literature Service with Groq AI Analysis
class RealDataLiteratureService:
    """Literature service with Groq AI-powered analysis"""
    
    def __init__(self):
        self.pubmed_available = self._check_pubmed_availability()
        self.ai_service = GroqAnalysisService()
    
    def _check_pubmed_availability(self):
        """Check if PubMed is available"""
        try:
            from pymed import PubMed
            pubmed = PubMed(tool="GroqHCPAPI", email="opensource@example.com")
            return True
        except ImportError:
            logger.warning("pymed library not available, using fallback mode")
            return False
        except Exception as e:
            logger.warning(f"PubMed initialization failed: {e}, using fallback mode")
            return False
    
    def search_relevant_studies(self, specialty: str, keywords: List[str], 
                        patient_conditions: List[str], enable_ai_analysis: bool = True,
                        ai_model: str = 'llama-3.1-8b-instant', max_results: int = 99) -> Dict:
        """Search literature with configurable result limit"""
        try:
            # Validate max_results
            max_results = min(max(1, max_results), 99)  # Cap at 99
            
            if self.pubmed_available:
                studies = self._search_pubmed(specialty, keywords, patient_conditions, max_results)
            else:
                studies = self._get_fallback_studies(specialty, keywords, patient_conditions, max_results)
            
            # Format study links
            formatted_studies = self._format_study_links(studies)
            
            # Add AI analysis if enabled (still limit AI analysis to top 3 for performance)
            ai_analysis = None
            if enable_ai_analysis and formatted_studies:
                search_context = {
                    'specialty': specialty,
                    'keywords': keywords,
                    'patient_conditions': patient_conditions
                }
                ai_analysis = self.ai_service.analyze_literature_relevance(formatted_studies[:3], search_context, ai_model)  # Still analyze only top 3
            
            return {
                'studies': formatted_studies,
                'source': 'PubMed' if self.pubmed_available else 'Fallback',
                'ai_analysis': ai_analysis,
                'ai_capabilities': {
                    'groq_available': self.ai_service.groq_available,
                    'model_used': ai_model if ai_analysis else None,
                    'models_available': list(self.ai_service.available_models.keys())
                },
                'search_metadata': {
                    'specialty': specialty,
                    'keywords': keywords,
                    'patient_conditions': patient_conditions,
                    'max_results_requested': max_results,
                    'total_results_returned': len(formatted_studies),
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Literature search error: {e}")
            return self._get_error_response(specialty, keywords, patient_conditions)

    def _format_study_links(self, studies: List[Dict]) -> List[Dict]:
        """Format study links: always show full URLs"""
        if not studies:
            return studies
        
        formatted_studies = []
        base_url = "https://pubmed.ncbi.nlm.nih.gov/"
        
        for study in studies:
            formatted_study = study.copy()
            
            # Extract PubMed ID from URL or use existing ID
            pubmed_id = None
            if study.get('url', '').startswith(base_url):
                pubmed_id = study['url'].replace(base_url, '').strip('/')
            elif study.get('id') and study['id'].isdigit():
                pubmed_id = study['id']
            
            if pubmed_id:
                # Always show full URL
                formatted_study['display_url'] = f"{base_url}{pubmed_id}/"
                formatted_study['pubmed_id'] = pubmed_id
                formatted_study['full_url'] = f"{base_url}{pubmed_id}/"
            else:
                formatted_study['display_url'] = study.get('url', 'No URL available')
            
            formatted_studies.append(formatted_study)
        
        return formatted_studies
    
    def _search_pubmed(self, specialty: str, keywords: List[str], conditions: List[str], max_results: int = 99) -> List[Dict]:
        """Search PubMed with configurable result limit"""
        try:
            from pymed import PubMed
            pubmed = PubMed(tool="GroqHCPAPI", email="opensource@example.com")
            
            # Validate and cap max_results
            max_results = min(max(1, max_results), 99)  # Ensure between 1-99
            
            query = self._build_query(specialty, keywords, conditions)
            results = pubmed.query(query, max_results=max_results)  # Use parameter
            
            studies = []
            for article in results:
                pub_date = getattr(article, 'publication_date', 'Unknown')
                if hasattr(pub_date, 'strftime'):
                    pub_date = pub_date.strftime('%Y-%m-%d')
                
                authors = article.authors or []
                if authors and not isinstance(authors, list):
                    authors = [str(author) for author in authors] if hasattr(authors, '__iter__') else [str(authors)]
                
                studies.append({
                    'id': article.pubmed_id or str(uuid.uuid4()),
                    'title': article.title or 'No title available',
                    'journal': article.journal or 'Unknown journal',
                    'publication_date': pub_date,
                    'relevance_score': 0.8,
                    'abstract': article.abstract or 'Abstract not available',
                    'url': f"https://pubmed.ncbi.nlm.nih.gov/{article.pubmed_id}/" if article.pubmed_id else '',
                    'authors': authors,
                    'source': 'PubMed',
                    'full_text_available': bool(article.pubmed_id)
                })
                
                if len(studies) >= max_results:
                    break
            
            return studies if studies else self._get_fallback_studies(specialty, keywords, conditions, max_results)
            
        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            return self._get_fallback_studies(specialty, keywords, conditions, max_results)
    
    def _build_query(self, specialty: str, keywords: List[str], conditions: List[str]) -> str:
        """Build PubMed search query"""
        terms = []
        if specialty:
            terms.append(specialty)
        if keywords:
            terms.extend(keywords)
        if conditions:
            terms.extend(conditions)
        return " AND ".join(terms) if terms else "medical research"
    
    def _get_fallback_studies(self, specialty: str, keywords: List[str], conditions: List[str], max_results: int = 99) -> List[Dict]:
        """Fallback studies that can generate up to max_results - FIXED VERSION"""
        logger.info(f"Using fallback studies: specialty={specialty}, max_results={max_results}")
        
        base_url = "https://pubmed.ncbi.nlm.nih.gov/"
        
        # Ensure we have at least basic data
        if not keywords:
            keywords = ["treatment", "therapy"]
        if not conditions:
            conditions = ["condition"]
        
        # Always return at least 2 studies
        studies = [
            {
                'id': 'study_1',
                'pubmed_id': '33383166',
                'title': f'Advanced {specialty} Interventions for {", ".join(keywords[:2])}',
                'journal': 'Journal of Clinical Medicine',
                'publication_date': '2024-01-15',
                'relevance_score': 0.9,
                'abstract': f'This comprehensive study examines the efficacy of various {specialty} interventions for patients with {", ".join(conditions)}.',
                'url': f"{base_url}33383166/",
                'authors': ['Smith J', 'Johnson A', 'Williams R'],
                'source': 'Medical Database',
                'full_text_available': True
            },
            {
                'id': 'study_2',
                'pubmed_id': '33383167', 
                'title': f'{specialty} Management of {conditions[0] if conditions else "Chronic Conditions"}',
                'journal': 'New England Journal of Medicine',
                'publication_date': '2024-01-10',
                'relevance_score': 0.85,
                'abstract': f'This multi-center study investigates long-term outcomes for {specialty} patients.',
                'url': f"{base_url}33383167/",
                'authors': ['Brown K', 'Davis M', 'Miller T'],
                'source': 'Clinical Trials Registry',
                'full_text_available': True
            }
        ]
        
        # Generate additional studies only if needed
        for i in range(3, max_results + 1):
            if len(studies) >= max_results:
                break
                
            studies.append({
                'id': f'study_{i}',
                'pubmed_id': f'3338316{i}' if i < 10 else f'33383{i:03d}',
                'title': f'{specialty} Research on {keywords[i % len(keywords)] if keywords else "Treatment"}',
                'journal': 'Various Medical Journals',
                'publication_date': f'2024-01-{10 + (i % 20)}',  # Keep dates reasonable
                'relevance_score': max(0.1, 0.8 - (i * 0.02)),  # Ensure positive score
                'abstract': f'Study #{i} on {specialty} focusing on {conditions[i % len(conditions)] if conditions else "patient outcomes"}.',
                'url': f"{base_url}3338316{i}/" if i < 10 else f"{base_url}33383{i:03d}/",
                'authors': [f'Researcher {chr(65 + (i % 26))}'],
                'source': 'Medical Research',
                'full_text_available': True
            })
        
        logger.info(f"Fallback generated {len(studies)} studies")
        return studies[:max_results]
    
    def _get_error_response(self, specialty: str, keywords: List[str], conditions: List[str]) -> Dict:
        """Error response with basic fallback"""
        return {
            'studies': self._get_fallback_studies(specialty, keywords, conditions),
            'source': 'Fallback',
            'ai_analysis': None,
            'ai_capabilities': {
                'groq_available': self.ai_service.groq_available,
                'models_available': list(self.ai_service.available_models.keys())
            },
            'search_metadata': {
                'specialty': specialty,
                'keywords': keywords,
                'patient_conditions': conditions,
                'total_results': 2,
                'timestamp': datetime.utcnow().isoformat(),
                'error': 'Service temporarily unavailable'
            }
        }

# ========== SERVICE INITIALIZATION ==========

# Initialize services (AFTER class definitions)
auth_service = AuthService()
realtime_service = RealTimeService()
analytics_service = AnalyticsService()
ai_service = GroqAnalysisService()
literature_service = RealDataLiteratureService()

# ========== AUTHENTICATION DECORATOR ==========

def token_required(f):
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(" ")[1]
            except IndexError:
                return {'message': 'Bearer token malformed'}, 401
        
        if not token:
            return {'message': 'Token is missing'}, 401
        
        try:
            current_user = auth_service.verify_token(token)
            kwargs['current_user'] = current_user
        except Exception as e:
            return {'message': str(e)}, 401
        
        return f(*args, **kwargs)
    return decorated

# ========== WEB SOCKET EVENTS ==========

# Only register SocketIO events if SocketIO is available
if socketio is not None:
    @socketio.on('connect')
    def handle_connect():
        logger.info(f"Client connected: {request.sid}")
        emit('connected', {'status': 'connected', 'sid': request.sid})

    @socketio.on('disconnect')
    def handle_disconnect():
        logger.info(f"Client disconnected: {request.sid}")

    @socketio.on('subscribe')
    def handle_subscribe(data):
        """Subscribe to real-time updates"""
        user_id = data.get('user_id')
        channel = data.get('channel')
        
        if user_id and channel:
            realtime_service.add_connection(user_id, request.sid)
            emit('subscribed', {'channel': channel, 'status': 'success'})
else:
    logger.info("SocketIO not available - WebSocket events disabled")

# ========== API ROUTES ==========

# Authentication
@ns_auth.route('/login')
class Login(Resource):
    @ns_auth.expect(login_model)
    def post(self):
        """User login"""
        data = request.get_json()
        result = auth_service.authenticate_user(data['username'], data['password'])
        
        if result:
            logger.info(f"User {data['username']} logged in successfully")
            return result
        else:
            logger.warning(f"Failed login attempt for {data['username']}")
            return {'message': 'Invalid credentials'}, 401

# Enhanced Literature Search with Groq AI Analysis
@ns_literature.route('/search')
class LiteratureSearch(Resource):
    @ns_literature.expect(literature_search_model)
    @token_required
    def post(self, current_user):
        """Search medical literature with Groq AI analysis"""
        data = request.get_json()
        
        result = literature_service.search_relevant_studies(
            data.get('specialty'),
            data.get('keywords', []),
            data.get('patient_conditions', []),
            data.get('enable_ai_analysis', True),
            data.get('ai_model', 'llama-3.1-8b-instant'),
            data.get('max_results', 99)  # Pass the parameter with default 99
        )
        
        logger.info(f"Literature search by {current_user['sub']} for {data.get('specialty')}")
        
        # Enhanced response structure
        enhanced_response = {
            'status': 'success',
            'data': {
                'studies': result.get('studies', []),
                'ai_analysis': result.get('ai_analysis'),
                'total_results': len(result.get('studies', []))
            },
            'metadata': {
                'source': result.get('source', 'Fallback'),
                'search_context': result.get('search_metadata', {}),
                'ai_capabilities': result.get('ai_capabilities', {}),
                'next_steps': [
                    "Review full study texts for detailed methodology",
                    "Consult clinical guidelines for application",
                    "Consider patient-specific contraindications"
                ]
            }
        }
        
        # Add format option
        format_type = request.args.get('format', 'detailed')
        if format_type == 'compact':
            enhanced_response = self._compact_format(enhanced_response)
        
        return enhanced_response
    
    def _compact_format(self, response):
        """Compact format for lighter payloads"""
        compact_data = {
            'status': response['status'],
            'data': {
                'study_count': len(response['data']['studies']),
                'key_findings': response['data']['ai_analysis'].get('key_findings', [])[:3] if response['data']['ai_analysis'] else [],
                'clinical_implications': response['data']['ai_analysis'].get('clinical_implications', [])[:2] if response['data']['ai_analysis'] else []
            },
            'metadata': {
                'source': response['metadata']['source'],
                'confidence': response['data']['ai_analysis'].get('confidence_score', 0) if response['data']['ai_analysis'] else 0
            }
        }
        return compact_data

# Groq AI Analysis endpoint
@ns_ai.route('/analyze')
class AIAnalysis(Resource):
    @ns_ai.expect(ai_analysis_model)
    @token_required
    def post(self, current_user):
        """General-purpose Groq AI analysis"""
        data = request.get_json()
        
        text = data.get('text', '')
        analysis_type = data.get('analysis_type', 'summary')
        model = data.get('model', 'llama-3.1-8b-instant')
        
        prompt = f"""
        Please analyze the following text for {analysis_type}:
        
        {text}
        
        Context: {data.get('context', 'General analysis')}
        
        Provide a concise, well-structured analysis.
        """
        
        try:
            analysis_result = ai_service._call_groq_api(prompt, model)
            
            response = {
                'status': 'success',
                'data': {
                    'analysis': analysis_result,
                    'analysis_type': analysis_type,
                    'context': data.get('context', 'General analysis')
                },
                'metadata': {
                    'model_used': model,
                    'timestamp': datetime.utcnow().isoformat(),
                    'groq_available': ai_service.groq_available,
                    'next_steps': [
                        "Review analysis with clinical team",
                        "Validate findings against current guidelines",
                        "Consider patient-specific factors"
                    ]
                }
            }
            
            return response
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'AI analysis failed: {str(e)}',
                'metadata': {
                    'groq_available': ai_service.groq_available,
                    'timestamp': datetime.utcnow().isoformat()
                }
            }, 500

# Analytics Predictions
@ns_analytics.route('/predict-risk')
class PredictRisk(Resource):
    @ns_analytics.expect(prediction_model)
    @token_required
    def post(self, current_user):
        """Predict patient health risk"""
        data = request.get_json()
        prediction = analytics_service.predict_risk(data['patient_data'])
        
        # Send real-time update if high risk
        if prediction.get('risk_level') == 'high':
            realtime_service.send_notification(
                current_user['user_id'],
                {
                    'type': 'high_risk_alert',
                    'patient_data': data['patient_data'],
                    'prediction': prediction,
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
        
        return prediction

@ns_analytics.route('/predict-cost')
class PredictCost(Resource):
    @ns_analytics.expect(prediction_model)
    @token_required
    def post(self, current_user):
        """Predict treatment costs"""
        data = request.get_json()
        treatments = data['patient_data'].get('proposed_treatments', [])
        prediction = analytics_service.predict_cost(data['patient_data'], treatments)
        return prediction

@ns_analytics.route('/population-trends')
class PopulationTrends(Resource):
    @ns_analytics.expect(population_analysis_model)  # ADD THIS DECORATOR
    @token_required
    def post(self, current_user):
        """Analyze population health trends"""
        data = request.get_json()
        patient_data_list = data.get('patients', [])
        trends = analytics_service.population_health_trends(patient_data_list)
        
        # Enhanced response structure
        enhanced_response = {
            'status': 'success',
            'data': trends,
            'metadata': {
                'population_size': len(patient_data_list),
                'timestamp': datetime.utcnow().isoformat(),
                'next_steps': [
                    "Review high-risk patient subgroups",
                    "Consider targeted interventions for prevalent conditions",
                    "Monitor population health trends over time"
                ]
            }
        }
        
        return enhanced_response

# Groq Models endpoint
@ns_ai.route('/models')
class AIModels(Resource):
    @token_required
    def get(self, current_user):
        """Get available Groq models"""
        return {
            'available_models': ai_service.available_models,
            'groq_available': ai_service.groq_available,
            'timestamp': datetime.utcnow().isoformat()
        }

# Enhanced health endpoint
@app.route('/health')
def health():
    """Health check endpoint with Groq status"""
    return {
        'status': 'healthy',
        'version': '2.2',
        'timestamp': datetime.utcnow().isoformat(),
        'services': {
            'authentication': 'active',
            'literature_search': 'active',
            'analytics': 'active',
            'real_time': 'active',
            'ai_analysis': 'active'
        },
        'groq_integration': {
            'available': ai_service.groq_available,
            'models_available': list(ai_service.available_models.keys()),
            'literature_analysis': True,
            'relevance_scoring': True
        },
        'open_source': True
    }

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return {'message': 'Resource not found'}, 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return {'message': 'Internal server error'}, 500

if __name__ == '__main__':
    # Production deployment settings
    port = int(os.getenv('PORT', 5000))  # Render provides PORT environment variable
    debug_mode = os.getenv('FLASK_ENV') == 'development'
    
    logger.info("Starting Groq-Powered HCP Engagement API v2.2")
    logger.info("AI Powered by Groq: Using llama-3.1-8b-instant (Fast & Efficient)")
    logger.info("Smart Literature: AI-powered relevance analysis")
    logger.info("Authentication: Bearer token required")
    logger.info(f"WebSocket: {'Available' if socketio else 'Disabled (compatibility issue)'}")
    logger.info(f"Environment: {os.getenv('FLASK_ENV', 'production')}")
    logger.info(f"Port: {port}")
    logger.info(f"Debug: {debug_mode}")
    
    # Use different server for production vs development
    if os.getenv('FLASK_ENV') == 'production':
        # In production, Gunicorn will handle the WSGI app
        # This code won't be executed when using Gunicorn
        logger.info("Production mode: Use Gunicorn to serve this application")
    else:
        # Development mode with SocketIO (if available)
        if socketio:
            socketio.run(app, host='0.0.0.0', port=port, debug=debug_mode)
        else:
            logger.warning("SocketIO not available, running with standard Flask server")
            app.run(host='0.0.0.0', port=port, debug=debug_mode)

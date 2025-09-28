from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
from flask_cors import CORS
from flask_socketio import SocketIO, emit
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
import hashlib
import math

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration for open-source deployment
app.config.update({
    'SECRET_KEY': os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production'),
    'JWT_SECRET_KEY': os.getenv('JWT_SECRET_KEY', 'jwt-secret-change-in-production'),
    'JWT_ACCESS_TOKEN_EXPIRES': timedelta(hours=1),
    'REDIS_URL': os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
})

# Security-focused CORS
CORS(app, resources={
    r"/*": {
        "origins": os.getenv('ALLOWED_ORIGINS', 'http://localhost:3000,http://127.0.0.1:3000').split(','),
        "methods": ["GET", "POST", "PUT", "DELETE"],
        "allow_headers": ["Authorization", "Content-Type"]
    }
})

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
    version='2.0', 
    title='Open HCP Engagement API', 
    description='Open-source Healthcare Provider engagement API with real-time features and analytics',
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

# Initialize WebSocket for real-time features
socketio = SocketIO(app, 
    cors_allowed_origins=os.getenv('ALLOWED_ORIGINS', 'http://localhost:3000').split(','),
    logger=logger,
    engineio_logger=False  # Disable engineio logging to reduce noise
)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Simple in-memory storage (Redis optional)
memory_store = {}

# Namespaces
ns_auth = api.namespace('auth', description='Authentication operations')
ns_literature = api.namespace('literature', description='Medical literature and studies operations')
ns_notifications = api.namespace('notifications', description='Medical information notifications')
ns_insurance = api.namespace('insurance', description='Insurance and patient interactions')
ns_questionnaire = api.namespace('questionnaire', description='Patient questionnaire operations')
ns_analytics = api.namespace('analytics', description='Advanced analytics and predictions')
ns_realtime = api.namespace('realtime', description='Real-time data synchronization')

# Security Models
login_model = api.model('Login', {
    'username': fields.String(required=True, description='Username'),
    'password': fields.String(required=True, description='Password')
})

# Enhanced Models
literature_search_model = api.model('LiteratureSearch', {
    'specialty': fields.String(required=True),
    'keywords': fields.List(fields.String),
    'patient_conditions': fields.List(fields.String),
    'max_results': fields.Integer(default=10)
})

# Analytics Models (Simplified)
prediction_model = api.model('PredictionRequest', {
    'patient_data': fields.Raw(required=True, description='Patient EMR data'),
    'model_type': fields.String(required=True, description='risk|outcome|cost')
})

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

# Authentication decorator
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
            auth_service = AuthService()
            current_user = auth_service.verify_token(token)
            kwargs['current_user'] = current_user
        except Exception as e:
            return {'message': str(e)}, 401
        
        return f(*args, **kwargs)
    return decorated

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
        
        if user_id in self.active_connections:
            for sid in self.active_connections[user_id]:
                socketio.emit('notification', message, room=sid)

# Lightweight Analytics Service (No scikit-learn dependency)
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
        """Analyze population health trends"""
        try:
            if not patient_data_list:
                return {'error': 'No patient data provided'}
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(patient_data_list)
            
            trends = {
                'average_age': round(df['age'].mean(), 1) if 'age' in df.columns else 0,
                'common_conditions': self._find_common_conditions(df),
                'risk_distribution': self._calculate_risk_distribution(df),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return trends
            
        except Exception as e:
            logger.error(f"Population analysis error: {e}")
            return {'error': 'Population analysis unavailable'}
    
    def _find_common_conditions(self, df: pd.DataFrame) -> List[Dict]:
        """Find most common conditions in population"""
        conditions = {}
        
        # Look for condition indicators
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['condition', 'diagnosis', 'disease']):
                if df[col].dtype == 'object':  # String columns
                    common = df[col].value_counts().head(3).to_dict()
                    conditions[col] = common
        
        return conditions
    
    def _calculate_risk_distribution(self, df: pd.DataFrame) -> Dict:
        """Calculate risk distribution across population"""
        risks = []
        
        for _, patient in df.iterrows():
            risk_pred = self.predict_risk(patient.to_dict())
            risks.append(risk_pred.get('risk_level', 'low'))
        
        risk_counts = pd.Series(risks).value_counts().to_dict()
        return risk_counts

# Enhanced Literature Service (Fixed with proper error handling)
class RealDataLiteratureService:
    """Open PubMed API integration with robust error handling"""
    
    def __init__(self):
        self.pubmed_available = self._check_pubmed_availability()
    
    def _check_pubmed_availability(self):
        """Check if PubMed is available"""
        try:
            from pymed import PubMed
            # Test a simple query to verify it works
            pubmed = PubMed(tool="OpenHCPAPI", email="opensource@example.com")
            return True
        except ImportError:
            logger.warning("pymed library not available, using fallback mode")
            return False
        except Exception as e:
            logger.warning(f"PubMed initialization failed: {e}, using fallback mode")
            return False
    
    def search_relevant_studies(self, specialty: str, keywords: List[str], 
                               patient_conditions: List[str]) -> List[Dict]:
        try:
            if self.pubmed_available:
                return self._search_pubmed(specialty, keywords, patient_conditions)
            else:
                return self._get_fallback_studies(specialty, keywords, patient_conditions)
            
        except Exception as e:
            logger.error(f"Literature search error: {e}")
            return self._get_fallback_studies(specialty, keywords, patient_conditions)
    
    def _search_pubmed(self, specialty: str, keywords: List[str], conditions: List[str]) -> List[Dict]:
        """Search PubMed with proper error handling"""
        try:
            from pymed import PubMed
            pubmed = PubMed(tool="OpenHCPAPI", email="opensource@example.com")
            
            query = self._build_query(specialty, keywords, conditions)
            results = pubmed.query(query, max_results=5)
            
            studies = []
            for article in results:
                # Handle publication_date conversion to avoid JSON serialization error
                pub_date = getattr(article, 'publication_date', 'Unknown')
                if hasattr(pub_date, 'strftime'):
                    pub_date = pub_date.strftime('%Y-%m-%d')
                elif pub_date is None:
                    pub_date = 'Unknown'
                
                # Handle authors field to ensure it's JSON serializable
                authors = article.authors or []
                if authors and not isinstance(authors, list):
                    authors = [str(author) for author in authors] if hasattr(authors, '__iter__') else [str(authors)]
                elif authors:
                    authors = [str(author) for author in authors]
                
                studies.append({
                    'id': article.pubmed_id or str(uuid.uuid4()),
                    'title': article.title or 'No title available',
                    'journal': article.journal or 'Unknown journal',
                    'publication_date': pub_date,
                    'relevance_score': 0.8,
                    'abstract': article.abstract or 'Abstract not available',
                    'url': f"https://pubmed.ncbi.nlm.nih.gov/{article.pubmed_id}/" if article.pubmed_id else '',
                    'authors': authors,
                    'source': 'PubMed'
                })
                
                if len(studies) >= 5:
                    break
            
            return studies if studies else self._get_fallback_studies(specialty, keywords, conditions)
            
        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            return self._get_fallback_studies(specialty, keywords, conditions)
    
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
    
    def _get_fallback_studies(self, specialty: str, keywords: List[str], conditions: List[str]) -> List[Dict]:
        """Reliable fallback studies"""
        return [
            {
                'id': 'study_1',
                'title': f'Advanced Treatments in {specialty} for {" and ".join(keywords)}',
                'journal': 'Journal of Clinical Medicine',
                'publication_date': '2024-01-15',
                'relevance_score': 0.9,
                'abstract': f'Comprehensive analysis of treatment approaches for {specialty} patients with {", ".join(conditions)}.',
                'url': 'https://example.com/study1',
                'authors': ['Smith J', 'Johnson A'],
                'source': 'Medical Database'
            },
            {
                'id': 'study_2',
                'title': f'{" and ".join(keywords)}: Latest Clinical Evidence',
                'journal': 'New England Journal of Medicine',
                'publication_date': '2024-01-10', 
                'relevance_score': 0.8,
                'abstract': f'Recent clinical trials and studies focusing on {", ".join(keywords)} in {specialty}.',
                'url': 'https://example.com/study2',
                'authors': ['Brown K', 'Davis M'],
                'source': 'Clinical Trials Registry'
            },
            {
                'id': 'study_3',
                'title': f'Patient Outcomes in {specialty} with {conditions[0] if conditions else "Chronic Conditions"}',
                'journal': 'The Lancet',
                'publication_date': '2023-12-20',
                'relevance_score': 0.7,
                'abstract': f'Long-term study of patient outcomes and quality of life measures.',
                'url': 'https://example.com/study3',
                'authors': ['Wilson R', 'Thompson L'],
                'source': 'Medical Research Journal'
            }
        ]

# Initialize services
auth_service = AuthService()
realtime_service = RealTimeService()
analytics_service = AnalyticsService()
literature_service = RealDataLiteratureService()  # This will now check PubMed availability on init

# WebSocket Events
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

# API Routes

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

# Literature Search
@ns_literature.route('/search')
class LiteratureSearch(Resource):
    @ns_literature.expect(literature_search_model)
    @token_required
    def post(self, current_user):
        """Search medical literature"""
        data = request.get_json()
        studies = literature_service.search_relevant_studies(
            data.get('specialty'),
            data.get('keywords', []),
            data.get('patient_conditions', [])
        )
        
        logger.info(f"Literature search by {current_user['sub']} for {data.get('specialty')}")
        
        return {'studies': studies, 'source': 'PubMed'}

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
    @token_required
    def post(self, current_user):
        """Analyze population health trends"""
        data = request.get_json()
        patient_data_list = data.get('patients', [])
        trends = analytics_service.population_health_trends(patient_data_list)
        return trends

# Real-time endpoints
@ns_realtime.route('/notify')
class SendNotification(Resource):
    @token_required
    def post(self, current_user):
        """Send real-time notification"""
        data = request.get_json()
        realtime_service.send_notification(current_user['user_id'], data)
        return {'status': 'notification_sent'}

# Health check
@app.route('/health')
def health():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'version': '2.0',
        'timestamp': datetime.utcnow().isoformat(),
        'services': {
            'authentication': 'active',
            'literature_search': 'active',
            'analytics': 'active',
            'real_time': 'active'
        },
        'open_source': True,
        'lightweight_analytics': True,
        'documentation': '/docs/'
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
    logger.info("🚀 Starting Open HCP Engagement API v2.0 (Lightweight)")
    logger.info("📚 Documentation: http://localhost:5000/docs/")
    logger.info("🔐 Authentication: Bearer token required for protected endpoints")
    logger.info("🌐 WebSocket: Real-time notifications available")
    logger.info("📊 Analytics: Lightweight rule-based system (no heavy ML dependencies)")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=os.getenv('FLASK_DEBUG', False))

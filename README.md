# Groq-Powered HCP Engagement API

[![API Version](https://img.shields.io/badge/version-2.2-blue.svg)](https://github.com/aandrx/hcp-engagement-api)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/flask-2.3.3-orange.svg)](https://flask.palletsprojects.com)
[![Groq AI](https://img.shields.io/badge/AI-Groq%20Powered-purple.svg)](https://groq.com)

A modern, AI-powered Healthcare Provider (HCP) engagement platform that combines medical literature search with advanced Groq AI analysis for enhanced clinical decision-making.

## Features

### AI-Powered Analysis
- **Groq Lightning-Fast AI**: Ultra-fast inference using Llama 3.1, Mixtral, and Gemma models
- **Intelligent Literature Analysis**: AI-powered relevance scoring and clinical implications
- **Multi-Model Support**: Choose from multiple Groq models based on your needs
- **Clinical Context Understanding**: Contextual analysis based on specialty and patient conditions

### Literature & Research
- **PubMed Integration**: Real-time access to medical literature via PubMed API
- **Smart Search**: Specialty-specific search with keyword and condition filtering
- **Relevance Scoring**: AI-driven relevance assessment for search results
- **Comprehensive Abstracts**: Detailed article summaries with publication data

### Healthcare Analytics
- **Risk Prediction**: Rule-based patient risk assessment
- **Cost Estimation**: Treatment cost prediction with complexity factors
- **Population Health**: Trend analysis across patient populations
- **Real-time Insights**: Live analytics with WebSocket notifications

### Enterprise Security
- **JWT Authentication**: Secure token-based authentication
- **Role-Based Access**: Provider and admin role management
- **CORS Protection**: Configurable cross-origin resource sharing
- **API Rate Limiting**: Built-in protection against abuse

### Real-Time Features
- **WebSocket Support**: Live notifications and updates
- **Event Streaming**: Real-time clinical alerts and insights
- **Multi-User Support**: Concurrent user session management

## Quick Start

### Prerequisites
- Python 3.8+
- Redis (optional, for caching)
- Groq API key (get one at [console.groq.com](https://console.groq.com))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/aandrx/hcp-engagement-api.git
   cd hcp-engagement-api
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Configure your environment**
   ```bash
   # Required: Add your Groq API key
   GROQ_API_KEY=your_groq_api_key_here
   
   # Security keys (generate secure random strings)
   SECRET_KEY=your_flask_secret_key_here_64_characters_long
   JWT_SECRET_KEY=your_jwt_secret_key_here_64_characters_long
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the API**
   - API Documentation: `http://localhost:5000/docs/`
   - Health Check: `http://localhost:5000/health`

## API Documentation

### Authentication

All protected endpoints require a Bearer token in the Authorization header.

#### Login
```http
POST /auth/login
Content-Type: application/json

{
  "username": "demo_provider",
  "password": "demo123"
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "user": {
    "username": "demo_provider",
    "role": "provider",
    "specialty": "Cardiology"
  }
}
```

### Literature Search with AI Analysis

Search medical literature with Groq AI-powered analysis:

```http
POST /literature/search
Authorization: Bearer YOUR_TOKEN
Content-Type: application/json

{
  "specialty": "Cardiology",
  "keywords": ["heart failure", "statins", "mortality reduction"],
  "patient_conditions": ["hypertension", "diabetes", "hyperlipidemia"],
  "max_results": 5,
  "enable_ai_analysis": true,
  "ai_model": "llama-3.1-8b-instant"
}
```

**Response:**
```json
{
  "studies": [
    {
      "id": "study_1",
      "title": "Advanced Cardiology Interventions for heart failure, statins, mortality reduction",
      "journal": "Journal of Clinical Medicine",
      "publication_date": "2024-01-15",
      "relevance_score": 0.9,
      "abstract": "This comprehensive study examines...",
      "url": "https://pubmed.ncbi.nlm.nih.gov/...",
      "authors": ["Smith J", "Johnson A"]
    }
  ],
  "ai_analysis": {
    "summary": "The reviewed articles provide strong evidence for...",
    "key_findings": [
      "Combination therapy reduces mortality by 30%",
      "ACE inhibitors show significant benefit in heart failure patients"
    ],
    "clinical_implications": [
      "Consider combination ACE inhibitor and beta-blocker therapy",
      "Monitor for contraindications in diabetic patients"
    ],
    "confidence_score": 0.85,
    "model_used": "llama-3.1-8b-instant"
  },
  "ai_capabilities": {
    "groq_available": true,
    "model_used": "llama-3.1-8b-instant",
    "models_available": [
      "llama-3.1-8b-instant",
      "llama-3.1-70b-versatile",
      "mixtral-8x7b-32768",
      "gemma2-9b-it"
    ]
  }
}
```

### Direct AI Analysis

Analyze any clinical text using Groq AI:

```http
POST /ai/analyze
Authorization: Bearer YOUR_TOKEN
Content-Type: application/json

{
  "text": "Heart failure patients with hypertension often benefit from ACE inhibitors and beta-blockers. Recent studies show combination therapy can reduce mortality by up to 30%.",
  "analysis_type": "clinical_implications",
  "model": "llama-3.1-8b-instant",
  "context": "Cardiology patient with heart failure and hypertension"
}
```

### Risk Prediction

Assess patient health risks using clinical data:

```http
POST /analytics/predict-risk
Authorization: Bearer YOUR_TOKEN
Content-Type: application/json

{
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
```

**Response:**
```json
{
  "risk_score": 0.78,
  "risk_level": "high",
  "risk_factors": ["hypertension", "diabetes", "hyperlipidemia", "obesity"],
  "confidence": 0.85,
  "method": "rule_based_analysis"
}
```

### Available AI Models

Get list of available Groq models:

```http
GET /ai/models
Authorization: Bearer YOUR_TOKEN
```

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `GROQ_API_KEY` | Your Groq API key | - | Yes |
| `SECRET_KEY` | Flask secret key | `dev-secret-key` | Yes |
| `JWT_SECRET_KEY` | JWT signing key | `jwt-secret` | Yes |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379/0` | No |
| `ALLOWED_ORIGINS` | CORS allowed origins | `http://localhost:3000` | No |
| `FLASK_DEBUG` | Enable debug mode | `False` | No |

### AI Model Selection

Choose the right Groq model for your use case:

- **llama-3.1-8b-instant**: Fast, efficient, good for general analysis
- **llama-3.1-70b-versatile**: More capable, better for complex medical reasoning
- **mixtral-8x7b-32768**: Balanced performance and speed
- **gemma2-9b-it**: Specialized for instruction following

## Testing

Run the comprehensive test suite:

```bash
# Start the API server first
python app.py

# In another terminal, run tests
python test_api.py
```

The test suite includes:
- Authentication testing
- Groq AI integration verification
- Literature search functionality
- Multiple AI model testing
- Real-time features validation

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   HCP API       │    │   External      │
│   (React/Vue)   │◄──►│   (Flask)       │◄──►│   Services      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                        │
                              │                        ├─ Groq AI API
                              │                        ├─ PubMed API
                              │                        └─ Redis Cache
                              │
                       ┌─────────────────┐
                       │   WebSocket     │
                       │   (Real-time)   │
                       └─────────────────┘
```

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
```

### Production Considerations

- Use a production WSGI server (Gunicorn, uWSGI)
- Set up Redis for improved caching
- Configure proper logging and monitoring
- Implement rate limiting and API quotas
- Use HTTPS in production
- Set strong secret keys

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Groq](https://groq.com) for ultra-fast AI inference
- [PubMed](https://pubmed.ncbi.nlm.nih.gov/) for medical literature access
- [Flask](https://flask.palletsprojects.com) for the web framework
- Medical community for inspiration and feedback

## Support

- Email: opensource@example.com
- Issues: [GitHub Issues](https://github.com/aandrx/hcp-engagement-api/issues)
- Documentation: [API Docs](http://localhost:5000/docs/)

---

**Built with care for the healthcare community**

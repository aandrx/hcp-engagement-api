# Groq-Powered HCP Engagement API

[![API Version](https://img.shields.io/badge/version-2.2-blue.svg)](https://github.com/aandrx/hcp-engagement-api)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/flask-2.3.3-orange.svg)](https://flask.palletsprojects.com)
[![Groq AI](https://img.shields.io/badge/AI-Groq%20Powered-purple.svg)](https://groq.com)

A comprehensive AI-powered Healthcare Provider (HCP) engagement platform that revolutionizes clinical decision-making through intelligent medical literature analysis, risk prediction, and population health insights powered by ultra-fast Groq AI models.

## What This API Does

This platform transforms how healthcare providers access and analyze medical information by combining:

- **Intelligent Literature Search**: Searches PubMed with AI-powered relevance analysis
- **Clinical Decision Support**: Provides actionable insights from medical research
- **Risk Assessment**: Predicts patient health risks using clinical parameters
- **Population Analytics**: Analyzes health trends across patient populations
- **Real-time AI Analysis**: Lightning-fast Groq AI processing for immediate insights

## Why It's Helpful

### For Healthcare Providers
- **Save Time**: Get AI-summarized literature reviews in seconds instead of hours
- **Improve Decisions**: Access evidence-based recommendations tailored to patient conditions
- **Reduce Risk**: Identify high-risk patients early with predictive analytics
- **Stay Current**: Access the latest medical research with contextual AI analysis

### For Healthcare Organizations
- **Population Insights**: Understand health trends across patient groups
- **Cost Optimization**: Predict treatment costs and resource allocation
- **Quality Improvement**: Evidence-based clinical protocols and guidelines
- **Real-time Monitoring**: Live alerts and notifications for critical insights

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

## Core Analysis Capabilities

### 1. AI-Powered Literature Search with Clinical Context

The API searches PubMed and provides intelligent analysis tailored to specific clinical scenarios:

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

**Detailed Analysis Response:**
```json
{
  "status": "success",
  "data": {
    "studies": [
      {
        "id": "study_12345",
        "title": "Combination ACE Inhibitor and Beta-Blocker Therapy in Heart Failure: A Meta-Analysis of Mortality Outcomes",
        "journal": "Journal of the American College of Cardiology",
        "publication_date": "2024-03-15",
        "relevance_score": 0.92,
        "abstract": "This meta-analysis of 15 randomized controlled trials involving 12,847 patients demonstrates that combination ACE inhibitor and beta-blocker therapy reduces all-cause mortality by 28% in heart failure patients with reduced ejection fraction...",
        "url": "https://pubmed.ncbi.nlm.nih.gov/38421567",
        "authors": ["Johnson M", "Smith K", "Williams R"]
      }
    ],
    "ai_analysis": {
      "summary": "The literature strongly supports combination ACE inhibitor and beta-blocker therapy for heart failure patients, particularly those with comorbid hypertension and diabetes. Evidence shows significant mortality reduction with proper monitoring.",
      "key_findings": [
        "Combination ACE inhibitor and beta-blocker therapy reduces all-cause mortality by 28% in heart failure patients",
        "Statins provide additional cardiovascular protection in diabetic patients with heart failure",
        "Early initiation of combination therapy within 30 days of diagnosis improves outcomes",
        "Patients with diabetes require closer monitoring for hyperkalemia with ACE inhibitors"
      ],
      "clinical_implications": [
        "Initiate combination ACE inhibitor (enalapril 10mg BID) and beta-blocker (metoprolol 25mg BID) therapy early",
        "Add moderate-intensity statin therapy (atorvastatin 40mg daily) for diabetic patients",
        "Monitor potassium levels every 2 weeks for first month, then monthly",
        "Consider patient-specific contraindications and drug interactions"
      ],
      "confidence_score": 0.89,
      "limitations": [
        "Most studies focused on heart failure with reduced ejection fraction",
        "Limited data on patients over 80 years old"
      ],
      "model_used": "llama-3.1-8b-instant"
    }
  },
  "metadata": {
    "search_timestamp": "2024-03-20T10:30:00Z",
    "total_studies_found": 127,
    "studies_analyzed": 5,
    "next_steps": [
      "Review current patient medications for contraindications",
      "Assess baseline kidney function and electrolyte levels",
      "Develop monitoring protocol for combination therapy"
    ],
    "source": "pubmed_groq_enhanced"
  }
}
```

### 2. Direct Clinical Text Analysis

Transform any clinical text into actionable insights using Groq AI:

```http
POST /ai/analyze
Authorization: Bearer YOUR_TOKEN
Content-Type: application/json

{
  "text": "65-year-old male with heart failure (EF 35%), hypertension, diabetes, and recent hospitalization for fluid overload. Current medications include furosemide 40mg daily. Patient reports shortness of breath on exertion and ankle swelling.",
  "analysis_type": "clinical_implications",
  "model": "llama-3.1-8b-instant",
  "context": "Cardiology consultation for heart failure management"
}
```

**Analysis Response:**
```json
{
  "status": "success",
  "data": {
    "analysis": "This patient presents with heart failure with reduced ejection fraction (HFrEF) and multiple comorbidities requiring evidence-based guideline-directed medical therapy. Key priorities include:\n\n1. **Medication Optimization**: Initiate ACE inhibitor (or ARB if ACE-intolerant) and beta-blocker therapy as first-line treatments. Consider starting with low doses and titrating upward.\n\n2. **Diuretic Management**: Current furosemide dose may need adjustment based on fluid status and kidney function.\n\n3. **Diabetes Management**: Ensure optimal glycemic control as diabetes accelerates cardiovascular disease progression.\n\n4. **Monitoring Requirements**: Regular assessment of kidney function, electrolytes, and symptoms.\n\n5. **Lifestyle Interventions**: Sodium restriction (<2g daily), daily weights, and appropriate activity level.",
    "analysis_type": "clinical_implications",
    "context": "Cardiology consultation for heart failure management"
  },
  "metadata": {
    "model_used": "llama-3.1-8b-instant",
    "timestamp": "2024-03-20T10:35:00Z",
    "groq_available": true,
    "next_steps": [
      "Review current medication list for contraindications",
      "Order baseline labs (BUN, creatinine, electrolytes, BNP)",
      "Schedule follow-up in 2 weeks for medication titration",
      "Refer to heart failure educator for lifestyle counseling"
    ]
  }
}

### 3. Comprehensive Risk Prediction

Assess cardiovascular and overall health risks using clinical parameters:

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
    "smoking": 1,
    "family_history": 1,
    "exercise_minutes_week": 60
  },
  "model_type": "cardiovascular_risk"
}
```

**Detailed Risk Assessment Response:**
```json
{
  "status": "success",
  "data": {
    "risk_score": 0.78,
    "risk_level": "high",
    "risk_category": "cardiovascular",
    "10_year_risk_percentage": 23.4,
    "risk_factors": [
      {
        "factor": "hypertension",
        "severity": "moderate",
        "contribution": 0.25
      },
      {
        "factor": "pre_diabetes", 
        "severity": "mild",
        "contribution": 0.18
      },
      {
        "factor": "hyperlipidemia",
        "severity": "moderate", 
        "contribution": 0.22
      },
      {
        "factor": "obesity",
        "severity": "moderate",
        "contribution": 0.13
      }
    ],
    "protective_factors": [
      {
        "factor": "regular_exercise",
        "benefit": "cardiovascular_protection"
      }
    ],
    "recommendations": [
      "Initiate antihypertensive therapy (ACE inhibitor or ARB)",
      "Start moderate-intensity statin therapy",
      "Implement lifestyle modifications for weight loss",
      "Monitor glucose levels and consider metformin",
      "Increase exercise to 150 minutes per week"
    ],
    "confidence": 0.89,
    "method": "enhanced_rule_based_analysis"
  },
  "metadata": {
    "timestamp": "2024-03-20T10:40:00Z",
    "next_followup": "3_months",
    "monitoring_required": [
      "Blood pressure monitoring twice weekly",
      "Lipid panel in 6 weeks",
      "HbA1c in 3 months"
    ]
  }
}

### 4. Population Health Analytics

Analyze health trends and patterns across patient populations:

```http
POST /analytics/population-trends
Authorization: Bearer YOUR_TOKEN
Content-Type: application/json

{
  "patients": [
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
      "glucose": 95,
      "cholesterol": 240,
      "bmi": 28,
      "smoking": 0,
      "conditions": ["hypertension"]
    }
  ]
}
```

**Population Analysis Response:**
```json
{
  "status": "success",
  "data": {
    "population_size": 4,
    "average_age": 60.0,
    "average_risk_score": 0.64,
    "risk_distribution": {
      "low": 1,
      "moderate": 1,
      "high": 2
    },
    "age_groups": {
      "45_54": 1,
      "55_64": 1,
      "65_74": 2
    },
    "risk_factors_prevalence": {
      "hypertension": {
        "count": 4,
        "prevalence": 100.0
      },
      "diabetes": {
        "count": 2,
        "prevalence": 50.0
      },
      "hyperlipidemia": {
        "count": 3,
        "prevalence": 75.0
      },
      "obesity": {
        "count": 2,
        "prevalence": 50.0
      }
    },
    "recommendations": [
      "Implement population-wide hypertension management protocol",
      "Develop diabetes prevention program for at-risk patients",
      "Consider cholesterol screening initiative"
    ]
  },
  "metadata": {
    "analysis_timestamp": "2024-03-20T10:45:00Z",
    "population_insights": [
      "High prevalence of cardiovascular risk factors",
      "Opportunity for preventive care interventions",
      "Need for comprehensive risk management protocols"
    ]
  }
}
```

### 5. Available AI Models

Get list of available Groq models and their capabilities:

```http
GET /ai/models
Authorization: Bearer YOUR_TOKEN
```

**AI Models Response:**
```json
{
  "groq_available": true,
  "models": {
    "llama-3.1-8b-instant": {
      "name": "Llama 3.1 8B Instant",
      "speed": "ultra_fast",
      "best_for": "Quick analysis, general medical queries",
      "max_tokens": 8192
    },
    "llama-3.1-70b-versatile": {
      "name": "Llama 3.1 70B Versatile", 
      "speed": "fast",
      "best_for": "Complex medical reasoning, detailed analysis",
      "max_tokens": 32768
    },
    "mixtral-8x7b-32768": {
      "name": "Mixtral 8x7B",
      "speed": "balanced",
      "best_for": "Multi-language support, diverse medical contexts",
      "max_tokens": 32768
    },
    "gemma2-9b-it": {
      "name": "Gemma 2 9B IT",
      "speed": "fast",
      "best_for": "Instruction following, structured analysis",
      "max_tokens": 8192
    }
  },
  "default_model": "llama-3.1-8b-instant",
  "recommendation": "Use llama-3.1-8b-instant for most clinical tasks, llama-3.1-70b-versatile for complex cases requiring detailed reasoning"
}

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

### AI Model Selection Guide

Choose the optimal Groq model based on your clinical use case:

| Model | Best For | Speed | Use Cases |
|-------|----------|-------|-----------|
| **llama-3.1-8b-instant** | General clinical analysis | Ultra-fast | Literature summaries, basic risk assessment, clinical notes analysis |
| **llama-3.1-70b-versatile** | Complex medical reasoning | Fast | Differential diagnosis, complex case analysis, detailed treatment planning |
| **mixtral-8x7b-32768** | Multi-specialty contexts | Balanced | Cross-specialty consultations, comprehensive reviews |
| **gemma2-9b-it** | Structured analysis | Fast | Guideline adherence checking, protocol development |

### Response Formats

The API supports two response formats:

- **Detailed Format** (default): Complete analysis with full context, recommendations, and metadata
- **Compact Format**: Streamlined responses for mobile applications and quick queries

Add `?format=compact` to any endpoint for compact responses:
```http
POST /literature/search?format=compact
```

## Testing & Validation

Run the comprehensive test suite to validate all AI-powered features:

```bash
# Start the API server first
python app.py

# In another terminal, run the comprehensive test suite
python test_api.py
```

### Test Coverage

The automated test suite validates:

- **Authentication & Security**: JWT token generation and validation
- **Groq AI Integration**: All available models and response quality
- **Literature Search**: PubMed integration with AI analysis
- **Risk Prediction**: Clinical parameter validation and risk scoring
- **Population Analytics**: Multi-patient analysis and trend identification
- **Real-time Features**: WebSocket connections and notifications
- **Error Handling**: Graceful degradation when AI services are unavailable

### Sample Test Output

```
Starting Groq-Powered HCP API Demo
============================================================

TEST: Groq Literature Search
============================================================
Testing COMPACT format:
PASS Status: success
Studies Found: 5
AI Confidence: 0.89
Summary: The literature strongly supports combination ACE inhibitor...

Testing DETAILED format:
PASS Status: success
Studies Found: 5
Key Findings:
 1. Combination therapy reduces mortality by 28%
 2. Early initiation improves outcomes
 3. Monitor for hyperkalemia in diabetic patients

TEST: Direct Groq Analysis
============================================================
Analysis Type: clinical_implications
Model Used: llama-3.1-8b-instant
Full Analysis Result:
This patient requires evidence-based heart failure management...

GROQ AI DEMO SUMMARY
============================================================
Tests Completed: 4
Successful: 4
Failed: 0
Success Rate: 100.0%

Features Demonstrated:
• Ultra-fast Groq AI inference
• Multiple response formats (compact/detailed)  
• Structured clinical insights
• Actionable next steps
• Multi-model analysis support
• Population health analytics
```

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

## Key Benefits Summary

### Clinical Impact
- **Evidence-Based Care**: Access to AI-analyzed medical literature for informed decisions
- **Risk Mitigation**: Early identification of high-risk patients through predictive analytics
- **Time Efficiency**: Reduce literature review time from hours to seconds
- **Quality Improvement**: Consistent, guideline-based recommendations

### Technical Advantages  
- **Ultra-Fast Processing**: Groq AI delivers sub-second response times
- **Scalable Architecture**: Handles multiple concurrent users and requests
- **Flexible Integration**: RESTful API with comprehensive documentation
- **Real-Time Capabilities**: WebSocket support for live notifications

### Data & Privacy
- **Secure Authentication**: JWT-based security with role-based access
- **No Patient Data Storage**: Processes data in memory without persistence
- **CORS Protection**: Configurable security for web applications
- **Audit Logging**: Comprehensive request and response logging

## Acknowledgments

- [Groq](https://groq.com) for revolutionary ultra-fast AI inference capabilities
- [PubMed](https://pubmed.ncbi.nlm.nih.gov/) for comprehensive medical literature access
- [Flask](https://flask.palletsprojects.com) for the robust web framework
- Healthcare providers and medical researchers for valuable feedback and requirements

## Support

- Email: opensource@example.com
- Issues: [GitHub Issues](https://github.com/aandrx/hcp-engagement-api/issues)
- Documentation: [API Docs](http://localhost:5000/docs/)

---

**Built with care for the healthcare community**

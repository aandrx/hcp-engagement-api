Groq-Powered HCP Engagement API
=================================

[![Issues](https://img.shields.io/github/issues/aandrx/hcp-engagement-api.svg)](https://github.com/aandrx/hcp-engagement-api/issues)
[![API Version](https://img.shields.io/badge/version-2.2-blue.svg)](https://github.com/aandrx/hcp-engagement-api)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/flask-2.3.3-orange.svg)](https://flask.palletsprojects.com)
[![Groq AI](https://img.shields.io/badge/AI-Groq%20Powered-purple.svg)](https://groq.com)

#### created for HackGT 12
#### build the demo here: [demo repo](https://github.com/aandrx/demo-hcp-engagement-api)

## Description:

A comprehensive AI-powered Healthcare Provider (HCP) engagement API that provides intelligent medical literature analysis, risk prediction, and population health insights powered by ultra-fast Groq AI models. Transform clinical decision-making with evidence-based recommendations in seconds.

### Basic Usage - Literature Search

```bash
curl -X POST https://your-api-domain.com/literature/search \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "specialty": "Cardiology",
    "keywords": ["heart failure", "statins"],
    "patient_conditions": ["hypertension"],
    "max_results": 3,
    "enable_ai_analysis": true
  }'
```

Response

```json
{
  "status": "success",
  "data": {
    "studies": [
      {
        "title": "Combination ACE Inhibitor and Beta-Blocker Therapy in Heart Failure",
        "journal": "Journal of the American College of Cardiology",
        "publication_date": "2024-03-15",
        "relevance_score": 0.92,
        "authors": ["Johnson M", "Smith K"]
      }
    ],
    "ai_analysis": {
      "summary": "The literature strongly supports combination therapy for heart failure patients with comorbid hypertension.",
      "key_findings": [
        "Combination therapy reduces mortality by 28%",
        "Early initiation improves outcomes"
      ],
      "confidence_score": 0.89
    }
  }
}
```

### Advanced Usage - Direct AI Analysis

Analyze any clinical text using Groq AI:

```bash
curl -X POST https://your-api-domain.com/ai/analyze \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "65-year-old male with heart failure (EF 35%), hypertension, diabetes",
    "analysis_type": "clinical_implications",
    "model": "llama-3.1-8b-instant"
  }'
```

Response

```json
{
  "status": "success",
  "data": {
    "analysis": "This patient requires evidence-based guideline-directed medical therapy. Key priorities include ACE inhibitor and beta-blocker therapy initiation.",
    "analysis_type": "clinical_implications"
  },
  "metadata": {
    "model_used": "llama-3.1-8b-instant",
    "next_steps": [
      "Review current medications for contraindications",
      "Order baseline labs (BUN, creatinine, electrolytes)"
    ]
  }
}
```

### Risk Prediction

Assess patient cardiovascular risk:

```bash
curl -X POST https://your-api-domain.com/analytics/predict-risk \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_data": {
      "age": 65,
      "systolic_bp": 150,
      "glucose": 130,
      "cholesterol": 260,
      "bmi": 32,
      "smoking": 1
    }
  }'
```

Response

```json
{
  "status": "success",
  "data": {
    "risk_score": 0.78,
    "risk_level": "high",
    "10_year_risk_percentage": 23.4,
    "risk_factors": [
      {
        "factor": "hypertension",
        "severity": "moderate",
        "contribution": 0.25
      }
    ],
    "recommendations": [
      "Initiate antihypertensive therapy",
      "Start moderate-intensity statin therapy"
    ]
  }
}
```

### AI Model Support

You can specify different Groq models for various use cases by using the `model` parameter:

```bash
curl -X POST https://your-api-domain.com/ai/analyze \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Complex clinical case requiring detailed analysis",
    "model": "llama-3.1-70b-versatile"
  }'
```

#### Supported AI Models

| Model | Best For | Speed | Use Cases |
|-------|----------|-------|-----------|
| `llama-3.1-8b-instant` | General analysis | Ultra-fast | Literature summaries, basic risk assessment |
| `llama-3.1-70b-versatile` | Complex reasoning | Fast | Differential diagnosis, detailed treatment planning |
| `mixtral-8x7b-32768` | Multi-specialty | Balanced | Cross-specialty consultations |
| `gemma2-9b-it` | Structured analysis | Fast | Guideline adherence, protocol development |

### Response Formats

You can request compact responses for mobile applications by adding the `format` parameter:

```bash
curl -X POST https://your-api-domain.com/literature/search?format=compact \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"specialty": "Cardiology", "keywords": ["heart failure"]}'
```

Response (Compact)

```json
{
  "status": "success",
  "data": {
    "study_count": 5,
    "key_findings": ["Combination therapy reduces mortality by 28%"],
    "confidence": 0.89
  }
}
```

### Authentication

All endpoints require authentication. First, obtain a token:

```bash
curl -X POST https://your-api-domain.com/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "demo_provider",
    "password": "demo123"
  }'
```

Response

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

### Documentation

<div align="center">
  <p>
    <a href="http://localhost:5000/docs/">
      <img width="200" height="52" src="https://img.shields.io/badge/Swagger_API_Docs-85EA2D.svg?logo=swagger&logoColor=black&style=for-the-badge" />
    </a>
  </p>
  <p>
    <a href="http://localhost:5000/health">
      <img width="200" height="52" src="https://img.shields.io/badge/Health_Check-00D9FF.svg?logo=statuspage&logoColor=white&style=for-the-badge" />
    </a>
  </p>
</div>

## Local Development

### Prerequisites

- Python 3.8+
- Groq API key (get one at [console.groq.com](https://console.groq.com))

### Installation

```bash
git clone https://github.com/aandrx/hcp-engagement-api.git
cd hcp-engagement-api
pip install -r requirements.txt
```

### Environment Setup

```bash
cp .env.example .env
# Edit .env with your configuration:
# GROQ_API_KEY=your_groq_api_key_here
```

### Basic Usage

```bash
python app.py
```

The API will be available at `http://localhost:5000`

### Testing

```bash
# Run comprehensive test suite
python test_api.py
```

## Key Features

- **Lightning-Fast AI**: Ultra-fast Groq AI inference (sub-second responses)
- **Medical Literature Search**: PubMed integration with AI-powered relevance analysis
- **Risk Prediction**: Cardiovascular and health risk assessment
- **Population Analytics**: Health trend analysis across patient populations
- **Multiple AI Models**: Choose from Llama 3.1, Mixtral, and Gemma models
- **Real-time Features**: WebSocket support for live notifications
- **Enterprise Security**: JWT authentication with role-based access

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/auth/login` | POST | Authenticate and get access token |
| `/literature/search` | POST | Search medical literature with AI analysis |
| `/ai/analyze` | POST | Direct AI analysis of clinical text |
| `/analytics/predict-risk` | POST | Patient risk prediction |
| `/analytics/population-trends` | POST | Population health analysis |
| `/ai/models` | GET | Available AI models |
| `/health` | GET | API health check |

### Dedication && Mission

<div align="center">
<p>This API is dedicated to healthcare providers worldwide who work tirelessly to improve patient outcomes through evidence-based medicine.</p>
  <p>Our mission is to democratize access to AI-powered clinical decision support, making advanced medical analysis available to healthcare providers regardless of their organization's size or resources.</p>
  
  <p>If you find this API helpful in your clinical practice, please consider:</p>
  <p><strong>Contributing to open healthcare initiatives</strong></p>
  <p><strong>Sharing feedback to improve clinical workflows</strong></p>
  <p><strong>Supporting medical education and research</strong></p>
  
  <p align="justify">Every API call represents a potential improvement in patient care. We believe that by providing healthcare providers with instant access to AI-analyzed medical literature and clinical insights, we can collectively raise the standard of care and improve health outcomes globally.</p>

</div>

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with care for the healthcare community**

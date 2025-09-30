# Environment Variables Setup for Render Deployment

This document lists all environment variables needed for deploying the HCP Engagement API to Render.

## Required Environment Variables

### 🔐 Security (Required)
```
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-key-here
```
**Note**: Render can auto-generate these using `generateValue: true` in render.yaml

### 🤖 AI Integration (Required for AI Features)
```
GROQ_API_KEY=your-groq-api-key-here
```
**How to get**: Sign up at [console.groq.com](https://console.groq.com) and create an API key

### 🌐 Application Settings
```
FLASK_ENV=production
PYTHON_VERSION=3.11.0
```

### 📊 Optional Services
```
REDIS_URL=redis://localhost:6379/0
```
**Note**: Only needed if you want to use Redis for caching. Can use Render's Redis addon.

## Setting Up in Render Dashboard

1. **Navigate to your service** in Render dashboard
2. **Go to Environment tab**
3. **Add each variable**:
   - Key: Variable name (e.g., `GROQ_API_KEY`)
   - Value: Your actual value
   - Type: Keep as "Plain text" for most values

## Getting Your Groq API Key

1. Visit [console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key and add it to your Render environment variables

## Security Best Practices

- ✅ Never commit API keys to GitHub
- ✅ Use Render's environment variables for secrets
- ✅ Let Render auto-generate SECRET_KEY and JWT_SECRET_KEY
- ✅ Rotate API keys periodically
- ❌ Don't use production keys in development

## Verification

After setting up environment variables, your API will:
- ✅ Start successfully without errors
- ✅ Pass the `/health` endpoint check
- ✅ Show Groq integration as "available" in health status
- ✅ Allow AI-powered literature analysis

## Troubleshooting

**API starts but Groq features don't work:**
- Check if GROQ_API_KEY is set correctly
- Verify the API key is valid at console.groq.com
- Check application logs for Groq API errors

**Authentication errors:**
- Ensure SECRET_KEY and JWT_SECRET_KEY are set
- They should be long, random strings (32+ characters)

**Service won't start:**
- Check if all required environment variables are set
- Review Render deployment logs for specific errors

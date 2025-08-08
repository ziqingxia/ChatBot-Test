# RAG System Streamlit Deployment Guide

## Overview
This guide explains how to deploy your RAG (Retrieval-Augmented Generation) chatbot system on Streamlit Cloud.

## Key Changes Made for Streamlit

### 1. **New Files Created**
- `streamlit_app.py` - Main Streamlit application
- `database_web.py` - Web-compatible database handler
- `requirements.txt` - Dependencies for deployment
- `DEPLOYMENT_GUIDE.md` - This guide

### 2. **Major Changes Required**

#### A. **Replace Command-Line Interface with Web UI**
- **Before**: `main.py` used `input()` and `print()` for interaction
- **After**: `streamlit_app.py` uses Streamlit widgets and session state

#### B. **Handle Session State**
- **Before**: Conversation state was lost between interactions
- **After**: Uses `st.session_state` to persist conversation across interactions

#### C. **CUDA/CPU Compatibility**
- **Before**: Hardcoded CUDA usage in `database.py`
- **After**: `database_web.py` automatically detects and uses available device

#### D. **File Path Handling**
- **Before**: Absolute paths in config
- **After**: Relative paths for web deployment

## Deployment Steps

### 1. **Prepare Your Code**

#### Update Configuration Paths
Edit `configs/config.yaml` to use relative paths:

```yaml
DATABASE:
  ROOT_PATH: "./database"  # Changed from absolute path
DICTIONARY:
  ROOT_PATH: "./dictionary"  # Changed from absolute path
REFINE_KNOWLEDGE:
  PHRASE_PATH: "./dictionary"  # Changed from absolute path
IN_CONTEXT:
  EXAMPLE_PATH: "./samples/conversations.json"  # Changed from absolute path
CHATBOT:
  HISTORY_PATH: "./history"  # Changed from absolute path
```

#### Update Streamlit App Import
In `streamlit_app.py`, change the import from:
```python
from database import RAGKnowledgeBase
```
to:
```python
from database_web import RAGKnowledgeBase
```

### 2. **Environment Setup**

#### Local Testing
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run streamlit_app.py
```

#### Streamlit Cloud Deployment
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set the main file path to `streamlit_app.py`
5. Add environment variables for your OpenAI API key

### 3. **Environment Variables**

#### For Streamlit Cloud
Add these secrets in your Streamlit Cloud dashboard:
```
OPENAI_API_KEY = "your-openai-api-key-here"
```

#### For Local Development
Create a `.streamlit/secrets.toml` file:
```toml
OPENAI_API_KEY = "your-openai-api-key-here"
```

### 4. **File Structure for Deployment**
```
your-repo/
├── streamlit_app.py          # Main Streamlit app
├── database_web.py           # Web-compatible database
├── chatbot.py               # Chatbot logic
├── fewshot.py               # Few-shot learning
├── prompts.py               # Prompt templates
├── configs/
│   └── config.yaml          # Configuration (with relative paths)
├── database/                # Your RAG database
├── dictionary/              # Your dictionary data
├── samples/                 # Sample conversations
├── history/                 # Chat history
├── requirements.txt         # Dependencies
└── README.md
```

## Key Features of the Streamlit App

### 1. **Event Selection**
- Dropdown menu to select conversation type
- Preview of event description and roles

### 2. **Chat Interface**
- Real-time chat with message history
- Styled message bubbles for user and assistant
- Conversation controls (new conversation, clear history)

### 3. **Database Management**
- File upload for adding new PDFs to database
- Real-time processing feedback

### 4. **Responsive Design**
- Mobile-friendly interface
- Custom CSS styling
- Sidebar for controls

## Troubleshooting

### Common Issues

#### 1. **CUDA Out of Memory**
- The web version automatically falls back to CPU
- Consider reducing batch sizes or model sizes

#### 2. **File Path Errors**
- Ensure all paths in `config.yaml` are relative
- Check that all required directories exist

#### 3. **OpenAI API Errors**
- Verify your API key is correct
- Check your OpenAI account has sufficient credits
- Ensure the API key has proper permissions

#### 4. **Memory Issues**
- Streamlit Cloud has memory limits
- Consider using smaller models or reducing data size
- Use `@st.cache_resource` for expensive operations

### Performance Optimization

#### 1. **Caching**
The app uses Streamlit's caching:
```python
@st.cache_resource
def load_config():
    # This will only run once per session
```

#### 2. **Lazy Loading**
- RAG system loads only when needed
- Database embeddings are cached

#### 3. **Error Handling**
- Graceful fallbacks for missing data
- User-friendly error messages

## Security Considerations

### 1. **API Key Security**
- Never commit API keys to version control
- Use environment variables or Streamlit secrets
- Consider using API key rotation

### 2. **File Upload Security**
- Validate file types and sizes
- Sanitize uploaded content
- Consider virus scanning for uploaded files

### 3. **Data Privacy**
- Be aware of what data is stored in session state
- Consider data retention policies
- Inform users about data usage

## Monitoring and Maintenance

### 1. **Logging**
- Add logging for debugging
- Monitor API usage and costs
- Track user interactions

### 2. **Updates**
- Regularly update dependencies
- Monitor for security vulnerabilities
- Keep OpenAI API version current

### 3. **Backup**
- Regularly backup your database and dictionary
- Version control your configuration
- Document any customizations

## Next Steps

1. **Test Locally**: Run `streamlit run streamlit_app.py` to test
2. **Deploy to Streamlit Cloud**: Follow the deployment steps above
3. **Customize**: Modify the UI and functionality as needed
4. **Scale**: Consider adding more features like user authentication, analytics, etc.

## Support

If you encounter issues:
1. Check the Streamlit documentation
2. Review the error logs in Streamlit Cloud
3. Test locally to isolate issues
4. Consider the troubleshooting section above 
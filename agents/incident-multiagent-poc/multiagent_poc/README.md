# Incident Management Assistant

A multi-agent system for incident management and resolution using Streamlit, Gemini AI, and ChromaDB.

## Overview

This application helps users manage and resolve incidents by:
- Analyzing incident descriptions
- Searching a knowledge base for similar issues
- Generating step-by-step resolution guides

The system uses multiple AI agents working together:
1. **Triage Agent**: Analyzes incidents and extracts severity and keywords
2. **Knowledge Base**: Stores and retrieves relevant documents
3. **Search Agent**: Finds external information (mocked in this version)
4. **Resolution Coach**: Generates step-by-step resolution guides

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root with your Google API key:
   ```
   GOOGLE_API_KEY='your_actual_gemini_api_key'
   ```

## Running the Application

To run the Streamlit application:

```bash
cd multiagent_poc
streamlit run streamlit_app.py
```

## Features

### Chat Interface
- Describe incidents in natural language
- Receive AI-generated resolution steps
- View chat history

### PDF Upload
- Upload PDF documents to the knowledge base
- Documents are automatically processed and indexed
- Knowledge from these documents is used to help resolve incidents

### Analysis View
- View detailed analysis of processed incidents
- See triage results, knowledge base matches, and resolution steps

## Architecture

The application consists of several components:

- **Streamlit UI**: User interface for interacting with the system
- **Agents**: AI-powered components that perform specific tasks
  - Triage Agent: Analyzes incidents
  - Resolution Coach: Generates resolution steps
- **Knowledge Base**: ChromaDB vector database for document storage and retrieval
- **PDF Processing**: Tools for extracting and indexing text from PDF documents

## Migration from Chainlit

This application was migrated from a Chainlit-based UI to Streamlit. The migration preserves all functionality while adding:
- Improved UI with tabs and visual elements
- Better progress indicators
- Enhanced error handling
- More detailed analysis view
- Improved chat history management

## Requirements

- Python 3.8+
- Streamlit
- Google Generative AI (Gemini)
- ChromaDB
- PyPDF
- Phidata
- Pydantic
- Python-dotenv
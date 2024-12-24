
# LLAMA RAG and Chat Application

An intelligent document analysis and chat application powered by LLAMA models.

## Features
- Document Q&A (RAG) with PDF and Excel support
- Interactive chat interface
- Multiple LLAMA model support
- Customizable response behaviors
- Streaming responses
- Context-aware conversations

## Setup
1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Create a `.env` file with your Groq API key:
```bash
GROQ_API_KEY=your_api_key_here
```
4. Run the application:
```bash
streamlit run Rag.py
```

## Environment Variables
- `GROQ_API_KEY`: Your Groq API key (required)

## Usage
1. Select mode (RAG or Chat) from the sidebar
2. For RAG mode:
   - Upload a PDF or Excel file
   - Ask questions about the document
3. For Chat mode:
   - Simply start chatting with the model
4. Adjust model and behavior settings in the sidebar

## Requirements
- Python 3.8+
- See requirements.txt for package dependencies

# 📄 PDF RAG Assistant

A modern, streamlined RAG (Retrieval-Augmented Generation) application for intelligent PDF document analysis powered by LangChain and Groq.

## ✨ Features

- **📄 PDF-Only Focus**: Streamlined interface for PDF document analysis
- **🤖 Modern LangChain**: Built with LangChain Expression Language (LCEL) for optimal performance
- **🎨 Enhanced UI/UX**: Clean, intuitive Streamlit interface with modern design
- **⚡ Fast Processing**: Optimized document chunking and vector search
- **🔍 Smart Retrieval**: Advanced similarity search with source citations
- **🔧 Configurable**: Adjustable model parameters and advanced settings

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd LLama-rag-chat

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file in the project root:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Run the Application

```bash
streamlit run Rag.py
```

## 🎯 How to Use

1. **Upload PDF**: Click "Choose a PDF file" and select your document
2. **Process Document**: Click "Process Document" to analyze and index the PDF
3. **Ask Questions**: Type your questions about the document content
4. **Get Answers**: Receive AI-powered responses with source citations

## 🔧 Configuration Options

### Models Available
- **llama-3.1-8b-instant** (Meta): Fast 8B parameter model for quick responses
- **llama-3.3-70b-versatile** (Meta): Most capable 70B parameter model for complex reasoning
- **meta-llama/llama-guard-4-12b** (Meta): Specialized 12B parameter safety and content moderation model
- **openai/gpt-oss-120b** (OpenAI): Large 120B parameter open-source model for advanced tasks
- **openai/gpt-oss-20b** (OpenAI): Balanced 20B parameter model for performance and efficiency

All models support up to 131,072 tokens context window for processing long documents.

### Advanced Settings
- **Temperature**: Control response creativity (0.0 = focused, 1.0 = creative)
- **Chunk Overlap**: Adjust text processing overlap for better context

## 🏗️ Architecture

### Modern LangChain Implementation
- **LCEL (LangChain Expression Language)**: Modern chain composition
- **Retrieval Chains**: Optimized document retrieval and answer generation
- **Vector Stores**: FAISS for efficient similarity search
- **Embeddings**: HuggingFace sentence transformers for semantic understanding

### UI/UX Improvements
- **Responsive Layout**: Two-column design for optimal workflow
- **Visual Feedback**: Progress indicators, success/error messages
- **Session Management**: Persistent chat history and document state
- **Modern Components**: Enhanced file upload, collapsible settings

## 📋 Requirements

- Python 3.8+
- Internet connection (for model downloads)
- Groq API key

## 🔒 API Keys

This application requires a Groq API key for LLM access. Sign up at [Groq](https://groq.com/) to get your free API key.

## 🎨 Design Philosophy

- **Simplicity**: Focus on PDF RAG without unnecessary features
- **Performance**: Optimized for speed and accuracy
- **User Experience**: Intuitive interface with clear visual hierarchy
- **Modularity**: Clean, maintainable code architecture

## 🚧 What's Changed

### Removed Features
- ❌ Chat mode (standalone conversation)
- ❌ Excel file support
- ❌ Legacy Groq client implementation

### Added Features
- ✅ Modern LangChain LCEL patterns
- ✅ Enhanced Streamlit UI/UX
- ✅ Better error handling and user feedback
- ✅ Improved document processing pipeline
- ✅ Source citation system

## 🤝 Contributing

Feel free to open issues or submit pull requests to improve the application.

## 📄 License

This project is open source. Feel free to use and modify as needed.
import os
import tempfile
from typing import List, Optional
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# LangChain imports with modern patterns
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
load_dotenv()

# Modern Streamlit page configuration
st.set_page_config(
    page_title="📄 PDF RAG Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
AVAILABLE_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "meta-llama/llama-guard-4-12b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b"
]

# Model information for UI
MODEL_INFO = {
    "llama-3.1-8b-instant": {
        "max_tokens": 131072,
        "description": "Meta LLAMA 3.1 with 8B parameters. Fast and efficient for quick responses.",
        "developer": "Meta"
    },
    "llama-3.3-70b-versatile": {
        "max_tokens": 131072,
        "description": "Meta LLAMA 3.3 with 70B parameters. Most capable model for complex reasoning.",
        "developer": "Meta"
    },
    "meta-llama/llama-guard-4-12b": {
        "max_tokens": 131072,
        "description": "Meta LLAMA Guard 4 with 12B parameters. Specialized safety and content moderation model.",
        "developer": "Meta"
    },
    "openai/gpt-oss-120b": {
        "max_tokens": 131072,
        "description": "OpenAI GPT OSS with 120B parameters. Large open-source model for advanced tasks.",
        "developer": "OpenAI"
    },
    "openai/gpt-oss-20b": {
        "max_tokens": 131072,
        "description": "OpenAI GPT OSS with 20B parameters. Balanced performance and efficiency.",
        "developer": "OpenAI"
    }
}

class ModernPDFRAG:
    """Modern PDF RAG application using LangChain Expression Language (LCEL)"""
    
    def __init__(self):
        self.initialize_session_state()
        self.setup_embeddings()
        
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = None
        if 'processed_files' not in st.session_state:
            st.session_state.processed_files = []
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
    def setup_embeddings(self):
        """Setup embedding model with error handling"""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            st.error(f"Failed to load embeddings model: {str(e)}")
            st.error("Please check your internet connection.")
            self.embeddings = None
    
    def process_pdf(self, uploaded_file) -> bool:
        """Process uploaded PDF file using modern LangChain patterns"""
        if not self.embeddings:
            st.error("Embeddings model not available.")
            return False
            
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name
            
            # Load PDF using LangChain
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()
            
            # Modern text splitting with optimal parameters
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            splits = text_splitter.split_documents(documents)
            
            # Create vector store
            vector_store = FAISS.from_documents(splits, self.embeddings)
            
            # Store in session state
            st.session_state.vector_store = vector_store
            st.session_state.processed_files.append(uploaded_file.name)
            
            # Cleanup
            os.unlink(temp_file_path)
            
            return True
            
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            return False
    
    def create_rag_chain(self, model_name: str, temperature: float = 0.1):
        """Create RAG chain using modern LangChain patterns"""
        if not GROQ_API_KEY:
            st.error("Please set your GROQ_API_KEY in the .env file")
            return None
            
        if not st.session_state.vector_store:
            st.warning("Please upload and process a PDF first.")
            return None
        
        # Initialize LLM
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=model_name,
            temperature=temperature,
            max_tokens=4096  # Increased for better responses with new models
        )
        
        # Create retriever
        retriever = st.session_state.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        # Create prompt template
        system_prompt = """You are a helpful AI assistant that answers questions based on the provided PDF document context.

Use the following pieces of context to answer the user's question. If you don't know the answer based on the context, just say that you don't have enough information to answer the question.

Context: {context}

Question: {input}

Provide a clear, concise, and helpful answer based on the context above."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        # Create document processing chain
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        
        # Create full RAG chain
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        return rag_chain
    
    def format_sources(self, response):
        """Format source documents for display"""
        if 'context' not in response:
            return ""
            
        sources = []
        for i, doc in enumerate(response['context'], 1):
            metadata = doc.metadata
            page = metadata.get('page', 'Unknown')
            source = metadata.get('source', 'Unknown')
            sources.append(f"**Source {i}:** {Path(source).name}, Page {page + 1}")
        
        return "\n\n**📚 Sources:**\n" + "\n".join(sources) if sources else ""
    
    def render_modern_ui(self):
        """Render modern Streamlit UI with improved design"""
        
        # Check API key
        if not GROQ_API_KEY:
            st.error("🔑 GROQ_API_KEY not found. Please add it to your .env file.")
            st.code("GROQ_API_KEY=your_groq_api_key_here")
            st.stop()
        
        # Header with modern styling
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="color: #2E8B57; margin-bottom: 0.5rem;">📄 PDF RAG Assistant</h1>
            <p style="color: #666; font-size: 1.1rem;">Ask questions about your PDF documents with AI</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar configuration
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Model selection
            model_name = st.selectbox(
                "🤖 Select Model",
                AVAILABLE_MODELS,
                help="Choose the AI model for processing your questions",
                format_func=lambda x: f"{MODEL_INFO[x]['developer']}: {x.split('/')[-1] if '/' in x else x}"
            )
            
            # Display model info
            if model_name in MODEL_INFO:
                st.info(f"📋 {MODEL_INFO[model_name]['description']}")
                st.caption(f"🔧 Developer: {MODEL_INFO[model_name]['developer']} | 🎯 Context: {MODEL_INFO[model_name]['max_tokens']:,} tokens")
            
            st.divider()
            
            # Advanced settings in expander
            with st.expander("🔧 Advanced Settings"):
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1,
                    step=0.1,
                    help="Controls randomness in responses. Lower = more focused, Higher = more creative"
                )
                
                chunk_overlap = st.slider(
                    "Chunk Overlap",
                    min_value=50,
                    max_value=300,
                    value=200,
                    step=50,
                    help="Overlap between text chunks for better context"
                )
            
            st.divider()
            
            # Clear session button
            if st.button("🗑️ Clear Session", use_container_width=True):
                st.session_state.vector_store = None
                st.session_state.processed_files = []
                st.session_state.chat_history = []
                st.rerun()
        
        # Main content area
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📤 Upload PDF")
            
            # File upload with modern styling
            uploaded_file = st.file_uploader(
                "Choose a PDF file",
                type=['pdf'],
                help="Upload a PDF document to analyze"
            )
            
            if uploaded_file:
                # Display file info
                file_size = uploaded_file.size / (1024 * 1024)  # Convert to MB
                st.success(f"✅ **{uploaded_file.name}**")
                st.caption(f"📊 Size: {file_size:.2f} MB")
                
                # Process button
                if st.button("🔄 Process Document", use_container_width=True, type="primary"):
                    with st.spinner("🔍 Processing PDF..."):
                        if self.process_pdf(uploaded_file):
                            st.success("✅ PDF processed successfully!")
                            st.balloons()
                        else:
                            st.error("❌ Failed to process PDF. Please try again.")
            
            # Show processed files
            if st.session_state.processed_files:
                st.subheader("📋 Processed Files")
                for i, filename in enumerate(st.session_state.processed_files, 1):
                    st.write(f"{i}. {filename}")
        
        with col2:
            st.subheader("💬 Ask Questions")
            
            # Question input
            question = st.text_input(
                "Your Question",
                placeholder="What is this document about?",
                help="Type your question about the uploaded PDF"
            )
            
            # Ask button
            if st.button("🔍 Ask Question", use_container_width=True, type="primary"):
                if not question.strip():
                    st.warning("⚠️ Please enter a question.")
                elif not st.session_state.vector_store:
                    st.warning("⚠️ Please upload and process a PDF first.")
                else:
                    self.handle_question(question, model_name, temperature)
            
            # Display chat history
            if st.session_state.chat_history:
                st.subheader("📜 Chat History")
                
                # Reverse order to show newest first
                for i, (q, a) in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
                    with st.expander(f"Q{len(st.session_state.chat_history)-i}: {q[:50]}..."):
                        st.write("**Question:**", q)
                        st.write("**Answer:**", a)
    
    def handle_question(self, question: str, model_name: str, temperature: float):
        """Handle user question with modern RAG chain"""
        try:
            # Create RAG chain
            rag_chain = self.create_rag_chain(model_name, temperature)
            
            if not rag_chain:
                return
            
            # Process question
            with st.spinner("🤔 Thinking..."):
                response = rag_chain.invoke({"input": question})
                
                # Display answer
                st.markdown("### 🤖 Answer")
                st.write(response['answer'])
                
                # Display sources
                sources = self.format_sources(response)
                if sources:
                    st.markdown(sources)
                
                # Add to chat history
                st.session_state.chat_history.append((question, response['answer']))
                
        except Exception as e:
            st.error(f"❌ Error processing question: {str(e)}")
            st.error("Please try again or check your API key.")

def main():
    """Main application entry point"""
    app = ModernPDFRAG()
    app.render_modern_ui()

if __name__ == "__main__":
    main()
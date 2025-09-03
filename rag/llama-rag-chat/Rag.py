import os
from dotenv import load_dotenv
import streamlit as st
import httpx
import tempfile
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
# Load environment variables
load_dotenv()
# Display beta development notice
st.warning("This application is currently in beta phase. Some features may be experimental.")

# Constants
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

AVAILABLE_MODELS = ["llama3-70b-8192", "llama-3.3-70b-versatile"]

# Initialize Groq client only if API key is available
client = None
if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)

# Define model info and behavior presets
model_info = {
    "llama3-70b-8192": {
        "max_tokens": 8192,
        "description": "LLAMA 3.0 with 70 billion parameters and 8192 max tokens. Suitable for large-scale tasks."
    },
    "llama-3.3-70b-versatile": {
        "max_tokens": 4096,
        "description": "LLAMA 3.3 with 70 billion parameters. Versatile model for a variety of tasks."
    },
    "llama-3.2-90b-vision-preview": {
        "max_tokens": 2048,
        "description": "LLAMA 3.2 with 90 billion parameters. Preview model with vision capabilities."
    },
    "llama-3.1-8b-instant": {
        "max_tokens": 1024,
        "description": "LLAMA 3.1 with 8 billion parameters. Instant model for quick responses."
    }
}

behavior_presets = {
    "Creative": {
        "temperature": 0.9,
        "top_p": 0.9,
        "description": "More creative and diverse responses"
    },
    "Balanced": {
        "temperature": 0.5,
        "top_p": 0.7,
        "description": "Balance between creativity and accuracy"
    },
    "Precise": {
        "temperature": 0.2,
        "top_p": 0.3,
        "description": "More focused and deterministic responses"
    }
}

class PDFQA:
    def __init__(self):
        self.initialize_session_state()
        # Try to use real embeddings, fall back to graceful error handling
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
        except Exception as e:
            st.error(f"Failed to load embeddings model: {str(e)}")
            st.error("Please check your internet connection. RAG functionality will be limited.")
            st.info("You can still use Chat mode without document upload.")
            # Create a mock embeddings for basic functionality
            self.embeddings = None

    @staticmethod
    def initialize_session_state():
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = None
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        if 'clear_input' not in st.session_state:
            st.session_state.clear_input = False

    def process_file(self, file, file_type):
        file_type = file_type.lower()
        
        # Determine correct file extension based on uploaded file
        if file_type == "pdf":
            extension = "pdf"
        else:  # Excel files
            extension = "xlsx" if file.name.endswith('.xlsx') else "xls"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name

            if not temp_file_path or not os.path.exists(temp_file_path):
                st.error("Temporary file path is invalid or does not exist.")
                return False

            try:
                if file_type == "pdf":
                    loader = PyPDFLoader(temp_file_path)
                    documents = loader.load()
                elif file_type == "excel":
                    # Use pandas instead of UnstructuredExcelLoader
                    documents = self._process_excel_with_pandas(temp_file_path)
                else:
                    st.error(f"Unsupported file type: {file_type}")
                    return False

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,  # Reduced chunk size for better context
                    chunk_overlap=100  # Increased overlap for better context continuity
                )
                texts = text_splitter.split_documents(documents)
                
                if not self.embeddings:
                    st.error("Embeddings model not available. Cannot create vector store.")
                    return False
                    
                st.session_state.vector_store = FAISS.from_documents(texts, self.embeddings)
                return True

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                return False
            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

    def _process_excel_with_pandas(self, file_path):
        """Process Excel file using pandas and create clean text sections"""
        from langchain.schema import Document
        
        try:
            # Read all sheets
            excel_data = pd.read_excel(file_path, sheet_name=None)
            documents = []
            
            for sheet_name, df in excel_data.items():
                # Create clean text representation
                text_parts = [f"Sheet: {sheet_name}"]
                
                # Add column headers
                if not df.empty:
                    headers = df.columns.tolist()
                    text_parts.append(f"Columns: {', '.join(str(h) for h in headers)}")
                    
                    # Add row data with context
                    for idx, row in df.iterrows():
                        row_text = []
                        for col, value in row.items():
                            if pd.notna(value):
                                row_text.append(f"{col}: {value}")
                        if row_text:
                            text_parts.append(f"Row {idx + 1}: {'; '.join(row_text)}")
                
                # Create document with metadata
                content = "\n".join(text_parts)
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": file_path,
                        "sheet": sheet_name,
                        "type": "excel"
                    }
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            st.error(f"Error processing Excel file with pandas: {str(e)}")
            return []

    def validate_file(self, file, file_type):
        """Validate file size and type"""
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        
        # Check file size
        if file.size > MAX_FILE_SIZE:
            st.error(f"File size ({file.size / (1024*1024):.1f}MB) exceeds maximum allowed size (10MB).")
            return False
        
        # Check file type
        if file_type.lower() == "pdf" and not file.name.lower().endswith('.pdf'):
            st.error("Please upload a valid PDF file.")
            return False
        elif file_type.lower() == "excel" and not (file.name.lower().endswith('.xlsx') or file.name.lower().endswith('.xls')):
            st.error("Please upload a valid Excel file (.xlsx or .xls).")
            return False
        
        return True

    def generate_response(self, query, model_name, temperature=0.1, top_p=0.7, max_tokens=1024, enable_streaming=False):
        try:
            if not st.session_state.vector_store:
                return "Please upload and process a file first."
            
            if not self.embeddings:
                return "Embeddings model not available. Please check your internet connection and restart the application."

            relevant_docs = st.session_state.vector_store.similarity_search(query, k=5)
            context = "\n".join([doc.page_content for doc in relevant_docs])
            
            prompt = f"""
            Based on the following context, please answer the question.
            
            Context:
            {context}

            Question: {query}

            Provide a clear and concise answer based on the context provided.
            """
            
            messages = [{"role": "user", "content": prompt}]
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=enable_streaming
            )

            if enable_streaming:
                response = ""
                for chunk in completion:
                    if chunk.choices[0].delta.content:
                        response += chunk.choices[0].delta.content
            else:
                response = completion.choices[0].message.content
            
            # Add citations and sources
            sources = []
            for i, doc in enumerate(relevant_docs, 1):
                metadata = doc.metadata
                source_info = f"Source {i}: "
                if 'source' in metadata:
                    source_info += f"File: {os.path.basename(metadata['source'])}"
                if 'sheet' in metadata:
                    source_info += f", Sheet: {metadata['sheet']}"
                if 'page' in metadata:
                    source_info += f", Page: {metadata['page']}"
                sources.append(source_info)
            
            if sources:
                response += f"\n\n**Sources:**\n" + "\n".join(sources)
            
            return response

        except Exception as e:
            return f"Error generating response: {str(e)}"

    def get_completion(self, messages, model_name, stream=True):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                stream=stream,
                stop=None,
            )
            
            if stream:
                return completion  # Return the stream object directly
            else:
                return completion.choices[0].message.content
        except Exception as e:
            return f"An error occurred: {str(e)}"

    def handle_submit(self):
        if st.session_state.user_input.strip():
            current_input = st.session_state.user_input
            
            # Store the current input in conversation history
            st.session_state.conversation_history.append({"role": "user", "content": current_input})
            
            with st.spinner("Thinking..."):
                # Limit the conversation history to the last 10 messages for context
                limited_history = st.session_state.conversation_history[-10:]
                
                if self.enable_streaming:
                    # Create a single placeholder for streaming response
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    # Stream the response
                    for chunk in self.get_completion(limited_history, self.model_name, stream=True):
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            # Update the placeholder with current response
                            message_placeholder.markdown(f"**Assistant:** {full_response}")
                    
                    # Add the full response to conversation history
                    st.session_state.conversation_history.append({"role": "assistant", "content": full_response})
                    # Clear the placeholder after adding to history
                    message_placeholder.empty()
                else:
                    # Non-streaming response
                    response = self.get_completion(limited_history, self.model_name, stream=False)
                    st.session_state.conversation_history.append({"role": "assistant", "content": response})
            
            # Set a flag to clear input on next rerun
            st.session_state.clear_input = True

    def render_ui(self):
        # Check for API key first
        if not GROQ_API_KEY:
            st.error("GROQ_API_KEY not found in environment variables. Please set your API key.")
            st.stop()
        
        global client
        if not client:
            client = Groq(api_key=GROQ_API_KEY)
        

        # Sidebar settings
        with st.sidebar:
            st.header("Settings")
            mode = st.radio("Select Mode", ["RAG", "Chat"])  # Move the mode selection here
            self.model_name = st.selectbox("Select Model", AVAILABLE_MODELS)
            
            # Add streaming toggle
            self.enable_streaming = st.checkbox("Enable Streaming", value=True)
            
            if st.button("Clear Conversation", key="clear_sidebar"):
                st.session_state.conversation_history = []
                st.session_state.vector_store = None
            
            # Display model description
            st.info(model_info[self.model_name]['description'])
            
            # Add behavior preset selector
            st.markdown("---")
            st.subheader("Response Behavior")
            selected_behavior = st.select_slider(
                "Select Response Style",
                options=list(behavior_presets.keys()),
                value="Balanced"
            )
            
            # Show behavior description
            st.caption(behavior_presets[selected_behavior]["description"])
            
            # Add controls for temperature and other supported parameters
            st.markdown("---")
            st.subheader("Advanced Parameters")
            # Initialize parameters with preset values but allow manual override
            self.temperature = st.slider("Temperature", 0.0, 1.0, 
                              value=behavior_presets[selected_behavior]["temperature"])
            self.top_p = st.slider("Top P", 0.0, 1.0, 
                              value=behavior_presets[selected_behavior]["top_p"])
            self.max_tokens = st.slider("Max Tokens", 1, model_info[self.model_name]['max_tokens'], 
                              model_info[self.model_name]['max_tokens'])

        # Set the title based on the selected mode
        if mode == "RAG":
            st.title("Insight in Your Document")
        else:
            st.title("LLAMA Chat")

        if mode == "RAG":
            file_type = st.radio("Select file type", ["PDF", "Excel"], horizontal=True)
            
            accepted_types = ["pdf"] if file_type == "PDF" else ["xlsx", "xls"]
            
            uploaded_file = st.file_uploader(
                f"Upload a {file_type} file", 
                type=accepted_types
            )

            if uploaded_file and st.button("Process Document"):
                if self.validate_file(uploaded_file, file_type):
                    with st.spinner(f"Processing {file_type}..."):
                        if self.process_file(uploaded_file, file_type.lower()):
                            st.success(f"File processed successfully!")
                        # Reset file position for reprocessing
                        uploaded_file.seek(0)

            user_input = st.text_input("Enter your question:", key="user_input")
            
            if st.button("Send") and user_input:
                with st.spinner("Generating response..."):
                    response = self.generate_response(
                        user_input, 
                        self.model_name, 
                        self.temperature, 
                        self.top_p, 
                        self.max_tokens, 
                        self.enable_streaming
                    )
                    st.markdown("**Answer:**\n" + response)

        elif mode == "Chat":
            # Create form for better Enter key handling
            with st.form(key="chat_form", clear_on_submit=True):
                # Create two columns for input with better alignment
                col1, col2 = st.columns([10, 1])
                
                with col1:
                    if "user_input" not in st.session_state:
                        st.session_state.user_input = ""
                    
                    if st.session_state.clear_input:
                        st.session_state.user_input = ""
                        st.session_state.clear_input = False
                    
                    user_input = st.text_input(
                        "Message",  # Added label
                        key="user_input",
                        value=st.session_state.user_input,
                        placeholder="Type your message here...",
                        label_visibility="collapsed"
                    )
                
                with col2:
                    submit_button = st.form_submit_button(
                        "➤",
                        use_container_width=True
                    )

            # Handle form submission
            if submit_button and user_input.strip():
                self.handle_submit()

            # Chat container and rest of the display code
            chat_container = st.container()

            # Display conversation history in chronological order (newest at bottom)
            with chat_container:
                for message in st.session_state.conversation_history:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])

if __name__ == "__main__":
    pdf_qa_app = PDFQA()
    pdf_qa_app.render_ui()
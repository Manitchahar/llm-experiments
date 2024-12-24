import os
from dotenv import load_dotenv
import streamlit as st
import httpx
import tempfile
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from unstructured.partition.xlsx import partition_xlsx

# ...existing code...

def get_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except KeyError:
        return os.getenv("GROQ_API_KEY")

# Load environment variables
load_dotenv()
# Display beta development notice
st.warning("This application is currently in beta phase. Some features may be experimental.")

# Constants
GROQ_API_KEY = get_api_key()
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables or Streamlit secrets")

AVAILABLE_MODELS = ["llama3-70b-8192", "llama-3.3-70b-versatile"]
MAX_HISTORY_LENGTH = 5

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)
client._client._transport = httpx.HTTPTransport(verify=False)

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
        "top_k": 80,
        "description": "More creative and diverse responses"
    },
    "Balanced": {
        "temperature": 0.5,
        "top_p": 0.7,
        "top_k": 50,
        "description": "Balance between creativity and accuracy"
    },
    "Precise": {
        "temperature": 0.2,
        "top_p": 0.3,
        "top_k": 20,
        "description": "More focused and deterministic responses"
    }
}

class PDFQA:
    def __init__(self):
        self.initialize_session_state()
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.memory = ConversationBufferWindowMemory(k=10)  # Correct initialization

    @staticmethod
    def initialize_session_state():
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = None
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        if 'clear_input' not in st.session_state:
            st.session_state.clear_input = False

    def process_file(self, file, file_type):
        file_type = file_type.lower()  # Convert to lowercase for consistent comparison
        extension = "pdf" if file_type == "pdf" else "xlsx"  # Default to xlsx for Excel
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name

            if not temp_file_path or not os.path.exists(temp_file_path):
                st.error("Temporary file path is invalid or does not exist.")
                return False

            try:
                if file_type == "pdf":
                    loader = PyPDFLoader(temp_file_path)
                elif file_type == "excel":
                    loader = UnstructuredExcelLoader(temp_file_path, mode="elements")
                else:
                    st.error(f"Unsupported file type: {file_type}")
                    return False

                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,  # Reduced chunk size for better context
                    chunk_overlap=100  # Increased overlap for better context continuity
                )
                texts = text_splitter.split_documents(documents)
                st.session_state.vector_store = FAISS.from_documents(texts, self.embeddings)
                return True

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                return False
            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

    def generate_response(self, query, model_name):
        try:
            if not st.session_state.vector_store:
                return "Please upload and process a file first."

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
                temperature=0.1,
                max_tokens=1024,
                stream=True
            )

            response = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    response += chunk.choices[0].delta.content
            
            self.memory.save_context({"input": query}, {"output": response})
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
                response = ""
                for chunk in completion:
                    if chunk.choices[0].delta.content:
                        response += chunk.choices[0].delta.content
                return response
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
                    # Create a placeholder for streaming response
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    # Stream the response
                    for chunk in self.get_completion(limited_history, self.model_name, stream=True):
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            # Update the placeholder with the current response
                            with message_placeholder.container():
                                st.chat_message("assistant").write(full_response)
                    
                    # Add the full response to conversation history
                    st.session_state.conversation_history.append({"role": "assistant", "content": full_response})
                    # Clear the placeholder
                    message_placeholder.empty()
                else:
                    # Non-streaming response
                    response = self.get_completion(limited_history, self.model_name, stream=False)
                    st.session_state.conversation_history.append({"role": "assistant", "content": response})
            
            # Set a flag to clear input on next rerun
            st.session_state.clear_input = True

    def render_ui(self):
        

        # Sidebar settings
        with st.sidebar:
            st.header("Settings")
            mode = st.radio("Select Mode", ["RAG", "Chat"])  # Move the mode selection here
            self.model_name = st.selectbox("Select Model", list(model_info.keys()))
            
            # Add streaming toggle
            self.enable_streaming = st.checkbox("Enable Streaming", value=True)
            
            if st.button("Clear Conversation", key="clear_sidebar"):
                st.session_state.conversation_history = []
            
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
            
            # Add controls for temperature, top_k, and other parameters
            st.markdown("---")
            st.subheader("Advanced Parameters")
            # Initialize parameters with preset values but allow manual override
            self.temperature = st.slider("Temperature", 0.0, 1.0, 
                              value=behavior_presets[selected_behavior]["temperature"])
            self.top_k = st.slider("Top K", 1, 100, 
                             value=behavior_presets[selected_behavior]["top_k"])
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
                with st.spinner(f"Processing {file_type}..."):
                    if self.process_file(uploaded_file, file_type.lower()):
                        st.success(f"File processed successfully!")

            user_input = st.text_input("Enter your question:", key="user_input")
            
            if st.button("Send") and user_input:
                with st.spinner("Generating response..."):
                    response = self.generate_response(user_input, self.model_name)
                    st.markdown("**Answer:**\n" + response)

            # Show conversation history
            memory_variables = self.memory.load_memory_variables({})
            if "history" in memory_variables and memory_variables["history"]:
                st.markdown("### Recent Conversation")
                history_lines = memory_variables["history"].split("\n")[-MAX_HISTORY_LENGTH:]
                st.markdown("\n".join(history_lines))

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

            # Display conversation history in reverse order
            with chat_container:
                for message in reversed(st.session_state.conversation_history):
                    with st.chat_message(message["role"]):
                        st.write(message["content"])

if __name__ == "__main__":
    pdf_qa_app = PDFQA()
    pdf_qa_app.render_ui()
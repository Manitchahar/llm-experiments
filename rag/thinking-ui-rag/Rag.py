import streamlit as st
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from streamlit.runtime.caching import cache_resource
import numpy as np
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
import nltk
from docling.document_converter import DocumentConverter
import chromadb

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load environment variables
load_dotenv()

# Initialize session state for persistent storage
if "processed_files" not in st.session_state:
    st.session_state["processed_files"] = set()
if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None

st.set_page_config(page_title="RAG with Semantic Chunking", layout="centered")
st.title("RAG with Semantic Chunking")

@cache_resource
def get_embedding_model():
    """Cache the SentenceTransformer model instance."""
    return SentenceTransformer('all-MiniLM-L6-v2')

def create_semantic_chunks(text):
    """
    Create semantically meaningful chunks using recursive splitting and semantic verification
    """
    # First, use RecursiveCharacterTextSplitter for initial chunking
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", ""],
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )
    initial_chunks = text_splitter.split_text(text)
    
    # Semantic verification using sentence embeddings
    model = get_embedding_model()
    sentences = nltk.sent_tokenize(text)
    if not sentences: # Handle case where text results in no sentences
        return initial_chunks if initial_chunks else []
        
    embeddings = model.encode(sentences)
    
    # Compute semantic similarity between consecutive sentences
    semantic_chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(len(sentences) - 1):
        similarity = util.cos_sim(embeddings[i], embeddings[i+1])
        
        if similarity < 0.6 or len(' '.join(current_chunk)) > 500: # Corrected from &lt;
            semantic_chunks.append(' '.join(current_chunk))
            current_chunk = []
        current_chunk.append(sentences[i + 1])
    
    if current_chunk:
        semantic_chunks.append(' '.join(current_chunk))
    
    return semantic_chunks if semantic_chunks else initial_chunks

def process_document(file):
    """Process document with Docling and create semantic chunks"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_path = tmp_file.name

    converter = DocumentConverter()
    with st.spinner("Extracting text with Docling..."):
        result = converter.convert(tmp_path)
        text = result.document.export_to_markdown()
    
    with st.spinner("Creating semantic chunks..."):
        chunks = create_semantic_chunks(text)
        
    # Initialize or get vector store
    # For Chroma, SentenceTransformer can be passed directly as the embedding function
    # if we ensure the model is loaded once or passed correctly.
    # Langchain's Chroma wrapper handles this.
    embedding_model = get_embedding_model()
    
    if st.session_state["vector_store"] is None:
        # Chroma uses an embedding function that takes a list of texts and returns embeddings.
        # The SentenceTransformer model's .encode() method fits this.
        st.session_state["vector_store"] = Chroma(
            collection_name="semantic_chunks_collection", # Ensure a unique collection name
            embedding_function=embedding_model # Pass the model instance itself
        )
    
    # Add chunks to vector store
    if chunks: # Ensure there are chunks to add
        st.session_state["vector_store"].add_texts(chunks)
    st.session_state["processed_files"].add(file.name)
    
    return chunks

def query_documents(query):
    """Query the vector store and return relevant chunks"""
    if st.session_state["vector_store"] is None:
        return []
    
    # The vector store (Chroma) uses its configured embedding function for the query
    results = st.session_state["vector_store"].similarity_search(query, k=3)
    return [doc.page_content for doc in results]

# File upload UI
uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, image, etc.)", type=["pdf", "docx", "jpg", "jpeg", "png"])
if uploaded_file and uploaded_file.name not in st.session_state["processed_files"]:
    chunks = process_document(uploaded_file)
    st.success(f"Document processed! Created {len(chunks)} semantic chunks")

# Display processed files
if st.session_state["processed_files"]:
    st.subheader("Processed Documents:")
    for file_name in st.session_state["processed_files"]: # Iterate over file_name
        st.write(f"- {file_name}")

# Query UI
query = st.text_input("Ask a question about your documents:")
if query:
    relevant_chunks = query_documents(query)
    st.subheader("Relevant Context:")
    for i, chunk in enumerate(relevant_chunks, 1):
        with st.expander(f"Context {i}"):
            st.write(chunk)
    
    # Generate response using Groq
    if relevant_chunks:
        chat_model = ChatGroq(
            api_key=os.getenv('GROQ_API_KEY'),
            model="qwen-qwq-32b" # Ensure this model is available/correct
        )
        
        prompt = f"""Based on the following context, answer the question:
        
Context:
{' '.join(relevant_chunks)}

Question: {query}
Answer:"""
        
        with st.spinner("Generating answer..."):
            response = chat_model.invoke(prompt)
            st.subheader("Answer:")
            st.write(response.content)
    elif st.session_state["vector_store"] is not None: # If query was made but no chunks found
        st.info("No relevant context found in the processed documents for your query.")
    elif not st.session_state["processed_files"]:
        st.warning("Please upload and process a document before asking a question.")

# Note: The `if __name__ == "__main__":` block is not strictly necessary for Streamlit scripts
# as they are run from top to bottom when executed with `streamlit run your_script.py`.
# The previous `run_rag_app()` function has been integrated directly into the script flow.
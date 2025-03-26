from typing import List, Dict
import os
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from phi.document import Document
from phi.document.reader.pdf import PDFReader as PhiPDFReader # Use Phi's reader for consistency if needed, but pypdf is simpler for basic text extraction
from phi.utils.log import logger

# --- Configuration ---
CHROMA_DB_PATH = "multiagent_poc/chroma_db"
COLLECTION_NAME = "knowledge_base"
# Ensure you have GOOGLE_API_KEY in your .env file or environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

# Simple embedding function wrapper for ChromaDB if not using its built-ins
# This might be needed if ChromaDB's Google integration requires specific setup
# For now, let's try passing Phi's embedder directly if possible, or use Chroma's default
# ef = embedding_functions.DefaultEmbeddingFunction() # Chroma's default
# Or configure Google PaLM directly if Chroma supports it easily:
# ef = embedding_functions.GooglePalmEmbeddingFunction(api_key=GOOGLE_API_KEY, model_name="text-embedding-004")

# --- ChromaDB Client ---
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Get or create the collection
# Try using Phi's embedder; if it fails, we might need a wrapper or Chroma's built-in
try:
    # ChromaDB expects an EmbeddingFunction protocol object. Phi's might not match directly.
    # Let's use Chroma's helper for Google Embeddings if available and straightforward
    # Assuming text-embedding-004 is compatible with the GoogleGenerativeAiEmbeddingFunction
    ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=GOOGLE_API_KEY, model_name="models/text-embedding-004")
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef
    )
    logger.info(f"Using GoogleGenerativeAiEmbeddingFunction for collection '{COLLECTION_NAME}'.")
except Exception as e:
    logger.warning(f"Could not initialize ChromaDB with GoogleGenerativeAiEmbeddingFunction: {e}. Falling back to default.")
    ef = embedding_functions.DefaultEmbeddingFunction() # Fallback to default sentence transformer
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef
    )


# --- PDF Processing and Indexing ---

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a PDF file."""
    logger.info(f"Extracting text from: {pdf_path}")
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n" # Add newline between pages
        logger.info(f"Successfully extracted text from {pdf_path}")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    """Splits text into overlapping chunks."""
    logger.info(f"Chunking text (length: {len(text)})...")
    if not text:
        return []
    # Simple chunking for now
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
        if start >= len(text):
             break # Avoid infinite loop on very short overlaps/texts
    logger.info(f"Created {len(chunks)} chunks.")
    return chunks

def add_pdf_to_knowledge_base(pdf_path: str, pdf_id: str):
    """Processes a PDF and adds its content to the ChromaDB collection."""
    logger.info(f"Processing PDF: {pdf_path} with ID: {pdf_id}")

    # 1. Extract Text
    full_text = extract_text_from_pdf(pdf_path)
    if not full_text:
        logger.error(f"No text extracted from {pdf_path}. Skipping.")
        return False

    # 2. Chunk Text
    text_chunks = chunk_text(full_text)
    if not text_chunks:
        logger.error(f"No chunks created from {pdf_path}. Skipping.")
        return False

    # 3. Prepare for ChromaDB
    documents = text_chunks
    metadatas = [{"source": pdf_id, "chunk": i} for i in range(len(documents))]
    ids = [f"{pdf_id}_chunk_{i}" for i in range(len(documents))]

    # Optional: Check for existing chunks to avoid duplicates (can be slow for large updates)
    # existing_items = collection.get(ids=ids)
    # if existing_items and existing_items['ids']:
    #    logger.warning(f"Found {len(existing_items['ids'])} existing chunks for {pdf_id}. Skipping add operation for these.")
       # Potentially filter out existing ids before adding, or use collection.upsert

    # 4. Add to Collection (using ChromaDB's batching)
    # Embeddings are generated automatically by the collection's embedding_function
    try:
        logger.info(f"Adding {len(documents)} chunks to collection '{COLLECTION_NAME}'...")
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Successfully added content from {pdf_id} to the knowledge base.")
        return True
    except Exception as e:
        logger.error(f"Error adding documents to ChromaDB for {pdf_id}: {e}")
        # Consider more specific error handling (e.g., API errors, DB errors)
        return False

# --- Querying ---

def query_knowledge_base(query_text: str, n_results: int = 5) -> List[Dict]:
    """Queries the knowledge base for relevant document chunks."""
    logger.info(f"Querying knowledge base for: '{query_text}'")
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances'] # Include distances for relevance scoring
        )
        logger.info(f"Found {len(results.get('ids', [])[0]) if results else 0} results.")

        # Format results nicely
        formatted_results = []
        if results and results.get('ids') and results['ids'][0]:
             ids = results['ids'][0]
             docs = results['documents'][0]
             metadatas = results['metadatas'][0]
             distances = results['distances'][0]
             for i in range(len(ids)):
                 formatted_results.append({
                     "id": ids[i],
                     "document": docs[i],
                     "metadata": metadatas[i],
                     "distance": distances[i]
                 })
        return formatted_results
    except Exception as e:
        logger.error(f"Error querying ChromaDB: {e}")
        return []

# --- Example Usage (Optional) ---
if __name__ == "__main__":
    # Create a dummy PDF for testing if needed, or use an existing one
    # Example: add_pdf_to_knowledge_base("path/to/your/document.pdf", "doc_001")

    search_query = "What are the main challenges?"
    print(f"\n--- Querying for: '{search_query}' ---")
    query_results = query_knowledge_base(search_query)

    if query_results:
        print("Found relevant documents:")
        for res in query_results:
            print(f"  - ID: {res['id']} (Distance: {res['distance']:.4f})")
            print(f"    Source: {res['metadata'].get('source', 'N/A')}")
            print(f"    Text: {res['document'][:150]}...") # Print snippet
    else:
        print("No relevant documents found.")

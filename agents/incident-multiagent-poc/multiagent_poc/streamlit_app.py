import streamlit as st
from agents import (
    TriageAgent,
    search_agent_mock,
    ResolutionCoachAgent,
    IncidentContext
)
from knowledge_base import add_pdf_to_knowledge_base, query_knowledge_base
from typing import Optional, List, Dict
import os
import tempfile
from pathlib import Path
import time
import json

# Set page configuration
st.set_page_config(
    page_title="ResolveX",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FFF8E1;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .error-box {
        background-color: #FFEBEE;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stExpander {
        border: none !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;
    }
    .source-rag {
        background-color: #E3F2FD;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #1E88E5;
    }
    .source-internet {
        background-color: #FFF8E1;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #FFA000;
    }
    .source-ai {
        background-color: #E8F5E9;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #43A047;
    }
    .source-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    .badge-rag {
        background-color: #1E88E5;
        color: white;
    }
    .badge-internet {
        background-color: #FFA000;
        color: white;
    }
    .badge-ai {
        background-color: #43A047;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Ensure GOOGLE_API_KEY is set
if not os.getenv("GOOGLE_API_KEY"):
    st.warning("""
    ⚠️ WARNING: GOOGLE_API_KEY environment variable not set.
    Please create a .env file in the multiagent_poc directory with:
    GOOGLE_API_KEY='your_actual_gemini_api_key'
    """)

# --- Constants ---
PDF_UPLOAD_DIR = "multiagent_poc/uploaded_pdfs"
os.makedirs(PDF_UPLOAD_DIR, exist_ok=True)

# --- Helper Functions ---
def format_knowledge_results(results: List[Dict]) -> str:
    """Formats ChromaDB query results into a string."""
    if not results:
        return "No relevant information found in the knowledge base."

    formatted = "Found the following relevant information in the knowledge base:\n"
    for i, res in enumerate(results):
        metadata = res.get('metadata', {})
        source = metadata.get('source', 'N/A') if metadata else 'N/A'
        distance = res.get('distance', float('inf'))
        relevance_score = 1 - distance if distance != float('inf') else 0
        document = res.get('document', '')

        formatted += f"\n--- Result {i+1} (Source: {source}, Relevance Score: {relevance_score:.4f}) ---\n"
        formatted += f"{document}\n"
    return formatted

def extract_rag_sources(kb_results: List[Dict]) -> List[Dict]:
    """Extract source information from knowledge base results"""
    sources = []
    for res in kb_results:
        metadata = res.get('metadata', {})
        source = metadata.get('source', 'N/A') if metadata else 'N/A'
        distance = res.get('distance', float('inf'))
        relevance_score = 1 - distance if distance != float('inf') else 0
        document = res.get('document', '')
        
        sources.append({
            "source": source,
            "relevance": relevance_score,
            "content": document[:300] + ("..." if len(document) > 300 else "")
        })
    return sources

def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if 'triage_agent' not in st.session_state:
        st.session_state.triage_agent = TriageAgent()
    if 'coach_agent' not in st.session_state:
        st.session_state.coach_agent = ResolutionCoachAgent()
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "ResolveX ready. Please describe the incident."}]
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "Chat"
    if 'resolution_steps' not in st.session_state:
        st.session_state.resolution_steps = None
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'source_attribution' not in st.session_state:
        st.session_state.source_attribution = {}

def process_incident(incident_description: str):
    """Process incident description and generate response"""
    st.session_state.processing = True
    
    triage_agent = st.session_state.triage_agent
    coach_agent = st.session_state.coach_agent
    
    # Create a placeholder for the progress bar
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Initialize progress bar
    progress_bar = progress_placeholder.progress(0)
    status_placeholder.info("Starting incident processing...")
    
    # Initialize source attribution
    source_attribution = {
        "rag_sources": [],
        "internet_sources": [],
        "reasoning_process": ""
    }
    
    # Triage Step
    try:
        status_placeholder.info("Triaging incident...")
        context = triage_agent.process_incident(incident_description)
        progress_bar.progress(25)
        
        if not (context and context.severity and context.keywords):
            status_placeholder.error("Triage failed to extract context.")
            st.session_state.processing = False
            progress_placeholder.empty()
            status_placeholder.empty()
            return None
            
    except Exception as e:
        status_placeholder.error(f"Triage Error: {e}")
        st.session_state.processing = False
        progress_placeholder.empty()
        status_placeholder.empty()
        return None
    
    # Information Gathering
    knowledge_info = "Not available"
    search_info = "Not available"
    kb_results = []
    
    try:
        # Query Knowledge Base
        status_placeholder.info("Querying knowledge base...")
        kb_results = query_knowledge_base(incident_description, n_results=3)
        knowledge_info = format_knowledge_results(kb_results)
        
        # Extract RAG sources for attribution
        source_attribution["rag_sources"] = extract_rag_sources(kb_results)
        
        progress_bar.progress(50)
        
        # Mock Search
        status_placeholder.info("Searching external sources...")
        if context.keywords:
            search_info = search_agent_mock(context.keywords)
            # Add internet source
            source_attribution["internet_sources"].append(
                f"External search for keywords: {', '.join(context.keywords)}"
            )
        progress_bar.progress(75)
        
    except Exception as e:
        status_placeholder.warning(f"Information gathering error: {e}")
        # Continue anyway with what we have
    
    # Resolution Coach
    try:
        status_placeholder.info("Generating resolution steps...")
        resolution_steps = coach_agent.generate_steps(
            incident_description=incident_description,
            triage_context=context,
            knowledge_info=knowledge_info,
            search_info=search_info
        )
        
        # Add reasoning process
        source_attribution["reasoning_process"] = f"""
        <p>The AI analyzed the incident with severity <strong>{context.severity}</strong> and keywords <strong>{', '.join(context.keywords)}</strong>.</p>
        
        <p>It combined information from:</p>
        <ul>
            <li>The knowledge base ({len(source_attribution['rag_sources'])} relevant documents)</li>
            <li>External search results</li>
            <li>Its own understanding of incident resolution best practices</li>
        </ul>
        
        <p>The resolution steps were generated by synthesizing all available information and applying 
        incident management principles to create a structured approach to resolving the issue.</p>
        """
        
        progress_bar.progress(100)
        status_placeholder.success("Processing complete!")
        
        # Store results in session state
        st.session_state.last_context = context
        st.session_state.last_knowledge_info = knowledge_info
        st.session_state.last_search_info = search_info
        st.session_state.resolution_steps = resolution_steps
        st.session_state.source_attribution = source_attribution
        
        # Add assistant response with full resolution steps
        st.session_state.messages.append({
            "role": "assistant", 
            "content": f"**Resolution Steps:**\n\n{resolution_steps}"
        })
        
        # Clear progress indicators after a short delay
        time.sleep(1)
        progress_placeholder.empty()
        status_placeholder.empty()
        
        st.session_state.processing = False
        return resolution_steps
        
    except Exception as e:
        status_placeholder.error(f"Resolution Coach Error: {e}")
        progress_placeholder.empty()
        status_placeholder.empty()
        st.session_state.processing = False
        return None

def display_chat():
    """Display the chat interface"""
    st.markdown('<div class="main-header">ResolveX</div>', unsafe_allow_html=True)
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # If this is an assistant message with resolution steps and we have source attribution
                if (message["role"] == "assistant" and 
                    "Resolution Steps" in message.get("content", "") and 
                    st.session_state.source_attribution):
                    
                    with st.expander("📊 View Source Attribution"):
                        st.markdown("### Where did this information come from?")
                        
                        # Display RAG sources
                        if st.session_state.source_attribution.get('rag_sources'):
                            st.markdown("#### Knowledge Base (RAG) Sources:")
                            for i, source in enumerate(st.session_state.source_attribution['rag_sources']):
                                st.markdown(f"""
                                <div class="source-rag">
                                    <span class="source-badge badge-rag">RAG</span>
                                    <strong>Source:</strong> {source['source']}<br>
                                    <strong>Relevance:</strong> {source['relevance']:.2f}<br>
                                    <strong>Content:</strong> {source['content']}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.markdown("No knowledge base sources were used.")
                            
                        # Display internet sources
                        if st.session_state.source_attribution.get('internet_sources'):
                            st.markdown("#### Internet Sources:")
                            for i, source in enumerate(st.session_state.source_attribution['internet_sources']):
                                st.markdown(f"""
                                <div class="source-internet">
                                    <span class="source-badge badge-internet">WEB</span>
                                    {source}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.markdown("No internet sources were used.")
                            
                        # Display reasoning process
                        if st.session_state.source_attribution.get('reasoning_process'):
                            st.markdown("#### AI Reasoning Process:")
                            st.markdown(f"""
                            <div class="source-ai">
                                <span class="source-badge badge-ai">AI</span>
                                {st.session_state.source_attribution['reasoning_process']}
                            </div>
                            """, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Describe the incident", disabled=st.session_state.processing):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Rerun to display the user message immediately
        st.rerun()

def display_upload_pdf():
    """Display the Knowledge Base upload interface"""
    st.markdown('<div class="main-header">Knowledge Base</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Upload PDF documents to build the knowledge base. These documents will be used to provide context for incident resolution.</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Select a PDF file (max 100MB)",
        type=["pdf"],
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        if uploaded_file.name in st.session_state.uploaded_files:
            st.warning(f"File '{uploaded_file.name}' has already been uploaded.")
        else:
            file_path = os.path.join(PDF_UPLOAD_DIR, uploaded_file.name)
            
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    # Save the file
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Add to knowledge base
                    success = add_pdf_to_knowledge_base(pdf_path=file_path, pdf_id=uploaded_file.name)
                    
                    if success:
                        st.session_state.uploaded_files.append(uploaded_file.name)
                        st.success(f"Successfully added '{uploaded_file.name}' to the knowledge base.")
                        
                        # Add a button to return to chat
                        if st.button("Return to Chat"):
                            st.session_state.current_tab = "Chat"
                            st.rerun()
                    else:
                        st.error(f"Failed to process '{uploaded_file.name}'.")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # Display list of uploaded files
    if st.session_state.uploaded_files:
        st.markdown("### Uploaded Files")
        for file in st.session_state.uploaded_files:
            st.markdown(f"- {file}")

def display_analysis():
    """Display the analysis of the last processed incident"""
    if not hasattr(st.session_state, 'last_context'):
        st.info("No incident has been processed yet. Please describe an incident in the Chat tab.")
        return
    
    st.markdown('<div class="main-header">Incident Analysis</div>', unsafe_allow_html=True)
    
    # Triage Results
    st.markdown('<div class="sub-header">Triage Results</div>', unsafe_allow_html=True)
    context = st.session_state.last_context
    st.markdown(f"""
    <div class="info-box">
        <strong>Severity:</strong> {context.severity}<br>
        <strong>Keywords:</strong> {', '.join(context.keywords)}
    </div>
    """, unsafe_allow_html=True)
    
    # Knowledge Base Results
    st.markdown('<div class="sub-header">Knowledge Base Results</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box">
        {st.session_state.last_knowledge_info.replace('\n', '<br>')}
    </div>
    """, unsafe_allow_html=True)
    
    # External Search Results
    st.markdown('<div class="sub-header">External Search Results</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box">
        {st.session_state.last_search_info.replace('\n', '<br>')}
    </div>
    """, unsafe_allow_html=True)
    
    # Resolution Steps
    st.markdown('<div class="sub-header">Resolution Steps</div>', unsafe_allow_html=True)
    
    # Fix: Check if resolution_steps is None before calling replace()
    resolution_content = ""
    if st.session_state.resolution_steps:
        resolution_content = st.session_state.resolution_steps.replace('\n', '<br>')
    
    st.markdown(f"""
    <div class="success-box">
        {resolution_content}
    </div>
    """, unsafe_allow_html=True)
    
    # Source Attribution
    if hasattr(st.session_state, 'source_attribution') and st.session_state.source_attribution:
        st.markdown('<div class="sub-header">Source Attribution</div>', unsafe_allow_html=True)
        
        # Create tabs for different source types
        source_tabs = st.tabs(["Knowledge Base", "Internet", "AI Reasoning"])
        
        # Knowledge Base tab
        with source_tabs[0]:
            if st.session_state.source_attribution.get('rag_sources'):
                for i, source in enumerate(st.session_state.source_attribution['rag_sources']):
                    st.markdown(f"""
                    <div class="source-rag">
                        <span class="source-badge badge-rag">RAG</span>
                        <strong>Source:</strong> {source['source']}<br>
                        <strong>Relevance:</strong> {source['relevance']:.2f}<br>
                        <strong>Content:</strong> {source['content']}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No knowledge base sources were used.")
        
        # Internet tab
        with source_tabs[1]:
            if st.session_state.source_attribution.get('internet_sources'):
                for i, source in enumerate(st.session_state.source_attribution['internet_sources']):
                    st.markdown(f"""
                    <div class="source-internet">
                        <span class="source-badge badge-internet">WEB</span>
                        {source}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No internet sources were used.")
        
        # AI Reasoning tab
        with source_tabs[2]:
            if st.session_state.source_attribution.get('reasoning_process'):
                st.markdown(f"""
                <div class="source-ai">
                    <span class="source-badge badge-ai">AI</span>
                    {st.session_state.source_attribution['reasoning_process']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No AI reasoning process information available.")

def main():
    """Main function to run the Streamlit app"""
    # Initialize session state
    initialize_session_state()
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/technical-support.png", width=80)
        st.markdown("## Navigation")
        
        # Navigation tabs
        selected_tab = st.radio(
            "Select a page:",
            ["Chat", "Knowledge Base", "Analysis"],
            index=["Chat", "Knowledge Base", "Analysis"].index(st.session_state.current_tab.replace("Upload PDF", "Knowledge Base"))
        )
        
        # Update current tab in session state
        if selected_tab != st.session_state.current_tab:
            st.session_state.current_tab = selected_tab
            st.rerun()
        
        # Add some information in the sidebar
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This application helps you manage and resolve incidents by:
        - Analyzing incident descriptions
        - Searching knowledge base for similar issues
        - Generating step-by-step resolution guides
        """)
        
        # Add a clear chat button
        if st.button("Clear Chat History"):
            st.session_state.messages = [{"role": "assistant", "content": "ResolveX ready. Please describe the incident."}]
            st.session_state.resolution_steps = None
            st.session_state.source_attribution = {}
            st.rerun()
    
    # Display the selected tab
    if st.session_state.current_tab == "Chat":
        display_chat()
        
        # Process the latest user message if not already processing
        if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user" and not st.session_state.processing:
            with st.spinner("Processing incident..."):
                process_incident(st.session_state.messages[-1]["content"])
            st.rerun()
            
    elif st.session_state.current_tab in ["Upload PDF", "Knowledge Base"]:
        display_upload_pdf()
        
    elif st.session_state.current_tab == "Analysis":
        display_analysis()

if __name__ == "__main__":
    main()

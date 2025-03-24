import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import PDFPlumberLoader
import os
import re
import pandas as pd
from typing import List, Dict, Any
# New imports for Email Automation
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from langchain_google_community.gmail.utils import build_resource_service
from langchain_google_community import GmailToolkit
from email.message import EmailMessage
import base64
import json

# --- Constants ---
GOOGLE_API_KEY = "REDACTED_SECRET"
GROQ_API_KEY = "REDACTED_SECRET"
NAME_PATTERN = re.compile(r"^(.*?)\n")
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_PATTERN = re.compile(r"\b\d{10}\b")

# --- Model Initialization ---
def initialize_models() -> tuple[LLMChain, GoogleGenerativeAIEmbeddings]:
    """Initialize LLM chain and embeddings model."""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=None,
        timeout=None,
        api_key=GROQ_API_KEY
    )
    
    prompt_template = PromptTemplate(
        input_variables=["job_title", "skills", "experience"],
        template="Generate a job description for the position of {job_title} with the following skills: {skills} and {experience} years of experience."
    )
    
    return (
        LLMChain(llm=llm, prompt=prompt_template),
        GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
    )

llm_chain, embeddings = initialize_models()

# --- Email Automation Helper Functions ---
EMAIL_PROMPT_TEMPLATE = """You are a professional email assistant. Create an email based on these details:
    
**Recipient**: {recipient}
**Purpose**: {reason}

Requirements:
- Use proper business letter format
- Maintain {tone} tone
- 3-5 clear paragraphs
- Appropriate opening/closing

Return only the email body text:"""

def init_gmail():
    """Initialize Gmail API connection with automatic token refresh"""
    try:
        credentials = Credentials.from_authorized_user_file(
            "token.json",
            scopes=["https://mail.google.com/"]
        ) if os.path.exists("token.json") else None
        
        if not credentials or not credentials.valid:
            if credentials and credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    "credentials.json",
                    ["https://mail.google.com/"]
                )
                credentials = flow.run_local_server(port=0)
            
            with open("token.json", "w") as token:
                token.write(credentials.to_json())
                
        return GmailToolkit(api_resource=build_resource_service(credentials=credentials))
    
    except Exception as e:
        raise RuntimeError(f"Gmail initialization failed: {str(e)}")

def create_email_message(recipient: str, subject: str, body: str) -> str:
    """Create MIME email message compatible with Gmail API"""
    message = EmailMessage()
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body)
    return base64.urlsafe_b64encode(message.as_bytes()).decode()

def generate_email(recipient: str, reason: str, tone: str = "professional") -> str:
    """Generate email content using ChatGroq"""
    try:
        prompt = EMAIL_PROMPT_TEMPLATE.format(
            recipient=recipient,
            reason=reason,
            tone=tone
        )
        # Reusing the ChatGroq model from our earlier initialization for email generation
        return llm_chain.llm.invoke(prompt).content
    except Exception as e:
        raise RuntimeError(f"Generation failed: {str(e)}")
        
def send_email(recipient: str, subject: str, body: str) -> dict:
    """Send email through Gmail API"""
    try:
        if not recipient or "@" not in recipient:
            raise ValueError("Invalid recipient email address")
        if not subject.strip():
            raise ValueError("Email subject required")
        
        toolkit = init_gmail()
        service = toolkit.api_resource
        
        raw_message = create_email_message(recipient, subject, body)
        response = service.users().messages().send(
            userId="me",
            body={"raw": raw_message}
        ).execute()
        
        return {
            "status": "sent",
            "message_id": response["id"],
            "thread_id": response["threadId"]
        }
        
    except Exception as e:
        error_info = {
            "error_type": type(e).__name__,
            "message": str(e),
            "gmail_error": getattr(e, "content", None)
        }
        raise RuntimeError(f"Failed to send email: {json.dumps(error_info, indent=2)}")

# --- Streamlit App ---
st.title("Hiring Tool")
page = st.sidebar.radio("Select Page", ["JD Generator", "Resume Ranker", "Email Automation"])

# JD Generator Page
if page == "JD Generator":
    st.title("JD Generator")
    job_title = st.text_input("Job Title")
    skills = st.text_area("Skills (comma-separated)")
    experience = st.number_input("Experience (years)", min_value=0)

    if st.button("Generate Job Description"):
        if job_title and skills and experience:
            try:
                job_description = llm_chain.run(job_title=job_title, skills=skills, experience=experience)
                st.subheader("Generated Job Description")
                st.write(job_description)
            except Exception as e:
                st.error(f"Error generating JD: {e}")
        else:
            st.error("Please fill all fields")

# Resume Ranker Page
elif page == "Resume Ranker":
    st.title("Resume Ranker")
    uploaded_files = st.file_uploader("Upload PDF Resumes", type=["pdf"], accept_multiple_files=True)
    job_description = st.text_area("Paste Job Description", height=200)

    if st.button("Rank Resumes"):
        if uploaded_files and job_description:
            try:
                resume_texts = []
                for pdf_file in uploaded_files:
                    with open(pdf_file.name, "wb") as f:
                        f.write(pdf_file.getbuffer())
                    loader = PDFPlumberLoader(pdf_file.name)
                    docs = loader.load()
                    text = " ".join([doc.page_content for doc in docs])
                    resume_texts.append(text)
                    os.remove(pdf_file.name)

                job_embedding = embeddings.embed_query(job_description)
                resume_embeddings = embeddings.embed_documents(resume_texts)
                similarities = cosine_similarity(resume_embeddings, [job_embedding])
                ranked_indices = similarities.flatten().argsort()[::-1]

                results = []
                for idx in ranked_indices:
                    text = resume_texts[idx]
                    name = re.search(r"^(.*?)\n", text).group(1).strip() if re.search(r"^(.*?)\n", text) else "Unknown"
                    email = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)
                    phone = re.search(r"\b\d{10}\b", text)
                    results.append({
                        "Rank": len(results)+1,
                        "Name": name,
                        "Contact": email.group(0) if email else phone.group(0) if phone else "N/A"
                    })
                
                st.table(pd.DataFrame(results))
            except Exception as e:
                st.error(f"Error processing resumes: {e}")
        else:
            st.error("Please upload resumes and enter a job description")

# Email Automation Page
elif page == "Email Automation":
    st.title("Email Automation")
    recipient = st.text_input("Recipient Email", placeholder="user@example.com")
    subject = st.text_input("Subject", placeholder="Meeting Request")
    reason = st.text_area("Email Content Purpose", placeholder="Key points or message purpose")
    tone = st.selectbox("Tone", ["professional", "friendly", "formal"])

    if st.button("Generate Draft 💌"):
        if not all([recipient, subject, reason]):
            st.error("Please fill all required fields")
        else:
            with st.spinner("Consulting AI assistant..."):
                try:
                    draft = generate_email(recipient, reason, tone)
                    st.session_state.draft = draft
                    st.success("Draft generated!")
                except Exception as e:
                    st.error(f"Generation failed: {str(e)}")

    if 'draft' in st.session_state:
        st.subheader("Generated Draft")
        edited_draft = st.text_area("Edit your draft", 
                                    value=st.session_state.draft, 
                                    height=300,
                                    help="Make any final edits before sending")
    
        if st.button("Send Email 🚀"):
            with st.spinner("Sending..."):
                try:
                    result = send_email(recipient, subject, edited_draft)
                    del st.session_state.draft
                    st.success("Email sent successfully!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Sending failed: {str(e)}")
                    st.json(json.loads(str(e).split("Failed to send email: ")[1]))
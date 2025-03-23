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
from langchain_core.prompts import ChatPromptTemplate
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- API Key Handling ---
google_api_key = "REDACTED_SECRET"
groq_api_key = "REDACTED_SECRET"

# --- Model Setup ---
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=None,
    timeout=None,
    api_key=groq_api_key
)

prompt_template = PromptTemplate(
    input_variables=["job_title", "skills", "experience"],
    template="Generate a job description for the position of {job_title} with the following skills: {skills} and {experience} years of experience."
)

llm_chain = LLMChain(llm=llm, prompt=prompt_template)

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=google_api_key)

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
class EmailAutomation:
    def __init__(self, llm):
        self.llm = llm
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.sender_email = st.secrets["email"]["sender"]
        self.sender_password = st.secrets["email"]["password"]

    def draft_email(self, recipient, purpose, details):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional recruitment email assistant"),
            ("human", "Draft email to {recipient} about {purpose}. Include: {details}")
        ])
        return (prompt | self.llm).invoke({
            "recipient": recipient,
            "purpose": purpose,
            "details": details
        }).content

    def send_email(self, to_email, subject, body):
        msg = MIMEMultipart()
        msg['From'] = self.sender_email
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        try:
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            server.send_message(msg)
            server.quit()
            return "Email sent successfully"
        except Exception as e:
            raise Exception(f"Failed to send email: {str(e)}")

    def automate_interview_invite(self, candidate_info):
        email_body = self.draft_email(
            candidate_info['name'],
            "Interview Invitation",
            f"Position: {candidate_info['position']}\nDetails: {candidate_info['interview_details']}"
        )
        return self.send_email(
            candidate_info['email'],
            f"Interview for {candidate_info['position']}",
            email_body
        )

if page == "Email Automation":
    st.title("Email Automation")
    email_automation = EmailAutomation(llm)
    
    candidate_name = st.text_input("Candidate Name")
    candidate_email = st.text_input("Email")
    job_title = st.text_input("Position")
    interview_details = st.text_area("Interview Details")
    
    if st.button("Send Invite"):
        if all([candidate_name, candidate_email, job_title, interview_details]):
            try:
                result = email_automation.automate_interview_invite({
                    "name": candidate_name,
                    "email": candidate_email,
                    "position": job_title,
                    "interview_details": interview_details
                })
                st.success(f"Email sent successfully! Server response: {result}")
            except Exception as e:
                st.error(f"Error sending email: {str(e)}")
        else:
            st.error("Please fill all fields")

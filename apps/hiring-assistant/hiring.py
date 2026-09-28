import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# Set up the ChatGroq model
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=None,
    timeout=None,
    api_key=os.getenv("GROQ_API_KEY")
)

# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["job_title", "skills", "experience"],
    template="Generate a job description for the position of {job_title} with the following skills: {skills} and {experience} years of experience."
)

# Create the LLM chain
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

# Streamlit app
st.title("JD Generator")

# Input fields
job_title = st.text_input("Job Title")
skills = st.text_area("Skills (comma-separated)")
experience = st.number_input("Experience (years)", min_value=0)

# Generate job description
if st.button("Generate Job Description"):
    if job_title and skills and experience:
        job_description = llm_chain.run(job_title=job_title, skills=skills, experience=experience)
        st.subheader("Generated Job Description")
        st.write(job_description)
    else:
        st.error("Please fill in all fields.")

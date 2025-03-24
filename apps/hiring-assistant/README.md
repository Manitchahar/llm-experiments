# hiring

## Hiring Tool

This Streamlit application provides a suite of tools to assist with the hiring process, including a Job Description Generator, Resume Ranker, and Email Automation.

### Features

-   **JD Generator**: Generates job descriptions based on job title, skills, and experience level.
-   **Resume Ranker**: Ranks uploaded resumes based on their similarity to a provided job description.
-   **Email Automation**: Generates and sends emails for various purposes.

### Setup

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up environment variables:**

    -   Create a `.env` file in the root directory.
    -   Add your embedding model key and Groq API key to the `.env` file:

        ```
        Groq api key=<groq api key>
        TRANSFORMER_API_KEY=<google text embeddings>  
        Use Hugging Face Transformers embeddings if needed.
        ```

4.  **Set up Gmail API:**

    -   Enable the Gmail API in the Google Cloud Console.
    -   Download the `credentials.json` file and place it in the root directory.
    -   The application will automatically create a `token.json` file upon first use to store authentication tokens.

### Usage

1.  **Run the Streamlit application:**

    ```bash
    streamlit run Resume.py
    ```

2.  **Access the application in your browser** at the address provided by Streamlit (usually `http://localhost:8501`).

### Pages

-   **JD Generator**: Enter the job title, required skills, and experience level to generate a job description.
-   **Resume Ranker**: Upload PDF resumes and paste a job description to rank the resumes based on their relevance.
-   **Email Automation**: Enter recipient email, subject, and purpose to generate and send emails.

### Dependencies

-   streamlit
-   langchain-groq
-   langchain-core
-   langchain-google-genai
-   scikit-learn
-   langchain-community
-   google-auth-httplib2
-   google-auth-oauthlib
-   email-validator
-   python-dotenv
-   pandas

### License
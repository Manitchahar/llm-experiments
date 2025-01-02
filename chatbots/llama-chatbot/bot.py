import streamlit as st
import httpx
from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")
client = Groq(api_key=GROQ_API_KEY)
client._client._transport = httpx.HTTPTransport(verify=False)

# Model name
model_name = "llama-3.3-70b-versatile"

# Response options
response_options = {
    "Short": {"max_tokens": 256, "temperature": 0.6, "top_p": 0.6},
    "Balanced": {"max_tokens": 1024, "temperature": 0.7, "top_p": 0.7},
    "Long": {"max_tokens": 2048, "temperature": 0.8, "top_p": 0.8}
}

def get_completion(messages, temperature, top_p, max_tokens, stream=False):
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=stream,
            stop=None,
        )
        
        if not stream:
            return completion.choices[0].message.content
        return completion

    except Exception as e:
        return f"An error occurred: {str(e)}"

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Main chat interface
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@700&display=swap');

    .title {
        font-family: 'Inter', sans-serif !important;
        font-size: 56px !important;
        font-weight: 700 !important;
        text-align: center !important;
        color: #262730 !important; /* Streamlit's dark gray for text */
        margin-bottom: 10px !important;
        background: linear-gradient(90deg, #1C83E1, #00D1B2); /* Streamlit's blue and teal */
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 5s ease infinite;
    }

    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .subtitle {
        font-family: 'Inter', sans-serif !important;
        font-size: 24px !important;
        color: #6B7280 !important; 
        text-align: center !important;
        margin-top: -10px !important;
        font-weight: 400 !important;
        opacity: 0;
        animation: fadeIn 2s ease 1s forwards;
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    .glow {
        text-shadow: 0 0 10px rgba(28, 131, 225, 0.3), 0 0 20px rgba(0, 209, 178, 0.3); /* Subtle glow using Streamlit's colors */
    }
    </style>
    <div>
        <h1 class="title glow" id="title">LLAMA Chat</h1>
        <p class="subtitle" id="subtitle">Your intelligent assistant powered by LLaMA 3</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.header("Settings")
    selected_option = st.radio("Choose response length:", ["Short", "Balanced", "Long"])
    st.write(f"Current Settings: {selected_option}")
    st.write(f"max_tokens: {response_options[selected_option]['max_tokens']}")
    st.write(f"temperature: {response_options[selected_option]['temperature']}")
    st.write(f"top_p: {response_options[selected_option]['top_p']}")
    if st.button("Clear Chat"):
        st.session_state.conversation_history = []

# Input form container at the top
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Message:", key="user_input", placeholder="Type your message here...")
    submit_button = st.form_submit_button("Send")

    if submit_button and user_input.strip():
        st.session_state.conversation_history.append({"role": "user", "content": user_input})
        
        with st.spinner("Thinking..."):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                *st.session_state.conversation_history[-5:]
            ]
            
            # Get parameters based on selected option
            params = response_options[selected_option]
            
            # Pass parameters to get_completion with streaming enabled
            response = get_completion(
                messages,
                temperature=params["temperature"],
                top_p=params["top_p"],
                max_tokens=params["max_tokens"],
                stream=True
            )
            
            # Initialize a placeholder for the assistant's response
            assistant_response_placeholder = st.empty()
            assistant_response = ""
            
            # Stream the response
            for chunk in response:
                if chunk.choices[0].delta.content:
                    assistant_response += chunk.choices[0].delta.content
                    assistant_response_placeholder.markdown(assistant_response)
            
            st.session_state.conversation_history.append({"role": "assistant", "content": assistant_response})
        
        st.rerun()

# Chat messages container below
chat_container = st.container()
with chat_container:
    for message in reversed(st.session_state.conversation_history):
        with st.chat_message(message["role"]):
            st.write(message["content"])
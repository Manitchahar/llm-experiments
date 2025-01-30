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

# Model options - Limited to specified models
model_options = {
    "Llama3.3-70B-Versatile": "llama-3.3-70b-versatile",
    "DeepSeek-V2-70B": "deepseek-r1-distill-llama-70b" # User-friendly name, API identifier
}

# Default model name
default_model_name = "Llama3.3-70B-Versatile" 
model_name = model_options[default_model_name] # Initialize with API identifier

# Response options
response_options = {
    "Short": {"max_tokens": 256, "temperature": 0.6, "top_p": 0.6},
    "Balanced": {"max_tokens": 1024, "temperature": 0.7, "top_p": 0.7},
    "Long": {"max_tokens": 2048, "temperature": 0.8, "top_p": 0.8}
}

def get_completion(messages, model, temperature, top_p, max_tokens, stream=False):
    """
    Calls the Groq API to get a chat completion.

    Args:
        messages (list): List of message dictionaries for the conversation history.
        model (str): The model name to use.
        temperature (float): Temperature parameter for generation.
        top_p (float): Top_p parameter for generation.
        max_tokens (int): Maximum tokens in the response.
        stream (bool): Whether to use streaming.

    Returns:
        str or Iterable: Completion content if not streaming, otherwise stream of completion chunks.
                       Returns an error message string if an exception occurs during non-streaming,
                       and None for stream on error (with error message displayed via st.error).
    """
    try:
        completion = client.chat.completions.create(
            model=model,
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
        if stream:
            st.error(f"Error during streaming: {str(e)}")
            return None # Signal error during streaming by returning None
        return f"An error occurred: {str(e)}"


# Initialize session state for conversation history
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
        <p class="subtitle" id="subtitle">Your intelligent assistant powered by LLaMA 3 & DeepSeek</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar for settings
with st.sidebar:
    st.header("Settings")

    with st.expander("Model Selection", expanded=True): # Model settings expander (initially expanded)
        model_selector_key = "model_selector"
        selected_model_name = st.selectbox(
            "Choose Model:",
            options=list(model_options.keys()),
            index=list(model_options.keys()).index(default_model_name),
            key=model_selector_key
        )
        model_name = model_options[selected_model_name]
        st.markdown(f"**Current Model:** {selected_model_name}") # Use markdown for bold text

    with st.expander("Response Options"): # Response options expander (initially collapsed)
        selected_option = st.radio("Choose response length:", ["Short", "Balanced", "Long"])
        st.markdown(f"**Response Length:** {selected_option}") # Use markdown for bold text
        st.markdown(f"**max_tokens:** {response_options[selected_option]['max_tokens']}") # Use markdown for bold text
        st.markdown(f"**temperature:** {response_options[selected_option]['temperature']}") # Use markdown for bold text
        st.markdown(f"**top_p:** {response_options[selected_option]['top_p']}") # Use markdown for bold text

    if st.button("Clear Chat"):
        st.session_state.conversation_history = []
        st.rerun()


# Input form container at the top
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Message:", key="user_input", placeholder="Type your message here...")
    submit_button = st.form_submit_button("Send")

    if submit_button and user_input.strip():
        st.session_state.conversation_history.append({"role": "user", "content": user_input})

        with st.spinner("Thinking..."):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                *st.session_state.conversation_history[-5:] # Keep last 5 messages for context
            ]

            # Get parameters based on selected response length option
            params = response_options[selected_option]

            # Get completion from Groq API with streaming
            response = get_completion(
                messages,
                model=model_name, # Use selected model_name
                temperature=params["temperature"],
                top_p=params["top_p"],
                max_tokens=params["max_tokens"],
                stream=True
            )

            # Initialize placeholder for streaming response
            assistant_response_placeholder = st.empty()
            assistant_response = ""

            # Stream the response chunks
            if response is not None: # Check if response is not None (error case)
                for chunk in response:
                    if chunk and chunk.choices[0].delta.content:
                        assistant_response += chunk.choices[0].delta.content
                        assistant_response_placeholder.markdown(assistant_response)
                    elif chunk is None: # Explicitly handle None chunk (error signal) if needed more granularly
                        break # Exit streaming loop if get_completion returned None
            # else: response is None, error already displayed by get_completion


            st.session_state.conversation_history.append({"role": "assistant", "content": assistant_response})

        st.rerun() # Rerun to display new messages


# Chat messages container below
chat_container = st.container()
with chat_container:
    if not st.session_state.conversation_history:
        st.info("Welcome! Choose a model and response length from the sidebar to start chatting.")
    for message in reversed(st.session_state.conversation_history): # Reverse to show latest at bottom
        with st.chat_message(message["role"]): # Use st.chat_message for styled chat bubbles
            st.write(message["content"])

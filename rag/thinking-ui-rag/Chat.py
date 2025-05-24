import re
import base64
import streamlit as st
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set Streamlit page configuration
st.set_page_config(page_title="Mini ChatGPT", layout="centered") # Made title consistent

GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
MODEL = "qwen-qwq-32b"

def format_reasoning_response(thinking_content):
    """Basic formatting for thinking content (e.g., stripping whitespace)."""
    return thinking_content.strip()

def display_message(message):
    """Display a single message in the chat interface."""
    role = "user" if message["role"] == "user" else "assistant"
    with st.chat_message(role):
        if role == "assistant":
            display_assistant_message(message["content"])
        else:
            st.markdown(message["content"])

def display_assistant_message(content):
    """Display assistant message with thinking content if present, with status box for all assistant messages."""
    pattern = r"<think>(.*?)</think>"
    think_match = re.search(pattern, content, re.DOTALL)
    if think_match:
        think_content = think_match.group(1)
        response_content = content.replace(f"<think>{think_content}</think>", "")
        # Show "Thinking complete!" status for all assistant messages with <think> tags
        with st.status("Thinking complete!", state="complete", expanded=False):
            st.markdown(format_reasoning_response(think_content))
        st.markdown(response_content)
    else:
        st.markdown(content)

def display_chat_history():
    """Display all previous messages in the chat history."""
    for message in st.session_state["messages"]:
        if message["role"] != "system":  # Skip system messages
            display_message(message)

@st.cache_resource
def get_chat_model():
    return ChatGroq(api_key=GROQ_API_KEY, model=MODEL)

def handle_user_input():
    """Handle new user input and generate assistant response."""
    if user_input := st.chat_input("Type your message here..."):
        st.session_state["messages"].append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        current_thinking_status = None # To hold the status object for access in except block
        with st.chat_message("assistant"):
            try:
                # Prepare prompt and get model
                history_messages = [
                    f"{msg['role'].capitalize()}: {msg['content']}"
                    for msg in st.session_state["messages"] if msg["role"] != "system"
                ]
                prompt = "\n".join(history_messages) + "\n" # Ensure trailing newline

                chat_model = get_chat_model()
                
                thinking_content_raw = "" # Raw content between <think> tags
                response_content_final = ""
                
                with st.status("Thinking...", expanded=True) as status_obj:
                    current_thinking_status = status_obj # Assign to outer variable
                    think_placeholder = st.empty()
                    
                    # Get the response and handle streaming
                    response = chat_model.invoke(prompt)
                    content = response.content
                    
                    # Extract thinking and response parts
                    pattern = r"<think>(.*?)</think>"
                    think_match = re.search(pattern, content, re.DOTALL)
                    
                    if think_match:
                        thinking_content_raw = think_match.group(1)  # Get content between tags
                        # Show thinking content character by character to simulate streaming
                        displayed_thinking = ""
                        for char in thinking_content_raw:
                            displayed_thinking += char
                            think_placeholder.markdown(format_reasoning_response(displayed_thinking)) # Apply formatting
                        
                        response_content_final = content.replace(f"<think>{thinking_content_raw}</think>", "")
                        current_thinking_status.update(label="Thinking complete!", state="complete", expanded=False)
                    else:
                        response_content_final = content
                        current_thinking_status.update(label="Response ready", state="complete", expanded=False)
                
                # Show final response (after status is collapsed)
                st.markdown(response_content_final)
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": (f"<think>{thinking_content_raw}</think>" if thinking_content_raw else "") + response_content_final
                })

            except Exception as e:
                st.error(f"Groq API error: {e}")
                if current_thinking_status: # If status object was created and an error occurred
                    current_thinking_status.update(label="Error during processing", state="error", expanded=False)

def main():
    """Main function to handle the chat interface and streaming responses."""
    try:
        img_base64 = base64.b64encode(open("OIP.jpg", "rb").read()).decode()
        st.markdown(f"""
        # Mini ChatGPT powered by <img src="data:image/png;base64,{img_base64}" width="170" style="vertical-align: -20px;">
        """, unsafe_allow_html=True)
    except FileNotFoundError:
        st.markdown("# Mini ChatGPT") # Fallback title if image not found
        st.warning("OIP.jpg not found. Displaying text title only.")

    st.markdown("<h4 style='text-align: center;'>With thinking UI! 💡</h4>", unsafe_allow_html=True)

    display_chat_history()
    handle_user_input()

if __name__ == "__main__":
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
    main()
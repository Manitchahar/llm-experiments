import gradio as gr
import platform # Add platform import
from chains import generate_replies # Import the function from chains.py
import os
from dotenv import load_dotenv
import pytesseract
from PIL import Image
import io

# --- Tesseract Configuration (OS-dependent) ---
# Set the path for Windows if running locally.
# On Linux (like Hugging Face Spaces), pytesseract should find it automatically
# if installed via packages.txt.
if platform.system() == "Windows":
    # Replace with your actual path if different
    tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    else:
        print(f"WARNING: Tesseract path specified but not found: {tesseract_path}")
        print("         Screenshot feature may fail if Tesseract is not in system PATH.")
# No 'else' needed - on other systems, pytesseract searches PATH by default.
# -----------------------------------------

# Load environment variables
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    print("WARNING: GROQ_API_KEY not found in .env file. The app might not work.")

# --- OCR Helper Function ---
def extract_text_from_image(image_input):
    """
    Extracts text from an uploaded image using pytesseract.
    'image_input' is expected to be a PIL Image object from Gradio.
    """
    if image_input is None:
        return None, "No image provided."

    try:
        # image_input from gr.Image is already a PIL image
        text = pytesseract.image_to_string(image_input, lang='eng+hin') # Specify languages (English + Hindi)
        print(f"--- OCR Result ---\n{text}\n------------------") # Log OCR result
        if not text.strip():
            return None, "OCR could not detect any text in the image."
        return text.strip(), None # Return extracted text and no error
    except pytesseract.TesseractNotFoundError:
        print("ERROR: Tesseract executable not found. Please install Tesseract and/or set the tesseract_cmd path.")
        return None, "ERROR: Tesseract OCR is not installed or configured correctly on the server."
    except Exception as e:
        print(f"Error during OCR processing: {e}")
        return None, f"An error occurred during image processing: {e}"

# --- Main Gradio Function ---
def get_rizz_replies(received_message_text, screenshot_image, my_last_message, situation):
    """
    Wrapper function to handle text or image input, call OCR if needed,
    and then call the backend chain to return replies.
    """
    received_message = None
    error_message = None

    # 1. Prioritize Screenshot Input
    if screenshot_image is not None:
        print("Processing screenshot...")
        extracted_text, ocr_error = extract_text_from_image(screenshot_image)
        if ocr_error:
            return f"Screenshot Error: {ocr_error}" # Return OCR error to user
        if extracted_text:
             received_message = extracted_text
             print(f"Using text from screenshot: '{received_message[:100]}...'") # Log confirmation
        # If OCR failed to find text but no error occurred, we might fall through to text input

    # 2. Use Text Input if no message derived from image
    if received_message is None and received_message_text:
         received_message = received_message_text.strip()
         print("Using text from input box.")

    # 3. Check if we have a message to process
    if not received_message:
        return "Please enter the message you received OR upload a screenshot of it."

    # 4. Call the backend chain
    print(f"Generating replies for: '{received_message[:100]}...'")
    try:
        replies = generate_replies(
            received_message=received_message,
            my_last_message=my_last_message,
            situation=situation
        )
        return replies
    except Exception as e:
        print(f"Error in Gradio interface calling generate_replies: {e}")
        # Provide a user-friendly error message
        return f"Sorry, an error occurred generating replies: {e}. Check API key and model status."

# --- Gradio Interface Definition ---
with gr.Blocks(theme=gr.themes.Soft(), title="Rizzly") as demo:
    gr.Markdown("# Rizzly 🔥 - AI Reply Generator")
    gr.Markdown("Paste the chat message OR upload a screenshot below and get witty reply suggestions!")

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Option 1: Enter Text Manually")
            received_msg_input = gr.Textbox(
                label="Message You Received",
                placeholder="Paste message here...",
                lines=4,
                interactive=True
            )
            gr.Markdown("### Option 2: Upload Screenshot")
            screenshot_input = gr.Image(
                label="Upload Screenshot",
                type="pil", # Return PIL image object
                interactive=True
            )

            with gr.Accordion("Optional Context (Improves Replies)", open=False):
                last_msg_input = gr.Textbox(
                    label="Your Last Message (Optional)",
                    placeholder="What did you last say?",
                    lines=2,
                    interactive=True
                )
                situation_input = gr.Textbox(
                    label="Brief Situation (Optional)",
                    placeholder="e.g., 'First date follow-up', 'Met at party'",
                    lines=1,
                    interactive=True
                )
            generate_button = gr.Button("✨ Generate Rizz Replies ✨", variant="primary")

        with gr.Column(scale=3):
            output_replies = gr.Markdown(label="Suggested Replies", value="Your suggested replies will appear here...")

    # --- Event Handling ---
    generate_button.click(
        fn=get_rizz_replies,
        inputs=[received_msg_input, screenshot_input, last_msg_input, situation_input], # Added screenshot_input
        outputs=output_replies
    )

    gr.Markdown("--- \n *Powered by Groq & Langchain*")

# --- Launch the App ---
if __name__ == "__main__":
    # Set server_name="0.0.0.0" to allow connections from your local network
    # Use demo.launch(share=True) for a temporary public link (requires login)
    demo.launch(server_name="0.0.0.0", share=True)
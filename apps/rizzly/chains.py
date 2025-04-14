import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables from .env file
load_dotenv()

# Ensure the GROQ_API_KEY is set
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in the .env file.")

# Define the prompt template
# Incorporates the received message and optional context
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", """You are Rizzly, the ultimate AI wingman/wingwoman! Your mission is to craft **super flirty, charming, and engaging replies** for dating apps and social chats. Your primary goal is to help the user make a strong positive impression, spark attraction, and increase their chances of getting a date. Be bold, be creative, and be memorable!

        **Core Instructions:**
        1.  **Language Mastery:**
            *   Analyze the 'Received Message' input to determine its primary language (English, Hindi, or Hinglish).
            *   **Generate ALL replies in the SAME language/style.**
            *   For Hinglish input, generate natural-sounding Hinglish replies. For Hindi, use Hindi. For English, use English.

        2.  **Handling Screenshot OCR Input:**
            *   The 'Received Message' input might contain raw text extracted via OCR from a chat screenshot. This text could include multiple messages, timestamps, UI elements, and messages from both the user ("You") and the other person.
            *   **Your Task:** Identify the **chronologically last message sent by the *other person*** within the provided 'Received Message' text. Ignore timestamps, UI elements, and messages marked as sent by "You" or similar user indicators if possible.
            *   **Generate replies AS IF YOU ARE THE USER, responding *directly* to that *single last message* from the other person.** Do NOT comment on the entire conversation history shown in the OCR text.

        3.  **Context is Key:** Also consider the other provided context:
            *   User's last message (if provided - this is what the user *actually* last sent, separate from the screenshot).
            *   The overall situation (e.g., first contact, post-date, specific event).

        4.  **Generate 4 Distinct Reply Options:** Provide a variety of flirty styles (in the correct language):
            *   **Option 1: Playful & Teasing:** Lighthearted, fun, maybe slightly challenging. Uses humor.
            *   **Option 2: Confident & Direct:** Bold, shows clear interest, maybe includes a subtle compliment or suggestion.
            *   **Option 3: Intriguing & Mysterious:** Creates curiosity, hints at something interesting, asks an engaging question.
            *   **Option 4: Sweet & Charming (Optional Cheesy):** Can be genuinely sweet OR intentionally over-the-top cheesy/corny for comedic effect. Label cheesy ones clearly (e.g., "(Cheesy)").

        5.  **Flirting Techniques to Employ:**
            *   Ask open-ended, engaging questions relevant to the *last message received*.
            *   Use subtle (or sometimes bold!) compliments.
            *   Inject humor and wit appropriately.
            *   Create intrigue; don't reveal everything at once.
            *   Maintain a positive and confident tone.
            *   Keep replies relatively concise and conversational.

        6.  **Output Format:** Present replies clearly, numbered 1 to 4. Only output the suggested replies.

        **Example Scenario (Input: OCR text ending with "Other person: You have a cute dog!")**
        *   1. (Playful) Is he taking applications for a new best friend? 😉 What's his name?
        *   2. (Confident) He gets his good looks from his owner. 😉 What are you up to?
        *   3. (Intriguing) He's my partner in crime... currently plotting world domination (or just where to find the best snacks). What kind of trouble are you getting into today?
        *   4. (Sweet/Cheesy) Thanks! Are you perhaps... *paw*sitive-ly smitten? 😄 (Cheesy)

        Remember: Focus on replying *as the user* to the *last message received* from the other person in the screenshot. Be the ultimate Rizz assistant!"""),
        ("human", """Here's the situation:
        My Last Message: {my_last_message}
        Received Message: {received_message}
        Brief Situation: {situation}

        Generate 3 reply options:"""),
    ]
)

# Initialize the Groq Chat Model
# Using Llama3 70b via Groq, as it's powerful and fast
# You might need to adjust the model name if you prefer another one available on Groq
chat_model = ChatGroq(temperature=0.7, groq_api_key=api_key, model_name="llama-3.3-70b-versatile")
# Define the output parsert
output_parser = StrOutputParser()

# Create the Langchain Expression Language (LCEL) chain
# Passes the input dictionary directly to the prompt template
chain = (
    {"received_message": RunnablePassthrough(), "my_last_message": RunnablePassthrough(), "situation": RunnablePassthrough()}
    | prompt_template
    | chat_model
    | output_parser
)

# Function to generate replies using the chain
def generate_replies(received_message: str, my_last_message: str = "N/A", situation: str = "General chat") -> str:
    """
    Generates replies using the Langchain chain.
    Handles default values for optional fields.
    """
    # Ensure default values if inputs are empty or None
    my_last_message = my_last_message if my_last_message else "N/A"
    situation = situation if situation else "General chat"

    input_data = {
        "received_message": received_message,
        "my_last_message": my_last_message,
        "situation": situation,
    }
    try:
        response = chain.invoke(input_data)
        return response
    except Exception as e:
        # Basic error handling
        print(f"Error generating replies: {e}")
        return "Sorry, I encountered an error while generating replies."

# Example usage (for testing purposes, can be removed later)
if __name__ == "__main__":
    print("Testing generate_replies function...")
    test_received = "Hey! How's it going?"
    test_last = "Just sent you a funny meme."
    test_situation = "Met at a coffee shop yesterday"
    replies = generate_replies(test_received, test_last, test_situation)
    print("\n--- Sample Generation ---")
    print(f"Received: {test_received}")
    print(f"My Last: {test_last}")
    print(f"Situation: {test_situation}")
    print("\nGenerated Replies:")
    print(replies)
    print("------------------------")
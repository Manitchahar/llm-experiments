import gradio as gr
import time

# Placeholder function for generating replies
# In the future, this will call the Langchain logic from chains.py
def generate_rizz(received_message, my_last_message, situation):
    print("Inputs received:")
    print(f"  Received Message: {received_message}")
    print(f"  My Last Message: {my_last_message}")
    print(f"  Situation: {situation}")

    # Dummy delay to simulate processing
    time.sleep(1)

    # Dummy replies - replace with actual AI generation later
    dummy_replies = [
        f"Dummy Reply 1 based on '{received_message}'",
        f"Dummy Reply 2 (context: {situation})",
        "Dummy Reply 3 - Be witty!",
    ]
    # Join replies into a single string with newlines for display
    return "\n\n".join(dummy_replies)

# Define Gradio Interface components
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# Rizzly 🔥\nYour AI Wingman for generating flirty replies.")

    with gr.Row():
        with gr.Column(scale=2):
            received_msg_input = gr.Textbox(
                label="Message Received",
                placeholder="Paste the message you received here...",
                lines=3
            )
            my_last_msg_input = gr.Textbox(
                label="My Last Message (Optional Context)",
                placeholder="What did you say before this?",
                lines=2
            )
            situation_input = gr.Textbox(
                label="Brief Situation (Optional Context)",
                placeholder="e.g., First date follow-up, Met at party, Tinder match",
                lines=1
            )
            generate_button = gr.Button("✨ Generate Rizz Replies ✨", variant="primary")

        with gr.Column(scale=1):
            output_replies = gr.Textbox(
                label="Suggested Replies",
                placeholder="AI-generated replies will appear here...",
                lines=10,
                interactive=False # Make output non-editable
            )

    # Connect button click to the function
    generate_button.click(
        fn=generate_rizz,
        inputs=[received_msg_input, my_last_msg_input, situation_input],
        outputs=output_replies
    )

# Launch the Gradio app
if __name__ == "__main__":
    app.launch()
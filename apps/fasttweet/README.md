# FastTweet

AI-assisted tweet drafting with research, generation, and reflection loops built with LangGraph, Groq, Tavily Search, and Streamlit.

## What It Does

- Searches for recent context on a topic with Tavily.
- Generates a draft tweet with a Groq-hosted Llama model.
- Runs a reflection loop to critique and improve the draft.
- Shows the final tweet, research snippets, and generation process in a Streamlit UI.

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Create a local environment file:

   ```powershell
   Copy-Item .env.example .env
   ```

3. Add your API keys to `.env`:

   ```env
   GROQ_API_KEY=your_groq_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

4. Run the app:

   ```bash
   streamlit run fastbot.py
   ```

## Security Note

Do not commit `.env` files or API keys. Use `.env.example` for placeholders only.

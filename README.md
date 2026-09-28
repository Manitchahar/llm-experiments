# llm-experiments

Early LLM projects: RAG, agents, chatbots, and small AI apps. Most were built in 2024–2025 while learning LangChain, Groq, CrewAI, MCP, and Streamlit. Each folder was its own repository; they're consolidated here with full commit history preserved.

These are learning prototypes, not maintained products. For current work, see [gemini-claw](https://github.com/Manitchahar/gemini-claw), [RocketGuard-1b](https://github.com/Manitchahar/RocketGuard-1b), [inference-engineering](https://github.com/Manitchahar/inference-engineering), and [ADK-Agents](https://github.com/Manitchahar/ADK-Agents).

## Projects

### Agents

| Project | What it is | Stack |
| --- | --- | --- |
| [incident-multiagent-poc](agents/incident-multiagent-poc) | Four-agent incident-management PoC: triage (severity + keywords), knowledge lookup, search, and a resolution coach that drafts troubleshooting steps. | Phidata, Gemini, ChromaDB, Streamlit |
| [yt-summariser-crewai](agents/yt-summariser-crewai) | Two-agent crew (researcher + writer) that searches a YouTube channel's videos and writes a blog post from them. | CrewAI |
| [mcp-chainlit](agents/mcp-chainlit) | Minimal MCP server that exposes a PowerShell tool, with a Chainlit chat client. **Runs arbitrary shell commands, so local use only.** | MCP (FastMCP), Chainlit |

### RAG

| Project | What it is | Stack |
| --- | --- | --- |
| [llama-rag-chat](rag/llama-rag-chat) | Chat with uploaded PDFs using embeddings + retrieval, with selectable models. | LangChain, Groq, sentence-transformers, Streamlit |
| [thinking-ui-rag](rag/thinking-ui-rag) | "Mini ChatGPT" UI that parses `<think>` blocks and shows the model's reasoning separately from its answer. | LangChain, Groq, Streamlit |

### Chatbots

| Project | What it is | Stack |
| --- | --- | --- |
| [llama-chatbot](chatbots/llama-chatbot) | Simple streaming chatbot over Llama models. | Groq, Streamlit |

### Apps

| Project | What it is | Stack |
| --- | --- | --- |
| [fasttweet](apps/fasttweet) | Tweet drafting with a research → generate → reflect loop. | LangGraph, Groq, Tavily, Streamlit |
| [edugenie](apps/edugenie) | AI tutor that builds personalized lesson plans and finds relevant courses. | LangChain, Groq |
| [hiring-assistant](apps/hiring-assistant) | JD generator, resume ranker (embedding similarity), and Gmail-based candidate outreach. | LangChain, Groq, Gemini embeddings, Gmail API, Streamlit |
| [rizzly](apps/rizzly) | Paste a chat message, get suggested replies with optional context. | LangChain, Groq, Gradio |

### ML basics

| Project | What it is |
| --- | --- |
| [ml-basics](ml/ml-basics) | Linear regression and Word2Vec embedding notebooks. |

## Running a project

Each folder is self-contained:

```bash
cd rag/llama-rag-chat
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp ../../.env.example .env   # then fill in the keys the project needs
streamlit run Rag.py
```

API keys are read from environment variables / a local `.env` file, which is git-ignored. Never commit it.

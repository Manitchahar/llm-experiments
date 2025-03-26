# Multi-Agent Incident Management PoC Plan

## Objective

Create a simple and fast Proof of Concept (PoC) for a multi-agent incident management system using Streamlit for the UI.

## Technology Stack

*   **UI Framework:** Streamlit
*   **LLM:** Google Gemini API (Requires API key)
* 

## Agent Definitions

1.  **Triage Agent:** Classifies incident severity (High/Medium/Low) and extracts context keywords using Gemini.
2.  **Knowledge Agent:** Takes context keywords, returns relevant past incident snippets (mocked in PoC).
3.  **Search Agent:** Takes context keywords, returns external search results (mocked in PoC).
4.  **Resolution Coach Agent:** Takes incident description, context, knowledge, and search results. Generates step-by-step troubleshooting suggestions and logs them to `resolutions.json`.

## Architecture & Orchestration

The flow is implemented in streamlit_app.py with agent logic in agents.py:

```mermaid
graph LR
    A[User Input (Incident via Streamlit)] --> B(Triage Agent);
    B -- Severity & Context --> C{Main App};
    C -- Context --> D(Knowledge Agent);
    C -- Context --> E(Search Agent);
    D -- Past Incidents --> C;
    E -- Search Results --> C;
    C -- All Data --> F(Resolution Coach Agent);
    F -- Resolution Steps --> H[(resolutions.json)];
    F -- Final Steps --> I{Output (Streamlit UI)};
```

## Implementation Steps

1.  **Setup Project:** Initialize Python environment, install `streamlit`, `google-generativeai`. Configure Gemini API key (e.g., via environment variable `GOOGLE_API_KEY`).
2.  **Define Agents:** Implement agent logic in agents.py.
3.  **Build Streamlit UI (streamlit_app.py):**
    *   Use Streamlit widgets for input/output
    *   Call agent functions with user input
4.  **Testing:** Run `streamlit run streamlit_app.py` and test with various incident descriptions. Verify agent outputs and `resolutions.json` updates.

## Key Files

* streamlit_app.py - Main application UI
* agents.py - Agent implementations
* knowledge_base.py - Knowledge retrieval
* resolutions.json - Stores resolution history

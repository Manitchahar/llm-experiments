# Multi-Agent Incident Management PoC Plan

## Objective

Create a simple and fast Proof of Concept (PoC) for a multi-agent incident management system using Chainlit for the UI and Phidata for the agent framework.

## Technology Stack

*   **UI Framework:** Chainlit
*   **Agent Framework:** Phidata
*   **LLM:** Google Gemini API (Requires API key)
* 

## Agent Definitions

1.  **Triage Agent:** Classifies incident severity (High/Medium/Low) and extracts context keywords using Gemini.
2.  **Knowledge Agent:** Takes context keywords, returns *mocked* relevant past incident snippets.
3.  **Search Agent:** Takes context keywords, returns *mocked* external search results.
4.  **Resolution Coach Agent:** Takes incident description, context, mocked knowledge, and mocked search results. Uses Gemini to synthesize step-by-step troubleshooting suggestions.
5.  **Documentation Agent:** Takes the final resolution steps, formats them, and appends them to `resolutions.json`.

## Architecture & Orchestration

The flow will be managed within the Chainlit application, calling Phidata agents:

```mermaid
graph LR
    A[User Input (Incident via Chainlit)] --> B(Triage Agent);
    B -- Severity & Context --> C{Orchestrator};
    C -- Context --> D(Knowledge Agent);
    C -- Context --> E(Search Agent);
    D -- Mocked Past Incidents --> C;
    E -- Mocked Search Results --> C;
    C -- All Data --> F(Resolution Coach Agent);
    F -- Troubleshooting Steps --> C;
    C -- Resolution Steps --> G(Documentation Agent);
    G -- Logs to JSON --> H[(resolutions.json)];
    F -- Final Steps --> I{Output (Chainlit UI)};
```

## Implementation Steps

1.  **Setup Project:** Initialize Python environment, install `phidata`, `chainlit`, `google-generativeai`, `pydantic`. Configure Gemini API key (e.g., via environment variable `GOOGLE_API_KEY`).
2.  **Define Phidata Agents:** Create Python classes/functions for each agent, incorporating mocked data sources where specified.
3.  **Implement Orchestration:** Define the sequence of agent calls.
4.  **Build Chainlit UI (`app.py`):**
    *   Use `@cl.on_chat_start` for setup.
    *   Use `@cl.on_message` to handle user input, trigger orchestration, and display results using `cl.Message` and `cl.Step`.
5.  **Create Mock Data:** Implement simple Python functions/dictionaries for Knowledge and Search agents.
6.  **Testing:** Run `chainlit run app.py -w` and test with various incident descriptions. Verify agent outputs and `resolutions.json` updates.

## Key Files

*  Everthing in one files except the json, mock or rag data


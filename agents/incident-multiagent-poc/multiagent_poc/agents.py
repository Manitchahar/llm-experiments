import os
from dotenv import load_dotenv
from typing import List, Dict, Optional
import json

from phi.assistant import Assistant
from phi.llm.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from pydantic import BaseModel, Field

# Load environment variables (for GOOGLE_API_KEY)
load_dotenv()

# --- Agent Definitions ---

class IncidentContext(BaseModel):
    severity: Optional[str] = Field(None, description="Estimated severity: High, Medium, or Low")
    keywords: Optional[List[str]] = Field(None, description="List of 3-5 keywords summarizing the incident")

class TriageAgent(Assistant):
    def __init__(self, **kwargs):
        super().__init__(
            name="TriageAgent",
            llm=Gemini(model="gemini-pro"),
            description="You are an expert incident triage specialist. Analyze the user's incident description and determine its severity and key context.",
            output_model=IncidentContext,
            instructions=[
                "Read the incident description carefully.",
                "Classify the severity as 'High', 'Medium', or 'Low' based on potential impact and urgency.",
                "Extract 3-5 relevant keywords that capture the core issue.",
                "Provide ONLY the JSON output with severity and keywords.",
            ],
            # debug_mode=True, # Uncomment for detailed logs
            **kwargs
        )

    def process_incident(self, incident_description: str) -> IncidentContext:
        """Processes the incident description and returns severity and keywords."""
        return self.run(incident_description)

# --- Mocked Agents (as functions for simplicity in PoC) ---

def knowledge_agent_mock(keywords: List[str]) -> str:
    """Mocks retrieving relevant past incidents."""
    print(f"--- Knowledge Agent (Mock) searching for: {keywords} ---")
    # In a real scenario, this would query a vector database
    mock_results = {
        "database": "Found similar past incident #123: 'Database connection errors after deployment'. Resolved by rolling back.",
        "network": "Found knowledge base article: 'Troubleshooting intermittent network latency'. Suggests checking firewall rules.",
        "login": "Found past incident #456: 'Users unable to login via SSO'. Root cause was expired certificate.",
        "slow": "Found performance tuning guide: 'Optimizing web server response times'.",
    }
    # Simple keyword matching for mock
    for keyword in keywords:
        if keyword.lower() in mock_results:
            return mock_results[keyword.lower()]
    return "No highly relevant past incidents found in mock data."

def search_agent_mock(keywords: List[str]) -> str:
    """Mocks performing external searches."""
    print(f"--- Search Agent (Mock) searching for: {keywords} ---")
    # In a real scenario, this would use DuckDuckGo, Tavily, etc.
    # Example using DuckDuckGo tool if we wanted a real search:
    # search_results = DuckDuckGo().run(" ".join(keywords))
    # return search_results[:500] # Limit length
    return f"Mock search result for '{' '.join(keywords)}': Found external blog post discussing similar error messages and potential OS-level causes."

# --- Resolution Coach Agent ---

class ResolutionCoachAgent(Assistant):
     def __init__(self, **kwargs):
        super().__init__(
            name="ResolutionCoachAgent",
            llm=Gemini(model="gemini-pro"),
            description="You are an expert troubleshooting coach. Synthesize information to provide actionable resolution steps.",
            instructions=[
                "Receive the original incident description, triage context, knowledge base info, and external search results.",
                "Combine all information.",
                "Generate a clear, step-by-step troubleshooting guide for the user.",
                "Focus on actionable steps.",
                "If information is conflicting, state it and suggest verification steps.",
            ],
            # debug_mode=True,
            **kwargs
        )

     def generate_steps(self, incident_description: str, triage_context: IncidentContext, knowledge_info: str, search_info: str) -> str:
        """Generates troubleshooting steps based on combined information."""
        prompt = f"""
        Original Incident: {incident_description}

        Triage Context:
        Severity: {triage_context.severity}
        Keywords: {', '.join(triage_context.keywords) if triage_context.keywords else 'N/A'}

        Relevant Knowledge Found:
        {knowledge_info}

        Relevant External Info Found:
        {search_info}

        Based on all the above information, provide a step-by-step troubleshooting guide:
        """
        return self.run(prompt)

# --- Documentation Agent ---

class DocumentationAgent:
    def __init__(self, file_path: str = "resolutions.json"):
        self.file_path = file_path

    def log_resolution(self, incident_description: str, steps: str):
        """Logs the incident and its resolution steps to a JSON file."""
        print(f"--- Documentation Agent logging to: {self.file_path} ---")
        log_entry = {
            "incident": incident_description,
            "resolution_steps": steps,
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }
        try:
            data = []
            if os.path.exists(self.file_path):
                with open(self.file_path, 'r') as f:
                    try:
                        data = json.load(f)
                        if not isinstance(data, list): # Ensure it's a list
                            data = [data]
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode existing JSON in {self.file_path}. Starting fresh list.")
                        data = [] # Reset if file is corrupt
            data.append(log_entry)
            with open(self.file_path, 'w') as f:
                json.dump(data, f, indent=2)
            print("--- Logging successful ---")
        except Exception as e:
            print(f"Error logging resolution: {e}")

# Example Usage (for testing purposes)
if __name__ == "__main__":
    test_incident = "Users are reporting intermittent database connection errors when trying to access the reporting module. This started after the deployment last night."

    print("--- Testing Triage Agent ---")
    triage_agent = TriageAgent()
    context = triage_agent.process_incident(test_incident)
    print(f"Triage Result: {context}")

    if context and context.keywords:
        print("\n--- Testing Knowledge Agent (Mock) ---")
        knowledge = knowledge_agent_mock(context.keywords)
        print(f"Knowledge Result: {knowledge}")

        print("\n--- Testing Search Agent (Mock) ---")
        search = search_agent_mock(context.keywords)
        print(f"Search Result: {search}")

        print("\n--- Testing Resolution Coach Agent ---")
        coach_agent = ResolutionCoachAgent()
        steps = coach_agent.generate_steps(test_incident, context, knowledge, search)
        print(f"Resolution Steps:\n{steps}")

        print("\n--- Testing Documentation Agent ---")
        doc_agent = DocumentationAgent()
        doc_agent.log_resolution(test_incident, steps)
    else:
        print("Triage failed, cannot proceed.")
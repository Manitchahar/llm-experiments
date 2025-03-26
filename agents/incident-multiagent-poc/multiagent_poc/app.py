import chainlit as cl
from agents import (
    TriageAgent,
    knowledge_agent_mock,
    search_agent_mock,
    ResolutionCoachAgent,
    DocumentationAgent,
    IncidentContext # Import the context model
)
import os

# Ensure GOOGLE_API_KEY is set, otherwise provide instructions
if not os.getenv("GOOGLE_API_KEY"):
    print("--------------------------------------------------")
    print("WARNING: GOOGLE_API_KEY environment variable not set.")
    print("Please create a .env file in the multiagent_poc directory with:")
    print("GOOGLE_API_KEY='your_actual_gemini_api_key'")
    print("--------------------------------------------------")
    # You might want to exit or raise an error here in a real app
    # For PoC, we'll let it proceed, but Gemini calls will fail.

@cl.on_chat_start
async def start():
    # Initialize agents (or they can be initialized within on_message)
    cl.user_session.set("triage_agent", TriageAgent())
    cl.user_session.set("coach_agent", ResolutionCoachAgent())
    # Documentation agent needs the path, ensure it's relative to where app.py is run
    cl.user_session.set("doc_agent", DocumentationAgent(file_path="resolutions.json"))

    await cl.Message(
        content="Incident Management Assistant ready. Please describe the incident."
    ).send()

@cl.on_message
async def main(message: cl.Message):
    incident_description = message.content

    # Retrieve agents from session
    triage_agent: TriageAgent = cl.user_session.get("triage_agent")
    coach_agent: ResolutionCoachAgent = cl.user_session.get("coach_agent")
    doc_agent: DocumentationAgent = cl.user_session.get("doc_agent")

    context: Optional[IncidentContext] = None
    knowledge_info: str = "Not available"
    search_info: str = "Not available"
    resolution_steps: str = "Could not generate resolution steps."

    # --- Triage Step ---
    async with cl.Step(name="Triage Incident") as triage_step:
        triage_step.input = incident_description
        try:
            # Run triage agent (synchronous function needs to be run in executor)
            context = await cl.make_async(triage_agent.process_incident)(incident_description)
            if context and context.severity and context.keywords:
                triage_step.output = f"Severity: {context.severity}\nKeywords: {', '.join(context.keywords)}"
            else:
                 triage_step.output = "Triage failed to extract context."
                 await cl.Message(content="Triage failed. Cannot proceed.").send()
                 return # Stop processing if triage fails
        except Exception as e:
            triage_step.output = f"Triage Error: {e}"
            await cl.Message(content=f"An error occurred during triage: {e}").send()
            return

    # --- Information Gathering Step (Knowledge & Search - Mocked) ---
    if context and context.keywords:
        async with cl.Step(name="Gather Information (Mocked)") as gather_step:
            gather_step.input = f"Keywords: {', '.join(context.keywords)}"
            # Run mock agents (synchronous functions need to be run in executor)
            knowledge_info = await cl.make_async(knowledge_agent_mock)(context.keywords)
            search_info = await cl.make_async(search_agent_mock)(context.keywords)
            gather_step.output = f"Knowledge Base Info:\n{knowledge_info}\n\nExternal Search Info:\n{search_info}"
    else:
         # This case should ideally be caught by the triage failure check above,
         # but added for robustness.
         await cl.Message(content="Cannot gather information without triage context.").send()
         return


    # --- Resolution Coach Step ---
    if context: # Ensure context is not None
        async with cl.Step(name="Generate Resolution Steps") as coach_step:
             coach_step.input = f"Incident: {incident_description}\nKnowledge: {knowledge_info}\nSearch: {search_info}"
             try:
                # Run coach agent (synchronous function needs to be run in executor)
                resolution_steps = await cl.make_async(coach_agent.generate_steps)(
                    incident_description=incident_description,
                    triage_context=context,
                    knowledge_info=knowledge_info,
                    search_info=search_info
                )
                coach_step.output = resolution_steps
             except Exception as e:
                coach_step.output = f"Resolution Coach Error: {e}"
                resolution_steps = f"Error generating steps: {e}" # Update variable for logging

    # --- Documentation Step ---
    async with cl.Step(name="Document Resolution") as doc_step:
        doc_step.input = f"Incident: {incident_description}\nSteps: {resolution_steps}"
        try:
            # Run documentation agent (synchronous function needs to be run in executor)
            await cl.make_async(doc_agent.log_resolution)(
                incident_description=incident_description,
                steps=resolution_steps
            )
            doc_step.output = f"Incident and steps logged to {doc_agent.file_path}"
        except Exception as e:
            doc_step.output = f"Documentation Error: {e}"

    # --- Final Output ---
    await cl.Message(
        content=f"**Resolution Steps:**\n{resolution_steps}"
    ).send()
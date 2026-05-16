import os
from typing import Annotated, List, TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


llm = ChatGroq(
    model="llama-3.3-70b-specdec",
    temperature=0.5,
    groq_api_key=require_env("GROQ_API_KEY"),
)

tool = TavilySearchResults(
    max_results=3,
    tavily_api_key=require_env("TAVILY_API_KEY"),
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                "You are a Twitter expert assigned to craft outstanding tweets. "
                "Generate the most engaging and impactful tweet possible based on the user's request. "
                "If the user provides feedback, refine and enhance your previous attempts accordingly."
            )
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
generate_chain = generation_prompt | llm

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                "You are a Twitter influencer known for engaging content and sharp insights. "
                "Review and critique the user's tweet with specific suggestions to improve depth, style, and impact."
            )
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
reflect_chain = reflection_prompt | llm


class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    research: List[str]


def research_node(state: State):
    """Fetch research and inject it into the message flow."""
    topic = state["messages"][-1].content
    results = tool.invoke({"query": f"Latest information about {topic}"})
    research_contents = [res["content"] for res in results]
    research_message = SystemMessage(
        content="RELEVANT RESEARCH:\n"
        + "\n\n".join(f"- {content[:300]}..." for content in research_contents)
    )
    return {
        "research": research_contents,
        "messages": [*state["messages"], research_message],
    }


def generation_node(state: State):
    """Generate a tweet with the conversation and research context."""
    result = generate_chain.invoke({"messages": state["messages"]})
    return {"messages": [*state["messages"], result]}


def reflection_node(state: State):
    """Provide constructive feedback on the generated tweet."""
    messages = state["messages"]
    feedback_context = [
        HumanMessage(content=msg.content)
        if isinstance(msg, AIMessage)
        else AIMessage(content=msg.content)
        for msg in messages
        if not isinstance(msg, SystemMessage)
    ]

    result = reflect_chain.invoke({"messages": feedback_context})
    return {"messages": [*messages, HumanMessage(content=result.content)]}


builder = StateGraph(State)
builder.add_node("research", research_node)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.set_entry_point("research")
builder.add_edge("research", "generate")

MAX_ITERATIONS = 3


def should_continue(state: State):
    ai_count = sum(1 for msg in state["messages"] if isinstance(msg, AIMessage))
    return END if ai_count >= MAX_ITERATIONS else "reflect"


builder.add_conditional_edges("generate", should_continue)
builder.add_edge("reflect", "generate")
graph = builder.compile()


def generate_tweet(topic: str):
    """Generate a tweet and return the graph response."""
    return graph.invoke(
        {
            "messages": [HumanMessage(content=f"Create a tweet about {topic}")],
            "research": [],
        }
    )


if __name__ == "__main__":
    st.set_page_config(page_title="FastTweet", page_icon="✍️")

    st.title("FastTweet - AI Tweet Generator")
    st.write("Generate researched tweet drafts with an AI critique loop.")

    topic = st.text_input(
        "Enter tweet topic:",
        placeholder="e.g., AI agents, climate tech, startup lessons",
    )

    if st.button("Generate Tweet"):
        if not topic:
            st.error("Please enter a topic first.")
            st.stop()

        with st.spinner("Generating tweet..."):
            try:
                response = generate_tweet(topic)
                messages = response.get("messages", [])
                research = response.get("research", [])

                final_tweet = ""
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage):
                        final_tweet = msg.content
                        break

                st.markdown("## Final Tweet")
                st.success(final_tweet or "No tweet generated.")

                with st.expander("Research Sources", expanded=False):
                    if research:
                        for i, res in enumerate(research, 1):
                            st.write(f"**Source {i}:**")
                            st.write(res[:250] + "...")
                            st.divider()
                    else:
                        st.write("No research sources available.")

                with st.expander("Generation Process", expanded=False):
                    for msg in messages:
                        if isinstance(msg, AIMessage):
                            st.markdown("**AI:**")
                        elif isinstance(msg, HumanMessage):
                            st.markdown("**User:**")
                        else:
                            st.markdown("**System:**")
                        st.code(msg.content)
                        st.divider()

            except Exception as exc:
                import traceback

                st.error(f"An error occurred: {exc}")
                st.code(traceback.format_exc(), language="python")

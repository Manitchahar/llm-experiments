import os
from typing import List, Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize components
llm = ChatGroq(
    model='llama-3.3-70b-specdec', 
    temperature=0.5,
    groq_api_key=os.getenv('GROQ_API_KEY')
)
tool = TavilySearchResults(
    max_results=3,
    tavily_api_key=os.getenv('TAVILY_API_KEY')
)

# Define prompts (keep unchanged as requested)
generation_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(
        content='''You are a Twitter expert assigned to craft outstanding tweets.
            Generate the most engaging and impactful tweet possible based on the user's request.
            If the user provides feedback, refine and enhance your previous attempts accordingly for maximum engagement.'''
    ),
    MessagesPlaceholder(variable_name='messages'),
])
generate_chain = generation_prompt | llm

reflection_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(
        content='''You are a Twitter influencer known for your engaging content and sharp insights.
            Review and critique the user's tweet.
            Provide constructive feedback, focusing on enhancing its depth, style, and overall impact.
            Offer specific suggestions to make the tweet more compelling and engaging for their audience.'''
    ),
    MessagesPlaceholder(variable_name='messages'),
])
reflect_chain = reflection_prompt | llm

# State definition
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    research: List[str]

# Core functions (improved)
def research_node(state: State):
    """Fetch research and inject into message flow"""
    topic = state["messages"][-1].content
    results = tool.invoke({"query": f"Latest information about {topic}"})
    research_contents = [res["content"] for res in results]
    research_message = SystemMessage(
        content="RELEVANT RESEARCH:\n" + "\n\n".join(
            f"• {content[:300]}..." for content in research_contents
        )
    )
    return {
        "research": research_contents,
        "messages": [*state["messages"], research_message]
    }

def generation_node(state: State):
    """Generate tweet with full context including research"""
    result = generate_chain.invoke({'messages': state["messages"]})
    return {"messages": [*state["messages"], result]}

def reflection_node(state: State):
    """Provide constructive feedback on generated tweet"""
    messages = state["messages"]
    
    # Create mirrored perspective for feedback
    feedback_context = [
        HumanMessage(content=msg.content) if isinstance(msg, AIMessage)
        else AIMessage(content=msg.content)
        for msg in messages if not isinstance(msg, SystemMessage)
    ]
    
    # Get feedback from reflection chain
    res = reflect_chain.invoke({'messages': feedback_context})
    return {"messages": [*messages, HumanMessage(content=res.content)]}

# Build graph with improved flow
builder = StateGraph(State)
builder.add_node("do_research", research_node)      # Changed from "research"
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)

# Set up workflow
builder.set_entry_point("do_research")              # Changed from "research"
builder.add_edge("do_research", "generate")         # Changed from "research"

# Improved conditional branching
MAX_ITERATIONS = 3
def should_continue(state: State):
    ai_count = sum(1 for msg in state["messages"] if isinstance(msg, AIMessage))
    return END if ai_count >= MAX_ITERATIONS else "reflect"

builder.add_conditional_edges("generate", should_continue)
builder.add_edge("reflect", "generate")

graph = builder.compile()

import streamlit as st

# Enhanced execution with better output formatting
def generate_tweet(topic: str):
    """Generate tweet and return the full response"""
    response = graph.invoke({
        "messages": [HumanMessage(content=f"Create a tweet about {topic}")],
        "research": []  # Initialize research list
    })
    return response

if __name__ == "__main__":
    st.set_page_config(page_title="TweetBot", page_icon="🐦")
    
    st.title("🐦 TweetBot - AI Tweet Generator")
    st.write("Generate engaging tweets with AI assistance")
    
    # User input
    topic = st.text_input("Enter tweet topic:", placeholder="e.g., AI technology, climate change, etc.")
    
    if st.button("Generate Tweet"):
        if topic:
            with st.spinner("Generating tweet..."):
                try:
                    response = generate_tweet(topic)
                    
                    # Extract messages and research from response
                    messages = response.get("messages", [])
                    research = response.get("research", [])
                    
                    # Find and display the final tweet
                    final_tweet = ""
                    for msg in reversed(messages):
                        if isinstance(msg, AIMessage):
                            final_tweet = msg.content
                            break
                    
                    # Display final tweet first
                    st.markdown("## 🐦 Final Tweet")
                    st.success(final_tweet or "No tweet generated.")
                    
                    # Display research summary
                    with st.expander("📚 Research Sources", expanded=False):
                        if research:
                            for i, res in enumerate(research, 1):
                                st.write(f"**Source {i}:**")
                                st.write(res[:250] + "...")
                                st.divider()
                        else:
                            st.write("No research sources available.")
                    
                    # Display generation process
                    with st.expander("🔄 Generation Process", expanded=False):
                        for msg in messages:
                            if isinstance(msg, AIMessage):
                                st.markdown("**🤖 AI:**")
                            elif isinstance(msg, HumanMessage):
                                st.markdown("**🧑 User:**")
                            else:  # SystemMessage
                                st.markdown("**📚 System:**")
                            st.code(msg.content)
                            st.divider()
                
                except Exception as e:
                    import traceback
                    st.error(f"An error occurred: {str(e)}")
                    st.code(traceback.format_exc(), language="python")
        else:
            st.error("Please enter a topic first!")
from crewai import Agent
from tools import yt_tool
from langchain import ChatGroq

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.5,
    api_key="REDACTED_SECRET"
)

blog_researcher = Agent(
    role='blog researcher from Youtube videos',
    goal='Get relevant video content on the topic {topic} from YT channel',
    verbose=True,
    memory=True,
    backstory=(
        "Expert in understanding videos in AI, Data Science, Machine Learning "
        "and Gen AI and providing suggestions"
    ),
    llm=llm,
    tools=[yt_tool],
    allow_delegation=False,
)

blog_writer = Agent(
    role='blog writer',
    goal='Narrate compelling tech stories about the video {topic} from YT channel',
    verbose=True,
    memory=True,
    backstory=(
        "Expert in understanding videos in AI, Data Science, Machine Learning "
        "and Gen AI and providing suggestions"
    ),
    llm=llm,
    tools=[yt_tool],
    allow_delegation=False,
)
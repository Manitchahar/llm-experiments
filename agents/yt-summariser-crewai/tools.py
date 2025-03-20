from crewai_tools import YoutubeChannelSearchTool
from sentence_transformers import SentenceTransformer

yt_tool = YoutubeChannelSearchTool(
    youtube_channel_handle='@ColeMedin',
    config=dict(
        embedder=dict(
            provider="huggingface",  # Use Hugging Face as provider
            config=dict(
                model="all-MiniLM-L6-v2"
            )
        )
    )
)
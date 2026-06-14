import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
LLM_MODEL: str = "anthropic/claude-3.5-haiku"

TAVILY_MAX_RESULTS: int = 6
RAG_TOP_K: int = 5
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 150

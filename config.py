"""Central application configuration."""
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() in {"1", "true", "yes", "on"}

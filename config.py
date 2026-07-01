"""Environment configuration and the shared Ollama client."""
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")
PORT = int(os.getenv("PORT", "8001"))
HOST = os.getenv("HOST", "0.0.0.0")

client = AsyncOpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

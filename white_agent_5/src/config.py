import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Project Paths
    ROOT_DIR = Path(__file__).parent.parent
    DATA_DIR = ROOT_DIR / "data"
    
    # Server Settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 9002))
    
    # LLM Settings
    DEFAULT_PROVIDER = "openrouter"
    DEFAULT_MODEL = "deepseek/deepseek-chat"

    # RAG Settings
    MAX_SEARCH_RESULTS = 10  # Reduced from 15 to fit in 100s timeout
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

config = Config()

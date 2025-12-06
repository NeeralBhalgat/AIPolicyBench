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
    DEFAULT_PROVIDER = "openai"
    DEFAULT_MODEL = "gpt-4o"
    
    # RAG Settings
    MAX_SEARCH_RESULTS = 15
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

config = Config()

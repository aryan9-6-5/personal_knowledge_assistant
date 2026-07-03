import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env from project root (one level up from backend/)
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    GROQ_API_KEY: str = ""
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL: str = "llama-3.3-70b-versatile"

    # Chunking
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150

    # Retrieval
    TOP_K: int = 5
    RERANK_TOP_K: int = 3
    SEMANTIC_WEIGHT: float = 0.6
    KEYWORD_WEIGHT: float = 0.4

    # LLM
    MAX_TOKENS: int = 2048
    TEMPERATURE: float = 0.1
    MAX_CONVERSATION_HISTORY: int = 10

    # Storage
    CHROMA_PERSIST_DIR: str = ".chroma_db"
    UPLOAD_DIR: str = "uploads"
    COLLECTION_NAME: str = "knowledge_base"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

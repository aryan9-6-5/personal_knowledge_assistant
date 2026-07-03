"""Local embedding service using sentence-transformers."""

import logging
from typing import List
from sentence_transformers import SentenceTransformer
from config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generates embeddings locally using sentence-transformers."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL, device="cpu")
        self.dimension = self.model.get_sentence_embedding_dimension()
        self._initialized = True
        logger.info(f"Embedding model loaded. Dimension: {self.dimension}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts, returns list of float vectors."""
        if not texts:
            return []
        embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return [emb.tolist() for emb in embeddings]

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""
        return self.embed([query])[0]

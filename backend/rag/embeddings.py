"""Local embedding service: sentence-transformers (default) or ONNX via fastembed.

Set EMBEDDING_BACKEND=onnx for low-memory hosts. It runs the same MiniLM weights
without PyTorch, so vectors are compatible with the default backend."""

import logging
from typing import List
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
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL} ({settings.EMBEDDING_BACKEND})")
        self.backend = settings.EMBEDDING_BACKEND
        if self.backend == "onnx":
            from fastembed import TextEmbedding
            self.model = TextEmbedding(settings.EMBEDDING_MODEL)
            self.dimension = len(next(iter(self.model.embed(["dimension probe"]))))
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(settings.EMBEDDING_MODEL, device="cpu")
            self.dimension = self.model.get_sentence_embedding_dimension()
        self._initialized = True
        logger.info(f"Embedding model loaded. Dimension: {self.dimension}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts, returns list of float vectors."""
        if not texts:
            return []
        if self.backend == "onnx":
            # fastembed returns L2-normalised vectors for MiniLM
            return [emb.tolist() for emb in self.model.embed(texts)]
        embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return [emb.tolist() for emb in embeddings]

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""
        return self.embed([query])[0]

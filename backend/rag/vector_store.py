"""ChromaDB vector store with hybrid search using Reciprocal Rank Fusion."""

import hashlib
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi

from config import settings
from rag.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages document storage and hybrid retrieval with ChromaDB."""

    def __init__(self):
        self.embedding_service = EmbeddingService()

        self.client = chromadb.Client(
            ChromaSettings(
                anonymized_telemetry=False,
                persist_directory=settings.CHROMA_PERSIST_DIR,
                is_persistent=True,
            )
        )

        self.collection = self.client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        # In-memory BM25 indices scoped per user — rebuilt on startup and after doc changes
        self._bm25_corpora: Dict[str, List[Dict[str, Any]]] = {}
        self._bm25_indices: Dict[str, BM25Okapi] = {}
        self._rebuild_bm25_index()

        logger.info(
            f"VectorStore initialized. Collection '{settings.COLLECTION_NAME}' "
            f"has {self.collection.count()} vectors."
        )

    def _rebuild_bm25_index(self):
        """Rebuild the BM25 index from stored documents, grouped by user."""
        try:
            result = self.collection.get(include=["documents", "metadatas"])
            self._bm25_corpora = {}
            self._bm25_indices = {}
            if result["documents"]:
                # Group by user_id
                for doc, meta, id_ in zip(
                    result["documents"], result["metadatas"], result["ids"]
                ):
                    user_id = str(meta.get("user_id", "public"))
                    if user_id not in self._bm25_corpora:
                        self._bm25_corpora[user_id] = []
                    self._bm25_corpora[user_id].append({
                        "content": doc,
                        "metadata": meta,
                        "id": id_,
                    })
                
                # Build BM25 index for each user's corpus
                for user_id, corpus in self._bm25_corpora.items():
                    tokenized = [doc["content"].lower().split() for doc in corpus]
                    self._bm25_indices[user_id] = BM25Okapi(tokenized)
            else:
                self._bm25_corpora = {}
                self._bm25_indices = {}
        except Exception as e:
            logger.error(f"BM25 index rebuild failed: {e}")
            self._bm25_corpora = {}
            self._bm25_indices = {}

    def add_chunks(
        self,
        chunks: List[str],
        metadatas: List[Dict[str, Any]],
        doc_id: str,
        user_id: str,
    ) -> int:
        """Add document chunks to the vector store. Returns count of chunks added."""
        if not chunks:
            return 0

        # Generate embeddings
        embeddings = self.embedding_service.embed(chunks)

        # Generate unique IDs based on content hash + index
        ids = []
        for i, chunk in enumerate(chunks):
            content_hash = hashlib.md5(chunk.encode()).hexdigest()[:12]
            ids.append(f"{doc_id}_{content_hash}_{i}")

        # Enrich metadata
        enriched_metadatas = []
        for i, meta in enumerate(metadatas):
            enriched_metadatas.append(
                {
                    **meta,
                    "doc_id": doc_id,
                    "user_id": user_id,
                    "chunk_index": i,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        # Upsert to ChromaDB
        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=enriched_metadatas,
        )

        # Rebuild BM25 index
        self._rebuild_bm25_index()

        logger.info(f"Added {len(chunks)} chunks for document {doc_id} of user {user_id}")
        return len(chunks)

    def delete_document(self, doc_id: str):
        """Delete all chunks belonging to a document."""
        try:
            results = self.collection.get(
                where={"doc_id": doc_id},
                include=["metadatas"],
            )
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                self._rebuild_bm25_index()
                logger.info(f"Deleted {len(results['ids'])} chunks for doc {doc_id}")
        except Exception as e:
            logger.error(f"Delete error for doc {doc_id}: {e}")

    def semantic_search(
        self, query_embedding: List[float], user_id: str, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Pure semantic (vector) search scoped to a user."""
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            where={"user_id": user_id},
            include=["documents", "metadatas", "distances"],
        )

        items = []
        for i in range(len(results["ids"][0])):
            items.append(
                {
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": 1.0 - results["distances"][0][i],  # cosine distance → similarity
                    "id": results["ids"][0][i],
                }
            )
        return items

    def keyword_search(self, query: str, user_id: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """BM25 keyword search scoped to a user."""
        bm25_index = self._bm25_indices.get(user_id)
        bm25_corpus = self._bm25_corpora.get(user_id, [])
        if not bm25_index or not bm25_corpus:
            return []

        query_tokens = query.lower().split()
        scores = bm25_index.get_scores(query_tokens)

        scored_docs = [
            {**bm25_corpus[i], "score": float(scores[i])}
            for i in range(len(scores))
            if scores[i] > 0
        ]
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:top_k]

    def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        user_id: str,
        top_k: int = 10,
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining semantic and keyword results
        using Reciprocal Rank Fusion (RRF) scoped to a user.
        """
        semantic_results = self.semantic_search(query_embedding, user_id, top_k * 2)
        keyword_results = self.keyword_search(query, user_id, top_k * 2)

        # Build RRF scores
        rrf_scores: Dict[str, float] = {}
        content_map: Dict[str, Dict[str, Any]] = {}

        for rank, item in enumerate(semantic_results):
            key = item["id"]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (rrf_k + rank + 1)
            content_map[key] = item

        for rank, item in enumerate(keyword_results):
            key = item["id"]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (rrf_k + rank + 1)
            if key not in content_map:
                content_map[key] = item

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

        results = []
        for id_ in sorted_ids[:top_k]:
            item = content_map[id_]
            item["rrf_score"] = rrf_scores[id_]
            results.append(item)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get global collection statistics (used for general checks)."""
        count = self.collection.count()

        # Count unique documents
        doc_ids = set()
        if count > 0:
            try:
                result = self.collection.get(include=["metadatas"])
                for meta in result["metadatas"]:
                    if "doc_id" in meta:
                        doc_ids.add(meta["doc_id"])
            except Exception:
                pass

        return {
            "total_chunks": count,
            "total_documents": len(doc_ids),
            "status": "ready",
        }

    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get collection statistics for a specific user."""
        try:
            results = self.collection.get(
                where={"user_id": user_id},
                include=["metadatas"],
            )
            count = len(results["ids"])
            doc_ids = set()
            for meta in results["metadatas"]:
                if "doc_id" in meta:
                    doc_ids.add(meta["doc_id"])
            return {
                "total_chunks": count,
                "total_documents": len(doc_ids),
                "status": "ready",
            }
        except Exception as e:
            logger.error(f"Failed to get user stats: {e}")
            return {
                "total_chunks": 0,
                "total_documents": 0,
                "status": "ready",
            }

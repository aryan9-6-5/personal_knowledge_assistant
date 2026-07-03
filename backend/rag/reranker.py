"""Cross-encoder re-ranking for improved retrieval precision."""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Try to load cross-encoder; fall back to score-based ranking if unavailable
_reranker_model = None
_reranker_available = False

def _load_reranker():
    global _reranker_model, _reranker_available
    try:
        from sentence_transformers import CrossEncoder
        _reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
        _reranker_available = True
        logger.info("Cross-encoder re-ranker loaded successfully.")
    except Exception as e:
        logger.warning(f"Cross-encoder unavailable, using score-based ranking: {e}")
        _reranker_available = False


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Re-rank candidates using a cross-encoder model.
    Falls back to existing scores if the model isn't available.
    """
    global _reranker_model, _reranker_available

    if not candidates:
        return []

    # Lazy load
    if _reranker_model is None and not _reranker_available:
        _load_reranker()

    if _reranker_available and _reranker_model is not None:
        # Prepare query-document pairs
        pairs = [(query, candidate["content"]) for candidate in candidates]

        try:
            scores = _reranker_model.predict(pairs)

            for i, score in enumerate(scores):
                candidates[i]["rerank_score"] = float(score)

            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
            return candidates[:top_k]
        except Exception as e:
            logger.error(f"Re-ranking failed, falling back: {e}")

    # Fallback: use existing scores (rrf_score or score)
    candidates.sort(
        key=lambda x: x.get("rrf_score", x.get("score", 0)),
        reverse=True,
    )
    return candidates[:top_k]

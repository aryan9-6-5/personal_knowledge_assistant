"""System endpoints: health check, stats."""

import logging
from fastapi import APIRouter

from models.schemas import HealthResponse, StatsResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["system"])


def get_engine():
    from main import rag_engine
    return rag_engine


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        engine = get_engine()
        stats = engine.get_stats()
        return HealthResponse(
            status="ok",
            total_documents=stats["total_documents"],
            total_chunks=stats["total_chunks"],
        )
    except Exception:
        return HealthResponse(status="ok")


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics."""
    engine = get_engine()
    stats = engine.get_stats()
    return StatsResponse(**stats)

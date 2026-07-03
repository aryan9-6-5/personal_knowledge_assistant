"""FastAPI application entry point."""

import logging
import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Global RAG engine instance
rag_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG engine on startup, cleanup on shutdown."""
    global rag_engine
    from rag.engine import RAGEngine
    from rag.reranker import _load_reranker

    logger.info("Starting RAG engine initialization...")
    rag_engine = RAGEngine()
    logger.info("RAG engine ready.")
    
    logger.info("Eagerly loading cross-encoder re-ranker...")
    _load_reranker()
    
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Personal Knowledge Assistant API",
    description="Advanced RAG-powered document Q&A system",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
from routers.documents import router as documents_router
from routers.chat import router as chat_router
from routers.system import router as system_router

app.include_router(documents_router)
app.include_router(chat_router)
app.include_router(system_router)

# Serve frontend static files in production
frontend_dist = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dist")
if os.path.isdir(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

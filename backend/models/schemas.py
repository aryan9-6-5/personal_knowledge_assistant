"""Pydantic models for API request/response schemas."""

from pydantic import BaseModel
from typing import List, Optional


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None


class UrlIngestRequest(BaseModel):
    url: str


class SourceInfo(BaseModel):
    source: str
    page: int
    relevance_score: float
    content: str


class DocumentInfo(BaseModel):
    id: str
    name: str
    chunks: int
    status: str
    created_at: str = ""


class DocumentUploadResponse(BaseModel):
    processed: int
    documents: List[DocumentInfo]


class StatsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    status: str
    active_conversations: int = 0


class HealthResponse(BaseModel):
    status: str
    total_documents: int = 0
    total_chunks: int = 0

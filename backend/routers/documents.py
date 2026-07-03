"""Document management API endpoints."""

import logging
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException

from models.schemas import DocumentInfo, DocumentUploadResponse, UrlIngestRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["documents"])


def get_engine():
    """Get RAG engine from app state. Set in main.py lifespan."""
    from main import rag_engine
    return rag_engine


@router.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """List all ingested documents."""
    engine = get_engine()
    docs = engine.get_documents()
    return [
        DocumentInfo(
            id=d.id,
            name=d.name,
            chunks=d.chunks,
            status=d.status,
            created_at=d.created_at,
        )
        for d in docs
    ]


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process document files (PDF, TXT, MD)."""
    engine = get_engine()
    results = []

    for file in files:
        try:
            content = await file.read()
            file_path = await engine.doc_processor.save_uploaded_file(content, file.filename)
            record = await engine.add_document(file_path, file.filename)
            results.append(
                DocumentInfo(
                    id=record.id,
                    name=record.name,
                    chunks=record.chunks,
                    status=record.status,
                    created_at=record.created_at,
                )
            )
        except Exception as e:
            logger.error(f"Failed to process {file.filename}: {e}")
            results.append(
                DocumentInfo(
                    id="error",
                    name=file.filename,
                    chunks=0,
                    status="failed",
                )
            )

    processed = sum(1 for r in results if r.status == "processed")
    return DocumentUploadResponse(processed=processed, documents=results)


@router.post("/documents/url", response_model=DocumentInfo)
async def ingest_url(req: UrlIngestRequest):
    """Ingest content from a web URL."""
    engine = get_engine()
    try:
        record = await engine.add_url(req.url)
        return DocumentInfo(
            id=record.id,
            name=record.name,
            chunks=record.chunks,
            status=record.status,
            created_at=record.created_at,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest URL: {str(e)}")


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and its chunks."""
    engine = get_engine()
    if engine.delete_document(doc_id):
        return {"success": True}
    raise HTTPException(status_code=404, detail="Document not found")

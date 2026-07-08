"""Chat API endpoint with SSE streaming."""

import json
import logging
from fastapi import APIRouter, Request, Depends
from fastapi.responses import StreamingResponse

from models.schemas import ChatRequest
from auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


def get_engine():
    from main import rag_engine
    return rag_engine


@router.post("/chat")
async def chat(
    req: ChatRequest,
    current_user: dict = Depends(get_current_user)
):
    """Query the knowledge base with SSE streaming response scoped to current user."""
    engine = get_engine()

    async def event_stream():
        try:
            async for event in engine.query_stream(req.message, current_user["id"], req.conversation_id):
                event_type = event["event"]
                data = json.dumps(event["data"])
                yield f"event: {event_type}\ndata: {data}\n\n"
        except Exception as e:
            logger.error(f"Chat stream error: {e}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete("/chat/history")
async def clear_history(current_user: dict = Depends(get_current_user)):
    """Clear conversation history and uploaded documents for the current user."""
    engine = get_engine()
    engine.clear_conversation(current_user["id"])
    engine.clear_all_documents(current_user["id"])
    return {"success": True}

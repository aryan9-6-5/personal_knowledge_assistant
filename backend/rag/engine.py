"""Core RAG engine — orchestrates the full retrieval-augmented generation pipeline."""

import logging
import uuid
from typing import List, Dict, Any, AsyncGenerator, Optional
from dataclasses import dataclass, field

from config import settings
from rag.embeddings import EmbeddingService
from rag.vector_store import VectorStore
from rag.document_processor import DocumentProcessor
from rag.reranker import rerank
from rag.llm import LLMService

logger = logging.getLogger(__name__)


@dataclass
class DocumentRecord:
    """Tracks an ingested document."""
    id: str
    name: str
    chunks: int
    status: str  # processing | processed | failed
    created_at: str = ""


class RAGEngine:
    """
    Advanced RAG pipeline:
    1. Query rephrasing
    2. HyDE (Hypothetical Document Embedding)
    3. Hybrid retrieval (semantic + BM25 with RRF)
    4. Cross-encoder re-ranking
    5. Context assembly
    6. Streaming LLM generation with source attribution
    """

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.doc_processor = DocumentProcessor()
        self.llm = LLMService()

        logger.info("RAG Engine initialized.")

    async def add_document(self, file_path: str, filename: str, user_id: int) -> DocumentRecord:
        """Process and add a document to the knowledge base."""
        doc_id = str(uuid.uuid4())
        
        # Save initial document record in database
        import database
        from datetime import datetime
        created_at = datetime.now().isoformat()
        database.add_document(
            doc_id=doc_id,
            user_id=user_id,
            name=filename,
            chunks=0,
            status="processing",
            created_at=created_at,
            file_path=file_path
        )

        try:
            chunks, metadatas = self.doc_processor.process_file(file_path, filename)
            # Pass user_id to vector store!
            chunk_count = self.vector_store.add_chunks(chunks, metadatas, doc_id, str(user_id))

            # Update document status in database
            database.update_document_status(doc_id, user_id, "processed", chunk_count)
            logger.info(f"Document '{filename}' added: {chunk_count} chunks for user {user_id}")
            
            return DocumentRecord(
                id=doc_id,
                name=filename,
                chunks=chunk_count,
                status="processed",
                created_at=created_at
            )
        except Exception as e:
            database.update_document_status(doc_id, user_id, "failed", 0)
            logger.error(f"Failed to process '{filename}': {e}")
            raise

    async def add_url(self, url: str, user_id: int) -> DocumentRecord:
        """Ingest a web URL into the knowledge base."""
        doc_id = str(uuid.uuid4())
        from urllib.parse import urlparse
        parsed = urlparse(url)
        name = f"{parsed.hostname}{parsed.path}"
        from datetime import datetime
        created_at = datetime.now().isoformat()

        import database
        database.add_document(
            doc_id=doc_id,
            user_id=user_id,
            name=name,
            chunks=0,
            status="processing",
            created_at=created_at
        )

        try:
            chunks, metadatas = self.doc_processor.process_url(url)
            chunk_count = self.vector_store.add_chunks(chunks, metadatas, doc_id, str(user_id))
            
            database.update_document_status(doc_id, user_id, "processed", chunk_count)
            
            return DocumentRecord(
                id=doc_id,
                name=name,
                chunks=chunk_count,
                status="processed",
                created_at=created_at
            )
        except Exception as e:
            database.update_document_status(doc_id, user_id, "failed", 0)
            logger.error(f"Failed to ingest URL '{url}': {e}")
            raise

    def delete_document(self, doc_id: str, user_id: int) -> bool:
        """Remove a document from the knowledge base."""
        import database
        doc = database.get_document(doc_id, user_id)
        if not doc:
            return False
        
        # Delete from ChromaDB
        self.vector_store.delete_document(doc_id)
        
        # Delete from SQLite
        database.delete_user_document(doc_id, user_id)
        
        # Delete file if exists
        file_path = doc.get("file_path")
        if file_path and os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except Exception as e:
                logger.error(f"Failed to delete file {file_path}: {e}")
        return True

    def clear_all_documents(self, user_id: int):
        """Remove all documents for a user from database, ChromaDB, and uploads folder."""
        import database
        docs = database.get_user_documents(user_id)
        for doc in docs:
            self.delete_document(doc["id"], user_id)

    def get_documents(self, user_id: int) -> List[DocumentRecord]:
        """List all documents for a user."""
        import database
        docs = database.get_user_documents(user_id)
        return [
            DocumentRecord(
                id=d["id"],
                name=d["name"],
                chunks=d["chunks"],
                status=d["status"],
                created_at=d["created_at"]
            )
            for d in docs
        ]

    async def _retrieve(self, query: str, user_id: int) -> List[Dict[str, Any]]:
        """
        Full retrieval pipeline:
        1. Query rephrasing
        2. HyDE (Hypothetical Document Embedding)
        3. Hybrid search (semantic + BM25 with RRF)
        4. Cross-encoder re-ranking
        5. Context assembly
        """
        # Step 1: Rephrase for better retrieval
        rephrased = await self.llm.rephrase_query(query)
        logger.info(f"Rephrased query: '{query}' → '{rephrased}'")

        # Step 2: HyDE — embed hypothetical answer for better semantic match
        hyde_answer = await self.llm.generate_hypothetical_answer(query)
        hyde_embedding = self.embedding_service.embed_query(hyde_answer)

        # Also embed the rephrased query
        query_embedding = self.embedding_service.embed_query(rephrased)

        # Step 3: Hybrid search with both embeddings, take the best
        results_hyde = self.vector_store.hybrid_search(
            rephrased, hyde_embedding, str(user_id), top_k=settings.TOP_K * 2
        )
        results_query = self.vector_store.hybrid_search(
            rephrased, query_embedding, str(user_id), top_k=settings.TOP_K * 2
        )

        # Merge and deduplicate
        seen_ids = set()
        merged = []
        for item in results_hyde + results_query:
            if item["id"] not in seen_ids:
                seen_ids.add(item["id"])
                merged.append(item)

        # Step 4: Re-rank
        reranked = rerank(query, merged, top_k=settings.RERANK_TOP_K)

        return reranked

    async def query_stream(
        self,
        question: str,
        user_id: int,
        conversation_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Full RAG query with streaming. Yields events:
        - {"event": "token", "data": {"token": "..."}}
        - {"event": "sources", "data": {"sources": [...]}}
        - {"event": "done", "data": {"conversation_id": "..."}}
        """
        import database
        conv_id = conversation_id or str(uuid.uuid4())

        # If it's a new conversation, create it in database
        if not database.conversation_exists(conv_id, user_id):
            title = question[:30] + "..." if len(question) > 30 else question
            database.create_conversation(conv_id, user_id, title)

        # Retrieve relevant chunks scoped to user
        relevant_docs = await self._retrieve(question, user_id)

        # Build sources for attribution
        sources = []
        for doc in relevant_docs:
            meta = doc.get("metadata", {})
            sources.append({
                "source": meta.get("source", "Unknown"),
                "page": meta.get("page", 0),
                "relevance_score": round(
                    doc.get("rerank_score", doc.get("rrf_score", doc.get("score", 0))),
                    3,
                ),
                "content": doc["content"][:300],
            })

        # Build context
        if relevant_docs:
            context = "\n\n---\n\n".join(
                f"[Source: {doc.get('metadata', {}).get('source', 'Unknown')}, "
                f"Page {doc.get('metadata', {}).get('page', '?')}]\n{doc['content']}"
                for doc in relevant_docs
            )
        else:
            context = "No relevant documents found in the knowledge base."

        # Build messages with conversation history from database
        history = database.get_conversation_history(conv_id)

        system_prompt = f"""You are a knowledgeable AI assistant that answers questions based on the provided document context.

Guidelines:
1. Answer based ONLY on the provided context. If the answer isn't in the context, say so clearly.
2. Be thorough but concise. Use markdown formatting for readability.
3. Cite sources naturally (e.g., "According to [document name]...").
4. If multiple sources agree, synthesize the information.
5. Never fabricate information not present in the context.

Document Context:
{context}"""

        messages = [{"role": "system", "content": system_prompt}]
        # Assuming database.get_conversation_history returns a list of dictionaries with 'role' and 'content'
        messages.extend(history[-settings.MAX_CONVERSATION_HISTORY:])
        messages.append({"role": "user", "content": question})

        # Stream the response
        full_response = ""
        async for token in self.llm.generate_stream(messages):
            full_response += token
            yield {"event": "token", "data": {"token": token}}

        # Save user and assistant messages in database
        import json
        user_msg_id = str(uuid.uuid4())
        assistant_msg_id = str(uuid.uuid4())
        
        database.add_message(user_msg_id, conv_id, "user", question)
        database.add_message(
            assistant_msg_id, 
            conv_id, 
            "assistant", 
            full_response, 
            json.dumps(sources) if sources else None
        )

        # Send sources
        if sources:
            yield {"event": "sources", "data": {"sources": sources}}

        yield {"event": "done", "data": {"conversation_id": conv_id}}

    def clear_conversation(self, user_id: int, conversation_id: str = None):
        """Clear conversation history."""
        import database
        database.clear_user_conversations(user_id)

    def get_stats(self, user_id: int) -> Dict[str, Any]:
        """Get system statistics scoped to user."""
        import database
        user_stats = self.vector_store.get_user_stats(str(user_id))
        conversations = database.get_user_conversations(user_id)
        return {
            "total_documents": user_stats["total_documents"],
            "total_chunks": user_stats["total_chunks"],
            "status": user_stats["status"],
            "active_conversations": len(conversations),
        }

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

        # Document registry (in-memory, backed by ChromaDB metadata)
        self.documents: Dict[str, DocumentRecord] = {}
        self._load_document_registry()

        # Conversation histories
        self.conversations: Dict[str, List[Dict[str, str]]] = {}

        logger.info("RAG Engine initialized.")

    def _load_document_registry(self):
        """Rebuild document registry from ChromaDB metadata."""
        stats = self.vector_store.get_stats()
        if stats["total_chunks"] > 0:
            try:
                result = self.vector_store.collection.get(include=["metadatas"])
                doc_chunks: Dict[str, Dict[str, Any]] = {}
                for meta in result["metadatas"]:
                    doc_id = meta.get("doc_id", "unknown")
                    if doc_id not in doc_chunks:
                        doc_chunks[doc_id] = {
                            "name": meta.get("source", "Unknown"),
                            "count": 0,
                            "timestamp": meta.get("timestamp", ""),
                        }
                    doc_chunks[doc_id]["count"] += 1

                for doc_id, info in doc_chunks.items():
                    self.documents[doc_id] = DocumentRecord(
                        id=doc_id,
                        name=info["name"],
                        chunks=info["count"],
                        status="processed",
                        created_at=info["timestamp"],
                    )
                logger.info(f"Loaded {len(self.documents)} documents from registry.")
            except Exception as e:
                logger.error(f"Failed to load document registry: {e}")

    async def add_document(self, file_path: str, filename: str) -> DocumentRecord:
        """Process and add a document to the knowledge base."""
        doc_id = str(uuid.uuid4())
        record = DocumentRecord(
            id=doc_id,
            name=filename,
            chunks=0,
            status="processing",
        )
        self.documents[doc_id] = record

        try:
            chunks, metadatas = self.doc_processor.process_file(file_path, filename)
            chunk_count = self.vector_store.add_chunks(chunks, metadatas, doc_id)

            record.chunks = chunk_count
            record.status = "processed"
            logger.info(f"Document '{filename}' added: {chunk_count} chunks")
        except Exception as e:
            record.status = "failed"
            logger.error(f"Failed to process '{filename}': {e}")
            raise

        return record

    async def add_url(self, url: str) -> DocumentRecord:
        """Ingest a web URL into the knowledge base."""
        doc_id = str(uuid.uuid4())
        from urllib.parse import urlparse
        parsed = urlparse(url)
        name = f"{parsed.hostname}{parsed.path}"

        record = DocumentRecord(id=doc_id, name=name, chunks=0, status="processing")
        self.documents[doc_id] = record

        try:
            chunks, metadatas = self.doc_processor.process_url(url)
            chunk_count = self.vector_store.add_chunks(chunks, metadatas, doc_id)
            record.chunks = chunk_count
            record.status = "processed"
        except Exception as e:
            record.status = "failed"
            logger.error(f"Failed to ingest URL '{url}': {e}")
            raise

        return record

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document from the knowledge base."""
        if doc_id not in self.documents:
            return False
        self.vector_store.delete_document(doc_id)
        del self.documents[doc_id]
        return True

    def clear_all_documents(self):
        """Remove all documents from the knowledge base and clear the uploads folder."""
        # Delete from vector store and memory
        for doc_id in list(self.documents.keys()):
            self.delete_document(doc_id)
            
        # Delete from disk
        import os
        from config import settings
        if os.path.exists(settings.UPLOAD_DIR):
            for filename in os.listdir(settings.UPLOAD_DIR):
                file_path = os.path.join(settings.UPLOAD_DIR, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    logger.error(f"Failed to delete {file_path}: {e}")

    def get_documents(self) -> List[DocumentRecord]:
        """List all documents."""
        return list(self.documents.values())

    async def _retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Full retrieval pipeline:
        1. Rephrase query
        2. Generate HyDE answer and embed it
        3. Hybrid search (semantic + keyword with RRF)
        4. Re-rank with cross-encoder
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
            rephrased, hyde_embedding, top_k=settings.TOP_K * 2
        )
        results_query = self.vector_store.hybrid_search(
            rephrased, query_embedding, top_k=settings.TOP_K * 2
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
        conversation_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Full RAG query with streaming. Yields events:
        - {"event": "token", "data": {"token": "..."}}
        - {"event": "sources", "data": {"sources": [...]}}
        - {"event": "done", "data": {"conversation_id": "..."}}
        """
        conv_id = conversation_id or str(uuid.uuid4())

        # Retrieve relevant chunks
        relevant_docs = await self._retrieve(question)

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

        # Build messages with conversation history
        history = self.conversations.get(conv_id, [])

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
        messages.extend(history[-settings.MAX_CONVERSATION_HISTORY:])
        messages.append({"role": "user", "content": question})

        # Stream the response
        full_response = ""
        async for token in self.llm.generate_stream(messages):
            full_response += token
            yield {"event": "token", "data": {"token": token}}

        # Update conversation history
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": full_response})
        if len(history) > settings.MAX_CONVERSATION_HISTORY * 2:
            history = history[-(settings.MAX_CONVERSATION_HISTORY * 2):]
        self.conversations[conv_id] = history

        # Send sources
        if sources:
            yield {"event": "sources", "data": {"sources": sources}}

        yield {"event": "done", "data": {"conversation_id": conv_id}}

    def clear_conversation(self, conversation_id: str = None):
        """Clear conversation history."""
        if conversation_id:
            self.conversations.pop(conversation_id, None)
        else:
            self.conversations.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        vs_stats = self.vector_store.get_stats()
        return {
            "total_documents": len(self.documents),
            "total_chunks": vs_stats["total_chunks"],
            "status": vs_stats["status"],
            "active_conversations": len(self.conversations),
        }

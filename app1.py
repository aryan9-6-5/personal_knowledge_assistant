
import os
import streamlit as st
import requests
import aiohttp
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import hashlib
from datetime import datetime
import logging
from pathlib import Path
from rank_bm25 import BM25Okapi

# Third-party imports

from chromadb.utils.embedding_functions import HuggingFaceEmbeddingFunction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain.schema import Document
import tiktoken
from chromadb import Client
from chromadb.config import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

@dataclass
class Config:
    """Configuration class for the RAG system"""
    
    GROQ_API_KEY: str = ""
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL: str = "llama3-70b-8192"
    
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_TOKENS: int = 4000
    TOP_K: int = 3
    SIMILARITY_THRESHOLD: float = 0.7
    INDEX_NAME: str = "personal-knowledge-assistant"
    VECTOR_DIMENSION: int = 384
    MAX_RETRIES: int = 3
    BACKOFF_FACTOR: float = 2.0

# =============================================================================
# CORE CLASSES
# =============================================================================
from sentence_transformers import SentenceTransformer

class LocalEmbeddingFunction:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"):
        self.model = SentenceTransformer(model_name, device=device)

    def __call__(self, input):
        return [self.model.encode(text).tolist() for text in input]
class GroqLLMService:
    """Handles LLM interactions using Groq API"""
    
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def generate_response_async(self, messages: List[Dict[str, str]], max_tokens: int = 1000, retries: int = Config.MAX_RETRIES) -> str:
        """Asynchronously generate response using Groq API with retry logic"""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "top_p": 0.9,
            "stream": False
        }
        
        async with aiohttp.ClientSession(headers=self.headers) as session:
            for attempt in range(retries):
                try:
                #    testing-library
                    async with session.post(self.api_url, json=payload) as response:
                        response.raise_for_status()
                        result = await response.json()
                        return result['choices'][0]['message']['content']
                except aiohttp.ClientResponseError as e:
                    if e.status == 429 and attempt < retries - 1:  # Rate limit
                        await asyncio.sleep(Config.BACKOFF_FACTOR ** attempt)
                    else:
                        logger.error(f"Groq API error: {e}")
                        raise Exception(f"Failed to generate response: {e}")
            raise Exception("Max retries exceeded for Groq API")
    
    async def rephrase_query(self, query: str) -> str:
        """Rephrase query for better retrieval"""
        messages = [
            {"role": "system", "content": "Rephrase the following question to make it clearer and more specific for a document search."},
            {"role": "user", "content": f"Original question: {query}"}
        ]
        return await self.generate_response_async(messages, max_tokens=100)

class VectorStore:
    """Manages vector storage and retrieval using Chroma DB"""

    def __init__(self, embedding_function, index_name: str = "personal-knowledge-assistant"):
        self.embedding_function = embedding_function

        # ✅ Correct separation
        settings = Settings(
            anonymized_telemetry=False,
            persist_directory=".chroma"
        )

        # ✅ tenant and database must go here
        self.client = Client(
            settings=settings,
            tenant="default_tenant",
            database="default_database"
        )

        self.collection = self.client.get_or_create_collection(
            name=index_name,
            embedding_function=self.embedding_function
        )

        logger.info(f"Connected to Chroma DB collection: {index_name}")
    def _initialize_chromadb(self):
        """Initialize Chroma DB connection (already handled in __init__)"""
        pass  # No additional initialization needed for Chroma DB

    def upsert_documents(self, documents: List[Document], embeddings: List[List[float]]):
        """Store documents and their embeddings in Chroma DB"""
        try:
            ids = []
            metadatas = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                doc_id = hashlib.md5(doc.page_content.encode()).hexdigest()
                ids.append(f"{doc_id}_{i}")
                metadatas.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get('source', 'unknown'),
                    "page": doc.metadata.get('page', 0),
                    "timestamp": datetime.now().isoformat()
                })

            self.collection.add(
                documents=[doc.page_content for doc in documents],
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Upserted {len(documents)} vectors to Chroma DB")
        except Exception as e:
            logger.error(f"Upsert error: {e}")
            raise

    def similarity_search(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            # Each 'metadatas', 'documents', and 'distances' is a List[List[Any]], outer list = queries, inner = results
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            documents = results['documents'][0]

            return [
                {
                    "content": metadatas[i]['content'],
                    "source": metadatas[i].get('source', 'unknown'),
                    "score": distances[i],  # Distance (lower is better)
                    "page": metadatas[i].get('page', 0)
                }
                for i in range(len(metadatas))
            ]
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise


    def hybrid_search(self, query: str, query_embedding: List[float], documents: List[Document], top_k: int = 3) -> List[Dict[str, Any]]:
        """Combine semantic and keyword-based search"""
        try:
            # Semantic search
            semantic_results = self.similarity_search(query_embedding, top_k * 2)
            
            # Keyword search with BM25
            tokenized_docs = [doc.page_content.split() for doc in documents]
            bm25 = BM25Okapi(tokenized_docs)
            scores = bm25.get_scores(query.split())
            keyword_results = [
                {"content": doc.page_content, "source": doc.metadata.get('source', 'unknown'), "score": score, "page": doc.metadata.get('page', 0)}
                for doc, score in zip(documents, scores)
            ]
            
            # Combine results (weighted average)
            combined_results = []
            for sem_doc in semantic_results:
                for kw_doc in keyword_results:
                    if sem_doc['content'] == kw_doc['content']:
                        combined_score = 0.7 * (1 - sem_doc['score']) + 0.3 * kw_doc['score']  # Invert distance to score
                        combined_results.append({**sem_doc, "score": combined_score})
                        break
            
            # Sort and filter top_k results
            combined_results.sort(key=lambda x: x['score'], reverse=True)
            return combined_results[:top_k]
        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            raise

class DocumentProcessor:
    """Handles document loading and processing"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def adaptive_chunk_size(self, content: str) -> int:
        """Determine chunk size based on content type"""
        if "```" in content or len(content.split()) < 50:
            return 500  # Smaller chunks for code or tables
        return Config.CHUNK_SIZE
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """Load and process PDF file"""
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            chunk_size = self.adaptive_chunk_size(documents[0].page_content)
            self.text_splitter.chunk_size = chunk_size
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            logger.error(f"PDF loading error: {e}")
            raise
    
    def load_text(self, file_path: str) -> List[Document]:
        """Load and process text file"""
        try:
            loader = TextLoader(file_path)
            documents = loader.load()
            chunk_size = self.adaptive_chunk_size(documents[0].page_content)
            self.text_splitter.chunk_size = chunk_size
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            logger.error(f"Text loading error: {e}")
            raise
    
    def load_web_url(self, url: str) -> List[Document]:
        """Load and process web URL"""
        try:
            loader = WebBaseLoader(url)
            documents = loader.load()


            chunk_size = self.adaptive_chunk_size(documents[0].page_content)
            self.text_splitter.chunk_size = chunk_size
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            logger.error(f"Web loading error: {e}")
            raise
    
    async def process_uploaded_file(self, uploaded_file) -> List[Document]:
        """Process uploaded file from Streamlit"""
        try:
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            if uploaded_file.name.endswith('.pdf'):
                documents = self.load_pdf(temp_path)
            elif uploaded_file.name.endswith('.txt'):
                documents = self.load_text(temp_path)
            else:
                raise ValueError(f"Unsupported file type: {uploaded_file.name}")
            
            os.remove(temp_path)
            return documents
        except Exception as e:
            logger.error(f"File processing error: {e}")
            raise

class RAGSystem:
    """Main RAG system orchestrator"""
    
    def __init__(self, config: Config):
        self.config = config
        self.embedding_function = LocalEmbeddingFunction(model_name=config.EMBEDDING_MODEL)

        self.llm_service = GroqLLMService(
            config.GROQ_API_KEY,
            config.LLM_MODEL
        )
        self.vector_store = VectorStore(
            self.embedding_function,
            config.INDEX_NAME
        )
        self.document_processor = DocumentProcessor(
            config.CHUNK_SIZE,
            config.CHUNK_OVERLAP
        )
        self.documents = []  # Store documents for hybrid search
    
    
    async def add_documents(self, documents: List[Document]):
        """Add documents to the knowledge base"""
        try:
            self.documents.extend(documents)
            contents = [doc.page_content for doc in documents]
            # Compute embeddings locally
            embeddings = self.embedding_function(contents)
            
            # Ensure embeddings is a List[List[float]]
            if isinstance(embeddings, dict):
                logger.warning("Embeddings returned as dict, attempting to extract list")
                # Check for common dictionary keys or structure
                if 'embeddings' in embeddings:
                    embeddings = embeddings['embeddings']
                else:
                    raise ValueError(f"Unexpected embedding dict format: {embeddings}")
            
            # Convert to List[List[float]] if necessary
            if not isinstance(embeddings, list) or not all(isinstance(emb, list) for emb in embeddings):
                # Handle case where embeddings might be a numpy array or other format
                embeddings = [emb.tolist() if isinstance(emb, np.ndarray) else emb for emb in embeddings]
            
            # Validate embeddings format
            if not all(isinstance(emb, list) and all(isinstance(x, float) for x in emb) for emb in embeddings):
                raise ValueError(f"Embeddings format invalid: {embeddings}")
            
            self.vector_store.upsert_documents(documents, embeddings)
            logger.info(f"Added {len(documents)} document chunks to knowledge base")
        except Exception as e:
            logger.error(f"Document addition error: {e}")
            raise
    
    async def query(self, question: str) -> Dict[str, Any]:
        """Query the knowledge base"""
        try:
            # Rephrase query
            rephrased_query = await self.llm_service.rephrase_query(question)
            query_embedding = self.embedding_function([rephrased_query])[0]  # Compute query embedding locally
            
            # Perform hybrid search
            relevant_docs = self.vector_store.hybrid_search(
                rephrased_query, 
                query_embedding, 
                self.documents, 
                top_k=self.config.TOP_K
            )
            
            if not relevant_docs:
                return {
                    "answer": "I couldn't find relevant information in the knowledge base to answer your question.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            context = "\n\n".join([
                f"Source: {doc['source']} (Page: {doc['page']})\nContent: {doc['content']}"
                for doc in relevant_docs
            ])
            
            messages = [
                {
                    "role": "system",
                    "content": """You are a helpful assistant that answers questions based on the provided context. 
                    Follow these guidelines:
                    1. Answer only based on the provided context
                    2. If the context doesn't contain enough information, say so
                    3. Be concise but comprehensive
                    4. Cite sources when possible
                    5. Don't make up information not in the context"""
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
                }
            ]
            
            answer = await self.llm_service.generate_response_async(messages, max_tokens=self.config.MAX_TOKENS)
            
            return {
                "answer": answer,
                "sources": [
                    {
                        "source": doc["source"],
                        "page": doc["page"],
                        "relevance_score": doc["score"]
                    }
                    for doc in relevant_docs
                ],
                "confidence": max([doc["score"] for doc in relevant_docs])
            }
        except Exception as e:
            logger.error(f"Query error: {e}")
            raise

# =============================================================================
# STREAMLIT APPLICATION
# =============================================================================

def load_config() -> Config:
    """Load configuration from Streamlit secrets or environment variables"""
    config = Config()
    
    try:
        config.GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    except:
        config.GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    
    if not config.GROQ_API_KEY:
        raise ValueError("Missing required API key: GROQ_API_KEY")
    
    return config

async def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Personal Knowledge Assistant",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Personal Knowledge Assistant")
    st.markdown("Upload documents and ask questions about them!")
    
    config = load_config()
    
    try:
        if 'rag_system' not in st.session_state:
            with st.spinner("Initializing RAG system..."):
                st.session_state.rag_system = RAGSystem(config)
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        return
    
    with st.sidebar:
        st.header("📄 Document Management")
        
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=['pdf', 'txt'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("Process Documents"):
                progress_bar = st.progress(0)
                try:
                    for i, uploaded_file in enumerate(uploaded_files):
                        documents = await st.session_state.rag_system.document_processor.process_uploaded_file(uploaded_file)
                        await st.session_state.rag_system.add_documents(documents)
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    st.success(f"Successfully processed {len(uploaded_files)} documents!")
                except Exception as e:
                    st.error(f"Error processing documents: {e}")
                finally:
                    progress_bar.empty()
        
        st.subheader("🌐 Add Web Content")
        url = st.text_input("Enter URL:")
        
        if url and st.button("Process URL"):
            try:
                with st.spinner("Processing web content..."):
                    documents = st.session_state.rag_system.document_processor.load_web_url(url)
                    await st.session_state.rag_system.add_documents(documents)
                    st.success("Web content processed successfully!")
            except Exception as e:
                st.error(f"Error processing URL: {e}")
    
    st.header("❓ Ask Questions")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Sources"):
                    for source in message["sources"]:
                        st.markdown(f"- **{source['source']}** (Page: {source['page']}, Relevance: {source['relevance_score']:.2f})")
    
    if question := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = await st.session_state.rag_system.query(question)
                    st.markdown(response["answer"])
                    if response["sources"]:
                        with st.expander("📚 Sources"):
                            for source in response["sources"]:
                                st.markdown(f"- **{source['source']}** (Page: {source['page']}, Relevance: {source['relevance_score']:.2f})")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response["sources"]
                    })
                except Exception as e:
                    error_msg = f"Error generating response: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    asyncio.run(main())

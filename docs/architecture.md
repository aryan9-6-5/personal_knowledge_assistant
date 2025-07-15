# System Architecture
A cloud-based RAG system using Groq (LLM), Hugging Face (embeddings), Pinecone (vector database), and Streamlit (UI).
- DocumentProcessor: Chunks PDFs/text/URLs with adaptive chunking.
- EmbeddingService: Generates embeddings with async API calls.
- VectorStore: Stores/retrieves embeddings with hybrid search.
- GroqLLMService: Generates responses with query rephrasing.
- RAGSystem: Orchestrates the pipeline.
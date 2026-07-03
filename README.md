# 🧠 Personal Knowledge Assistant

An **advanced RAG (Retrieval-Augmented Generation)** application that lets you upload documents and ask questions about them with AI-powered answers and source citations.

## Features

- **Advanced RAG Pipeline**: Query rephrasing → HyDE → Hybrid search (Semantic + BM25 with RRF) → Cross-encoder re-ranking → Streaming LLM generation
- **Document Ingestion**: Upload PDFs, text files, or ingest web URLs
- **Streaming Chat**: Real-time token-by-token response via Server-Sent Events
- **Source Attribution**: Every answer cites documents with page numbers and relevance scores
- **Conversation Memory**: Multi-turn dialogue with context preservation
- **Dark/Light Theme**: Premium UI with glassmorphism and micro-animations
- **Keyboard Shortcuts**: Ctrl+K (focus input), Ctrl+B (toggle sidebar)

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React + Vite |
| Backend | FastAPI (Python) |
| LLM | Groq API (Llama 3.3 70B) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2, local) |
| Vector DB | ChromaDB (persistent) |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Search | Hybrid: Cosine + BM25 with Reciprocal Rank Fusion |

## Setup

### 1. Environment Variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key at [console.groq.com](https://console.groq.com/)

### 2. Backend

```bash
cd backend
pip install -r requirements.txt
python main.py
```

The API server starts at `http://localhost:8000`.

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

The dev server starts at `http://localhost:5173` with API proxy to the backend.

## Usage

1. **Upload documents** — Drag & drop PDFs/text files in the sidebar, or paste a URL
2. **Ask questions** — Type in the chat and get AI answers grounded in your documents
3. **View sources** — Expand the "Sources" section on any answer to see cited passages

## Architecture

```
personal-knowledge-assistant/
├── backend/                    # FastAPI Python backend
│   ├── main.py                 # Entry point
│   ├── config.py               # Settings
│   ├── rag/                    # RAG engine modules
│   │   ├── engine.py           # Pipeline orchestrator
│   │   ├── embeddings.py       # Local sentence-transformers
│   │   ├── vector_store.py     # ChromaDB + hybrid search
│   │   ├── document_processor.py # Chunking with context
│   │   ├── reranker.py         # Cross-encoder re-ranking
│   │   └── llm.py              # Groq streaming LLM
│   ├── models/schemas.py       # Pydantic models
│   └── routers/                # API endpoints
│       ├── chat.py             # SSE streaming chat
│       ├── documents.py        # Document CRUD
│       └── system.py           # Health + stats
├── frontend/                   # React + Vite frontend
│   └── src/
│       ├── App.jsx             # Main app
│       ├── components/         # UI components
│       └── lib/api.js          # API client with SSE
└── .env                        # API keys
```
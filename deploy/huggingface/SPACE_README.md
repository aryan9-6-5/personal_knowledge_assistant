---
title: Personal Knowledge Assistant
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

Advanced RAG assistant: query rephrasing, HyDE, BM25 + dense hybrid retrieval with
Reciprocal Rank Fusion, cross-encoder re-ranking, and streaming answers with source
citations. FastAPI backend, React frontend, Groq (gpt-oss-120b).

Source code: https://github.com/aryan9-6-5/personal_knowledge_assistant

API docs: `/docs`. Create an account in the app, upload a PDF or text file, then ask questions.
Demo limits: 30 chat requests per hour per user, 10 MB per upload.

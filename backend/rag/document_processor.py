"""Document loading and intelligent chunking."""

import os
import hashlib
import logging
import tempfile
from typing import List, Dict, Any, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import settings

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document loading, chunking, and metadata extraction."""

    def __init__(self):
        self.default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        # Smaller chunks for code-heavy content
        self.code_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""],
        )

        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    def _detect_content_type(self, text: str) -> str:
        """Detect if content is code-heavy, tabular, or prose."""
        code_indicators = text.count("```") + text.count("def ") + text.count("class ")
        if code_indicators > 3:
            return "code"
        return "prose"

    def _get_splitter(self, text: str) -> RecursiveCharacterTextSplitter:
        """Get appropriate splitter based on content type."""
        content_type = self._detect_content_type(text)
        if content_type == "code":
            return self.code_splitter
        return self.default_splitter

    def process_pdf(self, file_path: str, filename: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Process a PDF file into chunks with metadata."""
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        chunks = []
        metadatas = []

        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if not text.strip():
                continue

            splitter = self._get_splitter(text)
            page_chunks = splitter.split_text(text)

            for chunk in page_chunks:
                if len(chunk.strip()) < 20:
                    continue
                # Contextual chunking: prepend source info
                contextual_chunk = f"[Source: {filename}, Page {page_num}]\n{chunk}"
                chunks.append(contextual_chunk)
                metadatas.append({
                    "source": filename,
                    "page": page_num,
                    "content_type": self._detect_content_type(text),
                })

        logger.info(f"PDF '{filename}': {len(reader.pages)} pages → {len(chunks)} chunks")
        return chunks, metadatas

    def process_text(self, file_path: str, filename: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Process a text/markdown file into chunks with metadata."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        if not text.strip():
            return [], []

        splitter = self._get_splitter(text)
        raw_chunks = splitter.split_text(text)

        chunks = []
        metadatas = []
        for i, chunk in enumerate(raw_chunks):
            if len(chunk.strip()) < 20:
                continue
            page = (i // 3) + 1  # Approximate page numbers
            contextual_chunk = f"[Source: {filename}, Section {page}]\n{chunk}"
            chunks.append(contextual_chunk)
            metadatas.append({
                "source": filename,
                "page": page,
                "content_type": self._detect_content_type(text),
            })

        logger.info(f"Text '{filename}': {len(chunks)} chunks")
        return chunks, metadatas

    def process_url(self, url: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Process a web URL into chunks."""
        import httpx
        from bs4 import BeautifulSoup

        response = httpx.get(url, timeout=30, follow_redirects=True)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style tags
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        title = soup.title.string if soup.title else url

        if not text.strip():
            return [], []

        splitter = self._get_splitter(text)
        raw_chunks = splitter.split_text(text)

        chunks = []
        metadatas = []
        for i, chunk in enumerate(raw_chunks):
            if len(chunk.strip()) < 20:
                continue
            contextual_chunk = f"[Source: {title}]\n{chunk}"
            chunks.append(contextual_chunk)
            metadatas.append({
                "source": title,
                "page": i + 1,
                "url": url,
                "content_type": "web",
            })

        logger.info(f"URL '{url}': {len(chunks)} chunks")
        return chunks, metadatas

    async def save_uploaded_file(self, file_content: bytes, filename: str) -> str:
        """Save uploaded file to disk and return path."""
        file_hash = hashlib.md5(file_content).hexdigest()[:8]
        safe_name = f"{file_hash}_{filename}"
        file_path = os.path.join(settings.UPLOAD_DIR, safe_name)

        with open(file_path, "wb") as f:
            f.write(file_content)

        return file_path

    def process_file(self, file_path: str, filename: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Route file processing based on extension."""
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".pdf":
            return self.process_pdf(file_path, filename)
        elif ext in (".txt", ".md", ".markdown", ".rst", ".csv"):
            return self.process_text(file_path, filename)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

"""Groq LLM service with streaming support."""

import logging
from typing import List, Dict, AsyncGenerator

from groq import AsyncGroq
from config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Handles all LLM interactions via Groq API."""

    def __init__(self):
        self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        self.model = settings.LLM_MODEL

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = None,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from Groq API."""
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens or settings.MAX_TOKENS,
                temperature=settings.TEMPERATURE,
                top_p=0.9,
                stream=True,
            )
            async for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    yield token
        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            yield f"\n\n[Error generating response: {str(e)}]"

    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = None,
    ) -> str:
        """Generate a complete response (non-streaming)."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens or settings.MAX_TOKENS,
                temperature=settings.TEMPERATURE,
                top_p=0.9,
                stream=False,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            raise

    async def rephrase_query(self, query: str) -> str:
        """Rephrase a user query for better retrieval."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a search query optimizer. Rephrase the following question "
                    "to be more specific and better suited for document retrieval. "
                    "Return ONLY the rephrased query, nothing else."
                ),
            },
            {"role": "user", "content": query},
        ]
        try:
            return await self.generate(messages, max_tokens=150)
        except Exception:
            return query  # Fall back to original

    async def generate_hypothetical_answer(self, query: str) -> str:
        """
        HyDE: Generate a hypothetical answer to embed for better retrieval.
        The hypothetical answer is closer to the document space than the query.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "Write a short, factual paragraph that would answer the following question. "
                    "Write as if you are a knowledgeable expert. Keep it under 100 words. "
                    "Do not say 'I don't know'. Just provide a plausible answer."
                ),
            },
            {"role": "user", "content": query},
        ]
        try:
            return await self.generate(messages, max_tokens=200)
        except Exception:
            return query

"""Query embedding with instruction prefix via local Ollama model."""

from __future__ import annotations

import logging
from typing import List
from pinecone import Pinecone

from config.settings import settings

logger = logging.getLogger(__name__)


class QueryEmbedder:
    """Encode user queries with instruction prompt through Pinecone API."""

    def __init__(self) -> None:
        """Initialize query embedding client."""
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.model_name = settings.PINECONE_EMBED_MODEL

    def embed(self, query: str) -> List[float]:
        """Embed a query string with instruction prefix."""
        try:
            embeddings = self.pc.inference.embed(
                model=self.model_name,
                inputs=[query],
                parameters={"input_type": "query"}
            )
            return embeddings.data[0].values
        except Exception as exc:
            logger.exception("Query embedding failed: %s", exc)
            raise

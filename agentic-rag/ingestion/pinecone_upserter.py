"""Pinecone upsert utilities for dense+sparse hybrid vectors."""

from __future__ import annotations

import logging
from typing import Dict, List

from pinecone import Pinecone, ServerlessSpec

from config.settings import settings

logger = logging.getLogger(__name__)


class PineconeUpserter:
    """Manage Pinecone index lifecycle and vector upserts."""

    def __init__(self) -> None:
        """Initialize Pinecone client and index."""
        try:
            self.client = Pinecone(api_key=settings.PINECONE_API_KEY)
            self._ensure_index()
            self.index = self.client.Index(settings.PINECONE_INDEX)
        except Exception as exc:
            logger.exception("Pinecone initialization failed: %s", exc)
            raise

    def _ensure_index(self) -> None:
        """Create index if it does not already exist."""
        try:
            indexes = self.client.list_indexes()
            existing = set()
            if hasattr(indexes, "names"):
                existing = set(indexes.names())
            else:
                existing = {idx["name"] for idx in indexes}
            if settings.PINECONE_INDEX not in existing:
                self.client.create_index(
                    name=settings.PINECONE_INDEX,
                    dimension=settings.PINECONE_DIMENSION,
                    metric=settings.PINECONE_METRIC,
                    spec=ServerlessSpec(cloud=settings.PINECONE_CLOUD, region=settings.PINECONE_REGION),
                )
        except Exception as exc:
            logger.exception("Failed to ensure Pinecone index: %s", exc)
            raise

    def upsert_chunks(self, chunks: List[Dict]) -> None:
        """Upsert chunks with both dense and sparse vectors."""
        try:
            vectors = []
            for chunk in chunks:
                metadata = {k: v for k, v in chunk.items() if k not in {"dense_vector", "sparse_vector"} and v is not None}
                vectors.append(
                    {
                        "id": chunk["id"],
                        "values": chunk["dense_vector"],
                        "sparse_values": chunk["sparse_vector"],
                        "metadata": metadata,
                    }
                )
            total = 0
            batch_size = 25
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i : i + batch_size]
                self.index.upsert(vectors=batch, namespace=settings.PINECONE_NAMESPACE)
                total += len(batch)
                logger.info("Upserted batch %s-%s (%s vectors)", i + 1, i + len(batch), len(batch))
            logger.info("Upserted %s vectors to Pinecone index %s", total, settings.PINECONE_INDEX)
        except Exception as exc:
            logger.exception("Pinecone upsert failed: %s", exc)
            raise

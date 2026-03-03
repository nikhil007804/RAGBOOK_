"""Sparse encoding utilities with BM25Encoder."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

from pinecone_text.sparse import BM25Encoder

from config.settings import settings

logger = logging.getLogger(__name__)


class SparseBM25Encoder:
    """Wrapper over BM25Encoder fit/save/load/encode operations."""

    def __init__(self, state_path: str | None = None) -> None:
        """Initialize sparse encoder wrapper."""
        self.state_path = Path(state_path or settings.BM25_STATE_PATH)
        self.encoder = BM25Encoder()

    def fit(self, chunks: List[Dict]) -> None:
        """Fit BM25 on chunk texts."""
        try:
            corpus = [chunk["text"] for chunk in chunks]
            self.encoder.fit(corpus)
        except Exception as exc:
            logger.exception("BM25 fit failed: %s", exc)
            raise

    def save(self) -> None:
        """Persist BM25 state to disk."""
        try:
            if hasattr(self.encoder, "dump"):
                self.encoder.dump(str(self.state_path))
                return
            payload = self.encoder.get_params()
            self.state_path.write_text(json.dumps(payload), encoding="utf-8")
        except Exception as exc:
            logger.exception("Saving BM25 params failed: %s", exc)
            raise

    def load(self) -> None:
        """Load BM25 state from disk."""
        try:
            if hasattr(self.encoder, "load"):
                self.encoder = self.encoder.load(str(self.state_path))
                return
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.encoder.set_params(**payload)
        except Exception as exc:
            logger.exception("Loading BM25 params failed: %s", exc)
            raise

    def encode_documents(self, chunks: List[Dict]) -> List[Dict]:
        """Encode chunk texts into sparse vectors."""
        try:
            for chunk in chunks:
                chunk["sparse_vector"] = self.encoder.encode_documents(chunk["text"])
            return chunks
        except Exception as exc:
            logger.exception("Sparse document encoding failed: %s", exc)
            raise

    def encode_query(self, query: str) -> Dict:
        """Encode user query into sparse vector."""
        try:
            return self.encoder.encode_queries(query)
        except Exception as exc:
            logger.exception("Sparse query encoding failed: %s", exc)
            raise

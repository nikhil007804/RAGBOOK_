"""Hybrid retrieval utilities using Pinecone dense+sparse queries."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from pinecone import Pinecone

from config.settings import settings
from ingestion.sparse_encoder import SparseBM25Encoder

logger = logging.getLogger(__name__)

CODE_KEYWORDS = ("implement", "code", "example", "show me", "function")
FIGURE_KEYWORDS = ("diagram", "figure", "architecture", "fig")


def detect_query_type(query: str) -> str:
    """Detect query type from keywords."""
    try:
        lowered = query.lower()
        if any(kw in lowered for kw in CODE_KEYWORDS):
            return "code"
        if any(kw in lowered for kw in FIGURE_KEYWORDS):
            return "figure"
        return "conceptual"
    except Exception as exc:
        logger.exception("Query type detection failed: %s", exc)
        raise


def resolve_alpha(query_type: str, alpha_override: Optional[float] = None) -> float:
    """Resolve alpha with override and query-type defaults."""
    try:
        if alpha_override is not None:
            return alpha_override
        if query_type == "code":
            return 0.3
        if query_type == "conceptual":
            return 0.8
        return settings.HYBRID_ALPHA
    except Exception as exc:
        logger.exception("Alpha resolution failed: %s", exc)
        raise


class HybridRetriever:
    """Run dense/sparse/hybrid Pinecone queries."""

    def __init__(self) -> None:
        """Initialize Pinecone index and BM25 encoder."""
        try:
            client = Pinecone(api_key=settings.PINECONE_API_KEY)
            self.index = client.Index(settings.PINECONE_INDEX)
            self.sparse_encoder = SparseBM25Encoder()
            self.sparse_encoder.load()
        except Exception as exc:
            logger.exception("HybridRetriever initialization failed: %s", exc)
            raise

    def retrieve(
        self,
        query_text: str,
        dense_vector: List[float],
        top_k: int = 20,
        alpha: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], str, float]:
        """Return dense, sparse, and hybrid result lists."""
        try:
            query_type = detect_query_type(query_text)
            effective_alpha = resolve_alpha(query_type, alpha)
            sparse_vector = self.sparse_encoder.encode_query(query_text)

            dense_result = self.index.query(
                namespace=settings.PINECONE_NAMESPACE,
                vector=dense_vector,
                top_k=top_k,
                include_metadata=True,
            )
            sparse_result = self.index.query(
                namespace=settings.PINECONE_NAMESPACE,
                vector=[0.0] * len(dense_vector),
                sparse_vector=sparse_vector,
                top_k=top_k,
                include_metadata=True,
            )

            scaled_dense = [v * effective_alpha for v in dense_vector]
            scaled_sparse = {
                "indices": sparse_vector.get("indices", []),
                "values": [v * (1 - effective_alpha) for v in sparse_vector.get("values", [])],
            }
            hybrid_result = self.index.query(
                namespace=settings.PINECONE_NAMESPACE,
                vector=scaled_dense,
                sparse_vector=scaled_sparse,
                top_k=top_k,
                include_metadata=True,
            )
            return (
                _extract_matches(dense_result),
                _extract_matches(sparse_result),
                _extract_matches(hybrid_result),
                query_type,
                effective_alpha,
            )
        except Exception as exc:
            logger.exception("Hybrid retrieval failed: %s", exc)
            raise


def _extract_matches(result: Any) -> List[Dict[str, Any]]:
    """Normalize Pinecone query result to a list of match dicts."""
    try:
        if result is None:
            return []
        
        matches = []
        if isinstance(result, dict):
            matches = result.get("matches", [])
        elif hasattr(result, "matches"):
            matches = list(result.matches)
        
        normalized = []
        for m in matches:
            if isinstance(m, dict):
                normalized.append(dict(m))
            elif hasattr(m, "to_dict"):
                normalized.append(m.to_dict())
            else:
                m_dict = {"id": getattr(m, "id", "")}
                if hasattr(m, "score"):
                    m_dict["score"] = m.score
                if hasattr(m, "metadata"):
                    m_dict["metadata"] = m.metadata
                normalized.append(m_dict)
        return normalized
    except Exception as exc:
        logger.exception("Failed to extract Pinecone matches: %s", exc)
        return []

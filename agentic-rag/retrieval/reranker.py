"""Rerank retrieved chunks using Pinecone Inference reranker."""

from __future__ import annotations

import logging
from typing import Any, Dict, List
from pinecone import Pinecone

from config.settings import settings

logger = logging.getLogger(__name__)


class BGereranker:
    """Apply Pinecone Inference to reduce top-k results to top-n."""

    def __init__(self) -> None:
        """Initialize Pinecone client for reranking."""
        try:
            self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            self.model = "bge-reranker-v2-m3" # Pinecone's hosted BGE reranker
        except Exception as exc:
            logger.exception("Reranker initialization failed: %s", exc)
            raise

    def rerank(self, query: str, results: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        """Rerank query-result pairs and return top_n results."""
        try:
            if not results:
                return []
            
            # Extract texts for Pinecone
            documents = []
            for idx, item in enumerate(results):
                text = item.get("metadata", {}).get("text", "")
                # BGE-reranker-v2-m3 has a 1024 token limit per query+document
                # 1 token is approx ~3.5 chars. Truncating to 2500 chars to be safe.
                if len(text) > 2500:
                    text = text[:2500]
                
                documents.append({
                    "id": str(idx),
                    "text": text
                })

            response = self.pc.inference.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_n=top_n,
                return_documents=False
            )

            # Map scores back to original results using the returned indices
            scored = []
            for record in response.data:
                idx = int(record.document.id) if hasattr(record.document, 'id') else int(record.index)
                item = results[idx]
                item["rerank_score"] = float(record.score)
                scored.append(item)
            
            # Sort just to be safe, though Pinecone returns them sorted
            scored.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            return scored
        except Exception as exc:
            logger.exception("Reranking failed: %s", exc)
            raise

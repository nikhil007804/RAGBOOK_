"""API router for RAG query and health endpoints."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pinecone import Pinecone

from api.models import QueryRequest, QueryResponse
from cache.cache_manager import CacheManager
from cache.key_builder import build_cache_key
from config.settings import settings
from generation.llm_client import GeminiClient
from generation.prompt_builder import build_prompt
from generation.response_formatter import parse_and_validate_response
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.query_embedder import QueryEmbedder
from retrieval.reranker import BGereranker
from retrieval.rrf_fusion import rrf_fuse

logger = logging.getLogger(__name__)

router = APIRouter()


@lru_cache(maxsize=1)
def _components() -> Dict[str, Any]:
    """Initialize heavy service dependencies once."""
    try:
        return {
            "cache_manager": CacheManager(),
            "query_embedder": QueryEmbedder(),
            "retriever": HybridRetriever(),
            "reranker": BGereranker(),
            "llm_client": GeminiClient(),
        }
    except Exception as exc:
        logger.exception("Component initialization failed: %s", exc)
        raise


@router.post("/query", response_model=QueryResponse)
def query_book(request: QueryRequest) -> QueryResponse:
    """Cache-first query endpoint with full retrieval/generation pipeline."""
    try:
        components = _components()
        cache_manager: CacheManager = components["cache_manager"]
        query_embedder: QueryEmbedder = components["query_embedder"]
        retriever: HybridRetriever = components["retriever"]
        reranker: BGereranker = components["reranker"]
        llm_client: GeminiClient = components["llm_client"]

        cache_key = build_cache_key(request.query)
        cached = cache_manager.check_cache(cache_key)
        if cached:
            return QueryResponse(
                answer=cached.get("answer", ""),
                citations=cached.get("citations", []),
                confidence=cached.get("confidence", "low"),
                source="cache",
            )

        dense_vector = query_embedder.embed(request.query)
        dense_results, sparse_results, hybrid_results, query_type, effective_alpha = retriever.retrieve(
            query_text=request.query,
            dense_vector=dense_vector,
            top_k=settings.RETRIEVAL_TOP_K,
            alpha=request.alpha,
        )
        fused = rrf_fuse([dense_results, sparse_results, hybrid_results], k=60, top_k=settings.RETRIEVAL_TOP_K)
        reranked = reranker.rerank(request.query, fused, top_n=settings.RERANK_TOP_N)
        prompt = build_prompt(request.query, reranked)
        raw_response = llm_client.generate(prompt)
        parsed = parse_and_validate_response(raw_response)
        parsed["query_type"] = _normalize_query_type(query_type)

        api_payload: Dict[str, Any] = {
            "answer": parsed["answer"],
            "citations": parsed["citations"],
            "confidence": parsed["confidence"],
            "source": "pipeline",
            "query_type": parsed["query_type"],
            "alpha": effective_alpha,
        }
        cache_manager.save_cache(cache_key, api_payload)

        return QueryResponse(
            answer=api_payload["answer"],
            citations=api_payload["citations"],
            confidence=api_payload["confidence"],
            source="pipeline",
        )
    except Exception as exc:
        logger.exception("Query endpoint failed: %s", exc)
        _components.cache_clear()
        raise HTTPException(status_code=500, detail="Query pipeline failed") from exc


@router.get("/health")
def health() -> Dict[str, str]:
    """Health check for API, Redis, and Pinecone connectivity."""
    redis_status = "disconnected"
    pinecone_status = "disconnected"
    try:
        components = _components()
        cache_manager: CacheManager = components["cache_manager"]
        if cache_manager.client.ping():
            redis_status = "connected"
    except Exception as exc:
        logger.exception("Redis health check failed: %s", exc)
        _components.cache_clear()

    try:
        pinecone_client = Pinecone(api_key=settings.PINECONE_API_KEY)
        pinecone_client.describe_index(settings.PINECONE_INDEX)
        pinecone_status = "connected"
    except Exception as exc:
        logger.exception("Pinecone health check failed: %s", exc)
        _components.cache_clear()

    return {"status": "ok", "redis": redis_status, "pinecone": pinecone_status}


def _normalize_query_type(query_type: str) -> str:
    """Map retrieval type to required LLM output categories."""
    try:
        mapping = {"conceptual": "prose", "code": "code", "figure": "figure"}
        return mapping.get(query_type, "prose")
    except Exception as exc:
        logger.exception("Query type normalization failed: %s", exc)
        return "prose"

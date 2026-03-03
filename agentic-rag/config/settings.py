"""Application settings loaded from environment variables."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    """Container for runtime configuration."""

    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_TEMPERATURE: float = 0.0
    GEMINI_MAX_OUTPUT_TOKENS: int = 4096

    PINECONE_API_KEY: str = ""
    PINECONE_INDEX: str = "agentic-rag"
    PINECONE_DIMENSION: int = 1024
    PINECONE_METRIC: str = "dotproduct"
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"

    BGE_EMBED_MODEL: str = "BAAI/bge-large-en-v1.5"
    BGE_RERANK_MODEL: str = "BAAI/bge-reranker-large"
    BGE_QUERY_INSTR: str = "Represent this sentence for searching relevant passages:"
    PINECONE_EMBED_MODEL: str = "multilingual-e5-large"

    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    CACHE_TTL_DAYS: int = 30

    HYBRID_ALPHA: float = 0.6
    RETRIEVAL_TOP_K: int = 20
    RERANK_TOP_N: int = 5

    PDF_PATH: str = ""
    BM25_STATE_PATH: str = "bm25_params.json"
    PINECONE_NAMESPACE: str = "default"


def _load_settings() -> Settings:
    """Load settings from `.env` and process environment with defaults."""
    try:
        load_dotenv()
        return Settings(
            GEMINI_API_KEY=os.getenv("GEMINI_API_KEY", ""),
            PINECONE_API_KEY=os.getenv("PINECONE_API_KEY", ""),
            REDIS_HOST=os.getenv("REDIS_HOST", "localhost"),
            REDIS_PORT=int(os.getenv("REDIS_PORT", "6379")),
            CACHE_TTL_DAYS=int(os.getenv("CACHE_TTL_DAYS", "30")),
            HYBRID_ALPHA=float(os.getenv("HYBRID_ALPHA", "0.6")),
            RETRIEVAL_TOP_K=int(os.getenv("RETRIEVAL_TOP_K", "20")),
            RERANK_TOP_N=int(os.getenv("RERANK_TOP_N", "5")),
            PDF_PATH=os.getenv("PDF_PATH", ""),
            BM25_STATE_PATH=os.getenv("BM25_STATE_PATH", "bm25_params.json"),
            PINECONE_NAMESPACE=os.getenv("PINECONE_NAMESPACE", "default"),
            PINECONE_EMBED_MODEL=os.getenv("PINECONE_EMBED_MODEL", "multilingual-e5-large"),
        )
    except Exception as exc:
        logger.exception("Failed to load settings: %s", exc)
        raise


settings = _load_settings()

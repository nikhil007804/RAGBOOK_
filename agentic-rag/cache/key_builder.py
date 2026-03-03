"""Cache key helpers."""

from __future__ import annotations

import hashlib
import logging

logger = logging.getLogger(__name__)


def build_cache_key(query: str) -> str:
    """Build cache key as `rag:` + md5(normalized query)."""
    try:
        normalized = query.strip().lower()
        digest = hashlib.md5(normalized.encode("utf-8")).hexdigest()
        return f"rag:{digest}"
    except Exception as exc:
        logger.exception("Failed to build cache key: %s", exc)
        raise

"""Redis cache manager for full RAG JSON responses."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

import redis

from config.settings import settings

logger = logging.getLogger(__name__)


class CacheManager:
    """Read/write cache entries with fixed TTL."""

    def __init__(self) -> None:
        """Initialize Redis connection."""
        try:
            self.client = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True)
        except Exception as exc:
            logger.exception("Redis client initialization failed: %s", exc)
            raise

    def check_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached value by key."""
        try:
            raw = self.client.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            logger.exception("Cache read failed for key=%s: %s", key, exc)
            return None

    def save_cache(self, key: str, payload: Dict[str, Any]) -> None:
        """Save payload with 30-day TTL."""
        try:
            ttl_seconds = 60 * 60 * 24 * settings.CACHE_TTL_DAYS
            self.client.setex(key, ttl_seconds, json.dumps(payload))
        except Exception as exc:
            logger.exception("Cache write failed for key=%s: %s", key, exc)
            raise

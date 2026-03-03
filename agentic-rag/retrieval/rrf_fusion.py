"""Reciprocal Rank Fusion (RRF) for multiple ranked lists."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def rrf_fuse(result_lists: List[List[Dict[str, Any]]], k: int = 60, top_k: int = 20) -> List[Dict[str, Any]]:
    """Fuse ranked result lists with RRF score(d)=sum(1/(k+rank))."""
    try:
        score_map: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}

        for ranked_list in result_lists:
            for rank, item in enumerate(ranked_list, start=1):
                doc_id = item["id"]
                score_map[doc_id] = score_map.get(doc_id, 0.0) + (1.0 / (k + rank))
                doc_map[doc_id] = item

        ranked = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        fused: List[Dict[str, Any]] = []
        for doc_id, score in ranked:
            doc = dict(doc_map[doc_id])
            doc["rrf_score"] = score
            fused.append(doc)
        return fused
    except Exception as exc:
        logger.exception("RRF fusion failed: %s", exc)
        raise

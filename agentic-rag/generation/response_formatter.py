"""Response parser and validator for strict JSON output."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def parse_and_validate_response(raw_json: str) -> Dict[str, Any]:
    """Parse LLM JSON output and enforce expected schema."""
    try:
        raw_json = raw_json.strip()
        if raw_json.startswith("```json"):
            raw_json = raw_json[7:]
        if raw_json.startswith("```"):
            raw_json = raw_json[3:]
        if raw_json.endswith("```"):
            raw_json = raw_json[:-3]
        raw_json = raw_json.strip()
        
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError:
            payload = {
                "answer": raw_json if len(raw_json) > 0 else "Not found in the document",
                "citations": [],
                "confidence": "low",
                "query_type": "prose"
            }
        answer = payload.get("answer", "Not found in the document")
        citations = payload.get("citations", [])
        confidence = payload.get("confidence", "low")
        query_type = payload.get("query_type", "prose")

        if not isinstance(citations, list):
            citations = []
        normalized_citations: List[Dict[str, Any]] = []
        for c in citations:
            if not isinstance(c, dict):
                continue
            normalized_citations.append(
                {
                    "page": int(c.get("page", 0)) if str(c.get("page", "0")).isdigit() else 0,
                    "chapter": str(c.get("chapter", "")),
                    "excerpt": str(c.get("excerpt", ""))[:300],
                }
            )

        if confidence not in {"high", "medium", "low"}:
            confidence = "low"
        if query_type not in {"prose", "code", "figure"}:
            query_type = "prose"

        return {
            "answer": answer,
            "citations": normalized_citations,
            "confidence": confidence,
            "query_type": query_type,
        }
    except Exception as exc:
        logger.exception("Response parsing/validation failed: %s", exc)
        return {
            "answer": "Not found in the document",
            "citations": [],
            "confidence": "low",
            "query_type": "prose",
        }

"""Prompt construction for grounded generation."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
You are an expert assistant on Agentic Design Patterns book.
Answer using the context below. You may synthesize and summarize information from the context. Cite page numbers where helpful.
If the answer cannot be reasonably inferred from the context, your `answer` field SHOULD BE EXACTLY: "Not found in the document."
Always respond in valid JSON matching the exact schema provided. DO NOT output plain text.

CONTEXT:
{context}

QUESTION: {query}

RESPOND IN THIS EXACT JSON FORMAT:
{{
  "answer": "...",
  "citations": [{{"page": ..., "chapter": "...", "excerpt": "..."}}],
  "confidence": "high|medium|low",
  "query_type": "prose|code|figure"
}}
"""


def build_prompt(query: str, reranked_chunks: List[Dict[str, Any]]) -> str:
    """Build a grounded prompt from top reranked chunks."""
    try:
        context_blocks = []
        for item in reranked_chunks:
            md = item.get("metadata", {})
            context_blocks.append(
                (
                    f"[id={item.get('id')}] page={md.get('page_start')} "
                    f"chapter={md.get('chapter_title')} type={md.get('content_type')}\n"
                    f"{md.get('text', '')}"
                )
            )
        context = "\n\n---\n\n".join(context_blocks)
        return PROMPT_TEMPLATE.format(context=context, query=query)
    except Exception as exc:
        logger.exception("Prompt building failed: %s", exc)
        raise

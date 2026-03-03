"""Hybrid chunking pipeline for prose, code, figure context, and bridge chunks."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

PYTHON_KEYWORDS = ("def ", "class ", "import ", "return", "async def ", "@")
CODE_QUERY_PATTERN = re.compile(r"^\s*(def |class |import |from |return|async def |@)", re.IGNORECASE)
DEF_CLASS_PATTERN = re.compile(r"^\s*(def |class |async def )", re.IGNORECASE)
FIGURE_PATTERN = re.compile(r"(Fig\.\s*\d+|Figure\s+\d+)", re.IGNORECASE)


def count_tokens(text: str, model_hint: str = "cl100k_base") -> int:
    """Count tokens using tiktoken."""
    try:
        enc = tiktoken.get_encoding(model_hint)
        return len(enc.encode(text))
    except Exception as exc:
        logger.exception("Token counting failed: %s", exc)
        raise


def _split_text_semantic(text: str, chunk_size: int = 450, chunk_overlap: int = 80) -> List[str]:
    """Split text with recursive separators."""
    try:
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n\n", "\n\n", "\n"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=lambda s: count_tokens(s),
            is_separator_regex=False,
        )
        return [c.strip() for c in splitter.split_text(text) if c.strip()]
    except Exception as exc:
        logger.exception("Semantic split failed: %s", exc)
        raise


def detect_content_type(text: str) -> str:
    """Detect content type with exact rules for prose/code/figure context."""
    try:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            return "prose"

        keyword_or_indent_lines = 0
        code_lines = 0
        for line in lines:
            normalized = line.lstrip()
            has_indent = line.startswith("    ")
            has_keyword = any(kw in normalized for kw in PYTHON_KEYWORDS)
            if has_indent or has_keyword:
                keyword_or_indent_lines += 1
            if CODE_QUERY_PATTERN.search(line):
                code_lines += 1

        if FIGURE_PATTERN.search(text):
            return "figure_context"
        if code_lines > 0:
            return "code"

        ratio = keyword_or_indent_lines / max(1, len(lines))
        return "prose" if ratio < 0.2 else "code"
    except Exception as exc:
        logger.exception("Content type detection failed: %s", exc)
        raise


def _extract_section_title(text: str) -> str:
    """Extract a section title heuristic from a chunk."""
    try:
        for line in text.splitlines():
            clean = line.strip()
            if clean and len(clean) < 120:
                return clean
        return "Untitled Section"
    except Exception as exc:
        logger.exception("Section title extraction failed: %s", exc)
        raise


def _split_code_blocks(text: str) -> List[str]:
    """Split code while avoiding mid-function boundaries."""
    try:
        lines = text.splitlines()
        boundaries = [0]
        for i, line in enumerate(lines):
            if DEF_CLASS_PATTERN.match(line):
                boundaries.append(i)
        boundaries.append(len(lines))
        boundaries = sorted(set(boundaries))

        blocks: List[str] = []
        for start_idx, end_idx in zip(boundaries[:-1], boundaries[1:]):
            block = "\n".join(lines[start_idx:end_idx]).strip()
            if block:
                blocks.append(block)
        if not blocks:
            blocks = [text.strip()]

        final_blocks: List[str] = []
        current = ""
        for block in blocks:
            candidate = f"{current}\n{block}".strip() if current else block
            if count_tokens(candidate) <= 600:
                current = candidate
            else:
                if current:
                    final_blocks.append(current.strip())
                current = block
        if current:
            final_blocks.append(current.strip())
        return final_blocks
    except Exception as exc:
        logger.exception("Code split failed: %s", exc)
        raise


def _find_neighboring_paragraphs(page_text: str, center_text: str) -> Dict[str, str]:
    """Find preceding and following paragraphs around matched figure text."""
    try:
        paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]
        idx = 0
        for i, p in enumerate(paragraphs):
            if center_text[:80] in p:
                idx = i
                break
        before = paragraphs[idx - 1] if idx - 1 >= 0 else ""
        after = paragraphs[idx + 1] if idx + 1 < len(paragraphs) else ""
        return {"before": before, "after": after}
    except Exception as exc:
        logger.exception("Neighbor paragraph lookup failed: %s", exc)
        raise


def build_chunks(pdf_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build hierarchical + semantic + content-aware chunks."""
    try:
        chunks: List[Dict[str, Any]] = []
        chapter_counts: Dict[str, int] = {}

        for page in pdf_data["pages"]:
            page_number = page["page_number"]
            raw_text = page["text"].strip()
            if not raw_text:
                continue

            content_type = detect_content_type(raw_text)
            chapter_id = page["chapter_id"]
            chapter_counts.setdefault(chapter_id, 0)

            if content_type == "prose":
                splits = _split_text_semantic(raw_text, 450, 80)
                for split in splits:
                    chapter_counts[chapter_id] += 1
                    chunks.append(
                        _make_chunk(
                            page=page,
                            chunk_text=split,
                            content_type="prose",
                            chunk_index=chapter_counts[chapter_id],
                        )
                    )

            elif content_type == "code":
                code_splits = _split_code_blocks(raw_text)
                prose_prefix = _preceding_prose_prefix(raw_text)
                for block in code_splits:
                    combined = f"{prose_prefix}\n\n{block}".strip() if prose_prefix else block
                    chapter_counts[chapter_id] += 1
                    chunks.append(
                        _make_chunk(
                            page=page,
                            chunk_text=combined,
                            content_type="code",
                            chunk_index=chapter_counts[chapter_id],
                            language="python",
                        )
                    )

            elif content_type == "figure_context":
                for para in [p.strip() for p in raw_text.split("\n\n") if p.strip()]:
                    fig_match = FIGURE_PATTERN.search(para)
                    if not fig_match:
                        continue
                    fig_id = fig_match.group(1).replace(" ", "")
                    neighbors = _find_neighboring_paragraphs(raw_text, para)
                    synthetic = f"{neighbors['before']}\n\n{para}\n\n{neighbors['after']}".strip()
                    chapter_counts[chapter_id] += 1
                    chunks.append(
                        _make_chunk(
                            page=page,
                            chunk_text=synthetic,
                            content_type="figure_context",
                            chunk_index=chapter_counts[chapter_id],
                            figure_id=fig_id,
                        )
                    )
            else:
                chapter_counts[chapter_id] += 1
                chunks.append(
                    _make_chunk(
                        page=page,
                        chunk_text=raw_text,
                        content_type="prose",
                        chunk_index=chapter_counts[chapter_id],
                    )
                )

        bridge_chunks = _build_bridge_chunks(chunks)
        return chunks + bridge_chunks
    except Exception as exc:
        logger.exception("Chunk building failed: %s", exc)
        raise


def _build_bridge_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create bridge chunks within same chapter only."""
    try:
        bridges: List[Dict[str, Any]] = []
        by_chapter: Dict[str, List[Dict[str, Any]]] = {}
        enc = tiktoken.get_encoding("cl100k_base")
        for chunk in chunks:
            by_chapter.setdefault(chunk["chapter_id"], []).append(chunk)
        for chapter_id, items in by_chapter.items():
            sorted_items = sorted(items, key=lambda c: c["chunk_index_in_chapter"])
            for i in range(len(sorted_items) - 1):
                first = sorted_items[i]
                second = sorted_items[i + 1]
                tail_tokens = enc.encode(first["text"])[-100:]
                head_tokens = enc.encode(second["text"])[:100]
                bridge_text = enc.decode(tail_tokens + head_tokens).strip()
                bridge_id = f"{chapter_id}_p{first['page_start']}_bridge_{i+1}"
                bridges.append(
                    {
                        **first,
                        "id": bridge_id,
                        "text": bridge_text,
                        "token_count": len(tail_tokens + head_tokens),
                        "page_end": second["page_end"],
                        "content_type": "bridge",
                        "language": None,
                        "figure_id": None,
                        "chunk_index_in_chapter": 100000 + i + 1,
                        "retrieval_priority": "low",
                    }
                )
        return bridges
    except Exception as exc:
        logger.exception("Bridge chunk generation failed: %s", exc)
        raise


def _make_chunk(
    page: Dict[str, Any],
    chunk_text: str,
    content_type: str,
    chunk_index: int,
    language: Optional[str] = None,
    figure_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build normalized chunk metadata schema."""
    try:
        token_count = count_tokens(chunk_text)
        chapter_id = page["chapter_id"]
        chunk_id = f"{chapter_id}_p{page['page_number']}_{content_type}_{chunk_index}"
        has_figure = bool(FIGURE_PATTERN.search(chunk_text))
        has_code = bool(CODE_QUERY_PATTERN.search(chunk_text))
        return {
            "id": chunk_id,
            "text": chunk_text,
            "token_count": token_count,
            "part_id": page["part_id"],
            "part_title": page["part_title"],
            "chapter_id": chapter_id,
            "chapter_title": page["chapter_title"],
            "section_title": _extract_section_title(chunk_text),
            "page_start": page["page_number"],
            "page_end": page["page_number"],
            "chunk_index_in_chapter": chunk_index,
            "content_type": content_type,
            "has_code_reference": has_code,
            "has_figure_reference": has_figure,
            "language": language,
            "figure_id": figure_id,
        }
    except Exception as exc:
        logger.exception("Chunk metadata creation failed: %s", exc)
        raise


def _preceding_prose_prefix(page_text: str) -> str:
    """Find preceding prose paragraph for code context prefix."""
    try:
        paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]
        prefix = ""
        for para in paragraphs:
            if CODE_QUERY_PATTERN.search(para):
                break
            prefix = para
        return prefix
    except Exception as exc:
        logger.exception("Failed to compute prose prefix: %s", exc)
        return ""

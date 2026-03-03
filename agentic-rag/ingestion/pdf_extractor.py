"""PDF extraction utilities based on pdfplumber."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

import pdfplumber

logger = logging.getLogger(__name__)


def parse_toc_from_page1(page1_text: str) -> List[Dict[str, Any]]:
    """Parse a lightweight TOC structure from page 1 text."""
    try:
        entries: List[Dict[str, Any]] = []
        current_part = {"id": "part_0", "title": "Unknown Part"}
        chapter_count = 0
        part_count = 0

        for raw_line in page1_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            part_match = re.match(r"^(Part\s+\w+)\s*[:-]?\s*(.+)?$", line, flags=re.IGNORECASE)
            if part_match:
                part_count += 1
                part_name = part_match.group(1).strip()
                part_title = (part_match.group(2) or part_name).strip()
                part_id = f"part_{part_count}"
                current_part = {"id": part_id, "title": part_title}
                continue

            chapter_match = re.match(r"^(?:Chapter\s+)?(\d+)\.?\s+(.+?)\s+(\d+)$", line, flags=re.IGNORECASE)
            if chapter_match:
                chapter_count += 1
                entries.append(
                    {
                        "part_id": current_part["id"],
                        "part_title": current_part["title"],
                        "chapter_id": f"ch{chapter_match.group(1)}",
                        "chapter_title": chapter_match.group(2).strip(),
                        "start_page": int(chapter_match.group(3)),
                    }
                )

        if not entries:
            logger.warning("No TOC entries parsed from page 1; fallback metadata will be used.")
        else:
            for idx in range(len(entries) - 1):
                entries[idx]["end_page"] = entries[idx + 1]["start_page"] - 1
            entries[-1]["end_page"] = 10**9
        return entries
    except Exception as exc:
        logger.exception("Failed to parse TOC: %s", exc)
        raise


def extract_pdf_pages(pdf_path: str) -> Dict[str, Any]:
    """Extract text and page metadata for every page in a PDF."""
    try:
        pages: List[Dict[str, Any]] = []
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            page1_text = pdf.pages[0].extract_text() or ""
            toc_entries = parse_toc_from_page1(page1_text)

            for index, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                chapter_meta = _find_chapter_for_page(index, toc_entries)
                pages.append(
                    {
                        "page_number": index,
                        "text": text,
                        "part_id": chapter_meta["part_id"],
                        "part_title": chapter_meta["part_title"],
                        "chapter_id": chapter_meta["chapter_id"],
                        "chapter_title": chapter_meta["chapter_title"],
                    }
                )

        return {"total_pages": total_pages, "toc_entries": toc_entries, "pages": pages}
    except Exception as exc:
        logger.exception("PDF extraction failed for %s: %s", pdf_path, exc)
        raise


def _find_chapter_for_page(page_number: int, toc_entries: List[Dict[str, Any]]) -> Dict[str, str]:
    """Resolve chapter metadata for a page number."""
    try:
        for entry in toc_entries:
            if entry["start_page"] <= page_number <= entry.get("end_page", 10**9):
                return {
                    "part_id": entry["part_id"],
                    "part_title": entry["part_title"],
                    "chapter_id": entry["chapter_id"],
                    "chapter_title": entry["chapter_title"],
                }
        return {
            "part_id": "part_0",
            "part_title": "Unknown Part",
            "chapter_id": "ch0",
            "chapter_title": "Unknown Chapter",
        }
    except Exception as exc:
        logger.exception("Failed chapter lookup for page %s: %s", page_number, exc)
        raise

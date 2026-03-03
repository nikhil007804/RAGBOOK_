"""Tests for hybrid chunker behavior."""

from ingestion.chunker import build_chunks, detect_content_type


def test_detect_content_type_code() -> None:
    """Code detection should classify python-like text as code."""
    text = "def foo(x):\n    return x + 1\n"
    assert detect_content_type(text) == "code"


def test_detect_content_type_figure() -> None:
    """Figure detection should classify figure reference text."""
    text = "As shown in Figure 3, the architecture has three layers."
    assert detect_content_type(text) == "figure_context"


def test_build_chunks_generates_bridge_within_chapter() -> None:
    """Bridge chunks should be generated for adjacent chapter chunks."""
    pdf_data = {
        "pages": [
            {
                "page_number": 10,
                "text": "This is paragraph one.\n\nThis is paragraph two.",
                "part_id": "part_1",
                "part_title": "Part One",
                "chapter_id": "ch1",
                "chapter_title": "Intro",
            },
            {
                "page_number": 11,
                "text": "This is paragraph three.\n\nThis is paragraph four.",
                "part_id": "part_1",
                "part_title": "Part One",
                "chapter_id": "ch1",
                "chapter_title": "Intro",
            },
        ]
    }
    chunks = build_chunks(pdf_data)
    assert any(chunk["content_type"] == "bridge" for chunk in chunks)

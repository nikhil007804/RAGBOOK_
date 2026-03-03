"""Tests for retrieval helpers."""

from retrieval.hybrid_retriever import detect_query_type, resolve_alpha
from retrieval.rrf_fusion import rrf_fuse


def test_detect_query_type_code() -> None:
    """Code keyword should produce code query type."""
    assert detect_query_type("show me code example for planner function") == "code"


def test_resolve_alpha_defaults() -> None:
    """Alpha defaults should follow query class rules."""
    assert resolve_alpha("code", None) == 0.3
    assert resolve_alpha("conceptual", None) == 0.8


def test_rrf_fusion_merges_rank_lists() -> None:
    """RRF should merge lists and retain top documents."""
    dense = [{"id": "a", "metadata": {"text": "x"}}, {"id": "b", "metadata": {"text": "y"}}]
    sparse = [{"id": "b", "metadata": {"text": "y"}}, {"id": "c", "metadata": {"text": "z"}}]
    fused = rrf_fuse([dense, sparse], k=60, top_k=3)
    ids = [x["id"] for x in fused]
    assert "b" in ids

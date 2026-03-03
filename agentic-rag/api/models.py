"""Pydantic models for API I/O."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Incoming query request."""

    query: str = Field(..., min_length=1)
    alpha: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class Citation(BaseModel):
    """Citation payload."""

    page: int
    chapter: str
    excerpt: str


class QueryResponse(BaseModel):
    """Outgoing query response."""

    answer: str
    citations: List[Citation]
    confidence: Literal["high", "medium", "low"]
    source: Literal["cache", "pipeline"]

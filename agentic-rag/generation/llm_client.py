"""Gemini client wrapper for deterministic JSON generation."""

from __future__ import annotations

import logging

import google.generativeai as genai

from config.settings import settings

logger = logging.getLogger(__name__)


class GeminiClient:
    """Client for Gemini 2.5 Flash with fixed generation settings."""

    def __init__(self) -> None:
        """Initialize Gemini SDK and model."""
        try:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(
                model_name=settings.GEMINI_MODEL,
                generation_config={
                    "temperature": settings.GEMINI_TEMPERATURE,
                    "max_output_tokens": settings.GEMINI_MAX_OUTPUT_TOKENS,
                    "response_mime_type": "application/json",
                },
            )
        except Exception as exc:
            logger.exception("Gemini client initialization failed: %s", exc)
            raise

    def generate(self, prompt: str) -> str:
        """Generate raw JSON text from prompt."""
        try:
            response = self.model.generate_content(prompt)
            return (response.text or "").strip()
        except Exception as exc:
            logger.exception("Gemini generation failed: %s", exc)
            raise

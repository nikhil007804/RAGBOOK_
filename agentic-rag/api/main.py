"""FastAPI application bootstrap."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from api.router import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create FastAPI app and include routes."""
    try:
        app = FastAPI(title="agentic-rag", version="1.0.0")
        app.include_router(router)
        return app
    except Exception as exc:
        logger.exception("Failed to create FastAPI app: %s", exc)
        raise


app = create_app()

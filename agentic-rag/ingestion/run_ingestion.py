"""Master ingestion runner: extract -> chunk -> embed -> sparse -> upsert."""

from __future__ import annotations

import logging

from config.settings import settings
from ingestion.chunker import build_chunks
from ingestion.embedder import ChunkEmbedder
from ingestion.pdf_extractor import extract_pdf_pages
from ingestion.pinecone_upserter import PineconeUpserter
from ingestion.sparse_encoder import SparseBM25Encoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def run() -> None:
    """Run complete ingestion pipeline in strict order."""
    try:
        logger.info("Starting extraction from PDF: %s", settings.PDF_PATH)
        pdf_data = extract_pdf_pages(settings.PDF_PATH)

        logger.info("Building hybrid chunks")
        chunks = build_chunks(pdf_data)

        logger.info("Embedding %s chunks with BGE", len(chunks))
        embedder = ChunkEmbedder()
        chunks = embedder.embed_chunks(chunks)

        logger.info("Fitting and encoding BM25 sparse vectors")
        sparse = SparseBM25Encoder()
        sparse.fit(chunks)
        sparse.save()
        chunks = sparse.encode_documents(chunks)

        logger.info("Upserting vectors to Pinecone")
        upserter = PineconeUpserter()
        upserter.upsert_chunks(chunks)
        logger.info("Ingestion completed successfully")
    except Exception as exc:
        logger.exception("Ingestion failed: %s", exc)
        raise


if __name__ == "__main__":
    run()

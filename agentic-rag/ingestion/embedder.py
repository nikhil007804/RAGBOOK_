"""Dense embedding for document chunks using local Ollama model."""

from __future__ import annotations

import logging
from typing import Dict, List
import time
from pinecone import Pinecone

from config.settings import settings

logger = logging.getLogger(__name__)


class ChunkEmbedder:
    """Embeds chunk texts through Pinecone inference API."""

    def __init__(self) -> None:
        """Initialize Pinecone embedding client config."""
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.model_name = settings.PINECONE_EMBED_MODEL

    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Embed chunks and append dense vectors."""
        try:
            batch_texts: List[str] = []
            batch_indices: List[int] = []
            for idx, chunk in enumerate(chunks):
                text = chunk["text"]
                if len(text) > 2000:
                    text = text[:2000] # Approximate trim to avoid massive chunks
                batch_texts.append(text)
                batch_indices.append(idx)

                if len(batch_texts) >= 96:
                    for _ in range(5):
                        try:
                            embeddings = self.pc.inference.embed(
                                model=self.model_name,
                                inputs=batch_texts,
                                parameters={"input_type": "passage", "truncate": "END"}
                            )
                            break
                        except Exception as e:
                            if "429" in str(e):
                                logger.warning("Rate limited by Pinecone. Sleeping for 10 seconds...")
                                time.sleep(10)
                            else:
                                raise
                    for target_idx, record in zip(batch_indices, embeddings.data):
                        chunks[target_idx]["dense_vector"] = record.values
                    batch_texts.clear()
                    batch_indices.clear()

            if batch_texts:
                for _ in range(5):
                    try:
                        embeddings = self.pc.inference.embed(
                            model=self.model_name,
                            inputs=batch_texts,
                            parameters={"input_type": "passage", "truncate": "END"}
                        )
                        break
                    except Exception as e:
                        if "429" in str(e):
                            logger.warning("Rate limited by Pinecone. Sleeping for 10 seconds...")
                            time.sleep(10)
                        else:
                            raise
                for target_idx, record in zip(batch_indices, embeddings.data):
                    chunks[target_idx]["dense_vector"] = record.values
            return chunks
        except Exception as exc:
            logger.exception("Chunk embedding failed: %s", exc)
            raise

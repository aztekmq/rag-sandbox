"""Embedding utilities backed by Snowflake Arctic models."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Iterable, List

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "Snowflake/snowflake-arctic-embed-xs"


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Load and cache the embedding model."""

    logger.info("Loading embedding model: %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    logger.info("Embedding model loaded successfully")
    return model


def embed_documents(texts: Iterable[str]) -> List[List[float]]:
    """Embed a batch of documents with detailed logging."""

    texts_list = list(texts)
    logger.debug("Embedding %d documents", len(texts_list))
    model = get_embedder()
    embeddings = model.encode(texts_list, show_progress_bar=False).tolist()
    logger.debug("Generated %d embeddings", len(embeddings))
    return embeddings


def embed_query(text: str) -> List[float]:
    """Embed a single query string for retrieval."""

    logger.debug("Embedding query text of length %d", len(text))
    model = get_embedder()
    vector = model.encode([text], show_progress_bar=False)[0].tolist()
    logger.debug("Query embedding created")
    return vector

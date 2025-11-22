"""Embedding utilities backed by Snowflake Arctic models.

The embedder is configured to run fully offline by default to keep container
deployments from reaching Hugging Face or other public endpoints. Operators can
override this behavior by setting the ``ALLOW_HF_INTERNET`` environment
variable to ``true`` and providing ``EMBEDDING_MODEL_ID`` to point at a
downloadable repository. Otherwise, the embedder will only load models from the
local ``EMBEDDING_MODEL_DIR`` path and raise explicit errors when the assets are
missing.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Iterable, List

from pathlib import Path

from sentence_transformers import SentenceTransformer

from app.config import ALLOW_HF_INTERNET, EMBEDDING_MODEL_DIR, EMBEDDING_MODEL_ID

logger = logging.getLogger(__name__)

MODEL_NAME = EMBEDDING_MODEL_ID


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Load and cache the embedding model."""

    logger.info(
        "Loading embedding model: %s (offline=%s)", MODEL_NAME, not ALLOW_HF_INTERNET
    )

    # Avoid unexpected Hugging Face calls by requiring the assets to exist
    # locally when offline defaults are in effect. This early guard makes the
    # failure mode explicit and keeps the loader from attempting network
    # resolution through the hub libraries.
    model_path = Path(MODEL_NAME)
    if not ALLOW_HF_INTERNET and not model_path.exists():
        raise FileNotFoundError(
            f"Embedding model directory '{model_path}' is missing. Place the"
            f" downloaded assets under '{EMBEDDING_MODEL_DIR}' or set"
            " ALLOW_HF_INTERNET=true to permit a one-time download."
        )
    try:
        model = SentenceTransformer(
            MODEL_NAME,
            cache_folder=str(EMBEDDING_MODEL_DIR),
            local_files_only=not ALLOW_HF_INTERNET,
        )
    except OSError as exc:  # pragma: no cover - defensive logging path
        logger.error(
            "Embedding assets not found in %s. Ensure the model is present on disk or"
            " allow internet downloads via ALLOW_HF_INTERNET=true. Error: %s",
            EMBEDDING_MODEL_DIR,
            exc,
        )
        raise

    logger.info("Embedding model loaded successfully from %s", EMBEDDING_MODEL_DIR)
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

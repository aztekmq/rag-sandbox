"""Demonstration script for a minimal RAG workflow using ChromaDB.

This module builds an in-memory vector store from a handful of IBM MQ reference
snippets and answers a user-supplied query. The implementation favors verbose
logging and extensive documentation to align with international programming
standards and auditability best practices.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import chromadb
from chromadb.utils import embedding_functions

LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
DEFAULT_EMBEDDING_MODEL_ID = os.getenv(
    "EMBEDDING_MODEL_ID", "Snowflake/snowflake-arctic-embed-xs"
)


@dataclass
class Document:
    """Structured representation of a text record to be indexed."""

    title: str
    content: str


class DemoRAG:
    """Small Retrieval-Augmented Generation helper built for demonstrations."""

    def __init__(self, embedding_model_id: str) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("Initializing DemoRAG with model '%s'", embedding_model_id)
        self.embedding_model_id = embedding_model_id
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_id, trust_remote_code=False
        )
        self.client = chromadb.Client(chromadb.config.Settings(anonymized_telemetry=False))
        self.collection = self.client.create_collection(
            name="demo_documents",
            embedding_function=self.embedding_function,
            metadata={"purpose": "demo_01"},
        )
        self.logger.debug("ChromaDB collection 'demo_documents' created successfully")

    def index_documents(self, documents: Iterable[Document]) -> None:
        """Add documents to the vector store with verbose tracing."""

        titles: List[str] = []
        contents: List[str] = []
        ids: List[str] = []

        for idx, document in enumerate(documents, start=1):
            doc_id = f"doc-{idx:03d}"
            ids.append(doc_id)
            titles.append(document.title)
            contents.append(document.content)
            self.logger.debug("Prepared document %s titled '%s' for indexing", doc_id, document.title)

        self.collection.upsert(documents=contents, ids=ids, metadatas=[{"title": t} for t in titles])
        self.logger.info("Indexed %d documents into ChromaDB", len(ids))

    def query(self, text: str, results: int = 3) -> List[dict]:
        """Retrieve the most relevant documents for the provided query text."""

        self.logger.debug("Running similarity search for query: %s", text)
        response = self.collection.query(query_texts=[text], n_results=results)
        hits: List[dict] = []

        for idx, (doc_id, content, metadata, distance) in enumerate(
            zip(
                response.get("ids", [[]])[0],
                response.get("documents", [[]])[0],
                response.get("metadatas", [[]])[0],
                response.get("distances", [[]])[0],
            ),
            start=1,
        ):
            hit = {
                "rank": idx,
                "id": doc_id,
                "title": metadata.get("title", "untitled"),
                "content": content,
                "distance": distance,
            }
            hits.append(hit)
            self.logger.debug("Hit %d -> %s", idx, hit)

        self.logger.info("Retrieved %d results for the query", len(hits))
        return hits


def configure_logging(log_level: str) -> None:
    """Configure root logging with both console and file handlers."""

    logger = logging.getLogger()
    logger.setLevel(log_level)

    formatter = logging.Formatter(LOG_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_path = Path("data/logs/demo_01.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.debug("Logging configured. Level: %s, File: %s", log_level, log_path)


def build_sample_documents() -> List[Document]:
    """Generate a small corpus describing the RAG sandbox capabilities."""

    return [
        Document(
            title="Architecture Overview",
            content=(
                "The sandbox pairs a llama.cpp language model with a local Chroma vector store. "
                "Embeddings rely on the Snowflake Arctic model to keep inference offline-first."
            ),
        ),
        Document(
            title="Logging Defaults",
            content=(
                "All services emit DEBUG-level logs to stdout and data/logs/app.log. "
                "This ensures auditability for IBM MQ reference experiments."
            ),
        ),
        Document(
            title="Operational Notes",
            content=(
                "PDFs live under data/pdfs, while embeddings persist in data/chroma_db. "
                "A prewarm thread primes caches at startup for faster first responses."
            ),
        ),
        Document(
            title="Ollama Stack",
            content=(
                "Helper scripts launch an Ollama server plus a Gradio UI for quick local testing. "
                "Commands run with verbose tracing to simplify debugging."
            ),
        ),
    ]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for interactive querying."""

    parser = argparse.ArgumentParser(description="Run a minimal RAG similarity search demo.")
    parser.add_argument(
        "query",
        type=str,
        help="Question or search phrase to run against the in-memory vector store.",
    )
    parser.add_argument(
        "--model",
        dest="model",
        default=DEFAULT_EMBEDDING_MODEL_ID,
        help=(
            "Sentence Transformer model identifier. Defaults to the Snowflake Arctic embedding model "
            "to stay consistent with the main stack's offline behavior."
        ),
    )
    parser.add_argument(
        "--results",
        dest="results",
        type=int,
        default=3,
        help="Number of similarity hits to return (default: 3).",
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        default="DEBUG",
        help="Logging level (DEBUG, INFO, WARNING, ERROR). Default is DEBUG for traceability.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the demo: configure logging, index, and query."""

    args = parse_args()
    configure_logging(args.log_level)
    logger = logging.getLogger("demo_01")
    logger.debug("Starting demo with args: %s", args)

    rag = DemoRAG(embedding_model_id=args.model)
    sample_docs = build_sample_documents()
    rag.index_documents(sample_docs)

    logger.info("Executing query: %s", args.query)
    hits = rag.query(args.query, results=args.results)

    logger.info("Top %d results:", len(hits))
    for hit in hits:
        logger.info(
            "#%d | %s | distance=%.4f\n%s", hit["rank"], hit["title"], hit["distance"], hit["content"]
        )


if __name__ == "__main__":
    main()

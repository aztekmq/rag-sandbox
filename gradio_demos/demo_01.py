"""Gradio-powered RAG playground that bridges to a local Ollama instance.

This module builds an in-memory ChromaDB vector store from a handful of IBM MQ
reference snippets, exposes a Gradio chat UI, and forwards contextualized
queries to a locally running Ollama server (e.g., started via Docker). The
implementation favors verbose logging and extensive documentation to align with
international programming standards and auditability best practices.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import chromadb
import requests

logger = logging.getLogger(__name__)


def _ensure_huggingface_shim() -> None:
    """Guarantee Gradio's optional Hugging Face hooks do not block local use.

    Some Gradio versions import ``get_token`` and ``whoami`` from
    ``huggingface_hub`` at module load time. Older hub releases omit these
    helpers, which can cause an ImportError even when Hugging Face connectivity
    is not required. To preserve a fully local Ollama workflow, we proactively
    patch missing attributes with lightweight no-op shims.
    """

    try:
        hf_hub = importlib.import_module("huggingface_hub")
        shim_applied = False

        if not hasattr(hf_hub, "get_token"):
            shim_applied = True

            def _noop_get_token(*_: object, **__: object) -> None:
                logger.debug("huggingface_hub.get_token shimmed to a no-op")
                return None

            setattr(hf_hub, "get_token", _noop_get_token)

        if not hasattr(hf_hub, "whoami"):
            shim_applied = True

            def _noop_whoami(*_: object, **__: object) -> dict:
                logger.debug("huggingface_hub.whoami shimmed to a no-op")
                return {}

            setattr(hf_hub, "whoami", _noop_whoami)

        if shim_applied:
            logger.info(
                "Applied Hugging Face compatibility shims; running Ollama demo fully locally"
            )
    except ModuleNotFoundError:
        logger.info(
            "huggingface_hub not installed; proceeding with local-only Ollama configuration"
        )
    except Exception as exc:  # pragma: no cover - defensive guard for unexpected import issues
        logger.exception("Unexpected Hugging Face shim failure: %s", exc)


_ensure_huggingface_shim()
import gradio as gr

LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
DEFAULT_EMBEDDING_MODEL_ID = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


@dataclass
class Document:
    """Structured representation of a text record to be indexed."""

    title: str
    content: str


class OllamaEmbeddingFunction:
    """Minimal embedding adapter that delegates to a local Ollama instance."""

    def __init__(self, base_url: str, model: str, timeout: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(
            "Initialized embedding function | base_url=%s | model=%s | timeout=%ss",
            self.base_url,
            self.model,
            self.timeout,
        )

    def __call__(self, texts: Iterable[str]) -> List[List[float]]:
        """Return embeddings for the provided texts via the Ollama HTTP API."""

        embeddings: List[List[float]] = []
        for text in texts:
            payload = {"model": self.model, "prompt": text}
            self.logger.debug(
                "Requesting embedding | url=%s/api/embeddings | model=%s | payload_length=%d",
                self.base_url,
                self.model,
                len(text),
            )

            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                content = response.json()
                vector = content.get("embedding")

                if not vector:
                    raise ValueError("Ollama response missing 'embedding' field")

                embeddings.append(vector)
                self.logger.debug(
                    "Received embedding with %d dimensions from Ollama", len(vector)
                )
            except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
                self.logger.exception("Failed to retrieve embedding from Ollama: %s", exc)
                raise

        self.logger.info("Generated %d embeddings via Ollama", len(embeddings))
        return embeddings


class DemoRAG:
    """Small Retrieval-Augmented Generation helper built for demonstrations."""

    def __init__(self, embedding_model_id: str, embedding_url: str) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(
            "Initializing DemoRAG with embedding model '%s' at '%s'",
            embedding_model_id,
            embedding_url,
        )
        self.embedding_model_id = embedding_model_id
        self.embedding_function = OllamaEmbeddingFunction(
            base_url=embedding_url, model=embedding_model_id
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


@dataclass
class OllamaConfig:
    """Configuration block for communicating with the Ollama server."""

    base_url: str
    model: str


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
                "Embeddings rely on the Ollama embedding endpoint to keep inference fully local."
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
    """Parse CLI arguments for the Gradio + Ollama demo."""

    parser = argparse.ArgumentParser(
        description=(
            "Launch a Gradio UI that retrieves demo documents and forwards the "
            "augmented prompt to a locally running Ollama model."
        )
    )
    parser.add_argument(
        "--model",
        dest="model",
        default=DEFAULT_EMBEDDING_MODEL_ID,
        help=(
            "Ollama embedding model to use for vectorization. Defaults to the Nomic embed model "
            "to keep the pipeline fully local."
        ),
    )
    parser.add_argument(
        "--results",
        dest="results",
        type=int,
        default=3,
        help="Number of similarity hits to return per user message (default: 3).",
    )
    parser.add_argument(
        "--ollama-url",
        dest="ollama_url",
        default=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        help="Base URL for the Ollama HTTP service (default: http://localhost:11434).",
    )
    parser.add_argument(
        "--ollama-model",
        dest="ollama_model",
        default=os.getenv("OLLAMA_MODEL", "llama3"),
        help="Ollama model identifier to query (default: llama3).",
    )
    parser.add_argument(
        "--embedding-url",
        dest="embedding_url",
        default=os.getenv("OLLAMA_EMBED_URL"),
        help=(
            "Base URL for the Ollama embedding endpoint. Defaults to the value of --ollama-url "
            "when omitted."
        ),
    )
    parser.add_argument(
        "--host",
        dest="host",
        default=os.getenv("GRADIO_HOST", "0.0.0.0"),
        help="Host interface for the Gradio server (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        dest="port",
        type=int,
        default=int(os.getenv("GRADIO_PORT", "7861")),
        help="Port for the Gradio server (default: 7861).",
    )
    parser.add_argument(
        "--share",
        dest="share",
        action="store_true",
        help="Enable Gradio share links for remote testing (disabled by default).",
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        default="DEBUG",
        help="Logging level (DEBUG, INFO, WARNING, ERROR). Default is DEBUG for traceability.",
    )
    return parser.parse_args()


def build_augmented_prompt(user_message: str, hits: List[dict]) -> str:
    """Compose a structured prompt that embeds retrieved context."""

    context_lines = [f"{hit['title']}: {hit['content']}" for hit in hits]
    context_block = "\n".join(context_lines)
    return (
        "You are a helpful assistant answering questions about the RAG sandbox demo. "
        "Use the provided context when it is relevant.\n\n"
        f"Context:\n{context_block}\n\nUser: {user_message}\nAssistant:"
    )


def stream_ollama_response(prompt: str, config: OllamaConfig):
    """Stream tokens from the Ollama chat endpoint for Gradio display."""

    logger = logging.getLogger("stream_ollama_response")
    logger.debug("Preparing request to Ollama | url=%s | model=%s", config.base_url, config.model)

    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }

    try:
        with requests.post(
            f"{config.base_url}/api/chat",
            json=payload,
            stream=True,
            timeout=120,
        ) as response:
            response.raise_for_status()
            logger.info("Ollama connection established; streaming response")

            for line in response.iter_lines():
                if not line:
                    continue

                data = json.loads(line.decode("utf-8"))
                token = data.get("message", {}).get("content", "")

                if token:
                    logger.debug("Streaming token: %s", token)
                    yield token
    except requests.RequestException as exc:
        logger.exception("Failed to stream response from Ollama: %s", exc)
        yield f"[Error] Unable to contact Ollama at {config.base_url}: {exc}"


def build_gradio_app(rag: DemoRAG, config: OllamaConfig, results: int) -> gr.Blocks:
    """Create the Gradio interface that orchestrates retrieval and generation."""

    logger = logging.getLogger("build_gradio_app")
    logger.debug("Constructing Gradio app with results=%d", results)

    def respond(message: str, history: list[list[str]]):
        logger.info("Received user message: %s", message)
        hits = rag.query(message, results=results)
        prompt = build_augmented_prompt(message, hits)
        logger.debug("Built augmented prompt: %s", prompt)
        return stream_ollama_response(prompt, config)

    chatbot = gr.ChatInterface(
        respond,
        title="RAG + Ollama Demo",
        description=(
            "Retrieves context from a tiny ChromaDB collection and forwards the "
            "augmented prompt to a local Ollama model for generation."
        ),
    )

    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # Retrieval-Augmented Generation with Ollama

            This UI retrieves reference snippets from an in-memory ChromaDB collection and
            streams responses from a locally running Ollama model. Ensure your Ollama Docker
            container is running and accessible at the configured URL before starting the demo.
            """
        )
        demo_chat = chatbot.render()
        gr.Markdown("Logs are written to `data/logs/demo_01.log` for auditability.")
        gr.HTML("""<small>Verbose logging is enabled to simplify debugging.</small>""")

    logger.info("Gradio interface constructed successfully")
    return demo


def main() -> None:
    """Entrypoint for the Gradio + Ollama demonstration."""

    args = parse_args()
    configure_logging(args.log_level)
    logger = logging.getLogger("demo_01")
    logger.debug("Starting demo with args: %s", args)

    embedding_url = args.embedding_url or args.ollama_url
    logger.info(
        "Using embedding configuration | url=%s | model=%s", embedding_url, args.model
    )

    rag = DemoRAG(embedding_model_id=args.model, embedding_url=embedding_url)
    rag.index_documents(build_sample_documents())

    ollama_config = OllamaConfig(base_url=args.ollama_url, model=args.ollama_model)
    logger.info(
        "Initializing Gradio server | host=%s | port=%s | ollama_url=%s | ollama_model=%s",
        args.host,
        args.port,
        ollama_config.base_url,
        ollama_config.model,
    )

    app = build_gradio_app(rag, ollama_config, results=args.results)
    app.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

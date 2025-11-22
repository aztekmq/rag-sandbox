"""Core retrieval-augmented generation (RAG) pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import chromadb
from chromadb.config import Settings
from llama_cpp import Llama

from app.config import CHROMA_DIR, MODEL_N_CTX, MODEL_PATH, MODEL_THREADS, PDF_DIR
from app.utils.embeddings import embed_documents, embed_query
from app.utils.pdf_ingest import ingest_pdf_files

logger = logging.getLogger(__name__)


class ModelPathError(FileNotFoundError):
    """Raised when the configured MODEL_PATH is missing or invalid."""

    def __init__(self, model_path: Path):
        super().__init__(
            f"Model path does not exist: {model_path}. Download the GGUF to this location or set MODEL_PATH to an existing file."
        )
        self.model_path = model_path


class RagEngine:
    """Encapsulates vector store, embeddings, and LLM inference."""

    def __init__(self) -> None:
        if not MODEL_PATH:
            raise ValueError("MODEL_PATH must be set to the GGUF file")

        model_path = Path(MODEL_PATH)
        logger.debug("Resolved model path to %s", model_path)
        if not model_path.is_file():
            logger.error(
                "Model path does not exist: %s. Ensure the GGUF file is mounted or download it via README instructions.",
                model_path,
            )
            raise ModelPathError(model_path)

        logger.info("Initializing RAG engine with model at %s", model_path)
        logger.debug("Configuring Chroma client with telemetry disabled via Settings")

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(
                anonymized_telemetry=False,
            ),
        )
        self.collection = self.client.get_or_create_collection(
            name="ibm-mq-docs", metadata={"description": "IBM MQ reference"}
        )
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=MODEL_N_CTX,
            n_threads=MODEL_THREADS,
            verbose=True,
        )
        logger.info("RAG engine ready")

    def ingest(self, chunk_size: int = 1200, chunk_overlap: int = 200) -> str:
        pdf_paths = list(Path(PDF_DIR).glob("*.pdf"))
        logger.info("Starting ingestion for %d PDFs", len(pdf_paths))
        chunks, metadata = ingest_pdf_files(pdf_paths, chunk_size, chunk_overlap)

        if not chunks:
            logger.warning("No chunks produced during ingestion")
            return "No PDFs found to ingest."

        embeddings = embed_documents(chunks)
        ids = [f"chunk-{i}" for i in range(len(chunks))]
        self.collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadata)
        logger.info("Ingested %d chunks into vector store", len(chunks))
        return f"Ingested {len(chunks)} chunks from {len(pdf_paths)} PDFs"

    def delete_document(self, source_name: str) -> str:
        logger.info("Deleting document from vector store: %s", source_name)
        results = self.collection.get(where={"source": source_name})
        ids = results.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)
            logger.info("Removed %d chunks for document %s", len(ids), source_name)
            return f"Removed {len(ids)} chunks for {source_name}"
        logger.warning("No chunks found for %s", source_name)
        return f"No entries found for {source_name}"

    def list_documents(self) -> List[str]:
        metadata = self.collection.get()["metadatas"]
        if not metadata:
            return []
        docs = sorted({item.get("source", "unknown") for item in metadata})
        logger.debug("Available documents: %s", docs)
        return docs

    def query(self, question: str, top_k: int = 4) -> str:
        logger.info("Querying vector store with question: %s", question)
        query_embedding = embed_query(question)
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        contexts = results.get("documents", [[]])[0]
        if not contexts:
            logger.warning("No context retrieved; ensure PDFs are ingested")
            return "I could not find relevant content. Please ingest PDFs in admin mode first."
        context_text = "\n".join(contexts)
        logger.debug("Retrieved %d contexts for generation", len(contexts))

        prompt = (
            "You are an assistant that answers questions about IBM MQ using the provided context. "
            "Keep answers concise and cite the source document names.\n\nContext:\n"
            f"{context_text}\n\nQuestion: {question}\nAnswer:"
        )

        completion = self.llm(
            prompt,
            temperature=0.2,
            max_tokens=512,
            stop=["User:", "Question:"],
        )
        answer = completion["choices"][0]["text"].strip()
        logger.info("Generated answer with %d tokens", len(answer.split()))
        return answer


_engine: RagEngine | None = None


def get_engine() -> RagEngine:
    global _engine
    if _engine is None:
        _engine = RagEngine()
    return _engine


def ingest_pdfs() -> str:
    engine, error_msg = _safe_get_engine()
    if error_msg:
        return error_msg
    return engine.ingest()


def query_rag(question: str) -> str:
    engine, error_msg = _safe_get_engine()
    if error_msg:
        return error_msg
    return engine.query(question)


def get_documents_list() -> List[str]:
    engine, error_msg = _safe_get_engine()
    if error_msg:
        return []
    return engine.list_documents()


def delete_document(source_name: str) -> str:
    engine, error_msg = _safe_get_engine()
    if error_msg:
        return error_msg
    return engine.delete_document(source_name)


def _safe_get_engine() -> Tuple[RagEngine | None, str]:
    """Return a ready RAG engine or a user-facing error message.

    This helper centralizes defensive handling for model path failures so that UI
    interactions degrade gracefully when the GGUF file is missing. Verbose
    logging captures the full exception chain for operators while end users
    receive actionable guidance instead of a stack trace.
    """

    try:
        return get_engine(), ""
    except ModelPathError as exc:
        logger.exception(
            "RAG engine initialization failed because the model file is missing at %s",
            exc.model_path,
        )
        return None, (
            "The language model file is missing. Please download the GGUF to "
            f"{MODEL_PATH} or set MODEL_PATH to an existing file before retrying."
        )
    except Exception:
        logger.exception("Unexpected failure while preparing the RAG engine")
        return None, "The RAG engine is unavailable due to an unexpected error. Check logs for details."

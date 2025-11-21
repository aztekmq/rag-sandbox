"""Core retrieval-augmented generation (RAG) pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings
from llama_cpp import Llama

from app.config import CHROMA_DIR, MODEL_N_CTX, MODEL_PATH, MODEL_THREADS, PDF_DIR
from app.utils.embeddings import embed_documents, embed_query
from app.utils.pdf_ingest import ingest_pdf_files

logger = logging.getLogger(__name__)


class RagEngine:
    """Encapsulates vector store, embeddings, and LLM inference."""

    def __init__(self) -> None:
        if not MODEL_PATH:
            raise ValueError("MODEL_PATH must be set to the GGUF file")
        logger.info("Initializing RAG engine with model at %s", MODEL_PATH)

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="ibm-mq-docs", metadata={"description": "IBM MQ reference"}
        )
        self.llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=MODEL_N_CTX,
            n_threads=MODEL_THREADS,
            verbose=False,
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
    engine = get_engine()
    return engine.ingest()


def query_rag(question: str) -> str:
    engine = get_engine()
    return engine.query(question)


def get_documents_list() -> List[str]:
    engine = get_engine()
    return engine.list_documents()


def delete_document(source_name: str) -> str:
    engine = get_engine()
    return engine.delete_document(source_name)

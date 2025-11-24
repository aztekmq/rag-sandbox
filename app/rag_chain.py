"""Core retrieval-augmented generation (RAG) pipeline with live ETA telemetry.

The module now instruments retrieval, prefill, and token generation phases with
fine-grained timestamps so user interfaces can surface accurate progress
indicators. Streaming generation yields incremental updates that include
tokens-per-second measurements and remaining-time estimates derived from actual
hardware throughput, providing a responsive experience even on constrained
deployments.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Tuple

import chromadb
from chromadb.config import Settings
from llama_cpp import Llama

from app.config import CHROMA_DIR, MODEL_N_CTX, MODEL_PATH, MODEL_THREADS, PDF_DIR
from app.utils.embeddings import embed_documents, embed_query
from app.utils.pdf_ingest import ingest_pdf_files

logger = logging.getLogger(__name__)
# Suppress noisy telemetry errors from upstream dependencies while retaining
# detailed application-level logs for debugging.
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
logger.debug("Chromadb telemetry loggers set to CRITICAL to avoid noisy stack traces")


class ModelPathError(FileNotFoundError):
    """Raised when the configured MODEL_PATH is missing."""

    def __init__(self, model_path: Path):
        super().__init__(
            f"Model path does not exist: {model_path}. Download the GGUF to this location or set MODEL_PATH to an existing file."
        )
        self.model_path = model_path


class ModelValidationError(ValueError):
    """Raised when the configured MODEL_PATH exists but is not a valid GGUF file."""

    def __init__(self, model_path: Path, detail: str):
        message = (
            "The model file is present but appears corrupted or incompatible. "
            f"{detail}"
        )
        super().__init__(message)
        self.model_path = model_path
        self.detail = detail


@dataclass
class GenerationProgress:
    """Represents a streaming update during retrieval or generation.

    Attributes:
        stage: Semantic phase marker (``retrieval``, ``prefill_estimate``,
            ``prefill_complete``, ``generation``, or ``done``).
        detail: Human-readable status string for logging and UI surfaces.
        retrieval_seconds: Observed retrieval duration, when available.
        prefill_seconds: Observed or estimated prefill duration in seconds.
        prompt_tokens: Number of tokens in the assembled prompt.
        decode_tokens: Count of generated tokens so far.
        tokens_per_second: Measured decode throughput.
        eta_seconds: Estimated remaining seconds for the current stage.
        partial_answer: Accumulated model text so far.
        total_prompt_seconds: Historical average prompt eval, useful for telemetry.
    """

    stage: str
    detail: str
    retrieval_seconds: float | None = None
    prefill_seconds: float | None = None
    prompt_tokens: int | None = None
    decode_tokens: int = 0
    tokens_per_second: float | None = None
    eta_seconds: float | None = None
    partial_answer: str = ""
    total_prompt_seconds: float | None = None


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

        self._validate_model_file(model_path)

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
        self.prompt_eval_tps_history: list[float] = []
        self.decode_tps_history: list[float] = []
        self.retrieval_history: list[float] = []
        logger.info("RAG engine ready")

    # ------------------------------------------------------------------
    # Throughput estimation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_average(values: list[float], default: float) -> float:
        """Return the average of ``values`` or a sensible fallback with logging."""

        if not values:
            logger.debug("No historical values available; using default fallback %.2f", default)
            return default

        average = sum(values) / len(values)
        logger.debug("Computed rolling average %.3f from %d samples", average, len(values))
        return average

    def _estimate_prompt_eval_tps(self) -> float:
        """Estimate prompt-eval throughput using history or a defensive default."""

        # Conservative default keeps ETAs realistic for CPU-only environments.
        return self._safe_average(self.prompt_eval_tps_history, default=25.0)

    def _estimate_decode_tps(self) -> float:
        """Estimate decode throughput using history or a defensive default."""

        return self._safe_average(self.decode_tps_history, default=15.0)

    # ------------------------------------------------------------------
    # Retrieval and generation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_model_file(model_path: Path) -> None:
        """Ensure the configured GGUF model exists and contains the expected header.

        The llama.cpp loader emits opaque errors when presented with an empty or
        non-GGUF file. This proactive check inspects the file size and first
        bytes to confirm the ``GGUF`` magic signature, logging verbose context
        for operators and raising :class:`ModelValidationError` with actionable
        remediation guidance when the file appears corrupted.
        """

        logger.debug("Validating GGUF model integrity at %s", model_path)

        if model_path.suffix.lower() != ".gguf":
            detail = "Expected a .gguf extension; please provide a valid GGUF model file."
            logger.error(detail)
            raise ModelValidationError(model_path, detail)

        minimum_size_bytes = 1024
        file_size = model_path.stat().st_size
        logger.debug("Model file size: %d bytes", file_size)
        if file_size < minimum_size_bytes:
            detail = (
                f"File is too small ({file_size} bytes). Download the complete GGUF artifact before retrying."
            )
            logger.error(detail)
            raise ModelValidationError(model_path, detail)

        with model_path.open("rb") as handle:
            header = handle.read(4)

        if header != b"GGUF":
            detail = "Missing GGUF magic header; the file may be corrupted or an incorrect download."
            logger.error(detail)
            raise ModelValidationError(model_path, detail)

        logger.info("Model file %s passed GGUF validation", model_path)

    def ingest(self, chunk_size: int = 1200, chunk_overlap: int = 200) -> str:
        pdf_paths = list(Path(PDF_DIR).glob("*.pdf"))
        logger.info("Starting ingestion for %d PDFs", len(pdf_paths))
        chunks, metadata = ingest_pdf_files(pdf_paths, chunk_size, chunk_overlap)

        if not chunks:
            logger.warning("No chunks produced during ingestion")
            return "No PDFs found to ingest."

        try:
            embeddings = embed_documents(chunks)
        except FileNotFoundError as exc:
            logger.exception(
                "Embedding model assets missing; ensure offline bundle exists or enable internet downloads"
            )
            return (
                "Embedding assets are missing for the Snowflake Arctic model. "
                "Place the model files under the configured EMBEDDING_MODEL_DIR or set "
                "ALLOW_HF_INTERNET=true to allow automatic downloads."
            )
        except Exception:
            logger.exception("Unexpected failure while generating embeddings during ingestion")
            return "Embedding failed due to an unexpected error. Check logs for details."

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

    def stream_query(self, question: str, top_k: int = 4, max_tokens: int = 512) -> Generator[GenerationProgress, None, None]:
        """Stream retrieval and generation progress with live ETA estimates."""

        logger.debug(
            "Entering stream_query with question=%r, top_k=%d, max_tokens=%d",
            question,
            top_k,
            max_tokens,
        )
        logger.info("Querying vector store with question: %s", question)
        retrieval_start = time.perf_counter()
        query_embedding = embed_query(question)
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        retrieval_elapsed = time.perf_counter() - retrieval_start
        self.retrieval_history.append(retrieval_elapsed)
        logger.info("Retrieval completed in %.3fs", retrieval_elapsed)

        contexts = results.get("documents", [[]])[0]
        if not contexts:
            logger.warning("No context retrieved; ensure PDFs are ingested")
            yield GenerationProgress(
                stage="retrieval",
                detail="No context found during retrieval.",
                retrieval_seconds=retrieval_elapsed,
            )
            return

        context_text = "\n".join(contexts)
        logger.debug("Retrieved %d contexts for generation", len(contexts))

        prompt = (
            "You are an assistant that answers questions about IBM MQ using the provided context. "
            "Keep answers concise and cite the source document names.\n\nContext:\n"
            f"{context_text}\n\nQuestion: {question}\nAnswer:"
        )

        prompt_tokens = len(self.llm.tokenize(prompt.encode("utf-8")))
        estimated_prefill = prompt_tokens / self._estimate_prompt_eval_tps()
        logger.debug(
            "Prompt length %d tokens; estimated prefill %.2fs using avg tps %.2f",
            prompt_tokens,
            estimated_prefill,
            self._estimate_prompt_eval_tps(),
        )

        yield GenerationProgress(
            stage="retrieval",
            detail=f"Retrieval completed in {retrieval_elapsed:.2f}s; estimating prefill.",
            retrieval_seconds=retrieval_elapsed,
            prompt_tokens=prompt_tokens,
            prefill_seconds=estimated_prefill,
            total_prompt_seconds=estimated_prefill,
        )

        decode_generator = self.llm(
            prompt,
            temperature=0.2,
            max_tokens=max_tokens,
            stop=["User:", "Question:"],
            stream=True,
        )

        generation_start = time.perf_counter()
        partial_answer = ""
        first_token_time: float | None = None
        tokens_generated = 0
        for index, chunk in enumerate(decode_generator):
            token_text = chunk.get("choices", [{}])[0].get("text", "")
            partial_answer += token_text
            tokens_generated += 1

            if first_token_time is None:
                first_token_time = time.perf_counter()
                prefill_duration = first_token_time - generation_start
                prompt_eval_tps = prompt_tokens / prefill_duration if prefill_duration > 0 else None
                if prompt_eval_tps:
                    self.prompt_eval_tps_history.append(prompt_eval_tps)
                logger.info(
                    "Prefill completed in %.3fs for %d tokens (%.2f tok/s)",
                    prefill_duration,
                    prompt_tokens,
                    prompt_eval_tps or -1.0,
                )
                yield GenerationProgress(
                    stage="prefill_complete",
                    detail=(
                        f"Prefill finished in {prefill_duration:.2f}s. Preparing live token ETA."
                    ),
                    retrieval_seconds=retrieval_elapsed,
                    prefill_seconds=prefill_duration,
                    prompt_tokens=prompt_tokens,
                    partial_answer=partial_answer,
                    total_prompt_seconds=prefill_duration,
                )
                continue

            decode_elapsed = time.perf_counter() - first_token_time
            tokens_per_second = tokens_generated / decode_elapsed if decode_elapsed > 0 else None
            eta_remaining = (
                (max_tokens - tokens_generated) / tokens_per_second if tokens_per_second else None
            )

            yield GenerationProgress(
                stage="generation",
                detail="Streaming tokens with live ETA updates.",
                retrieval_seconds=retrieval_elapsed,
                prefill_seconds=first_token_time - generation_start if first_token_time else None,
                prompt_tokens=prompt_tokens,
                decode_tokens=tokens_generated,
                tokens_per_second=tokens_per_second,
                eta_seconds=eta_remaining,
                partial_answer=partial_answer,
            )

        total_generation = time.perf_counter() - generation_start
        if first_token_time:
            decode_duration = total_generation - (first_token_time - generation_start)
            decode_tps = tokens_generated / decode_duration if decode_duration > 0 else None
            if decode_tps:
                self.decode_tps_history.append(decode_tps)
            logger.info(
                "Generation completed in %.3fs (%d tokens @ %.2f tok/s)",
                decode_duration,
                tokens_generated,
                decode_tps or -1.0,
            )

        yield GenerationProgress(
            stage="done",
            detail="Generation complete.",
            retrieval_seconds=retrieval_elapsed,
            prefill_seconds=(first_token_time - generation_start) if first_token_time else None,
            prompt_tokens=prompt_tokens,
            decode_tokens=tokens_generated,
            tokens_per_second=(self.decode_tps_history[-1] if self.decode_tps_history else None),
            eta_seconds=0.0,
            partial_answer=partial_answer.strip(),
        )

    def query(self, question: str, top_k: int = 4) -> str:
        """Generate a full response without streaming, preserving compatibility."""

        final_answer = ""
        for update in self.stream_query(question, top_k=top_k):
            final_answer = update.partial_answer or final_answer
        logger.debug("Non-streaming query produced %d characters", len(final_answer))
        return final_answer


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


def stream_query_rag(question: str, max_tokens: int = 512) -> Generator[GenerationProgress, None, None]:
    engine, error_msg = _safe_get_engine()
    if error_msg:
        yield GenerationProgress(stage="error", detail=error_msg)
        return

    yield from engine.stream_query(question, max_tokens=max_tokens)


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
    except ModelValidationError as exc:
        logger.exception(
            "RAG engine initialization failed because the GGUF file is invalid: %s",
            exc.detail,
        )
        return None, (
            "The language model file is present but invalid. Re-download the GGUF to "
            f"{MODEL_PATH} and try again. Details: {exc.detail}"
        )
    except Exception:
        logger.exception("Unexpected failure while preparing the RAG engine")
        return None, "The RAG engine is unavailable due to an unexpected error. Check logs for details."

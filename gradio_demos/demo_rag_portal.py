"""Gradio-based retrieval portal for multi-format document ingestion and Q&A.

This utility provides a compact, Microsoft-inspired interface for uploading PDF,
Word, text, and Markdown files into a local Chroma vector store. Users can view
and manage the repository (including deletions) and ask retrieval-augmented
questions over the ingested content. Logging is verbose by default to simplify
troubleshooting and align with international programming documentation
standards.
"""

from __future__ import annotations

import argparse
import logging
import mimetypes
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

os.environ["GRADIO_ANALYTICS_ENABLED"] = "false"

import chromadb
from chromadb.utils import embedding_functions
import docx
import gradio as gr
import pypdfium2

# ---------------------------------------------------------------------------
# Workaround for Gradio/gradio_client boolean-schema bugs in API info parsing
# ---------------------------------------------------------------------------
try:
    from gradio_client import utils as grc_utils  # type: ignore[attr-defined]

    _orig_get_type = getattr(grc_utils, "get_type", None)
    _orig_inner_json_schema_to_python_type = getattr(
        grc_utils, "_json_schema_to_python_type", None
    )

    if _orig_get_type is not None:

        def _safe_get_type(schema):
            """
            Some Gradio versions pass bare booleans into get_type(), which used to
            cause:
                TypeError: argument of type 'bool' is not iterable
            when get_type() checked for keys like "const" in the schema.
            Treat True/False schemas as a generic "Any" type for API-info purposes.
            """
            if isinstance(schema, bool):
                return "Any"
            return _orig_get_type(schema)

        grc_utils.get_type = _safe_get_type  # type: ignore[assignment]

    if _orig_inner_json_schema_to_python_type is not None:

        def _safe_inner_json_schema_to_python_type(schema, defs=None):
            """
            Inner converter used by json_schema_to_python_type. It can also receive
            bare booleans via $ref/$defs plumbing. When that happens, short-circuit
            to a generic "Any" type instead of raising APIInfoParseError.
            """
            if isinstance(schema, bool):
                return "Any"
            return _orig_inner_json_schema_to_python_type(schema, defs)

        grc_utils._json_schema_to_python_type = _safe_inner_json_schema_to_python_type  # type: ignore[assignment]

except Exception as patch_exc:  # pragma: no cover - defensive logging
    logging.getLogger(__name__).warning(
        "Failed to patch gradio_client schema utils: %s", patch_exc
    )
# ---------------------------------------------------------------------------

try:
    import sentence_transformers  # noqa: F401
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "sentence-transformers is required. Install with `pip install -r requirements.txt` "
        "or `pip install sentence-transformers`."
    ) from exc

LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
DEFAULT_DB_PATH = Path("data/rag_portal/chroma")
DEFAULT_UPLOAD_PATH = Path("data/rag_portal/uploads")
DEFAULT_COLLECTION_NAME = "rag_portal_documents"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 7863


@dataclass
class RAGConfig:
    """Runtime configuration for the retrieval portal."""

    db_path: Path = DEFAULT_DB_PATH
    upload_path: Path = DEFAULT_UPLOAD_PATH
    collection_name: str = DEFAULT_COLLECTION_NAME
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    log_level: str = "DEBUG"


class DocumentLoader:
    """Load supported file formats into plain text for downstream chunking."""

    SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md", ".markdown"}

    def __init__(self, upload_path: Path) -> None:
        self.upload_path = upload_path
        self.logger = logging.getLogger(self.__class__.__name__)
        self.upload_path.mkdir(parents=True, exist_ok=True)
        self.logger.debug("Initialized DocumentLoader | path=%s", self.upload_path)

    def save_upload(self, file_path: Path) -> Path:
        """Persist an uploaded file into the managed repository."""

        destination = self.upload_path / file_path.name
        self.logger.debug("Saving upload | source=%s | destination=%s", file_path, destination)
        shutil.copy(file_path, destination)
        return destination

    def load_text(self, file_path: Path) -> str:
        """Dispatch to the appropriate loader based on file extension."""

        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return self._load_pdf(file_path)
        if suffix == ".docx":
            return self._load_docx(file_path)
        if suffix in {".txt", ".md", ".markdown"}:
            return self._load_plain_text(file_path)

        raise ValueError(f"Unsupported file type: {suffix}")

    def _load_pdf(self, file_path: Path) -> str:
        """Extract text from a PDF using PyPDFium for reliability."""

        self.logger.debug("Loading PDF file: %s", file_path)
        text_parts: List[str] = []
        with pypdfium2.PdfDocument(file_path) as pdf:
            for page_index, page in enumerate(pdf, start=1):
                textpage = page.get_textpage()
                extracted = textpage.get_text_range()
                self.logger.debug(
                    "Extracted PDF page | file=%s | page=%d | chars=%d",
                    file_path,
                    page_index,
                    len(extracted),
                )
                text_parts.append(extracted)
        return "\n".join(text_parts)

    def _load_docx(self, file_path: Path) -> str:
        """Extract text content from Microsoft Word ``.docx`` files."""

        self.logger.debug("Loading DOCX file: %s", file_path)
        document = docx.Document(file_path)
        paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)

    def _load_plain_text(self, file_path: Path) -> str:
        """Read plaintext or Markdown files with UTF-8 decoding."""

        self.logger.debug("Loading text/markdown file: %s", file_path)
        return file_path.read_text(encoding="utf-8")


class VectorStore:
    """Wrapper around a persistent Chroma collection with embeddings."""

    def __init__(self, config: RAGConfig) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.config.db_path.mkdir(parents=True, exist_ok=True)
        self.logger.debug("Initializing Chroma client | path=%s", self.config.db_path)
        self.client = chromadb.PersistentClient(path=str(self.config.db_path))
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.config.embedding_model
        )
        self.collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            embedding_function=embedding_fn,
            metadata={"description": "RAG portal document store"},
        )
        self.logger.debug(
            "Vector store ready | collection=%s | model=%s",
            self.config.collection_name,
            self.config.embedding_model,
        )

    def add_document(self, source_path: Path, text: str) -> int:
        """Chunk and insert a document into the collection."""

        chunks = self._chunk_text(text)
        source_id = source_path.name
        metadata_list = [
            {"source_file": source_id, "chunk_index": idx, "source_path": str(source_path)}
            for idx in range(len(chunks))
        ]
        ids = [f"{source_id}-{uuid.uuid4()}" for _ in chunks]
        self.logger.info(
            "Adding document to vector store | file=%s | chunks=%d", source_id, len(chunks)
        )
        self.collection.add(ids=ids, documents=chunks, metadatas=metadata_list)
        return len(chunks)

    def delete_by_source(self, source_files: Sequence[str]) -> None:
        """Remove all vectors associated with the provided source files."""

        for source in source_files:
            self.logger.info("Deleting vectors for source file: %s", source)
            self.collection.delete(where={"source_file": source})

    def query(self, message: str, top_k: int = 4) -> list[tuple[str, dict]]:
        """Retrieve the most relevant chunks for the provided query."""

        self.logger.debug("Running similarity search | query_length=%d", len(message))
        results = self.collection.query(query_texts=[message], n_results=top_k)
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        return list(zip(documents, metadatas))

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 80) -> List[str]:
        """Split raw text into overlapping windows for embedding."""

        normalized = " ".join(text.split())
        chunks: List[str] = []
        start = 0
        while start < len(normalized):
            end = start + chunk_size
            chunks.append(normalized[start:end])
            start = end - overlap
        return chunks


class RAGPortal:
    """Coordinate document ingestion, management, and retrieval."""

    def __init__(self, config: RAGConfig) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.loader = DocumentLoader(config.upload_path)
        self.store = VectorStore(config)
        self.logger.debug("RAGPortal initialized | upload_path=%s", config.upload_path)

    def ingest_files(self, file_paths: list[Path]) -> str:
        """Process uploaded files into the vector store."""

        if not file_paths:
            return "No files were provided."

        status_lines: List[str] = []
        for file_path in file_paths:
            try:
                normalized = Path(getattr(file_path, "name", file_path))
                saved_path = self.loader.save_upload(normalized)
                text = self.loader.load_text(saved_path)
                chunks = self.store.add_document(saved_path, text)
                status_lines.append(f"{saved_path.name}: indexed {chunks} chunks")
                self.logger.info("Ingestion complete | file=%s | chunks=%d", saved_path, chunks)
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.exception("Failed to ingest file: %s", file_path)
                status_lines.append(f"{file_path}: failed -> {exc}")
        return "\n".join(status_lines)

    def repository_table(self) -> list[list[str | float]]:
        """Return a structured view of files stored in the upload directory."""

        records: list[list[str | float]] = []
        for file_path in sorted(self.config.upload_path.glob("*")):
            stats = file_path.stat()
            records.append(
                [
                    file_path.name,
                    (mimetypes.guess_type(file_path.name)[0] or "unknown"),
                    round(stats.st_size / 1024, 2),
                    datetime.fromtimestamp(stats.st_mtime).isoformat(timespec="seconds"),
                ]
            )
        self.logger.debug("Repository table generated | files=%d", len(records))
        return records

    def delete_files(self, file_names: list[str]) -> str:
        """Delete files and associated vectors."""

        if not file_names:
            return "No files selected for deletion."

        removed: list[str] = []
        for name in file_names:
            target = self.config.upload_path / name
            if target.exists():
                target.unlink()
                removed.append(name)
                self.logger.info("Removed file from repository: %s", name)
            else:
                self.logger.warning("Attempted to remove missing file: %s", name)
        if removed:
            self.store.delete_by_source(removed)
        return f"Removed: {', '.join(removed)}" if removed else "No files removed."

    def answer_question(self, message: str, history: list[list[str]]) -> str:
        """Generate a retrieval-augmented response using stored documents."""

        if not message.strip():
            return "Please enter a question to search the knowledge base."

        results = self.store.query(message)
        if not results:
            return "No documents are available yet. Please upload content first."

        response_lines = ["Top matches:"]
        for idx, (doc, meta) in enumerate(results, start=1):
            snippet = doc[:240].strip()
            response_lines.append(
                f"{idx}. {meta.get('source_file', 'unknown')} "
                f"(chunk {meta.get('chunk_index', '?')}): {snippet}"
            )
        response_lines.append(
            "\nFor refined answers, continue asking follow-up questions after adding more documents."
        )
        self.logger.debug(
            "RAG response prepared | query_length=%d | hits=%d", len(message), len(results)
        )
        return "\n".join(response_lines)


# Interface construction ----------------------------------------------------


def configure_logging(log_level: str) -> None:
    """Initialize console and file logging with verbose formatting."""

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    formatter = logging.Formatter(LOG_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    log_path = Path("data/logs/rag_portal.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    root_logger.debug(
        "Logging initialized | level=%s | file=%s | handlers=%d",
        log_level,
        log_path,
        len(root_logger.handlers),
    )


def build_interface(portal: RAGPortal) -> gr.Blocks:
    """Assemble the Gradio layout with a landing menu and a dedicated
    document management page that looks like a CRUD database interface.
    """

    css = """
    .portal-card {
        border: 1px solid #d6d6d6;
        border-radius: 10px;
        padding: 12px;
        background: #f7f8fa;
    }
    .compact-button {min-width: 160px;}
    .portal-header {font-size: 20px; font-weight: 600; margin-bottom: 8px; color: #1f3b73;}
    .app-menu {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
        padding: 8px 12px;
        border-radius: 10px;
        background: #e9edf5;
        border: 1px solid #d0d5e5;
    }
    .menu-title {
        font-size: 18px;
        font-weight: 600;
        color: #1f3b73;
    }
    .menu-right {
        display: flex;
        gap: 8px;
        align-items: center;
    }
    .menu-label {
        font-size: 13px;
        font-weight: 500;
        color: #4b5563;
    }
    """

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"), css=css) as demo:
        # Top-level app title + description
        gr.Markdown(
            "## Document Discovery Portal\n"
            "Use the menu to switch between **Q&A** and **Document Management** views.",
            elem_classes=["portal-header"],
        )

        # Menu bar controlling which "page" is visible
        with gr.Row(elem_classes=["app-menu"]):
            with gr.Column(scale=3):
                gr.Markdown("**Library Console**", elem_classes=["menu-title"])
            with gr.Column(scale=1):
                with gr.Row(elem_classes=["menu-right"]):
                    gr.Markdown("**View:**", elem_classes=["menu-label"])
                    view_selector = gr.Radio(
                        choices=["Q&A", "Document Management"],
                        value="Q&A",
                        label="",
                        interactive=True,
                    )

        # ----------------------
        # Q&A LANDING PAGE VIEW
        # ----------------------
        with gr.Column(visible=True) as qa_view:
            gr.Markdown(
                "### Ask the Library\n"
                "Pose questions and receive retrieval-augmented responses from your indexed documents."
            )

            gr.ChatInterface(
                fn=portal.answer_question,
                title="Retrieval Chat",
                description="Questions are answered using the most relevant document snippets.",
                chatbot=gr.Chatbot(height=280, bubble_full_width=True),
                textbox=gr.Textbox(
                    placeholder="Ask about your uploaded documents",
                    lines=2,
                ),
                submit_btn="Search",
                analytics_enabled=False,
            )

        # ----------------------------------
        # DOCUMENT MANAGEMENT / CRUD VIEW
        # ----------------------------------
        with gr.Column(visible=False) as doc_view:
            gr.Markdown(
                "### Document Management\n"
                "Upload, inspect, and curate your repository like a CRUD database interface."
            )

            with gr.Row(equal_height=True):
                # LEFT: "Create / Ingest" panel
                with gr.Column(scale=2, elem_classes=["portal-card"]):
                    gr.Markdown("#### Create & Index Records")
                    upload = gr.File(
                        label="Drop PDF, DOCX, TXT, or Markdown files",
                        file_count="multiple",
                        file_types=[".pdf", ".docx", ".txt", ".md", ".markdown"],
                    )

                    with gr.Row():
                        ingest_button = gr.Button(
                            "Ingest Files (Create)",
                            variant="primary",
                            elem_classes=["compact-button"],
                        )
                        clear_upload_button = gr.Button(
                            "Clear Selection",
                            elem_classes=["compact-button"],
                        )

                    ingest_status = gr.Textbox(
                        label="Ingestion log (Create operations)",
                        lines=4,
                    )

                    ingest_button.click(
                        fn=lambda files: portal.ingest_files(files or []),
                        inputs=upload,
                        outputs=ingest_status,
                    )
                    clear_upload_button.click(
                        fn=lambda: None,
                        inputs=None,
                        outputs=upload,
                    )

                # RIGHT: "Read / Update / Delete" panel
                with gr.Column(scale=2, elem_classes=["portal-card"]):
                    gr.Markdown("#### Repository Records (Read / Delete)")

                    repo_table = gr.Dataframe(
                        headers=["Name", "Type", "Size (KB)", "Updated"],
                        datatype=["str", "str", "number", "str"],
                        interactive=False,
                        wrap=True,
                        label="Current Records",
                    )

                    with gr.Row():
                        refresh_btn = gr.Button(
                            "Refresh Records",
                            elem_classes=["compact-button"],
                        )
                        # Placeholder for "Update" in a CRUD-style toolbar (no-op currently)
                        dummy_update_btn = gr.Button(
                            "Update (N/A)",
                            interactive=False,
                            elem_classes=["compact-button"],
                        )

                    delete_dropdown = gr.Dropdown(
                        label="Select records to delete",
                        multiselect=True,
                        choices=[],
                    )

                    delete_btn = gr.Button(
                        "Delete Selected",
                        variant="stop",
                        elem_classes=["compact-button"],
                    )
                    delete_status = gr.Textbox(
                        label="Repository updates (Delete operations)",
                        lines=2,
                    )

                    # helpers for table + dropdown
                    def _build_repo_state():
                        table = portal.repository_table()
                        names = [row[0] for row in table]
                        dropdown_update = gr.update(choices=names, value=[])
                        return table, dropdown_update

                    refresh_btn.click(
                        fn=_build_repo_state,
                        inputs=None,
                        outputs=[repo_table, delete_dropdown],
                    )

                    def delete_and_refresh(selected: list[str]):
                        status = portal.delete_files(selected or [])
                        table, dropdown_update = _build_repo_state()
                        return status, table, dropdown_update

                    delete_btn.click(
                        fn=delete_and_refresh,
                        inputs=delete_dropdown,
                        outputs=[delete_status, repo_table, delete_dropdown],
                    )

        # -----------------------
        # MENU VIEW SWITCH LOGIC
        # -----------------------

        def switch_view(choice: str):
            """Toggle which page is visible based on the menu selection."""
            if choice == "Q&A":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)

        view_selector.change(
            fn=switch_view,
            inputs=view_selector,
            outputs=[qa_view, doc_view],
        )

        # Initial repository load for the Document Management view
        demo.load(
            fn=_build_repo_state,
            inputs=None,
            outputs=[repo_table, delete_dropdown],
        )

    return demo


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse CLI options for launching the RAG portal."""

    parser = argparse.ArgumentParser(
        description="Run a Gradio interface for uploading documents and querying them via retrieval.",
    )
    parser.add_argument(
        "--host",
        dest="host",
        default=DEFAULT_HOST,
        help="Server host (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        dest="port",
        type=int,
        default=DEFAULT_PORT,
        help="Server port (default: 7863).",
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        default="DEBUG",
        help="Logging level; DEBUG recommended for verbose diagnostics.",
    )
    parser.add_argument(
        "--embedding-model",
        dest="embedding_model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="SentenceTransformer model used for vectorization.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    """Entrypoint for the RAG document portal."""

    args = parse_args(argv)
    configure_logging(args.log_level)
    logger = logging.getLogger("rag_portal_main")
    logger.info("Starting RAG portal | host=%s | port=%s", args.host, args.port)

    config = RAGConfig(
        host=args.host,
        port=args.port,
        embedding_model=args.embedding_model,
    )
    portal = RAGPortal(config)

    interface = build_interface(portal)
    interface.queue().launch(
        server_name=config.host,
        server_port=config.port,
        share=False,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
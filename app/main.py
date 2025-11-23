"""Gradio application exposing admin and user interfaces.

This module builds a responsive, role-aware Gradio interface styled with an
Apple-inspired aesthetic. It embraces verbose logging and comprehensive
documentation to comply with international programming standards while making
it easy to debug user journeys through the RAG workspace. Distinct landing
experiences exist for general users (chat-focused) and administrators (document
operations), with frictionless transitions between login, workspace, and logout
states.
"""

from __future__ import annotations

import inspect
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import gradio as gr
import gradio.blocks as gr_blocks
import gradio.routes as gr_routes
import gradio_client.utils as gr_client_utils

from app.auth import authenticate
from app.config import PDF_DIR, SHARE_INTERFACE
from app.rag_chain import delete_document, get_documents_list, ingest_pdfs, query_rag

logger = logging.getLogger(__name__)


@dataclass
class SessionRecord:
    """Represents a conversational session for a specific user and role."""

    session_id: str
    title: str
    history: list[tuple[str, str]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def as_choice(self) -> tuple[str, str]:
        """Return a Gradio-friendly ``(label, value)`` tuple for dropdowns."""

        timestamp = self.updated_at.strftime("%b %d • %H:%M")
        label = f"{self.title} · {timestamp}"
        return label, self.session_id


SESSION_STORE: Dict[str, Dict[str, SessionRecord]] = {}


def _session_key(role: str | None, username: str | None) -> str:
    """Build a consistent key for storing sessions per user and role."""

    safe_role = (role or "user").lower()
    safe_user = (username or safe_role).strip() or safe_role
    key = f"{safe_role}:{safe_user}"
    logger.debug("Computed session bucket key: %s", key)
    return key


def _get_session_bucket(role: str, username: str) -> Dict[str, SessionRecord]:
    """Return the session bucket for the given role/user, creating it if missing."""

    key = _session_key(role, username)
    if key not in SESSION_STORE:
        logger.info("Initializing session bucket for %s", key)
        SESSION_STORE[key] = {}
    return SESSION_STORE[key]


def _format_session_title(bucket: Dict[str, SessionRecord]) -> str:
    """Generate a human-readable session title based on bucket size."""

    next_index = len(bucket) + 1
    title = f"Conversation {next_index}"
    logger.debug("Generated session title: %s", title)
    return title


def _ensure_default_session(role: str, username: str) -> SessionRecord:
    """Ensure at least one session exists for the user, returning the active record."""

    bucket = _get_session_bucket(role, username)
    if bucket:
        logger.debug("Session bucket already populated for %s", _session_key(role, username))
        return next(iter(bucket.values()))

    session_id = str(uuid.uuid4())
    record = SessionRecord(session_id=session_id, title=_format_session_title(bucket))
    bucket[session_id] = record
    logger.info("Created default session %s for %s", session_id, _session_key(role, username))
    return record


def _format_session_meta(record: SessionRecord) -> str:
    """Render a compact summary for the active session."""

    duration = record.updated_at - record.created_at
    return (
        f"**Active session:** {record.title}\n"
        f"Created: {record.created_at:%b %d %H:%M UTC} · Updated: {record.updated_at:%b %d %H:%M UTC}\n"
        f"Turns: {len(record.history)} · Lifespan: {duration.total_seconds():.0f}s"
    )


def _list_session_choices(role: str, username: str) -> list[tuple[str, str]]:
    """Return display choices for all sessions scoped to the provided user."""

    bucket = _get_session_bucket(role, username)
    choices = sorted(bucket.values(), key=lambda rec: rec.updated_at, reverse=True)
    logger.debug("Listing %d session choices for %s", len(choices), _session_key(role, username))
    return [record.as_choice() for record in choices]


def _persist_history(session_id: str, role: str, username: str, history: list[tuple[str, str]]):
    """Persist chat history to the in-memory store with verbose logging."""

    bucket = _get_session_bucket(role, username)
    if session_id not in bucket:
        logger.warning("Session %s missing for %s; creating on-the-fly", session_id, _session_key(role, username))
        bucket[session_id] = SessionRecord(session_id=session_id, title=_format_session_title(bucket))

    bucket[session_id].history = history
    bucket[session_id].updated_at = datetime.utcnow()
    logger.info(
        "Persisted %d turns to session %s for %s", len(history), session_id, _session_key(role, username)
    )


def _load_session(role: str, username: str, session_id: str | None) -> SessionRecord:
    """Load a session record, falling back to the default if the ID is missing."""

    bucket = _get_session_bucket(role, username)
    if session_id and session_id in bucket:
        logger.debug(
            "Loaded requested session %s for %s", session_id, _session_key(role, username)
        )
        return bucket[session_id]

    logger.info("Requested session not found; returning default for %s", _session_key(role, username))
    return _ensure_default_session(role, username)


def _delete_session(role: str, username: str, session_id: str | None) -> SessionRecord:
    """Delete a session and return the next available record."""

    bucket = _get_session_bucket(role, username)
    if session_id and session_id in bucket:
        bucket.pop(session_id)
        logger.info("Deleted session %s for %s", session_id, _session_key(role, username))
    elif session_id:
        logger.warning(
            "Attempted to delete nonexistent session %s for %s", session_id, _session_key(role, username)
        )

    return _ensure_default_session(role, username)


def login(
    username: str,
    password: str,
    role: str,
) -> Tuple[gr.update, gr.update, str, str, str]:
    """Authenticate user and toggle interface visibility.

    The function preserves explicit logging to trace authentication outcomes and
    primes downstream components with the detected role and display name so that
    both user and administrator dashboards render the correct controls.
    """

    if authenticate(username, password, role):
        safe_username = username or role.title()
        logger.info("User %s logged in as %s", safe_username, role)
        role_banner = f"Logged in as {role.capitalize()} — welcome, {safe_username}."
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            role_banner,
            role,
            safe_username,
        )

    logger.warning("Failed login attempt for user %s", username or "<blank>")
    return gr.update(visible=True), gr.update(visible=False), "Wrong credentials!", "", ""


def _resolve_uploaded_file(file: gr.File | str | Path) -> tuple[Path, Path]:
    """Resolve a Gradio uploaded file to source and destination paths.

    Gradio's ``File`` component may deliver several object shapes across
    versions (e.g., ``NamedString`` wrappers or temporary file handles).
    This helper inspects common attributes to locate the on-disk temporary
    file while preserving the original filename for storage. Verbose logging
    documents each decision branch so operators can trace upload handling in
    production environments that require auditable behavior under
    international programming standards.
    """

    candidate_sources = []

    # ``name`` is typically the temporary file path for NamedString wrappers, but
    # can also be just the display name. We still log it to aid debugging.
    if getattr(file, "name", None):
        candidate_sources.append(Path(file.name))

    # ``path`` is present on some file-like objects returned by Gradio.
    if getattr(file, "path", None):
        candidate_sources.append(Path(file.path))

    # ``value`` is used by certain ``NamedString`` wrappers to hold the temp path.
    if getattr(file, "value", None):
        candidate_sources.append(Path(file.value))

    # If the object itself is path-like, use it as a final fallback.
    if isinstance(file, (str, Path)):
        candidate_sources.append(Path(file))

    for source in candidate_sources:
        logger.debug("Inspecting candidate upload source: %s", source)
        if source.exists():
            display_name = getattr(file, "orig_name", getattr(file, "name", source.name))
            sanitized_name = Path(display_name).name
            destination = PDF_DIR / sanitized_name
            logger.debug(
                "Resolved upload source=%s destination=%s (display=%s)",
                source,
                destination,
                sanitized_name,
            )
            return source, destination

    logger.error("Upload resolution failed; evaluated sources: %s", candidate_sources)
    raise FileNotFoundError(f"Unable to locate uploaded file for object {file!r}")


def upload_and_index(files: list[gr.File | str | Path] | None) -> str:
    """Persist uploaded PDFs and trigger re-indexing with resilient handling."""

    if not files:
        logger.warning("Upload invoked without files; returning guidance message")
        return "No files uploaded"

    failures: list[str] = []

    for file in files:
        try:
            source, destination = _resolve_uploaded_file(file)
            destination.write_bytes(Path(source).read_bytes())
            logger.info("Saved PDF upload from %s to %s", source, destination)
        except Exception as exc:  # noqa: BLE001 - intentional breadth for resilience
            logger.exception("Failed to persist uploaded file %r", file)
            failures.append(str(exc))

    if failures:
        failure_msg = "\n".join(failures)
        logger.error("One or more uploads failed: %s", failure_msg)
        return f"Upload errors encountered:\n{failure_msg}"

    return ingest_pdfs()


def _render_library_overview(documents: list[str]) -> str:
    """Render a simple, human-readable library overview for administrators."""

    if not documents:
        return "No documents ingested yet. Upload PDFs to build your knowledge base."

    entries = "\n".join([f"• {name}" for name in documents])
    return f"**{len(documents)} active documents**\n\n{entries}"


def show_admin_controls(role_str: str):
    """Reveal admin-only widgets when applicable.

    This function centralizes the toggling logic for administrator-specific
    controls. It also refreshes the document library so the admin landing page
    immediately reflects the current RAG corpus.
    """

    is_admin = "admin" in (role_str or "").lower()
    documents = refresh_documents() if is_admin else []
    selected_doc = documents[0] if documents else None
    logger.debug("Setting admin control visibility: %s", is_admin)
    overview_text = _render_library_overview(documents)
    return [
        gr.update(visible=is_admin),
        gr.update(visible=is_admin),
        gr.update(visible=is_admin),
        gr.update(visible=is_admin, choices=documents, value=selected_doc),
        gr.update(visible=is_admin),
        gr.update(visible=is_admin, value=overview_text),
    ]


def refresh_documents() -> list[str]:
    """Refresh the dropdown listing available documents."""

    docs = get_documents_list()
    logger.info("Document list refreshed with %d entries", len(docs))
    return docs


def refresh_library(status: str | None = None) -> tuple[gr.update, str]:
    """Return an updated document dropdown alongside a formatted overview.

    The function is designed for Gradio event callbacks so that ingest, delete,
    and manual refresh actions all hydrate the "Document Studio" tab with a
    consistent snapshot of the active PDFs. It optionally prepends a
    human-readable status to the overview when ``status`` is supplied.
    """

    docs = refresh_documents()
    default_choice = docs[0] if docs else None
    overview = _render_library_overview(docs)
    if status:
        overview = f"{status}\n\n{overview}"

    logger.debug(
        "Library state prepared with %d documents (default=%s)", len(docs), default_choice
    )
    return gr.update(choices=docs, value=default_choice), overview


def handle_delete(doc: str) -> str:
    """Remove a document and its chunks from the vector store."""

    if not doc:
        return "Select a document to delete."
    return delete_document(doc)


def delete_and_refresh(doc: str) -> tuple[gr.update, str]:
    """Delete a document and refresh the library snapshot for the UI."""

    status = handle_delete(doc)
    return refresh_library(status)


def ingest_and_refresh(files: list[gr.File | str | Path] | None) -> tuple[gr.update, str]:
    """Upload PDFs, trigger indexing, and rehydrate the document dropdown."""

    status = upload_and_index(files)
    return refresh_library(status)


def _session_selection_payload(role: str, username: str, record: SessionRecord) -> tuple:
    """Return updates for session selection UI when a session changes."""

    choices = _list_session_choices(role, username)
    meta = _format_session_meta(record)
    logger.debug(
        "Prepared session payload for %s with %d choices", _session_key(role, username), len(choices)
    )
    return (
        gr.update(choices=choices, value=record.session_id),
        record.session_id,
        gr.update(value=record.history),
        meta,
        "Response time: --",
    )


def hydrate_sessions(role: str, username: str) -> tuple:
    """Ensure sessions exist after login and populate related widgets."""

    record = _ensure_default_session(role, username)
    logger.info("Hydrating sessions for %s", _session_key(role, username))
    return _session_selection_payload(role, username, record)


def start_new_session(role: str, username: str) -> tuple:
    """Create a brand-new session, returning updates for the selection UI."""

    bucket = _get_session_bucket(role, username)
    session_id = str(uuid.uuid4())
    record = SessionRecord(session_id=session_id, title=_format_session_title(bucket))
    bucket[session_id] = record
    logger.info("Started new session %s for %s", session_id, _session_key(role, username))
    return _session_selection_payload(role, username, record)


def select_session(session_id: str, role: str, username: str) -> tuple:
    """Switch the active session and surface its history in the chat UI."""

    record = _load_session(role, username, session_id)
    logger.info("Switching to session %s for %s", record.session_id, _session_key(role, username))
    return _session_selection_payload(role, username, record)


def remove_session(session_id: str | None, role: str, username: str) -> tuple:
    """Delete the selected session and refresh selection controls."""

    record = _delete_session(role, username, session_id)
    logger.info("Session %s removed; activating %s for %s", session_id, record.session_id, _session_key(role, username))
    return _session_selection_payload(role, username, record)


def clear_session_history(session_id: str, role: str, username: str) -> tuple:
    """Clear the active session while keeping the conversation record available."""

    record = _load_session(role, username, session_id)
    logger.info(
        "Clearing history for session %s belonging to %s", record.session_id, _session_key(role, username)
    )
    _persist_history(record.session_id, role, username, [])
    refreshed = _load_session(role, username, record.session_id)
    meta = _format_session_meta(refreshed)
    return gr.update(value=[]), "Response time: --", refreshed.session_id, meta


def respond(
    message: str,
    history: list[list[str]] | list[tuple[str, str]] | None,
    role_str: str,
    session_id: str,
    username: str,
):
    """Generate a response using the RAG engine while streaming status updates."""

    sanitized = (message or "").strip()
    active_role = role_str or "user"
    active_user = username or active_role
    record = _load_session(active_role, active_user, session_id)

    if not sanitized:
        logger.warning("Empty prompt submitted for session %s", record.session_id)
        meta = _format_session_meta(record)
        return "", record.history, "Please enter a prompt to continue.", record.session_id, meta

    logger.info(
        "Received query for session %s belonging to %s", record.session_id, _session_key(active_role, active_user)
    )
    start_time = time.perf_counter()

    existing_history = [(turn[0], turn[1]) for turn in (history or [])]
    working_history: list[tuple[str, str]] = list(existing_history)
    working_history.append((sanitized, ""))
    _persist_history(record.session_id, active_role, active_user, working_history)
    meta = _format_session_meta(record)
    yield "", working_history, "Gathering response...", record.session_id, meta

    answer = query_rag(sanitized)
    elapsed = time.perf_counter() - start_time
    working_history[-1] = (sanitized, answer)
    _persist_history(record.session_id, active_role, active_user, working_history)
    meta = _format_session_meta(_load_session(active_role, active_user, record.session_id))
    logger.info(
        "Response for session %s completed in %.3f seconds", record.session_id, elapsed
    )
    return "", working_history, f"Response time: {elapsed:.2f}s", record.session_id, meta


def logout(
    role_str: str,
) -> tuple[gr.update, gr.update, gr.update, str, gr.update, gr.update, gr.update, str, str, gr.update]:
    """Return to the landing page and clear transient state."""

    logger.info("Logout requested for role: %s", role_str or "unknown")
    cleared_chat = gr.update(value=[])
    cleared_docs = gr.update(choices=[], value=None)
    cleared_sessions = gr.update(choices=[], value=None)
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(value="", visible=False),
        "",
        cleared_chat,
        cleared_docs,
        cleared_sessions,
        "",
        "Select a session to begin.",
        "Response time: --",
    )


ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def _monkeypatch_gradio_api_info() -> None:
    """Provide defensive wrappers for Gradio API info generation.

    Gradio attempts to produce OpenAPI-style metadata on startup. Certain
    component combinations can surface a ``TypeError`` in
    ``gradio_client.utils.json_schema_to_python_type`` when boolean schema
    fragments are present. This wrapper stack retains verbose logging for
    debugging while shielding the application from startup failures, which
    aligns with the user's requirement for robust, traceable execution under
    international programming documentation standards.
    """

    original_schema_to_type = gr_client_utils.json_schema_to_python_type

    def _coerce_bool_schema(fragment):
        """Normalize boolean JSON schema fragments into dictionaries.

        The Gradio helper ``json_schema_to_python_type`` expects mapping-based
        schema nodes, but boolean fragments are valid JSON Schema shortcuts. To
        keep logging verbose while avoiding repeated stack traces, this helper
        recursively converts ``True``/``False`` into descriptive dictionary
        placeholders that the downstream translator can safely consume.
        """

        if isinstance(fragment, bool):
            return {"type": "boolean" if fragment else "null", "description": "coerced bool schema"}
        if isinstance(fragment, dict):
            return {key: _coerce_bool_schema(value) for key, value in fragment.items()}
        if isinstance(fragment, list):
            return [_coerce_bool_schema(item) for item in fragment]
        return fragment

    def safe_json_schema_to_python_type(schema, defs=None):  # type: ignore[override]
        """Handle permissive JSON schema nodes without raising exceptions.

        Gradio occasionally emits boolean schema fragments (``True``/``False``)
        that violate the assumptions of ``json_schema_to_python_type``. The
        patched implementation documents the behavior explicitly, coerces
        boolean fragments into safe dictionaries, and returns a readable
        placeholder type instead of bubbling up a ``TypeError``.
        """

        logger.debug("Translating JSON schema to Python type: %s", schema)

        coerced_schema = _coerce_bool_schema(schema)
        coerced_defs = _coerce_bool_schema(defs) if defs is not None else None

        try:
            signature = inspect.signature(original_schema_to_type)
            accepts_defs = len(signature.parameters) > 1
            logger.debug(
                "Resolved json_schema_to_python_type signature with defs support: %s", accepts_defs
            )
            if accepts_defs:
                return original_schema_to_type(coerced_schema, coerced_defs)
            return original_schema_to_type(coerced_schema)
        except TypeError:
            logger.exception(
                "Encountered non-iterable schema fragment; substituting unknown type for resilience"
            )
            return "unknown"
        except Exception:
            logger.exception(
                "Unexpected failure translating JSON schema; substituting unknown type for resilience"
            )
            return "unknown"

    gr_client_utils.json_schema_to_python_type = safe_json_schema_to_python_type
    logger.debug("Applied resilient gradio_client.utils.json_schema_to_python_type wrapper")

    if not hasattr(gr_routes, "api_info"):
        logger.warning("gradio.routes.api_info is unavailable; skipping monkeypatch")
    else:
        original_api_info = gr_routes.api_info

        def safe_api_info(serialize: bool = False):  # type: ignore[override]
            """Generate route API info with verbose error handling."""

            logger.debug(
                "Generating Gradio route API info (serialize=%s) with fault tolerance", serialize
            )
            try:
                return original_api_info(serialize=serialize)
            except Exception:
                logger.exception(
                    "Failed to generate Gradio route API info; returning minimal stub for stability"
                )
                return {"named_endpoints": {}, "unnamed_endpoints": []}

        gr_routes.api_info = safe_api_info
        logger.debug("Applied defensive gradio.routes.api_info wrapper")

    if hasattr(gr_blocks.Blocks, "get_api_info"):
        original_get_api_info = gr_blocks.Blocks.get_api_info

        def safe_get_api_info(self):  # type: ignore[override]
            """Return Blocks API info with robust logging and fallbacks.

            Gradio's schema translation can occasionally throw runtime errors when
            optional schema nodes are absent. This wrapper preserves the original
            behavior when possible, while emitting verbose diagnostics and
            returning a predictable stub structure if the underlying implementation
            fails. The stub retains the fields expected by the client so that the
            UI continues to function even when schema generation is impaired.
            """

            logger.debug("Generating Gradio Blocks API info with protective wrapper")
            try:
                return original_get_api_info(self)
            except Exception:
                logger.exception(
                    "Failed to build Gradio Blocks API info; supplying fallback schema with no endpoints"
                )
                return {
                    "named_endpoints": {},
                    "unnamed_endpoints": [],
                    "dependencies": {"root": None, "modules": [], "imports": [], "targets": []},
                }

        gr_blocks.Blocks.get_api_info = safe_get_api_info
        logger.debug("Patched gradio.blocks.Blocks.get_api_info with protective wrapper")
    else:
        logger.warning("gradio.blocks.Blocks.get_api_info is unavailable; no patch applied")


def build_ui() -> gr.Blocks:
    """Build the Gradio Blocks interface."""

    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="IBM MQ RAG Sandbox",
        css=(ASSETS_DIR / "custom.css").read_text() if (ASSETS_DIR / "custom.css").exists() else None,
    ) as demo:
        role_state = gr.State("")
        user_state = gr.State("")
        session_state = gr.State("")

        gr.Markdown(
            """
            # IBM MQ Knowledge Base
            Experience a streamlined, Apple-inspired interface for chatting with and curating your localized knowledge base.
            """,
            elem_classes=["page-title"],
        )

        with gr.Column(visible=True, elem_classes=["card", "login-card"]) as login_box:
            gr.Markdown(
                """
                ### Sign in
                Choose your role to access either the chat experience (user) or the document operations hub (admin).
                """,
                elem_classes=["card-title"],
            )
            with gr.Row(elem_classes=["login-grid"]):
                role = gr.Radio(
                    ["user", "admin"],
                    label="Login as",
                    value="user",
                    elem_classes=["pill-input"],
                )
                user = gr.Textbox(
                    label="Username",
                    placeholder="your.name",
                    elem_classes=["text-input"],
                )
                pwd = gr.Textbox(
                    label="Password",
                    type="password",
                    placeholder="••••••••",
                    elem_classes=["text-input"],
                )
            login_btn = gr.Button("Enter Workspace", elem_classes=["primary-btn", "full-width"])

        status = gr.Markdown(visible=False, elem_classes=["status-bar"])

        with gr.Column(visible=False, elem_classes=["workspace"], elem_id="workspace") as main_interface:
            with gr.Row(elem_classes=["workspace-header", "card"]):
                gr.Markdown("### Workspace", elem_classes=["card-title", "no-margin"])
                user_badge = gr.Markdown("", elem_classes=["badge"])
                logout_btn = gr.Button("Logout", elem_classes=["ghost-btn"])

            with gr.Row(elem_classes=["workspace-layout"]):
                with gr.Column(scale=3, elem_classes=["card", "sidebar"]):
                    gr.Markdown("#### Your sessions", elem_classes=["card-title", "no-margin"])
                    gr.Markdown(
                        "Manage conversations and quickly return to recent threads unique to your login. Use the controls below to open, start, or delete sessions without losing context.",
                        elem_classes=["muted"],
                    )
                    session_selector = gr.Dropdown(
                        label="Saved conversations",
                        choices=[],
                        value=None,
                        interactive=True,
                        elem_classes=["text-input", "session-dropdown"],
                    )
                    session_meta = gr.Markdown(elem_classes=["status-bar", "session-meta"], value="Select a session to begin.")
                    with gr.Row(elem_classes=["session-actions"]):
                        new_session_btn = gr.Button(
                            "New conversation",
                            elem_classes=["primary-btn", "full-width"],
                            variant="primary",
                        )
                        delete_session_btn = gr.Button(
                            "Delete session",
                            elem_classes=["danger-btn", "full-width"],
                            variant="secondary",
                        )

                with gr.Column(scale=9, elem_classes=["card", "content-shell"]):
                    with gr.Tabs(elem_classes=["tabset"]) as tabs:
                        with gr.Tab("Chat Experience", elem_id="chat-tab"):
                            gr.Markdown(
                                """
                                Engage with your IBM MQ knowledge base using a clean, ChatGPT-inspired conversational flow.
                                """,
                                elem_classes=["muted"],
                            )
                            gr.Markdown(
                                "## What can I help with?",
                                elem_classes=["chat-hero"],
                            )
                            response_timer = gr.Markdown(
                                "Response time: --",
                                elem_classes=["status-bar", "timer-bar"],
                            )
                            chatbot = gr.Chatbot(
                                height=420,
                                bubble_full_width=False,
                                layout="panel",
                                elem_classes=["chatbot", "chat-frame"],
                            )
                            with gr.Row(elem_classes=["chat-input-row"]):
                                msg = gr.Textbox(
                                    label="Ask about IBM MQ",
                                    placeholder="Ask a question or paste a log snippet...",
                                    elem_classes=["text-input", "chat-input"],
                                    scale=10,
                                )
                                clear = gr.Button("Clear", elem_classes=["ghost-btn", "icon-btn"], scale=2)

                        with gr.Tab("Document Studio", visible=False, elem_id="admin-tab") as admin_tab:
                            gr.Markdown(
                                "Administer the localized corpus: review, ingest, or prune documents powering the RAG pipeline.",
                                elem_classes=["muted"],
                            )
                            with gr.Row(elem_classes=["doc-studio-grid"]):
                                with gr.Column(scale=6, elem_classes=["card", "doc-panel"]):
                                    gr.Markdown("#### Library Health", elem_classes=["card-title", "no-margin"])
                                    library_overview = gr.Markdown(elem_classes=["status-bar", "library-overview"])
                                    doc_list = gr.Dropdown(
                                        choices=[],
                                        value=None,
                                        label="Active PDFs",
                                        info="Currently indexed sources feeding the RAG pipeline.",
                                        interactive=True,
                                        elem_classes=["text-input"],
                                    )
                                    with gr.Row():
                                        refresh_btn = gr.Button("Refresh Library", elem_classes=["ghost-btn"])
                                        delete_btn = gr.Button("Delete Selected", elem_classes=["danger-btn"])

                                with gr.Column(scale=5, elem_classes=["card", "doc-panel"]):
                                    gr.Markdown("#### Add New Sources", elem_classes=["card-title", "no-margin"])
                                    gr.Markdown(
                                        "Upload one or more PDFs to expand the knowledge base. Ingestion automatically reindexes the vector store with verbose logging for traceability.",
                                        elem_classes=["muted"],
                                    )
                                    file_upload = gr.File(
                                        label="Upload IBM MQ PDFs",
                                        file_count="multiple",
                                        file_types=[".pdf"],
                                        type="filepath",
                                        interactive=True,
                                    )
                                    index_btn = gr.Button("Ingest PDFs", elem_classes=["primary-btn", "full-width"])

            msg.submit(
                respond,
                [msg, chatbot, role_state, session_state, user_state],
                [msg, chatbot, response_timer, session_state, session_meta],
            )
            clear.click(
                clear_session_history,
                inputs=[session_state, role_state, user_state],
                outputs=[chatbot, response_timer, session_state, session_meta],
                queue=False,
            )

        login_btn.click(
            login,
            inputs=[user, pwd, role],
            outputs=[login_box, main_interface, status, role_state, user_state],
        ).then(
            show_admin_controls,
            inputs=role_state,
            outputs=[
                admin_tab,
                file_upload,
                index_btn,
                doc_list,
                delete_btn,
                library_overview,
            ],
        ).then(
            lambda r, u: gr.update(visible=True, value=f"{u} ({r})"),
            inputs=[role_state, user_state],
            outputs=user_badge,
        ).then(
            lambda _: gr.update(visible=True),
            inputs=role_state,
            outputs=status,
        ).then(
            hydrate_sessions,
            inputs=[role_state, user_state],
            outputs=[session_selector, session_state, chatbot, session_meta, response_timer],
        )

        refresh_btn.click(refresh_library, None, [doc_list, library_overview])
        index_btn.click(ingest_and_refresh, file_upload, [doc_list, library_overview])
        file_upload.change(refresh_library, None, [doc_list, library_overview])
        delete_btn.click(delete_and_refresh, doc_list, [doc_list, library_overview])
        new_session_btn.click(
            start_new_session,
            inputs=[role_state, user_state],
            outputs=[session_selector, session_state, chatbot, session_meta, response_timer],
        )
        session_selector.change(
            select_session,
            inputs=[session_selector, role_state, user_state],
            outputs=[session_selector, session_state, chatbot, session_meta, response_timer],
        )
        delete_session_btn.click(
            remove_session,
            inputs=[session_selector, role_state, user_state],
            outputs=[session_selector, session_state, chatbot, session_meta, response_timer],
        )
        logout_btn.click(
            logout,
            inputs=role_state,
            outputs=[
                login_box,
                main_interface,
                status,
                role_state,
                chatbot,
                doc_list,
                session_selector,
                session_state,
                session_meta,
                response_timer,
            ],
        )

    return demo


def main() -> None:
    """Start the Gradio application."""

    _monkeypatch_gradio_api_info()
    logger.info("Launching Gradio app on 0.0.0.0:7860")
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=SHARE_INTERFACE, show_api=False)


if __name__ == "__main__":
    main()

"""Modern Gradio UI for the mq-rag experience.

This module rebuilds the interface into a single, role-aware Blocks app that
resembles contemporary AI search tools. It preserves all backend hooks while
adding verbose logging and documentation that comply with international
programming standards. The UI uses a lightweight state machine to toggle
between login, search, and help experiences without launching multiple Gradio
apps.
"""

from __future__ import annotations

import inspect
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, TypedDict

import gradio as gr
import gradio_client.utils as client_utils

from app.auth import authenticate
from app.config import PDF_DIR, SHARE_INTERFACE
from app.rag_chain import (
    GenerationProgress,
    delete_document,
    get_documents_list,
    ingest_pdfs,
    query_rag,
    start_background_prewarm,
    stream_query_rag,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compatibility patches
# ---------------------------------------------------------------------------

# Gradio 4.44 can emit boolean JSON schemas for some dependency payloads when
# analytics metadata is present. The default ``get_api_info`` helper does not
# tolerate those values and raises ``TypeError: argument of type 'bool' is not
# iterable`` during startup, which stops the ASGI app from rendering. To keep
# the interface resilient, we wrap the original method with a defensive shim
# that logs the failure and returns an empty schema instead of bubbling the
# exception up through the request stack.
_ORIGINAL_GET_API_INFO = gr.Blocks.get_api_info
_ORIGINAL_JSON_SCHEMA_TO_PYTHON_TYPE = client_utils.json_schema_to_python_type


def _coerce_json_schema(schema: Any) -> Any:
    """Coerce JSON schema fragments into dictionaries to avoid TypeError crashes."""

    if isinstance(schema, bool):
        logger.debug("Coercing boolean JSON schema value %s to an empty object", schema)
        return {} if schema else {"not": {}}

    if isinstance(schema, dict):
        return {key: _coerce_json_schema(value) for key, value in schema.items()}

    if isinstance(schema, list):
        return [_coerce_json_schema(value) for value in schema]

    return schema


def _safe_json_schema_to_python_type(schema: Any, defs: dict | None = None) -> Any:
    """Wrap Gradio's schema converter to tolerate non-dict inputs produced upstream."""

    sanitized_schema = _coerce_json_schema(schema)
    sanitized_defs = _coerce_json_schema(defs or {}) if defs is not None else None

    try:
        return _ORIGINAL_JSON_SCHEMA_TO_PYTHON_TYPE(sanitized_schema, sanitized_defs)
    except TypeError:
        logger.debug(
            "Original json_schema_to_python_type signature does not accept defs; retrying with sanitized schema only"
        )
        try:
            return _ORIGINAL_JSON_SCHEMA_TO_PYTHON_TYPE(sanitized_schema)
        except Exception as exc:  # pragma: no cover - defensive path for future regressions
            logger.exception(
                "Fallback activated while converting JSON schema to python type; returning 'unknown'",
                exc_info=exc,
            )
            return "unknown"


if client_utils.json_schema_to_python_type is not _safe_json_schema_to_python_type:
    client_utils.json_schema_to_python_type = _safe_json_schema_to_python_type


def _safe_get_api_info(self, *args, **kwargs):
    """Safely compute API metadata, tolerating serialization edge cases."""

    try:
        return _ORIGINAL_GET_API_INFO(self, *args, **kwargs)
    except TypeError as exc:  # pragma: no cover - guard for upstream changes
        logger.exception(
            "Failed to build Gradio API schema; returning empty descriptors for stability.",
            exc_info=exc,
        )
        return {"named_endpoints": [], "unnamed_endpoints": []}


if gr.Blocks.get_api_info is not _safe_get_api_info:
    gr.Blocks.get_api_info = _safe_get_api_info


class AppState(TypedDict):
    """State container shared across Gradio callbacks.

    Using ``TypedDict`` prevents ambiguous JSON schema generation (which can
    occur when plain ``dict`` is used) and improves downstream documentation
    while keeping runtime behavior unchanged.
    """

    page: str
    role: str
    user: str
    session_id: str


# ---------------------------------------------------------------------------
# Session management helpers (preserved)
# ---------------------------------------------------------------------------


@dataclass
class SessionRecord:
    """Represents a conversational session for a specific user and role."""

    session_id: str
    title: str
    history: list[tuple[str, str]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def as_choice(self) -> tuple[str, str]:
        """Return a Gradio-friendly ``(label, value)`` tuple for selection widgets."""

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
    title = f"Session {next_index}"
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
    """Render a compact summary for the active session without rigid labels."""

    duration = record.updated_at - record.created_at
    summary = (
        f"{record.title} — Created: {record.created_at:%b %d %H:%M UTC} · "
        f"Updated: {record.updated_at:%b %d %H:%M UTC}\n"
        f"Turns: {len(record.history)} · Lifespan: {duration.total_seconds():.0f}s"
    )
    logger.debug("Formatted session metadata for %s: %s", record.session_id, summary)
    return summary


def _safe_markdown(value: str = "", **kwargs) -> gr.Markdown:
    """Create a Markdown component while removing unsupported kwargs.

    Gradio releases can differ slightly in accepted keyword arguments. Some
    deployments may still attempt to pass ``min_width`` for layout tweaking,
    which raises ``TypeError`` on versions that do not expose the parameter.
    This helper strips any unsupported arguments, logs the adjustment for
    traceability, and returns a compatible component to keep startup robust.

    Args:
        value: Initial markdown text rendered in the component.
        **kwargs: Arbitrary keyword arguments forwarded to ``gr.Markdown``.

    Returns:
        A configured ``gr.Markdown`` built with options supported by the
        active Gradio installation.
    """

    signature = inspect.signature(gr.Markdown.__init__)
    unsupported: list[str] = []

    for arg in ["min_width"]:
        if arg in kwargs and arg not in signature.parameters:
            unsupported.append(arg)
            kwargs.pop(arg)

    if unsupported:
        logger.warning(
            "Removed unsupported Markdown args %s for Gradio %s", unsupported, gr.__version__
        )

    return gr.Markdown(value, **kwargs)


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
        logger.debug("Loaded requested session %s for %s", session_id, _session_key(role, username))
        return bucket[session_id]

    logger.info("Requested session not found; returning default for %s", _session_key(role, username))
    return _ensure_default_session(role, username)


# ---------------------------------------------------------------------------
# Document helpers (preserved)
# ---------------------------------------------------------------------------


def _resolve_uploaded_file(file: gr.File | str | Path) -> tuple[Path, Path]:
    """Resolve a Gradio uploaded file to source and destination paths."""

    candidate_sources = []
    if getattr(file, "name", None):
        candidate_sources.append(Path(file.name))
    if getattr(file, "path", None):
        candidate_sources.append(Path(file.path))
    if getattr(file, "value", None):
        candidate_sources.append(Path(file.value))
    if isinstance(file, (str, Path)):
        candidate_sources.append(Path(file))

    for source in candidate_sources:
        logger.debug("Inspecting candidate upload source: %s", source)
        if source.exists():
            display_name = getattr(file, "orig_name", getattr(file, "name", source.name))
            sanitized_name = Path(display_name).name
            destination = PDF_DIR / sanitized_name
            logger.debug("Resolved upload source=%s destination=%s (display=%s)", source, destination, sanitized_name)
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


def refresh_documents() -> list[str]:
    """Refresh the dropdown listing available documents."""

    docs = get_documents_list()
    logger.info("Document list refreshed with %d entries", len(docs))
    return docs


def _build_document_rows(filter_text: str | None = None, docs: list[str] | None = None) -> list[list[str]]:
    """Construct document table rows with lightweight metadata for display."""

    documents = docs or refresh_documents()
    query = (filter_text or "").strip().lower()
    rows: list[list[str]] = []
    for name in documents:
        if query and query not in name.lower():
            continue

        path = PDF_DIR / name
        file_type = (path.suffix or ".pdf").replace(".", "").upper() or "PDF"
        try:
            stats = path.stat()
            size_kb = f"{stats.st_size / 1024:.1f} KB"
            ingested = datetime.utcfromtimestamp(stats.st_mtime).strftime("%b %d %H:%M UTC")
        except OSError:
            size_kb = "—"
            ingested = "—"

        rows.append([name, file_type, size_kb, ingested, "✖"])

    logger.debug("Prepared %d document rows after filtering for query %r", len(rows), query)
    return rows


def refresh_library(status: str | None = None, filter_text: str | None = None) -> tuple[list[list[str]], str]:
    """Return updated document table rows alongside a formatted overview."""

    docs = refresh_documents()
    overview = _render_library_overview(docs)
    if status:
        overview = f"{status}\n\n{overview}"

    logger.debug("Library state prepared with %d documents", len(docs))
    return _build_document_rows(filter_text, docs), overview


def handle_delete(doc: str) -> str:
    """Remove a document and its chunks from the vector store."""

    if not doc:
        return "Select a document to delete."
    return delete_document(doc)


def delete_and_refresh(doc: str) -> tuple[gr.update, str]:
    """Delete a document and refresh the library snapshot for the UI."""

    status = handle_delete(doc)
    rows, overview = refresh_library(status)
    return gr.update(value=rows), overview


def ingest_and_refresh(files: list[gr.File | str | Path] | None) -> tuple[gr.update, str]:
    """Upload PDFs, trigger indexing, and rehydrate the document dropdown."""

    status = upload_and_index(files)
    rows, overview = refresh_library(status)
    return gr.update(value=rows), overview


def filter_documents(query: str | None) -> tuple[gr.update, str]:
    """Filter the document grid with type-ahead search."""

    rows, overview = refresh_library(filter_text=query)
    return gr.update(value=rows), overview


def handle_document_action(
    selected: list[str] | str,
    event: gr.SelectData | None = None,
    query: str | None = None,
    state: AppState | None = None,
) -> tuple[gr.update, str, str]:
    """React to document table clicks for selection or deletion."""

    if state is None:
        logger.warning("Document action invoked without state; defaulting to empty context")
        state = {}
    role = state.get("role") or "user"
    filter_text = query or ""
    row_index, col_index = 0, 0
    if event is not None:
        try:
            row_index, col_index = event.index  # type: ignore[assignment]
        except Exception:  # noqa: BLE001
            row_index = getattr(event, "row", 0) or 0
            col_index = getattr(event, "column", 0) or 0

    docs = refresh_documents()
    rows = _build_document_rows(filter_text, docs)
    if not rows:
        logger.info("Document action invoked with no rows; returning overview only")
        return gr.update(value=rows), _render_library_overview(docs), ""

    target_index = max(0, min(row_index, len(rows) - 1))
    target_doc = rows[target_index][0]
    logger.info(
        "Document table click on row %d column %d for %s (role=%s)",
        row_index,
        col_index,
        target_doc,
        role,
    )

    if col_index == 4:
        if role != "admin":
            logger.warning("Non-admin attempted to delete document %s", target_doc)
            return gr.update(value=rows), _render_library_overview(docs) + "\n\nAdmin-only delete.", target_doc

        status = handle_delete(target_doc)
        updated_rows, overview = refresh_library(status, filter_text)
        return gr.update(value=updated_rows), overview, ""

    return gr.update(value=rows), _render_library_overview(docs), target_doc


def delete_selected_document(doc: str | None, query: str | None, state: AppState) -> tuple[gr.update, str, str]:
    """Delete the chosen document when allowed, preserving logging fidelity."""

    role = state.get("role") or "user"
    filter_text = query or ""
    target_doc = (doc or "").strip()
    logger.info("Delete button invoked for document %s by role %s", target_doc or "<empty>", role)

    if not target_doc:
        logger.warning("Delete attempted without selecting a document")
        rows, overview = refresh_library(filter_text=filter_text)
        return gr.update(value=rows), overview + "\n\nSelect a document before deleting.", target_doc

    if role != "admin":
        logger.warning("Non-admin attempted to delete document %s", target_doc)
        rows, overview = refresh_library(filter_text=filter_text)
        return gr.update(value=rows), overview + "\n\nAdmin-only delete.", target_doc

    status = handle_delete(target_doc)
    updated_rows, overview = refresh_library(status, filter_text)
    return gr.update(value=updated_rows), overview, ""


# ---------------------------------------------------------------------------
# Authentication and session hydration
# ---------------------------------------------------------------------------


def _detect_role(username: str, password: str) -> str | None:
    """Infer the user's role using the existing authentication helper."""

    if authenticate(username, password, "admin"):
        logger.info("User %s authenticated as admin", username)
        return "admin"
    if authenticate(username, password, "user"):
        logger.info("User %s authenticated as user", username)
        return "user"
    logger.warning("Authentication failed for user %s", username or "<blank>")
    return None


def hydrate_sessions(role: str, username: str) -> tuple:
    """Ensure sessions exist after login and populate related widgets."""

    record = _ensure_default_session(role, username)
    logger.info("Hydrating sessions for %s", _session_key(role, username))
    meta = _format_session_meta(record)
    return record.session_id, record.history, meta


def clear_session_history(session_id: str, state: AppState) -> tuple:
    """Clear the active session while keeping the conversation record available."""

    role = state.get("role") or "user"
    username = state.get("user") or role
    record = _load_session(role, username, session_id)
    logger.info("Clearing history for session %s belonging to %s", record.session_id, _session_key(role, username))
    _persist_history(record.session_id, role, username, [])
    refreshed = _load_session(role, username, record.session_id)
    meta = _format_session_meta(refreshed)
    return [], refreshed.session_id, meta


def respond(
    message: str,
    history: list[list[str]] | list[tuple[str, str]] | None,
    state: AppState,
    session_id: str,
    username: str,
):
    """Generate a response using the RAG engine while streaming chat updates only."""

    sanitized = (message or "").strip()
    active_role = state.get("role") or "user"
    active_user = username or active_role
    record = _load_session(active_role, active_user, session_id)

    if not sanitized:
        logger.warning("Empty prompt submitted for session %s", record.session_id)
        meta = _format_session_meta(record)
        return (
            "",
            record.history,
            record.session_id,
            meta,
            gr.update(interactive=True),
        )

    logger.info(
        "Received query for session %s belonging to %s", record.session_id, _session_key(active_role, active_user)
    )
    active_start = time.perf_counter()

    existing_history = [(turn[0], turn[1]) for turn in (history or [])]
    working_history: list[tuple[str, str]] = list(existing_history)
    working_history.append((sanitized, ""))
    _persist_history(record.session_id, active_role, active_user, working_history)

    partial_answer = ""
    logger.debug("Streaming response for session %s initiated", record.session_id)
    meta = _format_session_meta(_load_session(active_role, active_user, record.session_id))

    last_payload: tuple[tuple[tuple[str, str], ...], bool] | None = None

    def _emit_if_changed(interactive: bool):
        """Yield a UI update only when the visible payload changes."""

        nonlocal last_payload, meta
        snapshot = tuple(working_history)
        payload_key = (snapshot, interactive)

        if payload_key == last_payload:
            logger.debug(
                "Suppressing duplicate UI emission for session %s; state unchanged", record.session_id
            )
            return

        last_payload = payload_key
        logger.debug(
            "Emitting UI update for session %s with %d conversation turns", record.session_id, len(snapshot)
        )
        yield (
            "",
            working_history,
            record.session_id,
            meta,
            gr.update(interactive=interactive),
        )

    yield from _emit_if_changed(interactive=False)

    for progress in stream_query_rag(sanitized):
        if progress.partial_answer:
            partial_answer = progress.partial_answer
        working_history[-1] = (sanitized, partial_answer)
        _persist_history(record.session_id, active_role, active_user, working_history)
        meta = _format_session_meta(_load_session(active_role, active_user, record.session_id))
        if progress.stage == "error":
            error_text = progress.detail or "The RAG engine is unavailable."
            working_history[-1] = (sanitized, error_text)
            logger.error("Progress stream reported error for session %s: %s", record.session_id, error_text)
            _persist_history(record.session_id, active_role, active_user, working_history)
            yield from _emit_if_changed(interactive=True)
            return

        logger.debug(
            "Streaming update for session %s: stage=%s", record.session_id, progress.stage
        )
        yield from _emit_if_changed(interactive=False)

    elapsed = time.perf_counter() - active_start
    working_history[-1] = (sanitized, partial_answer)
    _persist_history(record.session_id, active_role, active_user, working_history)
    meta = _format_session_meta(_load_session(active_role, active_user, record.session_id))
    logger.info("Response for session %s completed in %.3f seconds", record.session_id, elapsed)
    yield from _emit_if_changed(interactive=True)


def clear_query_input(current_value: str) -> str:
    """Reset the hero query textbox while providing verbose audit logs."""

    length = len(current_value or "")
    logger.info("Clear button activated for hero query input; previous length=%d", length)
    return ""


# ---------------------------------------------------------------------------
# Page state management
# ---------------------------------------------------------------------------


def _set_page(state: AppState, page: str) -> AppState:
    """Return a new state dict with the page updated."""

    new_state: AppState = {**state, "page": page}
    logger.debug("Page transition to %s with state %s", page, new_state)
    return new_state


def _initial_state() -> AppState:
    """Default app state used at startup and after logout."""

    return {"page": "login", "role": "", "user": "", "session_id": ""}


def announce_login_attempt() -> tuple[dict[str, Any], dict[str, Any]]:
    """Surface an immediate status update while authentication runs.

    Gradio executes chained callbacks sequentially; the first callback
    returns quickly to update the UI so users see progress feedback while
    the second callback performs real authentication. This improves clarity
    when the server performs slower checks or model warmup.

    The return type intentionally uses dictionaries instead of
    ``gr.Update`` to remain compatible with Gradio releases that removed the
    ``Update`` alias, avoiding attribute errors during type resolution while
    still conveying the update payload shape.
    """

    logger.info("Login submission received; rendering in-progress indicator")
    return (
        gr.update(value="Authenticating…", visible=True),
        gr.update(value="", visible=False),
    )


def attempt_login(username: str, password: str, state: AppState) -> tuple:
    """Authenticate and transition to the search page when successful."""

    role = _detect_role(username, password)
    if not role:
        logger.warning("Login failed; keeping user on login page")
        return (
            state,
            gr.update(value="Authentication failed.", visible=True),
            gr.update(value="Invalid credentials. Please try again.", visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value=""),
            gr.update(value=""),
            "",
            [],
            "Session inactive. Log in to begin.",
            gr.update(interactive=True),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value=_build_document_rows()),
            _render_library_overview([]),
            "Browse and manage ingested documents.",
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            "Admin-only.",
        )

    safe_username = username or role.title()
    new_state = {
        "page": "search",
        "role": role,
        "user": safe_username,
        "session_id": "",
    }
    sessions = hydrate_sessions(role, safe_username)
    doc_rows, doc_overview = refresh_library()
    logger.info("Login succeeded for %s; transitioning to search page", safe_username)
    return (
        new_state,
        gr.update(value="Authenticated. Redirecting…", visible=True),
        gr.update(value="", visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(value=safe_username),
        gr.update(value=role.title()),
        sessions[0],
        sessions[1],
        sessions[2],
        gr.update(interactive=True),
        gr.update(value=safe_username),
        gr.update(value=role),
        gr.update(value=doc_rows),
        doc_overview,
        "Browse and manage ingested documents.",
        gr.update(interactive=role == "admin"),
        gr.update(interactive=role == "admin"),
        gr.update(interactive=role == "admin"),
        "Full access" if role == "admin" else "Read-only for non-admins.",
    )


def logout(state: AppState) -> tuple:
    """Return to the landing page and clear transient state."""

    logger.info("Logout requested for role: %s", state.get("role") or "unknown")
    cleared_state = _initial_state()
    return (
        cleared_state,
        gr.update(value="Ready to sign in.", visible=False),
        gr.update(value="", visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(value=""),
        gr.update(value=""),
        "",
        [],
        "Session ready. Conversations persist automatically.",
        gr.update(interactive=True),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=_build_document_rows()),
        _render_library_overview([]),
        "Browse and manage ingested documents.",
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        "Admin-only.",
    )


def toggle_page(target: str, state: AppState) -> tuple:
    """Toggle between search, help, and manage docs views while remaining logged in."""

    updated = _set_page(state, target)
    return (
        updated,
        gr.update(visible=target == "login"),
        gr.update(visible=target != "login"),
        gr.update(visible=target == "help"),
        gr.update(visible=target == "search"),
        gr.update(visible=target == "manage_docs"),
    )


def configure_doc_tools(state: AppState) -> tuple:
    """Configure Manage Docs interactivity and populate the table based on role."""

    is_admin = state.get("role") == "admin"
    tooltip = "Full access" if is_admin else "Read-only for non-admins."
    logger.debug("Configuring manage docs view for %s", state.get("role") or "anonymous")
    rows, overview = refresh_library()
    hint = overview + ("\n\nAdmin-only." if not is_admin else "")
    return (
        gr.update(interactive=is_admin),
        gr.update(interactive=is_admin),
        gr.update(interactive=is_admin),
        gr.update(value=rows),
        hint,
        tooltip,
    )


def open_manage_docs(state: AppState) -> tuple:
    """Navigate to the Manage Docs view with refreshed data and gating."""

    updated_state = _set_page(state, "manage_docs")
    is_admin = state.get("role") == "admin"
    rows, overview = refresh_library()
    logger.info("Opening Manage Docs for %s", state.get("user") or "unknown user")
    return (
        updated_state,
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(value=rows),
        overview,
        "Browse and manage ingested documents.",
        gr.update(interactive=is_admin),
        gr.update(interactive=is_admin),
        gr.update(interactive=is_admin),
        "Full access" if is_admin else "Read-only for non-admins.",
    )


def return_to_search(state: AppState) -> tuple:
    """Navigate back to the primary search view while staying logged in."""

    return toggle_page("search", state)


# ---------------------------------------------------------------------------
# UI assembly
# ---------------------------------------------------------------------------


CUSTOM_CSS = """
/* Ops dashboard theme: dark, compact, and data-dense to keep workflows visible. */
body {background: #0f1117; color: #e6e9ef;}

.gradio-container {
  font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
  width: 100%;
  max-width: 1400px;
  margin: 0 auto;
  padding: 6px 10px !important;
  gap: 8px !important;
}

/* Global compaction to limit scroll and keep content above the fold. */
.gradio-container .block,
.gradio-container .gr-block,
.gradio-container .gr-form,
.gradio-container .gr-panel,
.gradio-container .gr-box,
.gradio-container .form,
.gradio-container .wrap {
  margin: 4px 0 !important;
  padding: 4px !important;
  gap: 6px !important;
}

.gradio-container .row,
.gradio-container .gr-row,
.gradio-container .column,
.gradio-container .gr-column {
  gap: 8px !important;
  margin: 2px 0 !important;
}

.gradio-container .empty,
.gradio-container div:empty {
  display: none !important;
  height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
}

.dashboard {gap: 10px !important;}
.header-bar {
  background: #141824;
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 10px;
  padding: 8px 12px !important;
  align-items: center;
}

.header-left h3 {margin: 0 !important; font-size: 18px !important; color: #e6e9ef;}
.header-right {justify-content: flex-end; gap: 10px !important;}

.badge {background: rgba(255,255,255,0.06); color: #9aa3b2; padding: 6px 10px; border-radius: 8px; font-size: 12px;}

.body-row {align-items: stretch; gap: 10px !important;}
.nav-rail {
  min-width: 240px;
  max-width: 280px;
  background: #141824;
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 10px;
  padding: 10px 10px 12px !important;
  gap: 10px !important;
}
.nav-rail .section-title {font-size: 12px; letter-spacing: 0.04em; color: #9aa3b2; margin-top: 2px !important;}
.nav-rail .nav-buttons button {width: 100%; justify-content: flex-start;}
.nav-rail .history-panel {max-height: 320px; overflow-y: auto;}

.main-area {gap: 10px !important;}
.panel {
  background: #171b27;
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 10px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.35);
  padding: 8px 10px !important;
  gap: 8px !important;
}

.kpi-strip {gap: 8px !important;}
.kpi-card {background: #141824; border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 10px 12px !important; min-height: 72px; box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);}
.kpi-card h4 {margin: 0 0 4px 0 !important; font-size: 13px !important; color: #9aa3b2;}
.kpi-card p {margin: 0 !important; font-size: 18px !important; color: #e6e9ef; font-weight: 700;}

.workflow-panel textarea,
.workflow-panel input,
.workflow-panel .gr-textbox textarea {
  min-height: 90px;
}

.hero-input input,
.hero-input textarea {
  border-radius: 10px;
  border: 1px solid rgba(255,255,255,0.08);
  background: #0f1117;
  color: #e6e9ef;
  font-size: 16px;
  padding: 8px 10px;
  margin: 0 !important;
  min-height: 70px;
}

.hero-actions {align-items: center !important; gap: 8px !important; margin: 0 !important; padding: 0 !important;}
button.primary {background: linear-gradient(135deg, #1f9dd4, #3ec7f8); color: #0f1117; border-radius: 10px; border: 1px solid rgba(255,255,255,0.08);}
button, .btn {color: #e6e9ef !important; font-weight: 600; min-height: 38px !important;}
button.ghost {background: transparent; border: 1px solid rgba(255,255,255,0.12); color: #e6e9ef; border-radius: 10px;}
button.ghost:hover {background: rgba(255,255,255,0.05); border-color: rgba(255,255,255,0.18);}
.gr-button-primary:hover, button.primary:hover {filter: brightness(1.05);}

.status {color: #9aa3b2; font-size: 13px; margin: 0 !important;}
.session-table table {width: 100%;}
.session-table td:last-child, .docs-table td:last-child {text-align: center; width: 56px;}

/* Chat transcript cleanup to remove phantom glyphs and excessive whitespace. */
.chatbot {
  background: #0f1117;
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 10px;
  padding: 6px 8px !important;
  margin-top: 2px !important;
  box-shadow: none !important;
}

#search-view .gr-chatbot,
#search-view .gr-chatbot > div {
  margin: 0 !important;
  padding: 0 !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}

#search-view .gr-chatbot .wrap {
  display: flex !important;
  flex-direction: column !important;
  gap: 6px !important;
  padding: 0 !important;
  margin: 0 !important;
  align-items: stretch !important;
  min-height: 0 !important;
  scroll-padding: 0 !important;
}

#search-view .gr-chatbot .wrap > div {margin: 0 !important; padding: 0 !important;}

#search-view .gr-chatbot .wrap > div:empty {display: none !important; min-height: 0 !important;}

#search-view .gr-chatbot .message {
  margin: 0 !important;
  padding: 10px 12px !important;
  border-radius: 10px !important;
  background: #171b27 !important;
  border: 1px solid rgba(255,255,255,0.06) !important;
  box-shadow: none !important;
  text-align: left !important;
}

#search-view .gr-chatbot .message * {margin: 0 !important; padding: 0 !important;}

#search-view .gr-chatbot .message,
#search-view .gr-chatbot .message * {
  /* Preserve clarity by neutralizing pseudo quote glyphs injected by the framework. */
  quotes: none !important;
}

#search-view .gr-chatbot .message::before,
#search-view .gr-chatbot .message::after,
#search-view .gr-chatbot .message *::before,
#search-view .gr-chatbot .message *::after,
#search-view .gr-chatbot .message blockquote::before,
#search-view .gr-chatbot .message blockquote::after,
#search-view .gr-chatbot .message q::before,
#search-view .gr-chatbot .message q::after {content: none !important; display: none !important;}

#search-view .gr-chatbot .message .avatar,
#search-view .gr-chatbot .message .icon,
#search-view .gr-chatbot .message [data-testid*="avatar"] {display: none !important;}

.card {padding: 10px; background: #171b27; border: 1px solid rgba(255,255,255,0.06); border-radius: 10px;}
.help-page {line-height: 1.6; color: #d9deeb;}

@media (max-width: 900px){
  .layout-row{flex-direction:column;}
  .nav-rail{max-width:100%; width:100%;}
  .header-bar{flex-direction:column; align-items:flex-start;}
}
"""


def build_app() -> gr.Blocks:
    """Create the Blocks application with conditional rendering."""

    logger.info("Initializing Gradio Blocks interface with custom theming and verbose instrumentation")

    with gr.Blocks(
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(),
        analytics_enabled=False,
    ) as demo:
        app_state = gr.State(_initial_state())
        user_state = gr.State("")
        role_state = gr.State("")
        session_state = gr.State("")
        selected_doc_state = gr.State("")

        # ---------------------- Login Screen ----------------------
        with gr.Column(
            visible=True,
            elem_classes=["panel", "login-card"],
            elem_id="login-view",
        ) as login_view:
            _safe_markdown("## Welcome to MQ RAG Ops", elem_classes=["title"])
            _safe_markdown(
                "Sign in to explore AI-assisted answers backed by your MQ knowledge base.",
                elem_classes=["status"],
            )
            userid = gr.Textbox(label="User ID", placeholder="Enter your username", autofocus=True)
            password = gr.Textbox(label="Password", placeholder="Enter your password", type="password")
            login_status = _safe_markdown(
                "Ready to sign in.", visible=False, elem_classes=["status"]
            )
            login_error = _safe_markdown(visible=False, elem_classes=["status"], value="")

        # ---------------------- Workspace -------------------------
        with gr.Column(visible=False, elem_id="workspace", elem_classes=["dashboard"]) as workspace:
            with gr.Row(elem_classes=["header-bar"]):
                with gr.Column(elem_classes=["header-left"], scale=1):
                    _safe_markdown("### MQ RAG Ops Dashboard", elem_classes=["no-margin"])
                with gr.Column(elem_classes=["header-right"], scale=1):
                    user_badge = _safe_markdown("", elem_classes=["badge"])
                    role_badge = _safe_markdown("", elem_classes=["badge"])
                    _safe_markdown("<span class='badge'>Environment: Prod</span>", elem_classes=[])
                    logout_btn = gr.Button("Logout", variant="secondary", elem_classes=["ghost"], scale=0)

            with gr.Row(elem_classes=["body-row", "layout-row"]):
                # Sidebar / nav rail
                with gr.Column(elem_classes=["nav-rail"], scale=3):
                    _safe_markdown("#### Navigation", elem_classes=["section-title"])
                    with gr.Column(elem_classes=["nav-buttons"]):
                        manage_docs_btn = gr.Button(
                            "📄 Manage Docs", elem_classes=["ghost"], variant="secondary"
                        )
                        help_btn = gr.Button("❓ Help & FAQ", elem_classes=["ghost"], variant="secondary")
                        _safe_markdown("", visible=False)  # spacer for alignment
                    _safe_markdown("#### Session", elem_classes=["section-title"])
                    session_status_card = _safe_markdown(
                        "Session ready. Conversations persist automatically.",
                        elem_classes=["status"],
                    )
                    with gr.Column(elem_classes=["history-panel", "panel"]):
                        _safe_markdown(
                            "Minimal history list placeholder. Sessions update automatically for clarity.",
                            elem_classes=["status"],
                        )

                # Main content
                with gr.Column(scale=9, elem_classes=["main-area"], elem_id="main-content"):
                    with gr.Column(visible=True, elem_id="search-view", elem_classes=["panel"]) as search_view:
                        with gr.Row(elem_classes=["kpi-strip"]):
                            kpi_sessions = _safe_markdown(
                                "<div class='kpi-card'><h4>Active Session</h4><p>Live</p></div>",
                                elem_classes=["kpi-card"],
                            )
                            kpi_docs = _safe_markdown(
                                "<div class='kpi-card'><h4>Docs Indexed</h4><p>—</p></div>",
                                elem_classes=["kpi-card"],
                            )
                            kpi_latency = _safe_markdown(
                                "<div class='kpi-card'><h4>Last Response</h4><p>&lt;1s</p></div>",
                                elem_classes=["kpi-card"],
                            )

                        with gr.Row(elem_classes=["workflow-panel"]):
                            with gr.Column(scale=7, elem_classes=["panel"]):
                                _safe_markdown("### Ask MQ", elem_classes=["title", "no-margin"])
                                hero_query = gr.Textbox(
                                    label="Ask anything about MQ",
                                    placeholder="Search documentation, logs, or troubleshooting steps…",
                                    lines=2,
                                    elem_classes=["hero-input"],
                                )
                                with gr.Row(elem_classes=["hero-actions"]):
                                    hero_clear_btn = gr.Button(
                                        "Clear",
                                        elem_classes=["ghost"],
                                        variant="secondary",
                                        scale=1,
                                    )
                                    hero_submit_btn = gr.Button(
                                        "Submit",
                                        elem_classes=["primary"],
                                        variant="primary",
                                        scale=1,
                                    )

                            with gr.Column(scale=5, elem_classes=["panel"]):
                                _safe_markdown(
                                    "#### Workflow Notes", elem_classes=["no-margin", "status"]
                                )
                                _safe_markdown(
                                    "Tune your prompt and submit for contextual answers. Clear will reset the input while retaining the session.",
                                    elem_classes=["status"],
                                )
                                clear_btn = gr.Button("Reset Session", elem_classes=["ghost"], variant="secondary")

                        with gr.Row(elem_classes=["result-grid"]):
                            with gr.Column(scale=7):
                                with gr.Tabs(elem_classes=["panel", "result-tabs"]):
                                    with gr.Tab("Answer"):
                                        chatbot = gr.Chatbot(
                                            height=420, bubble_full_width=False, elem_classes=["chatbot"]
                                        )
                                    with gr.Tab("Sources"):
                                        _safe_markdown(
                                            "Evidence and citations will be summarized here when available.",
                                            elem_classes=["status"],
                                        )
                                    with gr.Tab("Logs"):
                                        _safe_markdown(
                                            "Streaming diagnostics appear here during long-running operations.",
                                            elem_classes=["status"],
                                        )

                            with gr.Column(scale=5):
                                with gr.Accordion("Session Diagnostics", open=True, elem_classes=["panel"]):
                                    _safe_markdown(
                                        "Monitor the state of your conversation, environment, and document coverage.",
                                        elem_classes=["status"],
                                    )
                                    session_meta = _safe_markdown("", elem_classes=["status"])

                    with gr.Column(visible=False, elem_id="manage-docs-view", elem_classes=["panel"]) as manage_docs_view:
                        with gr.Row():
                            _safe_markdown("### Manage Docs", elem_classes=["title", "no-margin"])
                            back_to_search_from_docs = gr.Button(
                                "Back to Search", elem_classes=["ghost"], variant="secondary"
                            )
                            docs_logout = gr.Button("Logout", elem_classes=["ghost"], variant="secondary")
                        with gr.Row():
                            doc_status = _safe_markdown(
                                "Browse and manage ingested documents.", elem_classes=["status"]
                            )
                            admin_hint = _safe_markdown("", elem_classes=["status"])
                        with gr.Row():
                            doc_search = gr.Textbox(
                                label="Search documents",
                                placeholder="Type to filter documents…",
                            )
                            delete_doc_btn = gr.Button(
                                "Delete Document", variant="stop", elem_classes=["ghost"], interactive=False
                            )
                        with gr.Row():
                            doc_table = gr.Dataframe(
                                headers=["Document", "Type", "Size", "Ingested", ""],
                                datatype=["str", "str", "str", "str", "str"],
                                row_count=(0, "dynamic"),
                                col_count=5,
                                value=[],
                                interactive=True,
                                wrap=True,
                                elem_classes=["docs-table"],
                            )
                        with gr.Row():
                            doc_upload = gr.File(
                                label="Upload PDFs",
                                file_count="multiple",
                                file_types=[".pdf"],
                                interactive=False,
                            )
                            ingest_btn = gr.Button(
                                "Ingest", elem_classes=["primary"], variant="primary", interactive=False
                            )
                        doc_overview = _safe_markdown("", elem_classes=["status"])

                    with gr.Column(visible=False, elem_id="help-view", elem_classes=["panel", "help-page"]) as help_view:
                        with gr.Row():
                            _safe_markdown("## Help & Onboarding", elem_classes=["no-margin"])
                            back_to_search = gr.Button("Back to Search", elem_classes=["primary"], variant="primary")
                        _safe_markdown(
                            """
**Overview**

Use this app to perform AI-powered search across your MQ knowledge base. Authenticate to access personalized sessions and document management.

 **Running a search**
 - Enter a query in the large search bar and press Enter.
 - The assistant responds with contextual answers and cites past turns.
 The assistant responds with contextual answers and cites past turns.

**Sessions**
- Sessions persist automatically so you can return to them later without manual cleanup.
- Use the Clear controls to reset history while staying in the same conversation.

**Manage Docs**
- Use the Manage Docs view to ingest PDFs and review the knowledge base.
- Administrators can upload, ingest, and delete documents feeding the RAG index.
- Standard users can browse documents but cannot modify them.

**FAQ & Troubleshooting**
- If authentication fails, verify credentials and try again.
- Slow responses? Check connectivity to the vector store and document index.
- Upload errors? Confirm PDF format and file size limits.
- Need more help? Contact your platform administrator.
                        """,
                            elem_classes=["help-page"],
                        )

        # ---------------------- Wiring ----------------------------

        # Login submits (Enter key)
        userid.submit(
            announce_login_attempt,
            inputs=[],
            outputs=[login_status, login_error],
        ).then(
            attempt_login,
            inputs=[userid, password, app_state],
            outputs=[
                app_state,
                login_status,
                login_error,
                login_view,
                workspace,
                help_view,
                search_view,
                manage_docs_view,
                user_badge,
                role_badge,
                session_state,
                chatbot,
                session_meta,
                hero_submit_btn,
                user_state,
                role_state,
                doc_table,
                doc_overview,
                doc_status,
                doc_upload,
                ingest_btn,
                delete_doc_btn,
                admin_hint,
            ],
        )
        password.submit(
            announce_login_attempt,
            inputs=[],
            outputs=[login_status, login_error],
        ).then(
            attempt_login,
            inputs=[userid, password, app_state],
            outputs=[
                app_state,
                login_status,
                login_error,
                login_view,
                workspace,
                help_view,
                search_view,
                manage_docs_view,
                user_badge,
                role_badge,
                session_state,
                chatbot,
                session_meta,
                hero_submit_btn,
                user_state,
                role_state,
                doc_table,
                doc_overview,
                doc_status,
                doc_upload,
                ingest_btn,
                delete_doc_btn,
                admin_hint,
            ],
        )

        # Search interactions
        hero_query.submit(
            respond,
            inputs=[hero_query, chatbot, app_state, session_state, user_state],
            outputs=[
                hero_query,
                chatbot,
                session_state,
                session_meta,
                hero_submit_btn,
            ],
        )
        hero_submit_btn.click(
            respond,
            inputs=[hero_query, chatbot, app_state, session_state, user_state],
            outputs=[
                hero_query,
                chatbot,
                session_state,
                session_meta,
                hero_submit_btn,
            ],
        )
        hero_clear_btn.click(
            clear_query_input,
            inputs=hero_query,
            outputs=hero_query,
            queue=False,
        )
        clear_btn.click(
            clear_session_history,
            inputs=[session_state, app_state],
            outputs=[chatbot, session_state, session_meta],
            queue=False,
        )

        # Document tools
        ingest_btn.click(
            ingest_and_refresh,
            inputs=doc_upload,
            outputs=[doc_table, doc_overview],
        )
        doc_search.change(
            filter_documents,
            inputs=doc_search,
            outputs=[doc_table, doc_overview],
        )
        doc_table.select(
            handle_document_action,
            inputs=[doc_search, app_state],
            outputs=[doc_table, doc_overview, selected_doc_state],
        )
        delete_doc_btn.click(
            delete_selected_document,
            inputs=[selected_doc_state, doc_search, app_state],
            outputs=[doc_table, doc_overview, selected_doc_state],
            # Align with current Gradio event signatures to preserve confirmation
            # prompts without triggering keyword errors during interface setup.
            js="() => confirm('Delete this document?')",
        )

        # Help navigation
        help_btn.click(
            lambda state: toggle_page("help", state),
            inputs=app_state,
            outputs=[app_state, login_view, workspace, help_view, search_view, manage_docs_view],
        )
        back_to_search.click(
            lambda state: toggle_page("search", state),
            inputs=app_state,
            outputs=[app_state, login_view, workspace, help_view, search_view, manage_docs_view],
        )
        manage_docs_btn.click(
            open_manage_docs,
            inputs=app_state,
            outputs=[
                app_state,
                login_view,
                workspace,
                help_view,
                search_view,
                manage_docs_view,
                doc_table,
                doc_overview,
                doc_status,
                doc_upload,
                ingest_btn,
                delete_doc_btn,
                admin_hint,
            ],
        )
        back_to_search_from_docs.click(
            return_to_search,
            inputs=app_state,
            outputs=[app_state, login_view, workspace, help_view, search_view, manage_docs_view],
        )

        # Logout
        logout_btn.click(
            logout,
            inputs=app_state,
            outputs=[
                app_state,
                login_status,
                login_error,
                login_view,
                workspace,
                help_view,
                search_view,
                manage_docs_view,
                user_badge,
                role_badge,
                session_state,
                chatbot,
                session_meta,
                hero_submit_btn,
                user_state,
                role_state,
                doc_table,
                doc_overview,
                doc_status,
                doc_upload,
                ingest_btn,
                delete_doc_btn,
                admin_hint,
            ],
        )
        docs_logout.click(
            logout,
            inputs=app_state,
            outputs=[
                app_state,
                login_status,
                login_error,
                login_view,
                workspace,
                help_view,
                search_view,
                manage_docs_view,
                user_badge,
                role_badge,
                session_state,
                chatbot,
                session_meta,
                hero_submit_btn,
                user_state,
                role_state,
                doc_table,
                doc_overview,
                doc_status,
                doc_upload,
                ingest_btn,
                delete_doc_btn,
                admin_hint,
            ],
        )

        # Configure document tools after login
        app_state.change(
            configure_doc_tools,
            inputs=app_state,
            outputs=[doc_upload, ingest_btn, delete_doc_btn, doc_table, doc_overview, admin_hint],
        )

    return demo


app = build_app()
logger.info("Starting background model prewarm to keep first response fast")
start_background_prewarm()


if __name__ == "__main__":
    app.launch(
        share=SHARE_INTERFACE,
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
    )

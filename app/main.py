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
    return record.session_id, record.history, meta, "Response time: --"


def clear_session_history(session_id: str, state: AppState) -> tuple:
    """Clear the active session while keeping the conversation record available."""

    role = state.get("role") or "user"
    username = state.get("user") or role
    record = _load_session(role, username, session_id)
    logger.info("Clearing history for session %s belonging to %s", record.session_id, _session_key(role, username))
    _persist_history(record.session_id, role, username, [])
    refreshed = _load_session(role, username, record.session_id)
    meta = _format_session_meta(refreshed)
    return [], "Response time: --", refreshed.session_id, meta


def _format_eta_status(progress: GenerationProgress) -> str:
    """Human-friendly status messaging for live ETA telemetry."""

    if progress.stage == "error":
        return progress.detail or "The RAG engine is unavailable."

    if progress.stage == "retrieval":
        retrieval_text = (
            f"Retrieval: {progress.retrieval_seconds:.2f}s" if progress.retrieval_seconds else "Retrieval pending"
        )
        prefill_hint = (
            f" · Prefill ETA ~{progress.prefill_seconds:.1f}s" if progress.prefill_seconds else ""
        )
        return retrieval_text + prefill_hint

    if progress.stage == "prefill_start":
        return "Preparing model input from retrieved context."

    if progress.stage == "prefill_complete":
        return (
            f"Prefill: {progress.prefill_seconds:.2f}s · Measuring token throughput for ETA."
            if progress.prefill_seconds
            else "Prefill completed. Measuring token throughput."
        )

    if progress.stage == "generation":
        if progress.tokens_per_second:
            remaining = (
                f"≈{progress.eta_seconds:.1f}s remaining" if progress.eta_seconds is not None else "estimating remaining time"
            )
            return f"Generating… {progress.tokens_per_second:.2f} tok/s · {remaining}"
        return "Generating… measuring token speed."

    if progress.stage == "done":
        retrieval = f"Retrieval: {progress.retrieval_seconds:.2f}s" if progress.retrieval_seconds else "Retrieval: --"
        prefill = f"Prefill: {progress.prefill_seconds:.2f}s" if progress.prefill_seconds else "Prefill: --"
        generation = (
            f"Generation tokens: {progress.decode_tokens} @ {progress.tokens_per_second:.2f} tok/s"
            if progress.tokens_per_second
            else f"Generation tokens: {progress.decode_tokens}"
        )
        return f"{retrieval} · {prefill} · {generation}"

    return progress.detail or "Working…"


def _stage_label(progress: GenerationProgress) -> str:
    """Map backend stage codes to user-facing stage descriptions."""

    if progress.stage == "retrieval":
        return "Retrieving context…"
    if progress.stage in {"prefill_start"}:
        return "Preparing model input…"
    if progress.stage in {"prefill_complete", "generation"}:
        return "Generating answer…"
    if progress.stage == "done":
        return "Finalizing…"
    if progress.stage == "error":
        return "Error"
    return "Working…"


def _format_elapsed_label(seconds: float | None) -> str:
    """Return a clock-style elapsed label suitable for the UI."""

    if seconds is None:
        return "Elapsed: --"

    minutes = int(seconds) // 60
    remaining = seconds - (minutes * 60)
    return f"Elapsed: {minutes:02d}:{remaining:04.1f}"


def respond(
    message: str,
    history: list[list[str]] | list[tuple[str, str]] | None,
    state: AppState,
    session_id: str,
    username: str,
):
    """Generate a response using the RAG engine while streaming status updates."""

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
            "Please enter a prompt to continue.",
            record.session_id,
            meta,
            "Ready to search.",
            False,
            gr.update(visible=False, value=""),
        )

    logger.info(
        "Received query for session %s belonging to %s", record.session_id, _session_key(active_role, active_user)
    )
    start_time = time.perf_counter()

    existing_history = [(turn[0], turn[1]) for turn in (history or [])]
    working_history: list[tuple[str, str]] = list(existing_history)
    working_history.append((sanitized, ""))
    _persist_history(record.session_id, active_role, active_user, working_history)

    partial_answer = ""
    error_detected = False
    for progress in stream_query_rag(sanitized):
        if progress.partial_answer:
            partial_answer = progress.partial_answer
        working_history[-1] = (sanitized, partial_answer)
        status_text = _format_eta_status(progress)
        stage_text = _stage_label(progress)
        _persist_history(record.session_id, active_role, active_user, working_history)
        meta = _format_session_meta(_load_session(active_role, active_user, record.session_id))
        if progress.stage == "error":
            error_detected = True
            logger.error("Progress stream reported error for session %s: %s", record.session_id, status_text)
            yield (
                "",
                working_history,
                status_text,
                record.session_id,
                meta,
                stage_text,
                True,
                gr.update(
                    value=f"{status_text} — Try again.",
                    visible=True,
                ),
            )
            return

        logger.debug(
            "Streaming update for session %s: stage=%s detail=%s", record.session_id, progress.stage, status_text
        )
        yield (
            "",
            working_history,
            status_text,
            record.session_id,
            meta,
            stage_text,
            error_detected,
            gr.update(visible=False, value=""),
        )

    elapsed = time.perf_counter() - start_time
    working_history[-1] = (sanitized, partial_answer)
    _persist_history(record.session_id, active_role, active_user, working_history)
    meta = _format_session_meta(_load_session(active_role, active_user, record.session_id))
    logger.info("Response for session %s completed in %.3f seconds", record.session_id, elapsed)
    yield (
        "",
        working_history,
        f"Response time: {elapsed:.2f}s",
        record.session_id,
        meta,
        "Finalizing…",
        error_detected,
        gr.update(visible=False, value=""),
    )


def begin_response_cycle(prompt: str, start_time: float | None) -> tuple:
    """Immediately display loading indicators and disable submission controls."""

    sanitized = (prompt or "").strip()
    new_start = time.perf_counter()
    thinking_text = "Retrieving context…"
    logger.info(
        "Starting response cycle at %.6f with prompt length=%d", new_start, len(sanitized)
    )
    return (
        new_start,
        gr.update(value=thinking_text, visible=True),
        gr.update(visible=True),
        gr.update(interactive=False),
        gr.update(value=thinking_text, visible=True),
        gr.update(value="Elapsed: 00:00.0", visible=True),
        gr.update(value="ETA: calculating…", visible=True),
        False,
        gr.update(visible=False, value=""),
    )


def finalize_response_cycle(current_timer: str, start_time: float | None, errored: bool) -> tuple:
    """Hide loading indicators, re-enable submission, and report elapsed time."""

    elapsed = (time.perf_counter() - start_time) if start_time else None
    done_text = (
        "Encountered an error while generating the answer." if errored else (current_timer or "Answered.")
    )
    elapsed_label = _format_elapsed_label(elapsed)
    logger.info(
        "Finalizing response cycle; elapsed=%.3f, previous_timer=%s, errored=%s",
        elapsed or -1.0,
        current_timer,
        errored,
    )
    return (
        gr.update(value=("Finalizing…" if not errored else "Error"), visible=False),
        gr.update(visible=False),
        gr.update(interactive=True),
        gr.update(value=done_text, visible=True),
        gr.update(value=elapsed_label, visible=True),
        gr.update(value=("ETA: --" if errored else "ETA: complete"), visible=not errored),
        gr.update(visible=errored, value=(done_text if errored else "")),
        False,
    )


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
            "Response time: --",
            gr.update(value="Ready to search.", visible=True),
            gr.update(visible=False),
            gr.update(value="Elapsed: 00:00.0", visible=False),
            gr.update(value="ETA: calculating…", visible=False),
            False,
            gr.update(visible=False, value=""),
            None,
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
        sessions[3],
        gr.update(value="Ready to search.", visible=True),
        gr.update(visible=False),
        gr.update(value="Elapsed: 00:00.0", visible=False),
        gr.update(value="ETA: calculating…", visible=False),
        False,
        gr.update(visible=False, value=""),
        None,
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
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(value="", visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(value=""),
        gr.update(value=""),
        "",
        [],
        "Session ready. Conversations persist automatically.",
        "Response time: --",
        gr.update(value="Ready to search.", visible=True),
        gr.update(visible=False),
        gr.update(value="Elapsed: 00:00.0", visible=False),
        gr.update(value="ETA: calculating…", visible=False),
        False,
        gr.update(visible=False, value=""),
        None,
        "",
        "",
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
/* Compact, standards-aligned styling to reduce whitespace while keeping the MQ UI legible. */
body {background: #0d0f12; color: #e8ebf1;}

.gradio-container {
  font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
  width: min(880px, calc(100% - 8px));
  margin: 0 auto;
  padding: 4px !important;
  gap: 6px !important;
}

/* Global compaction to minimize scroll and collapse empty wrappers. */
.gradio-container .block,
.gradio-container .gr-block,
.gradio-container .gr-form,
.gradio-container .gr-panel,
.gradio-container .gr-box,
.gradio-container .form,
.gradio-container .wrap {
  margin: 4px 0 !important;
  padding: 4px !important;
  gap: 4px !important;
}

.gradio-container .row,
.gradio-container .gr-row,
.gradio-container .column,
.gradio-container .gr-column {
  gap: 4px !important;
  margin: 2px 0 !important;
}

.gradio-container .empty,
.gradio-container div:empty {
  display: none !important;
  height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
}

.panel {background: #14171c; border: 1px solid #1f232b; border-radius: 14px; box-shadow: 0 8px 30px rgba(0,0,0,0.35);}
.sidebar {min-width: 280px; max-width: 320px; padding: 10px; gap: 8px;}

#search-view {
  padding: 4px 6px 6px !important;
  gap: 6px !important;
}

.hero-input input,
.hero-input textarea {
  border-radius: 14px;
  border: 1px solid #232834;
  background: #0f1117;
  color: #e8ebf1;
  font-size: 18px;
  padding: 10px 12px;
  margin: 0 !important;
}

.hero-actions {
  align-items: center !important;
  gap: 6px !important;
  margin: 2px 0 0 !important;
  padding: 0 !important;
}

.status-row {
  align-items: center !important;
  gap: 6px !important;
  flex-wrap: nowrap !important;
  padding: 2px 0 !important;
  margin: 0 !important;
  max-height: 40px;
  min-height: 28px;
}

.status-row .status,
#stage-indicator,
#elapsed-indicator,
#eta-indicator,
#loading-spinner {
  margin: 0 !important;
  padding: 0 !important;
  white-space: nowrap;
  line-height: 1.25;
}

button.primary {background: linear-gradient(135deg, #4b82f7, #8a6bff); color: #fff; border-radius: 12px; border: none;}
button, .btn {color: #f5f7ff !important; font-weight: 600;}
button.ghost {background: #1d2330; border: 1px solid #4a5770; color: #f5f7ff; border-radius: 10px;}
button.ghost:hover {background: #242c3a; border-color: #6a7aa0;}
.gr-button-primary:hover, button.primary:hover {filter: brightness(1.05);}
.session-table table {width: 100%;}
.session-table td:last-child, .docs-table td:last-child {text-align: center; width: 56px;}
.status {color: #9ea8c2; font-size: 13px; margin: 0 !important;}
.loading-spinner {width: 20px; height: 20px; border-radius: 50%; border: 3px solid #1f2937; border-top-color: #6fb1ff; animation: spin 0.9s linear infinite;}
@keyframes spin {to {transform: rotate(360deg);}}
.error-banner {background: rgba(128, 38, 38, 0.35); color: #f6dada; padding: 6px 10px; border-radius: 8px; border: 1px solid #a94040; font-weight: 600;}

.chatbot {
  background: #0f1117;
  border: 1px solid #1f232b;
  border-radius: 16px;
  padding: 6px 8px !important;
  margin-top: 4px !important;
}

#search-view .gr-chatbot,
#search-view .gr-chatbot > div {
  margin-top: 4px !important;
  padding-top: 0 !important;
}

#search-view .gr-chatbot .wrap {
  padding: 4px !important;
  gap: 6px !important;
}

.card {padding: 10px; background: #0f1117; border: 1px solid #1f232b; border-radius: 12px;}
.help-page {line-height: 1.6; color: #d9deeb;}

@media (max-width: 960px){.layout-row{flex-direction:column;} .sidebar{max-width:100%; width:100%;}}
"""

TIMER_SCRIPT = """
<script>
(() => {
  const HISTORY_KEY = "rag_latency_history";
  const MAX_SAMPLES = 20;
  let timerId = null;
  let startTimestamp = null;
  let lastStage = "";

  const loadHistory = () => {
    try {
      const raw = localStorage.getItem(HISTORY_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch (err) {
      console.warn("Unable to parse latency history", err);
      return [];
    }
  };

  const saveSample = (ms) => {
    const history = loadHistory();
    history.push(ms);
    if (history.length > MAX_SAMPLES) {
      history.splice(0, history.length - MAX_SAMPLES);
    }
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  };

  const formatElapsed = (ms) => {
    const totalSeconds = ms / 1000;
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = (totalSeconds - minutes * 60).toFixed(1).padStart(4, "0");
    return `${String(minutes).padStart(2, "0")}:${seconds}`;
  };

  const hideEta = () => {
    const etaEl = document.querySelector("#eta-indicator");
    if (etaEl) {
      etaEl.textContent = "ETA: --";
      etaEl.style.display = "none";
    }
  };

  const renderTelemetry = () => {
    if (!startTimestamp) return;
    const elapsedMs = Date.now() - startTimestamp;
    const elapsedEl = document.querySelector("#elapsed-indicator");
    const etaEl = document.querySelector("#eta-indicator");
    if (elapsedEl) {
      elapsedEl.textContent = `Elapsed: ${formatElapsed(elapsedMs)}`;
      elapsedEl.style.display = "";
    }
    if (etaEl) {
      const history = loadHistory();
      if (history.length < 3) {
        etaEl.textContent = "ETA: calculating…";
        etaEl.style.display = "";
      } else {
        const average = history.reduce((sum, value) => sum + value, 0) / history.length;
        const remaining = Math.max(0, average - elapsedMs);
        etaEl.textContent = `ETA: ~${(remaining / 1000).toFixed(1)}s`;
        etaEl.style.display = "";
      }
    }
  };

  const startTimer = () => {
    if (timerId) clearInterval(timerId);
    startTimestamp = Date.now();
    timerId = window.setInterval(renderTelemetry, 100);
    renderTelemetry();
  };

  const stopTimer = (recordSample) => {
    if (timerId) {
      clearInterval(timerId);
      timerId = null;
    }
    if (startTimestamp && recordSample) {
      saveSample(Date.now() - startTimestamp);
    }
    startTimestamp = null;
  };

  const handleStageChange = (text) => {
    const normalized = (text || "").toLowerCase();
    if (!normalized) return;
    if (normalized.startsWith("retrieving") || normalized.startsWith("preparing")) {
      startTimer();
      return;
    }
    if (normalized.startsWith("generating")) {
      if (!startTimestamp) {
        startTimer();
      }
      return;
    }
    if (normalized.startsWith("finalizing")) {
      stopTimer(true);
      return;
    }
    if (normalized.startsWith("error")) {
      stopTimer(false);
      hideEta();
    }
  };

  const attach = () => {
    const stageEl = document.querySelector("#stage-indicator");
    const errorEl = document.querySelector("#error-banner");
    if (!stageEl) {
      window.setTimeout(attach, 400);
      return;
    }

    window.setInterval(() => {
      const stageText = (stageEl.textContent || "").trim();
      if (stageText !== lastStage) {
        lastStage = stageText;
        handleStageChange(stageText);
      }
      if (errorEl && errorEl.textContent.trim()) {
        stopTimer(false);
        hideEta();
      }
      if (startTimestamp) {
        renderTelemetry();
      }
    }, 150);
  };

  attach();
})();
</script>
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
        start_time_state = gr.State(None)
        error_state = gr.State(False)
        gr.HTML(TIMER_SCRIPT, visible=False, elem_id="timer-script")

        # ---------------------- Login Screen ----------------------
        with gr.Column(visible=True, elem_classes=["panel"], elem_id="login-view") as login_view:
            _safe_markdown("## Welcome to MQ RAG Search", elem_classes=["title"])
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
        with gr.Row(visible=False, elem_id="workspace", elem_classes=["layout-row"]) as workspace:
            # Sidebar
            with gr.Column(elem_classes=["sidebar", "panel"], scale=3):
                with gr.Row():
                    _safe_markdown("### AI Search", elem_classes=["no-margin"])
                    logout_btn = gr.Button("Logout", variant="secondary", elem_classes=["ghost"], scale=0)
                with gr.Row():
                    user_badge = _safe_markdown("", elem_classes=["status"])
                    role_badge = _safe_markdown("", elem_classes=["status"])
                session_meta = _safe_markdown("Session ready. Conversations persist automatically.", elem_classes=["status"])
                manage_docs_btn = gr.Button("Manage Docs", elem_classes=["ghost"], variant="secondary")
                help_btn = gr.Button("Help & FAQ", elem_classes=["ghost"], variant="secondary")

            # Main content
            with gr.Column(scale=9, elem_classes=["panel"], elem_id="main-content"):
                with gr.Column(visible=True, elem_id="search-view") as search_view:
                    _safe_markdown("### MQ AI Search", elem_classes=["title"])
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
                    with gr.Row(elem_classes=["status-row"]):
                        loading_spinner = gr.HTML(
                            value="<div class='loading-spinner' aria-label='Loading'></div>",
                            visible=False,
                            elem_id="loading-spinner",
                        )
                        stage_md = _safe_markdown(
                            "Ready to search.", elem_classes=["status"], elem_id="stage-indicator"
                        )
                        elapsed_md = _safe_markdown(
                            "Elapsed: 00:00.0",
                            visible=False,
                            elem_classes=["status"],
                            elem_id="elapsed-indicator",
                        )
                        eta_md = _safe_markdown(
                            "ETA: calculating…",
                            visible=False,
                            elem_classes=["status"],
                            elem_id="eta-indicator",
                        )
                    with gr.Row(elem_classes=["status-row"]):
                        response_timer = _safe_markdown(
                            "Response time: --", elem_classes=["status"], elem_id="status-detail"
                        )
                        error_banner = _safe_markdown(
                            "",
                            visible=False,
                            elem_classes=["error-banner"],
                            elem_id="error-banner",
                        )
                    chatbot = gr.Chatbot(height=520, bubble_full_width=False, elem_classes=["chatbot"])
                    with gr.Row():
                        clear_btn = gr.Button("Clear", elem_classes=["ghost"], variant="secondary")

                with gr.Column(visible=False, elem_id="manage-docs-view") as manage_docs_view:
                    _safe_markdown("### Manage Docs", elem_classes=["title"])
                    with gr.Row():
                        back_to_search_from_docs = gr.Button(
                            "Back to Search", elem_classes=["ghost"], variant="secondary"
                        )
                        docs_logout = gr.Button("Logout", elem_classes=["ghost"], variant="secondary")
                    doc_status = _safe_markdown("Browse and manage ingested documents.", elem_classes=["status"])
                    doc_search = gr.Textbox(
                        label="Search documents",
                        placeholder="Type to filter documents…",
                    )
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
                    delete_doc_btn = gr.Button(
                        "Delete Document", variant="stop", elem_classes=["ghost"], interactive=False
                    )
                    with gr.Row():
                        doc_upload = gr.File(
                            label="Upload PDFs",
                            file_count="multiple",
                            file_types=[".pdf"],
                            interactive=False,
                        )
                        ingest_btn = gr.Button("Ingest", elem_classes=["primary"], variant="primary", interactive=False)
                    doc_overview = _safe_markdown("", elem_classes=["status"])
                    admin_hint = _safe_markdown("", elem_classes=["status"])

                with gr.Column(visible=False, elem_id="help-view", elem_classes=["help-page"]) as help_view:
                    _safe_markdown("## Help & Onboarding")
                    back_to_search = gr.Button("Back to Search", elem_classes=["primary"], variant="primary")
                    _safe_markdown(
                        """
**Overview**

Use this app to perform AI-powered search across your MQ knowledge base. Authenticate to access personalized sessions and document management.

**Running a search**
- Enter a query in the large search bar and press Enter.
- The assistant responds with contextual answers and cites past turns.
- Response time and session metadata remain visible above the chat stream.

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
                response_timer,
                stage_md,
                loading_spinner,
                elapsed_md,
                eta_md,
                error_state,
                error_banner,
                start_time_state,
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
                response_timer,
                stage_md,
                loading_spinner,
                elapsed_md,
                eta_md,
                error_state,
                error_banner,
                start_time_state,
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
            begin_response_cycle,
            inputs=[hero_query, start_time_state],
            outputs=[
                start_time_state,
                stage_md,
                loading_spinner,
                hero_submit_btn,
                response_timer,
                elapsed_md,
                eta_md,
                error_state,
                error_banner,
            ],
        ).then(
            respond,
            inputs=[hero_query, chatbot, app_state, session_state, user_state],
            outputs=[
                hero_query,
                chatbot,
                response_timer,
                session_state,
                session_meta,
                stage_md,
                error_state,
                error_banner,
            ],
        ).then(
            finalize_response_cycle,
            inputs=[response_timer, start_time_state, error_state],
            outputs=[
                stage_md,
                loading_spinner,
                hero_submit_btn,
                response_timer,
                elapsed_md,
                eta_md,
                error_banner,
                error_state,
            ],
        )
        hero_submit_btn.click(
            begin_response_cycle,
            inputs=[hero_query, start_time_state],
            outputs=[
                start_time_state,
                stage_md,
                loading_spinner,
                hero_submit_btn,
                response_timer,
                elapsed_md,
                eta_md,
                error_state,
                error_banner,
            ],
        ).then(
            respond,
            inputs=[hero_query, chatbot, app_state, session_state, user_state],
            outputs=[
                hero_query,
                chatbot,
                response_timer,
                session_state,
                session_meta,
                stage_md,
                error_state,
                error_banner,
            ],
        ).then(
            finalize_response_cycle,
            inputs=[response_timer, start_time_state, error_state],
            outputs=[
                stage_md,
                loading_spinner,
                hero_submit_btn,
                response_timer,
                elapsed_md,
                eta_md,
                error_banner,
                error_state,
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
            outputs=[chatbot, response_timer, session_state, session_meta],
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
                login_view,
                workspace,
                login_error,
                search_view,
                manage_docs_view,
                user_badge,
                role_badge,
                session_state,
                chatbot,
                session_meta,
                response_timer,
                stage_md,
                loading_spinner,
                elapsed_md,
                eta_md,
                error_state,
                error_banner,
                start_time_state,
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
                login_view,
                workspace,
                login_error,
                search_view,
                manage_docs_view,
                user_badge,
                role_badge,
                session_state,
                chatbot,
                session_meta,
                response_timer,
                stage_md,
                loading_spinner,
                elapsed_md,
                eta_md,
                error_state,
                error_banner,
                start_time_state,
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


if __name__ == "__main__":
    app.launch(
        share=SHARE_INTERFACE,
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
    )

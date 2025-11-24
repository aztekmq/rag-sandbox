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
from typing import Dict, Tuple, TypedDict

import gradio as gr

from app.auth import authenticate
from app.config import PDF_DIR, SHARE_INTERFACE
from app.rag_chain import delete_document, get_documents_list, ingest_pdfs, query_rag

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
    """Render a compact summary for the active session."""

    duration = record.updated_at - record.created_at
    return (
        f"**Active session:** {record.title}\n"
        f"Created: {record.created_at:%b %d %H:%M UTC} · Updated: {record.updated_at:%b %d %H:%M UTC}\n"
        f"Turns: {len(record.history)} · Lifespan: {duration.total_seconds():.0f}s"
    )


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
        logger.debug("Loaded requested session %s for %s", session_id, _session_key(role, username))
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
        logger.warning("Attempted to delete nonexistent session %s for %s", session_id, _session_key(role, username))

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


def refresh_library(status: str | None = None) -> tuple[gr.update, str]:
    """Return an updated document dropdown alongside a formatted overview."""

    docs = refresh_documents()
    default_choice = docs[0] if docs else None
    overview = _render_library_overview(docs)
    if status:
        overview = f"{status}\n\n{overview}"

    logger.debug("Library state prepared with %d documents (default=%s)", len(docs), default_choice)
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
    choices = _list_session_choices(role, username)
    meta = _format_session_meta(record)
    return (
        gr.update(choices=choices, value=record.session_id),
        record.session_id,
        record.history,
        meta,
        "Response time: --",
    )


def start_new_session(state: AppState) -> tuple:
    """Create a brand-new session, returning updates for the selection UI."""

    role = state.get("role") or "user"
    username = state.get("user") or role
    bucket = _get_session_bucket(role, username)
    session_id = str(uuid.uuid4())
    record = SessionRecord(session_id=session_id, title=_format_session_title(bucket))
    bucket[session_id] = record
    logger.info("Started new session %s for %s", session_id, _session_key(role, username))
    choices = _list_session_choices(role, username)
    meta = _format_session_meta(record)
    return gr.update(choices=choices, value=session_id), session_id, [], meta, "Response time: --"


def select_session(session_id: str, state: AppState) -> tuple:
    """Switch the active session and surface its history in the chat UI."""

    role = state.get("role") or "user"
    username = state.get("user") or role
    record = _load_session(role, username, session_id)
    logger.info("Switching to session %s for %s", record.session_id, _session_key(role, username))
    choices = _list_session_choices(role, username)
    meta = _format_session_meta(record)
    return gr.update(choices=choices, value=record.session_id), record.session_id, record.history, meta, "Response time: --"


def remove_session(session_id: str | None, state: AppState) -> tuple:
    """Delete the selected session and refresh selection controls."""

    role = state.get("role") or "user"
    username = state.get("user") or role
    record = _delete_session(role, username, session_id)
    logger.info("Session %s removed; activating %s for %s", session_id, record.session_id, _session_key(role, username))
    choices = _list_session_choices(role, username)
    meta = _format_session_meta(record)
    return gr.update(choices=choices, value=record.session_id), record.session_id, record.history, meta, "Response time: --"


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
        return "", record.history, "Please enter a prompt to continue.", record.session_id, meta

    logger.info("Received query for session %s belonging to %s", record.session_id, _session_key(active_role, active_user))
    start_time = time.perf_counter()

    existing_history = [(turn[0], turn[1]) for turn in (history or [])]
    working_history: list[tuple[str, str]] = list(existing_history)
    working_history.append((sanitized, ""))
    _persist_history(record.session_id, active_role, active_user, working_history)
    meta = _format_session_meta(record)
    yield "", working_history, "Thinking…", record.session_id, meta

    answer = query_rag(sanitized)
    elapsed = time.perf_counter() - start_time
    working_history[-1] = (sanitized, answer)
    _persist_history(record.session_id, active_role, active_user, working_history)
    meta = _format_session_meta(_load_session(active_role, active_user, record.session_id))
    logger.info("Response for session %s completed in %.3f seconds", record.session_id, elapsed)
    return "", working_history, f"Response time: {elapsed:.2f}s", record.session_id, meta


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


def attempt_login(username: str, password: str, state: AppState) -> tuple:
    """Authenticate and transition to the search page when successful."""

    role = _detect_role(username, password)
    if not role:
        logger.warning("Login failed; keeping user on login page")
        return (
            state,
            gr.update(value="Invalid credentials. Please try again.", visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(choices=[], value=None),
            "",
            [],
            "Select a session to begin.",
            "Response time: --",
            "",
            "",
        )

    safe_username = username or role.title()
    new_state = {
        "page": "search",
        "role": role,
        "user": safe_username,
        "session_id": "",
    }
    sessions = hydrate_sessions(role, safe_username)
    logger.info("Login succeeded for %s; transitioning to search page", safe_username)
    return (
        new_state,
        gr.update(value="", visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(value=safe_username),
        gr.update(value=role.title()),
        sessions[0],
        sessions[1],
        sessions[2],
        sessions[3],
        sessions[4],
        safe_username,
        role,
    )


def logout(state: AppState) -> tuple:
    """Return to the landing page and clear transient state."""

    logger.info("Logout requested for role: %s", state.get("role") or "unknown")
    cleared_state = _initial_state()
    return (
        cleared_state,
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(value="", visible=False),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(choices=[], value=None),
        "",
        [],
        "Select a session to begin.",
        "Response time: --",
        "",
        "",
    )


def toggle_page(target: str, state: AppState) -> tuple:
    """Toggle between search and help views while remaining logged in."""

    updated = _set_page(state, target)
    return (
        updated,
        gr.update(visible=target == "login"),
        gr.update(visible=target != "login"),
        gr.update(visible=target == "help"),
        gr.update(visible=target != "help"),
    )


def configure_doc_tools(state: AppState) -> tuple:
    """Configure Document Tools interactivity based on role."""

    is_admin = state.get("role") == "admin"
    tooltip = "Full access" if is_admin else "Admin-only"
    logger.debug("Configuring document tools for %s", state.get("role") or "anonymous")
    dropdown_update, overview = refresh_library()
    return (
        gr.update(interactive=is_admin),
        gr.update(interactive=is_admin),
        gr.update(interactive=is_admin),
        gr.update(interactive=is_admin),
        dropdown_update,
        gr.update(value=overview + ("\n\nAdmin-only." if not is_admin else "")),
        tooltip,
    )


# ---------------------------------------------------------------------------
# UI assembly
# ---------------------------------------------------------------------------


CUSTOM_CSS = """
body {background: #0d0f12; color: #e8ebf1;}
.gradio-container {font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;}
.panel {background: #14171c; border: 1px solid #1f232b; border-radius: 14px; box-shadow: 0 8px 30px rgba(0,0,0,0.35);}
.sidebar {min-width: 280px; max-width: 320px; padding: 12px; gap: 10px;}
.hero-input input, .hero-input textarea {border-radius: 14px; border: 1px solid #232834; background: #0f1117; color: #e8ebf1; font-size: 18px; padding: 16px 18px;}
button.primary {background: linear-gradient(135deg, #4b82f7, #8a6bff); color: #fff; border-radius: 12px; border: none;}
button.ghost {background: transparent; border: 1px solid #2c3342; color: #e8ebf1; border-radius: 10px;}
.status {color: #9ea8c2; font-size: 13px;}
.chatbot {background: #0f1117; border: 1px solid #1f232b; border-radius: 16px;}
.card {padding: 14px; background: #0f1117; border: 1px solid #1f232b; border-radius: 12px;}
.help-page {line-height: 1.6; color: #d9deeb;}
@media (max-width: 960px){.layout-row{flex-direction:column;} .sidebar{max-width:100%; width:100%;}}
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

        # ---------------------- Login Screen ----------------------
        with gr.Column(visible=True, elem_classes=["panel"], elem_id="login-view") as login_view:
            _safe_markdown("## Welcome to MQ RAG Search", elem_classes=["title"])
            _safe_markdown(
                "Sign in to explore AI-assisted answers backed by your MQ knowledge base.",
                elem_classes=["status"],
            )
            userid = gr.Textbox(label="User ID", placeholder="Enter your username", autofocus=True)
            password = gr.Textbox(label="Password", placeholder="Enter your password", type="password")
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
                new_session_btn = gr.Button("New Search / Session", elem_classes=["primary"], variant="primary")
                session_radio = gr.Radio(label="Recent Sessions", choices=[], interactive=True)
                session_meta = _safe_markdown("Select a session to begin.", elem_classes=["status"])
                with gr.Row():
                    view_btn = gr.Button("View", elem_classes=["ghost"], variant="secondary")
                    delete_session_btn = gr.Button("Delete", elem_classes=["ghost"], variant="secondary")

                with gr.Accordion("Document Tools", open=False):
                    doc_upload = gr.File(label="Upload PDFs", file_count="multiple", file_types=[".pdf"], interactive=False)
                    ingest_btn = gr.Button("Ingest", elem_classes=["primary"], variant="primary", interactive=False)
                    doc_list = gr.Dropdown(label="Documents", choices=[], interactive=False)
                    metadata = gr.Textbox(label="Metadata / Notes", placeholder="Admin can edit", interactive=False)
                    doc_delete = gr.Button("Delete Document", variant="secondary", interactive=False)
                    doc_overview = _safe_markdown("", elem_classes=["status"])
                    admin_hint = _safe_markdown("Admin-only.", elem_classes=["status"])

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
                    response_timer = _safe_markdown("Response time: --", elem_classes=["status"])
                    chatbot = gr.Chatbot(height=520, bubble_full_width=False, elem_classes=["chatbot"])
                    with gr.Row():
                        clear_btn = gr.Button("Clear", elem_classes=["ghost"], variant="secondary")
                        back_to_help_btn = gr.Button("Open Help", elem_classes=["ghost"], variant="secondary")

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
- "New Search / Session" starts a clean conversation.
- Select any recent session to view its history.
- "Delete" removes the highlighted session.

**Document tools**
- Administrators can upload, ingest, edit metadata, and delete documents feeding the RAG index.
- Standard users can browse documents but cannot modify them (controls disabled with an Admin-only tooltip).

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
            attempt_login,
            inputs=[userid, password, app_state],
            outputs=[
                app_state,
                login_error,
                login_view,
                workspace,
                help_view,
                user_badge,
                role_badge,
                session_radio,
                session_state,
                chatbot,
                session_meta,
                response_timer,
                user_state,
                role_state,
            ],
        )
        password.submit(
            attempt_login,
            inputs=[userid, password, app_state],
            outputs=[
                app_state,
                login_error,
                login_view,
                workspace,
                help_view,
                user_badge,
                role_badge,
                session_radio,
                session_state,
                chatbot,
                session_meta,
                response_timer,
                user_state,
                role_state,
            ],
        )

        # Search interactions
        hero_query.submit(
            respond,
            inputs=[hero_query, chatbot, app_state, session_state, user_state],
            outputs=[hero_query, chatbot, response_timer, session_state, session_meta],
        )
        clear_btn.click(
            clear_session_history,
            inputs=[session_state, app_state],
            outputs=[chatbot, response_timer, session_state, session_meta],
            queue=False,
        )

        # Session controls
        new_session_btn.click(
            start_new_session,
            inputs=app_state,
            outputs=[session_radio, session_state, chatbot, session_meta, response_timer],
        )
        view_btn.click(
            select_session,
            inputs=[session_radio, app_state],
            outputs=[session_radio, session_state, chatbot, session_meta, response_timer],
        )
        delete_session_btn.click(
            remove_session,
            inputs=[session_radio, app_state],
            outputs=[session_radio, session_state, chatbot, session_meta, response_timer],
        )

        # Document tools
        ingest_btn.click(
            ingest_and_refresh,
            inputs=doc_upload,
            outputs=[doc_list, doc_overview],
        )
        doc_delete.click(
            delete_and_refresh,
            inputs=doc_list,
            outputs=[doc_list, doc_overview],
        )
        doc_list.change(
            lambda doc: doc or "Select a document to preview.",
            inputs=doc_list,
            outputs=metadata,
        )

        # Help navigation
        help_btn.click(
            lambda state: toggle_page("help", state),
            inputs=app_state,
            outputs=[app_state, login_view, workspace, help_view, search_view],
        )
        back_to_search.click(
            lambda state: toggle_page("search", state),
            inputs=app_state,
            outputs=[app_state, login_view, workspace, help_view, search_view],
        )
        back_to_help_btn.click(
            lambda state: toggle_page("help", state),
            inputs=app_state,
            outputs=[app_state, login_view, workspace, help_view, search_view],
        )

        # Logout
        logout_btn.click(
            logout,
            inputs=app_state,
            outputs=[
                app_state,
                login_view,
                workspace,
                login_error,
                user_badge,
                role_badge,
                session_radio,
                session_state,
                chatbot,
                session_meta,
                response_timer,
                user_state,
                role_state,
            ],
        )

        # Configure document tools after login
        app_state.change(
            configure_doc_tools,
            inputs=app_state,
            outputs=[ingest_btn, doc_upload, doc_delete, metadata, doc_list, doc_overview, admin_hint],
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

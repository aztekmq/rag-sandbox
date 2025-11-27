"""Gradio UI with fixed sidebar, KPI strip, and tabbed results view.

This module rebuilds the Blocks layout to match the target operations dashboard
style described in the project manifest. A left-hand sidebar surfaces session
controls and recent history, while the main canvas stacks KPIs, query input, and
results tabs. Verbose logging is preserved to simplify debugging and satisfy
international programming standards for observability.
"""

from __future__ import annotations

import inspect
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple, TypedDict

import gradio as gr

from app.config import SHARE_INTERFACE
from app.rag_chain import query_rag, start_background_prewarm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# ---------------------------------------------------------------------------
# Runtime environment hardening
# ---------------------------------------------------------------------------

# Force ONNX Runtime to remain on CPU-only execution paths inside containers
# where GPU discovery fails (e.g., missing /sys/class/drm/*). Keeping the
# configuration explicit prevents noisy warnings during startup and clarifies
# the intended hardware target for operators. This follows international
# programming standards by documenting operational safeguards directly in code.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("ORT_DEVICE_ALLOWLIST", "cpu")
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")
logger.debug(
    "GPU execution disabled; using CPU-only ONNX providers and suppressing verbose ONNX warnings"
)


# ---------------------------------------------------------------------------
# Gradio compatibility shims
# ---------------------------------------------------------------------------

# Gradio 4.x removed the ``scale`` argument from :class:`gr.Row`. Older code
# paths (including legacy deployments) can still pass ``scale`` and raise a
# ``TypeError`` before the UI initializes. The shim below preserves verbose
# logging, strips the unsupported key for compatibility, and retains the
# original initializer for future debugging. It also provides a defensive
# fallback path that retries initialization with sanitized arguments so UI
# construction never fails purely because of keyword drift between Gradio
# releases.
_ROW_SIGNATURE = inspect.signature(gr.Row.__init__)
_ORIGINAL_ROW_INIT = gr.Row.__init__
_ROW_PATCH_APPLIED = False


def _sanitize_row_kwargs(raw_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of Row kwargs with deprecated keys removed.

    Centralizing this logic keeps both the runtime shim and the helper factory
    aligned while preserving verbose logging that explains every mutation.
    """

    cleaned = dict(raw_kwargs)

    if "scale" in cleaned:
        removed_scale = cleaned.pop("scale")
        logger.warning(
            "Removed unsupported 'scale' kwarg=%s from gr.Row for compatibility", removed_scale
        )

    allowed_keys = set(_ROW_SIGNATURE.parameters.keys()) - {"self"}
    unsupported = [key for key in list(cleaned.keys()) if key not in allowed_keys]
    if unsupported:
        for key in unsupported:
            removed = cleaned.pop(key)
            logger.warning(
                "Dropped unsupported gr.Row kwarg %s=%s to prevent TypeError", key, removed
            )

    logger.debug("Sanitized gr.Row kwargs from %s to %s", raw_kwargs, cleaned)
    return cleaned


def apply_gradio_row_shim() -> None:
    """Apply a compatibility shim that ignores deprecated ``scale`` arguments.

    The helper is idempotent and logs every action so operators can trace when
    layout compatibility fixes were installed. Centralizing the patch also keeps
    the module aligned with international programming standards for runtime
    documentation and observability.
    """

    global _ROW_PATCH_APPLIED

    if _ROW_PATCH_APPLIED:
        logger.debug("gr.Row shim already active; skipping duplicate patch")
        return

    def _patched_row_init(self, *args: Any, **kwargs: Any) -> None:
        """Patch ``gr.Row.__init__`` to drop deprecated arguments safely."""

        if kwargs:
            kwargs = _sanitize_row_kwargs(kwargs)

        try:
            _ORIGINAL_ROW_INIT(self, *args, **kwargs)
        except TypeError as exc:  # pragma: no cover - defensive path
            # Gradio versions without **kwargs on Row will still raise if a
            # legacy caller passes ``scale``. Strip the key and retry once so
            # the UI can continue initializing instead of crashing at startup.
            logger.error(
                "Retrying gr.Row init with sanitized kwargs after failure: %s", exc
            )
            sanitized_kwargs = {
                k: v for k, v in kwargs.items() if k in _ROW_SIGNATURE.parameters
            }
            _ORIGINAL_ROW_INIT(self, *args, **sanitized_kwargs)

    gr.Row.__init__ = _patched_row_init
    logger.info(
        "Patched gr.Row.__init__ to ignore deprecated 'scale' kwarg (Gradio %s)",
        gr.__version__,
    )

    _ROW_PATCH_APPLIED = True


apply_gradio_row_shim()


# ---------------------------------------------------------------------------
# Simple session store
# ---------------------------------------------------------------------------


@dataclass
class SessionRecord:
    """Persist a user's chat history and timestamps for the session."""

    session_id: str
    history: List[Tuple[str, str]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def render_history(self, limit: int = 6) -> str:
        """Render recent history as Markdown bullet items for the sidebar."""

        if not self.history:
            return "*No turns yet. Ask your first question to start a trail.*"

        lines: list[str] = ["### Recent Searches"]
        for idx, (query, _) in enumerate(self.history[-limit:], start=1):
            truncated = (query[:60] + "…") if len(query) > 60 else query
            lines.append(f"{idx}. {truncated}")
        return "\n".join(lines)


class AppState(TypedDict):
    """Typed dictionary stored in the Gradio state container."""

    user: str
    session_id: str
    active_view: str


SESSION_STORE: Dict[str, SessionRecord] = {}
DEFAULT_SOURCES: list[list[Any]] = [
    [
        "Knowledge Base",
        "Upload or sync documents to see citation details for each answer.",
        None,
    ]
]
DEFAULT_LOG_MESSAGE = "Awaiting first query."

ASSETS_DIR = Path(__file__).parent / "assets"
CUSTOM_CSS_PATH = ASSETS_DIR / "custom.css"


def _default_sources() -> list[list[Any]]:
    """Return a shallow copy of the default source rows for UI binding."""

    return [list(row) for row in DEFAULT_SOURCES]


def _get_or_create_session(username: str, session_id: str | None) -> SessionRecord:
    """Return an existing session or create a new one for this user."""

    if session_id and session_id in SESSION_STORE:
        logger.debug("Reusing existing session %s for user %s", session_id, username)
        return SESSION_STORE[session_id]

    new_id = session_id or str(uuid.uuid4())
    record = SessionRecord(session_id=new_id)
    SESSION_STORE[new_id] = record
    logger.info("Created new session %s for user %s", new_id, username)
    return record


# ---------------------------------------------------------------------------
# Chat handler
# ---------------------------------------------------------------------------


def _format_answer(query: str, answer: str) -> str:
    """Compose display-friendly Markdown for the main response panel."""

    return f"**Query:** *{query}*\n\n**AI Result:** {answer}"


def handle_query(
    message: str,
    state: AppState,
) -> Generator[
    Tuple[str, List[List[Any]], str, str, AppState, str, str], None, None
]:
    """Process a user query with lightweight streaming updates.

    The generator yields an immediate acknowledgement for responsiveness followed
    by the completed answer, refreshed session state, and sidebar history. Verbose
    logging tracks each step so operators can trace user interactions and
    troubleshoot issues rapidly.
    """

    username = state.get("user", "admin")
    session_id = state.get("session_id") or ""
    record = _get_or_create_session(username, session_id)

    text = (message or "").strip()
    if not text:
        logger.debug(
            "Received empty query for user=%s session=%s; prompting for input",
            username,
            record.session_id,
        )
        sidebar_md = record.render_history()
        yield (
            "Please enter a question to begin.",
            _default_sources(),
            DEFAULT_LOG_MESSAGE,
            "",
            state,
            "",
            sidebar_md,
        )
        return

    logger.info("Handling query for user=%s session=%s", username, record.session_id)
    sidebar_md = record.render_history()
    yield "Thinking...", _default_sources(), "Query received...", text, state, message, sidebar_md

    answer = query_rag(text)
    record.history.append((text, answer))
    record.updated_at = datetime.utcnow()
    state["session_id"] = record.session_id

    logger.debug(
        "Updated session %s with %d total turns", record.session_id, len(record.history)
    )

    formatted_answer = _format_answer(text, answer)
    sidebar_md = record.render_history()
    yield formatted_answer, _default_sources(), "Completed without errors.", answer, state, "", sidebar_md


def start_new_conversation(
    state: AppState,
) -> Tuple[str, List[List[Any]], str, str, AppState, str, str]:
    """Reset the chat for a fresh session while retaining the username."""

    logger.info("Starting a new conversation for user=%s", state.get("user", "admin"))
    state["session_id"] = ""
    greeting = "The AI is ready. Enter your query below."
    sidebar_md = SessionRecord(session_id="temp").render_history()
    return greeting, _default_sources(), DEFAULT_LOG_MESSAGE, "", state, "", sidebar_md


def toggle_sidebar(current_visibility: bool) -> Tuple[bool, gr.update]:
    """Flip the sidebar visibility state and emit an updated column config."""

    new_state = not current_visibility
    logger.debug("Sidebar visibility toggled to %s", new_state)
    return new_state, gr.Column.update(visible=new_state)


def switch_view(
    target_view: str, current_view: str
) -> Tuple[str, gr.update, gr.update, gr.update]:
    """Toggle visible panels based on the requested target view.

    The helper centralizes navigation logic for the sidebar buttons so that
    Search, Docs, and Help panels can be shown or hidden without modifying
    backend behavior. It logs both the requested target and the resulting state
    to support detailed troubleshooting sessions.
    """

    normalized_target = (target_view or "search").lower()
    allowed_views = {"search", "docs", "help"}
    resolved_view = normalized_target if normalized_target in allowed_views else "search"
    logger.info(
        "Switching view from %s to %s (resolved target=%s)",
        current_view,
        target_view,
        resolved_view,
    )

    search_visibility = gr.Column.update(visible=resolved_view == "search")
    docs_visibility = gr.Column.update(visible=resolved_view == "docs")
    help_visibility = gr.Column.update(visible=resolved_view == "help")

    return resolved_view, search_visibility, docs_visibility, help_visibility


# ---------------------------------------------------------------------------
# Minimalist CSS and layout helpers
# ---------------------------------------------------------------------------

INLINE_CSS = """
.layout-row { align-items: stretch; gap: 16px !important; }
.sidebar-card { gap: 8px !important; }
.sidebar-footer { color: #7a6c5d; font-size: 13px; }
.kpi-strip { gap: 10px !important; }
.kpi-card { min-width: 150px; }
.result-tabs .gr-panel { background: transparent; border: none; box-shadow: none; }
.result-tabs .gr-tabitem { padding: 0 !important; }
.input-panel .gr-textbox { margin-top: 0; }
"""


def _create_row(**kwargs: Any) -> gr.Row:
    """Return a Gradio row while filtering unsupported keyword arguments.

    Gradio 4.x removed the ``scale`` argument from :class:`gr.Row`. Passing it
    now raises a ``TypeError`` similar to ``Row.__init__() got an unexpected
    keyword argument 'scale'``. This helper defensively strips the key, logs the
    cleanup for observability, and constructs the row with the remaining
    options. This approach keeps the layout code concise while staying
    compatible with the current library contract.
    """

    filtered_kwargs = _sanitize_row_kwargs(kwargs)

    logger.debug("Creating gr.Row with sanitized kwargs: %s", filtered_kwargs)

    try:
        return gr.Row(**filtered_kwargs)
    except TypeError as exc:  # pragma: no cover - defensive path
        logger.error(
            "Retrying gr.Row construction after filtering unsupported kwargs: %s", exc
        )
        safe_kwargs = {
            key: value for key, value in filtered_kwargs.items() if key in _ROW_SIGNATURE.parameters
        }
        return gr.Row(**safe_kwargs)


# ---------------------------------------------------------------------------
# UI assembly
# ---------------------------------------------------------------------------


def build_app() -> gr.Blocks:
    """Build the Gradio Blocks interface with a fixed sidebar and tabs."""

    logger.info("Initializing MQ-RAG dashboard UI")
    logger.debug(
        "Preparing initial state and CSS for UI construction; avoiding deprecated Row scaling"
    )

    initial_state: AppState = {"user": "admin", "session_id": "", "active_view": "search"}
    combined_css = CUSTOM_CSS_PATH.read_text() + "\n" + INLINE_CSS

    with gr.Blocks(
        css=combined_css,
        theme=gr.themes.Soft(),
        analytics_enabled=False,
        title="MQ-RAG Assistant",
    ) as demo:
        app_state = gr.State(initial_state)
        sidebar_visible = gr.State(True)
        active_view_state = gr.State(initial_state["active_view"])

        logger.debug(
            "Assembling layout rows and sidebar components with Gradio %s (Row scale unsupported)",
            gr.__version__,
        )
        with _create_row(
            variant="panel",
            elem_classes=["layout-row", "input-row"],
            equal_height=True,
        ):
            # Sidebar (collapsible)
            with gr.Column(
                scale=1,
                min_width=260,
                visible=True,
                elem_classes=["nav-rail", "panel", "sidebar-card"],
            ) as sidebar_column:
                gr.Markdown("## MQ-RAG\nOps Desk")
                new_chat_btn = gr.Button("➕ New Search", variant="primary", size="sm")
                gr.Markdown("**Role:** Admin\n\n**Environment:** Sandbox")

                gr.Markdown("### Navigation", elem_classes=["section-title"])
                with _create_row(elem_classes=["nav-buttons"]):
                    search_nav_btn = gr.Button("🧭 Search", size="sm")
                    docs_nav_btn = gr.Button("📂 Docs", size="sm")
                    help_nav_btn = gr.Button("❓ Help", size="sm")

                history_panel = gr.Markdown(
                    value="*No turns yet. Ask your first question to start a trail.*",
                    elem_classes=["history-panel"],
                )

                gr.Markdown(
                    "---\n**Tip:** Toggle the menu with the ☰ button if you need more space.",
                    elem_classes=["sidebar-footer"],
                )

            # Main content
            with gr.Column(scale=4, elem_classes=["main-area", "center-stage"]):
                with _create_row(justify="between", elem_classes=["header-row"]):
                    gr.Markdown("### MQ-RAG Assistant", elem_classes=["title-text"])
                    toggle_icon = gr.Button(
                        "☰",
                        size="sm",
                        min_width=50,
                        elem_classes=["toggle-icon"],
                        variant="secondary",
                    )

                with _create_row(elem_classes=["badge-row", "header-meta"]):
                    gr.Markdown("<span class='badge accent-badge'>Live</span>", elem_classes=["header-badge"])
                    gr.Markdown("<span class='badge light-badge'>Ops Dashboard</span>", elem_classes=["header-badge"])

                gr.Markdown(
                    "Welcome back, admin. Ask a question to search the knowledge base or review recent answers in the tabs below.",
                    elem_classes=["greeting-subtitle"],
                )

                with _create_row(elem_classes=["kpi-strip"]):
                    gr.Markdown("#### Sessions\n0 active", elem_classes=["kpi-card"])
                    gr.Markdown("#### Docs Indexed\n2 indexed", elem_classes=["kpi-card"])
                    gr.Markdown("#### Latency\n< 5s", elem_classes=["kpi-card"])
                    gr.Markdown("#### Model\nArctic", elem_classes=["kpi-card"])

                logger.debug("Binding input row without deprecated row scale arguments")
                with gr.Column(elem_classes=["view-panel"], visible=True) as search_panel:
                    gr.Markdown("### Search", elem_classes=["section-heading"])
                    with _create_row(elem_classes=["input-panel"], equal_height=True):
                        query_input = gr.Textbox(
                            lines=2,
                            placeholder="Ask the AI a question...",
                            container=False,
                            scale=4,
                            elem_id="query-input",
                        )
                        search_btn = gr.Button(
                            "Send",
                            variant="primary",
                            scale=1,
                            size="sm",
                            elem_id="submit-button",
                        )

                    with gr.Tabs(elem_classes=["result-tabs"]):
                        with gr.Tab("Answer"):
                            answer_panel = gr.Markdown(
                                label="Search Result",
                                value="The AI is ready. Enter your query below.",
                                elem_classes=["response-box"],
                            )

                        with gr.Tab("Sources"):
                            source_output = gr.Dataframe(
                                headers=["Source", "Snippet", "Relevance"],
                                datatype=["str", "str", "number"],
                                row_count=3,
                                col_count=(3, "fixed"),
                                interactive=False,
                                value=_default_sources(),
                            )

                        with gr.Tab("Logs"):
                            log_panel = gr.Markdown(DEFAULT_LOG_MESSAGE)

                        with gr.Tab("Raw"):
                            raw_panel = gr.Textbox(
                                value="",
                                label="Raw Response",
                                interactive=False,
                                lines=6,
                            )

                with gr.Column(visible=False, elem_classes=["view-panel", "docs-panel"]) as docs_panel:
                    gr.Markdown("### Manage Docs", elem_classes=["section-heading"])
                    gr.Markdown(
                        "Sync files, monitor ingestion, and review source freshness in one place.",
                        elem_classes=["section-subhead"],
                    )
                    with _create_row(equal_height=True, elem_classes=["docs-actions"]):
                        gr.File(label="Upload documents", file_types=[".pdf", ".txt"], type="filepath")
                        gr.Button("Trigger Ingestion", variant="primary", size="sm")
                    gr.Markdown(
                        "**Recent activity**\n- Intake pending\n- No warnings logged",
                        elem_classes=["placeholder-card"],
                    )

                with gr.Column(visible=False, elem_classes=["view-panel", "help-panel"]) as help_panel:
                    gr.Markdown("### Help", elem_classes=["section-heading"])
                    gr.Markdown(
                        "Common questions, runbooks, and contact options for the ops team.",
                        elem_classes=["section-subhead"],
                    )
                    gr.Markdown(
                        "**FAQ**\n- How do I open the UI? Open at http://localhost:7860.\n- How do I refresh the index? Use the Docs panel.\n- Where are logs stored? Check the Logs tab after a search.\n- Need support? File a ticket via the ops channel.",
                        elem_classes=["placeholder-card"],
                    )

        # Wiring
        search_inputs = [query_input, app_state]
        search_outputs = [
            answer_panel,
            source_output,
            log_panel,
            raw_panel,
            app_state,
            query_input,
            history_panel,
        ]

        search_btn.click(fn=handle_query, inputs=search_inputs, outputs=search_outputs)
        query_input.submit(fn=handle_query, inputs=search_inputs, outputs=search_outputs)

        new_chat_btn.click(
            fn=start_new_conversation,
            inputs=[app_state],
            outputs=search_outputs,
        )

        toggle_icon.click(
            fn=toggle_sidebar,
            inputs=[sidebar_visible],
            outputs=[sidebar_visible, sidebar_column],
        )

        search_nav_btn.click(
            fn=lambda current_view: switch_view("search", current_view),
            inputs=[active_view_state],
            outputs=[active_view_state, search_panel, docs_panel, help_panel],
        )
        docs_nav_btn.click(
            fn=lambda current_view: switch_view("docs", current_view),
            inputs=[active_view_state],
            outputs=[active_view_state, search_panel, docs_panel, help_panel],
        )
        help_nav_btn.click(
            fn=lambda current_view: switch_view("help", current_view),
            inputs=[active_view_state],
            outputs=[active_view_state, search_panel, docs_panel, help_panel],
        )

    return demo


app = build_app()
logger.info("Starting background model prewarm")
start_background_prewarm()

if __name__ == "__main__":
    launch_url = "http://localhost:7860"
    logger.info("Launching MQ-RAG app at %s", launch_url)
    app.launch(
        share=SHARE_INTERFACE,
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
    )

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
# Gradio compatibility shims
# ---------------------------------------------------------------------------

# Gradio 4.x removed the ``scale`` argument from :class:`gr.Row`. Older code
# paths (including legacy deployments) can still pass ``scale`` and raise a
# ``TypeError`` before the UI initializes. The shim below preserves verbose
# logging, strips the unsupported key for compatibility, and retains the
# original initializer for future debugging.
# Apply the shim only when the current Gradio version lacks ``scale`` support.
_ROW_SIGNATURE = inspect.signature(gr.Row.__init__)
_ORIGINAL_ROW_INIT = gr.Row.__init__


def _patched_row_init(self, *args: Any, **kwargs: Any) -> None:
    """Patch ``gr.Row.__init__`` to drop deprecated arguments safely."""

    if "scale" in kwargs and "scale" not in _ROW_SIGNATURE.parameters:
        kwargs = dict(kwargs)
        kwargs.pop("scale")
        logger.warning(
            "Removed deprecated 'scale' kwarg from gr.Row for compatibility: %s", kwargs
        )
    _ORIGINAL_ROW_INIT(self, *args, **kwargs)


if "scale" not in _ROW_SIGNATURE.parameters:
    gr.Row.__init__ = _patched_row_init
    logger.debug(
        "Patched gr.Row.__init__ to ignore deprecated 'scale' kwarg (Gradio %s)",
        gr.__version__,
    )
else:
    logger.debug(
        "gr.Row.__init__ already accepts 'scale'; compatibility patch not applied (Gradio %s)",
        gr.__version__,
    )


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

    filtered_kwargs = dict(kwargs)
    if "scale" in filtered_kwargs:
        logger.warning(
            "Removing unsupported 'scale' from gr.Row args: %s", filtered_kwargs
        )
        filtered_kwargs.pop("scale")
    return gr.Row(**filtered_kwargs)


# ---------------------------------------------------------------------------
# UI assembly
# ---------------------------------------------------------------------------


def build_app() -> gr.Blocks:
    """Build the Gradio Blocks interface with a fixed sidebar and tabs."""

    logger.info("Initializing MQ-RAG dashboard UI")
    logger.debug(
        "Preparing initial state and CSS for UI construction; avoiding deprecated Row scaling"
    )

    initial_state: AppState = {"user": "admin", "session_id": ""}
    combined_css = CUSTOM_CSS_PATH.read_text() + "\n" + INLINE_CSS

    with gr.Blocks(
        css=combined_css,
        theme=gr.themes.Soft(),
        analytics_enabled=False,
        title="MQ-RAG Assistant",
    ) as demo:
        app_state = gr.State(initial_state)
        sidebar_visible = gr.State(True)

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
                    gr.Button("🧭 Search", size="sm")
                    gr.Button("📂 Docs", size="sm")
                    gr.Button("❓ Help", size="sm")

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
                with _create_row(justify="between"):
                    gr.Markdown("### MQ-RAG Assistant")
                    toggle_icon = gr.Button(
                        "☰",
                        size="sm",
                        min_width=50,
                        elem_classes=["toggle-icon"],
                        variant="secondary",
                    )

                gr.Markdown(
                    "Welcome back, admin. Ask a question to search the knowledge base or review recent answers in the tabs below.",
                    elem_classes=["greeting-subtitle"],
                )

                with _create_row(elem_classes=["kpi-strip"]):
                    gr.Markdown("#### Sessions\n0 active")
                    gr.Markdown("#### Docs Indexed\n2 indexed")
                    gr.Markdown("#### Latency\n< 5s")
                    gr.Markdown("#### Model\nArctic")

                logger.debug("Binding input row without deprecated row scale arguments")
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

    return demo


app = build_app()
logger.info("Starting background model prewarm")
start_background_prewarm()

if __name__ == "__main__":
    app.launch(
        share=SHARE_INTERFACE,
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
    )

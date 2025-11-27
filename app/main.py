"""Gradio UI with fixed sidebar, KPI strip, and tabbed results view.

Clean, modern ops-dashboard style RAG interface with full compatibility
for Gradio ≥4.0 (scale parameter removed from Row).
"""

from __future__ import annotations

import inspect
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

import gradio as gr

from app.config import SHARE_INTERFACE
from app.rag_chain import query_rag, start_background_prewarm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# ---------------------------------------------------------------------------
# Runtime environment hardening (CPU-only ONNX)
# ---------------------------------------------------------------------------
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("ORT_DEVICE_ALLOWLIST", "cpu")
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")
logger.debug("GPU execution disabled; forcing CPU-only ONNX Runtime")


# ---------------------------------------------------------------------------
# Gradio 4.x compatibility: safe Row creator (replaces deprecated 'scale')
# ---------------------------------------------------------------------------
_ROW_SIGNATURE = inspect.signature(gr.Row.__init__)


def _sanitize_row_kwargs(raw_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Remove deprecated or unsupported kwargs from gr.Row calls."""
    cleaned = dict(raw_kwargs)
    if "scale" in cleaned:
        removed = cleaned.pop("scale")
        logger.debug("Removed deprecated 'scale=%s' from gr.Row()", removed)

    allowed = set(_ROW_SIGNATURE.parameters.keys()) - {"self"}
    unsupported = [k for k in cleaned.keys() if k not in allowed]
    for k in unsupported:
        cleaned.pop(k)
        logger.debug("Dropped unsupported gr.Row kwarg: %s", k)

    return cleaned


def _create_row(**kwargs: Any) -> gr.Row:
    """Create a gr.Row with Gradio 4+ compatibility (no 'scale' allowed)."""
    sanitized = _sanitize_row_kwargs(kwargs)
    return gr.Row(**sanitized)


# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------
@dataclass
class SessionRecord:
    session_id: str
    history: List[Tuple[str, str]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def render_history(self, limit: int = 6) -> str:
        if not self.history:
            return "*No turns yet. Ask your first question to start a trail.*"

        lines = ["### Recent Searches"]
        for idx, (query, _) in enumerate(self.history[-limit:], start=1):
            truncated = (query[:60] + "…") if len(query) > 60 else query
            lines.append(f"{idx}. {truncated}")
        return "\n".join(lines)


class AppState(TypedDict, total=False):
    user: str
    session_id: str
    active_view: str


SESSION_STORE: Dict[str, SessionRecord] = {}
DEFAULT_SOURCES: list[list[Any]] = [
    ["Knowledge Base", "Upload or sync documents to see citation details for each answer.", None],
]
DEFAULT_LOG_MESSAGE = "Awaiting first query."

ASSETS_DIR = Path(__file__).parent / "assets"
CUSTOM_CSS_PATH = ASSETS_DIR / "custom.css"


def _default_sources() -> list[list[Any]]:
    return [list(row) for row in DEFAULT_SOURCES]


def _get_or_create_session(username: str, session_id: str | None) -> SessionRecord:
    if session_id and session_id in SESSION_STORE:
        logger.debug("Reusing session %s for user %s", session_id, username)
        return SESSION_STORE[session_id]

    new_id = session_id or str(uuid.uuid4())
    record = SessionRecord(session_id=new_id)
    SESSION_STORE[new_id] = record
    logger.info("Created new session %s for user %s", new_id, username)
    return record


# ---------------------------------------------------------------------------
# Query handling
# ---------------------------------------------------------------------------
def _format_answer(query: str, answer: str) -> str:
    return f"**Query:** *{query}*\n\n**AI Result:** {answer}"


def handle_query(
    message: str, state: AppState
) -> Generator[Tuple[str, List[List[Any]], str, str, AppState, str, str], None, None]:
    username = state.get("user", "admin")
    session_id = state.get("session_id", "")
    record = _get_or_create_session(username, session_id)

    text = (message or "").strip()
    if not text:
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

    logger.info("Processing query: %s (session=%s)", text[:60], record.session_id)
    sidebar_md = record.render_history()

    # Immediate acknowledgment
    yield (
        "Thinking...",
        _default_sources(),
        "Query received...",
        text,
        state,
        message,
        sidebar_md,
    )

    # Actual RAG call
    answer = query_rag(text)
    record.history.append((text, answer))
    record.updated_at = datetime.utcnow()
    state["session_id"] = record.session_id

    formatted_answer = _format_answer(text, answer)
    sidebar_md = record.render_history()

    yield (
        formatted_answer,
        _default_sources(),
        "Completed without errors.",
        answer,
        state,
        "",
        sidebar_md,
    )


def start_new_conversation(state: AppState) -> Tuple[str, List[List[Any]], str, str, AppState, str, str]:
    logger.info("Starting new conversation for user %s", state.get("user", "admin"))
    state["session_id"] = ""
    greeting = "The AI is ready. Enter your query below."
    sidebar_md = SessionRecord(session_id="temp").render_history()
    return greeting, _default_sources(), DEFAULT_LOG_MESSAGE, "", state, "", sidebar_md


def toggle_sidebar(current: bool) -> Tuple[bool, gr.Column]:
    new_state = not current
    logger.debug("Sidebar toggled to %s", new_state)
    return new_state, gr.Column.update(visible=new_state)


def switch_view(target: str, current: str) -> Tuple[str, gr.Column, gr.Column, gr.Column]:
    target = (target or "search").lower()
    resolved = "search" if target not in {"search", "docs", "help"} else target
    logger.info("View switch: %s → %s", current, resolved)
    return (
        resolved,
        gr.Column.update(visible=resolved == "search"),
        gr.Column.update(visible=resolved == "docs"),
        gr.Column.update(visible=resolved == "help"),
    )


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
INLINE_CSS = """
.layout-row { align-items: stretch; gap: 16px !important; }
.sidebar-card { gap: 8px !important; }
.sidebar-footer { color: #7a6c5d; font-size: 13px; }
.kpi-strip { gap: 10px !important; display: flex; flex-wrap: wrap; }
.kpi-card { min-width: 150px; padding: 12px; background: #f8f9fa; border-radius: 8px; }
.result-tabs .gr-panel { background: transparent; border: none; box-shadow: none; }
.input-panel .gr-textbox { margin-top: 0; }
"""

# ---------------------------------------------------------------------------
# UI Assembly
# ---------------------------------------------------------------------------
def build_app() -> gr.Blocks:
    logger.info("Building MQ-RAG dashboard UI (Gradio 4+ compatible)")

    combined_css = (CUSTOM_CSS_PATH.read_text() if CUSTOM_CSS_PATH.exists() else "") + "\n" + INLINE_CSS

    with gr.Blocks(
        css=combined_css,
        theme=gr.themes.Soft(),
        analytics_enabled=False,
        title="MQ-RAG Assistant",
    ) as demo:
        app_state = gr.State({"user": "admin", "session_id": "", "active_view": "search"})
        sidebar_visible = gr.State(True)
        active_view_state = gr.State("search")

        # Main layout: Sidebar + Content
        with _create_row(variant="panel", elem_classes=["layout-row"], equal_height=True):
            # === SIDEBAR ===
            with gr.Column(
                min_width=260,
                visible=True,
                elem_classes=["nav-rail", "panel", "sidebar-card"],
            ) as sidebar_col:
                gr.Markdown("## MQ-RAG\nOps Desk")
                gr.Button("New Search", variant="primary", size="sm").click(
                    start_new_conversation, app_state, outputs=None
                )
                gr.Markdown("**Role:** Admin\n\n**Environment:** Sandbox")
                gr.Markdown("### Navigation", elem_classes=["section-title"])

                with _create_row(elem_classes=["nav-buttons"]):
                    gr.Button("Search", size="sm").click(
                        lambda cv: switch_view("search", cv),
                        active_view_state,
                        [active_view_state, None, None, None],
                    )
                    gr.Button("Docs", size="sm").click(
                        lambda cv: switch_view("docs", cv),
                        active_view_state,
                        [active_view_state, None, None, None],
                    )
                    gr.Button("Help", size="sm").click(
                        lambda cv: switch_view("help", cv),
                        active_view_state,
                        [active_view_state, None, None, None],
                    )

                history_panel = gr.Markdown("*No turns yet.*", elem_classes=["history-panel"])
                gr.Markdown(
                    "---\n**Tip:** Toggle menu with ☰ for more space.",
                    elem_classes=["sidebar-footer"],
                )

            # === MAIN CONTENT ===
            with gr.Column(elem_classes=["main-area", "center-stage"]):
                with _create_row(justify="between", elem_classes=["header-row"]):
                    gr.Markdown("### MQ-RAG Assistant", elem_classes=["title-text"])
                    gr.Button("Menu", size="sm", elem_classes=["toggle-icon"], variant="secondary").click(
                        toggle_sidebar, sidebar_visible, [sidebar_visible, sidebar_col]
                    )

                with _create_row(elem_classes=["badge-row", "header-meta"]):
                    gr.Markdown("<span class='badge accent-badge'>Live</span>")
                    gr.Markdown("<span class='badge light-badge'>Ops Dashboard</span>")

                gr.Markdown("Welcome back, admin. Ask a question to search the knowledge base.")

                # KPI Strip
                with _create_row(elem_classes=["kpi-strip"]):
                    gr.Markdown("#### Sessions\n0 active", elem_classes=["kpi-card"])
                    gr.Markdown("#### Docs Indexed\n2 indexed", elem_classes=["kpi-card"])
                    gr.Markdown("#### Latency\n< 5s", elem_classes=["kpi-card"])
                    gr.Markdown("#### Model\nArctic", elem_classes=["kpi-card"])

                # === VIEW PANELS ===
                with gr.Column(visible=True, elem_classes=["view-panel"]) as search_panel:
                    gr.Markdown("### Search", elem_classes=["section-heading"])
                    with _create_row(elem_classes=["input-panel"], equal_height=True):
                        query_input = gr.Textbox(
                            lines=2,
                            placeholder="Ask the AI a question...",
                            container=False,
                            elem_id="query-input",
                        )
                        search_btn = gr.Button("Send", variant="primary", size="sm")

                    with gr.Tabs(elem_classes=["result-tabs"]):
                        with gr.Tab("Answer"):
                            answer_panel = gr.Markdown("The AI is ready. Enter your query below.")
                        with gr.Tab("Sources"):
                            source_output = gr.Dataframe(
                                headers=["Source", "Snippet", "Relevance"],
                                datatype=["str", "str", "number"],
                                value=_default_sources(),
                                interactive=False,
                            )
                        with gr.Tab("Logs"):
                            log_panel = gr.Markdown(DEFAULT_LOG_MESSAGE)
                        with gr.Tab("Raw"):
                            raw_panel = gr.Textbox("", label="Raw Response", interactive=False, lines=6)

                # Docs Panel
                with gr.Column(visible=False, elem_classes=["view-panel", "docs-panel"]) as docs_panel:
                    gr.Markdown("### Manage Docs")
                    gr.Markdown("Sync files, monitor ingestion, and review source freshness.")
                    with _create_row(equal_height=True):
                        gr.File(label="Upload documents", file_types=[".pdf", ".txt"], type="filepath")
                        gr.Button("Trigger Ingestion", variant="primary", size="sm")
                    gr.Markdown("**Recent activity**\n- Intake pending\n- No warnings logged")

                # Help Panel
                with gr.Column(visible=False, elem_classes=["view-panel", "help-panel"]) as help_panel:
                    gr.Markdown("### Help")
                    gr.Markdown("Common questions and runbooks.")
                    gr.Markdown(
                        "**FAQ**\n- Open UI at http://localhost:7860\n- Refresh index in Docs tab\n- Check Logs tab after search"
                    )

        # === Event Wiring ===
        outputs = [answer_panel, source_output, log_panel, raw_panel, app_state, query_input, history_panel]

        search_btn.click(handle_query, [query_input, app_state], outputs)
        query_input.submit(handle_query, [query_input, app_state], outputs)

        # Navigation buttons (update visibility via switch_view)
        for btn, view in [("Search", "search"), ("Docs", "docs"), ("Help", "help")]:
            btn_obj = next(c for c in sidebar_col.children if isinstance(c, gr.Button) and c.value == view)
            btn_obj.click(
                lambda cv, v=view: switch_view(v, cv),
                active_view_state,
                [active_view_state, search_panel, docs_panel, help_panel],
            )

    return demo


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
app = build_app()
logger.info("Starting background model prewarm task")
start_background_prewarm()

if __name__ == "__main__":
    logger.info("Launching MQ-RAG Assistant at http://0.0.0.0:7860")
    app.launch(
        share=SHARE_INTERFACE,
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
        quiet=False,
    )
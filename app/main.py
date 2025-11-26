"""Modern ChatGPT-style Gradio interface for MQ-RAG.

The main application interface is intentionally compact and minimal: a
collapsible sidebar, tight control spacing, and a bottom-aligned input replicate
the efficient feel of ChatGPT while keeping sources tucked away in an optional
accordion. Verbose logging is used throughout to simplify troubleshooting and
to align with international programming documentation standards.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Tuple, TypedDict

import gradio as gr

from app.config import SHARE_INTERFACE
from app.rag_chain import query_rag, start_background_prewarm

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


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
    """Compose the display-friendly Markdown for the main response panel."""
    return f"**Query:** *{query}*\n\n**AI Result:** {answer}"


def handle_query(
    message: str,
    state: AppState,
) -> Tuple[str, List[List[Any]], AppState, str]:
    """Process a user query and format UI outputs for the sleek interface."""

    username = state.get("user", "rob")
    session_id = state.get("session_id") or ""
    record = _get_or_create_session(username, session_id)

    text = (message or "").strip()
    if not text:
        logger.debug(
            "Received empty query for user=%s session=%s; prompting for input",
            username,
            record.session_id,
        )
        return (
            "Please enter a question to begin.",
            _default_sources(),
            state,
            "",
        )

    logger.info("Handling query for user=%s session=%s", username, record.session_id)

    answer = query_rag(text)
    record.history.append((text, answer))
    record.updated_at = datetime.utcnow()
    state["session_id"] = record.session_id

    logger.debug(
        "Updated session %s with %d total turns", record.session_id, len(record.history)
    )

    formatted_answer = _format_answer(text, answer)
    return formatted_answer, _default_sources(), state, ""


def start_new_conversation(state: AppState) -> Tuple[str, List[List[Any]], AppState, str]:
    """Reset the chat for a fresh session while retaining the username."""

    logger.info("Starting a new conversation for user=%s", state.get("user", "rob"))
    state["session_id"] = ""
    greeting = "The AI is ready. Enter your query below."
    return greeting, _default_sources(), state, ""


def toggle_sidebar(current_visibility: bool) -> Tuple[bool, gr.update]:
    """Flip the sidebar visibility state and emit an updated column config."""

    new_state = not current_visibility
    logger.debug("Sidebar visibility toggled to %s", new_state)
    return new_state, gr.Column.update(visible=new_state)


# ---------------------------------------------------------------------------
# Minimalist CSS and layout helpers
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
.gradio-container { padding: 0 !important; }
.svelte-17w9n8 { padding: 8px !important; }

/* Sidebar styling */
.sidebar { background: #f9f9fb; border-right: 1px solid #e6e8ec; gap: 12px; }
.sidebar .gr-button { justify-content: flex-start; }
.sidebar .gr-button-primary { width: 100%; }
.sidebar .gr-markdown { margin: 0; }

/* Main area */
.main-area { padding: 12px 16px; gap: 12px; }
.response-box label { display: none; }
.response-box { min-height: 140px; }
.toggle-icon button { padding: 4px 10px; line-height: 1; font-size: 1.2em; }
.input-row { align-items: center; }

/* Input aesthetics */
.input-row .gr-textbox { margin-top: 0; }
"""


# ---------------------------------------------------------------------------
# UI assembly
# ---------------------------------------------------------------------------


def build_app() -> gr.Blocks:
    """Build the Gradio Blocks interface with a collapsible sidebar."""
    logger.info("Initializing sleek MQ-RAG UI")

    initial_state: AppState = {"user": "rob", "session_id": ""}

    with gr.Blocks(
        css=CUSTOM_CSS, theme=gr.themes.Soft(), analytics_enabled=False, title="AI Assistant"
    ) as demo:
        app_state = gr.State(initial_state)
        sidebar_visible = gr.State(True)

        with gr.Row(equal_height=True):
            # Sidebar
            with gr.Column(
                scale=1, min_width=250, visible=True, elem_classes=["sidebar"]
            ) as sidebar_column:
                new_chat_btn = gr.Button("➕ New Search", variant="primary", size="sm")
                gr.Markdown("### 📜 History")
                gr.Markdown("- *Recent MQ deep dive*")
                gr.Markdown("- *RAG pipeline overview*")
                gr.Markdown("- *Throughput telemetry*")
                gr.Markdown("---")
                gr.Markdown("### 🛠️ Tools")
                with gr.Row():
                    gr.Button("⚙️ Settings", size="sm", min_width=0)
                    gr.Button("💾 Export", size="sm", min_width=0)
                    gr.Button("👤 Account", size="sm", min_width=0)

            # Main content
            with gr.Column(scale=4, elem_classes=["main-area"]):
                toggle_icon = gr.Button(
                    "☰", size="sm", min_width=50, elem_classes=["toggle-icon"]
                )

                main_output = gr.Markdown(
                    label="Search Result",
                    value="The AI is ready. Enter your query below.",
                    elem_classes=["response-box"],
                )

                with gr.Accordion("📚 View Sources & Context", open=False):
                    source_output = gr.Dataframe(
                        headers=["Source", "Snippet", "Relevance"],
                        datatype=["str", "str", "number"],
                        row_count=3,
                        col_count=(3, "fixed"),
                        interactive=False,
                        value=_default_sources(),
                    )

                with gr.Row(variant="panel", elem_classes=["input-row"]):
                    query_input = gr.Textbox(
                        lines=1,
                        placeholder="Ask the AI a question...",
                        container=False,
                        scale=4,
                    )
                    search_btn = gr.Button("Send", variant="primary", scale=1, size="sm")

        # Wiring
        search_inputs = [query_input, app_state]
        search_outputs = [main_output, source_output, app_state, query_input]

        search_btn.click(fn=handle_query, inputs=search_inputs, outputs=search_outputs)
        query_input.submit(fn=handle_query, inputs=search_inputs, outputs=search_outputs)

        new_chat_btn.click(
            fn=start_new_conversation,
            inputs=[app_state],
            outputs=[main_output, source_output, app_state, query_input],
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

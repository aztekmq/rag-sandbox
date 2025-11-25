"""
Claude-style landing page for MQ-RAG.

This refactor makes the Gradio UI visually mimic the Claude “Good afternoon, rob”
screen, but branded as “MQ-RAG AI”. It intentionally avoids extra panels and
controls that are not visible in the reference screenshot.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple, TypedDict, Any

import gradio as gr

from app.config import SHARE_INTERFACE
from app.rag_chain import query_rag, start_background_prewarm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simple session store
# ---------------------------------------------------------------------------

@dataclass
class SessionRecord:
    session_id: str
    history: List[Tuple[str, str]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


class AppState(TypedDict):
    user: str
    session_id: str


SESSION_STORE: Dict[str, SessionRecord] = {}


def _get_or_create_session(username: str, session_id: str | None) -> SessionRecord:
    """Return an existing session or create a new one for this user."""
    if session_id and session_id in SESSION_STORE:
        return SESSION_STORE[session_id]

    new_id = session_id or str(uuid.uuid4())
    record = SessionRecord(session_id=new_id)
    SESSION_STORE[new_id] = record
    logger.info("Created new session %s for user %s", new_id, username)
    return record


def _time_based_greeting(username: str) -> str:
    """Return a time-of-day aware greeting, e.g. 'Good afternoon, rob'."""
    now = datetime.now().hour
    if 5 <= now < 12:
        salutation = "Good morning"
    elif 12 <= now < 18:
        salutation = "Good afternoon"
    elif 18 <= now < 22:
        salutation = "Good evening"
    else:
        salutation = "Good night"
    return f"{salutation}, {username}"


# ---------------------------------------------------------------------------
# Chat handler
# ---------------------------------------------------------------------------

def respond(
    message: str,
    history: List[List[str]] | None,
    state: AppState,
) -> Tuple[str, List[List[str]], AppState]:
    """
    Minimal chat handler: send prompt to query_rag, append answer to history,
    clear the input box, and keep session state.
    """
    username = state.get("user", "rob")
    session_id = state.get("session_id") or ""
    record = _get_or_create_session(username, session_id)

    text = (message or "").strip()
    if not text:
        return "", history or [], state

    logger.info("Handling query for user=%s session=%s", username, record.session_id)

    # Call your RAG backend
    answer = query_rag(text)

    # Normalize history to list[list[str]]
    history_pairs: List[List[str]] = [list(turn) for turn in (history or [])]
    history_pairs.append([text, answer])

    record.history = [(h[0], h[1]) for h in history_pairs]
    record.updated_at = datetime.utcnow()
    state["session_id"] = record.session_id

    # Return cleared textbox, updated chat, and updated state
    return "", history_pairs, state


# ---------------------------------------------------------------------------
# CSS to mimic the Claude new-chat screen
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
body, .gradio-container {
    background: #FCF7EF;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
}

/* Layout */
.main-layout {
    display: flex;
    height: 100vh;
    max-width: 100vw;
}

/* Sidebar */
.sidebar {
    width: 260px;
    padding: 24px 20px;
    box-sizing: border-box;
    border-right: 1px solid #F0E6D9;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.sidebar-title {
    font-size: 26px;
    font-weight: 600;
    margin-bottom: 24px;
}

.sidebar-nav button {
    justify-content: flex-start;
    width: 100%;
    border-radius: 999px;
    margin-bottom: 4px;
    font-weight: 500;
}

.sidebar-nav button.primary-nav {
    background: #F6EFE3;
    border: none;
}

.sidebar-nav button.secondary-nav {
    background: transparent;
    border: none;
}

.sidebar-footer {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 14px;
    margin-top: 32px;
}

.sidebar-avatar {
    width: 28px;
    height: 28px;
    border-radius: 999px;
    background: #3C3C3C;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
}

/* Main panel */
.main-panel {
    flex: 1;
    display: flex;
    justify-content: center;
    align-items: flex-start;
    padding-top: 80px;
}

.main-inner {
    width: 720px;
}

/* Top pill */
.plan-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 999px;
    background: white;
    border: 1px solid #EFE3D3;
    font-size: 13px;
    color: #7A6A55;
    margin-bottom: 40px;
}

/* Greeting */
.greeting-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 18px;
}

.greeting-icon {
    font-size: 26px;
}

.greeting-text {
    font-size: 36px;
    font-weight: 500;
    color: #3A3226;
}

/* Prompt card */
.prompt-card {
    margin-top: 20px;
    background: #FFF9F2;
    border-radius: 24px;
    box-shadow: 0 18px 60px rgba(0,0,0,0.04);
    padding: 18px 20px 14px 20px;
}

.prompt-row {
    display: flex;
    gap: 12px;
    align-items: stretch;
}

.hero-input textarea {
    border: none !important;
    box-shadow: none !important;
    resize: none;
    background: transparent;
    font-size: 16px;
    padding-left: 0;
    padding-right: 0;
}

.hero-input label {
    display: none;
}

/* Accessories underneath textbox */
.accessory-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 10px;
}

.accessory-left {
    display: flex;
    gap: 6px;
}

.icon-btn button {
    width: 28px;
    height: 28px;
    border-radius: 999px;
    border: none;
    background: transparent;
    font-size: 15px;
}

.model-select .wrap {
    border-radius: 999px !important;
    background: white !important;
    border: 1px solid #E7D9C7 !important;
    font-size: 13px;
}

.send-btn button {
    width: 38px;
    height: 38px;
    border-radius: 999px;
    border: none;
    background: #F3C3A0;
    font-size: 18px;
}

/* Bottom banner under card */
.tools-banner {
    font-size: 13px;
    color: #8B7A64;
    margin-top: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.tools-icons {
    display: flex;
    gap: 6px;
}

.tools-icon-pill {
    width: 26px;
    height: 18px;
    border-radius: 999px;
    background: #E1D8C9;
}

/* Chatbot area: keep invisible border to resemble empty space */
.chatbot-container {
    margin-top: 32px;
}

.chatbot-container .gr-chatbot {
    border: none;
    background: transparent;
}
"""


# ---------------------------------------------------------------------------
# UI assembly
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    logger.info("Initializing Claude-style MQ-RAG UI")

    initial_state: AppState = {"user": "rob", "session_id": ""}

    with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft(), analytics_enabled=False) as demo:
        app_state = gr.State(initial_state)

        with gr.Row(elem_classes=["main-layout"]):

            # ---------------- Sidebar ----------------
            with gr.Column(elem_classes=["sidebar"]):
                with gr.Column():
                    gr.Markdown("**MQ-RAG AI**", elem_classes=["sidebar-title"])
                    with gr.Column(elem_classes=["sidebar-nav"]):
                        gr.Button("New chat", elem_classes=["primary-nav"])
                        gr.Button("Chats", elem_classes=["secondary-nav"])
                        gr.Button("Projects", elem_classes=["secondary-nav"])
                        gr.Button("Artifacts", elem_classes=["secondary-nav"])

                with gr.Column(elem_classes=["sidebar-footer"]):
                    gr.HTML('<div class="sidebar-avatar">R</div>')
                    gr.Markdown("**rob**  \nFree plan")

            # ---------------- Main panel ----------------
            with gr.Column(elem_classes=["main-panel"]):
                with gr.Column(elem_classes=["main-inner"]):

                    # Plan pill
                    gr.HTML('<div class="plan-pill">Free plan · <span style="text-decoration: underline;">Upgrade</span></div>')

                    # Greeting
                    greeting = _time_based_greeting("rob")
                    with gr.Row(elem_classes=["greeting-row"]):
                        gr.Markdown("✺", elem_classes=["greeting-icon"])
                        gr.Markdown(greeting, elem_classes=["greeting-text"])

                    # Prompt card
                    with gr.Column(elem_classes=["prompt-card"]):

                        with gr.Row(elem_classes=["prompt-row"]):
                            # Left: textbox + accessories
                            with gr.Column():
                                hero_query = gr.Textbox(
                                    value="",
                                    placeholder="How can I help you today?",
                                    lines=2,
                                    label="",
                                    elem_classes=["hero-input"],
                                )

                                with gr.Row(elem_classes=["accessory-row"]):
                                    # left icons
                                    with gr.Row(elem_classes=["accessory-left"]):
                                        gr.Button("+", elem_classes=["icon-btn"], variant="secondary")
                                        gr.Button("≡", elem_classes=["icon-btn"], variant="secondary")
                                        gr.Button("🕓", elem_classes=["icon-btn"], variant="secondary")
                                    # (right section intentionally blank in screenshot)
                                    gr.Markdown("")

                            # Right: model dropdown + send button
                            with gr.Column():
                                model_select = gr.Dropdown(
                                    choices=["Sonnet 4.5"],
                                    value="Sonnet 4.5",
                                    label="",
                                    elem_classes=["model-select"],
                                )
                                send_btn = gr.Button("↑", elem_classes=["send-btn"])

                        # Under-card tools banner
                        with gr.Row(elem_classes=["tools-banner"]):
                            gr.Markdown("Upgrade to connect your tools to MQ-RAG AI")
                            with gr.Row(elem_classes=["tools-icons"]):
                                gr.HTML('<div class="tools-icon-pill"></div>')
                                gr.HTML('<div class="tools-icon-pill"></div>')

                    # Chat output (looks like empty area beneath card)
                    with gr.Column(elem_classes=["chatbot-container"]):
                        chatbot = gr.Chatbot(label="", bubble_full_width=False)

        # ---------------- Wiring ----------------

        hero_query.submit(
            respond,
            inputs=[hero_query, chatbot, app_state],
            outputs=[hero_query, chatbot, app_state],
        )
        send_btn.click(
            respond,
            inputs=[hero_query, chatbot, app_state],
            outputs=[hero_query, chatbot, app_state],
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
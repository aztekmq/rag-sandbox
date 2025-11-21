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

import logging
from pathlib import Path
from typing import Tuple

import gradio as gr
import gradio.blocks as gr_blocks
import gradio.routes as gr_routes

from app.auth import authenticate
from app.config import PDF_DIR, SHARE_INTERFACE
from app.rag_chain import delete_document, get_documents_list, ingest_pdfs, query_rag

logger = logging.getLogger(__name__)


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


def upload_and_index(files: list[gr.File] | None) -> str:
    """Persist uploaded PDFs and trigger re-indexing."""

    if files is None:
        return "No files uploaded"

    for file in files:
        destination = PDF_DIR / file.name.split("/")[-1]
        file.save(destination)
        logger.info("Saved PDF upload to %s", destination)

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
    logger.debug("Setting admin control visibility: %s", is_admin)
    overview_text = _render_library_overview(documents)
    return [
        gr.update(visible=is_admin),
        gr.update(visible=is_admin),
        gr.update(visible=is_admin),
        gr.update(visible=is_admin, choices=documents),
        gr.update(visible=is_admin),
        gr.update(visible=is_admin, value=overview_text),
    ]


def refresh_documents() -> list[str]:
    """Refresh the dropdown listing available documents."""

    docs = get_documents_list()
    logger.info("Document list refreshed with %d entries", len(docs))
    return docs


def handle_delete(doc: str) -> str:
    """Remove a document and its chunks from the vector store."""

    if not doc:
        return "Select a document to delete."
    return delete_document(doc)


def respond(message: str, history: list[list[str]], role_str: str):
    """Generate a response using the RAG engine."""

    logger.info("Received query from %s role: %s", role_str or "unknown", message)
    answer = query_rag(message)
    history.append((message, answer))
    return "", history


def logout(role_str: str) -> tuple[gr.update, gr.update, gr.update, str, gr.update, gr.update]:
    """Return to the landing page and clear transient state."""

    logger.info("Logout requested for role: %s", role_str or "unknown")
    cleared_chat = gr.update(value=[])
    cleared_docs = gr.update(choices=[], value=None)
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(value="", visible=False),
        "",
        cleared_chat,
        cleared_docs,
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
            role = gr.Radio(["user", "admin"], label="Login as", value="user", elem_classes=["pill-input"])
            user = gr.Textbox(label="Username", placeholder="your.name", elem_classes=["text-input"])
            pwd = gr.Textbox(label="Password", type="password", placeholder="••••••••", elem_classes=["text-input"])
            login_btn = gr.Button("Enter Workspace", elem_classes=["primary-btn"])

        status = gr.Markdown(visible=False, elem_classes=["status-bar"])

        with gr.Column(visible=False, elem_classes=["card", "workspace"], elem_id="workspace") as main_interface:
            with gr.Row(elem_classes=["workspace-header"]):
                gr.Markdown("### Workspace", elem_classes=["card-title", "no-margin"])
                user_badge = gr.Markdown("", elem_classes=["badge"])
                logout_btn = gr.Button("Logout", elem_classes=["ghost-btn"])

            with gr.Tabs(elem_classes=["tabset"]) as tabs:
                with gr.Tab("Chat Experience", elem_id="chat-tab"):
                    gr.Markdown(
                        "Engage with your IBM MQ knowledge base using a clean, ChatGPT-inspired conversational flow.",
                        elem_classes=["muted"],
                    )
                    chatbot = gr.Chatbot(height=540, bubble_full_width=False, layout="panel", elem_classes=["chatbot"])
                    with gr.Row():
                        msg = gr.Textbox(
                            label="Ask about IBM MQ",
                            placeholder="Ask a question or paste a log snippet...",
                            elem_classes=["text-input", "chat-input"],
                        )
                        clear = gr.Button("Clear", elem_classes=["ghost-btn"])

                with gr.Tab("Document Studio", visible=False, elem_id="admin-tab") as admin_tab:
                    gr.Markdown(
                        "Administer the localized corpus: review, ingest, or prune documents powering the RAG pipeline.",
                        elem_classes=["muted"],
                    )
                    with gr.Row():
                        doc_list = gr.Dropdown(choices=[], label="Active Documents", interactive=True, elem_classes=["text-input"])
                        refresh_btn = gr.Button("Refresh Library", elem_classes=["ghost-btn"])
                    library_overview = gr.Markdown(elem_classes=["status-bar"])
                    with gr.Row():
                        file_upload = gr.File(
                            label="Upload IBM MQ PDFs",
                            file_count="multiple",
                            file_types=[".pdf"],
                            interactive=True,
                        )
                        index_btn = gr.Button("Ingest PDFs", elem_classes=["primary-btn"])
                        delete_btn = gr.Button("Delete Selected", elem_classes=["danger-btn"])

            msg.submit(respond, [msg, chatbot, role_state], [msg, chatbot])
            clear.click(lambda: None, None, chatbot, queue=False)

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
        )

        refresh_btn.click(refresh_documents, None, doc_list)
        index_btn.click(upload_and_index, file_upload, library_overview)
        file_upload.change(refresh_documents, None, doc_list)
        delete_btn.click(handle_delete, doc_list, library_overview)
        logout_btn.click(
            logout,
            inputs=role_state,
            outputs=[login_box, main_interface, status, role_state, chatbot, doc_list],
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

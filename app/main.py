"""Gradio application exposing admin and user interfaces."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import gradio as gr

from app.auth import authenticate
from app.config import PDF_DIR
from app.rag_chain import delete_document, get_documents_list, ingest_pdfs, query_rag

logger = logging.getLogger(__name__)


def login(username: str, password: str, role: str) -> Tuple[gr.Update, gr.Update, str]:
    """Authenticate user and toggle interface visibility."""

    if authenticate(username, password, role):
        logger.info("User %s logged in as %s", username, role)
        return gr.update(visible=False), gr.update(visible=True, value=f"Logged in as {role.capitalize()}"), ""
    logger.warning("Failed login attempt for user %s", username)
    return gr.update(visible=True), gr.update(visible=False), "Wrong credentials!"


def upload_and_index(files: list[gr.File] | None) -> str:
    """Persist uploaded PDFs and trigger re-indexing."""

    if files is None:
        return "No files uploaded"

    for file in files:
        destination = PDF_DIR / file.name.split("/")[-1]
        file.save(destination)
        logger.info("Saved PDF upload to %s", destination)

    return ingest_pdfs()


def show_admin_controls(role_str: str):
    """Reveal admin-only widgets when applicable."""

    is_admin = "admin" in role_str.lower()
    logger.debug("Setting admin control visibility: %s", is_admin)
    return [gr.update(visible=is_admin) for _ in range(4)]


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

    logger.info("Received query: %s", message)
    answer = query_rag(message)
    history.append((message, answer))
    return "", history


ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def build_ui() -> gr.Blocks:
    """Build the Gradio Blocks interface."""

    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="IBM MQ RAG Sandbox",
        css=(ASSETS_DIR / "custom.css").read_text() if (ASSETS_DIR / "custom.css").exists() else None,
    ) as demo:
        gr.Markdown("# IBM MQ Knowledge Base (Local RAG)")

        with gr.Column(visible=True) as login_box:
            gr.Markdown("### Login")
            role = gr.Radio(["user", "admin"], label="Login as", value="user")
            user = gr.Textbox(label="Username")
            pwd = gr.Textbox(label="Password", type="password")
            login_btn = gr.Button("Login")

        status = gr.Markdown(visible=False)

        with gr.Column(visible=False) as main_interface:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Admin Panel")
                    file_upload = gr.File(label="Upload IBM MQ PDFs", file_count="multiple", visible=False)
                    index_btn = gr.Button("Re-index all PDFs", visible=False)
                    doc_list = gr.Dropdown(choices=[], label="Documents", visible=False)
                    delete_btn = gr.Button("Delete selected", visible=False)

                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(height=600)
                    msg = gr.Textbox(label="Ask about IBM MQ")
                    clear = gr.Button("Clear")

            msg.submit(respond, [msg, chatbot, status], [msg, chatbot])
            clear.click(lambda: None, None, chatbot, queue=False)

        login_btn.click(
            login,
            inputs=[user, pwd, role],
            outputs=[login_box, main_interface, status],
        ).then(
            show_admin_controls,
            inputs=status,
            outputs=[file_upload, index_btn, doc_list, delete_btn],
        )

        index_btn.click(upload_and_index, file_upload, gr.Textbox())
        file_upload.change(refresh_documents, None, doc_list)
        delete_btn.click(handle_delete, doc_list, gr.Textbox())

    return demo


def main() -> None:
    """Start the Gradio application."""

    logger.info("Launching Gradio app on 0.0.0.0:7860")
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()

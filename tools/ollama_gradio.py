"""Gradio chat client for a local Ollama instance.

This module spins up a lightweight Gradio interface that proxies prompts to a
locally running Ollama HTTP server (e.g., started via ``scripts/run_ollama_docker.sh``).
The design emphasizes verbose, structured logging so operators can debug
connectivity and model behaviors easily in line with international programming
standards.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from typing import Generator, Iterable, Tuple

import gradio as gr
import requests

LOGGER = logging.getLogger(__name__)
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3"


def configure_logging(log_level: str = "DEBUG") -> None:
    """Configure root logging with a verbose, developer-friendly formatter."""

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    LOGGER.debug("Logging initialized at level: %s", log_level)


@dataclass(frozen=True)
class OllamaConfig:
    """Encapsulate target Ollama settings in a structured, typed record."""

    base_url: str = DEFAULT_OLLAMA_URL
    model: str = DEFAULT_MODEL

    @property
    def generate_url(self) -> str:
        """Return the fully qualified URL for the Ollama generate endpoint."""

        return f"{self.base_url.rstrip('/')}/api/generate"


def _stream_response(response: requests.Response) -> Iterable[str]:
    """Stream decoded content from the Ollama response with verbose logging."""

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        LOGGER.debug("[OLLAMA_STREAM] raw line: %s", line)
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            LOGGER.warning("Received non-JSON line from Ollama: %s", line)
            continue

        if payload.get("error"):
            raise RuntimeError(payload["error"])

        content = payload.get("response")
        if content:
            yield content


def query_ollama(prompt: str, config: OllamaConfig) -> Generator[str, None, None]:
    """Send a prompt to Ollama and yield streaming responses for Gradio."""

    LOGGER.info("Dispatching prompt to Ollama | model=%s | url=%s", config.model, config.generate_url)
    with requests.post(
        config.generate_url,
        json={"model": config.model, "prompt": prompt, "stream": True},
        stream=True,
        timeout=300,
    ) as response:
        response.raise_for_status()
        yield from _stream_response(response)


def build_interface(config: OllamaConfig) -> gr.Blocks:
    """Create the Gradio Blocks UI that proxies chat messages to Ollama."""

    with gr.Blocks(title="Ollama Gradio Client", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # Local Ollama Chat (Debug-Friendly)
            - Model: `{model}` at `{url}`
            - Verbose logging is active; see terminal output for request/response traces.
            """.format(model=config.model, url=config.generate_url)
        )

        chatbot = gr.Chatbot(height=400, label="Ollama Chat Stream")
        textbox = gr.Textbox(label="Enter your prompt", placeholder="Ask anything...", lines=3)

        def respond(message: str, history: list[Tuple[str, str]]) -> Generator[str, None, None]:
            LOGGER.info("New chat message received with %d prior turns", len(history))
            accumulated = ""
            for chunk in query_ollama(message, config):
                accumulated += chunk
                yield accumulated

        submit = gr.Button("Send to Ollama", variant="primary")
        textbox.submit(respond, inputs=[textbox, chatbot], outputs=chatbot)
        submit.click(respond, inputs=[textbox, chatbot], outputs=chatbot)

    return demo


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for configuring the Gradio client."""

    parser = argparse.ArgumentParser(description="Start a Gradio UI for a local Ollama server.")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Base URL for the Ollama service (default: %(default)s)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model ID to request from Ollama (default: %(default)s)")
    parser.add_argument("--host", default="0.0.0.0", help="Host for the Gradio server (default: %(default)s)")
    parser.add_argument("--port", default=7861, type=int, help="Port for the Gradio server (default: %(default)s)")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link (off by default for local security)")
    parser.add_argument("--log-level", default="DEBUG", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    return parser.parse_args()


def main() -> None:
    """Entrypoint for starting the Gradio Blocks app with verbose logging."""

    args = parse_args()
    configure_logging(args.log_level)
    config = OllamaConfig(base_url=args.ollama_url, model=args.model)
    LOGGER.info("Starting Gradio client | ollama_url=%s | model=%s | host=%s | port=%s", args.ollama_url, args.model, args.host, args.port)
    interface = build_interface(config)
    interface.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

"""Debug-friendly Gradio chat client for a local Ollama instance.

This script demonstrates how to connect Gradio's ``ChatInterface`` to a
locally running Ollama server. It emphasizes verbose logging so operators can
trace HTTP requests and streamed responses. The code follows international
programming standards via type hints, docstrings, and explicit error
handling.
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


@dataclass(frozen=True)
class OllamaConfig:
    """Configuration for interacting with the Ollama HTTP API."""

    base_url: str = DEFAULT_OLLAMA_URL
    model: str = DEFAULT_MODEL

    @property
    def chat_url(self) -> str:
        """Return the fully qualified chat endpoint URL."""

        return f"{self.base_url.rstrip('/')}/api/chat"

    @property
    def tags_url(self) -> str:
        """Return the URL for checking available models."""

        return f"{self.base_url.rstrip('/')}/api/tags"


def configure_logging(log_level: str = "DEBUG") -> None:
    """Configure verbose logging to aid troubleshooting."""

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    LOGGER.debug("Logging initialized at level: %s", log_level)


def ensure_ollama_ready(config: OllamaConfig) -> None:
    """Validate that Ollama is reachable and advertises models.

    Raises:
        RuntimeError: if the Ollama service cannot be contacted.
    """

    LOGGER.info("Probing Ollama availability at %s", config.tags_url)
    try:
        response = requests.get(config.tags_url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network guard
        raise RuntimeError(
            "Could not reach Ollama. Ensure the server is running at "
            f"{config.base_url} (e.g., run 'ollama serve')."
        ) from exc

    payload = response.json()
    available_models = [entry.get("name", "<unknown>") for entry in payload.get("models", [])]
    LOGGER.info("Ollama responded with %d model(s): %s", len(available_models), ", ".join(available_models))


def _stream_chat_response(response: requests.Response) -> Iterable[str]:
    """Yield chat content chunks from a streaming Ollama response."""

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        LOGGER.debug("[OLLAMA_STREAM] raw line: %s", line)
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            LOGGER.warning("Received non-JSON line from Ollama: %s", line)
            continue

        if error_message := payload.get("error"):
            raise RuntimeError(error_message)

        message = payload.get("message", {})
        content = message.get("content")
        if content:
            yield content


def chat_with_ollama(message: str, history: list[Tuple[str, str]], config: OllamaConfig) -> Generator[str, None, None]:
    """Send a chat message to Ollama and stream back the response."""

    LOGGER.info("New user message received | history_count=%d", len(history))
    ensure_ollama_ready(config)

    LOGGER.info("Dispatching payload to Ollama | model=%s | url=%s", config.model, config.chat_url)
    with requests.post(
        config.chat_url,
        json={"model": config.model, "messages": [{"role": "user", "content": message}], "stream": True},
        stream=True,
        timeout=300,
    ) as resp:
        if resp.status_code == requests.codes.not_found:
            raise RuntimeError(
                "Ollama did not expose /api/chat (404). Update Ollama or switch to /api/generate."
            )
        resp.raise_for_status()
        accumulated = ""
        for chunk in _stream_chat_response(resp):
            accumulated += chunk
            yield accumulated


def build_interface(config: OllamaConfig) -> gr.Blocks:
    """Create the Gradio Blocks interface wired to the Ollama chat endpoint."""

    chatbot = gr.Chatbot(height=400, label="Ollama Chat Stream")
    textbox = gr.Textbox(label="Enter your prompt", placeholder="Ask anything...", lines=3)

    demo = gr.ChatInterface(
        fn=lambda message, history: chat_with_ollama(message, history, config),
        chatbot=chatbot,
        textbox=textbox,
        title="Local Ollama Chat (debug ready)",
        description=(
            f"Model: `{config.model}` at `{config.chat_url}`. Verbose logging is enabled to trace requests and streamed responses."
        ),
        retry_btn=None,
        undo_btn=None,
        clear_btn="Clear history",
    )
    return demo


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for configuring the Gradio demo."""

    parser = argparse.ArgumentParser(description="Run a debug-friendly Gradio UI against a local Ollama server.")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Base URL for the Ollama service (default: %(default)s)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model ID to request from Ollama (default: %(default)s)")
    parser.add_argument("--host", default="0.0.0.0", help="Host for the Gradio server (default: %(default)s)")
    parser.add_argument("--port", default=7861, type=int, help="Port for the Gradio server (default: %(default)s)")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link (disabled by default)")
    parser.add_argument("--log-level", default="DEBUG", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    return parser.parse_args()


def main() -> None:
    """Entrypoint for the demo with verbose logging and safety checks."""

    args = parse_args()
    configure_logging(args.log_level)
    config = OllamaConfig(base_url=args.ollama_url, model=args.model)

    LOGGER.info(
        "Starting Gradio demo | ollama_url=%s | model=%s | host=%s | port=%s", args.ollama_url, args.model, args.host, args.port
    )
    interface = build_interface(config)
    interface.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

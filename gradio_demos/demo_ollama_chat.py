"""Standalone Gradio chat interface that proxies to a local Ollama server.

This utility mirrors the Gradio ``ChatInterface`` examples from the official
Gradio documentation and adapts them for an Ollama deployment launched via the
``scripts/run_ollama_docker.sh`` helper. The implementation favors verbose
logging and explicit documentation to satisfy international programming
standards while keeping the codebase easy to audit.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, List

import gradio as gr
import requests

LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3"


@dataclass
class OllamaChatConfig:
    """Configuration block describing how to reach the Ollama API."""

    base_url: str
    model: str
    timeout: int = 120


def configure_logging(log_level: str) -> None:
    """Initialize console and file handlers with verbose formatting.

    Parameters
    ----------
    log_level:
        Desired verbosity level (e.g., ``DEBUG``). DEBUG is recommended to
        simplify troubleshooting during local development.
    """

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    formatter = logging.Formatter(LOG_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    log_path = Path("data/logs/ollama_chat.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    root_logger.debug(
        "Logging initialized | level=%s | file=%s | handlers=%d",
        log_level,
        log_path,
        len(root_logger.handlers),
    )


def _format_history_for_ollama(history: List[List[str]]) -> List[dict]:
    """Translate Gradio chat history into Ollama's chat message schema."""

    messages: List[dict] = []
    for user_message, assistant_message in history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": assistant_message})
    return messages


def stream_ollama_chat(
    message: str, history: List[List[str]], config: OllamaChatConfig
) -> Generator[str, None, None]:
    """Yield response tokens from Ollama in a streaming fashion.

    This generator mirrors the Gradio ChatInterface streaming pattern illustrated
    in the upstream documentation while delegating the actual completion work to
    the Ollama ``/api/chat`` endpoint.
    """

    logger = logging.getLogger("stream_ollama_chat")
    logger.info(
        "Forwarding chat message to Ollama | url=%s | model=%s | history_items=%d",
        config.base_url,
        config.model,
        len(history),
    )

    payload = {
        "model": config.model,
        "messages": _format_history_for_ollama(history)
        + [{"role": "user", "content": message}],
        "stream": True,
    }

    try:
        with requests.post(
            f"{config.base_url.rstrip('/')}/api/chat",
            json=payload,
            stream=True,
            timeout=config.timeout,
        ) as response:
            response.raise_for_status()
            logger.debug("Ollama connection established; streaming tokens")

            for line in response.iter_lines():
                if not line:
                    continue

                data = json.loads(line.decode("utf-8"))
                token = data.get("message", {}).get("content", "")

                if token:
                    logger.debug("Received token chunk: %s", token)
                    yield token
    except requests.RequestException as exc:  # pragma: no cover - network guard
        logger.exception("Streaming failure when contacting Ollama: %s", exc)
        yield f"[Error] Unable to contact Ollama at {config.base_url}: {exc}"


def build_interface(config: OllamaChatConfig) -> gr.Blocks:
    """Construct the Gradio Blocks layout with an embedded ChatInterface."""

    logger = logging.getLogger("build_interface")
    logger.debug("Building Gradio interface for Ollama chat")

    chatbot = gr.ChatInterface(
        lambda message, history: stream_ollama_chat(message, history, config),
        title="Ollama Chat Gateway",
        description=(
            "Stream messages to a locally running Ollama instance started via the "
            "`scripts/run_ollama_docker.sh` helper."
        ),
        analytics_enabled=False,
    )

    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # Local Ollama Chat

            This interface proxies user conversations to a Docker-hosted Ollama
            server. Ensure the container is running (for example via
            `scripts/run_ollama_docker.sh`) and that the configured URL is
            reachable from this host. Verbose logging is enabled and written to
            `data/logs/ollama_chat.log` for debugging.
            """
        )
        chatbot.render()
        gr.HTML("""<small>Verbose logging is enabled to simplify diagnostics.</small>""")

    logger.info("Gradio Blocks application created successfully")
    return demo


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments controlling the chat interface runtime."""

    parser = argparse.ArgumentParser(
        description=(
            "Launch a Gradio ChatInterface that streams completions from a local "
            "Ollama deployment."
        )
    )
    parser.add_argument(
        "--ollama-url",
        dest="ollama_url",
        default=DEFAULT_OLLAMA_URL,
        help="Base URL for the Ollama service (default: http://localhost:11434).",
    )
    parser.add_argument(
        "--ollama-model",
        dest="ollama_model",
        default=DEFAULT_OLLAMA_MODEL,
        help="Model identifier served by Ollama (default: llama3).",
    )
    parser.add_argument(
        "--host",
        dest="host",
        default="0.0.0.0",
        help="Host interface for the Gradio server (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        dest="port",
        type=int,
        default=7862,
        help="Port for the Gradio server (default: 7862 to avoid clashes).",
    )
    parser.add_argument(
        "--share",
        dest="share",
        action="store_true",
        help="Enable Gradio share links for remote access (disabled by default).",
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        default="DEBUG",
        help="Logging level; DEBUG recommended for verbose tracing.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    """Entrypoint for the Ollama ChatInterface launcher."""

    args = parse_args(argv)
    configure_logging(args.log_level)
    logger = logging.getLogger("ollama_chat_main")
    logger.debug("Launching Ollama chat with args: %s", args)

    config = OllamaChatConfig(base_url=args.ollama_url, model=args.ollama_model)
    logger.info(
        "Preparing Gradio server | host=%s | port=%s | model=%s | url=%s",
        args.host,
        args.port,
        config.model,
        config.base_url,
    )

    app = build_interface(config)
    app.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

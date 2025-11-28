#!/usr/bin/env bash
# run_ollama_stack.sh
#
# Purpose:
#   Convenience launcher that boots a local Ollama Docker container and then
#   starts the Gradio bridge (tools/ollama_gradio.py) against it. Verbose logging
#   is enabled throughout so you can troubleshoot connectivity quickly.
#
# Usage:
#   ./scripts/run_ollama_stack.sh
#   OLLAMA_MODEL=llama3 ./scripts/run_ollama_stack.sh
#
# Environment variables:
#   * OLLAMA_MODEL: model to request from Ollama (defaults to "llama3").
#   * OLLAMA_URL: base URL for the Ollama HTTP API (defaults to "http://localhost:11434").
#   * GRADIO_PORT: port for the Gradio UI (defaults to 7861).

set -euo pipefail
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR%/scripts}"

# Step 1: ensure the Ollama container is running locally.
"${SCRIPT_DIR}/run_ollama_docker.sh"

# Step 2: start the Gradio UI that streams prompts to Ollama.
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
GRADIO_PORT="${GRADIO_PORT:-7861}"

cd "${ROOT_DIR}"
python tools/ollama_gradio.py \
  --ollama-url "${OLLAMA_URL}" \
  --model "${OLLAMA_MODEL}" \
  --host 0.0.0.0 \
  --port "${GRADIO_PORT}" \
  --log-level DEBUG

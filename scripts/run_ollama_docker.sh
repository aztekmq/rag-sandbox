#!/usr/bin/env bash
# run_ollama_docker.sh
#
# Purpose:
#   Start (or resume) a local Ollama model-serving instance inside Docker,
#   ensure it is reachable on the host, and pull specified models so they
#   are ready for use by local RAG / Gradio apps.
#
# Usage:
#   ./scripts/run_ollama_docker.sh
#
#   # Override container/image/port/models if desired:
#   CONTAINER_NAME=custom-ollama \
#   IMAGE=ollama/ollama:latest \
#   PORT_MAPPING=11434:11434 \
#   MODELS="llama3.1 snowflake-arctic-embed snowflake-arctic" \
#   ./scripts/run_ollama_docker.sh

set -euo pipefail
set -x

# Resolve script & repo root dirs
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# If a project-wide .env exists in the repo root, load it
if [[ -f "${ROOT_DIR}/.env" ]]; then
  echo "[INFO] Loading environment from ${ROOT_DIR}/.env"
  # Export all vars while sourcing, then revert behavior
  set -a
  # shellcheck disable=SC1090
  source "${ROOT_DIR}/.env"
  set +a
else
  echo "[INFO] No .env found at ${ROOT_DIR}/.env; using built-in defaults."
fi

CONTAINER_NAME="${CONTAINER_NAME:-local-ollama}"
IMAGE="${IMAGE:-ollama/ollama:latest}"
PORT_MAPPING="${PORT_MAPPING:-11434:11434}"
VOLUME_NAME="${VOLUME_NAME:-ollama}"

# Models to ensure are present inside the container.
MODELS="${MODELS:-llama3.1}"

# Whether to send a tiny test request to warm chat models (non-embed) after pulling.
WARM_MODELS="${WARM_MODELS:-true}"

#######################################
# Helper: fail with message
#######################################
die() {
  echo "[ERROR] $*" >&2
  exit 1
}

#######################################
# Helper: wait until HTTP endpoint is ready
# Arguments:
#   $1 - host (e.g., localhost)
#   $2 - port (e.g., 11434)
#   $3 - timeout in seconds (optional, default 60)
#######################################
wait_for_http() {
  local host="${1}"
  local port="${2}"
  local timeout="${3:-60}"
  local start_ts
  start_ts="$(date +%s)"

  echo "[INFO] Waiting for Ollama HTTP API at http://${host}:${port} (timeout: ${timeout}s)..."

  while true; do
    if curl -sSf "http://${host}:${port}/api/tags" >/dev/null 2>&1; then
      echo "[INFO] Ollama HTTP API is responding on ${host}:${port}."
      break
    fi

    local now_ts
    now_ts="$(date +%s)"
    if (( now_ts - start_ts > timeout )); then
      die "Timed out waiting for Ollama HTTP API on ${host}:${port}"
    fi

    sleep 2
  done
}

#######################################
# Helper: pull models inside the container
#######################################
pull_models() {
  local container="$1"
  local models="$2"

  if ! docker exec "${container}" which ollama >/dev/null 2>&1; then
    die "Ollama CLI not found inside container '${container}'. Is the image correct?"
  fi

  for model in ${models}; do
    echo "[INFO] Pulling model '${model}' inside container '${container}'..."
    docker exec "${container}" ollama pull "${model}"
  done
}

#######################################
# Helper: warm chat models (non-embedding)
#######################################
warm_chat_models() {
  local host="$1"
  local port="$2"
  local models="$3"

  for model in ${models}; do
    if [[ "${model}" == *"embed"* ]]; then
      echo "[INFO] Skipping warm-up for embedding model '${model}'."
      continue
    fi

    echo "[INFO] Warming up chat model '${model}' via HTTP API..."
    curl -sS -X POST "http://${host}:${port}/api/chat" \
      -H "Content-Type: application/json" \
      -d '{
        "model": "'"${model}"'",
        "messages": [{"role": "user", "content": "ping"}],
        "stream": false
      }' >/dev/null 2>&1 || echo "[WARN] Warm-up request for '${model}' did not succeed (check logs if needed)."
  done
}

#######################################
# Main script
#######################################

if ! command -v docker >/dev/null 2>&1; then
  die "Docker CLI not found. Please install Docker before running this script."
fi

HOST_PORT="${PORT_MAPPING%%:*}"

if docker ps -a --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}$"; then
  echo "[INFO] Container '${CONTAINER_NAME}' already exists. Starting (if not running)..."
  docker start "${CONTAINER_NAME}" >/dev/null
else
  echo "[INFO] Creating and starting new Ollama container '${CONTAINER_NAME}'..."
  docker run \
    --detach \
    --name "${CONTAINER_NAME}" \
    --publish "${PORT_MAPPING}" \
    --volume "${VOLUME_NAME}:/root/.ollama" \
    --restart unless-stopped \
    "${IMAGE}"
fi

echo "[INFO] Ollama container is running as '${CONTAINER_NAME}' and listening on ${PORT_MAPPING}."
echo "[INFO] View logs with: docker logs -f ${CONTAINER_NAME}"

wait_for_http "localhost" "${HOST_PORT}" 90
pull_models "${CONTAINER_NAME}" "${MODELS}"

if [[ "${WARM_MODELS}" == "true" ]]; then
  warm_chat_models "localhost" "${HOST_PORT}" "${MODELS}"
else
  echo "[INFO] WARM_MODELS is set to 'false'; skipping warm-up requests."
fi

echo "[INFO] All requested models are pulled: ${MODELS}"
echo "[INFO] Ollama Docker instance '${CONTAINER_NAME}' is fully initialized."
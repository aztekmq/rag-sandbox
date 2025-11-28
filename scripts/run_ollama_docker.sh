#!/usr/bin/env bash
# run_ollama_docker.sh
#
# Purpose:
#   Start (or resume) a local Ollama model-serving instance inside Docker with verbose logging enabled.
#   The script favors debuggability and predictable behavior, following international scripting standards.
#
# Usage:
#   ./scripts/run_ollama_docker.sh
#   CONTAINER_NAME=custom-ollama IMAGE=ollama/ollama:latest ./scripts/run_ollama_docker.sh
#
# Notes:
#   * Uses `set -x` for verbose command tracing.
#   * Leaves the container running in detached mode so other scripts (e.g., Gradio) can connect to port 11434.
#   * Persists model data in the named Docker volume `ollama` by default.

set -euo pipefail
set -x

CONTAINER_NAME="${CONTAINER_NAME:-local-ollama}"
IMAGE="${IMAGE:-ollama/ollama:latest}"
PORT_MAPPING="${PORT_MAPPING:-11434:11434}"
VOLUME_NAME="${VOLUME_NAME:-ollama}"

# Ensure Docker is available before proceeding to reduce opaque failures.
if ! command -v docker >/dev/null 2>&1; then
  echo "[ERROR] Docker CLI not found. Please install Docker before running this script." >&2
  exit 1
fi

# Start or resume the Ollama container with explicit restart behavior and persistent volume.
if docker ps -a --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}$"; then
  docker start "${CONTAINER_NAME}"
else
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

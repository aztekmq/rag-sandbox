#!/usr/bin/env bash
# Entry point for local development builds following international scripting practices.
# Ensures Docker Compose builds run with verbose logging and an isolated credential-free
# configuration to avoid registry helper errors during the BuildKit bootstrap phase.

mkdir -p models data

set -euo pipefail
set -x

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMP_DOCKER_CONFIG="$(mktemp -d)"

# Proactively remove any stale mq-rag container to guarantee a clean restart while
# emitting clear diagnostic output for traceability.
remove_previous_instance() {
  if docker ps -a --format '{{.Names}}' | grep -q '^mq-rag$'; then
    docker rm -f mq-rag
  fi
}
remove_previous_instance

cleanup() {
  rm -rf "${TEMP_DOCKER_CONFIG}"
}
trap cleanup EXIT

export DOCKER_CONFIG="${TEMP_DOCKER_CONFIG}"
export DOCKER_BUILDKIT=1
export BUILDKIT_PROGRESS=plain

clear
cd "${PROJECT_ROOT}"
docker compose up -d --build

#!/usr/bin/env bash
# Local dev launcher:
# - strict bash mode + trace
# - isolated DOCKER_CONFIG to avoid credential helper noise
# - BuildKit with plain (verbose) progress
# - clean restart of mq-rag service/container

set -Eeuo pipefail
set -x

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEMP_DOCKER_CONFIG="$(mktemp -d)"

cleanup() {
  rm -rf -- "${TEMP_DOCKER_CONFIG}"
}
trap cleanup EXIT

export DOCKER_CONFIG="${TEMP_DOCKER_CONFIG}"
export DOCKER_BUILDKIT=1
export BUILDKIT_PROGRESS=plain

cd -- "${PROJECT_ROOT}"

# Ensure expected local dirs exist (relative to project root).
mkdir -p -- models data

# Stop/remove any previous instance cleanly (covers container + network bits).
# If the service doesn't exist yet, this is a no-op.
docker compose down --remove-orphans || true

# Optional: uncomment if you really want a clear screen each run.
# clear

docker compose up -d --build
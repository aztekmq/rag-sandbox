#!/usr/bin/env bash
# Local dev launcher - UPDATE mode (no destroy)
# - Rebuilds image if code/model/config changed
# - Recreates container without touching volumes or losing data
# - Keeps existing container running until the new one is healthy

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

# Ensure local directories exist
mkdir -p -- models data

# Optional: prune unused images to save space (remove this line if you don't want it)
# docker image prune -a -f

# This is the key change:
# --build         → rebuild image if Dockerfile/code changed
# --force-recreate → recreate the container even if config is identical
# --no-deps       → don't touch other services (we only have one anyway)
# -d              → run detached
# We DO NOT run "docker compose down" anymore

docker compose up -d --build --force-recreate --no-deps

echo "Update complete - mq-rag container has been recreated with the latest image."
echo "Access it at http://localhost:7860"
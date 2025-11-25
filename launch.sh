#!/usr/bin/env bash
# Local development launcher for the mq-rag Gradio web UI.
# Responsibilities (aligned with International Programming Standards):
# - enforce strict Bash safety options with verbose tracing for debuggability
# - isolate Docker client configuration to avoid credential helper noise
# - reuse existing mq-rag Docker assets when possible, pruning only unneeded ones
# - rebuild and relaunch the Gradio container with minimal disruption

set -Eeuo pipefail
set -x

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEMP_DOCKER_CONFIG="$(mktemp -d)"

log() {
  local level="${1}"
  shift
  printf '[%s] %s\n' "${level}" "$*"
}

cleanup() {
  log INFO "Removing temporary Docker config at ${TEMP_DOCKER_CONFIG}"
  rm -rf -- "${TEMP_DOCKER_CONFIG}"
}
trap cleanup EXIT

export DOCKER_CONFIG="${TEMP_DOCKER_CONFIG}"
export DOCKER_BUILDKIT=1
export BUILDKIT_PROGRESS=plain

cd -- "${PROJECT_ROOT}"

# Ensure expected local dirs exist (relative to project root).
log INFO "Ensuring required data directories exist"
mkdir -p -- models data

# Remove any stopped mq-rag containers to prevent name conflicts while preserving
# running services that may still be healthy.
stopped_containers=$(docker ps -aq --filter "name=^mq-rag$" --filter "status=exited" || true)
if [[ -n "${stopped_containers}" ]]; then
  log INFO "Cleaning up stopped mq-rag containers"
  docker rm --force ${stopped_containers} || true
fi

# Prune unused mq-rag images left behind by rebuilds without touching unrelated assets.
log INFO "Pruning dangling mq-rag images"
docker image prune --force --filter "label=com.docker.compose.project=rag-sandbox" --filter "dangling=true" || true

# Optional: uncomment if you really want a clear screen each run.
# clear

# Rebuild only what is necessary and (re)start the Gradio service.
log INFO "Launching mq-rag with minimal redeploy"
docker compose up -d --build rag-sandbox

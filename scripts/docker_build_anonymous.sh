#!/usr/bin/env bash
# Purpose: Build the project Docker image using an isolated Docker config to force anonymous pulls.
# This script enables verbose logging to aid debugging and aligns with international scripting standards.
set -euo pipefail
set -x

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMP_DOCKER_CONFIG="$(mktemp -d)"

cleanup() {
  rm -rf "${TEMP_DOCKER_CONFIG}"
}
trap cleanup EXIT

export DOCKER_CONFIG="${TEMP_DOCKER_CONFIG}"
export DOCKER_BUILDKIT=1

# Forward all user-provided arguments to docker build to preserve flexibility.
docker build "${PROJECT_ROOT}" "$@"

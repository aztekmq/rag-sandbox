#!/usr/bin/env bash
# International Programming Standards compliant helper to fetch the Snowflake Arctic
# embedding model with verbose logging. This script is idempotent and can be rerun
# safely; it downloads assets only when missing. Internet access is required
# unless you have already mirrored the repository to the target directory.

set -euo pipefail

# Emit verbose, timestamped logs for every significant action to simplify
# debugging in heterogeneous environments.
log() {
  printf '%s [INFO] %s\n' "$(date -Iseconds)" "$*"
}

error() {
  printf '%s [ERROR] %s\n' "$(date -Iseconds)" "$*" >&2
}

# Defaults align with the application configuration and can be overridden via
# environment variables for custom deployments.
: "${EMBEDDING_MODEL_ID:=Snowflake/snowflake-arctic-embed-xs}"
: "${EMBEDDING_MODEL_DIR:=data/models/snowflake-arctic-embed-xs}"
export EMBEDDING_MODEL_ID EMBEDDING_MODEL_DIR

# Validate dependencies early with explicit guidance.
if ! command -v python >/dev/null 2>&1; then
  error "Python is required to download embeddings. Install Python 3.11+ and rerun."
  exit 1
fi

if ! python - <<'PY' >/dev/null 2>&1
import importlib.util
import sys

module_missing = importlib.util.find_spec("huggingface_hub") is None
sys.exit(1 if module_missing else 0)
PY
then
  log "Python package 'huggingface_hub' is missing; installing with 'python -m pip install --upgrade huggingface-hub'."
  if ! python -m pip install --upgrade --quiet huggingface-hub; then
    error "Failed to install 'huggingface-hub'. Ensure internet connectivity and pip permissions, then rerun."
    exit 1
  fi
  log "Dependency 'huggingface_hub' installed successfully."
fi

log "Preparing to download embedding model '${EMBEDDING_MODEL_ID}' into '${EMBEDDING_MODEL_DIR}'"
mkdir -p "${EMBEDDING_MODEL_DIR}"

python - <<PY
from __future__ import annotations
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

model_id = Path(os.environ["EMBEDDING_MODEL_ID"])
model_dir = Path(os.environ["EMBEDDING_MODEL_DIR"])

print(f"Downloading embedding model '{model_id}' to '{model_dir}' with verbose logging...")
try:
    snapshot_download(
        repo_id=str(model_id),
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
        cache_dir=str(model_dir),
    )
except Exception as exc:  # noqa: BLE001
    print(
        "Failed to download embeddings. Verify internet connectivity, Hugging Face",
        "credentials (if required), and repository spelling. Error:",
        exc,
        sep="\n",
        file=sys.stderr,
    )
    sys.exit(1)
else:
    print("Embedding assets are ready under", model_dir)
PY

log "Embedding download complete. If this directory is mounted into the container, ingestion should proceed without errors."

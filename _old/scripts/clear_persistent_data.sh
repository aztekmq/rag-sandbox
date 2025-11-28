#!/usr/bin/env bash
# Purpose: Clear persistent vector data and ingested PDFs to provide a clean state
# without re-downloading language or embedding models. Verbose logging is
# enabled to simplify debugging in heterogeneous environments, in line with
# international scripting standards.

set -euo pipefail
set -x

log() {
  printf '%s [INFO] %s\n' "$(date -Iseconds)" "$*"
}

error() {
  printf '%s [ERROR] %s\n' "$(date -Iseconds)" "$*" >&2
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHROMA_DIR="${PROJECT_ROOT}/data/chroma_db"
PDF_DIR="${PROJECT_ROOT}/data/pdfs"

safe_reset_dir() {
  local target_dir="$1"
  local label="$2"

  if [[ "${target_dir}" == "/" || "${target_dir}" == "${PROJECT_ROOT}" ]]; then
    error "Refusing to clear critical directory '${target_dir}'. Update script configuration."
    exit 1
  fi

  if [[ -d "${target_dir}" ]]; then
    log "Clearing ${label} directory at '${target_dir}'."
    rm -rf "${target_dir}"
  else
    log "${label} directory not found; creating a fresh location at '${target_dir}'."
  fi

  mkdir -p "${target_dir}"
  log "${label} directory reset complete."
}

log "Starting persistent data reset using PROJECT_ROOT='${PROJECT_ROOT}'."
safe_reset_dir "${CHROMA_DIR}" "Chroma vector store"
safe_reset_dir "${PDF_DIR}" "PDF ingestion"
log "Persistent vector and PDF data have been cleared. Models remain untouched."

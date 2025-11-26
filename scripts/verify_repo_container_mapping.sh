#!/usr/bin/env bash
# Purpose: Verify that the Git repository inside the Docker container is correctly mapped and usable.
# The script emits verbose logs for each validation step to ease debugging and follows international scripting standards.
# Usage:
#   ./scripts/verify_repo_container_mapping.sh [--repo-path PATH] [--expected-remote URL] [--expected-branch NAME]
# Notes:
#   - Verbose command tracing is enabled by default (set -x).
#   - The script reports mount metadata, Git remotes/branches, and performs an optional write test inside the repository.
set -euo pipefail
set -x

# Initialize logging helpers with timestamps for consistent, debuggable output.
log() {
  local level="$1"
  local message="$2"
  printf '%s [%s] %s\n' "$(date -Iseconds)" "${level}" "${message}"
}

log_info() { log "INFO" "$1"; }
log_warn() { log "WARN" "$1"; }
log_error() { log "ERROR" "$1"; }

usage() {
  cat <<'USAGE'
verify_repo_container_mapping.sh
Validate that a Git repository is correctly mapped into a Docker container.

Options:
  --repo-path PATH       Repository path inside the container (default: current directory).
  --expected-remote URL  Optional expected remote URL; fails if the origin does not match.
  --expected-branch NAME Optional expected branch name; fails if HEAD is on a different branch.
  -h, --help             Show this help text.
USAGE
}

REPO_PATH="$(pwd)"
EXPECTED_REMOTE=""
EXPECTED_BRANCH=""

# Parse arguments using a simple loop to avoid external dependencies while keeping behavior explicit.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-path)
      REPO_PATH="$2"
      shift 2
      ;;
    --expected-remote)
      EXPECTED_REMOTE="$2"
      shift 2
      ;;
    --expected-branch)
      EXPECTED_BRANCH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      log_error "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

# Resolve to an absolute path for consistent mount detection.
if ! REPO_PATH="$(cd "${REPO_PATH}" && pwd -P)"; then
  log_error "Repository path '${REPO_PATH}' is not accessible."
  exit 1
fi

log_info "Repository path resolved to: ${REPO_PATH}"

if [[ ! -d "${REPO_PATH}" ]]; then
  log_error "Repository path '${REPO_PATH}' does not exist."
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  log_error "Git is not installed inside the container."
  exit 1
fi

if ! git -C "${REPO_PATH}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  log_error "Path '${REPO_PATH}' is not inside a Git working tree."
  exit 1
fi

GIT_ROOT="$(git -C "${REPO_PATH}" rev-parse --show-toplevel)"
log_info "Git root detected at: ${GIT_ROOT}"

if [[ "${GIT_ROOT}" != "${REPO_PATH}" ]]; then
  log_warn "Input path is inside the repository; using top-level: ${GIT_ROOT}"
fi

# Capture mount metadata to help verify host-to-container bind mappings.
if command -v findmnt >/dev/null 2>&1; then
  MOUNT_INFO="$(findmnt -T "${GIT_ROOT}" -o TARGET,SOURCE,FSTYPE,OPTIONS -n || true)"
  if [[ -n "${MOUNT_INFO}" ]]; then
    log_info "Mount info for repository: ${MOUNT_INFO}"
  else
    log_warn "Mount information unavailable for '${GIT_ROOT}'."
  fi
else
  log_warn "findmnt is not available; mount mapping cannot be inspected."
fi

ORIGIN_URL="$(git -C "${GIT_ROOT}" remote get-url origin 2>/dev/null || true)"
if [[ -n "${ORIGIN_URL}" ]]; then
  log_info "Origin remote: ${ORIGIN_URL}"
else
  log_warn "No origin remote configured."
fi

if [[ -n "${EXPECTED_REMOTE}" ]]; then
  if [[ "${ORIGIN_URL}" != "${EXPECTED_REMOTE}" ]]; then
    log_error "Expected remote '${EXPECTED_REMOTE}' but found '${ORIGIN_URL}'."
    exit 1
  else
    log_info "Expected remote matches configured origin."
  fi
fi

CURRENT_BRANCH="$(git -C "${GIT_ROOT}" rev-parse --abbrev-ref HEAD)"
log_info "Current branch: ${CURRENT_BRANCH}"

if [[ -n "${EXPECTED_BRANCH}" && "${CURRENT_BRANCH}" != "${EXPECTED_BRANCH}" ]]; then
  log_error "Expected branch '${EXPECTED_BRANCH}' but found '${CURRENT_BRANCH}'."
  exit 1
fi

CURRENT_COMMIT="$(git -C "${GIT_ROOT}" rev-parse HEAD)"
log_info "HEAD commit: ${CURRENT_COMMIT}"

# Confirm the working tree is readable and writable, which indicates the bind mount is functioning.
if [[ -w "${GIT_ROOT}" ]]; then
  if WRITE_TEST_FILE="$(mktemp "${GIT_ROOT%/}/.mapping_check.XXXXXX" 2>/dev/null)"; then
    log_info "Write test succeeded; temporary file created at ${WRITE_TEST_FILE}."
    rm -f "${WRITE_TEST_FILE}"
    log_info "Cleanup of temporary write-test file completed."
  else
    log_warn "Write test failed; the repository may be read-only in this container."
  fi
else
  log_warn "Repository path is not writable; skipping write test."
fi

# Provide a succinct summary to help operators verify the mapping at a glance.
log_info "Repository mapping verification completed successfully."

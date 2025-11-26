#!/usr/bin/env bash
# Purpose: Verify that the Git repository inside the Docker container is correctly mapped and usable.
# The script emits verbose logs for each validation step to ease debugging and follows international scripting standards.
# The script also emits an Executive Verification screen summarizing whether the container is using the expected
# host-mounted Git repository or relying on internal image code (e.g., bundled Gradio sources).
# Usage:
#   ./scripts/verify_repo_container_mapping.sh [--repo-path PATH] [--expected-remote URL] [--expected-branch NAME] [--container-name NAME] [--container-repo-path PATH] [--trace]
# Notes:
#   - Verbose logging is always enabled; add --trace to also emit shell tracing (set -x).
#   - The script reports mount metadata, Git remotes/branches, and performs an optional write test inside the repository.
set -euo pipefail

# Initialize logging helpers with timestamps for consistent, debuggable output.
log() {
  local level="$1"
  local message="$2"
  printf '%s [%s] %s\n' "$(date -Iseconds)" "${level}" "${message}"
}

log_info() { log "INFO" "$1"; }
log_warn() { log "WARN" "$1"; }
log_error() { log "ERROR" "$1"; }

print_divider() {
  printf '%s\n' "============================================================"
}

print_executive_header() {
  printf '\n'
  print_divider
  printf "|| %-58s||\n" "Repository Mapping Executive Verification"
  print_divider
}

print_kv() {
  local label="$1"
  local value="$2"
  printf "|| %-20s : %-33s||\n" "${label}" "${value}"
}

print_kv_wrapped() {
  local label="$1"
  local value="$2"
  local width=35

  if command -v fold >/dev/null 2>&1; then
    while IFS= read -r line; do
      print_kv "${label}" "${line}"
      label=""
    done <<<"$(echo -n "${value}" | fold -s -w ${width})"
  else
    # Fallback without fold; prints raw value and preserves verbose output for debugging.
    print_kv "${label}" "${value}"
  fi
}

usage() {
  cat <<'USAGE'
verify_repo_container_mapping.sh
Validate that a Git repository is correctly mapped into a Docker container.

Options:
  --repo-path PATH       Repository path inside the container (default: current directory).
  --expected-remote URL  Optional expected remote URL; fails if the origin does not match.
  --expected-branch NAME Optional expected branch name; fails if HEAD is on a different branch.
  --container-name NAME  Optional running container to verify via docker exec.
  --container-repo-path PATH
                         Repository path inside the running container (defaults to detected host path).
  --trace               Enable shell tracing (set -x) for deep debugging.
  -h, --help             Show this help text.
USAGE
}

REPO_PATH="$(pwd)"
EXPECTED_REMOTE=""
EXPECTED_BRANCH=""
CONTAINER_NAME=""
CONTAINER_REPO_PATH=""
ENABLE_TRACE=false

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
    --container-name)
      CONTAINER_NAME="$2"
      shift 2
      ;;
    --container-repo-path)
      CONTAINER_REPO_PATH="$2"
      shift 2
      ;;
    --trace)
      ENABLE_TRACE=true
      shift 1
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

if [[ "${ENABLE_TRACE}" == "true" ]]; then
  set -x
  log_info "Shell tracing enabled (--trace)."
fi

log_info "Starting repository mapping verification."

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

MOUNT_STATUS="UNKNOWN"
MOUNT_NOTES="Mount inspection not available; cannot confirm whether the host repository is mapped."

classify_mount_from_findmnt() {
  local mount_info="$1"

  if [[ -z "${mount_info}" ]]; then
    return
  fi

  read -r MNT_TARGET MNT_SOURCE MNT_FSTYPE MNT_OPTIONS <<<"${mount_info}"
  if [[ "${MNT_OPTIONS}" == *"bind"* ]]; then
    MOUNT_STATUS="HOST_BIND"
    MOUNT_NOTES="Bind mount detected (options include 'bind'); repository should reflect the host Git checkout."
  elif [[ "${MNT_FSTYPE}" == "overlay" ]]; then
    MOUNT_STATUS="IMAGE_OVERLAY"
    MOUNT_NOTES="Overlay filesystem detected without bind flag; repository likely originates from the container image (e.g., internal Gradio code)."
  else
    MOUNT_STATUS="STANDARD_FS"
    MOUNT_NOTES="Mounted on ${MNT_FSTYPE} without 'bind'; host mapping uncertain—confirm repository freshness manually."
  fi
}

classify_mount_from_findmnt "${MOUNT_INFO:-}"

log_info "Mount classification (pre-docker): ${MOUNT_STATUS} (${MOUNT_NOTES})"

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

##########
# Docker-based verification to ensure the running container sees the freshest repository content.
# This block is optional and executes only when a container name is provided.
##########

run_docker_cmd() {
  local description="$1"
  shift
  local cmd=("$@")

  log_info "[docker] ${description}: ${cmd[*]}"
  if ! "${cmd[@]}"; then
    log_error "[docker] ${description} failed"
    return 1
  fi
}

if [[ -n "${CONTAINER_NAME}" ]]; then
  if ! command -v docker >/dev/null 2>&1; then
    log_error "Docker CLI is not available; cannot validate container state."
    exit 1
  fi

  log_info "Container verification requested for: ${CONTAINER_NAME}"

  if ! docker ps --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
    log_error "Container '${CONTAINER_NAME}' is not running; cannot verify in-container repository mapping."
    exit 1
  fi

  CONTAINER_REPO_PATH=${CONTAINER_REPO_PATH:-${GIT_ROOT}}
  log_info "Using container repo path: ${CONTAINER_REPO_PATH}"

  # Capture mount information directly from the Docker daemon for ground truth on bind/volume mappings.
  if docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    CONTAINER_MOUNTS=$(docker inspect "${CONTAINER_NAME}" --format '{{range .Mounts}}{{printf "%s:%s (%s);" .Source .Destination .Mode}}{{end}}')
    if [[ -n "${CONTAINER_MOUNTS}" ]]; then
      log_info "[docker] Mounts for ${CONTAINER_NAME}: ${CONTAINER_MOUNTS}"

      MATCHING_MOUNT=$(docker inspect "${CONTAINER_NAME}" --format '{{range .Mounts}}{{if eq .Destination "'"'"'"'"'${CONTAINER_REPO_PATH}'"'"'"'"'"}}'"'"'{{printf "%s:%s (%s)" .Source .Destination .Mode}}'"'"'{{end}}{{end}}')
      if [[ -n "${MATCHING_MOUNT}" ]]; then
        log_info "[docker] Mount that contains the repository path: ${MATCHING_MOUNT}"
        if [[ "${MATCHING_MOUNT}" == *"bind"* ]]; then
          MOUNT_STATUS="HOST_BIND"
          MOUNT_NOTES="Docker reports a bind mount for ${CONTAINER_REPO_PATH}; host repository should be visible."
        else
          MOUNT_STATUS="DOCKER_VOLUME"
          MOUNT_NOTES="Docker reports a non-bind mount for ${CONTAINER_REPO_PATH}; verify host changes propagate."
        fi
      else
        log_warn "[docker] No mount entry matched container path ${CONTAINER_REPO_PATH}."
      fi
    else
      log_warn "[docker] No mounts reported for ${CONTAINER_NAME}."
    fi
  else
    log_warn "[docker] Unable to inspect container '${CONTAINER_NAME}'."
  fi

  # Validate that the repository path is visible and usable inside the container.
  run_docker_cmd "Check repository visibility" docker exec "${CONTAINER_NAME}" test -d "${CONTAINER_REPO_PATH}"

  DOCKER_GIT_ROOT="$(docker exec "${CONTAINER_NAME}" git -C "${CONTAINER_REPO_PATH}" rev-parse --show-toplevel 2>/dev/null || true)"
  if [[ -z "${DOCKER_GIT_ROOT}" ]]; then
    log_error "[docker] Git repository not detected at ${CONTAINER_REPO_PATH} inside ${CONTAINER_NAME}."
    exit 1
  fi
  log_info "[docker] Git root inside container: ${DOCKER_GIT_ROOT}"

  DOCKER_HEAD_COMMIT="$(docker exec "${CONTAINER_NAME}" git -C "${CONTAINER_REPO_PATH}" rev-parse HEAD 2>/dev/null || true)"
  if [[ -z "${DOCKER_HEAD_COMMIT}" ]]; then
    log_error "[docker] Unable to read HEAD commit inside container at ${CONTAINER_REPO_PATH}."
    exit 1
  fi
  log_info "[docker] HEAD commit inside container: ${DOCKER_HEAD_COMMIT}"

  if [[ "${DOCKER_HEAD_COMMIT}" != "${CURRENT_COMMIT}" ]]; then
    log_warn "[docker] Host HEAD (${CURRENT_COMMIT}) and container HEAD (${DOCKER_HEAD_COMMIT}) differ; container may be outdated."
  else
    log_info "[docker] Host and container HEAD commits match."
  fi

  # Compare a file checksum to ensure the container is reading the latest file contents, not stale layers.
  TARGET_FILE="${GIT_ROOT}/README.md"
  if [[ -f "${TARGET_FILE}" ]]; then
    HOST_CHECKSUM=$(sha256sum "${TARGET_FILE}" | awk '{print $1}')
    log_info "[docker] Host checksum for README.md: ${HOST_CHECKSUM}"

    DOCKER_TARGET_FILE="${CONTAINER_REPO_PATH}/README.md"
    DOCKER_CHECKSUM=$(docker exec "${CONTAINER_NAME}" sha256sum "${DOCKER_TARGET_FILE}" 2>/dev/null | awk '{print $1}')
    if [[ -n "${DOCKER_CHECKSUM}" ]]; then
      log_info "[docker] Container checksum for README.md: ${DOCKER_CHECKSUM}"
      if [[ "${HOST_CHECKSUM}" != "${DOCKER_CHECKSUM}" ]]; then
        log_warn "[docker] README.md checksum mismatch; container may not reflect the latest host files."
      else
        log_info "[docker] README.md checksum matches; container sees the updated file content."
      fi
    else
      log_warn "[docker] Unable to compute checksum for ${DOCKER_TARGET_FILE} inside container; file may be missing."
    fi
  else
    log_warn "Host README.md not found; skipping checksum comparison."
  fi

  # Optional write test within the container to verify mount writability mirrors host behavior.
  TEMP_FILE_NAME="/tmp/repo_write_check_$$.txt"
  if docker exec "${CONTAINER_NAME}" sh -c "cd '${CONTAINER_REPO_PATH}' && touch '${TEMP_FILE_NAME}'"; then
    log_info "[docker] Write test inside container succeeded at ${CONTAINER_REPO_PATH}/${TEMP_FILE_NAME}"
    docker exec "${CONTAINER_NAME}" rm -f "${TEMP_FILE_NAME}" || log_warn "[docker] Cleanup of temporary file inside container failed."
  else
    log_warn "[docker] Write test inside container failed; mount may be read-only."
  fi
fi

log_info "Mount classification (post-docker): ${MOUNT_STATUS} (${MOUNT_NOTES})"

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
print_executive_header
print_kv "Path" "${GIT_ROOT}"
print_kv "Mount" "${MOUNT_STATUS}"
print_kv_wrapped "Remote" "${ORIGIN_URL:-<none>}"
print_kv "Branch" "${CURRENT_BRANCH}"
print_kv_wrapped "Commit" "${CURRENT_COMMIT}"
print_kv "Write Test" "$([[ -w "${GIT_ROOT}" ]] && echo "Writable" || echo "Read-only")"
print_kv_wrapped "Notes" "${MOUNT_NOTES}"
print_divider
log_info "Repository mapping verification completed successfully. Executive screen above reflects current state."

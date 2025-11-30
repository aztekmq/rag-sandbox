#!/bin/bash

# --- Default Configuration ---
DEFAULT_OLLAMA_HOST="http://localhost:11434"
ENDPOINT="/api/tags"

# If environment variable is set, use it. Otherwise, use default.
OLLAMA_HOST="${OLLAMA_HOST:-$DEFAULT_OLLAMA_HOST}"

# --- Parse Command-Line Arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        -H|--host)
            # Normalize: allow host:port, or full http://host:port
            if [[ "$2" == http* ]]; then
                OLLAMA_HOST="$2"
            else
                OLLAMA_HOST="http://$2"
            fi
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--host <URL-or-host:port>]"
            echo
            echo "Examples:"
            echo "  $0"
            echo "  $0 --host http://127.0.0.1:11434"
            echo "  OLLAMA_HOST=http://10.0.0.20:11434 $0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Run with --help for usage."
            exit 1
            ;;
    esac
done


# --- Script Functions ---
check_dependency() {
    if ! command -v "$1" &> /dev/null
    then
        echo "Error: Required command '$1' is not installed." >&2
        echo "Please install it (e.g., 'sudo apt install $1' or 'brew install $1')." >&2
        exit 1
    fi
}

# --- Main Execution ---

# 1. Check dependencies
check_dependency "curl"
check_dependency "jq"

echo "🔎 Querying Ollama instance at ${OLLAMA_HOST}..."
echo "---"

# 2. Get JSON from Ollama
MODEL_JSON=$(curl -sS "${OLLAMA_HOST}${ENDPOINT}")
CURL_STATUS=$?

if [ $CURL_STATUS -ne 0 ]; then
    echo "❌ Error: Could not connect to Ollama server." >&2
    echo "   Make sure it's running at ${OLLAMA_HOST}." >&2
    exit 1
fi

# 3. Extract models
MODEL_NAMES=$(echo "${MODEL_JSON}" | jq -r '.models[].name')

if [ -z "$MODEL_NAMES" ]; then
    echo "⚠️ No models found."
else
    echo "✅ Available Models:"
    echo "${MODEL_NAMES}" | while read -r model; do
        echo "- $model"
    done
fi

echo "---"
#!/bin/bash

# --- Configuration ---
OLLAMA_HOST="http://localhost:11434"
ENDPOINT="/api/tags"

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

# 1. Check for required tools
check_dependency "curl"
check_dependency "jq"

echo "🔎 Querying Ollama instance at ${OLLAMA_HOST}..."
echo "---"

# 2. Execute curl and pipe the JSON output to jq
# -s: Silent mode (suppress progress meter)
# -S: Show error only if silent is used
MODEL_JSON=$(curl -sS "${OLLAMA_HOST}${ENDPOINT}")

# Check if the curl command was successful
if [ $? -ne 0 ]; then
    echo "❌ Error: Could not connect to Ollama server or retrieve data." >&2
    echo "   Ensure Ollama is running and accessible at ${OLLAMA_HOST}." >&2
    exit 1
fi

# 3. Use jq to parse the JSON and extract the model names
# .models[]: Selects every item in the "models" array.
# .name: Extracts the value of the "name" key from each item.
# -r: Raw output (removes surrounding quotes).
MODEL_NAMES=$(echo "${MODEL_JSON}" | jq -r '.models[].name')

if [ -z "$MODEL_NAMES" ]; then
    echo "⚠️ No models found in the Ollama instance."
else
    echo "✅ Found Available Models:"
    echo "${MODEL_NAMES}" | while read -r model; do
        echo "- $model"
    done
fi

echo "---"
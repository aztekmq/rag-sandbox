#!/usr/bin/env bash
set -Eeuo pipefail
set -x

cd "$(dirname "${BASH_SOURCE[0]}")"

# Just restart the container — your code is already mounted live
docker compose restart rag-sandbox || docker compose up -d

echo "Hot reload complete! Changes are live."
echo "http://localhost:7860"
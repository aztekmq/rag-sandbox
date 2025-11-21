#!/usr/bin/env bash
# Entry point for local development builds following international scripting practices.
# Clears the terminal for readability and launches the stack with rebuild enabled.

set -euo pipefail

clear
docker compose up -d --build

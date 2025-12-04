# RAG Sandbox (IBM MQ)

A single-container, production-ready Retrieval Augmented Generation (RAG) sandbox for IBM MQ documentation. The stack is tuned for 2025 best practices and runs fully offline after model downloads, with verbose logging throughout so activity is always auditable.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quickstart](#quickstart)
- [Launching with Verbose Logging](#launching-with-verbose-logging)
- [Embedding Options](#embedding-options)
- [Operations](#operations)
- [Logging and Observability](#logging-and-observability)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Security](#security)

## Overview
Retrieval Augmented Generation keeps answers grounded in real documents:
1. **Retrieval** – the system searches IBM MQ PDFs to pull relevant passages.
2. **Generation** – the language model summarizes those passages into a clear response.

## Features
- **Admin mode**: upload IBM MQ PDFs, re-index, delete documents, and inspect verbose logs.
- **User mode**: simplified chat-only interface without upload rights.
- **Offline-first**: llama.cpp inference plus local embeddings keep data private and deterministic.
- **Persistent storage**: PDFs, Chroma DB, and logs survive container restarts via mounted volumes.
- **Audit-friendly logging**: DEBUG-level output by default to stdout and `data/logs/app.log` for reproducibility.

## Architecture
- **Language model**: `Meta-Llama-3.1-8B-Instruct.Q5_K_M.gguf` served by `llama.cpp` (override with `MODEL_PATH`).
- **Embedding model**: `Snowflake/snowflake-arctic-embed-xs` loaded locally via `sentence-transformers` (configurable through `EMBEDDING_MODEL_ID` and `EMBEDDING_MODEL_DIR`).
- **Vector store**: Chroma persistent client under `data/chroma_db`.
- **PDF ingestion**: `pypdfium2` text extraction with 1,000-character chunks and 200-character overlap.
- **Interface**: Gradio 4 WebUI running entirely in the container.
- **Prewarm thread**: loads embeddings and llama.cpp on startup and generates a single token to prime caches, cutting first-response latency while emitting detailed logs.

## Prerequisites
- Docker and Docker Compose.
- Enough CPU/RAM to run the 8B model (align `MODEL_THREADS` with physical cores).
- Local folders to mirror container mounts: `models/` for GGUF weights and `data/` for embeddings, PDFs, vector store, and logs.

## Quickstart
1. **Clone and enter the repo**
   ```bash
   git clone https://github.com/yourname/rag-sandbox.git
   cd rag-sandbox
   ```

2. **Create persistence folders**
   ```bash
   mkdir -p models data
   ```

3. **Download the llama.cpp model (one-time)**
   ```bash
   ./download_llm.sh
   ```
   The script stores `Meta-Llama-3.1-8B-Instruct.Q5_K_M.gguf` in `models/` with verbose progress logging.

4. **(Optional) Pre-download embeddings for offline use**
   ```bash
   ./scripts/download_embedding.sh
   ```
   Assets land in `data/models/snowflake-arctic-embed-xs` so embedding calls never leave the host.

5. **Create `.env` for credentials and paths**
   ```env
   ADMIN_USERNAME=admin
   ADMIN_PASSWORD=change_me_strong_password
   USER_USERNAME=user
   USER_PASSWORD=mquser2025
   MODEL_PATH=/app/models/Meta-Llama-3.1-8B-Instruct.Q5_K_M.gguf
   ALLOW_HF_INTERNET=false
   EMBEDDING_MODEL_ID=Snowflake/snowflake-arctic-embed-xs
   EMBEDDING_MODEL_DIR=/app/data/models/snowflake-arctic-embed-xs
   LOG_LEVEL=DEBUG
   SHARE_INTERFACE=false
   ```
   `MODEL_PATH` must reference the exact GGUF file inside the container. Confirm with `docker exec -it mq-rag ls -l /app/models` if unsure.

## Launching with Verbose Logging
Run the Compose helper once setup is complete:
```bash
./launch.sh
```
`launch.sh` exports `DOCKER_CONFIG`, `DOCKER_BUILDKIT`, and `BUILDKIT_PROGRESS=plain` to keep Docker output verbose and credential-helper free. The Gradio UI becomes available at `http://localhost:7860`.

### Manual Docker build (optional)
```bash
docker build -t rag-sandbox .
docker run -d -p 7860:7860 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  --env-file .env \
  --name mq-rag \
  rag-sandbox
```

### Anonymous Docker builds
If credential helpers cause issues, run:
```bash
./scripts/docker_build_anonymous.sh -t rag-sandbox .
```
The script uses a temporary `DOCKER_CONFIG` and `set -x` for traceable command logs. `launch.sh` mirrors this behavior for Compose runs.

## Embedding Options
- **Offline-first (default)**: keep `ALLOW_HF_INTERNET=false` and use the pre-downloaded assets in `data/models/snowflake-arctic-embed-xs` mounted to `/app/data/models/snowflake-arctic-embed-xs`.
- **Allow one-time download**: set `ALLOW_HF_INTERNET=true` and start the stack; the model downloads into the mounted `data/` volume for future offline runs.
- **Custom repository**: set `EMBEDDING_MODEL_ID=org/repo-name` and optionally `EMBEDDING_MODEL_DIR` to point at a different local path.

## Operations
- **Upload PDFs (Admin)**: add IBM MQ PDFs via the left panel, then click **Re-index all PDFs**. Files persist under `/app/data/pdfs`.
- **Delete documents (Admin)**: choose a document and click **Delete selected** to remove its chunks from Chroma.
- **Chat (User/Admin)**: ask IBM MQ questions; responses combine retrieval and llama.cpp generation.
- **Ollama + Gradio test stack**: run `./scripts/run_ollama_docker.sh` for a local Ollama server with verbose tracing, then `python tools/ollama_gradio.py --host 0.0.0.0 --port 7861 --ollama-url http://localhost:11434 --model llama3` for a debugging UI. `./scripts/run_ollama_stack.sh` executes both sequentially.

## Logging and Observability
- Default `LOG_LEVEL=DEBUG` mirrors logs to stdout and `data/logs/app.log` for auditability.
- `config.py` ensures required directories (PDFs, Chroma DB, logs, models) exist and initializes structured logging during startup.
- Prewarm and ingestion routines emit progress, timing, and error details so you can trace activity per international scripting and programming standards.
- **Ingestion audit script**: run `python tools/ingestion_audit.py data/pdfs` (replace with your PDF folder or individual files) to validate extraction, chunking, embedding, and retrieval end to end. The script writes verbose logs to `data/logs/ingestion_audit.log` and a JSON summary to `data/reports/ingestion_audit_report.json` so you can spot empty pages, missing text, or weak retrieval coverage before relying on the graph.

## Troubleshooting
- **Import errors (e.g., `ModuleNotFoundError: No module named 'app'`)**: run as a module (`python -m app.main`) or use Docker to preserve the package context.
- **Missing embeddings**: verify `data/models/snowflake-arctic-embed-xs` exists or temporarily set `ALLOW_HF_INTERNET=true` to download automatically.
- **Slow first answer**: the background prewarm thread loads embeddings and llama.cpp on startup; keep the container running so caches stay warm.
- **Persistence checks**: confirm `/app/data/pdfs` and `/app/data/chroma_db` are mounted; empty mounts lead to missing documents or stale retrieval results.
- **Reset vector store**: remove `data/chroma_db` and re-run ingestion from the admin panel to rebuild embeddings after major document changes.

## Development
Install dependencies locally (Python 3.11 recommended):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app.main
```

## Security
- Store real secrets in `.env` or Docker secrets; never commit them.
- Default credentials are for local testing only—update them before production use.

## Reference Layout
```
rag-sandbox/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── app/
│   ├── main.py               # Gradio UI (admin + user), login flow
│   ├── rag_chain.py          # RAG engine: embeddings, vector store, llama.cpp inference
│   ├── auth.py               # Role-based auth helpers
│   ├── config.py             # Paths, env vars, logging setup
│   ├── assets/
│   │   └── custom.css        # Simple theming
│   └── utils/
│       ├── pdf_ingest.py     # PyPDFium text extraction + chunking
│       └── embeddings.py     # Snowflake Arctic embeddings
├── data/                     # PDFs, Chroma DB, logs (mounted volume)
├── models/                   # GGUF model download target (mounted volume)
└── README.md
```

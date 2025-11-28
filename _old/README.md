# RAG Sandbox (IBM MQ)

A single-container, production-ready Retrieval Augmented Generation (RAG) sandbox for IBM MQ documentation. The stack is built for 2025 best practices: llama.cpp inference with Llama-3.1-8B-Instruct Q5_K_M GGUF, Gradio 4 for a clean WebUI, Chroma persistence, PyPDFium parsing, and Snowflake Arctic embeddings. Everything runs fully offline—no external API calls once models are downloaded.

---

### 🔍 **What is RAG? (Simple Explanation)**

Imagine you ask a super-smart robot a question. But sometimes the robot doesn't *remember* everything. So instead of guessing, the robot goes to look things up **right when you ask**.

RAG is basically the robot doing two steps:

---

### **1️⃣ Retrieval — “Let me go look that up.”**

The robot searches through a big collection of documents, files, or notes to find the most helpful information. It’s like Googling, but inside the robot’s brain.

---

### **2️⃣ Generation — “Now I’ll explain it to you.”**

After finding the info, the robot reads it and then writes a clear, helpful answer in its own words.

---

### 🧠 **Why is this useful?**

Because instead of making stuff up, the robot uses **real information** it just found. So answers are:

* more accurate
* more detailed
* based on real sources
* less likely to be wrong

---

### 💬 **Example**

**You ask:** *“What were the main causes of the Civil War?”*
**The robot:**

1. Searches your history textbook and notes.
2. Finds the paragraphs about slavery, states’ rights, and economics.
3. Writes an easy-to-understand answer using what it found.

---

### 📦 **In short:**

**RAG = Search + Explain** — a robot that first *looks things up* and then *gives you a smart answer.*

## Features
- **Admin mode**: upload IBM MQ PDFs, re-index, delete documents, view verbose logs, manage the knowledge base.
- **User mode**: simple chat interface with no upload rights.
- **Offline-first**: CPU/GPU llama.cpp backend and local embeddings keep data private.
- **Persistent storage**: PDFs, Chroma DB, and logs survive container restarts.

## Default local RAG components
All core components are local-first with verbose logging enabled by default so you can trace behavior easily:

- **Language model**: `Meta-Llama-3.1-8B-Instruct.Q5_K_M.gguf` served through `llama.cpp` (configurable via `MODEL_PATH`).
- **Embedding model**: `Snowflake/snowflake-arctic-embed-xs` loaded locally via `sentence-transformers` (configurable with `EMBEDDING_MODEL_ID` and `EMBEDDING_MODEL_DIR`).
- **Vector store**: Chroma persistent client stored under `data/chroma_db`.
- **PDF ingestion**: `pypdfium2` text extraction with 1,000-character chunks and 200-character overlap.
- **Interface**: Gradio 4, single-container deployment.
- **Logging**: `LOG_LEVEL=DEBUG` by default, mirrored to stdout and `data/logs/app.log` for audit-friendly diagnostics.

## Quickstart (run these before `./launch.sh`)
1. **Clone and enter the repo**
   ```bash
   git clone https://github.com/yourname/rag-sandbox.git
   cd rag-sandbox
   ```

2. **Prepare local folders for persistence**
   ```bash
   mkdir -p models data
   ```
   These mirror the container mounts at `/app/models` and `/app/data` so models, Chroma indexes, and verbose logs persist between runs.

3. **Download the llama.cpp model (one-time, large download)**
   Use the bundled helper to avoid typos and keep logging verbose:
   ```bash
   ./download_llm.sh
   ```
   The script saves `Meta-Llama-3.1-8B-Instruct.Q5_K_M.gguf` into `models/`. The filename is case-sensitive; altering it will return `404 Not Found`.

4. **(Optional but recommended for offline use) Download embeddings locally**
   ```bash
   ./scripts/download_embedding.sh
   ```
   This installs `huggingface_hub` if missing, then downloads `Snowflake/snowflake-arctic-embed-xs` into `data/models/snowflake-arctic-embed-xs` so embedding calls never reach the internet.

5. **Set credentials, paths, and logging defaults**
   Create `.env` (ignored by git) so Docker Compose and the app share consistent defaults:
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

   The `MODEL_PATH` value must point to the exact GGUF file *inside* the container. When running with Docker, mount your downloaded model into `/app/models` and confirm the path with:
   ```bash
   docker exec -it mq-rag ls -l /app/models
   docker exec -it mq-rag printenv MODEL_PATH
   ```

### Embedding model (offline friendly)

The sentence-transformer embedder is configured to run offline by default so the
container will not attempt to reach Hugging Face. You have two options:

- **Allow one-time download with internet access**: Set `ALLOW_HF_INTERNET=true`
  in your `.env` and start the stack. The container downloads
  `Snowflake/snowflake-arctic-embed-xs` into `/app/data/models/snowflake-arctic-embed-xs`.
  The mounted `./data` volume keeps the assets for future offline runs.

- **Pre-download the embedding assets manually** (no internet required later):
  run `./scripts/download_embedding.sh` from the project root. The assets land
  in `data/models/snowflake-arctic-embed-xs` and are mounted automatically into
  `/app/data/models/snowflake-arctic-embed-xs` via Docker Compose.

If you want to use a different embedding repository, set
`EMBEDDING_MODEL_ID=org/repo-name` and optionally `EMBEDDING_MODEL_DIR` to a
custom path. Public Gradio share links are disabled by default
(`SHARE_INTERFACE=false`) to keep the UI local-only; enable them only when the
host has outbound connectivity by setting both `ALLOW_HF_INTERNET=true` and
`SHARE_INTERFACE=true`.

### Launch the stack with verbose logging

Run the Compose helper once all setup steps are complete:

```bash
./launch.sh
```

`launch.sh` uses an isolated Docker config and `BUILDKIT_PROGRESS=plain` to keep
image pulls and build output verbose for easier debugging. The app listens on
http://localhost:7860 once healthy.

### Host time synchronization

Docker Compose mounts `/etc/localtime` and `/etc/timezone` from the host into
the container so clock settings and timezone data match exactly. This keeps
timestamped DEBUG logs consistent across host and container. If your platform
does not expose `/etc/timezone` (rare on some minimal distributions), set `TZ`
in your `.env` to your host timezone (for example, `TZ=America/New_York`) before
running `./launch.sh`.

### Login defaults
- User mode → `user` / `mquser2025`
- Admin mode → `admin` / your password

### CPU optimization (stop scalar llama.cpp builds)

The Dockerfile now forces `llama-cpp-python` to compile from source with
`AVX2`, `FMA`, and `F16C` enabled (`CMAKE_ARGS` + `FORCE_CMAKE=1`). This avoids
the scalar fallback observed in some prebuilt wheels that can push prefill time
for the 8B model above three minutes. If you rebuild locally and want to verify
the optimized path, inspect the build logs for the enabled CPU features and keep
`MODEL_THREADS` aligned with your host’s physical cores for best throughput.

## Manual Docker build (optional)
```bash
docker build -t rag-sandbox .
docker run -d -p 7860:7860 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  --env-file .env \
  --name mq-rag \
  rag-sandbox
```

## Force anonymous Docker builds (avoids credential helper issues)
If your Docker host is configured with credential helpers that are unavailable in this environment, use the helper scripts below to build anonymously with verbose logging for easier debugging:

```bash
./scripts/docker_build_anonymous.sh -t rag-sandbox .
```

The script sets a temporary, empty `DOCKER_CONFIG` directory so BuildKit pulls `docker/dockerfile:1` anonymously, then cleans up after the build. All arguments are forwarded to `docker build` so you can pass additional flags as needed. To achieve the same effect for Docker Compose builds, run:

```bash
./launch.sh
```

`launch.sh` mirrors the anonymous-build behavior for Compose, exporting `DOCKER_CONFIG`, `DOCKER_BUILDKIT`, and `BUILDKIT_PROGRESS` to keep registry pulls verbose and credential-free.

## Maintenance and troubleshooting scripts (all emit verbose logs)

Each helper script is designed with explicit logging so activities can be traced in line with international scripting standards:

- `./scripts/download_embedding.sh`: Idempotently downloads the Snowflake Arctic embedding model (or an override defined by `EMBEDDING_MODEL_ID`) into `data/models/snowflake-arctic-embed-xs` with timestamped progress messages. The script auto-installs `huggingface_hub` when missing and exits with actionable guidance on failure. Override `EMBEDDING_MODEL_DIR` to place the assets elsewhere and re-run safely.
- `./scripts/docker_build_anonymous.sh`: Builds the Docker image using a temporary `DOCKER_CONFIG` to force anonymous pulls while preserving `docker build` arguments. The script enables `set -x` for command-level tracing, ensuring credential-helper issues are immediately visible.
- `./scripts/clear_persistent_data.sh`: Clears Chroma vector store and ingested PDF folders under `data/` while leaving models intact. Safety checks prevent destructive deletions, and log statements mark each reset step so you can confirm the storage state before restarting the stack.
- `./scripts/verify_repo_container_mapping.sh`: Verifies that the Git repository is correctly mapped into the container by reporting mount metadata, validating the configured `origin`, checking the current branch/commit, and performing a safe write test. Verbose tracing (`set -x`) and timestamped log lines make the mapping audit easy to debug.

## Project Structure
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

## Why the first answer was slow (and how it's mitigated now)

Both the embedding model and the llama.cpp GGUF weights are memory-mapped on
first use. On a fresh container boot this one-time load can take several
minutes; subsequent questions are faster because the models stay resident in
memory. To avoid making your first user query wait for this warm-up, the
application now starts a background prewarm thread during startup. The thread
does three things with verbose logging so you can audit progress:

1. Loads the embedding model (and downloads it if `ALLOW_HF_INTERNET=true`).
2. Instantiates the llama.cpp engine with your configured `MODEL_PATH`.
3. Runs a single-token generation to prime caches without consuming noticeable
   resources.

Keep the container running to retain this warm state. Model assets, Chroma
indexes, and logs continue to persist on disk between runs via the mounted
`data/` and `models/` volumes; if you restart the container, the background
prewarm automatically runs again so first-question latency steadily improves
once the thread completes.

## Behind-the-scenes ingest flow

1. **Upload handling and persistence**
   * Incoming Gradio uploads (or file paths) are normalized to real paths, preserving original names while sanitizing them for the PDF storage directory (`data/pdfs`). Verbose debug/info logs trace each candidate path and final resolution. If nothing is uploaded, the handler short-circuits with a warning. Failures are collected with stack traces to keep the UI response actionable.

   * Successfully resolved files are written to the PDF directory before ingestion begins, ensuring the vectorization step always reads from a stable on-disk location. All writes and errors are logged for traceability.

2. **PDF text extraction and chunking**
   * The ingestion utility spins up a `PdfIngestor` with configurable chunk sizing/overlap, logging the parameters up front. Each PDF is opened with PyPDFium, page-by-page text is extracted with debug-level progress, and the full document text is split into overlapping chunks while recording byte offsets for metadata. The module emits info-level summaries for every file and a roll-up of total PDFs and chunks processed.

   * Nonexistent PDFs are skipped with warnings to avoid hard failures during batch runs.

3. **Embedding generation and vector-store write**
   * The RAG engine gathers all PDFs from the ingest directory, runs the chunker, and then calls the embedding layer. It handles missing embedding assets explicitly—logging exceptions and returning guidance on placing the Snowflake Arctic model locally or enabling downloads—before proceeding. Successful runs upsert chunk embeddings, raw text, and metadata into the Chroma collection and log the totals for auditing.

   * The embedding helper enforces offline-first behavior, verifying assets exist under `EMBEDDING_MODEL_DIR` (or downloading them when allowed), loading the model once via an LRU cache, and providing detailed logging around asset discovery and encoding throughput.

4. **Operational safeguards and telemetry**
   * Global configuration sets DEBUG-level logging by default, writes to both console and `data/logs/app.log`, and ensures required directories (PDFs, Chroma DB, logs, models) exist before ingestion. Offline defaults prevent accidental network calls unless explicitly enabled via `ALLOW_HF_INTERNET`, aligning with the request for verbose, auditable operations.

   * Ingestion status is surfaced back to the UI through `ingest_pdfs()` responses, allowing administrators to see success counts or actionable error messages after each run.

All steps follow verbose logging conventions so activities can be debugged easily and remain aligned with international programming and scripting standards.

## Operations
- **Upload PDFs (Admin)**: Add IBM MQ PDFs via the left panel, then click **Re-index all PDFs**. Files are stored in `/app/data/pdfs`.
- **Delete documents (Admin)**: Select a document from the dropdown and click **Delete selected** to remove its chunks from Chroma.
- **Chat (User/Admin)**: Ask IBM MQ questions; responses come from local retrieval + llama.cpp generation.

## Logging and Observability
- Logs are written to stdout and mirrored to `data/logs/app.log`; inspect with `docker logs mq-rag` or by reading the file from the mounted volume.
- Verbose logging is enabled by default (`LOG_LEVEL=DEBUG` via docker compose); lower to `INFO` if you want quieter output once the system is stable.
- If you upgrade dependencies, rebuild the container or virtual environment to ensure `pypdfium2`, `sentence-transformers`, and `llama-cpp-python` remain in sync with your platform.

## Troubleshooting and Debugging
- **Import errors (e.g., `ModuleNotFoundError: No module named 'app'`)**: Always run the service as a module so Python resolves the `app` package correctly. Use `python -m app.main` locally or keep the default container command.
- **Embedding assets missing**: Ensure `data/models/snowflake-arctic-embed-xs` exists or set `ALLOW_HF_INTERNET=true` to let the container download the model.
- **Enable/adjust verbosity**: Set `LOG_LEVEL=DEBUG` (default) for detailed tracing in both stdout and `data/logs/app.log`. If troubleshooting noisy dependencies, briefly bump to `INFO`, then revert to `DEBUG` to retain rich context.
- **Validate environment variables**: Run `printenv | sort` inside the container to confirm credentials, `MODEL_PATH`, and logging settings are present. Missing values often lead to authentication failures or model loading errors.
- **Check persisted artifacts**: Confirm `/app/data/pdfs` and `/app/data/chroma_db` are mounted. Empty mounts can explain missing documents or retrieval mismatches.
- **Reset vector store**: If responses seem stale after PDF changes, remove `data/chroma_db` and re-run ingestion from the admin panel to rebuild embeddings.
- **Tail logs while reproducing**: Use `docker compose logs -f rag-sandbox` during reproduction so warnings and stack traces remain time-correlated with UI actions.

## Notes and Best Practices
- The container runs entirely offline after the initial model download.
- Ensure adequate CPU/RAM for the 8B model; adjust `MODEL_THREADS` and `MODEL_N_CTX` via environment variables if needed.
- For clean shutdowns and persistence, prefer `docker compose` with mounted volumes as shown above.

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

## Codex prompt manifest workflow
- The authoritative orientation guide for agents lives at `CODEX_MANIFEST.md`. Read it before proposing UI or UX changes.
- Regenerate the auto-discovery sections any time the repo layout shifts:
  ```bash
  python tools/generate_codex_manifest.py
  ```
  The script uses verbose logging so you can audit what was scanned.
- Use this prompt header to ensure agents honor the manifest and the UI-focused definition of done:
  > Read CODEX_MANIFEST.md first and treat it as authoritative context. Follow its Hard Rules + Definition of Done. If any part conflicts with your task, stop and explain the conflict.
- Keep the **Change Log** section of `CODEX_MANIFEST.md` up to date when architecture or entrypoints move.

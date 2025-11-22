# RAG Sandbox (IBM MQ)

A single-container, production-ready Retrieval Augmented Generation (RAG) sandbox for IBM MQ documentation. The stack is built for 2025 best practices: llama.cpp inference with Llama-3.1-8B-Instruct Q5_K_M GGUF, Gradio 4 for a clean WebUI, Chroma persistence, PyPDFium parsing, and Snowflake Arctic embeddings. Everything runs fully offline—no external API calls once models are downloaded.

## RAG in Plain Language

Here’s **Retrieval Augmented Generation (RAG)** explained like you’re in **8th grade**:

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

### Login defaults
- User mode → `user` / `mquser2025`
- Admin mode → `admin` / your password

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

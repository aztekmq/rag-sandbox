# RAG Sandbox (IBM MQ)

A single-container, production-ready Retrieval Augmented Generation (RAG) sandbox for IBM MQ documentation. The stack is built for 2025 best practices: llama.cpp inference with Llama-3.1-8B-Instruct Q5_K_M GGUF, Gradio 4 for a clean WebUI, Chroma persistence, Docling PDF parsing, and Snowflake Arctic embeddings. Everything runs fully offline—no external API calls once models are downloaded.

## Features
- **Admin mode**: upload IBM MQ PDFs, re-index, delete documents, view verbose logs, manage the knowledge base.
- **User mode**: simple chat interface with no upload rights.
- **Offline-first**: CPU/GPU llama.cpp backend and local embeddings keep data private.
- **Persistent storage**: PDFs, Chroma DB, and logs survive container restarts.

## Quickstart
1. **Clone and enter the repo**
   ```bash
   git clone https://github.com/yourname/rag-sandbox.git
   cd rag-sandbox
   ```

2. **Download the model (once, large download)**
   ```bash
   mkdir -p models
   wget -O models/llama-3.1-8b-instruct-q5_k_m.gguf \
     https://huggingface.co/QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf
   ```

3. **Set credentials and model path**
   Create `.env` (ignored by git):
   ```env
   ADMIN_USERNAME=admin
   ADMIN_PASSWORD=change_me_strong_password
   USER_USERNAME=user
   USER_PASSWORD=mquser2025
   MODEL_PATH=/app/models/llama-3.1-8b-instruct-q5_k_m.gguf
   LOG_LEVEL=INFO
   ```

4. **Build and run via Docker Compose**
   ```bash
   docker compose up -d --build
   ```
   The app listens on http://localhost:7860.

5. **Login**
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
│       ├── pdf_ingest.py     # Docling PDF parsing + chunking
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
- Verbose logging is enabled by default (`LOG_LEVEL=INFO`); set `LOG_LEVEL=DEBUG` for deeper traceability during ingestion and inference.

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
python app/main.py
```

## Security
- Store real secrets in `.env` or Docker secrets; never commit them.
- Default credentials are for local testing only—update them before production use.

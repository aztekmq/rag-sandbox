# RAG Sandbox (IBM MQ)

A single-container, production-ready Retrieval Augmented Generation (RAG) sandbox for IBM MQ documentation. The stack is built for 2025 best practices: llama.cpp inference with Llama-3.1-8B-Instruct Q5_K_M GGUF, Gradio 4 for a clean WebUI, Chroma persistence, Docling PDF parsing, and Snowflake Arctic embeddings. Everything runs fully offline—no external API calls once models are downloaded.

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
- Python 3.11 builders should rebuild after pulling updates: Docling is pinned to `1.7.2` to pull a compatible `docling-parse` release and avoid installation errors.

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

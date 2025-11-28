# CODEX MANIFEST — Gradio App Orientation

## 0) TL;DR
This is a Gradio-based web app for IBM MQ / RAG analysis. Your job is to improve UI/UX and front-end layout without touching backend logic.

---

## 1) Project Purpose (1–2 paragraphs)
- What this app does:
  - Provides a role-aware Gradio dashboard to query an IBM MQ knowledge base using a local RAG pipeline (llama.cpp + Chroma + Snowflake Arctic embeddings).
  - Supports document ingestion, sessionized chat, and admin tooling while keeping verbose logging for debugging.
- Primary users:
  - Admins / MQ engineers for ingestion and maintenance
  - Analysts / operators for question answering
- Primary user flows:
  1) Login → open tool → ask question → review answer/sources/logs
  2) Admin → manage docs → reindex → validate ingestion

---

## 2) Repo Map (authoritative)
### Entrypoints
- Main Gradio app:
  - `app/main.py` — creates Blocks layout and wires callbacks
- Secondary UI modules:
  - Inline within `app/main.py` (panels, callbacks, state helpers)
- Backend / logic boundary (do not modify unless asked):
  - `app/rag_chain.py`, `app/utils/*.py` (embeddings, ingestion), `app/auth.py`, `app/config.py`

### Styling
- CSS file(s):
  - `app/assets/custom.css` (loaded via `CUSTOM_CSS` in `app/main.py`)
- Inline Blocks css:
  - `CUSTOM_CSS` string defined in `app/main.py`

---

## 3) Current UI Structure (quick outline)
- Login view (`login-view`): Column with welcome copy, user/password inputs, status markdown
- Workspace (`workspace`):
  - Header bar: left title, right badges for user/role/env, logout button
  - Sidebar (`nav-rail`): navigation buttons (Manage Docs, Help & FAQ), session status, placeholder history card
  - Main area (`main-content`):
    - Search view (`search-view`): KPI strip, "Ask MQ" textbox + submit button, example chips, chatbot output with toggles for sources and raw response, secondary inputs for temperature/top_k/context, source table and query timing stats
    - Help view (`help-view`): rich markdown FAQ and usage guidance
    - Manage Docs view (`manage-docs-view`): upload + ingest button, document table/overview/status, delete document control, admin hint copy
- Known pain points from code comments: session history is placeholder-only; KPI values are static; layout tuned for desktop widths

---

## 4) Target UI Style
**Style:** Data-dense Ops Dashboard (Datadog/Grafana vibe)

### Layout target
- Fixed top header bar
- Left tool/history rail ~260px
- Main work area with:
  - KPI strip
  - Primary workflow panel
  - Results grid with tabs

### Definition of Done (must satisfy)
1. Sidebar fixed 260px with scrollable history/tools sections
2. KPI strip visible above fold on 1366×768
3. No section has >16px vertical gap
4. Compact inputs ≤36px height
5. Results in tabs: Answer / Sources / Logs / Raw
6. CSS in `app/assets/custom.css` and loaded globally
7. No horizontal scrolling except tables/log panes
8. Backend logic untouched

---

## 5) Hard Rules for Codex
- UI only unless prompt explicitly says otherwise.
- Do **not** change inference, RAG, MQ parsing, or file ingestion logic.
- Only edit:
  - `app/main.py`
  - `app/assets/custom.css`
  - Optional UI helper modules if added under `app/`
- Before editing, you must: (1) summarize current UI structure from cited code, (2) propose a plan, (3) then implement via diffs.
- If more files seem necessary, STOP and explain.

---

## 6) Common Tasks + Where to Implement
- Tighten spacing / layout density:
  - Edit `app/assets/custom.css` + container gaps in `app/main.py`
- Move advanced options into accordions:
  - Within relevant columns/rows in `app/main.py`
- Results reorganization (tabs/panels):
  - Chatbot + result tables in `app/main.py` (search view)
- Sidebar/history shell:
  - `nav-rail` column and related markdown in `app/main.py`

---

## 7) How to Run / Test
```bash
# local run
python -m app.main
# or via launcher
./launch.sh
# regenerate auto skeleton
python tools/generate_codex_manifest.py
```

Expected test:

* App loads without error on http://localhost:7860
* Layout fits above fold at 1366×768
* Primary run flow unchanged

---

## 8) Change Log
* Keep this section updated when architecture changes.

---

## Prompt Header (reuse in Codex prompts)
Read CODEX_MANIFEST.md first and treat it as authoritative context. Follow its Hard Rules + Definition of Done. If any part conflicts with your task, stop and explain the conflict.

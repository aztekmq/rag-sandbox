"""Application configuration management.

This module centralizes paths, credentials, and model configuration using
environment variables while keeping sensible defaults for local testing. All
values are designed to be safe for production when secrets are provided through
a `.env` file or Docker environment variables. The configuration follows the
International Programming Standards for documentation and readability so that
operators can audit and tune runtime parameters with confidence. Verbose logging
and explicit offline controls are enabled by default to aid debugging and avoid
unexpected network calls on isolated hosts.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

load_dotenv()

# Enforce offline defaults for Hugging Face backed libraries unless explicitly
# overridden. This prevents containers from attempting to reach external
# services during local or air-gapped deployments.
ALLOW_HF_INTERNET: Final[bool] = (
    os.getenv("ALLOW_HF_INTERNET", "false").strip().lower() == "true"
)
if not ALLOW_HF_INTERNET:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = BASE_DIR / "data"
PDF_DIR: Final[Path] = DATA_DIR / "pdfs"
CHROMA_DIR: Final[Path] = DATA_DIR / "chroma_db"
LOG_DIR: Final[Path] = DATA_DIR / "logs"
# Case-sensitive filename that matches the upstream GGUF artifact; altering the
# casing can cause 404 responses during downloads on case-sensitive filesystems.
DEFAULT_MODEL_FILENAME: Final[str] = "Meta-Llama-3.1-8B-Instruct.Q5_K_M.gguf"
DEFAULT_MODEL_PATH: Final[Path] = (
    Path(os.getenv("MODEL_PATH", BASE_DIR / "models" / DEFAULT_MODEL_FILENAME))
    .expanduser()
    .resolve()
)
EMBEDDING_MODEL_DIR: Final[Path] = Path(
    os.getenv("EMBEDDING_MODEL_DIR", DATA_DIR / "models" / "snowflake-arctic-embed-xs")
)
EMBEDDING_MODEL_ID: Final[str] = os.getenv(
    "EMBEDDING_MODEL_ID", "Snowflake/snowflake-arctic-embed-xs"
)

ADMIN_USER: Final[str] = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASS: Final[str] = os.getenv("ADMIN_PASSWORD", "change_me_strong_password")
USER_USER: Final[str] = os.getenv("USER_USERNAME", "user")
USER_PASS: Final[str] = os.getenv("USER_PASSWORD", "mquser2025")

MODEL_PATH: Final[Path] = DEFAULT_MODEL_PATH
MODEL_N_CTX: Final[int] = int(os.getenv("MODEL_N_CTX", 4096))
MODEL_THREADS: Final[int] = int(os.getenv("MODEL_THREADS", 8))
# Explicitly disable Gradio tunneling by default to keep traffic local. Operators
# can opt-in by setting SHARE_INTERFACE=true when internet access is permitted.
SHARE_INTERFACE_REQUESTED: Final[bool] = (
    os.getenv("SHARE_INTERFACE", "false").strip().lower() == "true"
)

LOG_LEVEL: Final[str] = os.getenv("LOG_LEVEL", "DEBUG")
LOG_FORMAT: Final[str] = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"

for directory in (DATA_DIR, PDF_DIR, CHROMA_DIR, LOG_DIR, EMBEDDING_MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format=LOG_FORMAT,
)

LOG_FILE = LOG_DIR / "app.log"
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(log_level)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
logging.getLogger().addHandler(file_handler)

# Gradio share links require outbound connectivity. If the deployment is
# configured for offline operation, force sharing off and emit a clear warning
# so operators understand why no public URL is exposed.
SHARE_INTERFACE: bool = SHARE_INTERFACE_REQUESTED
if SHARE_INTERFACE_REQUESTED and not ALLOW_HF_INTERNET:
    logging.getLogger(__name__).warning(
        "SHARE_INTERFACE requested but ALLOW_HF_INTERNET is false; disabling public share "
        "link to keep the application local only."
    )
    SHARE_INTERFACE = False

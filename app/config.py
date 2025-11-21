"""Application configuration management.

This module centralizes paths, credentials, and model configuration using
environment variables while keeping sensible defaults for local testing.
All values are designed to be safe for production when secrets are provided
through a `.env` file or Docker environment variables.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

load_dotenv()

BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = BASE_DIR / "data"
PDF_DIR: Final[Path] = DATA_DIR / "pdfs"
CHROMA_DIR: Final[Path] = DATA_DIR / "chroma_db"
LOG_DIR: Final[Path] = DATA_DIR / "logs"

ADMIN_USER: Final[str] = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASS: Final[str] = os.getenv("ADMIN_PASSWORD", "change_me_strong_password")
USER_USER: Final[str] = os.getenv("USER_USERNAME", "user")
USER_PASS: Final[str] = os.getenv("USER_PASSWORD", "mquser2025")

MODEL_PATH: Final[str | None] = os.getenv("MODEL_PATH")
MODEL_N_CTX: Final[int] = int(os.getenv("MODEL_N_CTX", 4096))
MODEL_THREADS: Final[int] = int(os.getenv("MODEL_THREADS", 8))

LOG_LEVEL: Final[str] = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT: Final[str] = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"

for directory in (DATA_DIR, PDF_DIR, CHROMA_DIR, LOG_DIR):
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

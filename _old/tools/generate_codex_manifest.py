#!/usr/bin/env python3
"""Generate a first-pass CODEX_MANIFEST.md for this Gradio RAG repo.

The script is intentionally read-only apart from writing the manifest output.
It scans for UI entrypoints, styling assets, and backend boundaries so agents
have a concise orientation guide. Logging is verbose by default to simplify
debugging in line with international programming standards.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

ROOT = Path(".").resolve()
IGNORE_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", "dist", "build"}

GRADIO_HINTS = [
    re.compile(r"\bgr\.Blocks\s*\("),
    re.compile(r"\bgr\.Interface\s*\("),
    re.compile(r"\bimport\s+gradio\s+as\s+gr\b"),
    re.compile(r"\bBlocks\s*\(\s*css\s*="),
]

logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.DEBUG) -> None:
    """Configure module-level logging with a consistent, verbose formatter."""

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    logger.debug("Logging configured at %s", logging.getLevelName(level))


def iter_files() -> Iterable[Path]:
    """Yield candidate files while skipping ignored directories."""

    for path in ROOT.rglob("*"):
        if path.is_dir():
            continue
        if any(part in IGNORE_DIRS for part in path.parts):
            continue
        yield path


def find_gradio_files() -> List[Path]:
    """Return Python files that look like Gradio entrypoints or UI modules."""

    gradio_files: List[Path] = []
    for path in iter_files():
        if path.suffix != ".py":
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:  # pragma: no cover - defensive logging only
            logger.warning("Failed to read %s: %s", path, exc)
            continue
        if any(regex.search(text) for regex in GRADIO_HINTS):
            logger.debug("Detected Gradio hints in %s", path)
            gradio_files.append(path)
    return gradio_files


def detect_css_files() -> List[Path]:
    """Collect discovered CSS files for styling context."""

    css_files: List[Path] = []
    for path in iter_files():
        if path.suffix == ".css":
            css_files.append(path)
            logger.debug("Found CSS file: %s", path)
    return css_files


def guess_entrypoints(gradio_files: List[Path]) -> List[Path]:
    """Heuristically identify likely Gradio entrypoints from detected files."""

    candidates: List[Path] = []
    for path in gradio_files:
        name = path.name.lower()
        if path.parent == ROOT and name in {"app.py", "main.py", "server.py"}:
            candidates.append(path)
            logger.debug("Flagging root-level candidate entrypoint: %s", path)

    if not candidates:
        for path in gradio_files:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "gr.Blocks" in text:
                candidates.append(path)
                logger.debug("Fallback entrypoint candidate via Blocks(): %s", path)
    return candidates


def list_ui_dirs() -> List[Path]:
    """List directories that typically contain UI code."""

    ui_dirs: List[Path] = []
    for name in ["ui", "frontend", "web", "pages", "components", "app"]:
        directory = ROOT / name
        if directory.exists() and directory.is_dir():
            ui_dirs.append(directory)
            logger.debug("Registered UI directory: %s", directory)
    return ui_dirs


def list_backend_dirs() -> List[Path]:
    """List directories that likely contain backend or service logic."""

    backend_dirs: List[Path] = []
    for name in ["backend", "rag", "mq", "core", "services", "api", "logic", "app/utils"]:
        directory = ROOT / name
        if directory.exists() and directory.is_dir():
            backend_dirs.append(directory)
            logger.debug("Registered backend directory: %s", directory)
    return backend_dirs


def summarize_file(path: Path) -> dict:
    """Summarize Gradio building blocks used in a file for quick scanning."""

    text = path.read_text(encoding="utf-8", errors="ignore")
    summary = {
        "blocks": len(re.findall(r"gr\.Blocks\s*\(", text)),
        "interfaces": len(re.findall(r"gr\.Interface\s*\(", text)),
        "tabs": len(re.findall(r"gr\.Tabs\s*\(", text)),
        "accordions": len(re.findall(r"gr\.Accordion\s*\(", text)),
        "rows": len(re.findall(r"gr\.Row\s*\(", text)),
        "columns": len(re.findall(r"gr\.Column\s*\(", text)),
        "css_inline": bool(re.search(r"Blocks\s*\(\s*css\s*=", text)),
    }
    logger.debug("Summary for %s: %s", path, summary)
    return summary


def build_manifest_content(
    entrypoints: List[Path],
    gradio_files: List[Path],
    css_files: List[Path],
    ui_dirs: List[Path],
    backend_dirs: List[Path],
) -> str:
    """Construct the manifest markdown content."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append("# CODEX MANIFEST — AUTO-GENERATED SKELETON")
    lines.append(f"_Generated: {timestamp}_")
    lines.append("")
    lines.append("## 0) TL;DR")
    lines.append("This is a Gradio-based app. UI refactors are allowed; backend logic is not.")
    lines.append("")
    lines.append("## 1) Entrypoints (likely)")
    if entrypoints:
        for path in entrypoints:
            lines.append(f"- `{path}`")
    else:
        lines.append("- (Could not confidently detect root entrypoint.)")
    lines.append("")
    lines.append("## 2) Gradio UI Files")
    for path in gradio_files:
        stats = summarize_file(path)
        lines.append(
            "- `{}` — Blocks:{} Tabs:{} Acc:{} Rows:{} Cols:{} InlineCSS:{}".format(
                path, stats["blocks"], stats["tabs"], stats["accordions"], stats["rows"], stats["columns"], stats["css_inline"]
            )
        )
    lines.append("")
    lines.append("## 3) UI Directories")
    if ui_dirs:
        for directory in ui_dirs:
            lines.append(f"- `{directory}`")
    else:
        lines.append("- none detected")
    lines.append("")
    lines.append("## 4) CSS / Styling Files")
    if css_files:
        for path in css_files:
            lines.append(f"- `{path}`")
    else:
        lines.append("- none detected")
    lines.append("")
    lines.append("## 5) Backend / Logic Directories (do not modify)")
    if backend_dirs:
        for directory in backend_dirs:
            lines.append(f"- `{directory}`")
    else:
        lines.append("- none detected")
    lines.append("")
    lines.append("## 6) Known UI Issues")
    lines.append("- (Fill in: whitespace, layout fit, hierarchy, responsiveness, etc.)")
    lines.append("")
    lines.append("## 7) Target UI Style + Definition of Done")
    lines.append("- Style: Data-dense Ops Dashboard")
    lines.append("- (Fill in your checklist here.)")
    lines.append("")
    lines.append("## 8) Hard Rules for Codex")
    lines.append("- UI only unless explicitly asked.")
    lines.append("- Only edit app/ui/css files.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Generate CODEX_MANIFEST.md from repository structure."""

    configure_logging()
    logger.info("Starting manifest discovery under %s", ROOT)

    gradio_files = find_gradio_files()
    css_files = detect_css_files()
    entrypoints = guess_entrypoints(gradio_files)
    ui_dirs = list_ui_dirs()
    backend_dirs = list_backend_dirs()

    content = build_manifest_content(entrypoints, gradio_files, css_files, ui_dirs, backend_dirs)

    manifest_path = ROOT / "CODEX_MANIFEST.md"
    manifest_path.write_text(content, encoding="utf-8")
    logger.info("Wrote manifest to %s", manifest_path)


if __name__ == "__main__":
    main()

"""Disk cache keyed by content hash. Mirrors graphify's cache pattern."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def content_hash(text: str) -> str:
    """SHA256 of text string."""
    return hashlib.sha256(text.encode()).hexdigest()


def ensure_dirs(out_dir: Path = Path("ratm-out")) -> None:
    """Create ratm-out/ subdirectories."""
    (out_dir / "cache" / "summaries").mkdir(parents=True, exist_ok=True)
    (out_dir / "cache" / "ast").mkdir(parents=True, exist_ok=True)
    (out_dir / "chroma").mkdir(parents=True, exist_ok=True)


def load_summary(hash: str, out_dir: Path = Path("ratm-out")) -> str | None:
    """Load cached summary for content hash. Returns None on miss."""
    entry = out_dir / "cache" / "summaries" / f"{hash}.json"
    if not entry.exists():
        return None
    try:
        return json.loads(entry.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def save_summary(hash: str, summary: str, out_dir: Path = Path("ratm-out")) -> None:
    """Persist a summary string keyed by content hash."""
    ensure_dirs(out_dir)
    entry = out_dir / "cache" / "summaries" / f"{hash}.json"
    tmp = entry.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(summary), encoding="utf-8")
        tmp.replace(entry)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def load_ast(file_hash: str, out_dir: Path = Path("ratm-out")) -> dict | None:
    """Load cached AST extraction keyed by file hash. Returns None on miss."""
    entry = out_dir / "cache" / "ast" / f"{file_hash}.json"
    if not entry.exists():
        return None
    try:
        return json.loads(entry.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def save_ast(file_hash: str, data: dict, out_dir: Path = Path("ratm-out")) -> None:
    """Persist AST extraction result keyed by file hash."""
    ensure_dirs(out_dir)
    entry = out_dir / "cache" / "ast" / f"{file_hash}.json"
    tmp = entry.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(entry)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

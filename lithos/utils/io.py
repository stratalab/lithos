"""Filesystem helpers: atomic writes, hashing, JSON/YAML I/O, no-clobber guards.

Every generated artifact in Lithos is written through these helpers so writes are
atomic (no half-written manifests) and existing outputs are never silently
overwritten (PRD §20.7).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "atomic_write_bytes",
    "atomic_write_text",
    "ensure_dir",
    "ensure_new_dir",
    "read_json",
    "read_yaml",
    "sha256_bytes",
    "sha256_file",
    "write_json",
    "write_yaml",
]

StrPath = str | os.PathLike[str]


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: StrPath, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of a file, read in ``chunk_size`` chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_bytes(path: StrPath, data: bytes) -> Path:
    """Write ``data`` to ``path`` atomically (temp file in the same dir + replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        # Remove the temp file on any failure, then re-raise.
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
    return path


def atomic_write_text(path: StrPath, text: str, encoding: str = "utf-8") -> Path:
    """Atomically write ``text`` to ``path``."""
    return atomic_write_bytes(path, text.encode(encoding))


def read_json(path: StrPath) -> Any:
    """Load JSON from ``path``."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: StrPath, obj: Any, *, indent: int = 2, sort_keys: bool = False) -> Path:
    """Atomically write ``obj`` as JSON (trailing newline)."""
    text = json.dumps(obj, indent=indent, sort_keys=sort_keys, ensure_ascii=False)
    return atomic_write_text(path, text + "\n")


def read_yaml(path: StrPath) -> Any:
    """Load YAML from ``path`` (safe loader)."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: StrPath, obj: Any) -> Path:
    """Atomically write ``obj`` as YAML (block style, key order preserved)."""
    text = yaml.safe_dump(obj, sort_keys=False, allow_unicode=True)
    return atomic_write_text(path, text)


def ensure_dir(path: StrPath) -> Path:
    """Create ``path`` (and parents) if missing; return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_new_dir(path: StrPath, *, allow_existing: bool = False) -> Path:
    """Create a new directory, refusing to clobber an existing one (PRD §20.7).

    Pass ``allow_existing=True`` to deliberately reuse an existing directory
    (e.g. an explicit resume, PRD §9.9).
    """
    p = Path(path)
    if p.exists() and not allow_existing:
        raise FileExistsError(
            f"Refusing to clobber existing path: {p}. Pass allow_existing=True to reuse it."
        )
    p.mkdir(parents=True, exist_ok=allow_existing)
    return p

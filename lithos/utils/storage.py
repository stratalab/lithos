"""Pluggable artifact storage (PRD §26.5).

A thin ``fsspec`` wrapper so the durable home for shards, checkpoints, and
exported models is a *single config value*. Local disk by default; point
``base_uri`` at an S3-compatible bucket (AWS S3, Cloudflare R2, MinIO, Backblaze
B2) or GCS and nothing else in the codebase changes.

Credentials come from the environment (``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY``
for S3/R2, etc.), never from config files.

Training reads shards via mmap, which needs a local filesystem, so the pattern
is: durable copy in the object store, working copy staged to local disk — pull
shards before training, push checkpoints/artifacts after. ``Storage.put`` /
``Storage.get`` move whole files or directories between the two.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import fsspec
from pydantic import BaseModel, ConfigDict, Field

from lithos.utils.env import load_env


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Bare path -> local disk; otherwise a URI: s3://bucket, gs://bucket, memory://x
    base_uri: str = "data/store"
    endpoint_url: str | None = None  # S3-compatible endpoint (R2/MinIO/B2)
    region: str | None = None
    anon: bool = False
    options: dict[str, Any] = Field(default_factory=dict)  # extra fsspec storage_options


class Storage:
    """Filesystem-agnostic store rooted at ``cfg.base_uri``."""

    def __init__(self, cfg: StorageConfig) -> None:
        load_env()  # pick up R2/S3 credentials from a local .env if present
        self.cfg = cfg
        base_uri = cfg.base_uri
        if "://" not in base_uri:
            # Bare path -> local disk, resolved to an absolute path.
            self.fs = fsspec.filesystem("file")
            self.root = str(Path(base_uri).resolve())
        else:
            opts: dict[str, Any] = dict(cfg.options)
            client_kwargs: dict[str, Any] = {}
            # Endpoint/region from config, falling back to the standard env vars.
            endpoint = (
                cfg.endpoint_url
                or os.environ.get("AWS_ENDPOINT_URL_S3")
                or os.environ.get("AWS_ENDPOINT_URL")
            )
            region = cfg.region or os.environ.get("AWS_DEFAULT_REGION")
            if endpoint:
                client_kwargs["endpoint_url"] = endpoint
            if region:
                client_kwargs["region_name"] = region
            if client_kwargs:
                opts.setdefault("client_kwargs", {}).update(client_kwargs)
            if cfg.anon:
                opts["anon"] = True
            self.fs, self.root = fsspec.core.url_to_fs(base_uri, **opts)

    def uri(self, *parts: str) -> str:
        """Full backend URI for a path relative to the store root."""
        cleaned = [p.strip("/") for p in parts if p]
        return "/".join([self.root.rstrip("/"), *cleaned])

    def exists(self, rel: str) -> bool:
        return bool(self.fs.exists(self.uri(rel)))

    def open(self, rel: str, mode: str = "rb") -> Any:
        return self.fs.open(self.uri(rel), mode)

    def ls(self, rel: str = "") -> list[str]:
        return list(self.fs.ls(self.uri(rel)))

    def put(self, local_path: str | Path, rel: str) -> str:
        """Upload a local file or directory to ``rel`` in the store."""
        lp = Path(local_path)
        dest = self.uri(rel)
        parent = str(Path(rel).parent)
        if parent and parent != ".":
            self.fs.makedirs(self.uri(parent), exist_ok=True)
        if lp.is_dir():
            self.fs.put(str(lp), dest, recursive=True)
        else:
            # Stream through open() to the exact key (reliable across backends/R2).
            with open(lp, "rb") as src, self.fs.open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
        return dest

    def get(self, rel: str, local_path: str | Path) -> Path:
        """Download a file or directory from ``rel`` to ``local_path``."""
        src = self.uri(rel)
        local = Path(local_path)
        local.parent.mkdir(parents=True, exist_ok=True)
        if self.fs.isdir(src):
            # Target must not pre-exist, or fsspec nests src inside it.
            self.fs.get(src, str(local), recursive=True)
        else:
            with self.fs.open(src, "rb") as r, open(local, "wb") as w:
                shutil.copyfileobj(r, w)
        return local

    def write_bytes(self, rel: str, data: bytes) -> None:
        parent = str(Path(rel).parent)
        if parent and parent != ".":
            self.fs.makedirs(self.uri(parent), exist_ok=True)
        with self.open(rel, "wb") as f:
            f.write(data)

    def read_bytes(self, rel: str) -> bytes:
        with self.open(rel, "rb") as f:
            return bytes(f.read())

    def write_json(self, rel: str, obj: Any) -> None:
        self.write_bytes(rel, (json.dumps(obj, indent=2) + "\n").encode("utf-8"))

    def read_json(self, rel: str) -> Any:
        return json.loads(self.read_bytes(rel))


def load_storage(config_path: str | Path = "configs/storage.yaml") -> Storage:
    """Build a Storage from a config file, applying the LITHOS_STORAGE_BASE_URI override.

    The env override (typically set in .env) keeps the committed config generic
    while the actual bucket lives with the credentials.
    """
    from lithos.utils.config import load_and_validate

    load_env()
    cfg = load_and_validate(str(config_path), StorageConfig)
    override = os.environ.get("LITHOS_STORAGE_BASE_URI")
    if override:
        cfg = cfg.model_copy(update={"base_uri": override})
    return Storage(cfg)

"""Shared pytest fixtures/helpers for the dataset-inspector test suite."""

from __future__ import annotations

import os
import time
from typing import Optional

from dataset_inspector.config import Config
from dataset_inspector.core.models import Compression, FileMeta, FileType


def make_config(**overrides) -> Config:
    """A Config backed by an in-memory dict. Readers use `.get(key, default)` so {} is valid."""
    return Config(overrides or {})


def make_meta(path: str, file_type: FileType, compression: Compression = Compression.NONE,
              encoding: str = "utf-8") -> FileMeta:
    """Build a minimal FileMeta pointing at a real on-disk file (for reader unit tests)."""
    ap = os.path.abspath(path)
    return FileMeta(
        file_name=os.path.basename(ap),
        absolute_path=ap,
        relative_path=os.path.basename(ap),
        parent_dir=os.path.dirname(ap),
        dataset="testds",
        source="test",
        extension=os.path.splitext(ap)[1].lstrip("."),
        file_type=file_type,
        mime_type="application/octet-stream",
        compression=compression,
        size_bytes=os.path.getsize(ap) if os.path.exists(ap) else 0,
        mtime=time.time(),
        encoding=encoding,
    )

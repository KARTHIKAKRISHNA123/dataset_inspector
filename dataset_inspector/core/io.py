"""Streaming I/O primitives — the memory-safety backbone.

Every function here reads *bounded* amounts: a byte prefix, or one line/record at a time via
generators. Compression is streamed. Nothing calls `.read()` without a size, and nothing builds a
list of a file's contents.
"""

from __future__ import annotations

import contextlib
import gzip
import io
import json
import os
import zipfile
from typing import Any, Dict, Iterator, Optional, TextIO

from .models import Compression, FileMeta
from .exceptions import CorruptFileError, ReaderError


def read_head_bytes(path: str, n: int) -> bytes:
    """First `n` bytes from disk (no decompression). O(1) regardless of file size."""
    try:
        with open(path, "rb") as fh:
            return fh.read(n)
    except OSError as exc:
        raise CorruptFileError(f"cannot read {path}: {exc}") from exc


def read_member_head_bytes(archive_path: str, member: str, n: int) -> bytes:
    """First `n` bytes of a zip member, streamed (archive is NOT extracted)."""
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            with zf.open(member, "r") as fh:
                return fh.read(n)
    except (zipfile.BadZipFile, KeyError, OSError) as exc:
        raise CorruptFileError(f"cannot read member {member} of {archive_path}: {exc}") from exc


@contextlib.contextmanager
def open_binary(meta: FileMeta) -> Iterator[Any]:
    """Yield a streaming *binary* handle for a FileMeta (plain | gzip | zip-member)."""
    if meta.archive_path:
        zf = zipfile.ZipFile(meta.archive_path, "r")
        try:
            fh = zf.open(meta.archive_member, "r")
            yield fh
        except (KeyError, zipfile.BadZipFile, OSError) as exc:
            raise CorruptFileError(f"bad member {meta.archive_member}: {exc}") from exc
        finally:
            with contextlib.suppress(Exception):
                fh.close()  # type: ignore[has-type]
            zf.close()
        return
    if meta.compression is Compression.GZIP:
        try:
            with gzip.open(meta.absolute_path, "rb") as fh:
                yield fh
        except (OSError, EOFError) as exc:
            raise CorruptFileError(f"bad gzip {meta.absolute_path}: {exc}") from exc
        return
    try:
        with open(meta.absolute_path, "rb") as fh:
            yield fh
    except OSError as exc:
        raise CorruptFileError(f"cannot open {meta.absolute_path}: {exc}") from exc


@contextlib.contextmanager
def open_text(meta: FileMeta, encoding: Optional[str] = None) -> Iterator[TextIO]:
    """Yield a decoded text stream for a FileMeta. `errors='replace'` keeps noisy files usable."""
    enc = encoding or meta.encoding or "utf-8"
    with open_binary(meta) as binary:
        wrapper = io.TextIOWrapper(binary, encoding=enc, errors="replace", newline="")
        try:
            yield wrapper
        finally:
            with contextlib.suppress(Exception):
                wrapper.detach()  # don't close underlying handle twice


def iter_text_lines(meta: FileMeta, encoding: Optional[str] = None) -> Iterator[str]:
    """Stream newline-stripped decoded lines. One line resident at a time."""
    with open_text(meta, encoding) as stream:
        for line in stream:
            yield line.rstrip("\r\n")


def iter_jsonl_objects(meta: FileMeta, skip_bad: bool = True) -> Iterator[Dict[str, Any]]:
    """Stream parsed JSON objects from a JSONL file/member, one per line."""
    for line in iter_text_lines(meta):
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            if skip_bad:
                continue
            raise ReaderError(f"invalid JSON line in {meta.key}")
        yield obj if isinstance(obj, dict) else {"_value": obj}


def write_json(path: str, obj: Any) -> None:
    """Atomic small-JSON write (temp + rename) so a crash never leaves a half report."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

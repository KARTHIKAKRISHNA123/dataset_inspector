"""Lightweight MIME-type resolution.

The spec wants a `mime_type` per file. Python's stdlib `mimetypes` covers the common cases by
extension; we override for data formats it doesn't know (jsonl, parquet, gguf, docx) and fall back to
`application/octet-stream` for binary/unknown. Pure-stdlib, no `libmagic` dependency.
"""

from __future__ import annotations

import mimetypes
from typing import Optional

_OVERRIDES = {
    "jsonl": "application/x-ndjson",
    "ndjson": "application/x-ndjson",
    "json": "application/json",
    "csv": "text/csv",
    "tsv": "text/tab-separated-values",
    "tab": "text/tab-separated-values",
    "txt": "text/plain",
    "text": "text/plain",
    "xml": "application/xml",
    "parquet": "application/vnd.apache.parquet",
    "pq": "application/vnd.apache.parquet",
    "gguf": "application/octet-stream",   # binary model container
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "gz": "application/gzip",
    "zip": "application/zip",
}


def guess_mime(filename: str, extension: Optional[str] = None) -> str:
    ext = (extension or filename.rsplit(".", 1)[-1] if "." in filename else "").lower()
    if ext in _OVERRIDES:
        return _OVERRIDES[ext]
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"

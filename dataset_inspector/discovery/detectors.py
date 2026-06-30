"""Pure detection functions for discovery: compression, logical type, encoding, provenance.

All are functions of (path + small byte prefix) so they cost O(1) per file and are trivially unit
tested. Keeping them pure (no I/O beyond a passed-in prefix, no globals) is what lets the same logic
run over a local disk, a zip member, or an S3 object later.
"""

from __future__ import annotations

import gzip
import os
import zipfile
from typing import List, Optional, Tuple

from ..core.models import Compression, FileType

_GZIP_MAGIC = b"\x1f\x8b"
_ZIP_MAGIC = b"PK\x03\x04"
_PARQUET_MAGIC = b"PAR1"
_GGUF_MAGIC = b"GGUF"
_DOCX_MAGIC = b"PK\x03\x04"  # docx is a zip; disambiguated by extension/content

_EXT_TO_TYPE = {
    "txt": FileType.TXT, "text": FileType.TXT,
    "json": FileType.JSON,
    "jsonl": FileType.JSONL, "ndjson": FileType.JSONL,
    "csv": FileType.CSV,
    "tsv": FileType.TSV, "tab": FileType.TSV,
    "parquet": FileType.PARQUET, "pq": FileType.PARQUET,
    "docx": FileType.DOCX,
    "xml": FileType.XML,
    "gguf": FileType.GGUF,
}

# Path tokens → provenance label. Heuristic only.
_SOURCE_HINTS = [
    ("huggingface", "Hugging Face"), ("hf_", "Hugging Face"), ("hf-", "Hugging Face"),
    ("datasets--", "Hugging Face"), ("github", "GitHub"), ("gitlab", "GitLab"),
    ("kaggle", "Kaggle"), ("ai4bharat", "AI4Bharat"), ("aicoach", "AI Coach"),
    ("ai_coach", "AI Coach"), ("gov", "Government"), ("data.gov", "Government"),
    ("zenodo", "Research"), ("arxiv", "Research"), ("opus", "OPUS"),
]


def ext_of(path: str) -> str:
    return os.path.splitext(path)[1].lstrip(".").lower()


def strip_gzip_ext(path: str) -> str:
    return path[:-3] if path.lower().endswith(".gz") else path


def detect_compression(path: str, head: bytes) -> Compression:
    if head[:2] == _GZIP_MAGIC:
        return Compression.GZIP
    # .docx and .zip both start with PK; treat .docx as a logical type, not a container.
    if head[:4] == _ZIP_MAGIC and ext_of(path) != "docx":
        return Compression.ZIP
    e = ext_of(path)
    if e == "gz":
        return Compression.GZIP
    if e == "zip":
        return Compression.ZIP
    return Compression.NONE


def detect_file_type(path: str, compression: Compression, head: bytes) -> FileType:
    name = strip_gzip_ext(path) if compression is Compression.GZIP else path
    e = ext_of(name)
    if e in _EXT_TO_TYPE:
        return _EXT_TO_TYPE[e]
    sample = _decompressed_head(path, compression, head)
    return sniff_content_type(sample)


def _decompressed_head(path: str, compression: Compression, head: bytes) -> bytes:
    if compression is Compression.GZIP:
        try:
            with gzip.open(path, "rb") as fh:
                return fh.read(4096)
        except OSError:
            return head
    return head[:4096]


def sniff_content_type(sample: bytes) -> FileType:
    if sample[:4] == _GGUF_MAGIC:
        return FileType.GGUF
    if sample[:4] == _PARQUET_MAGIC:
        return FileType.PARQUET
    s = sample.lstrip()
    if not s:
        return FileType.UNKNOWN
    if s[:1] == b"<":
        return FileType.XML
    if s[:1] in (b"{", b"["):
        nl = s.find(b"\n")
        if s[:1] == b"{" and nl != -1 and s[nl:].lstrip().startswith(b"{"):
            return FileType.JSONL
        return FileType.JSON
    first_line = s.split(b"\n", 1)[0]
    if b"\t" in first_line:
        return FileType.TSV
    if b"," in first_line:
        return FileType.CSV
    return FileType.TXT


def detect_encoding(raw: bytes, fallback: str = "utf-8") -> Tuple[str, float]:
    if not raw:
        return fallback, 0.0
    try:
        from charset_normalizer import from_bytes  # type: ignore
        best = from_bytes(raw).best()
        if best is not None:
            return (best.encoding or fallback).lower(), max(0.0, min(1.0, 1.0 - float(best.chaos)))
    except Exception:
        pass
    try:
        raw.decode("utf-8")
        return "utf-8", 0.6
    except UnicodeDecodeError:
        return fallback, 0.0


def detect_source(path: str) -> str:
    low = path.lower()
    for token, label in _SOURCE_HINTS:
        if token in low:
            return label
    return "unknown"


def list_zip_members(archive_path: str, data_exts: List[str]) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                e = ext_of(strip_gzip_ext(info.filename))
                if e in data_exts:
                    out.append((info.filename, info.file_size))
    except (zipfile.BadZipFile, OSError):
        return []
    return out

"""Data contracts passed between inspector stages.

`FileMeta` is the output of *discovery* — pure filesystem/byte-level facts about one inspectable unit
(a loose file, or one member inside a .zip). The reader/preview/schema/stats stages enrich a `dict`
report keyed off this. We keep `FileMeta` a dataclass (cheap, typed) and the final report a plain
dict (flexible, matches the JSON output shape the spec asks for).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional

_SLOTS = {"slots": True} if sys.version_info >= (3, 10) else {}


class FileType(str, Enum):
    TXT = "txt"
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    TSV = "tsv"
    PARQUET = "parquet"
    DOCX = "docx"
    XML = "xml"
    GGUF = "gguf"
    UNKNOWN = "unknown"


class Compression(str, Enum):
    NONE = "none"
    GZIP = "gzip"
    ZIP = "zip"


class DatasetType(str, Enum):
    MONOLINGUAL = "Monolingual"
    BILINGUAL = "Bilingual"
    MULTILINGUAL = "Multilingual"
    UNKNOWN = "Unknown"


@dataclass(**_SLOTS)
class FileMeta:
    """Byte/filesystem-level facts about one inspectable unit."""

    file_name: str
    absolute_path: str
    relative_path: str
    parent_dir: str
    dataset: str                      # logical dataset name (first path component under root)
    source: str                       # provenance label (HuggingFace/GitHub/…) — heuristic
    extension: str
    file_type: FileType
    mime_type: str
    compression: Compression
    size_bytes: int
    mtime: float
    encoding: str = "utf-8"
    encoding_confidence: float = 0.0
    sha1_quick: Optional[str] = None
    archive_path: Optional[str] = None    # set when this is a member inside a .zip
    archive_member: Optional[str] = None
    permissions: Optional[str] = None     # octal string, e.g. "0o644"
    error: Optional[str] = None

    @property
    def key(self) -> str:
        """Stable identity for checkpoint/resume."""
        base = f"{self.archive_path}::{self.archive_member}" if self.archive_path else self.absolute_path
        return f"{base}|{self.sha1_quick or ''}"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["file_type"] = self.file_type.value
        d["compression"] = self.compression.value
        return d


@dataclass(**_SLOTS)
class LanguageHints:
    """Hint-only language signals. NOT a final language-ID decision."""

    path_hint: Optional[str] = None
    script_hint: Optional[str] = None
    dominant_script: Optional[str] = None
    script_distribution: Dict[str, float] = field(default_factory=dict)
    mixed_language: bool = False
    romanized: bool = False
    code_switching: bool = False
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(**_SLOTS)
class SchemaInfo:
    """Detected logical structure of a record-oriented file."""

    columns: List[str] = field(default_factory=list)
    n_columns: int = 0
    nested: bool = False
    text_column: Optional[str] = None
    language_column: Optional[str] = None
    source_column: Optional[str] = None
    target_column: Optional[str] = None
    metadata_columns: List[str] = field(default_factory=list)
    estimated_records: Optional[int] = None
    raw_schema: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

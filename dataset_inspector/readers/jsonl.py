"""JSONL / NDJSON reader. One object per line — the easiest format to stream safely."""

from __future__ import annotations

from typing import Any, Dict, Iterator, Optional

from ..core.io import iter_jsonl_objects
from ..core.models import FileMeta, FileType
from .base import BaseReader
from .registry import register_reader


@register_reader
class JsonlReader(BaseReader):
    file_types = (FileType.JSONL,)

    def iter_records(self, meta: FileMeta, max_records: Optional[int] = None
                     ) -> Iterator[Dict[str, Any]]:
        n = 0
        for obj in iter_jsonl_objects(meta, skip_bad=True):
            yield obj
            n += 1
            if max_records is not None and n >= max_records:
                return

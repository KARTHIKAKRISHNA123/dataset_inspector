"""Plain-text reader. Each non-empty line is one record `{"text": line}`. Fully streaming."""

from __future__ import annotations

from typing import Any, Dict, Iterator, Optional

from ..core.io import iter_text_lines
from ..core.models import FileMeta, FileType
from .base import BaseReader
from .registry import register_reader


@register_reader
class TxtReader(BaseReader):
    file_types = (FileType.TXT,)

    def iter_records(self, meta: FileMeta, max_records: Optional[int] = None
                     ) -> Iterator[Dict[str, Any]]:
        n = 0
        for line in iter_text_lines(meta):
            if not line.strip():
                continue
            yield {"text": line}
            n += 1
            if max_records is not None and n >= max_records:
                return

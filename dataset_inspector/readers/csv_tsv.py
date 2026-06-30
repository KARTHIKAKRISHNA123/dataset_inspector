"""CSV / TSV reader. Streams rows via the stdlib `csv` module (lazy over the text handle).

Delimiter handling:
  * TSV → tab, always.
  * CSV → `csv.Sniffer` on a small sample, falling back to ',' if sniffing is inconclusive.

Header handling: we treat the first row as a header when its cells look like field names (non-empty,
mostly non-numeric). Otherwise we synthesise `col_0, col_1, …`. `csv.reader` correctly handles
quoted fields containing commas/newlines, so we never mis-split.
"""

from __future__ import annotations

import csv
import io as _io
from typing import Any, Dict, Iterator, List, Optional

from ..core.io import open_text
from ..core.models import FileMeta, FileType
from .base import BaseReader
from .registry import register_reader

# csv has a global field-size limit that small by default; raise it for long text cells.
try:
    csv.field_size_limit(10 * 1024 * 1024)
except OverflowError:  # pragma: no cover - platform dependent
    csv.field_size_limit(2 ** 31 - 1)


def _looks_like_header(row: List[str]) -> bool:
    if not row:
        return False
    nonempty = [c for c in row if c.strip() != ""]
    if not nonempty:
        return False
    numericish = sum(1 for c in nonempty if c.strip().replace(".", "", 1).lstrip("-").isdigit())
    return numericish <= len(nonempty) // 2  # header if at most half the cells are numbers


@register_reader
class CsvTsvReader(BaseReader):
    file_types = (FileType.CSV, FileType.TSV)

    def _delimiter(self, meta: FileMeta) -> str:
        if meta.file_type is FileType.TSV:
            return "\t"
        with open_text(meta) as s:
            sample = s.read(16384)
        try:
            return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            return ","

    def _header(self, meta: FileMeta, delimiter: str):
        with open_text(meta) as s:
            reader = csv.reader(s, delimiter=delimiter)
            for row in reader:
                if any(c.strip() for c in row):
                    if _looks_like_header(row):
                        return [c.strip() or f"col_{i}" for i, c in enumerate(row)], True
                    return [f"col_{i}" for i in range(len(row))], False
        return [], False

    def iter_records(self, meta: FileMeta, max_records: Optional[int] = None
                     ) -> Iterator[Dict[str, Any]]:
        delimiter = self._delimiter(meta)
        header, has_header = self._header(meta, delimiter)
        n = 0
        with open_text(meta) as s:
            reader = csv.reader(s, delimiter=delimiter)
            for i, row in enumerate(reader):
                if i == 0 and has_header:
                    continue
                if not any(c.strip() for c in row):
                    continue
                if header and len(row) <= len(header):
                    rec = {header[j]: row[j] for j in range(len(row))}
                elif header:
                    rec = {header[j]: row[j] for j in range(len(header))}
                    rec["_extra"] = delimiter.join(row[len(header):])
                else:
                    rec = {f"col_{j}": v for j, v in enumerate(row)}
                yield rec
                n += 1
                if max_records is not None and n >= max_records:
                    return

    def describe(self, meta: FileMeta) -> Dict[str, Any]:
        delimiter = self._delimiter(meta)
        header, has_header = self._header(meta, delimiter)
        return {
            "delimiter": "\\t" if delimiter == "\t" else delimiter,
            "has_header": has_header,
            "columns": header,
            "n_columns": len(header),
        }

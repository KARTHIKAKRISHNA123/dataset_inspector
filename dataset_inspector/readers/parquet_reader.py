"""Parquet reader — row-group / batch streaming via pyarrow. NEVER builds a full DataFrame.

Why parquet needs special care: a parquet file is columnar and can be hundreds of GB. The forbidden
move is `pandas.read_parquet(path)` / `table.to_pandas()`, which materialises every row in RAM. The
correct move is to read the **footer metadata** (schema + exact row count — free, no data scan) and
then iterate **record batches** lazily, touching only as many rows as the caller asks for.

  * `describe()`     → schema, columns, exact `num_rows` (from footer), row-group count. O(1) I/O.
  * `iter_records()` → `ParquetFile.iter_batches(batch_size=…)`, converted to python dicts batch by
                       batch with `max_records` honoured — so a preview reads exactly one small batch.

Limitations (documented honestly):
  * Random-access parquet needs a *seekable* file. Plain `.parquet` files are seekable; a parquet
    stored *inside* a .zip member or gzipped is not cheaply seekable, so for those we emit a clear
    note instead of buffering the whole member into RAM. (In practice datasets ship parquet
    uncompressed-at-the-container level, because parquet already compresses internally.)
  * If `pyarrow` isn't installed, the reader degrades to a metadata-only note rather than crashing
    the whole inspection — install with `pip install ".[parquet]"`.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, Optional

from ..core.models import FileMeta, FileType
from ..logging_utils import get_logger
from .base import BaseReader
from .registry import register_reader

_log = get_logger("readers.parquet")


def _pyarrow():
    """Import pyarrow.parquet lazily so the dependency is optional."""
    try:
        import pyarrow.parquet as pq  # type: ignore
        return pq
    except Exception:
        return None


@register_reader
class ParquetReader(BaseReader):
    file_types = (FileType.PARQUET,)

    def _seekable(self, meta: FileMeta) -> bool:
        # Only plain on-disk parquet is cheaply seekable for row-group reads.
        return meta.archive_member is None and meta.compression.value == "none"

    def iter_records(self, meta: FileMeta, max_records: Optional[int] = None
                     ) -> Iterator[Dict[str, Any]]:
        pq = _pyarrow()
        if pq is None or not self._seekable(meta):
            return  # describe() reports why; nothing to stream
        try:
            pf = pq.ParquetFile(meta.absolute_path)
        except Exception as exc:  # noqa: BLE001 - corrupt/unsupported parquet → skip gracefully
            _log.warning("parquet open failed %s: %s", meta.key, exc)
            return
        batch_size = max_records if (max_records and max_records < 1024) else 1024
        emitted = 0
        try:
            for batch in pf.iter_batches(batch_size=batch_size):
                cols = batch.schema.names
                columns = [batch.column(i).to_pylist() for i in range(len(cols))]
                for row_idx in range(batch.num_rows):
                    rec = {cols[c]: columns[c][row_idx] for c in range(len(cols))}
                    yield rec
                    emitted += 1
                    if max_records is not None and emitted >= max_records:
                        return
        except Exception as exc:  # noqa: BLE001
            _log.warning("parquet read error %s: %s", meta.key, exc)
            return

    def describe(self, meta: FileMeta) -> Dict[str, Any]:
        pq = _pyarrow()
        if pq is None:
            return {"parquet": "pyarrow not installed (pip install '.[parquet]') — metadata skipped"}
        if not self._seekable(meta):
            return {"parquet": "compressed/in-archive parquet not seekable for streaming preview"}
        try:
            pf = pq.ParquetFile(meta.absolute_path)
            schema = pf.schema_arrow
            columns = list(schema.names)
            types = {name: str(schema.field(name).type) for name in columns}
            return {
                "columns": columns,
                "n_columns": len(columns),
                "num_rows": pf.metadata.num_rows,          # EXACT, from footer (no data scan)
                "num_row_groups": pf.num_row_groups,
                "arrow_schema": types,
            }
        except Exception as exc:  # noqa: BLE001
            return {"parquet_error": str(exc)}

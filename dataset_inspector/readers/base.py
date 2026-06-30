"""The reader plugin interface.

A reader's single responsibility: turn ONE `FileMeta` into a lazy stream of **records** (dicts), and
optionally describe its structure. It does NOT decide which column is "text", classify the dataset,
or compute statistics — those are separate stages that consume the record stream. This separation is
what makes the plugin set small and each reader testable in isolation.

Two methods:
  * `iter_records(meta)` — REQUIRED. Lazy generator of dict records. For line formats, wrap each line
    as `{"text": line}`. MUST NOT load the whole file.
  * `describe(meta)`     — OPTIONAL. Cheap structural facts that only the reader can know without
    streaming the records (e.g. parquet schema from the footer, gguf metadata header, xml root tag).
    Default: empty dict.

`max_records` lets callers bound how much a reader yields (preview/stats sample) so even a 1 TB file
is touched only as much as needed.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Iterator, Optional, Sequence

from ..config import Config
from ..core.models import FileMeta, FileType


class BaseReader(abc.ABC):
    file_types: Sequence[FileType] = ()

    def __init__(self, config: Config):
        self.config = config

    def handles(self, meta: FileMeta) -> bool:
        return meta.file_type in self.file_types

    @abc.abstractmethod
    def iter_records(self, meta: FileMeta, max_records: Optional[int] = None
                     ) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError

    def describe(self, meta: FileMeta) -> Dict[str, Any]:
        return {}

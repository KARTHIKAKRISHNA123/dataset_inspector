"""Capability-based reader registry (the OCP/DIP seam).

Readers register with `@register_reader`. The registry instantiates them and resolves the right one
for a FileMeta by asking each `handles()`. Adding a format = write a reader + decorate it; nothing
here changes.
"""

from __future__ import annotations

from typing import List, Optional, Type

from ..config import Config
from ..core.exceptions import UnsupportedFormatError
from ..core.models import FileMeta
from .base import BaseReader

_REGISTRY: List[Type[BaseReader]] = []


def register_reader(cls: Type[BaseReader]) -> Type[BaseReader]:
    if cls not in _REGISTRY:
        _REGISTRY.append(cls)
    return cls


class ReaderRegistry:
    def __init__(self, config: Config):
        self.config = config
        self._instances: List[BaseReader] = [cls(config) for cls in _REGISTRY]

    def resolve(self, meta: FileMeta) -> BaseReader:
        for r in self._instances:
            if r.handles(meta):
                return r
        raise UnsupportedFormatError(f"no reader for {meta.file_type.value} ({meta.key})")

    def try_resolve(self, meta: FileMeta) -> Optional[BaseReader]:
        try:
            return self.resolve(meta)
        except UnsupportedFormatError:
            return None

    def registered(self) -> List[str]:
        return [type(r).__name__ for r in self._instances]

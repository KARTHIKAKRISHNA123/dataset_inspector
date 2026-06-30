"""Append-only resume log (crash-safe). Same robust design used across the pipeline.

Before inspecting a file we check `is_done(key)`; after writing its report we `mark_done(key)`.
The key includes the quick content hash, so a *changed* file (same path, new bytes) is re-inspected.
"""

from __future__ import annotations

import os
from typing import Set

from .exceptions import CheckpointError


class CheckpointManager:
    def __init__(self, path: str):
        self.path = path
        self._done: Set[str] = set()
        self._fh = None
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                for line in fh:
                    k = line.rstrip("\n")
                    if k:
                        self._done.add(k)
        except OSError as exc:
            raise CheckpointError(f"cannot read checkpoint {self.path}: {exc}") from exc

    def is_done(self, key: str) -> bool:
        return key in self._done

    def mark_done(self, key: str) -> None:
        if key in self._done:
            return
        if self._fh is None:
            self._fh = open(self.path, "a", encoding="utf-8")
        self._fh.write(key + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._done.add(key)

    def count(self) -> int:
        return len(self._done)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

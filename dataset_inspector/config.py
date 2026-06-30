"""YAML configuration with dotted-path access, deep-merge, and `--set key=value` overrides.

Loaded once at startup and dependency-injected into stages. Light by design (no pydantic): the
inspector reads config constantly on hot paths, so we avoid per-access validation overhead and just
validate required keys explicitly where needed.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional

import yaml

from .core.exceptions import ConfigError


class Config:
    def __init__(self, data: Dict[str, Any]):
        self._data = data

    @classmethod
    def load(cls, default_path: str, override_path: Optional[str] = None,
             overrides: Optional[List[str]] = None) -> "Config":
        data = cls._read(default_path)
        if override_path:
            data = cls._merge(data, cls._read(override_path))
        for kv in overrides or []:
            cls._apply(data, kv)
        return cls(data)

    @staticmethod
    def _read(path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            raise ConfigError(f"config not found: {path}")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"bad YAML {path}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ConfigError(f"top-level YAML in {path} must be a mapping")
        return loaded

    @staticmethod
    def _merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
        out = copy.deepcopy(base)
        for k, v in over.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = Config._merge(out[k], v)
            else:
                out[k] = v
        return out

    @staticmethod
    def _apply(data: Dict[str, Any], kv: str) -> None:
        if "=" not in kv:
            raise ConfigError(f"--set expects key=value, got {kv!r}")
        key, raw = kv.split("=", 1)
        value = yaml.safe_load(raw)
        node = data
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ConfigError(f"cannot set {key}: {p} not a mapping")
        node[parts[-1]] = value

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def section(self, name: str) -> Dict[str, Any]:
        sec = self.get(name, {})
        return sec if isinstance(sec, dict) else {}

    @property
    def raw(self) -> Dict[str, Any]:
        return self._data

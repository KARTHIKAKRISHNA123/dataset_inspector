"""Structured logging + a Heartbeat for progress/memory lines during long inspections."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from typing import Optional

try:
    import psutil  # type: ignore
    _PROC = psutil.Process(os.getpid())
except Exception:  # pragma: no cover
    _PROC = None


def current_rss_mb() -> Optional[float]:
    if _PROC is None:
        return None
    try:
        return _PROC.memory_info().rss / (1024 * 1024)
    except Exception:  # pragma: no cover
        return None


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO", file: Optional[str] = None,
                  json_format: bool = False) -> logging.Logger:
    logger = logging.getLogger("dataset_inspector")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False
    fmt: logging.Formatter = _JsonFormatter() if json_format else logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%H:%M:%S")
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    logger.addHandler(console)
    if file:
        os.makedirs(os.path.dirname(os.path.abspath(file)), exist_ok=True)
        fh = RotatingFileHandler(file, maxBytes=50 * 1024 * 1024, backupCount=5, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"dataset_inspector.{name}")


class Heartbeat:
    """O(1)-memory progress reporter. Emits a line every `every` items."""

    def __init__(self, logger: logging.Logger, every: int = 200, unit: str = "files",
                 log_memory: bool = True):
        self.logger = logger
        self.every = max(1, every)
        self.unit = unit
        self.log_memory = log_memory
        self.count = 0
        self._start = time.time()
        self._last = self._start

    def tick(self, n: int = 1) -> None:
        self.count += n
        if self.count % self.every == 0:
            now = time.time()
            rate = self.every / max(1e-9, now - self._last)
            mem = current_rss_mb() if self.log_memory else None
            extra = f" | rss={mem:.0f}MB" if mem is not None else ""
            self.logger.info("… %s %s | %.0f %s/s%s", f"{self.count:,}", self.unit, rate,
                             self.unit, extra)
            self._last = now

    def done(self) -> None:
        elapsed = time.time() - self._start
        self.logger.info("done: %s %s in %.1fs (%.0f %s/s)", f"{self.count:,}", self.unit,
                         elapsed, self.count / max(1e-9, elapsed), self.unit)

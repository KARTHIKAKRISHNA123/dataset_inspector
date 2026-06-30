"""GGUF reader — parse the metadata HEADER only; never read tensors. Stdlib `struct` only.

Why GGUF is handled completely differently from every other format
------------------------------------------------------------------
GGUF (the llama.cpp model container) is a **binary model file**, not a text dataset. The bytes are
mostly quantised tensor weights — gigabytes of numbers that are meaningless to a tokenizer-data
inspector and ruinous to load. So this reader produces **zero text records** and instead reads just
the small, self-describing **metadata header** at the front of the file:

    magic "GGUF" | version (u32) | tensor_count (u64) | metadata_kv_count (u64) | [ KV pairs … ]

We parse the KV pairs (architecture, quantization, context length, tokenizer info, and the size of
the token vocabulary) and then **STOP** — we never reach, read, or decode the tensor data section.
A hard byte budget caps how much of even the metadata we read, so a hostile/huge header can't blow
up memory. Large arrays (e.g. the 32k-entry token list) are *counted* but not stored — we keep only
a small sample, advancing the stream without retaining the bytes.

This is the concrete realisation of the rule "never load GGUF tensors into RAM": we treat GGUF as a
metadata-only format by design.
"""

from __future__ import annotations

import struct
from typing import Any, Dict, Iterator, List, Optional

from ..core.io import open_binary
from ..core.models import FileMeta, FileType
from ..logging_utils import get_logger
from .base import BaseReader
from .registry import register_reader

_log = get_logger("readers.gguf")

_MAGIC = b"GGUF"
_BUDGET_BYTES = 32 * 1024 * 1024      # never read more than 32 MiB of header, ever
_ARRAY_SAMPLE = 8                      # store at most this many elements of any array

# Fixed-width GGUF metadata value types → (struct format, byte size).
_SCALAR = {
    0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4), 5: ("i", 4),
    6: ("f", 4), 7: ("?", 1), 10: ("Q", 8), 11: ("q", 8), 12: ("d", 8),
}
_TYPE_STRING = 8
_TYPE_ARRAY = 9


class _Budget:
    """Tracks bytes read from the stream and aborts past the cap (defensive against bad headers)."""

    def __init__(self, stream, limit: int):
        self.s = stream
        self.limit = limit
        self.used = 0

    def read(self, n: int) -> bytes:
        if self.used + n > self.limit:
            raise _Truncated()
        b = self.s.read(n)
        if len(b) < n:
            raise _Truncated()
        self.used += len(b)
        return b


class _Truncated(Exception):
    """Raised when the metadata exceeds the byte budget or the file ends early."""


def _u(buf: _Budget, fmt: str, size: int) -> Any:
    return struct.unpack("<" + fmt, buf.read(size))[0]


def _gguf_string(buf: _Budget) -> str:
    n = _u(buf, "Q", 8)
    if n > buf.limit:                  # absurd length → corrupt
        raise _Truncated()
    return buf.read(n).decode("utf-8", "replace")


def _read_array(buf: _Budget) -> Dict[str, Any]:
    elem_type = _u(buf, "I", 4)
    count = _u(buf, "Q", 8)
    sample: List[Any] = []
    if elem_type == _TYPE_STRING:
        for i in range(count):
            s = _gguf_string(buf)               # must read to advance the stream
            if i < _ARRAY_SAMPLE:
                sample.append(s)
        return {"elem": "string", "len": count, "sample": sample}
    if elem_type in _SCALAR:
        fmt, size = _SCALAR[elem_type]
        for i in range(count):
            v = _u(buf, fmt, size)
            if i < _ARRAY_SAMPLE:
                sample.append(v)
        return {"elem": f"scalar:{elem_type}", "len": count, "sample": sample}
    # nested arrays are not used in practice; bail clearly
    raise _Truncated()


def _read_value(buf: _Budget, vtype: int) -> Any:
    if vtype == _TYPE_STRING:
        return _gguf_string(buf)
    if vtype == _TYPE_ARRAY:
        return _read_array(buf)
    if vtype in _SCALAR:
        fmt, size = _SCALAR[vtype]
        return _u(buf, fmt, size)
    raise _Truncated()


@register_reader
class GgufReader(BaseReader):
    file_types = (FileType.GGUF,)

    def iter_records(self, meta: FileMeta, max_records: Optional[int] = None
                     ) -> Iterator[Dict[str, Any]]:
        # GGUF carries no tokenizer-training text. Intentionally yields nothing.
        return iter(())

    def describe(self, meta: FileMeta) -> Dict[str, Any]:
        try:
            with open_binary(meta) as raw:
                buf = _Budget(raw, _BUDGET_BYTES)
                magic = buf.read(4)
                if magic != _MAGIC:
                    return {"gguf_error": f"bad magic {magic!r} (not a GGUF file)"}
                version = _u(buf, "I", 4)
                tensor_count = _u(buf, "Q", 8)
                kv_count = _u(buf, "Q", 8)

                metadata: Dict[str, Any] = {}
                keys: List[str] = []
                truncated = False
                for _ in range(kv_count):
                    try:
                        key = _gguf_string(buf)
                        vtype = _u(buf, "I", 4)
                        value = _read_value(buf, vtype)
                    except _Truncated:
                        truncated = True
                        break
                    keys.append(key)
                    # Keep scalars + array summaries; arrays already store only a small sample.
                    metadata[key] = value

                return self._summarise(version, tensor_count, kv_count, keys, metadata, truncated)
        except _Truncated:
            return {"gguf_error": "metadata exceeded byte budget or file truncated"}
        except Exception as exc:  # noqa: BLE001
            _log.warning("gguf parse failed %s: %s", meta.key, exc)
            return {"gguf_error": str(exc)}

    @staticmethod
    def _summarise(version, tensor_count, kv_count, keys, metadata, truncated) -> Dict[str, Any]:
        def g(*names):
            for n in names:
                if n in metadata:
                    return metadata[n]
            return None

        tokens = metadata.get("tokenizer.ggml.tokens")
        vocab_size = tokens["len"] if isinstance(tokens, dict) and "len" in tokens else \
            g("tokenizer.ggml.vocab_size", "llama.vocab_size")
        arch = g("general.architecture")
        ctx = g(f"{arch}.context_length" if arch else "", "llama.context_length")
        emb = g(f"{arch}.embedding_length" if arch else "", "llama.embedding_length")
        tok_keys = [k for k in keys if k.startswith("tokenizer.")]
        return {
            "gguf_version": version,
            "architecture": arch,
            "model_name": g("general.name"),
            "model_type": g("general.type"),
            "quantization_version": g("general.quantization_version"),
            "file_type": g("general.file_type"),
            "context_length": ctx,
            "embedding_length": emb,
            "vocab_size": vocab_size,
            "tensor_count": tensor_count,
            "metadata_kv_count": kv_count,
            "tokenizer_metadata_keys": tok_keys,
            "metadata_keys": keys[:60],
            "metadata_truncated": truncated,
        }

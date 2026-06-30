"""Streaming JSON reader.

JSON is the hardest format to stream because it's a single self-delimited document. A naive
`json.load(open(f))` loads the *entire* file — forbidden here. We implement a small character-level
state machine that yields **top-level array elements one at a time**, holding only the current
element in memory. This covers the overwhelmingly common dataset shape:

    [ {...}, {...}, {...}, ... ]              # array of records  → true streaming

For an object document `{ "data": [ ... ], "meta": {...} }` we read a *bounded* prefix (cap), parse
it, and stream the first array-valued field's elements. A genuinely huge single-object JSON is rare
for datasets; if one exceeds the cap we emit a clearly-flagged bounded preview record rather than
exhausting RAM. (To make even that case fully streaming, drop in an `ijson` based reader later — no
other code changes, thanks to the plugin registry.)

Complexity: array path is O(bytes) time, O(largest element) memory.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List, Optional

from ..core.io import open_text
from ..core.models import FileMeta, FileType
from .base import BaseReader
from .registry import register_reader

_ARRAY_KEYS = ["data", "rows", "records", "examples", "train", "items",
               "translations", "sentences", "documents"]
_DEFAULT_OBJECT_CAP = 8 * 1024 * 1024  # 8 MiB bounded read for object-shaped JSON


def _char_stream(stream, chunk: int) -> Iterator[str]:
    while True:
        buf = stream.read(chunk)
        if not buf:
            return
        for ch in buf:
            yield ch


def _loads(fragment: str) -> Optional[Dict[str, Any]]:
    fragment = fragment.strip()
    if not fragment:
        return None
    try:
        val = json.loads(fragment)
    except json.JSONDecodeError:
        return None
    if isinstance(val, dict):
        return val
    return {"_value": val}


def _stream_array(chars: Iterator[str], max_records: Optional[int]) -> Iterator[Dict[str, Any]]:
    """Yield elements of an array whose opening '[' has already been consumed."""
    buf: List[str] = []
    depth = 0
    in_str = False
    esc = False
    count = 0
    for ch in chars:
        if in_str:
            buf.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            buf.append(ch)
            continue
        if ch in "{[":
            depth += 1
            buf.append(ch)
            continue
        if ch == "]" and depth == 0:
            rec = _loads("".join(buf))
            if rec is not None:
                yield rec
            return
        if ch in "}]":
            depth -= 1
            buf.append(ch)
            continue
        if ch == "," and depth == 0:
            rec = _loads("".join(buf))
            buf = []
            if rec is not None:
                yield rec
                count += 1
                if max_records is not None and count >= max_records:
                    return
            continue
        buf.append(ch)
    # truncated stream (no closing ]) — flush whatever we have
    rec = _loads("".join(buf))
    if rec is not None:
        yield rec


@register_reader
class JsonReader(BaseReader):
    file_types = (FileType.JSON,)

    def iter_records(self, meta: FileMeta, max_records: Optional[int] = None
                     ) -> Iterator[Dict[str, Any]]:
        chunk = int(self.config.get("io.read_chunk_bytes", 1 << 20))
        with open_text(meta) as stream:
            chars = _char_stream(stream, chunk)
            first = None
            for ch in chars:
                if not ch.isspace():
                    first = ch
                    break
            if first == "[":
                yield from _stream_array(chars, max_records)
                return
        # Object-shaped (or scalar) document → bounded re-read.
        yield from self._object_records(meta, max_records)

    def _object_records(self, meta: FileMeta, max_records: Optional[int]
                        ) -> Iterator[Dict[str, Any]]:
        cap = int(self.config.get("io.json_object_cap", _DEFAULT_OBJECT_CAP))
        with open_text(meta) as s:
            data = s.read(cap + 1)
        if len(data) > cap:
            yield {"_note": "large top-level JSON object; bounded preview only "
                            "(install ijson reader for full streaming)",
                   "_partial": data[:1000]}
            return
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            yield {"_note": "unparseable JSON document", "_partial": data[:500]}
            return
        if isinstance(obj, list):
            count = 0
            for el in obj:
                yield el if isinstance(el, dict) else {"_value": el}
                count += 1
                if max_records is not None and count >= max_records:
                    return
            return
        if isinstance(obj, dict):
            # stream the first array-valued field (records); prefer known keys.
            keys = _ARRAY_KEYS + [k for k in obj.keys() if k not in _ARRAY_KEYS]
            for k in keys:
                v = obj.get(k)
                if isinstance(v, list) and v:
                    count = 0
                    for el in v:
                        yield el if isinstance(el, dict) else {"text": el}
                        count += 1
                        if max_records is not None and count >= max_records:
                            return
                    return
            yield obj  # no array field → the object itself is the single record
            return
        yield {"_value": obj}

    def describe(self, meta: FileMeta) -> Dict[str, Any]:
        # Shape hint without a full parse: peek the first non-space char.
        with open_text(meta) as s:
            head = s.read(2048).lstrip()
        shape = "array" if head[:1] == "[" else ("object" if head[:1] == "{" else "scalar")
        return {"json_shape": shape}

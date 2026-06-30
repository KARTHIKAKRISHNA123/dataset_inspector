"""XML reader — streaming via `xml.etree.ElementTree.iterparse`; malformed-safe; stdlib only.

The forbidden move is `ET.parse(path)`, which builds the WHOLE element tree in RAM — impossible for
a multi-GB XML dump (e.g. a Wikipedia export). Instead we use **iterparse**, an event-based pull
parser that yields elements as they close, letting us extract each element's text and then
**`elem.clear()`** it so the tree never accumulates. Memory stays O(current element subtree).

  * `iter_records()` → for every leaf-ish element with non-whitespace text, yield `{"text": ..,
    "tag": localname}`. We `clear()` processed elements and prune already-finished children of the
    root to keep memory flat across millions of nodes.
  * `describe()`     → root tag, namespaces, a bounded tag-frequency summary, and the tags that most
    often carry text (candidate "text elements").

Malformed XML: `iterparse` raises `ParseError` partway through. We catch it, log, and return what we
parsed so far — a truncated/garbled file yields a partial preview instead of killing the run.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, Optional
from xml.etree import ElementTree as ET

from ..core.io import open_binary
from ..core.models import FileMeta, FileType
from ..logging_utils import get_logger
from .base import BaseReader
from .registry import register_reader

_log = get_logger("readers.xml")


def _localname(tag: str) -> str:
    """Strip the `{namespace}` prefix etree puts on tags: '{http://x}row' → 'row'."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


@register_reader
class XmlReader(BaseReader):
    file_types = (FileType.XML,)

    def iter_records(self, meta: FileMeta, max_records: Optional[int] = None
                     ) -> Iterator[Dict[str, Any]]:
        n = 0
        try:
            with open_binary(meta) as raw:
                root = None
                for event, elem in ET.iterparse(raw, events=("start", "end")):
                    if event == "start" and root is None:
                        root = elem
                        continue
                    if event != "end":
                        continue
                    text = (elem.text or "").strip()
                    if text:
                        yield {"text": text, "tag": _localname(elem.tag)}
                        n += 1
                        if max_records is not None and n >= max_records:
                            elem.clear()
                            return
                    elem.clear()  # free this element's memory
                    # prune finished children off the root so it never grows unbounded
                    if root is not None and elem is not root:
                        for child in list(root):
                            if child is not elem:
                                root.remove(child)
                            break
        except ET.ParseError as exc:
            _log.warning("malformed XML %s: %s (returning partial)", meta.key, exc)
            return
        except Exception as exc:  # noqa: BLE001
            _log.warning("xml read error %s: %s", meta.key, exc)
            return

    def describe(self, meta: FileMeta) -> Dict[str, Any]:
        root_tag: Optional[str] = None
        namespaces: Dict[str, str] = {}
        tag_counts: Dict[str, int] = {}
        text_tags: Dict[str, int] = {}
        cap = 20000  # bounded scan for the structural summary
        seen = 0
        try:
            with open_binary(meta) as raw:
                for event, payload in ET.iterparse(raw, events=("start", "start-ns", "end")):
                    if event == "start-ns":
                        prefix, uri = payload
                        namespaces[prefix or "(default)"] = uri
                        continue
                    if event == "start":
                        if root_tag is None:
                            root_tag = _localname(payload.tag)
                        continue
                    # end
                    name = _localname(payload.tag)
                    tag_counts[name] = tag_counts.get(name, 0) + 1
                    if (payload.text or "").strip():
                        text_tags[name] = text_tags.get(name, 0) + 1
                    payload.clear()
                    seen += 1
                    if seen >= cap:
                        break
        except ET.ParseError as exc:
            return {"xml_error": f"malformed: {exc}", "root": root_tag}
        except Exception as exc:  # noqa: BLE001
            return {"xml_error": str(exc)}
        top_text = sorted(text_tags.items(), key=lambda kv: -kv[1])[:5]
        return {
            "root": root_tag,
            "namespaces": namespaces,
            "distinct_tags": len(tag_counts),
            "tag_frequency_sample": dict(sorted(tag_counts.items(), key=lambda kv: -kv[1])[:10]),
            "text_elements": [t for t, _ in top_text],
            "scanned_elements": seen,
        }

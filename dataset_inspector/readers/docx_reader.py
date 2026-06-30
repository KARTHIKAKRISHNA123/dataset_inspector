"""DOCX reader — extract reading-order text from Word documents; ignore images/drawings.

A .docx is a ZIP of XML parts; the body text lives in `word/document.xml`. We use `python-docx`,
which parses that part and exposes paragraphs and tables in document order. We yield:
  * headings and paragraphs as `{"text": ...}` records (skipping empty paragraphs), then
  * table cells row by row.
Images, charts, and drawings carry no tokenizer-relevant text, so we skip them (python-docx simply
doesn't surface them as paragraph text).

Honest limitations:
  * python-docx loads the document's XML into memory. For Word documents that's fine — a .docx is
    typically KB–MB, tiny next to the corpora this tool inspects. (A multi-GB .docx is not a thing.)
  * For a .docx stored *inside* a .zip member we'd need to extract it first; we skip those with a
    note rather than buffering, keeping the no-extract rule. Loose .docx files are the normal case.
  * Footnotes/endnotes and hyperlink URLs are out of scope for the *preview* (configurable handling
    belongs in the downstream cleaning phase, not the inspector).
  * Corrupt/partial .docx → caught and reported via the orchestrator's error field, never fatal.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, Optional

from ..core.models import FileMeta, FileType
from ..logging_utils import get_logger
from .base import BaseReader
from .registry import register_reader

_log = get_logger("readers.docx")


def _docx():
    try:
        import docx  # type: ignore  (python-docx)
        return docx
    except Exception:
        return None


@register_reader
class DocxReader(BaseReader):
    file_types = (FileType.DOCX,)

    def _loadable(self, meta: FileMeta) -> bool:
        return meta.archive_member is None  # loose .docx only (no in-zip extraction)

    def _open(self, meta: FileMeta):
        docx = _docx()
        if docx is None or not self._loadable(meta):
            return None
        try:
            return docx.Document(meta.absolute_path)
        except Exception as exc:  # noqa: BLE001 - corrupt docx
            _log.warning("docx open failed %s: %s", meta.key, exc)
            return None

    def iter_records(self, meta: FileMeta, max_records: Optional[int] = None
                     ) -> Iterator[Dict[str, Any]]:
        document = self._open(meta)
        if document is None:
            return
        n = 0
        # Paragraphs (headings + body) in document order.
        for para in document.paragraphs:
            text = (para.text or "").strip()
            if not text:
                continue
            style = getattr(getattr(para, "style", None), "name", "") or ""
            yield {"text": text, "style": style}
            n += 1
            if max_records is not None and n >= max_records:
                return
        # Then table cell text, row by row.
        for table in document.tables:
            for row in table.rows:
                cells = [(c.text or "").strip() for c in row.cells]
                line = " | ".join(c for c in cells if c)
                if not line:
                    continue
                yield {"text": line, "style": "table"}
                n += 1
                if max_records is not None and n >= max_records:
                    return

    def describe(self, meta: FileMeta) -> Dict[str, Any]:
        if _docx() is None:
            return {"docx": "python-docx not installed (pip install '.[docx]') — metadata skipped"}
        if not self._loadable(meta):
            return {"docx": "in-archive .docx skipped (no extraction)"}
        document = self._open(meta)
        if document is None:
            return {"docx_error": "could not open document"}
        headings = [p.text.strip() for p in document.paragraphs
                    if (getattr(getattr(p, "style", None), "name", "") or "").startswith("Heading")
                    and p.text.strip()]
        return {
            "n_paragraphs": sum(1 for p in document.paragraphs if p.text.strip()),
            "n_tables": len(document.tables),
            "headings_sample": headings[:5],
        }

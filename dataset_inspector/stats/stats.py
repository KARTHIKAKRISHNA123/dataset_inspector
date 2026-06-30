"""StreamingStats — accumulates dataset statistics over a record stream with bounded memory.

Everything here is an *online* (single-pass) statistic. We never keep the records: we keep running
counters and a *bounded* set of text hashes for a duplicate estimate. So whether a dataset has 10k or
10 billion records, the memory used by this accumulator is capped by config, not by the data.

Metrics produced (all explained in docs/phase notes):
  * n_records, missing (empty/absent text)
  * text length min / max / average (characters)
  * duplicate_estimate (exact-duplicate texts seen, bounded sampler)
  * script_distribution (normalised, from a character sample per record)
  * unicode: replacement_chars (U+FFFD → decoding damage), control_chars
  * encoding_quality = 1 - replacement_ratio  (1.0 = clean, lower = mojibake)
  * languages_seen (distinct values of a language field, if present)
"""

from __future__ import annotations

import hashlib
import unicodedata
from typing import Any, Dict, List, Optional

from ..core.scripts import script_histogram

# Field names whose value we treat as "the text" for length/script stats.
_DEFAULT_TEXT_FIELDS = ["text", "sentence", "content", "body", "sentence1", "src",
                        "source", "translation", "line", "raw"]


def extract_text(rec: Dict[str, Any], text_fields: Optional[List[str]] = None) -> str:
    """Pick the most likely text value from a record, robustly, per-record.

    Strategy: first configured field that holds a non-empty string; else the longest string value in
    the record (handles unknown schemas); else "". This is intentionally schema-agnostic so stats
    work even before/without global column detection.
    """
    fields = text_fields or _DEFAULT_TEXT_FIELDS
    for f in fields:
        v = rec.get(f)
        if isinstance(v, str) and v.strip():
            return v
    # fallback: longest string value (covers translation pairs, unknown keys)
    best = ""
    for v in rec.values():
        if isinstance(v, str) and len(v) > len(best):
            best = v
    return best


class StreamingStats:
    def __init__(self, config, language_field: Optional[str] = None):
        self.cfg = config
        self.text_fields: List[str] = list(config.get("schema.text_fields", _DEFAULT_TEXT_FIELDS))
        self.sample_chars = int(config.get("stats.sample_chars_for_script", 2000))
        self.dup_cap = 200000  # bounded hash sampler size
        self.language_field = language_field

        self.n = 0
        self.missing = 0
        self.len_min: Optional[int] = None
        self.len_max = 0
        self.len_sum = 0
        self.script_counts: Dict[str, int] = {}
        self.replacement = 0
        self.control = 0
        self.char_total = 0
        self._hashes: set = set()
        self.dup_estimate = 0
        self.dup_sampler_full = False
        self.languages_seen: set = set()

    def observe(self, rec: Dict[str, Any]) -> None:
        self.n += 1
        if self.language_field:
            lv = rec.get(self.language_field)
            if isinstance(lv, str) and lv.strip():
                self.languages_seen.add(lv.strip())

        text = extract_text(rec, self.text_fields)
        if not text:
            self.missing += 1
            return

        ln = len(text)
        self.len_min = ln if self.len_min is None else min(self.len_min, ln)
        self.len_max = max(self.len_max, ln)
        self.len_sum += ln

        # unicode quality on a bounded slice
        sample = text[: self.sample_chars]
        self.char_total += len(sample)
        for ch in sample:
            if ch == "�":
                self.replacement += 1
            elif unicodedata.category(ch) == "Cc" and ch not in "\t\n\r":
                self.control += 1
        for sc, c in script_histogram(sample).items():
            self.script_counts[sc] = self.script_counts.get(sc, 0) + c

        # bounded duplicate sampler
        if not self.dup_sampler_full:
            h = hashlib.sha1(text.encode("utf-8", "replace")).digest()
            if h in self._hashes:
                self.dup_estimate += 1
            else:
                self._hashes.add(h)
                if len(self._hashes) >= self.dup_cap:
                    self.dup_sampler_full = True

    def finalize(self) -> Dict[str, Any]:
        non_missing = self.n - self.missing
        avg = round(self.len_sum / non_missing, 2) if non_missing else 0.0
        total_scripts = sum(self.script_counts.values())
        dist = ({k: round(v / total_scripts, 4)
                 for k, v in sorted(self.script_counts.items(), key=lambda kv: -kv[1])}
                if total_scripts else {})
        enc_q = round(1.0 - (self.replacement / self.char_total), 4) if self.char_total else 1.0
        return {
            "records_sampled": self.n,
            "missing_text": self.missing,
            "text_length_min": self.len_min or 0,
            "text_length_max": self.len_max,
            "text_length_avg": avg,
            "duplicate_estimate": self.dup_estimate,
            "duplicate_sampler_saturated": self.dup_sampler_full,
            "script_distribution": dist,
            "unicode_replacement_chars": self.replacement,
            "unicode_control_chars": self.control,
            "encoding_quality": enc_q,
            "languages_seen": sorted(self.languages_seen)[:50],
        }

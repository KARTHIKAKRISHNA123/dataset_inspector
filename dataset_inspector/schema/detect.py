"""Schema detection + dataset-type classification over a bounded record sample.

Two responsibilities, both streaming and bounded-memory:

  * `SchemaSampler` — observe records one at a time and learn the *structure*: the union of column
    names, whether any value is nested (dict/list), inferred per-column types, and which columns play
    the special roles (text / language / source / target / metadata). It keeps only small dicts keyed
    by column name, never the records — so memory is O(#columns), not O(#records).

  * `classify_dataset` — turn the observed schema + streaming statistics + language hints into a
    Monolingual / Bilingual / Multilingual / Unknown label WITH a confidence and a human-readable
    reasoning string. This is deliberately explainable: the spec asks us to justify the call.

Why classification lives next to schema: the strongest signal for "how many languages" is structural
(a `language` column's distinct values, or the presence of source+target translation columns), which
is exactly what `SchemaSampler` discovers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..config import Config
from ..core.models import DatasetType, SchemaInfo

_DEFAULT_TEXT = ["text", "sentence", "content", "body", "sentence1", "src", "source",
                 "translation", "line", "raw"]
_DEFAULT_LANG = ["language", "lang", "lang_id", "language_code", "locale", "lid"]
_DEFAULT_SRC = ["source", "src", "source_text", "src_text", "en"]
_DEFAULT_TGT = ["target", "tgt", "target_text", "tgt_text", "translation"]


def _type_name(v: Any) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        return "list"
    if isinstance(v, dict):
        return "dict"
    if v is None:
        return "null"
    return type(v).__name__


class SchemaSampler:
    """Accumulates structural facts over a record stream. O(#columns) memory."""

    def __init__(self, config: Config):
        self.text_fields: List[str] = list(config.get("schema.text_fields", _DEFAULT_TEXT))
        self.language_fields: List[str] = list(config.get("schema.language_fields", _DEFAULT_LANG))
        self.source_fields: List[str] = list(config.get("schema.source_fields", _DEFAULT_SRC))
        self.target_fields: List[str] = list(config.get("schema.target_fields", _DEFAULT_TGT))
        self._columns: Dict[str, int] = {}        # insertion-ordered: column -> #records seen in
        self._types: Dict[str, str] = {}          # column -> first inferred type
        self.nested = False
        self.n = 0

    def observe(self, rec: Dict[str, Any]) -> None:
        self.n += 1
        if not isinstance(rec, dict):
            return
        for k, v in rec.items():
            self._columns[k] = self._columns.get(k, 0) + 1
            if k not in self._types:
                self._types[k] = _type_name(v)
            if isinstance(v, (dict, list)):
                self.nested = True

    def _pick(self, candidates: List[str]) -> Optional[str]:
        """First candidate field actually present, matched case-insensitively."""
        lower = {c.lower(): c for c in self._columns}
        for cand in candidates:
            if cand.lower() in lower:
                return lower[cand.lower()]
        return None

    def finalize(self, estimated_records: Optional[int] = None) -> SchemaInfo:
        cols = list(self._columns.keys())
        text_col = self._pick(self.text_fields)
        lang_col = self._pick(self.language_fields)
        src_col = self._pick(self.source_fields)
        tgt_col = self._pick(self.target_fields)
        # A column can be both "source" (en) and "text"; keep source/target distinct from metadata.
        special = {c for c in (text_col, lang_col, src_col, tgt_col) if c}
        metadata_cols = [c for c in cols if c not in special]
        return SchemaInfo(
            columns=cols,
            n_columns=len(cols),
            nested=self.nested,
            text_column=text_col,
            language_column=lang_col,
            source_column=src_col,
            target_column=tgt_col,
            metadata_columns=metadata_cols,
            estimated_records=estimated_records,
            raw_schema=dict(self._types),
        )


def classify_dataset(schema: SchemaInfo, stats: Dict[str, Any], hints: Dict[str, Any],
                     config: Config) -> Tuple[DatasetType, float, str, List[str]]:
    """Decide Monolingual/Bilingual/Multilingual/Unknown with confidence + reasoning.

    Evidence, strongest first:
      1. distinct values of a `language` column (structural, most reliable);
      2. presence of source+target translation columns → at least bilingual;
      3. number of distinct *scripts* holding a meaningful share (a script can map to several
         languages, so this is a lower bound on language count).
    """
    multi_min = int(config.get("classification.multilingual_min_scripts", 3))
    bi_n = int(config.get("classification.bilingual_scripts", 2))

    langs_seen: List[str] = list(stats.get("languages_seen", []) or [])
    scripts = [s for s, frac in (stats.get("script_distribution", {}) or {}).items()
               if isinstance(frac, (int, float)) and frac >= 0.15]
    has_pairs = bool(schema.source_column and schema.target_column)

    reasons: List[str] = []
    # Pick the dominant evidence count.
    if langs_seen:
        signal = len(langs_seen)
        reasons.append(f"`{schema.language_column}` column holds {signal} distinct language value(s): "
                       f"{', '.join(langs_seen[:8])}")
        evidence = "language-field"
    elif has_pairs:
        signal = 2
        reasons.append(f"translation columns present (source=`{schema.source_column}`, "
                       f"target=`{schema.target_column}`) → ≥2 languages")
        evidence = "translation-pair"
    else:
        signal = len(scripts)
        if scripts:
            reasons.append(f"{len(scripts)} script(s) with ≥15% share: {', '.join(scripts)}")
        else:
            reasons.append("no language field, no translation columns, no scripted text observed")
        evidence = "script-distribution"

    # Map signal → type.
    if signal >= multi_min:
        dtype = DatasetType.MULTILINGUAL
    elif signal == bi_n or (has_pairs and signal >= 2):
        dtype = DatasetType.BILINGUAL
    elif signal == 1:
        dtype = DatasetType.MONOLINGUAL
    else:
        dtype = DatasetType.UNKNOWN

    # Confidence: structural evidence is trusted more than pure script counts.
    base = {"language-field": 0.85, "translation-pair": 0.8, "script-distribution": 0.55}[evidence]
    if dtype is DatasetType.UNKNOWN:
        base = 0.2
    # If script evidence corroborates the language-field count, nudge confidence up.
    if evidence == "language-field" and len(scripts) >= min(signal, multi_min):
        base = min(0.95, base + 0.05)
        reasons.append(f"corroborated by {len(scripts)} distinct script(s)")

    # Languages we can name: prefer language-field values; else script-unique hints + path hint.
    languages_named = sorted(set(langs_seen))
    if not languages_named:
        for s in scripts:
            from ..core.scripts import unique_language_for_script
            u = unique_language_for_script(s)
            if u:
                languages_named.append(u)
        ph = hints.get("path_hint")
        if ph and ph not in languages_named:
            languages_named.append(ph)
        languages_named = sorted(set(languages_named))

    reasoning = "; ".join(reasons)
    return dtype, round(base, 2), reasoning, languages_named

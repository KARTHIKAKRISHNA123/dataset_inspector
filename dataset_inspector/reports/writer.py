"""ReportWriter — persist per-file reports and accumulate the master inventory.

Two output tiers, matching the spec:
  * Per file  : `outputs/reports/<dataset>/<file>.json` and `.md` (full detail + human view).
  * Master    : `outputs/master_inventory.{json,csv,md}` (one flat row per inspected file).

Streaming discipline: the master CSV and a master JSONL are **appended per file** as we go, so even
if the process dies, the partial master is on disk and consistent. The compact `master_inventory.json`
and `.md` are rendered at `finalize()` from small flat rows (a few dozen fields each). We keep those
flat rows in memory because the count is bounded by the number of *files* (hundreds–thousands), not
records (billions) — a deliberate, documented trade-off.
"""

from __future__ import annotations

import csv
import os
import re
from typing import Any, Dict, List

from ..core.io import write_json
from ..logging_utils import get_logger

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_name(name: str) -> str:
    s = _SAFE.sub("_", name).strip("_")
    return s or "file"


_FLAT_FIELDS = [
    "dataset", "file_name", "source", "file_type", "compression", "encoding",
    "language_hint", "dataset_type", "classification_confidence", "n_columns",
    "estimated_records", "records_sampled", "text_length_avg", "encoding_quality",
    "dominant_script", "mixed_language", "code_switching", "romanized",
    "errors", "relative_path",
]


class ReportWriter:
    def __init__(self, config, outdir: str):
        self.cfg = config
        self.log = get_logger("reports")
        self.per_dataset_dir = os.path.join(outdir, "reports")
        self.master_dir = outdir
        os.makedirs(self.per_dataset_dir, exist_ok=True)
        os.makedirs(self.master_dir, exist_ok=True)
        self.master_json = os.path.join(self.master_dir, "master_inventory.json")
        self.master_csv = os.path.join(self.master_dir, "master_inventory.csv")
        self.master_md = os.path.join(self.master_dir, "master_inventory.md")
        self.master_jsonl = os.path.join(self.master_dir, "master_inventory.jsonl")
        self.rows: List[Dict[str, Any]] = []
        self._csv_fh = open(self.master_csv, "w", encoding="utf-8", newline="")
        self._csv_writer = csv.DictWriter(self._csv_fh, fieldnames=_FLAT_FIELDS,
                                          extrasaction="ignore")
        self._csv_writer.writeheader()
        self._jsonl_fh = open(self.master_jsonl, "w", encoding="utf-8")

    # ------------------------------------------------------------------
    def write_report(self, report: Dict[str, Any]) -> None:
        dataset = _safe_name(report.get("dataset", "(root)"))
        fname = _safe_name(report.get("file_name", "file"))
        ddir = os.path.join(self.per_dataset_dir, dataset)
        os.makedirs(ddir, exist_ok=True)
        # uniquify in case two members share a name within a dataset
        base = os.path.join(ddir, fname)
        json_path = self._unique(base + ".json")
        write_json(json_path, report)
        self._write_markdown(json_path[:-5] + ".md", report)

        flat = self._flatten(report)
        self.rows.append(flat)
        self._csv_writer.writerow(flat)
        self._csv_fh.flush()
        import json as _json
        self._jsonl_fh.write(_json.dumps(flat, ensure_ascii=False) + "\n")
        self._jsonl_fh.flush()

    def _unique(self, path: str) -> str:
        if not os.path.exists(path):
            return path
        stem, ext = os.path.splitext(path)
        i = 1
        while os.path.exists(f"{stem}_{i}{ext}"):
            i += 1
        return f"{stem}_{i}{ext}"

    # ------------------------------------------------------------------
    def _flatten(self, r: Dict[str, Any]) -> Dict[str, Any]:
        stats = r.get("statistics", {}) or {}
        cls = r.get("classification", {}) or {}
        hints = r.get("language_hints", {}) or {}
        schema = r.get("schema", {}) or {}
        return {
            "dataset": r.get("dataset", ""),
            "file_name": r.get("file_name", ""),
            "source": r.get("source", ""),
            "file_type": r.get("file_type", ""),
            "compression": r.get("compression", ""),
            "encoding": r.get("encoding", ""),
            "language_hint": r.get("language_hint", "") or "",
            "dataset_type": r.get("dataset_type", ""),
            "classification_confidence": cls.get("confidence", ""),
            "n_columns": schema.get("n_columns", ""),
            "estimated_records": stats.get("estimated_records", "") if stats.get(
                "estimated_records") is not None else "",
            "records_sampled": stats.get("records_sampled", ""),
            "text_length_avg": stats.get("text_length_avg", ""),
            "encoding_quality": stats.get("encoding_quality", ""),
            "dominant_script": hints.get("dominant_script", "") or "",
            "mixed_language": hints.get("mixed_language", ""),
            "code_switching": hints.get("code_switching", ""),
            "romanized": hints.get("romanized", ""),
            "errors": len(r.get("errors", []) or []),
            "relative_path": (r.get("meta", {}) or {}).get("relative_path", ""),
        }

    # ------------------------------------------------------------------
    def _write_markdown(self, path: str, r: Dict[str, Any]) -> None:
        stats = r.get("statistics", {}) or {}
        cls = r.get("classification", {}) or {}
        hints = r.get("language_hints", {}) or {}
        schema = r.get("schema", {}) or {}
        prev = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(r.get("preview", []))) or "  (none)"
        md = [
            f"# {r.get('file_name')}  ({r.get('dataset')})",
            "",
            f"- **Source:** {r.get('source')}",
            f"- **Type / compression / encoding:** {r.get('file_type')} / "
            f"{r.get('compression')} / {r.get('encoding')}",
            f"- **Dataset type:** {r.get('dataset_type')} "
            f"(confidence {cls.get('confidence')}) — {cls.get('reasoning')}",
            f"- **Language hint:** {r.get('language_hint')} "
            f"(dominant script: {hints.get('dominant_script')}, "
            f"mixed={hints.get('mixed_language')}, code-switch={hints.get('code_switching')}, "
            f"romanized={hints.get('romanized')})",
            f"- **Columns ({schema.get('n_columns')}):** {', '.join(schema.get('columns', []))}",
            f"- **Text column:** {schema.get('text_column')} | "
            f"**Language column:** {schema.get('language_column')} | "
            f"**Src/Tgt:** {schema.get('source_column')}/{schema.get('target_column')}",
            "",
            "## Preview",
            prev,
            "",
            "## Statistics",
            f"- records sampled: {stats.get('records_sampled')}",
            f"- estimated records: {stats.get('estimated_records')}",
            f"- text length (min/avg/max): {stats.get('text_length_min')} / "
            f"{stats.get('text_length_avg')} / {stats.get('text_length_max')}",
            f"- missing text: {stats.get('missing_text')} | duplicate estimate: "
            f"{stats.get('duplicate_estimate')}",
            f"- encoding quality: {stats.get('encoding_quality')} | "
            f"script distribution: {stats.get('script_distribution')}",
        ]
        if r.get("format_metadata"):
            md += ["", "## Format metadata", "```json",
                   _pretty(r["format_metadata"]), "```"]
        if r.get("errors"):
            md += ["", "## Errors", *[f"- {e}" for e in r["errors"]]]
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(md))

    # ------------------------------------------------------------------
    def finalize(self) -> Dict[str, Any]:
        summary = self._summarise()
        write_json(self.master_json, {"summary": summary, "files": self.rows})
        self._write_master_md(summary)
        self._csv_fh.close()
        self._jsonl_fh.close()
        self.log.info("master inventory → %s (.json/.csv/.md/.jsonl)", self.master_dir)
        return summary

    def _summarise(self) -> Dict[str, Any]:
        from collections import Counter
        by_type: Counter = Counter()
        by_dtype: Counter = Counter()
        by_source: Counter = Counter()
        by_lang: Counter = Counter()
        for row in self.rows:
            by_type[row["file_type"]] += 1
            by_dtype[row["dataset_type"]] += 1
            by_source[row["source"]] += 1
            by_lang[row["language_hint"] or "und"] += 1
        return {
            "n_files": len(self.rows),
            "by_file_type": dict(by_type.most_common()),
            "by_dataset_type": dict(by_dtype.most_common()),
            "by_source": dict(by_source.most_common()),
            "by_language_hint": dict(by_lang.most_common()),
        }

    def _write_master_md(self, summary: Dict[str, Any]) -> None:
        lines = ["# Master Inventory", "",
                 f"**Files inspected:** {summary['n_files']}", ""]

        def block(title: str, d: Dict[str, Any]) -> None:
            lines.append(f"## {title}\n")
            lines.append("| key | count |")
            lines.append("|---|---|")
            for k, v in d.items():
                lines.append(f"| {k} | {v} |")
            lines.append("")

        block("By file type", summary["by_file_type"])
        block("By dataset type", summary["by_dataset_type"])
        block("By source", summary["by_source"])
        block("By language hint", summary["by_language_hint"])

        lines.append("## Files\n")
        lines.append("| dataset | file | type | comp | lang | dataset_type | conf | rows~ | errs |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in self.rows:
            lines.append(
                f"| {r['dataset']} | {r['file_name']} | {r['file_type']} | {r['compression']} | "
                f"{r['language_hint']} | {r['dataset_type']} | {r['classification_confidence']} | "
                f"{r['estimated_records']} | {r['errors']} |")
        with open(self.master_md, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))


def _pretty(obj: Any) -> str:
    import json
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)[:4000]
    except Exception:
        return str(obj)[:4000]

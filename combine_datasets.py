"""Combine every inspected dataset's TEXT into ONE streaming JSONL (memory-safe).

Reads the inspector's per-file reports (outputs/reports/**/*.json) for the absolute path + the
detected text column, opens each original file via the plugin readers, and appends
{"text": ..., "source": ...} to a single output file — one record at a time, never loading a whole
dataset into RAM.

Output is JSONL (one JSON object per line), NOT a single JSON array: a 20M-record array can't be
streamed or opened; JSONL can. Every tokenizer/`datasets` loader reads JSONL fine.

Run:
    cd /d/Intern/dataset-inspector && source .venv/Scripts/activate
    python combine_datasets.py --outputs ./outputs --out ./outputs/combined_dataset.jsonl
"""
from __future__ import annotations

import argparse
import json
import os

from dataset_inspector.config import Config
from dataset_inspector.core.models import Compression, FileMeta, FileType
from dataset_inspector.readers import ReaderRegistry


def _default_config() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "config", "default.yaml")


def _meta_from_report(r: dict) -> FileMeta | None:
    m = r.get("meta", {}) or {}
    ap = m.get("absolute_path")
    if not ap:
        return None
    try:
        ft = FileType(r.get("file_type") or m.get("file_type", "unknown"))
    except ValueError:
        ft = FileType.UNKNOWN
    try:
        comp = Compression(r.get("compression") or m.get("compression", "none"))
    except ValueError:
        comp = Compression.NONE
    name = os.path.basename(m.get("archive_member") or ap)
    return FileMeta(
        file_name=name, absolute_path=ap,
        relative_path=m.get("relative_path", name), parent_dir=os.path.dirname(ap),
        dataset=r.get("dataset") or m.get("dataset", ""), source=r.get("source") or "",
        extension=os.path.splitext(name)[1].lstrip("."), file_type=ft,
        mime_type="application/octet-stream", compression=comp, size_bytes=0, mtime=0.0,
        encoding=r.get("encoding") or m.get("encoding", "utf-8"),
        archive_path=m.get("archive_path"), archive_member=m.get("archive_member"),
    )


def _text_of(rec: dict, text_col: str | None) -> str | None:
    if text_col:
        v = rec.get(text_col)
        if isinstance(v, str) and v.strip():
            return v
        low = {k.lower(): k for k in rec}
        k = low.get(text_col.lower())
        v = rec.get(k) if k else None
        if isinstance(v, str) and v.strip():
            return v
    best = None
    for v in rec.values():
        if isinstance(v, str) and v.strip() and (best is None or len(v) > len(best)):
            best = v
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="./outputs", help="inspector outputs/ directory")
    ap.add_argument("--out", default=None, help="combined output path (.jsonl)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    reports_dir = os.path.join(args.outputs, "reports")
    out_path = args.out or os.path.join(args.outputs, "combined_dataset.jsonl")
    cfg = Config.load(args.config or _default_config())
    reg = ReaderRegistry(cfg)

    n_files = n_records = 0
    with open(out_path, "w", encoding="utf-8") as w:
        for root, _dirs, files in os.walk(reports_dir):
            for fn in sorted(files):
                if not fn.endswith(".json"):
                    continue
                try:
                    r = json.load(open(os.path.join(root, fn), encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                meta = _meta_from_report(r)
                if meta is None or not os.path.exists(meta.absolute_path):
                    continue
                reader = reg.try_resolve(meta)
                if reader is None:
                    continue
                text_col = (r.get("schema", {}) or {}).get("text_column")
                buf = []
                try:
                    for rec in reader.iter_records(meta):
                        t = _text_of(rec, text_col)
                        if t and t.strip():
                            buf.append(json.dumps(
                                {"text": t, "source": meta.relative_path}, ensure_ascii=False))
                            n_records += 1
                            if len(buf) >= 2000:
                                w.write("\n".join(buf) + "\n"); buf.clear()
                except Exception as exc:  # noqa: BLE001 - one bad file never stops the merge
                    print(f"  skip {meta.key}: {exc}")
                if buf:
                    w.write("\n".join(buf) + "\n")
                n_files += 1
                if n_files % 500 == 0:
                    print(f"  {n_files} files, {n_records:,} records -> {out_path}")
    print(f"done: {n_files} files, {n_records:,} records -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

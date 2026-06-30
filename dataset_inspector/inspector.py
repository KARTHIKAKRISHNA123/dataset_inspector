"""DatasetInspector — the orchestrator that turns one `FileMeta` into one complete report dict.

It composes the independent stages (reader → schema → statistics → language hints → classification)
without any of them knowing about each other. Each `inspect()` call is bounded-memory and makes at
most two short, *streaming* passes over the file, each capped by config:

    Pass 1 (schema + preview + hint-sample): read up to `schema_probe` records to learn columns,
            the language column, the first N preview texts, and a small text sample for script hints.
    Pass 2 (statistics): read up to `stats.max_records_sampled` records through the online
            StreamingStats accumulator.

Re-reading the first K records twice costs K record-reads, never the whole file — so a 100 GB JSONL
is touched only as much as the caps allow. Binary/metadata formats (GGUF, and parquet/docx when their
optional deps are absent) yield no text; we still emit their `describe()` metadata.

Failure policy: any reader exception is caught, recorded in `report["errors"]`, and inspection
continues — one broken file never aborts the batch (that's the orchestrator's job, per the exception
hierarchy).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .config import Config
from .core.exceptions import ReaderError
from .core.models import FileMeta, SchemaInfo
from .logging_utils import get_logger
from .readers import ReaderRegistry
from .schema.detect import SchemaSampler, classify_dataset
from .stats.stats import StreamingStats, extract_text
from .classify.language import path_language_hint, language_hints_from_text


class DatasetInspector:
    def __init__(self, config: Config):
        self.cfg = config
        self.registry = ReaderRegistry(config)
        self.log = get_logger("inspector")
        self.preview_n = int(config.get("preview.n_records", 2))
        self.max_chars = int(config.get("preview.max_chars_per_record", 500))
        self.schema_probe = max(self.preview_n, 2000)
        self.stats_cap = int(config.get("stats.max_records_sampled", 100000))
        self.hint_sample_chars = 4000

    # ------------------------------------------------------------------
    def inspect(self, meta: FileMeta) -> Dict[str, Any]:
        errors: List[str] = []
        if meta.error:
            errors.append(meta.error)

        reader = self.registry.try_resolve(meta)
        format_metadata: Dict[str, Any] = {}
        preview: List[str] = []
        schema_info = SchemaInfo()
        stats_dict: Dict[str, Any] = {}

        if reader is None:
            errors.append(f"unsupported format: {meta.file_type.value}")
            hints = language_hints_from_text("", path_language_hint(
                meta.archive_member or meta.absolute_path, meta.archive_path))
            return self._assemble(meta, schema_info, stats_dict, hints, preview,
                                  format_metadata, errors)

        # describe(): cheap format-specific metadata (parquet schema, gguf header, xml root, …)
        try:
            format_metadata = reader.describe(meta) or {}
        except Exception as exc:  # noqa: BLE001
            errors.append(f"describe failed: {exc}")

        # ---- Pass 1: schema + preview + hint text sample --------------------------------
        sampler = SchemaSampler(self.cfg)
        hint_buf: List[str] = []
        hint_len = 0
        sampled_bytes = 0
        try:
            for i, rec in enumerate(reader.iter_records(meta, max_records=self.schema_probe)):
                sampler.observe(rec)
                text = extract_text(rec)
                if text:
                    sampled_bytes += len(text)
                    if len(preview) < self.preview_n:
                        preview.append(text[: self.max_chars])
                    if hint_len < self.hint_sample_chars:
                        take = text[: self.hint_sample_chars - hint_len]
                        hint_buf.append(take)
                        hint_len += len(take)
        except ReaderError as exc:
            errors.append(f"read error (schema pass): {exc}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"unexpected error (schema pass): {exc}")

        est_records = self._estimate_records(meta, format_metadata, sampler.n, sampled_bytes)
        schema_info = sampler.finalize(estimated_records=est_records)

        # ---- language hints (path + script of the sampled text) -------------------------
        path_hint = path_language_hint(meta.archive_member or meta.absolute_path, meta.archive_path)
        hints = language_hints_from_text("".join(hint_buf), path_hint)

        # ---- Pass 2: streaming statistics ------------------------------------------------
        stats = StreamingStats(self.cfg, language_field=schema_info.language_column)
        try:
            for rec in reader.iter_records(meta, max_records=self.stats_cap):
                stats.observe(rec)
        except ReaderError as exc:
            errors.append(f"read error (stats pass): {exc}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"unexpected error (stats pass): {exc}")
        stats_dict = stats.finalize()
        stats_dict["file_size_bytes"] = meta.size_bytes
        stats_dict["estimated_records"] = est_records

        return self._assemble(meta, schema_info, stats_dict, hints, preview,
                              format_metadata, errors)

    # ------------------------------------------------------------------
    def _estimate_records(self, meta: FileMeta, fmt_meta: Dict[str, Any],
                          n_sampled: int, sampled_text_bytes: int) -> Optional[int]:
        # Exact when the format tells us (parquet footer).
        if isinstance(fmt_meta.get("num_rows"), int):
            return int(fmt_meta["num_rows"])
        # Cannot estimate reliably from a compressed file's on-disk size.
        if meta.compression.value != "none" or meta.archive_member:
            return None
        if n_sampled <= 0 or sampled_text_bytes <= 0:
            return None
        avg = sampled_text_bytes / n_sampled
        if avg <= 0:
            return None
        return int(meta.size_bytes / avg)

    def _assemble(self, meta: FileMeta, schema: SchemaInfo, stats: Dict[str, Any],
                  hints, preview: List[str], fmt_meta: Dict[str, Any],
                  errors: List[str]) -> Dict[str, Any]:
        dtype, conf, reasoning, langs = classify_dataset(
            schema, stats, hints.to_dict(), self.cfg)
        return {
            # headline fields (match the spec's output example) ---------------------------
            "dataset": meta.dataset,
            "file_name": meta.file_name,
            "source": meta.source,
            "file_type": meta.file_type.value,
            "compression": meta.compression.value,
            "encoding": meta.encoding,
            "language_hint": hints.path_hint or hints.script_hint,
            "dataset_type": dtype.value,
            "columns": schema.columns,
            "preview": preview,
            "statistics": stats,
            "schema": schema.to_dict(),
            # richer detail ---------------------------------------------------------------
            "classification": {
                "dataset_type": dtype.value,
                "confidence": conf,
                "reasoning": reasoning,
                "languages_seen": langs,
            },
            "language_hints": hints.to_dict(),
            "format_metadata": fmt_meta,
            "meta": meta.to_dict(),
            "errors": errors,
        }

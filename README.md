# dataset-inspector for multilingual bpe pipeline

A **standalone, streaming, disk-backed** tool that *inspects and profiles* heterogeneous datasets
**without preprocessing them**. Point it at a folder of 100+ datasets (HuggingFace dumps, GitHub
repos, research/government corpora, archives) and it produces, for every file, a complete picture:
metadata, format-aware preview, schema, dataset-type classification, language hints, and statistics —
**never loading a whole dataset into memory.**

This is the *understanding* layer that comes before any multilingual NLP pipeline. It is intentionally
independent of any tokenizer/training code.

## Core guarantee

> **Disk is primary storage; RAM is a temporary buffer.** No file is ever fully loaded. Parquet is
> read by row-group, XML by `iterparse`, GGUF reads only its metadata header (never tensors), ZIP
> members are streamed (never extracted), GZIP is streamed (never fully decompressed), JSON/JSONL are
> read record-by-record. Inspection is **resumable** via checkpoints.

## Supported formats

Built in: `txt, json, jsonl, csv, tsv, parquet, docx, xml, gguf`, plus `gzip` and `zip` containers.

Designed to extend (one new reader file, zero edits elsewhere): PDF, EPUB, HTML, Markdown, TAR,
TAR.GZ, Feather, Arrow, Avro, SQLite, LMDB.

## What it produces

Per file: a JSON + Markdown report under `outputs/reports/`.
Across the run: `outputs/master_inventory.{json,csv,md}`.

## Quick start

```bash
# 1. create a virtualenv
python -m venv .venv
. .venv/Scripts/activate          # Windows (PowerShell: .venv\Scripts\Activate.ps1)
# source .venv/bin/activate       # macOS/Linux

# 2. install. Core install handles txt/json/jsonl/csv/tsv/xml/gguf with ZERO extra deps.
pip install -e .                  # minimal
pip install -e ".[full]"          # adds parquet (pyarrow), docx (python-docx), psutil memory logs

# 3. inspect a tree of datasets (point --input at your Downloads folder)
python -m dataset_inspector --input "C:/Users/<you>/Downloads" --outdir ./outputs
# after install the console script also works:
dataset-inspect --input ./data --outdir ./outputs
```

Useful flags: `--limit 50` (smoke-test the first 50 files), `--no-resume` (ignore checkpoint),
`--set stats.max_records_sampled=20000` (override any config knob inline), `--config <path>`.

Re-running resumes automatically: files already inspected (keyed by path + 64 KiB content hash) are
skipped, so a crashed run continues where it stopped.

### Outputs

```
outputs/
├── master_inventory.json     summary + one flat row per file
├── master_inventory.csv      same rows, spreadsheet-friendly
├── master_inventory.md       human-readable tables
├── master_inventory.jsonl    streamed one-row-per-file (crash-safe)
├── reports/<dataset>/<file>.json   full per-file report (preview, schema, stats, classification)
├── reports/<dataset>/<file>.md     human-readable per-file report
├── logs/inspector.log
└── _checkpoints/inspect.done       resume log
```

### Run the tests

```bash
pip install -e ".[dev]"
pytest                # 19 unit + integration tests
```

## Layout

```
dataset-inspector/
├── config/default.yaml        all tunable knobs (chunk sizes, sample caps, thresholds)
├── pyproject.toml             packaging; console script `dataset-inspect`; optional dep groups
├── dataset_inspector/         the package
│   ├── __main__.py            enables `python -m dataset_inspector`
│   ├── cli.py                 argument parsing + the streaming, resumable run loop
│   ├── config.py              YAML config (dotted access, deep-merge, --set overrides)
│   ├── logging_utils.py       structured logging + throughput/memory Heartbeat
│   ├── inspector.py           ORCHESTRATOR: FileMeta → full report (2 bounded streaming passes)
│   ├── core/                  models, exceptions, streaming io, checkpoint, scripts, mime
│   ├── discovery/             recursive streaming file discovery + metadata (detectors + walker)
│   ├── readers/               PLUGIN format readers (base + registry + one file per format)
│   ├── schema/                schema/column detection + dataset-type classification
│   ├── classify/              language hints (script/path/mixed/romanized/code-switch)
│   ├── stats/                 streaming, bounded-memory statistics accumulator
│   └── reports/               per-file (json/md) + master inventory (json/csv/md/jsonl) writers
├── outputs/                   generated reports + checkpoints (gitignored)
└── tests/                     unit + reader + integration tests
```

Preview generation is part of the readers + orchestrator (the orchestrator asks each reader for the
first N records), so there is no separate `preview/` package — adding one would split a single
responsibility across two places.

### Adding a new format (the plugin contract)

1. Create `dataset_inspector/readers/<fmt>_reader.py`.
2. Subclass `BaseReader`, set `file_types = (FileType.X,)`, implement `iter_records()` (lazy!) and
   optionally `describe()`; decorate the class with `@register_reader`.
3. Add one import line to `dataset_inspector/readers/__init__.py`.

Nothing else changes — the registry resolves your reader by capability. (Add the new extension to
`FileType` + `discovery/detectors._EXT_TO_TYPE` if it's a brand-new logical type.)

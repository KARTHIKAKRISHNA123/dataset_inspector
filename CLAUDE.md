# CLAUDE.md — dataset-inspector

Context for AI assistants (and humans) working in this repo. Read this first.

## What this project is
A **standalone, streaming, plugin-based Dataset Inspector**. Point it at a folder of heterogeneous
datasets and it produces per-file **metadata reports** (type, compression, encoding, schema, detected
columns, preview, statistics, script/language *hints*, classification) — **without preprocessing** and
**without ever loading a whole file into RAM**. It is the *discovery* stage; a separate pipeline
(`../indic-multilingual-bpe-pipeline`) consumes its output.

## Core invariants (do not break)
1. **Streaming only.** Never `json.load`/`read()` a whole file, never build a full DataFrame, never
   extract a ZIP or fully decompress a GZIP. Readers are generators; Parquet is row-group read; GGUF
   reads only its header; ZIP members are opened as streams.
2. **Hints, not decisions.** The inspector emits *language hints* (path + Unicode script). Final
   language ID is NOT its job.
3. **Resumable.** Every run is checkpointed (append-only done-log); re-running skips finished files.
4. **Plugin readers (Open/Closed).** Add a format = add one `*_reader.py` that subclasses `BaseReader`
   and is decorated `@register_reader`. `readers/__init__.py` auto-imports all sibling modules, so no
   other file changes. Never add a big if/elif over extensions.

## Structure
```
dataset-inspector/
├── pyproject.toml            # console script: dataset-inspect = dataset_inspector.cli:main
├── config/default.yaml       # all knobs (extensions, sample bytes, thresholds)
├── dataset_inspector/
│   ├── __main__.py           # enables `python -m dataset_inspector`
│   ├── cli.py                # arg parsing + streaming run loop (walk → inspect → write reports)
│   ├── inspector.py          # ORCHESTRATOR: FileMeta → full report dict (2 bounded streaming passes)
│   ├── config.py             # YAML config, dotted access, --set overrides
│   ├── logging_utils.py      # structured logging + Heartbeat (throughput/memory)
│   ├── core/
│   │   ├── models.py         # FileMeta, FileType, Compression, SchemaInfo, LanguageHints (dataclasses)
│   │   ├── exceptions.py     # exception hierarchy (skip-vs-abort policy)
│   │   ├── io.py             # streaming primitives: read_head_bytes, open_binary/open_text, iter_jsonl
│   │   ├── checkpoint.py     # append-only resume log
│   │   ├── scripts.py        # Unicode script ranges (script ≠ language)
│   │   └── mime.py           # MIME resolution
│   ├── discovery/
│   │   ├── detectors.py      # pure fns: compression/type/encoding/source (magic bytes first)
│   │   └── walker.py         # DatasetWalker: recursive streaming walk → FileMeta generator
│   ├── readers/              # PLUGIN readers (auto-registered)
│   │   ├── base.py           # BaseReader ABC: iter_records() + describe()
│   │   ├── registry.py       # capability-based resolver
│   │   ├── txt.py jsonl.py json_reader.py csv_tsv.py
│   │   ├── parquet_reader.py # row-group streaming (pyarrow, optional)
│   │   ├── docx_reader.py    # python-docx (optional)
│   │   ├── xml_reader.py     # stdlib iterparse, malformed-safe
│   │   └── gguf_reader.py    # header metadata only, never tensors (stdlib struct)
│   ├── schema/detect.py      # SchemaSampler (columns/roles) + classify_dataset (mono/bi/multi)
│   ├── classify/language.py  # path + script language hints
│   └── stats/stats.py        # StreamingStats (online counters, bounded memory)
├── tests/                    # conftest + test_unit + test_readers + test_integration
└── outputs/                  # generated reports (gitignored)
```

## Outputs (what a run produces)
```
outputs/
├── master_inventory.{json,csv,md,jsonl}     # one flat row per inspected file
└── reports/<dataset>/<file>.json|.md        # full per-file report (the pipeline reads these)
```
Each report's `meta.absolute_path` + `schema.text_column/language_column/source_column/target_column`
are the fields the downstream pipeline uses to open and canonicalize the original files.

## Run
```bash
pip install -e ".[full]"     # core works with zero extra deps; full adds parquet+docx
python -m dataset_inspector --input "C:/Users/<you>/Downloads" --outdir ./outputs
pytest                        # unit + integration tests
```
Flags: `--limit N` (smoke test), `--no-resume`, `--set key=value`, `--config <yaml>`.

## Gotchas
- 8 readers must register (`ReaderRegistry(Config({})).registered()` → 8). If fewer, `readers/__init__.py`
  auto-discovery didn't run.
- fastText is NOT used here (LID is the pipeline's job). Optional deps: pyarrow, python-docx.
- Reports are the source of truth for the downstream pipeline — do not rename their fields casually.

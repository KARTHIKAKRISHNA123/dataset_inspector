"""Command-line entry point: `dataset-inspect` (or `python -m dataset_inspector`).

Wires the stages into one streaming, resumable run:

    walk(input) ─▶ for each FileMeta ─▶ inspect ─▶ write per-file report ─▶ mark checkpoint
                                                                       └▶ append master rows
    ─▶ finalize master inventory

Every file is independent, so a crash is recoverable: re-running skips files whose `key`
(path + quick content hash) is already in the checkpoint log. `--no-resume` forces a clean pass;
`--limit` caps the number of files (handy for a smoke test on a huge tree).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from .config import Config
from .core.checkpoint import CheckpointManager
from .core.exceptions import InspectorError
from .discovery.walker import DatasetWalker
from .inspector import DatasetInspector
from .logging_utils import Heartbeat, get_logger, setup_logging
from .reports import ReportWriter


def _default_config_path() -> str:
    """`<repo_root>/config/default.yaml`, where repo_root is the package's parent dir."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(pkg_dir)
    return os.path.join(repo_root, "config", "default.yaml")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dataset-inspect",
        description="Standalone streaming inspector/profiler for heterogeneous datasets.")
    p.add_argument("--input", "-i", required=True,
                   help="Root directory of raw datasets (e.g. your Downloads folder).")
    p.add_argument("--config", "-c", default=None,
                   help="Path to config YAML (default: <repo>/config/default.yaml).")
    p.add_argument("--override-config", default=None, help="Second YAML to deep-merge.")
    p.add_argument("--set", action="append", default=[], metavar="key=value",
                   help="Inline config override (repeatable), e.g. --set stats.max_records_sampled=20000")
    p.add_argument("--outdir", "-o", default=None, help="Override io.outdir.")
    p.add_argument("--limit", type=int, default=None, help="Inspect at most N files (smoke test).")
    p.add_argument("--no-resume", action="store_true", help="Ignore the checkpoint; inspect everything.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config or _default_config_path()

    try:
        config = Config.load(config_path, args.override_config, args.set)
    except InspectorError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    outdir = args.outdir or config.get("io.outdir", "./outputs")
    config.raw.setdefault("io", {})["outdir"] = outdir

    log = setup_logging(
        level=str(config.get("logging.level", "INFO")),
        file=str(config.get("logging.file", os.path.join(outdir, "logs", "inspector.log"))),
        json_format=bool(config.get("logging.json", False)),
    )
    clog = get_logger("cli")

    if not os.path.isdir(args.input):
        clog.error("input is not a directory: %s", args.input)
        return 2

    walker = DatasetWalker(config)
    inspector = DatasetInspector(config)
    writer = ReportWriter(config, outdir)
    ckpt_path = os.path.join(outdir, "_checkpoints", "inspect.done")
    checkpoint = None if args.no_resume else CheckpointManager(ckpt_path)

    hb = Heartbeat(clog, every=int(config.get("logging.heartbeat_files", 200)),
                   unit="files", log_memory=bool(config.get("logging.log_memory", True)))
    clog.info("inspecting %s → %s | readers: %s",
              args.input, outdir, ", ".join(inspector.registry.registered()))

    n_done = 0
    try:
        for meta in walker.walk(args.input):
            if checkpoint is not None and checkpoint.is_done(meta.key):
                continue
            try:
                report = inspector.inspect(meta)
                writer.write_report(report)
            except InspectorError as exc:           # recoverable: log + continue
                clog.warning("inspect failed for %s: %s", meta.key, exc)
            except Exception as exc:                 # noqa: BLE001 - never let one file kill the run
                clog.warning("unexpected error for %s: %s", meta.key, exc)
            if checkpoint is not None:
                checkpoint.mark_done(meta.key)
            hb.tick()
            n_done += 1
            if args.limit is not None and n_done >= args.limit:
                clog.info("reached --limit=%d, stopping", args.limit)
                break
    except KeyboardInterrupt:
        clog.warning("interrupted — re-run to resume from checkpoint.")
    finally:
        summary = writer.finalize()
        if checkpoint is not None:
            checkpoint.close()
    hb.done()
    clog.info("inspection complete: %d files | %s", summary["n_files"],
              ", ".join(f"{k}={v}" for k, v in summary["by_file_type"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

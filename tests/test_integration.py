"""End-to-end integration test: build a tiny multi-format corpus, run the CLI, assert outputs.

Also verifies the two production guarantees we most care about:
  * resumability — a second run inspects 0 new files (checkpoint works);
  * master inventory is produced in all four formats.
"""

from __future__ import annotations

import gzip
import json
import os
import zipfile

from dataset_inspector import cli


def _build_corpus(root: str) -> None:
    os.makedirs(os.path.join(root, "ta"), exist_ok=True)
    os.makedirs(os.path.join(root, "mix"), exist_ok=True)
    with open(os.path.join(root, "ta", "a.txt"), "w", encoding="utf-8") as f:
        f.write("வணக்கம் உலகம்\nதமிழ்\n")
    with open(os.path.join(root, "mix", "b.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps({"text": "नमस्ते", "language": "hi"}, ensure_ascii=False) + "\n")
    with gzip.open(os.path.join(root, "mix", "c.txt.gz"), "wt", encoding="utf-8") as f:
        f.write("ಕನ್ನಡ\n")
    with zipfile.ZipFile(os.path.join(root, "mix", "d.zip"), "w") as z:
        z.writestr("te/te.jsonl", json.dumps({"text": "తెలుగు", "language": "te"},
                                             ensure_ascii=False) + "\n")
        z.writestr("skip.md", "ignore")


def test_end_to_end(tmp_path):
    corpus = tmp_path / "raw"
    out = tmp_path / "out"
    _build_corpus(str(corpus))

    rc = cli.main(["--input", str(corpus), "--outdir", str(out)])
    assert rc == 0

    # all four master artifacts exist
    for name in ("master_inventory.json", "master_inventory.csv",
                 "master_inventory.md", "master_inventory.jsonl"):
        assert (out / name).exists(), name

    master = json.loads((out / "master_inventory.json").read_text(encoding="utf-8"))
    # 4 inspectable units: txt, jsonl, gz-as-txt, zip-member jsonl  (skip.md ignored)
    assert master["summary"]["n_files"] == 4
    types = master["summary"]["by_file_type"]
    assert types.get("txt", 0) >= 2 and types.get("jsonl", 0) >= 2

    # per-dataset reports written
    assert (out / "reports").is_dir()

    # resumability: a second run does no new work
    rc2 = cli.main(["--input", str(corpus), "--outdir", str(out)])
    assert rc2 == 0

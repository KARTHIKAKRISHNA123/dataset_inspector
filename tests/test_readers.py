"""Reader unit tests — each format streams correctly from a tiny on-disk file."""

from __future__ import annotations

import json
import struct

from dataset_inspector.core.models import Compression, FileType
from dataset_inspector.readers.txt import TxtReader
from dataset_inspector.readers.jsonl import JsonlReader
from dataset_inspector.readers.json_reader import JsonReader
from dataset_inspector.readers.csv_tsv import CsvTsvReader
from dataset_inspector.readers.xml_reader import XmlReader
from dataset_inspector.readers.gguf_reader import GgufReader
from tests.conftest import make_config, make_meta


def test_txt_reader_skips_blank_lines(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("one\n\n  \ntwo\n", encoding="utf-8")
    recs = list(TxtReader(make_config()).iter_records(make_meta(str(p), FileType.TXT)))
    assert [r["text"] for r in recs] == ["one", "two"]


def test_jsonl_reader_streams_objects(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text('{"text":"x"}\n\n{"text":"y"}\n', encoding="utf-8")
    recs = list(JsonlReader(make_config()).iter_records(make_meta(str(p), FileType.JSONL)))
    assert [r["text"] for r in recs] == ["x", "y"]


def test_json_reader_streams_array_with_limit(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps([{"text": i} for i in range(100)]), encoding="utf-8")
    recs = list(JsonReader(make_config()).iter_records(make_meta(str(p), FileType.JSON),
                                                       max_records=5))
    assert len(recs) == 5
    assert recs[0]["text"] == 0


def test_csv_reader_detects_header_and_rows(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("id,text\n1,hello\n2,world\n", encoding="utf-8")
    reader = CsvTsvReader(make_config())
    recs = list(reader.iter_records(make_meta(str(p), FileType.CSV)))
    assert recs == [{"id": "1", "text": "hello"}, {"id": "2", "text": "world"}]
    assert reader.describe(make_meta(str(p), FileType.CSV))["has_header"] is True


def test_tsv_reader(tmp_path):
    p = tmp_path / "a.tsv"
    p.write_text("src\ttgt\nGood\tநல்லது\n", encoding="utf-8")
    recs = list(CsvTsvReader(make_config()).iter_records(make_meta(str(p), FileType.TSV)))
    assert recs[0]["tgt"] == "நல்லது"


def test_xml_reader_streams_text_and_describes(tmp_path):
    p = tmp_path / "a.xml"
    p.write_text('<corpus><s>வணக்கம்</s><s>உலகம்</s></corpus>', encoding="utf-8")
    meta = make_meta(str(p), FileType.XML)
    recs = list(XmlReader(make_config()).iter_records(meta))
    assert [r["text"] for r in recs] == ["வணக்கம்", "உலகம்"]
    desc = XmlReader(make_config()).describe(meta)
    assert desc["root"] == "corpus"
    assert "s" in desc["text_elements"]


def test_xml_reader_malformed_is_safe(tmp_path):
    p = tmp_path / "bad.xml"
    p.write_text('<corpus><s>ok</s><s>broken', encoding="utf-8")   # never closed
    recs = list(XmlReader(make_config()).iter_records(make_meta(str(p), FileType.XML)))
    assert recs and recs[0]["text"] == "ok"          # partial, no exception


def _gstr(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def test_gguf_reader_metadata_only(tmp_path):
    p = tmp_path / "m.gguf"
    with open(p, "wb") as f:
        f.write(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 2))
        f.write(_gstr("general.architecture") + struct.pack("<I", 8) + _gstr("llama"))
        f.write(_gstr("tokenizer.ggml.tokens") + struct.pack("<I", 9)
                + struct.pack("<I", 8) + struct.pack("<Q", 3)
                + _gstr("a") + _gstr("b") + _gstr("c"))
    meta = make_meta(str(p), FileType.GGUF)
    reader = GgufReader(make_config())
    assert list(reader.iter_records(meta)) == []          # never yields text
    desc = reader.describe(meta)
    assert desc["architecture"] == "llama"
    assert desc["vocab_size"] == 3                          # from token array length
    assert desc["tensor_count"] == 0

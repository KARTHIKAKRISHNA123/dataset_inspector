"""Unit tests for the pure, format-agnostic pieces: detectors, scripts, schema, classifier."""

from __future__ import annotations

import gzip

from dataset_inspector.core.models import Compression, FileType, SchemaInfo
from dataset_inspector.core import scripts as S
from dataset_inspector.discovery import detectors as D
from dataset_inspector.schema.detect import SchemaSampler, classify_dataset
from tests.conftest import make_config


# ---- detectors -------------------------------------------------------------------------------
def test_detect_compression_magic_beats_extension(tmp_path):
    p = tmp_path / "data.txt"          # named .txt but actually gzip
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        fh.write("hello")
    head = open(p, "rb").read(8)
    assert D.detect_compression(str(p), head) is Compression.GZIP


def test_detect_file_type_inner_ext_for_gzip():
    assert D.detect_file_type("corpus.jsonl.gz", Compression.GZIP, b"") is FileType.JSONL


def test_sniff_jsonl_vs_json():
    assert D.sniff_content_type(b'{"a":1}\n{"a":2}\n') is FileType.JSONL
    assert D.sniff_content_type(b'[{"a":1},{"a":2}]') is FileType.JSON
    assert D.sniff_content_type(b"<root></root>") is FileType.XML
    assert D.sniff_content_type(b"GGUF\x03\x00") is FileType.GGUF


def test_detect_encoding_utf8():
    enc, conf = D.detect_encoding("வணக்கம்".encode("utf-8"))
    assert enc.replace("_", "-").lower().startswith("utf")
    assert 0.0 <= conf <= 1.0


def test_detect_source_hint():
    assert D.detect_source("/x/huggingface/datasets--ai4bharat/file.txt") == "Hugging Face"
    assert D.detect_source("/x/random/file.txt") == "unknown"


# ---- scripts ---------------------------------------------------------------------------------
def test_dominant_script_unique_languages():
    assert S.dominant_script("வணக்கம்") == "Tamil"
    assert S.unique_language_for_script("Tamil") == "ta"
    # Devanagari is shared → no unique language
    assert S.dominant_script("नमस्ते") == "Devanagari"
    assert S.unique_language_for_script("Devanagari") is None


def test_analyze_scripts_code_switching():
    info = S.analyze_scripts("hello வணக்கம் world உலகம்")
    assert info["code_switching"] is True
    assert "Tamil" in info["script_distribution"] and "Latin" in info["script_distribution"]


# ---- schema + classification -----------------------------------------------------------------
def test_schema_sampler_detects_roles():
    cfg = make_config(schema={"text_fields": ["text"], "language_fields": ["language"],
                              "source_fields": ["source"], "target_fields": ["target"]})
    s = SchemaSampler(cfg)
    s.observe({"text": "hi", "language": "en", "extra": 1})
    info = s.finalize()
    assert info.text_column == "text"
    assert info.language_column == "language"
    assert "extra" in info.metadata_columns
    assert info.n_columns == 3


def test_classify_translation_pair_is_bilingual():
    cfg = make_config()
    schema = SchemaInfo(columns=["source", "target"], n_columns=2,
                        source_column="source", target_column="target")
    dtype, conf, reason, langs = classify_dataset(schema, {"languages_seen": []}, {}, cfg)
    assert dtype.value == "Bilingual"
    assert conf > 0.5


def test_classify_multilingual_from_language_field():
    cfg = make_config()
    schema = SchemaInfo(columns=["text", "language"], language_column="language")
    stats = {"languages_seen": ["ta", "kn", "te", "en"], "script_distribution": {}}
    dtype, conf, reason, langs = classify_dataset(schema, stats, {}, cfg)
    assert dtype.value == "Multilingual"
    assert set(langs) >= {"ta", "kn", "te", "en"}

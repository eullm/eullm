"""Tests for the italgiure corpus validation script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_italgiure_corpus.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("validate_italgiure_corpus", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


mod = _load_module()


def _sample_record(**over) -> dict:
    rec = {
        "text": "La Corte di Cassazione " + "testo legale " * 50,
        "source": "italgiure",
        "sentence_id": "snpen/2026/13137",
        "article_num": "13137",
        "article_title": "Cassazione snpen Sez. 5, Sentenza n.13137/2026",
        "url": "https://example/snpen.pdf",
        "metadata": {
            "ecli": "ECLI:IT:CASS:2026:13137PEN",
            "kind": "snpen",
            "sezione": "5",
            "ssz": "0",
            "anno": "2026",
            "tipoprov": "Sentenza",
            "datdec": "20251210",
            "datdep": "20260409",
            "presidente": "CATENA",
            "relatore": "BELMONTE",
        },
    }
    rec.update(over)
    return rec


def _write_slice(dir: Path, kind: str, year: str, count: int, **rec_over) -> None:
    path = dir / f"italgiure_{kind}_{year}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for n in range(count):
            rec = _sample_record(**rec_over)
            rec["sentence_id"] = f"{kind}/{year}/{13000 + n}"
            rec["article_num"] = str(13000 + n)
            rec["metadata"] = dict(rec["metadata"], kind=kind, anno=year)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def test_validate_corpus_counts_and_splits(tmp_path: Path):
    _write_slice(tmp_path, "snpen", "2025", 3)
    _write_slice(tmp_path, "snciv", "2026", 2)

    stats = mod.validate_corpus(tmp_path)
    assert stats["total_records"] == 5
    assert stats["by_kind"] == {"snpen": 3, "snciv": 2}
    assert stats["by_year"] == {"2025": 3, "2026": 2}
    assert stats["by_kind_year"] == {"snpen_2025": 3, "snciv_2026": 2}
    assert stats["malformed_records"] == 0
    assert stats["duplicate_sentence_ids"] == 0
    assert stats["length_stats"]["min"] > 0


def test_validate_corpus_flags_missing_fields(tmp_path: Path):
    path = tmp_path / "italgiure_snpen_2026.jsonl"
    bad = _sample_record()
    bad.pop("text")
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(bad) + "\n")
        f.write(json.dumps(_sample_record()) + "\n")

    stats = mod.validate_corpus(tmp_path)
    assert stats["malformed_records"] == 1
    assert stats["total_records"] == 1
    assert any("text" in k for k in stats["issues"])


def test_validate_corpus_detects_duplicates(tmp_path: Path):
    path = tmp_path / "italgiure_snpen_2026.jsonl"
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(_sample_record()) + "\n")
        f.write(json.dumps(_sample_record()) + "\n")

    stats = mod.validate_corpus(tmp_path)
    assert stats["duplicate_sentence_ids"] == 1


def test_validate_corpus_empty_dir_raises(tmp_path: Path):
    with pytest.raises(SystemExit):
        mod.validate_corpus(tmp_path)


def test_validate_corpus_samples(tmp_path: Path):
    _write_slice(tmp_path, "snpen", "2026", 3)
    stats = mod.validate_corpus(tmp_path, sample_per_slice=2)
    assert len(stats["samples"]) == 1
    assert len(stats["samples"][0]["records"]) == 2

"""Tests for the structured-PII sweep over an already-anonymised corpus."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sweep_structured_pii.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sweep_structured_pii", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


mod = _load_module()


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_default_layers_exclude_name_layers():
    cfg = mod.build_config(list(mod.DEFAULT_LAYERS))
    # The whole point of the sweep: names are the first pass's job, and a
    # second per-chunk numbering would destroy [PERSONA_N] coherence.
    assert cfg.use_ner is False
    assert cfg.redact_allcaps_names is False
    assert cfg.redact_cf is True
    # Opt-in layers stay off unless asked for.
    assert cfg.redact_phone is False
    assert cfg.redact_address is False
    assert cfg.redact_birth is False


def test_opt_in_layer_is_enabled_when_requested():
    cfg = mod.build_config(["phone"])
    assert cfg.redact_phone is True
    assert cfg.redact_cf is False


def test_report_only_leaves_the_file_untouched(tmp_path):
    src = tmp_path / "train.jsonl"
    _write_jsonl(src, [{"text": "il ricorrente [PERSONA_1] (RRNMSM70S21G273H) deduce"}])
    before = src.read_bytes()

    counts, changed = mod.sweep_file(
        src,
        config=mod.build_config(list(mod.DEFAULT_LAYERS)),
        field="text",
        apply=False,
        show=False,
    )

    assert counts["codice_fiscale"] == 1
    assert changed == 1
    assert src.read_bytes() == before
    assert not (tmp_path / "train.jsonl.bak").exists()
    assert not (tmp_path / "train.jsonl.tmp").exists()


def test_apply_redacts_and_keeps_a_backup(tmp_path):
    src = tmp_path / "train.jsonl"
    _write_jsonl(
        src,
        [
            {"text": "il ricorrente [PERSONA_1] (RRNMSM70S21G273H) deduce", "id": 1},
            {"text": "la Corte accoglie il ricorso", "id": 2},
        ],
    )

    counts, changed = mod.sweep_file(
        src,
        config=mod.build_config(list(mod.DEFAULT_LAYERS)),
        field="text",
        apply=True,
        show=False,
    )

    assert counts["codice_fiscale"] == 1
    assert changed == 1

    out = _read_jsonl(src)
    assert "RRNMSM70S21G273H" not in out[0]["text"]
    assert "[CODICE_FISCALE]" in out[0]["text"]
    # [PERSONA_1] from the first pass must survive untouched.
    assert "[PERSONA_1]" in out[0]["text"]
    # Untouched records round-trip, other fields included.
    assert out[1]["text"] == "la Corte accoglie il ricorso"
    assert [r["id"] for r in out] == [1, 2]

    backup = _read_jsonl(tmp_path / "train.jsonl.bak")
    assert "RRNMSM70S21G273H" in backup[0]["text"]
    assert not (tmp_path / "train.jsonl.tmp").exists()


def test_sweep_is_idempotent(tmp_path):
    src = tmp_path / "train.jsonl"
    _write_jsonl(src, [{"text": "codice SPDFRC66M42F839P agli atti"}])
    cfg = mod.build_config(list(mod.DEFAULT_LAYERS))

    mod.sweep_file(src, config=cfg, field="text", apply=True, show=False)
    first = src.read_text(encoding="utf-8")

    counts, changed = mod.sweep_file(src, config=cfg, field="text", apply=True, show=False)

    assert sum(counts.values()) == 0
    assert changed == 0
    assert src.read_text(encoding="utf-8") == first


def test_records_without_the_text_field_survive(tmp_path):
    src = tmp_path / "train.jsonl"
    _write_jsonl(src, [{"content": "VLFABL75P66G273N"}, {"text": 42}])

    counts, changed = mod.sweep_file(
        src,
        config=mod.build_config(list(mod.DEFAULT_LAYERS)),
        field="text",
        apply=True,
        show=False,
    )

    assert sum(counts.values()) == 0
    assert changed == 0
    # A record the sweep does not understand is copied verbatim, never dropped.
    assert _read_jsonl(src) == [{"content": "VLFABL75P66G273N"}, {"text": 42}]


def test_malformed_json_is_preserved(tmp_path):
    src = tmp_path / "train.jsonl"
    src.write_text(
        '{"text": "MSTMTN76T03M208N"}\nnot json at all\n',
        encoding="utf-8",
    )

    mod.sweep_file(
        src,
        config=mod.build_config(list(mod.DEFAULT_LAYERS)),
        field="text",
        apply=True,
        show=False,
    )

    lines = src.read_text(encoding="utf-8").splitlines()
    assert "MSTMTN76T03M208N" not in lines[0]
    assert lines[1] == "not json at all"


def test_cli_reports_and_exits_nonzero_when_dirty(tmp_path, capsys):
    src = tmp_path / "train.jsonl"
    _write_jsonl(src, [{"text": "LMMMCS62P16Z404B"}])

    rc = mod.main([str(src)])

    assert rc == 1
    assert src.read_text(encoding="utf-8").count("LMMMCS62P16Z404B") == 1
    assert "codice_fiscale" in capsys.readouterr().out


def test_cli_exits_zero_on_a_clean_corpus(tmp_path):
    src = tmp_path / "val.jsonl"
    _write_jsonl(src, [{"text": "la Corte rigetta il ricorso"}])

    assert mod.main([str(src)]) == 0


def test_cli_rejects_an_unknown_layer(tmp_path):
    src = tmp_path / "train.jsonl"
    _write_jsonl(src, [{"text": "x"}])

    with pytest.raises(SystemExit):
        mod.main([str(src), "--layers", "cf,nomi"])

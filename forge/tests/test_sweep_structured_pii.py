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

    counts, changed, _ = mod.sweep_file(
        src,
        config=mod.build_config(list(mod.DEFAULT_LAYERS)),
        prefilter=mod.build_prefilter(list(mod.DEFAULT_LAYERS)),
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

    counts, changed, _ = mod.sweep_file(
        src,
        config=mod.build_config(list(mod.DEFAULT_LAYERS)),
        prefilter=mod.build_prefilter(list(mod.DEFAULT_LAYERS)),
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
    pre = mod.build_prefilter(list(mod.DEFAULT_LAYERS))

    mod.sweep_file(src, config=cfg, prefilter=pre, field="text", apply=True, show=False)
    first = src.read_text(encoding="utf-8")

    counts, changed, _ = mod.sweep_file(
        src, config=cfg, prefilter=pre, field="text", apply=True, show=False
    )

    assert sum(counts.values()) == 0
    assert changed == 0
    assert src.read_text(encoding="utf-8") == first


def test_records_without_the_text_field_survive(tmp_path):
    src = tmp_path / "train.jsonl"
    _write_jsonl(src, [{"content": "VLFABL75P66G273N"}, {"text": 42}])

    counts, changed, _ = mod.sweep_file(
        src,
        config=mod.build_config(list(mod.DEFAULT_LAYERS)),
        prefilter=mod.build_prefilter(list(mod.DEFAULT_LAYERS)),
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
        prefilter=mod.build_prefilter(list(mod.DEFAULT_LAYERS)),
        field="text",
        apply=True,
        show=False,
    )

    lines = src.read_text(encoding="utf-8").splitlines()
    assert "MSTMTN76T03M208N" not in lines[0]
    assert lines[1] == "not json at all"


# One sample per layer that the layer is known to redact. The prefilter must
# catch every one of them, JSON-encoded, or the sweep would skip a line it
# should have cleaned — the single way the fast path can be wrong.
_LAYER_SAMPLES = {
    "cf": "il ricorrente RRNMSM70S21G273H deduce",
    "piva": "la società con P.IVA 12345678901 ha presentato",
    "iban": "accredito su IBAN IT60X0542811101000000123456 intestato",
    "email": "notifica a mario.rossi@example.com in data",
    "phone": "reperibile al +39 333 1234567 per",
    "birth": "nato a Palermo il 21/11/1970 e residente",
    "address": "con studio in Via Roma 12, presso",
}


@pytest.mark.parametrize("layer", sorted(_LAYER_SAMPLES))
def test_prefilter_catches_everything_its_layer_redacts(layer):
    sample = _LAYER_SAMPLES[layer]
    _, stats = mod.anonymize_text(sample, config=mod.build_config([layer]))
    assert stats.total() > 0, f"sample for {layer!r} is not actually redacted"

    line = json.dumps({"text": sample}, ensure_ascii=False)
    prefilter = mod.build_prefilter([layer])
    assert any(p.search(line) for p in prefilter)


def test_every_selectable_layer_has_prefilter_patterns():
    assert set(mod.LAYER_PATTERNS) == set(mod.LAYERS)
    assert all(mod.LAYER_PATTERNS[layer] for layer in mod.LAYERS)


def test_clean_lines_are_copied_byte_for_byte(tmp_path):
    src = tmp_path / "train.jsonl"
    # Odd spacing, key order and a non-ASCII char: a JSON round-trip would
    # normalise all three, the fast path must not touch any of them.
    src.write_text(
        '{ "id":1,  "text" : "la Corte accoglie il ricorso perché fondato" }\n',
        encoding="utf-8",
    )
    before = src.read_bytes()

    mod.sweep_file(
        src,
        config=mod.build_config(list(mod.DEFAULT_LAYERS)),
        prefilter=mod.build_prefilter(list(mod.DEFAULT_LAYERS)),
        field="text",
        apply=True,
        show=False,
    )

    assert src.read_bytes() == before


def test_prefilter_hit_outside_the_text_field_changes_nothing(tmp_path):
    src = tmp_path / "train.jsonl"
    src.write_text(
        '{"text": "la Corte rigetta", "contact": "ufficio@example.com"}\n',
        encoding="utf-8",
    )
    before = src.read_bytes()

    counts, changed, _ = mod.sweep_file(
        src,
        config=mod.build_config(list(mod.DEFAULT_LAYERS)),
        prefilter=mod.build_prefilter(list(mod.DEFAULT_LAYERS)),
        field="text",
        apply=True,
        show=False,
    )

    # The sweep only ever rewrites `field`; a match elsewhere costs a parse
    # and nothing else, and the line is passed through untouched.
    assert sum(counts.values()) == 0
    assert changed == 0
    assert src.read_bytes() == before


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


# --- a line this sweep cannot read is not a line it has cleared --------------

def _broken_line(src: Path, body: str) -> None:
    """A record that matched the prefilter but will not parse: a truncated
    write, or a stray control character inside the JSON."""
    src.write_text(body + "\n", encoding="utf-8")


def test_a_malformed_line_is_redacted_rather_than_passed_through(tmp_path):
    src = tmp_path / "train.jsonl"
    _broken_line(src, '{"text": "contatto mario.rossi@example.it", broken')

    counts, _changed, uninspectable = mod.sweep_file(
        src,
        config=mod.build_config(list(mod.DEFAULT_LAYERS)),
        prefilter=mod.build_prefilter(list(mod.DEFAULT_LAYERS)),
        field="text",
        apply=True,
        show=False,
    )

    assert uninspectable == 1
    assert sum(counts.values()) == 1
    assert "mario.rossi@example.it" not in src.read_text(encoding="utf-8")


def test_a_malformed_line_without_a_prefilter_match_is_left_alone(tmp_path):
    src = tmp_path / "train.jsonl"
    src.write_text('{"text": "la Corte rigetta" oops\n', encoding="utf-8")

    _counts, _changed, uninspectable = mod.sweep_file(
        src,
        config=mod.build_config(list(mod.DEFAULT_LAYERS)),
        prefilter=mod.build_prefilter(list(mod.DEFAULT_LAYERS)),
        field="text",
        apply=True,
        show=False,
    )

    # Nothing here could have matched a layer, so there is nothing to report.
    assert uninspectable == 0


def test_the_cli_does_not_call_a_file_clean_it_could_not_read(tmp_path, capsys):
    src = tmp_path / "train.jsonl"
    _broken_line(src, '{"text": "scrivi a mario.rossi@example.it", broken')

    rc = mod.main([str(src)])
    out = capsys.readouterr().out

    assert rc == 1
    assert "not inspectable" in out or "not be parsed" in out
    assert "No structured PII found." not in out


def test_the_cli_still_exits_zero_when_every_line_parses(tmp_path):
    src = tmp_path / "train.jsonl"
    _write_jsonl(src, [{"text": "la Corte rigetta il ricorso"}])

    assert mod.main([str(src)]) == 0


def test_cli_rejects_an_unknown_layer(tmp_path):
    src = tmp_path / "train.jsonl"
    _write_jsonl(src, [{"text": "x"}])

    with pytest.raises(SystemExit):
        mod.main([str(src), "--layers", "cf,nomi"])


def test_star_field_scans_every_field_and_only_reports(tmp_path):
    import json as _json
    import subprocess
    import sys as _sys
    from pathlib import Path as _Path

    script = _Path(__file__).resolve().parents[1] / "scripts" / "sweep_structured_pii.py"
    f = tmp_path / "cards.jsonl"
    rows = [{"massima": "Il ricorrente, mario.rossi@example.com, propone ricorso."},
            {"massima": "Nessun dato."}]
    f.write_text("".join(_json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    before = f.read_text()
    p = subprocess.run([_sys.executable, str(script), str(f), "--field", "*", "--quiet"],
                       capture_output=True, text=True)
    assert p.returncode == 1 and "1 record" in p.stdout
    assert f.read_text() == before                          # nothing rewritten
    p = subprocess.run([_sys.executable, str(script), str(f), "--field", "text", "--quiet"],
                       capture_output=True, text=True)
    assert p.returncode == 0                                # 'text' alone misses it
    p = subprocess.run([_sys.executable, str(script), str(f), "--field", "*", "--apply"],
                       capture_output=True, text=True)
    assert p.returncode != 0

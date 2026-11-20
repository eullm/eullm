"""clean_corpus.py: nothing of a held-out set's articles, no question-like rows on dev ones."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "clean_corpus.py"
_spec = importlib.util.spec_from_file_location("clean_corpus", _PATH)
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")
    return path


def test_the_rule(tmp_path):
    held = _write(tmp_path / "practical-heldout.jsonl",
                  [{"id": "h1", "metadata": {"code": "cp", "articolo": "575"}}])
    dev = _write(tmp_path / "closed-book.jsonl", [
        {"id": "d1", "code": "cp", "articolo": "624"},
        {"id": "d2", "code": "cp", "articolo": "999-bis", "tipo": "esiste_falsa"},
    ])
    rows = _write(tmp_path / "rewrites-caso-a.jsonl", [
        {"code": "cp", "articolo": "575", "forma": "parafrasi", "teacher": "g"},   # held-out
        {"code": "cp", "articolo": "624", "forma": "caso", "teacher": "g"},        # dev, question
        {"code": "cp", "articolo": "624", "forma": "parafrasi", "teacher": "g"},   # dev, text: kept
        {"code": "cp", "articolo": "999-BIS", "forma": "inesistente", "source": "program"},
        {"code": "cc", "articolo": "2043", "forma": "caso", "teacher": "g"},       # kept
    ])
    _write(tmp_path / "rewrites-caso-a.rejects.jsonl", [{"id": "x", "reason": "length"}])
    out = tmp_path / "clean"
    m = cc.clean([rows, tmp_path / "rewrites-caso-a.rejects.jsonl"], out, [held], [dev],
                 set(cc.DEV_FORMS.split(",")))
    kept = [json.loads(x) for x in (out / "rewrites-caso-a.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert [(r["articolo"], r["forma"]) for r in kept] == [("624", "parafrasi"),
                                                           ("2043", "caso")]
    assert not (out / "rewrites-caso-a.rejects.jsonl").exists()
    assert m["rows"] == 2
    assert m["dropped"] == {"held-out article": 1, "development article, question-like form": 1,
                            "made-up number of a set": 1}
    f = m["files"]["rewrites-caso-a.jsonl"]
    assert f["by_teacher_and_form"] == {"g": {"caso": 1, "parafrasi": 1}}
    assert len(f["sha256"]) == 64
    assert m["sets"]["held_out"][str(held)]["articles"] == 1
    assert m["sets"]["development"][str(dev)]["made_up_numbers"] == 1
    saved = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert "text" not in json.dumps(saved["files"])        # counts only


def test_refuses_to_overwrite_its_input(tmp_path):
    rows = _write(tmp_path / "rewrites-a.jsonl", [{"code": "cp", "articolo": "1"}])
    try:
        cc.clean([rows], tmp_path, [], [], set())
    except SystemExit as e:
        assert "is the input" in str(e)
    else:
        raise AssertionError("overwrote its input")

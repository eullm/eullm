"""judge_agreement.py and paired.cohen_kappa: two judges over the same answers."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

from eullm_forge.eval.paired import cohen_kappa

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "judge_agreement.py"


def _load():
    spec = importlib.util.spec_from_file_location("judge_agreement", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_kappa_is_one_on_agreement_and_zero_at_chance():
    assert cohen_kappa([("correct", "correct"), ("wrong", "wrong")]) == 1.0
    # each rater says "correct" half the time, independently of the other
    pairs = [("correct", "correct"), ("correct", "wrong"), ("wrong", "correct"), ("wrong", "wrong")]
    assert abs(cohen_kappa(pairs)) < 1e-9
    assert cohen_kappa([("correct", "wrong"), ("wrong", "correct")]) < 0
    assert cohen_kappa([("correct", "correct")]) == 1.0
    assert math.isnan(cohen_kappa([]))


def test_agreement_counts_only_the_items_both_graded():
    mod = _load()
    a = {"q1": "correct", "q2": "partial", "q3": "wrong", "q4": "correct"}
    b = {"q1": "correct", "q2": "wrong", "q3": "wrong"}
    r = mod.agreement(a, b)
    assert r["n"] == 3 and abs(r["exact"] - 2 / 3) < 1e-9
    assert r["exact_bin"] == 1.0          # partial and wrong are both "not correct"
    assert r["correct_a"] == 1 and r["correct_b"] == 1


def _graded(d: Path, label: str, grades: dict[str, str]):
    d.mkdir(parents=True, exist_ok=True)
    (d / f"answers-{label}.graded.jsonl").write_text("".join(
        json.dumps({"id": i, "grade": g, "answer": "x"}) + "\n" for i, g in grades.items()), encoding="utf-8")


def test_the_script_reports_per_model_and_per_kind(tmp_path, capsys):
    mod = _load()
    a, b = tmp_path / "qwen", tmp_path / "mistral"
    ids = {"norm-contenuto-codice_civile-1": "correct", "norm-contenuto-codice_civile-2": "wrong",
           "norm-termine-codice_civile-3": "correct"}
    _graded(a, "m1", ids)
    _graded(b, "m1", {**ids, "norm-termine-codice_civile-3": "wrong"})
    _graded(a, "only-a", ids)
    assert mod.main([str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "1 models in both" in out and "m1" in out and "only-a" not in out
    assert "  contenuto" in out and "  termine" in out
    assert "correct 2 vs 1" in out


def test_no_common_model_is_said_so(tmp_path, capsys):
    mod = _load()
    _graded(tmp_path / "a", "m1", {"x": "correct"})
    _graded(tmp_path / "b", "m2", {"x": "correct"})
    assert mod.main([str(tmp_path / "a"), str(tmp_path / "b")]) == 1
    assert "no model graded in both" in capsys.readouterr().out

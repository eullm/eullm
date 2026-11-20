"""trap_summary.py: the traps of 36, and of 33 without the off-topic three."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "trap_summary.py"


def load():
    spec = importlib.util.spec_from_file_location("trap_summary", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def trap(i, expected, category="Premessa"):
    return {"id": i, "category": category, "question": "q", "metadata": {"expected": expected}}


TRAPS = [trap("t-art-1", "correzione"), trap("t-art-2", "correzione"),
         trap("t-plain-1", "risposta"), trap("t-fuori-imu", "astensione", "Fuori perimetro"),
         trap("t-fuori-casa", "astensione", "Fuori perimetro"),
         # same category, but a plain abstention test: it stays in the 33
         trap("t-statistiche", "astensione", "Fuori perimetro")]


def test_the_off_topic_traps_are_counted_apart():
    s = load().summary({t["id"]: t for t in TRAPS},
                       {"t-art-1": "correct", "t-art-2": "wrong", "t-plain-1": "correct",
                        "t-fuori-imu": "correct", "t-fuori-casa": "partial",
                        "t-statistiche": "correct"})
    assert (s["right"], s["n"], s["right_kept"], s["n_kept"]) == (4, 6, 3, 4)
    assert s["by_expected"] == {"correzione": 1, "astensione": 2, "risposta": 1}
    assert s["off_topic"] == {"casa": False, "imu": True}


def test_the_table_prints_counts_only(tmp_path, capsys):
    traps = tmp_path / "traps.jsonl"
    traps.write_text("".join(json.dumps(t) + "\n" for t in TRAPS), encoding="utf-8")
    graded = tmp_path / "graded"
    graded.mkdir()
    (graded / "answers-r2l-q4-hybnn.graded.jsonl").write_text("".join(
        json.dumps({"id": t["id"], "grade": "correct", "question": "SECRET"}) + "\n"
        for t in TRAPS), encoding="utf-8")
    assert load().main(["--traps", str(traps), "--graded", str(graded), "r2l", "none"]) == 0
    out = capsys.readouterr().out
    assert "6/6   4/4" in out and "none" in out and "no graded answers" in out
    assert "SECRET" not in out and "q" not in out.split()

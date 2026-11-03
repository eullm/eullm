"""tools/lumi/round_status.py: one round's points by group, label and state."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("round_status",
                                              ROOT / "tools" / "lumi" / "round_status.py")
round_status = importlib.util.module_from_spec(spec)
spec.loader.exec_module(round_status)


def put(queue, state, pid, **point):
    d = queue / state
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{pid}.json").write_text(json.dumps(dict(id=pid, **point)), encoding="utf-8")


def test_a_rounds_points_by_group_label_and_state(tmp_path, capsys):
    r = "r-b12032"
    put(tmp_path, "done", "m1", round=r, group="b12-moe", engine_label="v085")
    put(tmp_path, "running", "m2", round=r, group="b12-moe", engine_label="v085")
    (tmp_path / "running" / "m2.owner.json").write_text("{}", encoding="utf-8")
    put(tmp_path, "todo", "m3", round=r, group="b12-moe", engine_label="v087")
    put(tmp_path, "failed", "d1", round=r, group="b12-dsa", engine_label="v085",
        model="deepseek", notes=["PointError: server on port 18000 exited (134):\nlog"])
    put(tmp_path, "todo", "x", round="r-v085", group="b12-moe", engine_label="v085")
    (tmp_path / "todo" / "broken.json").write_text("{", encoding="utf-8")

    assert round_status.main(["--queue", str(tmp_path), "--round", r]) == 0
    out = capsys.readouterr().out.splitlines()
    rows = {tuple(line.split()[:2]): line.split()[2:] for line in out[1:]
            if not line.startswith("  ")}
    assert rows[("b12-moe", "v085")] == ["0", "1", "1", "0", "0"]
    assert rows[("b12-moe", "v087")] == ["1", "0", "0", "0", "0"]
    assert rows[("b12-dsa", "v085")] == ["0", "0", "0", "1", "0"]
    assert out[-1] == ("  failed b12-dsa v085 deepseek: "
                       "PointError: server on port 18000 exited (134):")


def test_an_empty_round_says_so(tmp_path, capsys):
    assert round_status.main(["--queue", str(tmp_path), "--round", "r-none"]) == 0
    assert "no point of this round" in capsys.readouterr().out

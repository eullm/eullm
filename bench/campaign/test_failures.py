"""tools/lumi/failures.py: a label's failed points, and the server's log of
the first of each group and hour."""

from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("failures", ROOT / "tools" / "lumi" / "failures.py")
failures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(failures)


def fail(queue, pid, t, **point):
    d = queue / "failed"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{pid}.json"
    path.write_text(json.dumps(dict(id=pid, **point)), encoding="utf-8")
    os.utime(path, (t, t))


def test_a_labels_failures_with_the_log_of_each_group_and_hour(tmp_path, capsys):
    t = time.mktime((2026, 11, 2, 3, 33, 0, 0, 0, -1))
    log = "PointError: server on port 18020 exited (1):\n" + "\n".join(
        f"server line {i}" for i in range(30)) + "\nError: the last line"
    fail(tmp_path, "a", t, engine_label="v085", group="scale-split", model="qwen3-32b",
         kv="f16/f16", gcds=2, notes=["an earlier try", log])
    fail(tmp_path, "b", t + 30, engine_label="v085", group="scale-split", model="qwen3-14b",
         kv="f16/f16", gcds=2, notes=[log])
    fail(tmp_path, "c", t + 3600 * 10, engine_label="v085", group="ram-480b",
         model="qwen3-coder-480b", kv="f16/f16", gcds=8,
         notes=["PointError: every request failed in repeat 0: ['timed out']"])
    fail(tmp_path, "d", t, engine_label="v0794", group="scale-split", model="qwen3-32b",
         notes=["not this label"])
    (tmp_path / "failed" / "e.json").write_text("{half", encoding="utf-8")

    assert failures.main(["--queue", str(tmp_path), "--label", "v085", "--log", "3"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "3 failed"
    assert out[1].startswith("02-11 03:33 scale-split") and "qwen3-32b" in out[1]
    assert out[1].endswith("PointError: server on port 18020 exited (1):")
    assert "not this label" not in "\n".join(out)
    # One note per group and hour: the 03:33 scale-split once, then ram-480b.
    heads = [line for line in out if line.startswith("---")]
    assert len(heads) == 2 and "qwen3-32b" in heads[0] and "ram-480b" in heads[1]
    i = out.index(heads[0])
    assert out[i + 1:i + 4] == ["server line 28", "server line 29", "Error: the last line"]


def test_no_failures_says_so(tmp_path, capsys):
    assert failures.main(["--queue", str(tmp_path), "--label", "v085"]) == 0
    assert capsys.readouterr().out.strip() == "0 failed"

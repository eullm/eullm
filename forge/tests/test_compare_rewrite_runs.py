"""compare_rewrite_runs.py: identical answers, first differences, thinking leaks."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compare_rewrite_runs.py"
_spec = importlib.util.spec_from_file_location("compare_rewrite_runs", _PATH)
crr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(crr)


def _run(d: Path, rows: dict[str, str]) -> Path:
    d.mkdir()
    (d / "rewrites-caso.jsonl").write_text(
        "".join(json.dumps({"id": k, "text": v}) + "\n" for k, v in rows.items()),
        encoding="utf-8")
    (d / "rewrites-caso.rejects.jsonl").write_text('{"id": "x"}\n', encoding="utf-8")
    return d


def test_against_the_base(tmp_path):
    a = _run(tmp_path / "llama", {"1": "uguale", "2": "abcdef", "3": "solo qui"})
    b = _run(tmp_path / "engine", {"1": "uguale", "2": "abcXef", "4": "<|channel>thought x"})
    runs = {d.name: crr.read_run(d) for d in (a, b)}
    out = crr.compare(runs, "llama")
    assert "thinking in 0" in out[0]
    assert "2 in common, 1 identical, first difference at median char 3" in out[1]
    assert "thinking in 1" in out[1]


def test_speed_from_the_log(tmp_path):
    log = tmp_path / "job.out"
    log.write_text("[rewrites] 98 kept in 2.0 min: 49/min\n"
                   "[rewrites] 198 kept in 4.2 min: 47/min\n", encoding="utf-8")
    assert crr.speed(log).startswith("[rewrites] 198 kept")

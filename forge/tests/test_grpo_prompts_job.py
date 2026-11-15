"""sbatch_grpo_prompts.slurm with make_grpo_prompts.py stubbed.

Pinned: the held-out and development exams are passed as exclusions, the
file appears only when the run finished (a GRPO chain waits on it), and a
file already there is kept rather than drawn again.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "forge" / "scripts" / "leonardo" / "sbatch_grpo_prompts.slurm"

pytestmark = pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None,
                                reason="needs POSIX bash")

STUB = r'''#!/usr/bin/env bash
echo "$@" > "$CALLS"
out=""
while [ $# -gt 0 ]; do [ "$1" = "--out" ] && out="$2"; shift; done
[ -n "${FAIL:-}" ] && { echo partial > "$out"; exit 1; }
printf '{"id": "p1"}\n{"id": "p2"}\n' > "$out"
'''


@pytest.fixture
def job(tmp_path):
    work = tmp_path / "work"
    (work / "eval").mkdir(parents=True)
    (work / "norms").mkdir()
    for name in ("eval/norm-exam-v3.jsonl", "eval/norm-exam-v4.jsonl",
                 "eval/norm-exam-devbig.jsonl", "norms/legislazione_a.chunks.jsonl"):
        (work / name).write_text("{}\n")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    py = bin_ / "python"
    py.write_text(STUB)
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    calls = tmp_path / "calls"
    out = work / "grpo" / "prompts-hyb06.jsonl"
    out.parent.mkdir()
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "WORK": str(work),
           "CALLS": str(calls), "GP_REPO": str(REPO), "GP_OUT": str(out),
           "EULLM_VENV": str(tmp_path / "nov")}
    env.pop("SLURM_JOB_ID", None)

    def run(**extra):
        r = subprocess.run(["bash", str(SCRIPT)], cwd=tmp_path, env={**env, **extra},
                           capture_output=True, text=True, encoding="utf-8", timeout=60)
        args = calls.read_text().split() if calls.exists() else []
        calls.unlink(missing_ok=True)
        return r, args

    return run, out, work


def test_prompts_are_drawn_hybrid_away_from_every_exam(job):
    run, out, work = job
    r, args = run()
    assert r.returncode == 0, r.stdout + r.stderr
    excluded = args[args.index("--exclude-exam") + 1:args.index("--per-code")]
    assert args[args.index("--judged") + 1] == "0"
    assert sorted(Path(p).name for p in excluded) == [
        "norm-exam-devbig.jsonl", "norm-exam-v3.jsonl", "norm-exam-v4.jsonl"]
    assert args[args.index("--embedder") + 1] == "Qwen/Qwen3-Embedding-0.6B"
    assert args[args.index("--reranker") + 1] == "Qwen/Qwen3-Reranker-0.6B"
    assert args[args.index("--seed") + 1] == "1"
    assert out.read_text().count("\n") == 2


def test_a_failed_run_leaves_no_file_and_a_finished_one_is_kept(job):
    run, out, _ = job
    r, _ = run(FAIL="1")
    assert r.returncode != 0 and not out.exists()
    r, _ = run()
    assert r.returncode == 0
    r, args = run()
    assert r.returncode == 0 and args == [] and "kept" in r.stdout


def test_a_missing_exam_stops_before_drawing(job):
    run, out, work = job
    (work / "eval" / "norm-exam-v4.jsonl").unlink()
    r, args = run()
    assert r.returncode == 1 and args == [] and "missing" in r.stderr
    assert not out.exists()

"""sbatch_ngram_train.slurm as a link of a chain, with ngram_train.py stubbed.

Pinned: a finished run (DONE) exits at once without starting Python, a
missing model or corpus fails before the GPU is used, and the options reach
the trainer, the mode included.
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
SCRIPT = REPO / "forge" / "scripts" / "leonardo" / "sbatch_ngram_train.slurm"

pytestmark = pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None,
                                reason="needs POSIX bash (on Windows, bash is WSL's)")

STUB = r'''#!/usr/bin/env bash
echo "$*" >> "$CALLS"
exit 0
'''


@pytest.fixture
def job(tmp_path):
    work = tmp_path / "work"
    mem = work / "eullm_runs" / "memory"
    mem.mkdir(parents=True)
    (mem / "program.jsonl").write_text('{"text": "a"}\n')
    (mem / "rewrites.jsonl").write_text('{"text": "b"}\n')
    (mem / "rewrites-b.jsonl").write_text('{"text": "c"}\n')
    model = work / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    py = bin_ / "python"
    py.write_text(STUB)
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    calls = tmp_path / "calls"
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "WORK": str(work),
           "CALLS": str(calls), "NG_REPO": str(REPO), "NG_MODEL": str(model),
           "EULLM_VENV": str(tmp_path / "nov"), "HEARTBEAT_INTERVAL": "1"}
    env.pop("SLURM_JOB_ID", None)

    def run(**extra):
        r = subprocess.run(["bash", str(SCRIPT)], cwd=tmp_path, env={**env, **extra},
                           capture_output=True, text=True, timeout=60)
        asked = calls.read_text().splitlines() if calls.exists() else []
        calls.unlink(missing_ok=True)
        return r, asked

    return run, work


def test_the_options_reach_the_trainer(job):
    run, work = job
    r, asked = run(NG_MODE="lora")
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(asked) == 1
    call = asked[0]
    assert "ngram_train.py" in call and "--mode lora" in call and "--stop-after 6600" in call
    assert "--epochs 3" in call
    assert str(work / "eullm_runs" / "lineA" / "a2-r2l-lora") in call
    for name in ("program.jsonl", "rewrites.jsonl", "rewrites-b.jsonl"):
        assert name in call


def test_a_finished_run_exits_without_python(job):
    run, work = job
    out = work / "eullm_runs" / "lineA" / "a2-r2l-ngram"
    out.mkdir(parents=True)
    (out / "DONE").write_text('{"steps": 9}')
    r, asked = run()
    assert r.returncode == 0 and asked == []
    assert "finished already" in r.stdout


def test_missing_inputs_fail_before_the_trainer(job, tmp_path):
    run, work = job
    r, asked = run(NG_MODEL=str(tmp_path / "nope"))
    assert r.returncode == 1 and asked == [] and "no model" in r.stderr
    r, asked = run(NG_DATA=str(tmp_path / "missing.jsonl"))
    assert r.returncode == 1 and asked == [] and "missing" in r.stderr
    r, asked = run(NG_MODE="full")
    assert r.returncode == 1 and asked == []

"""submit_gate.sh: the gate trained and a round examined through it, in one submission."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "leonardo" / "submit_gate.sh"

pytestmark = pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None,
                                reason="needs POSIX bash")


def _exe(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _setup(tmp_path: Path, queued: str = "") -> tuple[dict, Path]:
    work = tmp_path / "work"
    rnd = work / "eullm_runs" / "grpo" / "r2h"
    (rnd / "merged").mkdir(parents=True)
    (rnd / "merged" / "config.json").write_text("{}", encoding="utf-8")
    (rnd / "gguf").mkdir()
    (rnd / "gguf" / "r2h-q4_k_m.gguf").write_text("gguf", encoding="utf-8")
    (work / "eullm_runs" / "stage3").mkdir(parents=True)
    (work / "eullm_runs" / "stage3" / "openbook-v05.jsonl").write_text("{}\n", encoding="utf-8")
    (work / "eullm_runs" / "premise5").mkdir(parents=True)
    (work / "eullm_runs" / "premise5" / "grpo.jsonl").write_text("{}\n", encoding="utf-8")
    (work / "eval" / "lexia").mkdir(parents=True)
    (work / "eval" / "lexia" / "lexia-trappole.jsonl").write_text("{}\n", encoding="utf-8")
    (work / "eval" / "practical-dev.jsonl").write_text("{}\n", encoding="utf-8")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    log = tmp_path / "sbatch.log"
    _exe(bin_ / "sbatch", f"""#!/usr/bin/env bash
n=$(( $(wc -l < "{log}" 2>/dev/null || echo 0) + 100 ))
printf '%s|%s|STEP=%s|OB=%s|ARGS=%s|TAG=%s\\n' "$n" "$*" "${{GATE_STEP:-}}" \\
  "${{GATE_OPENBOOK:-}}" "${{EXAM_OPEN_ARGS:-}}" "${{EXAM_OPEN_TAG:-}}" >> "{log}"
echo "$n"
""")
    _exe(bin_ / "squeue", f"""#!/usr/bin/env bash
printf '%s' '{queued}'
""")
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "WORK": str(work)}
    return env, log


def _run(env, *args):
    return subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True,
                          text=True, encoding="utf-8", timeout=30)


def test_the_gate_is_trained_in_order_then_the_round_examined_through_it(tmp_path):
    env, log = _setup(tmp_path)
    r = _run(env, "r2h")
    assert r.returncode == 0, r.stderr + r.stdout
    calls = [ln.split("|") for ln in log.read_text(encoding="utf-8").splitlines()]
    steps = [c[2] for c in calls[:3]]
    assert steps == ["STEP=offtopic", "STEP=pairs", "STEP=train"]
    # the newest open-book pairs on disk
    assert calls[1][3].endswith("openbook-v05.jsonl")
    assert "afterok:100" in calls[1][1] and "afterok:101" in calls[2][1]
    exams = [c for c in calls[3:] if "sbatch_norm_exam" in c[1]]
    # traps, devbig in two links, practical; a note to the model by default
    assert [c[5] for c in exams] == ["TAG=hybnn-gate", "TAG=hybnn-gate", "TAG=hybnn-gate",
                                     "TAG=hyb-gate"]
    assert all("--gate " in c[4] and "--gate-mode note" in c[4] for c in exams)
    assert "afterok:102" in exams[0][1]
    assert sum("sbatch_judge" in c[1] for c in calls) == 3


def test_a_second_gate_gets_its_own_labels(tmp_path):
    env, log = _setup(tmp_path)
    r = _run({**env, "GATE_TAG": "gate2", "GATE_DIR": str(tmp_path / "gate2")}, "r2h")
    assert r.returncode == 0, r.stderr + r.stdout
    tags = {ln.split("|")[5] for ln in log.read_text(encoding="utf-8").splitlines()
            if "norm_exam" in ln}
    assert tags == {"TAG=hybnn-gate2", "TAG=hyb-gate2"}


def test_nothing_is_queued_twice_or_without_its_inputs(tmp_path):
    env, log = _setup(tmp_path, queued="eullm-gate-train\n")
    r = _run(env, "r2h")
    assert r.returncode == 0 and "already queued" in r.stdout
    assert not log.exists()
    env, log = _setup(tmp_path / "b")
    r = _run(env, "missing-round")
    assert r.returncode == 1 and "missing" in r.stderr
    assert not log.exists()

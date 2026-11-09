"""submit_memory_round.sh (JUPITER): training, GGUF and the three measurements, in order, once."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "forge" / "scripts" / "jupiter" / "submit_memory_round.sh"

pytestmark = pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None,
                                reason="needs POSIX bash")


def _exe(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(0o755)


def _setup(tmp_path):
    project = tmp_path / "e-aif-2026pg01-1434"
    home = project / "tester" / "forge"
    data, runs, private = home / "datasets", home / "runs", home / "private-eval"
    for d in (data / "memory", data / "openbook", data / "eval" / "leonardo", private,
              runs / "start"):
        d.mkdir(parents=True)
    (runs / "start" / "config.json").write_text("{}")
    for f in (data / "memory" / "general-it.jsonl", data / "openbook" / "pairs.jsonl",
              data / "eval" / "leonardo" / "closed-book.jsonl",
              data / "eval" / "leonardo" / "norm-exam-devbig.jsonl",
              private / "practical-heldout.jsonl"):
        f.write_text("{}\n")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    log, queue = tmp_path / "sbatch.log", tmp_path / "queue"
    log.write_text("")
    queue.write_text("")
    _exe(bin_ / "sbatch", f"""#!/bin/bash
L={log.as_posix()}
n=$(( $(wc -l < "$L") + 100 ))
echo "$n $* MODEL=${{MC_MODEL:-}} OUT=${{MC_OUT:-}} PREP=${{MC_PREPARE_ARGS:-}}" \
     "ARMS=${{VERT_ARMS:-}} PE=${{PE_MODELS:-}} CLOSED=${{PE_CLOSED:-}} RG=${{RG_MODELS:-}}" >> "$L"
echo "$n"
""")
    _exe(bin_ / "squeue", f"#!/bin/sh\ncat {queue.as_posix()}\n")
    env = {**os.environ, "PATH": bin_.as_posix() + ":" + os.environ.get("PATH", ""),
           "PROJECT": project.as_posix(), "USER": "tester", "SLURM_JOB_ID": "",
           "MC_PREPARE_ARGS": "--epochs 5"}
    for k in ("FORGE_HOME", "FORGE_VENV", "FORGE_DATA_DIR", "FORGE_RUN_DIR", "FORGE_EXTRA_LIBS",
              "FORGE_PRIVATE_EVAL", "LMOD_CMD", "MR_AFTER", "MR_LINKS", "MR_SFT", "PE_CLOSED"):
        env.pop(k, None)
    return env, runs, log, queue


def test_a_round_is_queued_in_order_and_once(tmp_path):
    env, runs, log, queue = _setup(tmp_path)
    run = ["bash", str(SCRIPT), "e1-x", str(runs / "start")]
    p = subprocess.run(run, env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stdout + p.stderr
    jobs = log.read_text().splitlines()
    link1, link2, vert, closed, open_, reg = jobs
    assert "--job-name=forge-mem-e1-x" in link1 and "--dependency" not in link1
    assert "PREP=--epochs 5" in link1 and f"OUT={runs / 'memory' / 'e1-x'}" in link1
    assert "--dependency=afterany:100" in link2
    assert "--dependency=afterany:101" in vert and "ARMS=e1-x" in vert
    for j in (closed, open_, reg):
        assert "--dependency=afterany:102" in j
    assert "CLOSED=1" in closed and "e1-x-q4_k_m.gguf" in closed
    assert "CLOSED= " in open_ and "practical-heldout=" not in closed
    assert "RG=e1-x=" in reg
    assert p.stdout.strip().splitlines()[-1] == "OPEN 104"

    queue.write_text("101\n")
    p = subprocess.run(run, env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0 and "already queued" in p.stdout
    assert len(log.read_text().splitlines()) == 6


def test_a_model_made_elsewhere_is_measured_without_training(tmp_path):
    env, runs, log, _ = _setup(tmp_path)
    env.update({"MR_LINKS": "0", "MR_SFT": str(runs / "grpo"), "MR_AFTER": "555"})
    p = subprocess.run(["bash", str(SCRIPT), "g9e-r2", str(runs / "grpo" / "g9e-r2" / "merged")],
                       env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stdout + p.stderr
    vert, closed, open_, reg = log.read_text().splitlines()
    assert "--dependency=afterany:555" in vert and "ARMS=g9e-r2" in vert
    assert str(runs / "grpo" / "g9e-r2" / "merged") in closed
    assert "--dependency=afterany:100" in open_ and "RG=g9e-r2=" in reg

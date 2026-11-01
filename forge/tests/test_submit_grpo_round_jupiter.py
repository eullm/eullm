"""submit_grpo_round.sh (JUPITER): one submission queues the whole round, in order.

The real script runs against a fake Slurm: sbatch records each job's options
and the GRPO/EVAL variables it was given, squeue answers from a file. Checked:
the prompts are filtered before anything is queued, the training waits for
the smoke test to end well, the exams for the last link, the judge for the
exams; and a second run while the round is queued submits nothing.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "forge" / "scripts" / "jupiter" / "submit_grpo_round.sh"

pytestmark = pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None,
                                reason="needs POSIX bash")


def _exe(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(0o755)


def _item(i, code, art):
    return {"id": f"q{i}", "domain": "legal", "lang": "it", "question": "?",
            "reference": "r", "metadata": {"code": code, "articolo": art}}


def _setup(tmp_path):
    project = tmp_path / "e-aif-2026pg01-1434"
    home = project / "tester" / "forge"
    data, runs, private = home / "datasets", home / "runs", home / "private-eval"
    for d in (data / "grpo" / "leonardo", data / "eval", private, home / "libs-grpo",
              home / "libs-qwen35", runs / "start-model"):
        d.mkdir(parents=True)
    (runs / "start-model" / "config.json").write_text("{}")
    rows = [{"id": f"p{i}", "prompt": [{"role": "user", "content": "Domanda?"}],
             "tipo": "pratica" if i % 2 else "premessa_termine", "keywords": [],
             "code": "codice_civile", "articolo": str(i)} for i in range(1, 9)]
    (data / "grpo" / "leonardo" / "prompts-practical.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    # The trap test is on article 3: its prompt must go.
    trap = json.dumps(_item(1, "codice_civile", "3"))
    (data / "eval" / "trap-test.jsonl").write_text(trap + "\n")
    for f in (data / "eval" / "trap-test-para.jsonl", data / "eval" / "norm-exam.jsonl",
              private / "lexia-trappole.jsonl", private / "practical-dev.jsonl"):
        f.write_text(json.dumps(_item(9, "codice_penale", "1")) + "\n")
    # START already answered the trap test: it is asked only the other sets.
    (runs / "teacher-eval").mkdir()
    (runs / "teacher-eval" / "answers-trap-test-dqp-d-q9b.jsonl").write_text("{}\n")

    bin_ = tmp_path / "bin"
    bin_.mkdir()
    log, queue = tmp_path / "sbatch.log", tmp_path / "queue"
    queue.write_text("")
    log.write_text("")
    _exe(bin_ / "sbatch", f"""#!/bin/bash
L={log.as_posix()}
n=$(( $(wc -l < "$L") + 100 ))
echo "$n $* NO_MERGE=${{GRPO_NO_MERGE:-}} ARGS=${{GRPO_ARGS:-}} OUT=${{GRPO_OUT:-}}" \
     "MODEL=${{EVAL_MODEL:-}} SETS=${{EVAL_SETS:-}} JUDGE=${{JUDGE_ANSWERS_DIR:-}}" >> "$L"
echo "$n"
""")
    _exe(bin_ / "squeue", f"#!/bin/sh\ncat {queue.as_posix()}\n")
    env = {**os.environ, "PATH": bin_.as_posix() + ":" + os.environ.get("PATH", ""),
           "PROJECT": project.as_posix(), "USER": "tester", "SLURM_JOB_ID": ""}
    for k in ("FORGE_HOME", "FORGE_VENV", "FORGE_DATA_DIR", "FORGE_RUN_DIR",
              "FORGE_EXTRA_LIBS", "FORGE_PRIVATE_EVAL", "GRPO_PROMPTS", "LMOD_CMD"):
        env.pop(k, None)
    return env, runs, log, queue


def _run(env, runs):
    return subprocess.run(["bash", str(SCRIPT), "r1", str(runs / "start-model"), "2", "200"],
                          env=env, capture_output=True, text=True, timeout=60)


def test_the_round_is_queued_in_order_and_once(tmp_path):
    env, runs, log, queue = _setup(tmp_path)
    p = _run(env, runs)
    assert p.returncode == 0, p.stdout + p.stderr
    prompts = runs / "grpo" / "r1" / "prompts.jsonl"
    kept = [json.loads(ln) for ln in prompts.read_text().splitlines()]
    assert sorted(r["articolo"] for r in kept) == ["1", "2", "4", "5", "6", "7", "8"]

    jobs = log.read_text().splitlines()
    smoke, link1, link2, ev, start_ev, judge = jobs
    assert "--time=00:40:00" in smoke and "NO_MERGE=1" in smoke and "--limit 48" in smoke
    assert smoke.endswith("JUDGE=") and "OUT=" + str(runs / "grpo" / "r1-smoke") in smoke
    assert "--dependency=afterok:100" in link1 and "--max-steps 200" in link1
    assert "NO_MERGE= " in link1 and "--time=6:00:00" in link1
    assert "--dependency=afterany:101" in link2
    assert "--dependency=afterany:102" in ev
    assert "MODEL=" + str(runs / "grpo" / "r1" / "merged") in ev
    assert all(f"{s}=" in ev for s in ("trap-test", "trap-test-para", "lexia-trappole",
                                       "practical-dev", "norm-exam"))
    assert "MODEL=" + str(runs / "start-model") in start_ev and "--dependency" not in start_ev
    assert "SETS=trap-test=" not in start_ev and "trap-test-para=" in start_ev
    assert "--dependency=afterany:103:104" in judge
    assert "JUDGE=" + str(runs / "teacher-eval") in judge
    assert "Domanda" not in p.stdout + p.stderr

    # Queued: a second run submits nothing.
    queue.write_text("102\n")
    p = _run(env, runs)
    assert p.returncode == 0 and "already queued" in p.stdout
    assert len(log.read_text().splitlines()) == 6


def test_start_is_asked_only_the_sets_named(tmp_path):
    env, runs, log, _ = _setup(tmp_path)
    env["GRPO_START_SETS"] = "norm-exam"
    p = _run(env, runs)
    assert p.returncode == 0, p.stdout + p.stderr
    start_ev = log.read_text().splitlines()[4]
    assert "MODEL=" + str(runs / "start-model") in start_ev
    assert "SETS=norm-exam=" in start_ev and "practical-dev=" not in start_ev


def test_a_round_waits_for_the_one_that_writes_its_start(tmp_path):
    env, runs, log, _ = _setup(tmp_path)
    env["GRPO_AFTER"] = "777"
    start = runs / "grpo" / "r2" / "merged"
    p = subprocess.run(["bash", str(SCRIPT), "r3", str(start), "2", "200"],
                       env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stdout + p.stderr
    jobs = log.read_text().splitlines()
    link1, link2, ev, judge = jobs           # no smoke test, START not asked
    assert "--dependency=afterany:777" in link1 and "--limit" not in link1
    assert "--dependency=afterany:100" in link2
    assert "MODEL=" + str(runs / "grpo" / "r3" / "merged") in ev
    assert "--dependency=afterany:102" in judge


def test_prompts_a_queued_job_writes_are_filtered_by_the_first_link(tmp_path):
    env, runs, log, _ = _setup(tmp_path)
    env["GRPO_AFTER"] = "555"
    later = tmp_path / "later.jsonl"            # written by the job the round waits for
    env["GRPO_PROMPTS"] = str(later)
    p = subprocess.run(["bash", str(SCRIPT), "r4", str(runs / "start-model"), "1", "50"],
                       env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stdout + p.stderr
    out = runs / "grpo" / "r4"
    assert not (out / "prompts.jsonl").exists()
    filt = (out / "filter.sh").read_text()
    assert "filter_grpo_prompts.py" in filt and str(later) in filt
    assert "--dependency=afterany:555" in log.read_text().splitlines()[0]


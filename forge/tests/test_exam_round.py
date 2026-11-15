"""The exam round, run as the watcher runs it, with sbatch stubbed.

Drawing the exam is real (make_norm_exam.py on made-up articles); only the
scheduler is fake. What is pinned: the exam leaves out the training pairs'
articles, an existing exam is reused rather than redrawn, the judge waits on
the exam (afterok), and a round without its exclusions refuses to start.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "forge" / "scripts" / "leonardo" / "sbatch_exam_round.slurm"

pytestmark = pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None,
                                reason="needs POSIX bash (on Windows, bash is WSL's)")

FILLER = " Il presente articolo contiene disposizioni di dettaglio sufficienti." * 3
RECORDS = [{"code": "codice_civile", "article_num": "", "chunk_index": 0,
            "text": f"Art. {n}. \n \n (Rubrica {n}). \n \n La domanda si propone entro "
                    f"sessanta giorni.{FILLER}"}
           for n in range(1, 61)]


def _exe(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def leonardo(tmp_path):
    work = tmp_path / "work"
    (work / "norms").mkdir(parents=True)
    (work / "norms" / "legislazione_x.chunks.jsonl").write_text(
        "\n".join(json.dumps(r) for r in RECORDS) + "\n")
    model = work / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    pairs = work / "openbook-pairs.jsonl"
    pairs.write_text("\n".join(json.dumps({"key": f"ob-g-codice_civile-{n}"})
                               for n in range(1, 21)) + "\n")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    _exe(bin_ / "sbatch", '#!/usr/bin/env bash\necho "$*" >> "$CALLS"\n'
                          'echo $((100 + $(wc -l < "$CALLS")))\n')
    _exe(bin_ / "python", f'#!/usr/bin/env bash\nexec "{sys.executable}" "$@"\n')
    calls = tmp_path / "sbatch-calls"
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "WORK": str(work),
           "CALLS": str(calls), "ROUND_REPO": str(REPO), "EULLM_VENV": str(tmp_path / "nov"),
           "EULLM_RUN_DIR": str(tmp_path / "run"),
           "ROUND_MODELS": f"v0.3={model} base={model}", "ROUND_PAIRS": str(pairs)}
    env.pop("SLURM_JOB_ID", None)

    def run(**extra):
        r = subprocess.run(["bash", str(SCRIPT)], cwd=tmp_path, env={**env, **extra},
                           capture_output=True, text=True, encoding="utf-8", timeout=120)
        return r, calls.read_text() if calls.exists() else ""

    return run, work


def test_a_round_draws_without_trained_articles_then_exam_then_judge(leonardo):
    run, work = leonardo
    r, calls = run()
    assert r.returncode == 0, r.stdout + r.stderr
    assert "20 trained articles left out" in r.stdout
    drawn = [json.loads(x)["metadata"] for x in
             (work / "eval" / "norm-exam-v2.jsonl").read_text().splitlines()]
    assert drawn
    assert not {int(m["articolo"]) for m in drawn if m["tipo"] != "inesistente"} & set(range(1, 21))
    exam, judge = calls.splitlines()
    assert "sbatch_norm_exam.slurm" in exam
    assert "--dependency=afterok:101" in judge and "sbatch_judge.slurm" in judge
    assert "Che cosa prevede" not in r.stdout   # counts only


def test_an_existing_exam_is_reused_not_redrawn(leonardo):
    run, work = leonardo
    run()
    exam = work / "eval" / "norm-exam-v2.jsonl"
    before = exam.read_text()
    r, _ = run()
    assert r.returncode == 0, r.stdout + r.stderr
    assert "reusing it, not redrawing" in r.stdout
    assert exam.read_text() == before


def test_a_round_without_its_exclusions_refuses(leonardo):
    run, _ = leonardo
    r, calls = run(ROUND_PAIRS="")
    assert r.returncode == 1
    assert "ROUND_PAIRS is required" in r.stderr
    assert not calls


def test_a_development_round_shares_no_article_with_the_held_out_exam(leonardo):
    run, work = leonardo
    run()                                         # the held-out exam
    held = work / "eval" / "norm-exam-v2.jsonl"
    dev = work / "eval" / "norm-exam-dev.jsonl"
    r, calls = run(ROUND_ITEMS=str(dev), ROUND_OUT=str(work / "eval" / "dev"),
                   ROUND_EXCLUDE_EXAM=str(held), ROUND_QUIET="0")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "articles of the other exam left out" in r.stdout

    def arts(p):
        return {(json.loads(x)["metadata"]["code"], json.loads(x)["metadata"]["articolo"])
                for x in p.read_text().splitlines()}
    assert any(json.loads(x)["metadata"]["tipo"] == "contenuto"
               for x in dev.read_text().splitlines())
    assert not arts(dev) & arts(held)


def test_a_round_asks_for_more_gpus_for_a_big_model(leonardo):
    run, _ = leonardo
    r, calls = run(ROUND_GPUS="2")
    assert r.returncode == 0, r.stdout + r.stderr
    exam = calls.splitlines()[0]
    assert "--gres=gpu:2" in exam and "--mem=240G" in exam and "--cpus-per-task=16" in exam


def test_a_development_round_draws_more_per_code_and_chains_its_links(leonardo):
    run, work = leonardo
    dev = work / "eval" / "norm-exam-devbig.jsonl"
    r, calls = run(ROUND_ITEMS=str(dev), ROUND_OUT=str(work / "eval" / "devbig"),
                   ROUND_PER_CODE="30", ROUND_CODES="codice_civile",
                   ROUND_EXAM_LINKS="3", ROUND_JUDGE_LINKS="2", ROUND_OPEN_ONLY="1")
    assert r.returncode == 0, r.stdout + r.stderr
    kinds = [json.loads(x)["metadata"]["tipo"] for x in dev.read_text().splitlines()]
    assert kinds.count("contenuto") == 30          # 40 untrained articles, 30 asked
    lines = calls.splitlines()
    assert len(lines) == 5
    e1, e2, e3, j1, j2 = lines
    assert "--dependency" not in e1
    assert "--dependency=afterany:101" in e2 and "--dependency=afterany:102" in e3
    assert all("sbatch_norm_exam.slurm" in x for x in (e1, e2, e3))
    # the judge waits for the last exam link to finish cleanly, then chains
    assert "--dependency=afterok:103" in j1 and "--dependency=afterany:104" in j2
    assert all("sbatch_judge.slurm" in x for x in (j1, j2))


def test_links_must_be_a_positive_number(leonardo):
    run, _ = leonardo
    r, calls = run(ROUND_EXAM_LINKS="0")
    assert r.returncode == 1 and "positive number" in r.stderr


def test_a_round_can_queue_its_own_report_after_the_last_judge_link(leonardo):
    run, work = leonardo
    graded = work / "eval" / "graded-v2"
    r, calls = run(ROUND_JUDGE_LINKS="2", ROUND_REPORT="v0.3-hyb base-hyb",
                   ROUND_REPORT_DIR=str(graded), ROUND_JUDGE_ARGS="--rubric v2")
    assert r.returncode == 0, r.stdout + r.stderr
    exam, j1, j2, report = calls.splitlines()
    assert "sbatch_judge.slurm" in j2 and "--dependency=afterany:102" in j2
    assert "--dependency=afterany:103" in report and "--partition=lrd_all_serial" in report
    assert f"--output={graded}/report.txt" in report
    assert "--baseline v0.3-hyb" in report and "--baseline base-hyb" in report
    assert graded.is_dir()
    assert f"-> {graded}/report.txt" in r.stdout


def test_without_a_report_the_round_says_where_the_grades_are(leonardo):
    run, _ = leonardo
    r, calls = run()
    assert len(calls.splitlines()) == 2 and "results will be in" in r.stdout

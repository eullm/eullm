"""`status.sh` on the situations it has actually met, with Slurm stubbed.

A morning check that cries wolf gets ignored, and one that misses the real
problem is worse than none. Its first run (2026-09-28) raised three alarms and
all three were noise: the last generation link exiting in seconds because
nothing was left, a watcher whose file had arrived minutes after its last
wake, and the package watcher waiting — correctly — for a training run. The
cases below pin both sides: the noise stays quiet, the real problems do not.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "leonardo" / "status.sh"

pytestmark = pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None,
                                reason="needs POSIX bash (on Windows, bash is WSL's)")


def _exe(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def run_status(tmp_path: Path, queue: list[tuple[str, str, str]],
               ended: list[tuple[str, str, str, str]],
               gpu_queue: list[str] | None = None, disk_pct: int = 50,
               extra_env: dict[str, str] | None = None) -> str:
    bin_ = tmp_path / "bin"
    bin_.mkdir(exist_ok=True)
    q_reason = "\\n".join("|".join(j) for j in queue)
    q_state = "\\n".join(f"{n} {s}" for n, s, _ in queue)
    q_name = "\\n".join(n for n, _, _ in queue)
    # what is on the GPU partition: by default one job, so that the tests of
    # other situations are not also an idle allocation
    q_gpu = "\\n".join(["1"] if gpu_queue is None else gpu_queue)
    _exe(bin_ / "squeue", f"""#!/usr/bin/env bash
case "$*" in
  *boost_usr_prod*) [ -n "{q_gpu}" ] && printf '{q_gpu}\\n' ;;
  *"%j|%T|%r"*) printf '{q_reason}\\n' ;;
  *"%j %T"*) printf '{q_state}\\n' ;;
  *) printf '{q_name}\\n' ;;
esac
""")
    rows = "\\n".join("|".join(e) for e in ended)
    short = "\\n".join(f"{n}|{s}" for _, n, s, _ in ended)
    _exe(bin_ / "sacct", f"""#!/usr/bin/env bash
case "$*" in
  *ElapsedRaw*) printf '6480|billing=32,cpu=32,gres/gpu=4,mem=480G,node=1\\n' ;;
  *JobID*) printf '{rows}\\n' ;;
  *) printf '{short}\\n' ;;
esac
""")
    _exe(bin_ / "df", f"""#!/usr/bin/env bash
printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n'
printf 'lustre 1000 {disk_pct * 10} 0 {disk_pct}%% /work\\n'
""")
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}",
           "EULLM_RUNS": str(tmp_path / "runs"), **(extra_env or {})}
    out = subprocess.run(["bash", str(SCRIPT), "2026-09-27T21:00"], env=env,
                         capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert out.returncode == 0, out.stderr
    return out.stdout


@pytest.fixture
def runs(tmp_path):
    (tmp_path / "runs" / "stage3" / "logs").mkdir(parents=True)
    return tmp_path / "runs" / "stage3"


def watcher_log(runs: Path, name: str, need: Path, left: int = 100) -> None:
    (runs / "logs" / f"{name}-9.out").write_text(
        f"[wait] 2026-09-28 12:34:09 not yet: {need}\n"
        f"[wait] 2026-09-28 12:34:09 will look again in 20 minutes (job 9, {left} tries left)\n",
        encoding="utf-8")


def test_this_mornings_three_false_alarms_stay_quiet(tmp_path, runs):
    (runs / "logs" / "eullm-gen-openbook-30.out").write_text(
        "[ob] nothing left to do -> x.done\n", encoding="utf-8")
    done = runs / "openbook-pairs.jsonl.done"
    done.write_text("done\n", encoding="utf-8")                      # arrived after the last wake
    watcher_log(runs, "wait-eullm-stage3", done)
    (runs / "logs" / "wait-eullm-stage3-pkg-8.out").write_text(
        f"[wait] 2026-09-28 12:34:09 not yet: {runs / 'sft' / 'adapter_config.json'}\n"
        "[wait] 2026-09-28 12:34:09 will look again in 20 minutes (job 8, 100 tries left)\n",
        encoding="utf-8")
    out = run_status(
        tmp_path,
        queue=[("wait-eullm-stage3", "PENDING", "BeginTime"),
               ("wait-eullm-stage3-pkg", "PENDING", "BeginTime"),
               ("eullm-stage3", "RUNNING", "None")],
        ended=[("30", "eullm-gen-openbook", "COMPLETED", "00:00:07"),
               ("1", "eullm-p2-amm", "TIMEOUT", "02:00:10")])
    assert "[!!]" not in out, out
    assert "nothing wrong found" in out
    assert "submits at its next wake" in out


def test_the_real_problems_are_flagged(tmp_path, runs):
    empty = runs / "openbook-pairs.jsonl.done"
    empty.touch()                                   # last night's bug
    watcher_log(runs, "wait-eullm-stage3", empty)
    out = run_status(
        tmp_path,
        queue=[("wait-eullm-stage3", "PENDING", "BeginTime"),
               ("eullm-p2-8b", "PENDING", "DependencyNeverSatisfied")],
        ended=[("2", "eullm-p2-amm", "COMPLETED", "00:00:03"),   # quota, gate, missing file
               ("3", "eullm-stage3", "FAILED", "00:12:00"),
               ("4", "eullm-p2-split", "OUT_OF_MEMORY", "00:40:00")])
    assert "exists but is EMPTY" in out
    assert "DependencyNeverSatisfied" in out
    assert "a link that short did no work" in out
    assert "eullm-stage3 ended FAILED" in out
    assert "ended OUT_OF_MEMORY" in out
    assert "5 thing(s) above need a look" in out


def test_a_stage3_link_after_training_finished_is_not_an_alarm(tmp_path, runs):
    """2026-09-30: the v0.4 chain's spare link resumed from the final
    checkpoint, saved the adapter again and ended in 2:43 — flagged as a
    link that did no work, when the work had been done by the link before."""
    (runs / "logs" / "eullm-stage3-59054377.out").write_text(
        "[s3] out    /w/sft-v04\n[stage3] adapter /w/sft-v04/adapter\n", encoding="utf-8")
    (runs / "logs" / "eullm-s3-it4b-v05-60.out").write_text(
        "[stage3] adapter already at /w/sft-it4b-v05/adapter: training finished, "
        "nothing left to do (move it away to train again)\n", encoding="utf-8")
    out = run_status(tmp_path, queue=[],
                     ended=[("59054377", "eullm-stage3", "COMPLETED", "00:02:43"),
                            ("60", "eullm-s3-it4b-v05", "COMPLETED", "00:00:40")])
    assert "[!!]" not in out, out
    assert "training finished (fine)" in out
    assert "60 eullm-s3-it4b-v05: ended in 00:00:40 with nothing left to do" in out


def test_a_grpo_link_after_training_finished_is_not_an_alarm(tmp_path, runs):
    """A GRPO link that resumed at the last step saved and ended, which is
    what grpo_train.py does with no steps left. Same case as the stage-3 link
    above, and it was raising the alarm: eullm-grpo* had been added to the list
    of names checked, but the escape only matched the stage-3 prefix."""
    (runs / "logs" / "eullm-grpo-77.out").write_text(
        "[grpo] resuming from /w/eullm_runs/grpo/v03/checkpoint-9000\n"
        "[grpo] adapter /w/eullm_runs/grpo/v03/adapter\n", encoding="utf-8")
    (runs / "logs" / "eullm-grpo-78.out").write_text(
        "[grpo] adapter already at /w/eullm_runs/grpo/v03/adapter: nothing "
        "left to do (move it away to train again)\n", encoding="utf-8")
    out = run_status(tmp_path, queue=[],
                     ended=[("77", "eullm-grpo", "COMPLETED", "00:04:31"),
                            ("78", "eullm-grpo", "COMPLETED", "00:00:12")])
    assert "[!!]" not in out, out
    assert "77 eullm-grpo: ended in 00:04:31, training finished (fine)" in out
    assert "78 eullm-grpo: ended in 00:00:12 with nothing left to do" in out


def test_a_named_stage3_link_that_did_nothing_is_flagged(tmp_path, runs):
    """A chain renamed with -J is still a training chain: a three-minute link
    that never reached the end of training needs a human."""
    (runs / "logs" / "eullm-s3-q35-9b-v04-61.out").write_text(
        "[s3] base   /w/Qwen3.5-9B\nTraceback (most recent call last):\n", encoding="utf-8")
    out = run_status(tmp_path, queue=[],
                     ended=[("61", "eullm-s3-q35-9b-v04", "COMPLETED", "00:03:00")])
    assert "61 eullm-s3-q35-9b-v04 ended COMPLETED after only 00:03:00" in out


def test_an_opd_link_is_watched_like_the_other_training_links(tmp_path):
    """OPD is the third training chain with the same two short endings, and
    the watchdog knew neither: eullm-opd* was not among the names checked,
    and the training-finished escape only matched the stage-3/grpo prefixes.
    A 21-second link that did nothing was silently counted; a link that saved
    the adapter or found nothing left would have been flagged next, once the
    name was added without the prefix.
    """
    opd = tmp_path / "runs" / "opd" / "logs"
    opd.mkdir(parents=True)
    (opd / "eullm-opd-11.out").write_text(
        "[opd] resuming at step 40\n"
        "[opd] adapter /w/eullm_runs/opd/v01/adapter\n", encoding="utf-8")
    (opd / "eullm-opd-12.out").write_text(
        "[opd] adapter already at /w/eullm_runs/opd/v01/adapter: nothing left to do\n",
        encoding="utf-8")
    (opd / "eullm-opd-13.out").write_text(
        "[opd] resuming at step 0\nTraceback (most recent call last):\n", encoding="utf-8")
    out = run_status(tmp_path, queue=[],
                     ended=[("11", "eullm-opd", "COMPLETED", "00:04:31"),
                            ("12", "eullm-opd", "COMPLETED", "00:00:12"),
                            ("13", "eullm-opd", "COMPLETED", "00:00:21")])
    assert "11 eullm-opd: ended in 00:04:31, training finished (fine)" in out
    assert "12 eullm-opd: ended in 00:00:12 with nothing left to do" in out
    assert "13 eullm-opd ended COMPLETED after only 00:00:21" in out


def test_jobs_that_are_short_on_purpose_are_not_stalled_links(tmp_path, runs):
    """Two jobs the eullm-p* glob took, and both are short on purpose.

    sbatch_quantize.slurm submits eullm-p3-gguf on lrd_all_serial with no
    --gres at all -- a CPU quantize, done in minutes. sbatch_backfill_probe's
    own header says eullm-probe "does nothing but report where it landed and
    exit", so it ends in seconds every time it runs. Neither resumes a chain,
    so neither has work it could have failed to do, and a morning check that
    cries wolf is one that gets ignored. The real chain link beside them keeps
    its alarm.
    """
    (runs / "logs" / "eullm-p3-gguf-62.out").write_text(
        "[quant] llama-quantize model-f16.gguf model-Q4_K_M.gguf\n", encoding="utf-8")
    (runs / "logs" / "eullm-probe-61.out").write_text("[probe] backfill window found\n",
                                                      encoding="utf-8")
    (runs / "logs" / "eullm-p2-8b-63.out").write_text("[distill] nothing at all happened\n",
                                                      encoding="utf-8")
    out = run_status(tmp_path, queue=[],
                     ended=[("61", "eullm-probe", "COMPLETED", "00:00:21"),
                            ("62", "eullm-p3-gguf", "COMPLETED", "00:04:30"),
                            ("63", "eullm-p2-8b", "COMPLETED", "00:00:21"),
                            ("64", "eullm-package-jpn17-q35-4b", "COMPLETED", "00:09:15")])
    assert "eullm-probe ended COMPLETED" not in out, out
    assert "eullm-p3-gguf ended COMPLETED" not in out, out
    assert "eullm-package-jpn17-q35-4b ended" not in out, out
    assert "63 eullm-p2-8b ended COMPLETED after only 00:00:21" in out
    assert "[!!] 1 thing(s) above need a look" in out, out


def test_grpo_progress_is_shown_and_a_stop_is_flagged(tmp_path, runs):
    logs = runs.parent / "grpo" / "logs"
    logs.mkdir(parents=True)
    (logs / "eullm-grpo-70.out").write_text(
        "[grpo] step 10/250 reward 0.712 (std 0.301) zero-std 0.45 kl 0.0010 len 88 31s/step\n"
        " 8%|##  | 20/250 [10:00<1:55:00, 30.0s/it]"     # tqdm's bar, no newline
        "[grpo] step 20/250 reward 0.744 (std 0.288) zero-std 0.48 kl 0.0021 len 85 30s/step\n",
        encoding="utf-8")
    (logs / "eullm-grpo-71.out").write_text(
        "[grpo] step 10/250 reward 0.990 (std 0.010) zero-std 0.98 kl 0.0001 len 60 30s/step\n"
        "[grpo] STOP: 20 steps in a row with 98% of groups all-equal (step 30): "
        "nothing left to learn from these prompts\n", encoding="utf-8")
    out = run_status(tmp_path, queue=[], ended=[])
    assert "     [grpo] step 20/250 reward 0.744" in out
    assert "[!!] eullm-grpo-71.out: STOP: 20 steps in a row" in out
    assert "1 thing(s) above need a look" in out


def test_an_idle_gpu_queue_is_flagged_and_a_queued_chain_is_not(tmp_path, runs):
    """2026-10-04: a day at 1.8 node-hours, noticed by the user, not by this."""
    out = run_status(tmp_path, queue=[("eullm-gguf-grpo-v04", "RUNNING", "None")],
                     ended=[], gpu_queue=[])
    assert "the allocation is idle" in out
    assert "57.6 local hours (1.8 node-hours) since" in out
    out = run_status(tmp_path, queue=[("eullm-grpo-r2-v04", "PENDING", "Dependency")],
                     ended=[], gpu_queue=["59332673"])
    assert "idle" not in out and "nothing wrong found" in out


def test_a_base_model_whose_scratch_copy_is_gone_is_flagged(tmp_path, runs):
    """2026-10-10: base models moved to the purged fast scratch behind
    symlinks. A purged copy leaves a link to nothing, and compute nodes
    cannot download it again."""
    work = tmp_path / "work"
    (work / "models").mkdir(parents=True)
    scratch = tmp_path / "scratch"
    (scratch / "Qwen3-14B").mkdir(parents=True)
    (work / "models" / "Qwen3-14B").symlink_to(scratch / "Qwen3-14B")
    (work / "models" / "Qwen3.8-27B").symlink_to(scratch / "Qwen3.8-27B")   # purged
    (work / "models" / "local").mkdir()
    out = run_status(tmp_path, [], [], extra_env={"WORK": str(work)})
    assert "[!!] " + str(work / "models" / "Qwen3.8-27B") + " points to" in out
    assert "Qwen3-14B points to" not in out and "local points to" not in out


def test_a_full_disk_is_flagged(tmp_path, runs):
    """2026-10-05: $WORK at 109% of its quota, found by a conversion failing."""
    out = run_status(tmp_path, queue=[("eullm-grpo", "RUNNING", "None")], ended=[],
                     disk_pct=94)
    assert "[!!] $WORK is 94% full" in out
    out = run_status(tmp_path, queue=[("eullm-grpo", "RUNNING", "None")], ended=[],
                     disk_pct=78)
    assert "78% full" in out and "nothing wrong found" in out


def test_a_full_scratch_is_flagged(tmp_path, runs):
    """2026-10-24: the scratch full too, every merge of a round failing there."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    out = run_status(tmp_path, queue=[("eullm-grpo", "RUNNING", "None")], ended=[],
                     disk_pct=94, extra_env={"EULLM_SCRATCH": str(scratch)})
    assert "[!!] the scratch is 94% full" in out
    out = run_status(tmp_path, queue=[("eullm-grpo", "RUNNING", "None")], ended=[],
                     disk_pct=94, extra_env={"EULLM_SCRATCH": str(tmp_path / "none")})
    assert "the scratch is" not in out


def test_the_allocation_left_and_the_pace_to_use_it_are_shown(tmp_path, runs, monkeypatch):
    """2026-10-06: ~13,000 local hours on course to be lost, worked out by hand."""
    monkeypatch.setenv("EULLM_BUDGET_HOURS", "100")
    monkeypatch.setenv("EULLM_BUDGET_START", "2026-09-02")
    monkeypatch.setenv("EULLM_BUDGET_END", "2999-01-01")
    out = run_status(tmp_path, queue=[("eullm-grpo", "RUNNING", "None")], ended=[])
    assert "allocation: 58 of 100 local hours used since 2026-09-02, 42 left for" in out
    assert "pace of the window above:" in out and "saldo -b is the bill" in out
    assert "[!!]" not in out
    monkeypatch.setenv("EULLM_BUDGET_END", "2000-01-01")
    assert "it has ended" in run_status(tmp_path, queue=[("eullm-grpo", "RUNNING", "None")],
                                        ended=[])


def test_the_bill_from_saldo_is_shown_with_the_even_pace(tmp_path, runs):
    """2026-10-13: a catch-up plan was worked out on this script's 63% while
    saldo, the bill, said 59.0%. Where saldo is, its figure is shown too."""
    bin_ = tmp_path / "bin"
    bin_.mkdir(exist_ok=True)
    _exe(bin_ / "saldo", """#!/usr/bin/env bash
echo "---------------------------------------------------------------------"
echo "account                start         end         total        localCluster   totConsumed     totConsumed     monthTotal     monthConsumed"
echo "                                             (local h)   Consumed(local h)     (local h)               %      (local h)         (local h)"
echo "---------------------------------------------------------------------"
echo "AIFAC_P02_1147      20000101    29991231         40000               23616         23616            59.0         19672               6818"
""")  # noqa: E501
    out = run_status(tmp_path, queue=[("eullm-grpo", "RUNNING", "None")], ended=[])
    assert "bill (saldo -b, AIFAC_P02_1147): 23616 of 40000 local hours, 59.0%; " \
           "the even pace is" in out
    assert "ahead" in out and "to use the rest by 29991231:" in out


def test_opd_progress_and_case_law_results_are_shown(tmp_path, runs, monkeypatch):
    """2026-10-06: the user asked how to tell that a pilot's first link had run."""
    logs = runs.parent / "opd" / "logs"
    logs.mkdir(parents=True)
    (logs / "eullm-opd-opd2-4b-80.out").write_text(
        "[opd] repo /w/eullm (fcebb63), student cuda:0, teacher 1,2\n"
        "[opd] step 1/60 kl 0.7665 len 174 53s/step\n"
        "[opd] step 2/60 kl 0.7001 len 180 45s/step\n"
        "[opd] step 33/60 kl 0.4194 len 210 45s/step\n"
        "[opd] time is up at step 33: saved, the next link carries on\n", encoding="utf-8")
    repo_logs = tmp_path / "repo-logs"
    repo_logs.mkdir()
    (repo_logs / "eullm-cds-exam-9.out").write_text(
        "[cds-exam] v04-4b-q4: 1200 questions; source ruling retrieved 0.350, cited 0.069; "
        "answers citing only rulings they were given 0.677 -> /w/answers-v04-4b-q4.jsonl\n",
        encoding="utf-8")
    monkeypatch.setenv("EULLM_REPO_LOGS", str(repo_logs))
    out = run_status(tmp_path, queue=[("eullm-opd", "RUNNING", "None")], ended=[])
    assert "     [opd] step 1/60 kl 0.7665" in out
    assert "     [opd] time is up at step 33" in out and "step 2/60" not in out
    assert "   [cds-exam] v04-4b-q4: 1200 questions; source ruling retrieved 0.350" in out
    assert "nothing wrong found" in out


def test_a_grpo_link_logged_in_the_checkout_is_read_there(tmp_path, runs, monkeypatch):
    # chains submitted from $WORK/eullm without -o write to its logs/
    repo_logs = tmp_path / "repo-logs"
    repo_logs.mkdir()
    (repo_logs / "eullm-grpo-jg-q35-9b-r2n-85.out").write_text(
        "[grpo] adapter already at /w/eullm_runs/grpo/jg-q35-9b-r2n/adapter: nothing "
        "left to do (move it away to train again)\n", encoding="utf-8")
    monkeypatch.setenv("EULLM_REPO_LOGS", str(repo_logs))
    out = run_status(tmp_path, queue=[],
                     ended=[("85", "eullm-grpo-jg-q35-9b-r2n", "COMPLETED", "00:01:17")])
    assert "85 eullm-grpo-jg-q35-9b-r2n: ended in 00:01:17 with nothing left to do" in out
    assert "a link that short did no work" not in out

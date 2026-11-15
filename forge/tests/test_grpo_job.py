"""sbatch_grpo.slurm with a judge: who gets which GPU, and the judge goes away.

Stubs stand in for nvidia-smi (three GPUs, numbered as Slurm hands them),
the venv's python, llama-server and accelerate; each records what it was
given. Pinned: the judge runs on the LAST of the job's GPUs, the policy on
the others with one process fewer, grpo_train.py gets the judge's URL, the
server is stopped when the job ends, and a job without a judge is unchanged.
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
SCRIPT = REPO / "forge" / "scripts" / "leonardo" / "sbatch_grpo.slurm"

pytestmark = pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None,
                                reason="needs POSIX bash")


def _exe(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def job(tmp_path):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    rec = tmp_path / "rec"
    rec.mkdir()
    _exe(bin_ / "nvidia-smi", "#!/bin/sh\nprintf 'GPU 0\\nGPU 1\\nGPU 2\\n'\n")
    _exe(bin_ / "python", """#!/bin/sh
case "$*" in *socket*) echo 41234 ;; esac
exit 0
""")
    _exe(bin_ / "accelerate", f"""#!/bin/sh
echo "$CUDA_VISIBLE_DEVICES" > {rec}/acc_devs
echo "$@" > {rec}/acc_args
""")
    server = tmp_path / "llama-server"
    _exe(server, f"""#!/bin/sh
echo "$CUDA_VISIBLE_DEVICES" > {rec}/srv_devs
echo "$@" > {rec}/srv_args
echo $$ > {rec}/srv_pid
exec sleep 30
""")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("{}\n")
    judge = tmp_path / "judge-q8_0.gguf"
    judge.write_bytes(b"GGUF")
    work = tmp_path / "work"
    work.mkdir()
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "WORK": str(work),
           "GRPO_REPO": str(REPO), "GRPO_MODEL": str(model), "GRPO_PROMPTS": str(prompts),
           "GRPO_OUT": str(tmp_path / "out"), "LCPP_SERVER": str(server),
           "EULLM_VENV": str(tmp_path / "nov"), "HEARTBEAT_INTERVAL": "1",
           "CUDA_VISIBLE_DEVICES": "1,2,3"}
    env.pop("SLURM_JOB_ID", None)

    def run(**extra):
        return subprocess.run(["bash", str(SCRIPT)], cwd=tmp_path, env={**env, **extra},
                              capture_output=True, text=True, encoding="utf-8", timeout=60)

    return run, rec, judge


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def test_the_judge_takes_the_last_gpu_and_the_policy_the_others(job):
    run, rec, judge = job
    r = run(GRPO_JUDGE_GGUF=str(judge))
    assert r.returncode == 0, r.stdout + r.stderr
    assert (rec / "srv_devs").read_text().strip() == "3"
    assert (rec / "acc_devs").read_text().strip() == "1,2"
    args = (rec / "acc_args").read_text().split()
    assert args[args.index("--num_processes") + 1] == "2"
    assert args[args.index("--judge-url") + 1] == "http://127.0.0.1:41234"
    srv = (rec / "srv_args").read_text().split()
    assert srv[srv.index("-m") + 1] == str(judge) and srv[srv.index("--port") + 1] == "41234"
    import time
    pid = int((rec / "srv_pid").read_text())
    for _ in range(50):
        if _gone(pid):
            break
        time.sleep(0.1)
    assert _gone(pid), "the judge server outlived the job"


def test_without_a_judge_every_gpu_trains(job):
    run, rec, _ = job
    r = run()
    assert r.returncode == 0, r.stdout + r.stderr
    args = (rec / "acc_args").read_text().split()
    assert args[args.index("--num_processes") + 1] == "3" and "--judge-url" not in args
    assert not (rec / "srv_args").exists()


def test_a_missing_judge_file_stops_before_training(job, tmp_path):
    run, rec, _ = job
    r = run(GRPO_JUDGE_GGUF=str(tmp_path / "nope.gguf"))
    assert r.returncode == 1 and "no judge GGUF" in r.stderr
    assert not (rec / "acc_args").exists()

"""heartbeat.sh: a job that fails at start-up cancels the rest of its chain.

On 2026-11-04 the 24 links of the A1 chains all died in 1-4 minutes of the
same error, each releasing the next through its afterany dependency. Pinned
here, with squeue and scancel stubbed: an early failure cancels every job
behind it, transitively, and nothing else; a late failure, a success and a
signal cancel nothing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HEARTBEAT = Path(__file__).resolve().parents[1] / "scripts" / "leonardo" / "heartbeat.sh"

pytestmark = pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None,
                                reason="needs POSIX bash")

# 101 is this job; 102 and 103 are its chain, 104 waits on 103 among others,
# 200 waits on an unrelated job, 300 has no dependency.
QUEUE = """102 afterany:101(unfulfilled)
103 afterany:102(unfulfilled)
104 afterok:999(unfulfilled),afterany:103(unfulfilled)
200 afterany:1010(unfulfilled)
300 (null)
"""


@pytest.fixture
def run(tmp_path):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "squeue").write_text(f"#!/bin/sh\ncat <<'EOF'\n{QUEUE}EOF\n")
    (bin_ / "scancel").write_text(f'#!/bin/sh\necho "$@" >> {tmp_path / "cancelled"}\n')
    for f in bin_.iterdir():
        f.chmod(0o755)

    def go(body: str, **env):
        script = f'source "{HEARTBEAT}"\nheartbeat_start\n{body}\n'
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30,
                           env={**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}",
                                "SLURM_JOB_ID": "101", "HEARTBEAT_INTERVAL": "1", **env})
        cancelled = tmp_path / "cancelled"
        out = cancelled.read_text().split() if cancelled.exists() else []
        cancelled.unlink(missing_ok=True)
        return r, out

    return go


def test_an_early_failure_cancels_the_chain_behind_it(run):
    r, cancelled = run("exit 1")
    assert r.returncode == 1
    assert sorted(cancelled) == ["102", "103", "104"]
    assert "cancelling the jobs queued behind it" in r.stdout


def test_a_failure_after_the_window_cancels_nothing(run):
    r, cancelled = run("sleep 2; exit 1", HEARTBEAT_FAST_FAIL="1")
    assert r.returncode == 1 and cancelled == []


def test_success_signal_and_switched_off_cancel_nothing(run):
    assert run("exit 0")[1] == []
    assert run("exit 143")[1] == []
    assert run("exit 1", HEARTBEAT_FAST_FAIL="0")[1] == []

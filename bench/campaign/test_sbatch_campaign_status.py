"""A LUMI campaign link killed at walltime must exit with the runner's status.

Slurm sends SIGTERM at the walltime; the wrapper forwards it to the Python
runner, which shuts down and returns 0. The interrupted `wait` has already
stored 143, and the old `wait || STATUS=$?` in the reaping loop never
replaced it, so Slurm recorded every normal 48 h end as FAILED
(NonZeroExitCode).

The test lifts the reaping loop out of the real script, so it cannot drift,
and starts it with STATUS=143 as the interrupted first wait leaves it.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "lumi" / "sbatch_campaign.slurm"

pytestmark = pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None,
                                reason="needs POSIX bash")


def _bash() -> str:
    return os.environ.get("EULLM_TEST_BASH") or shutil.which("bash") or "bash"


def test_the_reaping_loop_replaces_the_interrupted_waits_143(tmp_path):
    text = SCRIPT.read_text(encoding="utf-8")
    m = re.search(r'while kill -0 "\$RUNNER".*?\ndone', text, re.S)
    assert m, "the reaping loop moved; update this test with it"
    marker = tmp_path / "runner.started"
    (tmp_path / "runner.sh").write_text(
        "#!/bin/bash\n"
        f"touch {marker.as_posix()}\n"
        "sleep 1\n"
        "exit 0\n")
    (tmp_path / "runner.sh").chmod(0o755)
    wrapper = tmp_path / "wrapper.sh"
    wrapper.write_text(
        "#!/bin/bash\n"
        "set -uo pipefail\n"
        'bash "$1" &\n'
        "RUNNER=$!\n"
        f"while [ ! -f {marker.as_posix()} ]; do sleep 0.05; done\n"
        "STATUS=143  # the interrupted wait, as signal delivery leaves it\n"
        + m.group(0) + "\n"
        'exit "$STATUS"\n')
    p = subprocess.run([_bash(), str(wrapper), str(tmp_path / "runner.sh")],
                       capture_output=True, text=True, encoding="utf-8", timeout=15)
    assert p.returncode == 0, p.stdout + p.stderr

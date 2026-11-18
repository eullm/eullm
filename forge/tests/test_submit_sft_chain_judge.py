"""submit_sft_chain.sh must move a judge already waiting, not submit another.

At the normal end of a chain the last link has finished and the judge is
PENDING. `QUEUED` is empty, so the old `if [ -n "$QUEUED" ]` skipped the
lookup and a second judge was submitted; two judges scan and write the same
graded files. The test lifts the lookup block out of the real script.
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
SCRIPT = ROOT / "forge" / "scripts" / "jupiter" / "submit_sft_chain.sh"

pytestmark = pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None,
                                reason="needs POSIX bash")


def _bash() -> str:
    return os.environ.get("EULLM_TEST_BASH") or shutil.which("bash") or "bash"


def _exe(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def test_a_legacy_pending_judge_on_the_finished_tail_is_moved(tmp_path):
    text = SCRIPT.read_text(encoding="utf-8")
    m = re.search(r'judge=""\n.*?(?=\n\[ -n "\$judge" \])', text, re.S)
    assert m, "the judge lookup moved; update this test with it"
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    updates = tmp_path / "updates.log"
    _exe(bin_ / "squeue", "#!/bin/sh\n"
         'case "$*" in *forge-judge-ob1*) ;; *) printf "101\\n" ;; esac\n')
    _exe(bin_ / "scontrol", f"""#!/bin/sh
case "$*" in
  "show job 101") echo "JobId=101 Dependency=afterany:100" ;;
  "show job 100") echo "JobId=100 JobName=forge-sft-ob1" ;;
  update*) echo "$*" >> {updates.as_posix()} ;;
esac
""")
    _exe(bin_ / "sacct", "#!/bin/sh\nexit 0\n")
    wrapper = tmp_path / "wrapper.sh"
    wrapper.write_text(
        "#!/bin/bash\n"
        "set -uo pipefail\n"
        "NAME=forge-sft-ob1\nTAG=ob1\nJUDGE_NAME=forge-judge-ob1\nUSER=tester\n"
        "QUEUED=\nprev=102\n"
        + m.group(0) + "\n"
        'echo "JUDGE:$judge"\n', encoding="utf-8")
    env = {**os.environ, "PATH": bin_.as_posix() + ":" + os.environ.get("PATH", "")}
    p = subprocess.run([_bash(), str(wrapper)], env=env, capture_output=True, text=True, encoding="utf-8",
                       timeout=20)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "JUDGE:101" in p.stdout
    assert "JobId=101 Dependency=afterany:102" in updates.read_text(encoding="utf-8")

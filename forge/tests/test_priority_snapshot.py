"""priority_snapshot.run reads the scheduler's output, as users wrote it.

sprio and squeue print job names, and sacct prints everything the day had:
the snapshot is a record of those commands, so their UTF-8 output must not
depend on the host's default encoding.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "leonardo" / "priority_snapshot.py"
spec = importlib.util.spec_from_file_location("priority_snapshot", SCRIPT)
ps = importlib.util.module_from_spec(spec)
sys.modules["priority_snapshot"] = ps
spec.loader.exec_module(ps)


def test_scheduler_output_is_read_as_utf8(monkeypatch):
    class Out:
        stdout = "citt\u00e0-job RUNNING\n"
        stderr = ""
        returncode = 0

    seen = []

    def run(cmd, **kwargs):
        seen.append(kwargs)
        return Out()

    monkeypatch.setattr(ps.subprocess, "run", run)
    assert ps.run(["sprio"]) == ("citt\u00e0-job RUNNING\n", "")
    assert seen[0].get("encoding") == "utf-8"

"""Runs the shell test suite for `submit_chain.sh` under pytest.

The script is bash and its test suite is bash too — a stub `sbatch` on PATH
recording the arguments each link is submitted with, which is the only way to
check dependency wiring without a scheduler. CI runs `pytest`, so without this
wrapper that suite would sit in the tree and never execute, which for a test
guarding a silent failure is the same as not having it.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SUITE = Path(__file__).resolve().parent / "test_submit_chain.sh"


@pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None,
                    reason="needs POSIX bash (on Windows, bash is WSL's)")
def test_submit_chain_dependency_wiring():
    result = subprocess.run(
        ["bash", str(SUITE)],
        capture_output=True, text=True, encoding="utf-8", timeout=120,
    )
    # The suite prints one line per check; surface all of it on failure so the
    # report names which dependency came out wrong.
    assert result.returncode == 0, result.stdout + result.stderr

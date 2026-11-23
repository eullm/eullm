"""`archive_results.sh` must not destroy the day's good archive on failure.

tar opens its output before it reads the tree, so writing the final dated
file directly truncated the previous archive first; a disk-full or unreadable
operand then left a partial file under the documented name and the good copy
gone. Reruns on the same date are the documented workflow, so this was one
failed tar away from losing the safety copy.
"""

from __future__ import annotations

import datetime as dt
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "lumi" / "archive_results.sh"

pytestmark = pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None,
                                reason="needs POSIX bash (on Windows, bash is WSL's)")


def _bash() -> str:
    return os.environ.get("EULLM_TEST_BASH") or shutil.which("bash") or "bash"


def _env(tmp_path: Path, *, tar_dir: Path | None = None) -> dict[str, str]:
    repo = tmp_path / "repo"
    (repo / "bench" / "campaign").mkdir(parents=True)
    (repo / "bench" / "campaign" / "campaign.py").write_text(
        "raise SystemExit(0)\n", encoding="utf-8")
    campaign = tmp_path / "campaign"
    (campaign / "results").mkdir(parents=True)
    (campaign / "results" / "row.json").write_text("{}\n", encoding="utf-8")
    archive = tmp_path / "archive"
    archive.mkdir()
    env = {
        **os.environ,
        "EULLM_REPO": repo.as_posix(),
        "CAMPAIGN_DIR": campaign.as_posix(),
        "EULLM_ARCHIVE_DIR": archive.as_posix(),
        "PY": sys.executable,
        "USER": os.environ.get("USERNAME", "tester"),
    }
    if tar_dir:
        env["PATH"] = tar_dir.as_posix() + os.pathsep + env.get("PATH", "")
    return env


def _run(env: dict[str, str]):
    return subprocess.run([_bash(), str(SCRIPT)], env=env, capture_output=True, text=True,
                          encoding="utf-8")


def test_a_failed_tar_keeps_the_previous_archive(tmp_path):
    out = tmp_path / "archive" / f"results-{dt.date.today():%F}.tar.gz"
    tar_dir = tmp_path / "bin"
    tar_dir.mkdir()
    stub = tar_dir / "tar"
    stub.write_text('#!/bin/sh\nprintf PARTIAL > "$2"\nexit 2\n', encoding="utf-8")
    stub.chmod(0o755)
    env = _env(tmp_path, tar_dir=tar_dir)
    env["EULLM_ARCHIVE_DIR"] = out.parent.as_posix()
    out.write_bytes(b"OLD-GOOD-ARCHIVE")

    proc = _run(env)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert out.read_bytes() == b"OLD-GOOD-ARCHIVE"
    assert not list(out.parent.glob(".*.tar.gz"))
    assert "is unchanged" in proc.stderr


def test_a_successful_tar_replaces_the_day_file(tmp_path):
    out = tmp_path / "archive" / f"results-{dt.date.today():%F}.tar.gz"
    env = _env(tmp_path)
    env["EULLM_ARCHIVE_DIR"] = out.parent.as_posix()
    out.write_bytes(b"OLD-GOOD-ARCHIVE")

    proc = _run(env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert out.read_bytes() != b"OLD-GOOD-ARCHIVE"
    assert out.stat().st_size > 0
    assert not list(out.parent.glob(".*.tar.gz"))

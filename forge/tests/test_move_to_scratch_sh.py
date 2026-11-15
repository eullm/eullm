"""`move_to_scratch.sh`: copy, swap for a symlink, and never force.

The base models it moves are read by chains that are running while it runs,
so what is pinned is what a job opening the path finds afterwards (the same
files, through the symlink), and that a path it cannot move safely is left
exactly as it was.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "leonardo" / "move_to_scratch.sh"

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or not (shutil.which("bash") and shutil.which("rsync")),
    reason="needs POSIX bash and rsync")


def _model(root: Path, name: str) -> Path:
    d = root / name
    (d / "snapshots" / "abc").mkdir(parents=True)
    (d / "blobs").mkdir()
    (d / "blobs" / "w").write_bytes(b"weights" * 100)
    # the Hugging Face cache links snapshots to blobs relatively; they must survive
    (d / "snapshots" / "abc" / "model.safetensors").symlink_to("../../blobs/w")
    (d / "config.json").write_text("{}")
    return d


def _run(*args: str, margin: str = "0") -> str:
    out = subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, encoding="utf-8",
                         timeout=60, env={"PATH": "/usr/bin:/bin", "MOVE_MARGIN_GB": margin})
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_a_model_is_copied_then_replaced_by_a_symlink(tmp_path):
    work, scratch = tmp_path / "work", tmp_path / "scratch"
    src = _model(work, "Qwen3-14B")
    out = _run(str(scratch), str(src))
    assert f"[move] {src} -> {scratch / 'Qwen3-14B'}" in out
    assert src.is_symlink() and src.resolve() == (scratch / "Qwen3-14B").resolve()
    assert (src / "config.json").read_text() == "{}"
    assert (src / "snapshots" / "abc" / "model.safetensors").read_bytes() == b"weights" * 100
    assert not (work / "Qwen3-14B.old").exists()
    # run twice, as a resubmitted job would: nothing moves again
    assert "already a symlink" in _run(str(scratch), str(src))


def test_a_path_that_cannot_be_moved_safely_is_left_as_it_was(tmp_path):
    work, scratch = tmp_path / "work", tmp_path / "scratch"
    src = _model(work, "Qwen3.8-27B")
    # a margin larger than any test disk: never enough room
    out = _run(str(scratch), str(src), str(work / "missing"), margin="100000000")
    assert "skipped" in out and "not there" in out
    assert src.is_dir() and not src.is_symlink() and (src / "config.json").exists()
    assert not (scratch / "Qwen3.8-27B").exists()

"""`perplexity_compare.sh` re-measures a rewritten base instead of reusing it.

The base perplexity is cached under a key, because re-measuring costs twenty
minutes. The key named the base file but not its content, so rewriting the
base in place -- re-quantized to the same path, the documented way to build
it -- reused the old number and produced a plausible delta against a model
that was never measured. These tests run the real script with a stubbed
llama-perplexity, measure once, rewrite the base under the same name, and
read back whether the second round re-measures (the test_train_sh.py shape,
skipped on Windows).
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "perplexity_compare.sh"

pytestmark = pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None,
                                reason="needs POSIX bash (on Windows, bash is WSL's)")


def _exe(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def stage(tmp_path, monkeypatch):
    lcpp = tmp_path / "llama.cpp" / "build" / "bin"
    lcpp.mkdir(parents=True)
    calls = tmp_path / "calls.log"
    # A perplexity binary that answers at once and records what it measured.
    _exe(lcpp / "llama-perplexity",
         "#!/usr/bin/env bash\n"
         f'echo "measured $2" >> "{calls}"\n'
         'echo "Final estimate: PPL = 5.00"\n')
    monkeypatch.setenv("LCPP_DIR", str(tmp_path / "llama.cpp"))

    (tmp_path / "base-q4_k_m.gguf").write_bytes(b"base model weights, version one\n")
    (tmp_path / "student-q4_k_m.gguf").write_bytes(b"student weights\n")
    (tmp_path / "corpus.txt").write_text("held-out corpus text\n")
    return tmp_path, calls


def _run(stage_dir: Path, *extra: str):
    proc = subprocess.run(
        ["bash", str(SCRIPT),
         "--student", str(stage_dir / "student-q4_k_m.gguf"),
         "--base", str(stage_dir / "base-q4_k_m.gguf"),
         "--corpus", str(stage_dir / "corpus.txt"),
         "--chunks", "4",
         "--base-ppl-cache", str(stage_dir / "base-ppl.txt"),
         *extra],
        capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, proc.stderr
    return proc.stdout + proc.stderr


def _measured(calls: Path) -> list[str]:
    return calls.read_text(encoding="utf-8").splitlines() if calls.exists() else []


def test_a_rewritten_base_is_measured_again_not_recalled(stage):
    stage_dir, calls = stage
    first = _run(stage_dir)
    assert "PPL = 5.00" in first and _measured(calls) != []

    # Re-quantized to the same path: same name AND same byte count (same
    # shapes quantize to the same size), so only the mtime tells them apart.
    # Forced apart because fixtures are written within one second.
    base = stage_dir / "base-q4_k_m.gguf"
    base.write_bytes(b"base model weights, version two\n")
    os.utime(base, (base.stat().st_atime + 100, base.stat().st_mtime + 100))
    calls.unlink()

    second = _run(stage_dir)
    assert "not re-measuring" not in second
    assert "re-measuring" in second
    # ...and the stub proves it: the base was measured again.
    assert any("base-q4_k_m.gguf" in line for line in _measured(calls)), _measured(calls)


def test_an_untouched_base_is_recalled_not_measured(stage):
    stage_dir, calls = stage
    _run(stage_dir)
    calls.unlink()

    second = _run(stage_dir)
    assert "not re-measuring" in second
    assert not any("base-q4_k_m.gguf" in line for line in _measured(calls)), _measured(calls)

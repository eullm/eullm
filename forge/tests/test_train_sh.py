"""`train.sh` resumes from the checkpoint it should, with Slurm unnecessary.

The launcher used to rank `checkpoint-*` by mtime, so anything that touched a
checkpoint directory without writing one -- a touch, an rsync, the copy of
$WORK that runs between links -- reordered the list and the resume reloaded a
checkpoint thousands of steps old. distill.sh died this death and its header
still says so; the test pins the mtime fragments out of it. These tests hand
train.sh an output dir where the higher step is the older directory, with
`llamafactory-cli` stubbed on PATH, and read back which checkpoint the
resolved YAML resumes from.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train.sh"

pytestmark = pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None,
                                reason="needs POSIX bash (on Windows, bash is WSL's)")


def _exe(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def stage(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # llamafactory-cli records the resolved YAML it is given and stops.
    _exe(bin_dir / "llamafactory-cli",
         "#!/usr/bin/env bash\n"
         'if [ "$1" != "train" ]; then echo "unexpected: $*" >&2; exit 2; fi\n'
         'grep "^resume_from_checkpoint:" "$2" || echo "resume_from_checkpoint: <none>"\n')
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])

    data = tmp_path / "data"
    data.mkdir()
    (data / "train.jsonl").write_text('{"text": "x"}\n', encoding="utf-8")
    (data / "val.jsonl").write_text('{"text": "x"}\n', encoding="utf-8")

    out = tmp_path / "out"
    out.mkdir()
    config = tmp_path / "train.yaml"
    config.write_text(f"output_dir: {out}\nmodel_name_or_path: x\n", encoding="utf-8")
    return config, data, out


def _run(config: Path, data: Path) -> str:
    proc = subprocess.run(["bash", str(SCRIPT), str(config), str(data)],
                          capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_resume_takes_the_highest_step_not_the_newest_mtime(stage):
    config, data, out = stage
    (out / "checkpoint-9000").mkdir()
    (out / "checkpoint-100").mkdir()
    # the lower step is touched last: an rsync between links does this.
    now = time.time()
    os.utime(out / "checkpoint-9000", (now - 86400, now - 86400))
    os.utime(out / "checkpoint-100", (now, now))

    printed = _run(config, data)
    assert "resume_from_checkpoint: " in printed
    assert printed.rstrip().endswith("checkpoint-9000"), printed


def test_a_non_numeric_checkpoint_suffix_is_not_a_checkpoint(stage):
    config, data, out = stage
    (out / "checkpoint-100").mkdir()
    (out / "checkpoint-final").mkdir()

    printed = _run(config, data)
    assert printed.rstrip().endswith("checkpoint-100"), printed


def test_no_checkpoint_starts_fresh(stage):
    config, data, _ = stage
    printed = _run(config, data)
    assert "resume_from_checkpoint: <none>" in printed

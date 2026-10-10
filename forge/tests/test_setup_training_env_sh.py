"""`setup_training_env.sh` step 4 passes its variables as ENVIRONMENT.

The assignments sat AFTER the closing quote of `$PYTHON -c "..."`, so they
became argv[1:] while the Python read `os.environ`. Neither DATA_DIR nor
TARBALL_NAME is exported (they are assigned at the top of the script and are
not in its required-env list), so `os.environ['DATA_DIR']` raised KeyError
and `set -euo pipefail` aborted before the corpus was ever fetched.

The step is lifted verbatim out of the script and run under bash with
huggingface_hub stubbed, so it cannot drift from the shell it guards.
Skipped on Windows, where `bash` on PATH is WSL's relay.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "setup_training_env.sh"

pytestmark = pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None,
                                reason="needs POSIX bash (on Windows, bash is WSL's)")


def _step() -> str:
    """The command block of step 4, cut out of the script verbatim."""
    after_log = SCRIPT.read_text(encoding="utf-8").split("Downloading ")[1]
    return after_log.split("\n", 1)[1].split("\n\n", 1)[0]


def _stub_hub(tmp_path: Path) -> Path:
    stub = tmp_path / "stub"
    stub.mkdir(exist_ok=True)
    (stub / "huggingface_hub.py").write_text(
        "def hf_hub_download(**kw):\n"
        "    print('GOT', kw['repo_id'], kw['filename'], kw['local_dir'])\n"
        "    return kw['local_dir'] + '/x.tar.gz'\n")
    return stub


def test_the_corpus_download_sees_its_variables_in_the_environment(tmp_path):
    step = _step()
    # The assignments must precede the -c script, not follow its closing
    # quote: after it they are only argv, and os.environ never sees them.
    script_arg = step.index('-c "')
    for name in ("DATA_DIR=", "TARBALL_NAME=", "HF_DATASET="):
        assert step.rindex(name) < script_arg, \
            f"{name} must be a prefix of the command, before -c"

    stub = _stub_hub(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    run = f"""set -eu
DATA_DIR="{data}"
TARBALL_NAME="legal_it_pretraining.tar.gz"
HF_DATASET="stub/dataset"
export HF_DATASET PYTHONPATH="{stub}"
PYTHON="python"
{step}
"""
    proc = subprocess.run([shutil.which("bash"), "-c", run],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"GOT stub/dataset legal_it_pretraining.tar.gz {data}" in proc.stdout, proc.stdout


def test_the_step_needs_no_exported_variables(tmp_path):
    """The documented path: DATA_DIR and TARBALL_NAME are assigned at the top
    of the script and never exported, so the step must not read the
    environment for them. Setting them to empty proves the step still works."""
    stub = _stub_hub(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    run = f"""set -eu
DATA_DIR="{data}"
TARBALL_NAME="legal_it_pretraining.tar.gz"
export HF_DATASET="stub/dataset"
export PYTHONPATH="{stub}"
PYTHON="python"
{_step()}
"""
    proc = subprocess.run([shutil.which("bash"), "-c", run],
                          capture_output=True, text=True,
                          env={**os.environ, "DATA_DIR": "", "TARBALL_NAME": ""})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"GOT stub/dataset legal_it_pretraining.tar.gz {data}" in proc.stdout, proc.stdout
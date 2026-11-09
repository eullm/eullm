"""sbatch_product_eval.slurm (JUPITER): open book by default, closed book on demand.

The real script runs with fake python, eullm, taskset and nvidia-smi on PATH:
python records its arguments. Checked: by default every set is asked with the
retrieved norms; with PE_CLOSED=1 and PE_SETS no norm reaches legal_eval.py,
the answers carry the -q4c tag, and the closed-book suite is scored at the end.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "forge" / "scripts" / "jupiter" / "sbatch_product_eval.slurm"

pytestmark = pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None,
                                reason="needs POSIX bash")


def _exe(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(0o755)


def _setup(tmp_path):
    project = tmp_path / "e-aif-2026pg01-1434"
    home = project / "tester" / "forge"
    data, runs, private = home / "datasets", home / "runs", home / "private-eval"
    for d in (data / "eval", data / "norms-src", private, runs / "m" / "merged"):
        d.mkdir(parents=True)
    (runs / "m" / "merged" / "config.json").write_text("{}")
    (runs / "m" / "m.gguf").write_text("gguf")
    (data / "norms-src" / "legislazione_cc.chunks.jsonl").write_text("{}\n")
    for f in (data / "eval" / "trap-test.jsonl", data / "eval" / "trap-test-para.jsonl",
              data / "eval" / "norm-exam.jsonl", data / "eval" / "closed-book.jsonl",
              private / "lexia-trappole.jsonl", private / "practical-dev.jsonl"):
        f.write_text("{}\n")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    log = tmp_path / "python.log"
    _exe(bin_ / "python", f"#!/bin/bash\necho \"$*\" >> {log.as_posix()}\n")
    _exe(bin_ / "eullm", "#!/bin/sh\necho 'eullm 0.8.5'\n")
    _exe(bin_ / "taskset", "#!/bin/sh\nshift 2\nexec \"$@\"\n")
    _exe(bin_ / "nvidia-smi", "#!/bin/sh\nexit 0\n")
    env = {**os.environ, "PATH": bin_.as_posix() + ":" + os.environ.get("PATH", ""),
           "PROJECT": project.as_posix(), "USER": "tester", "SLURM_JOB_ID": "7",
           "EULLM_REPO": ROOT.as_posix(), "EULLM_BIN": (bin_ / "eullm").as_posix(),
           "PE_MODELS": f"m={runs / 'm' / 'merged'}={runs / 'm' / 'm.gguf'}"}
    for k in ("FORGE_HOME", "FORGE_VENV", "FORGE_DATA_DIR", "FORGE_RUN_DIR",
              "FORGE_PRIVATE_EVAL", "LMOD_CMD", "PE_SETS", "PE_CLOSED"):
        env.pop(k, None)
    return env, data, log


def _run(env):
    return subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True,
                          timeout=60)


def test_open_book_by_default(tmp_path):
    env, _, log = _setup(tmp_path)
    p = _run(env)
    assert p.returncode == 0, p.stdout + p.stderr
    asks = [ln for ln in log.read_text().splitlines() if "legal_eval.py" in ln]
    assert len(asks) == 5
    assert all("--norms" in a and "--no-note" in a and "-q4e" in a for a in asks)
    assert "score_closed_book" not in log.read_text()


def test_closed_book_asks_no_texts_and_scores_the_suite(tmp_path):
    env, data, log = _setup(tmp_path)
    env["PE_CLOSED"] = "1"
    env["PE_SETS"] = f"closed-book={data / 'eval' / 'closed-book.jsonl'}"
    p = _run(env)
    assert p.returncode == 0, p.stdout + p.stderr
    lines = log.read_text().splitlines()
    asks = [ln for ln in lines if "legal_eval.py" in ln]
    assert len(asks) == 1
    assert "--norms" not in asks[0] and "--embedder" not in asks[0]
    assert "--label closed-book-m-q4c" in asks[0]
    assert any("score_closed_book.py" in ln and "closed-book.jsonl" in ln for ln in lines)
    assert "closed book, no texts" in p.stdout

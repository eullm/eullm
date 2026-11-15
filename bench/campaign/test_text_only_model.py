"""text_only_model.sh makes <id>-text: the same weights, linked, and the same
manifest without the projector, so `--mtp` drafts on a model whose repository
ships one (a model with a projector loads on the sequential engine, which
does not draft; c06 measured no draft on LUMI, 24-10-2026)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "lumi" / "text_only_model.sh"

pytestmark = pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None,
                                reason="needs POSIX bash")


def run(store, *ids):
    env = dict(os.environ, EULLM_MODELS_DIR=str(store), PY=sys.executable, USER="tester",
               SBATCH_ACCOUNT="project_test", CAMPAIGN_DIR=str(store.parent / "campaign"))
    env.pop("EULLM_REPO", None)
    return subprocess.run(["bash", str(SCRIPT), *ids], env=env, capture_output=True,
                          text=True, encoding="utf-8", timeout=60)


def test_the_twin_has_the_weights_and_no_projector(tmp_path):
    store = tmp_path / "models"
    src = store / "qwen3.5-9b-mtp-gguf-q4_k_m"
    src.mkdir(parents=True)
    (src / "Qwen3.5-9B-Q4_K_M.gguf").write_bytes(b"GGUF weights")
    (src / "mmproj-F16.gguf").write_bytes(b"GGUF projector")
    (src / "manifest.json").write_text(json.dumps({
        "id": "qwen3.5-9b-mtp-gguf-q4_k_m", "name": "Qwen3.5 9B MTP", "status": "ready",
        "gguf_file": "Qwen3.5-9B-Q4_K_M.gguf", "mmproj_file": "mmproj-F16.gguf",
        "hf_repo": "unsloth/Qwen3.5-9B-MTP-GGUF"}))
    twin = store / "qwen3.5-9b-mtp-gguf-q4_k_m-text"
    # A projector left in the twin's directory goes: the engine would find it
    # there, manifest or not (#1118).
    twin.mkdir()
    (twin / "mmproj-F16.gguf").write_bytes(b"GGUF projector")
    for _ in range(2):  # a second run leaves the twin as it is
        p = run(store, "qwen3.5-9b-mtp-gguf-q4_k_m")
        assert p.returncode == 0, p.stdout + p.stderr
    assert sorted(os.listdir(twin)) == ["Qwen3.5-9B-Q4_K_M.gguf", "manifest.json"]
    assert (twin / "Qwen3.5-9B-Q4_K_M.gguf").read_bytes() == b"GGUF weights"
    assert os.path.samefile(twin / "Qwen3.5-9B-Q4_K_M.gguf", src / "Qwen3.5-9B-Q4_K_M.gguf")
    m = json.loads((twin / "manifest.json").read_text())
    assert m["id"] == "qwen3.5-9b-mtp-gguf-q4_k_m-text" and "mmproj_file" not in m
    assert m["gguf_file"] == "Qwen3.5-9B-Q4_K_M.gguf" and m["status"] == "ready"
    assert m["hf_repo"] == "unsloth/Qwen3.5-9B-MTP-GGUF"
    # The original keeps its projector.
    assert json.loads((src / "manifest.json").read_text())["mmproj_file"] == "mmproj-F16.gguf"


def test_a_model_not_in_the_store_is_an_error(tmp_path):
    (tmp_path / "models").mkdir()
    p = run(tmp_path / "models", "missing-model")
    assert p.returncode != 0 and "pull it first" in p.stderr

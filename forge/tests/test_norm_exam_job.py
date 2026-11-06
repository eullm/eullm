"""sbatch_norm_exam.slurm as one link of a chain, with legal_eval.py stubbed.

What is pinned: an answers file with one line per item is a finished run and
is not asked again, a shorter one is asked again, and EXAM_OPEN_ONLY drops the
closed-book pass. The model is never loaded: `python` is a stub that writes a
full answers file and records what it was asked.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "forge" / "scripts" / "leonardo" / "sbatch_norm_exam.slurm"

pytestmark = pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None,
                                reason="needs POSIX bash (on Windows, bash is WSL's)")

# Writes as many lines as the exam has to the --answers path and logs the label.
STUB = r'''#!/usr/bin/env bash
items=""; answers=""; label=""; how=""
while [ $# -gt 0 ]; do
    case "$1" in
        --items) items="$2"; shift ;;
        --answers) answers="$2"; shift ;;
        --label) label="$2"; shift ;;
        --embedder) how="$how+emb"; shift ;;
        --reranker) how="$how+rr"; shift ;;
        --gguf) how="$how+gguf:$(basename "$2")"; shift ;;
        --llama-server) shift ;;
        --ngram-memory) how="$how+mem:$(basename "$(dirname "$2")")"; shift ;;
        --adapter) how="$how+lora:$(basename "$(dirname "$2")")"; shift ;;
        --max-new-tokens) how="$how+max:$2"; shift ;;
    esac
    shift
done
echo "$label$how" >> "$CALLS"
[ -n "$answers" ] && sed 's/.*/{}/' "$items" > "$answers"
exit 0
'''


@pytest.fixture
def job(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    items = work / "exam-dev.jsonl"
    items.write_text("{}\n{}\n{}\n")
    model = work / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    out = work / "answers"
    out.mkdir()
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    py = bin_ / "python"
    py.write_text(STUB)
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    calls = tmp_path / "calls"
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "WORK": str(work),
           "CALLS": str(calls), "EXAM_REPO": str(REPO), "EXAM_ITEMS": str(items),
           "EXAM_OUT": str(out), "EXAM_MODELS": f"a={model} b={model}",
           "EXAM_NORMS": str(work / "norms.jsonl"), "EULLM_VENV": str(tmp_path / "nov"),
           # the heartbeat's sleep holds the output pipe until it wakes
           "HEARTBEAT_INTERVAL": "1"}
    env.pop("SLURM_JOB_ID", None)

    def run(**extra):
        r = subprocess.run(["bash", str(SCRIPT)], cwd=tmp_path, env={**env, **extra},
                           capture_output=True, text=True, timeout=60)
        asked = calls.read_text().split() if calls.exists() else []
        calls.unlink(missing_ok=True)
        return r, asked

    return run, out


def test_a_second_link_asks_only_what_the_first_did_not_finish(job):
    run, out = job
    (out / "answers-a.jsonl").write_text("{}\n{}\n{}\n")        # whole
    (out / "answers-a-open.jsonl").write_text("{}\n")           # cut short
    r, asked = run()
    assert r.returncode == 0, r.stdout + r.stderr
    assert asked == ["a-open", "b", "b-open"]
    assert "a: answered already, skipped" in r.stdout
    r, asked = run()
    assert r.returncode == 0 and asked == []


def test_open_only_skips_the_closed_book_pass(job):
    run, _ = job
    r, asked = run(EXAM_OPEN_ONLY="1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert asked == ["a-open", "b-open"]


def test_a_hybrid_round_is_labelled_apart_and_passes_the_models(job):
    run, out = job
    (out / "answers-a-open.jsonl").write_text("{}\n{}\n{}\n")   # the BM25 run, done
    r, asked = run(EXAM_OPEN_ONLY="1", EXAM_EMBEDDER="Qwen/Qwen3-Embedding-0.6B",
                   EXAM_RERANKER="Qwen/Qwen3-Reranker-4B", EXAM_OPEN_TAG="hyb")
    assert r.returncode == 0, r.stdout + r.stderr
    assert asked == ["a-hyb+emb+rr", "b-hyb+emb+rr"]
    assert (out / "answers-a-open.jsonl").read_text() == "{}\n{}\n{}\n"


def test_a_gguf_label_answers_with_the_gguf_and_the_others_do_not(job, tmp_path):
    run, _ = job
    gguf = tmp_path / "a-q4_k_m.gguf"
    gguf.write_bytes(b"GGUF")
    server = tmp_path / "llama-server"
    server.write_text("#!/bin/sh\n")
    server.chmod(0o755)
    r, asked = run(EXAM_OPEN_ONLY="1", EXAM_GGUF=f"a={gguf}", LCPP_SERVER=str(server))
    assert r.returncode == 0, r.stdout + r.stderr
    assert asked == ["a-open+gguf:a-q4_k_m.gguf", "b-open"]


def test_a_gguf_round_without_the_server_or_the_file_fails_before_asking(job, tmp_path):
    run, _ = job
    gguf = tmp_path / "a-q4_k_m.gguf"
    gguf.write_bytes(b"GGUF")
    r, asked = run(EXAM_GGUF=f"a={gguf}", LCPP_SERVER=str(tmp_path / "missing"))
    assert r.returncode == 1 and asked == []
    assert "no llama-server" in r.stderr
    server = tmp_path / "llama-server"
    server.write_text("#!/bin/sh\n")
    server.chmod(0o755)
    r, asked = run(EXAM_GGUF=f"a={tmp_path / 'nope.gguf'}", LCPP_SERVER=str(server))
    assert r.returncode == 1 and asked == [] and "no GGUF" in r.stderr
    proj = tmp_path / "mmproj-F16.gguf"
    proj.write_bytes(b"GGUF")
    r, asked = run(EXAM_GGUF=f"a={proj}", LCPP_SERVER=str(server))
    assert r.returncode == 1 and asked == [] and "vision projector" in r.stderr


def test_a_gguf_directory_answers_with_its_model_not_the_projector(job, tmp_path):
    run, _ = job
    d = tmp_path / "flash-next-q4"
    d.mkdir()
    for name in ("mmproj-F16.gguf", "Flash-Q4-00001-of-00003.gguf",
                 "Flash-Q4-00002-of-00003.gguf", "Flash-Q4-00003-of-00003.gguf"):
        (d / name).write_bytes(b"GGUF")
    server = tmp_path / "llama-server"
    server.write_text("#!/bin/sh\n")
    server.chmod(0o755)
    r, asked = run(EXAM_OPEN_ONLY="1", EXAM_GGUF=f"a={d}", LCPP_SERVER=str(server))
    assert r.returncode == 0, r.stdout + r.stderr
    assert asked == ["a-open+gguf:Flash-Q4-00001-of-00003.gguf", "b-open"]


def test_memory_and_adapter_labels_are_grafted_and_the_others_not(job, tmp_path):
    run, _ = job
    mem = tmp_path / "a2-ngram" / "memory"
    mem.mkdir(parents=True)
    (mem / "config.json").write_text("{}")
    lora = tmp_path / "a2-lora" / "adapter"
    lora.mkdir(parents=True)
    (lora / "adapter_config.json").write_text("{}")
    r, asked = run(EXAM_NORMS="", EXAM_MEMORY=f"a={mem}", EXAM_ADAPTER=f"b={lora}")
    assert r.returncode == 0, r.stdout + r.stderr
    assert asked == ["a+mem:a2-ngram", "b+lora:a2-lora"]


def test_a_missing_memory_fails_before_asking(job, tmp_path):
    run, _ = job
    r, asked = run(EXAM_MEMORY=f"a={tmp_path / 'nope'}")
    assert r.returncode == 1 and asked == [] and "no n-gram memory" in r.stderr
    r, asked = run(EXAM_ADAPTER=f"b={tmp_path / 'nope'}")
    assert r.returncode == 1 and asked == [] and "no LoRA adapter" in r.stderr


def test_exam_args_reach_every_run(job):
    run, _ = job
    r, asked = run(EXAM_ARGS="--max-new-tokens 1000")
    assert r.returncode == 0, r.stdout + r.stderr
    assert asked == ["a+max:1000", "a-open+max:1000", "b+max:1000", "b-open+max:1000"]

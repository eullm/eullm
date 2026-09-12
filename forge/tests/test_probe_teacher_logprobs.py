"""Tests for the P1 probe's reading of what vLLM returns.

The probe's job is to decide whether a pre-registered prediction held, so the
part worth testing is the judgement, not the inference. Two rules are easy to
get backwards and would silently invert the verdict:

  * position 0 has no preceding context and comes back as None — counting it
    as a short position would falsify P1 on every run;
  * vLLM may add the prompt's own token on top of the top-K, so K+1 entries
    is a pass.

vLLM is not importable here (and needs GPUs), so the shapes it returns are
reconstructed as plain dicts. That is the contract the probe reads.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = (Path(__file__).resolve().parents[1]
          / "scripts" / "pilot" / "probe_teacher_logprobs.py")


def _load():
    spec = importlib.util.spec_from_file_location("probe", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


probe = _load()


class FakeLogprob:
    """Stands in for vllm.sequence.Logprob: a normalised value plus a rank."""

    def __init__(self, logprob: float, rank: int) -> None:
        self.logprob = logprob
        self.rank = rank
        self.decoded_token = "x"


class FakeOutput:
    def __init__(self, prompt_logprobs) -> None:
        self.prompt_logprobs = prompt_logprobs


def entry(width: int) -> dict:
    return {i: FakeLogprob(-float(i) - 0.1, i + 1) for i in range(width)}


def test_first_position_is_not_counted_as_short():
    """vLLM returns None for position 0; it has no distribution to return."""
    out = FakeOutput([None] + [entry(64) for _ in range(9)])
    shape = probe.inspect_prompt_logprobs([out], top_k=64)
    assert shape["positions_scored"] == 9
    assert shape["positions_below_k"] == 0


def test_k_plus_one_entries_is_a_pass():
    """The prompt's own token may be added beyond the top-K."""
    out = FakeOutput([None] + [entry(65) for _ in range(4)])
    shape = probe.inspect_prompt_logprobs([out], top_k=64)
    assert shape["positions_below_k"] == 0
    assert shape["entries_per_position"] == [65]


def test_fewer_than_k_is_the_falsification():
    out = FakeOutput([None, entry(64), entry(32), entry(64)])
    shape = probe.inspect_prompt_logprobs([out], top_k=64)
    assert shape["positions_below_k"] == 1
    assert shape["entries_per_position"] == [32, 64]


def test_empty_prompt_logprobs_scores_nothing():
    """A backend that accepts the argument and ignores it must not pass."""
    shape = probe.inspect_prompt_logprobs([FakeOutput(None)], top_k=64)
    assert shape["positions_scored"] == 0


def test_value_attributes_are_recorded():
    """ADR-001 assumes raw logits; vLLM returns `logprob`. Record, don't assume."""
    out = FakeOutput([None, entry(64)])
    shape = probe.inspect_prompt_logprobs([out], top_k=64)
    assert "logprob" in shape["value_attributes"]


def test_positions_accumulate_across_sequences():
    outs = [FakeOutput([None] + [entry(64)] * 4),
            FakeOutput([None] + [entry(64)] * 6)]
    shape = probe.inspect_prompt_logprobs(outs, top_k=64)
    assert shape["positions_scored"] == 10


def test_load_samples_skips_short_and_malformed(tmp_path):
    data = tmp_path / "val.jsonl"
    data.write_text("\n".join([
        json.dumps({"text": "x" * 300}),
        "{not json",
        json.dumps({"text": "too short"}),
        json.dumps({"other": "y" * 300}),
        json.dumps({"text": "z" * 300}),
    ]), encoding="utf-8")
    got = probe.load_samples(data, n=10, seed=1, field="text")
    assert len(got) == 2


# ── offline model resolution ─────────────────────────────────────────────
# Compute nodes have no network. vLLM resolves a repo id against the Hub even
# when told to be offline, so a repo id has to become the prefetched snapshot
# before it reaches vLLM. Every case here corresponds to a way that can go
# wrong quietly — the worst being two jobs scoring against different revisions
# of "the same" model.

def test_repo_id_becomes_the_prefetched_snapshot(tmp_path, monkeypatch):
    snap = tmp_path / "hub" / "models--Qwen--Qwen3-4B-Base" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert probe.resolve_local_model("Qwen/Qwen3-4B-Base") == str(snap)


def test_refs_main_decides_when_several_snapshots_exist(tmp_path, monkeypatch):
    """Otherwise two jobs could score against different revisions in silence."""
    base = tmp_path / "hub" / "models--Qwen--Qwen3-4B-Base"
    old = base / "snapshots" / "old111"
    new = base / "snapshots" / "new222"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    (base / "refs").mkdir()
    (base / "refs" / "main").write_text("old111\n", encoding="utf-8")
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    # The ref wins even though the other directory is newer.
    assert probe.resolve_local_model("Qwen/Qwen3-4B-Base") == str(old)


def test_a_dangling_ref_falls_back_instead_of_failing(tmp_path, monkeypatch):
    base = tmp_path / "hub" / "models--Qwen--Qwen3-4B-Base"
    snap = base / "snapshots" / "real999"
    snap.mkdir(parents=True)
    (base / "refs").mkdir()
    (base / "refs" / "main").write_text("gone000", encoding="utf-8")
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert probe.resolve_local_model("Qwen/Qwen3-4B-Base") == str(snap)


def test_an_existing_path_is_left_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert probe.resolve_local_model(str(tmp_path)) == str(tmp_path)


def test_an_uncached_model_is_passed_through_unchanged(tmp_path, monkeypatch):
    """Not this function's job to fail — let vLLM say what it cannot find."""
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert probe.resolve_local_model("Qwen/Nothing-Here") == "Qwen/Nothing-Here"


def test_no_hf_home_is_passed_through_unchanged(monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    assert probe.resolve_local_model("Qwen/Qwen3-4B-Base") == "Qwen/Qwen3-4B-Base"


# ── cluster settings the script owns ─────────────────────────────────────

def test_flashinfer_sampler_is_off_by_default():
    """It JIT-compiles kernels against an nvcc this cluster does not have.

    Set in the module rather than in a shell: a variable typed into a
    terminal is not versioned, reaches a batch job only if sbatch exports it,
    and differs between login nodes. All three cost jobs on this allocation.
    """
    import os
    assert os.environ["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


def test_an_explicit_setting_still_wins(monkeypatch):
    """setdefault, so the flag can be flipped back to measure it again."""
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "1")
    _load()                                   # re-import with the override set
    import os
    assert os.environ["VLLM_USE_FLASHINFER_SAMPLER"] == "1"

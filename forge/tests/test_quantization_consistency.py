"""The teacher must be quantized identically in Phase 1 and Phase 2.

Phase 1 trains a LoRA adapter on top of a quantized frozen base. Phase 2
loads that adapter onto the teacher and uses the teacher's logits as the
distillation target. If the two phases quantize differently, the adapter is
applied to weights it was never trained against, and the resulting error is
not a degradation of training — it is the target the student is fitted to.

The two configs live in different files, are edited weeks apart, and express
the same fact in different vocabularies (`quantization_bit: 8` versus
`teacher_load_in_8bit: true`). That is precisely the shape of divergence this
project has already paid for more than once, so it is asserted rather than
documented.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

CONFIG_DIR = Path(__file__).resolve().parents[1] / "training" / "configs" / "leonardo"
PHASE1 = CONFIG_DIR / "continued_pt_qwen3_30b_a3b_leonardo.yaml"
PHASE2 = CONFIG_DIR / "distill_qwen3_30b_a3b_to_4b_leonardo.yaml"


def _load(path: Path) -> dict:
    assert path.is_file(), f"missing config: {path}"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _phase2_teacher_bits(cfg: dict) -> int | None:
    """Phase 2 spells quantization as two booleans; normalise to bits."""
    in8 = bool(cfg.get("teacher_load_in_8bit"))
    in4 = bool(cfg.get("teacher_load_in_4bit"))
    assert not (in8 and in4), "teacher_load_in_8bit and _in_4bit are both true"
    if in8:
        return 8
    if in4:
        return 4
    return None


def test_teacher_quantization_matches_across_phases():
    p1_bits = _load(PHASE1).get("quantization_bit")
    p2_bits = _phase2_teacher_bits(_load(PHASE2))
    assert p1_bits == p2_bits, (
        f"Phase 1 trains the adapter on a {p1_bits}-bit base while Phase 2 "
        f"loads the teacher at {p2_bits}-bit. Change both or neither."
    )


def test_phase2_points_at_the_phase1_output():
    p1 = _load(PHASE1)
    p2 = _load(PHASE2)
    out = str(p1["output_dir"]).rstrip("/")
    adapter = str(p2["teacher_adapter"]).rstrip("/")
    assert out == adapter, (
        f"Phase 2 reads the adapter from {adapter!r} but Phase 1 writes "
        f"{out!r}. This has already broken once, after a model rename."
    )


def test_phase1_does_not_shard_the_frozen_base():
    """No ZeRO, FSDP or any other parameter-sharding of a frozen model.

    Sharding exists to distribute what a trainable model needs — optimizer
    state, gradients, parameters under update. With 107 M trainable of
    30.6 B, there is nothing worth sharding, and the all-gather traffic it
    costs took three jobs to OOM before this was believed.
    """
    cfg = _load(PHASE1)
    forbidden = [k for k in ("deepspeed", "fsdp", "fsdp_config") if cfg.get(k)]
    assert not forbidden, (
        f"Phase 1 declares {forbidden}, which shards a frozen base. The base "
        "is quantized and replicated instead; see the config header."
    )


def test_phase1_quantizes_the_frozen_base():
    """A replicated base only fits because it is quantized."""
    cfg = _load(PHASE1)
    bits = cfg.get("quantization_bit")
    assert bits in (4, 8), (
        f"quantization_bit is {bits!r}. Without sharding, the base is "
        "replicated on every GPU: at bf16 that is 61 GB against 64 GB of "
        "card, before activations."
    )


def test_phase1_trains_only_adapters():
    cfg = _load(PHASE1)
    assert cfg.get("finetuning_type") == "lora", (
        "quantization_bit only makes sense with a frozen base; a full "
        "fine-tune of quantized weights is not what this config means."
    )

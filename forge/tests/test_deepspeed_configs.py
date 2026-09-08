"""Guards on the shipped DeepSpeed configs.

DeepSpeed validates `zero_optimization` with pydantic in `extra=forbid`
mode. JSON has no comment syntax, so the natural instinct is to add an
explanatory `"_note"` key next to the setting it explains — and inside that
object it is not an ignored comment, it is a fatal ValidationError that kills
the job. One did, eight minutes into a four-GPU allocation, after the
pre-flight and the dataset pass had already run:

    pydantic_core.ValidationError: 1 validation error for DeepSpeedZeroConfig
    _overlap_comm_note
      Extra inputs are not permitted [type=extra_forbidden]

The top level is not validated the same way, so that is where prose belongs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

CONFIG_DIR = Path(__file__).resolve().parents[1] / "training" / "configs"
DS_CONFIGS = sorted(CONFIG_DIR.glob("ds_*.json"))


def test_there_are_deepspeed_configs_to_check():
    # A glob that silently matches nothing would make every test below pass.
    assert DS_CONFIGS, f"no ds_*.json under {CONFIG_DIR}"


@pytest.mark.parametrize("path", DS_CONFIGS, ids=lambda p: p.name)
def test_config_is_valid_json(path: Path):
    json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", DS_CONFIGS, ids=lambda p: p.name)
def test_comment_keys_only_at_top_level(path: Path):
    """Underscore-prefixed keys are fine at the top level, fatal below it."""
    cfg = json.loads(path.read_text(encoding="utf-8"))

    def walk(node: object, trail: str) -> list[str]:
        if not isinstance(node, dict):
            return []
        bad = [
            f"{trail}.{k}" for k in node if isinstance(k, str) and k.startswith("_")
        ]
        for k, v in node.items():
            bad += walk(v, f"{trail}.{k}")
        return bad

    offenders = [b for k, v in cfg.items() for b in walk(v, k)]
    assert not offenders, (
        f"{path.name}: comment keys below the top level are rejected by "
        f"DeepSpeed's pydantic validation, not ignored: {offenders}. "
        "Move the prose into the top-level _README."
    )


@pytest.mark.parametrize("path", DS_CONFIGS, ids=lambda p: p.name)
def test_zero_optimization_keys_are_known(path: Path):
    """Catch typos and stray keys in the strictly-validated section."""
    known = {
        "stage",
        "allgather_partitions",
        "allgather_bucket_size",
        "overlap_comm",
        "reduce_scatter",
        "reduce_bucket_size",
        "contiguous_gradients",
        "round_robin_gradients",
        "offload_param",
        "offload_optimizer",
        "sub_group_size",
        "stage3_prefetch_bucket_size",
        "stage3_param_persistence_threshold",
        "stage3_max_live_parameters",
        "stage3_max_reuse_distance",
        "stage3_gather_16bit_weights_on_model_save",
        "zero_quantized_weights",
        "zero_hpz_partition_size",
    }
    cfg = json.loads(path.read_text(encoding="utf-8"))
    zero = cfg.get("zero_optimization")
    if zero is None:
        pytest.skip(f"{path.name} has no zero_optimization section")
    unknown = sorted(set(zero) - known)
    assert not unknown, (
        f"{path.name}: unknown key(s) in zero_optimization: {unknown}. "
        "DeepSpeed rejects these outright; check the spelling, or add the "
        "key here if it is a real option this DeepSpeed version supports."
    )

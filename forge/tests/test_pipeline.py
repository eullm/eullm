"""Tests for the pipeline orchestrator and profiles."""


import pytest

from eullm_forge.pipeline import (
    PipelineConfig,
    estimate_target_params,
    load_profile,
    run_pipeline,
)
from eullm_forge.profiles import list_profiles


def test_list_profiles():
    profiles = list_profiles()
    assert "legal_it" in profiles
    assert "medical_de" in profiles
    assert "finance_fr" in profiles


def test_load_legal_it_profile():
    config = load_profile("legal-it")
    assert config.base_model == "Qwen/Qwen3-14B"
    assert "it" in config.languages
    assert "en" in config.languages
    assert config.target_vram_gb == 8
    assert config.pruning.target_ratio == 0.5
    assert config.pruning.strategy == "mlp_first"
    assert config.distillation.temperature == 2.0
    assert config.quantization.bits == 4
    assert config.identity.identity_name == "EULLM Legal IT"


def test_load_medical_de_profile():
    config = load_profile("medical-de")
    assert config.base_model == "Qwen/Qwen3-14B"
    assert "de" in config.languages


def test_load_finance_fr_profile():
    config = load_profile("finance-fr")
    assert config.base_model == "Qwen/Qwen3-14B"
    assert "fr" in config.languages


def test_load_nonexistent_profile():
    with pytest.raises(FileNotFoundError):
        load_profile("nonexistent-profile")


def test_estimate_target_params():
    # 8GB VRAM / 0.6 GB per billion = ~13.3B target
    target = estimate_target_params(14.0, 8)
    assert 13.0 <= target <= 14.0

    # Target should not exceed source
    target = estimate_target_params(7.0, 16)
    assert target == 7.0


def test_pipeline_config_defaults():
    config = PipelineConfig()
    assert config.target_vram_gb == 8
    assert not config.skip_pruning
    assert not config.skip_distillation


# ── Stage ordering and the GGUF contract ────────────────────────────────
#
# Two regressions are guarded here. (1) The identity LoRA adapter used to be
# trained and then dropped: stage 5 exported the pre-LoRA model, so
# `--identity` silently produced a GGUF without the identity. (2) The shipped
# profiles asked for AWQ *and* a GGUF export, a combination llama.cpp's
# converter cannot process — the whole pipeline ran before failing at the very
# last stage.

def test_shipped_profiles_target_gguf_without_hf_quantization():
    from eullm_forge.quantize import METHOD_NONE

    for name in ("legal-it", "medical-de", "finance-fr"):
        config = load_profile(name)
        assert config.export.format == "gguf", name
        assert config.quantization.method == METHOD_NONE, (
            f"{name}: AWQ/GPTQ weights cannot be converted to GGUF; the "
            f"quantization for this target is export.quantization"
        )
        # The GGUF-level quantization is still requested.
        assert config.export.quantization == "q4_k_m", name


def test_awq_plus_gguf_export_is_rejected_up_front():
    from eullm_forge.pipeline import _validate_stage_combination
    from eullm_forge.quantize import METHOD_AWQ

    config = load_profile("legal-it")
    config.quantization.method = METHOD_AWQ
    with pytest.raises(ValueError, match="cannot be exported"):
        _validate_stage_combination(config)


def test_hf_quantization_allowed_for_non_gguf_targets():
    from eullm_forge.pipeline import _validate_stage_combination
    from eullm_forge.quantize import METHOD_AWQ

    config = load_profile("legal-it")
    config.quantization.method = METHOD_AWQ
    config.export.format = "safetensors"
    _validate_stage_combination(config)  # must not raise


def test_skipping_quantization_defuses_the_conflict():
    from eullm_forge.pipeline import _validate_stage_combination
    from eullm_forge.quantize import METHOD_AWQ

    config = load_profile("legal-it")
    config.quantization.method = METHOD_AWQ
    config.skip_quantization = True
    _validate_stage_combination(config)  # must not raise


def test_identity_stage_runs_before_quantization_and_is_merged():
    """The exported model must descend from the identity stage.

    Runs `run_pipeline` with every heavyweight stage stubbed out, and asserts
    on the path threaded between them — the bug was purely in that wiring, so
    it is observable without torch or a GPU.
    """
    import eullm_forge.pipeline as pipeline_mod

    calls: list[str] = []

    def fake_identity(cfg):
        calls.append(f"identity(base={cfg.model_path})")
        return "/tmp/eullm-test/identity-lora/adapter"

    def fake_merge(base, adapter, output_path=None):
        calls.append(f"merge(base={base}, adapter={adapter})")
        return output_path or "/tmp/eullm-test/identity-merged"

    def fake_quantize(model_path, cfg):
        calls.append(f"quantize(in={model_path}, method={cfg.method})")
        return model_path

    exported: dict[str, str] = {}

    def fake_export(cfg):
        exported["model_path"] = cfg.model_path
        calls.append(f"export(in={cfg.model_path})")
        return cfg.output_path

    config = load_profile("legal-it")
    config.output_dir = "/tmp/eullm-test"
    config.skip_pruning = True
    config.skip_distillation = True
    config.base_model = "/tmp/eullm-test/base"

    originals = (
        pipeline_mod.fine_tune_identity,
        pipeline_mod.merge_identity_adapter,
        pipeline_mod.quantize,
        pipeline_mod.export_gguf,
    )
    pipeline_mod.fine_tune_identity = fake_identity
    pipeline_mod.merge_identity_adapter = fake_merge
    pipeline_mod.quantize = fake_quantize
    pipeline_mod.export_gguf = fake_export
    try:
        run_pipeline(config)
    finally:
        (
            pipeline_mod.fine_tune_identity,
            pipeline_mod.merge_identity_adapter,
            pipeline_mod.quantize,
            pipeline_mod.export_gguf,
        ) = originals

    # The adapter is merged, not just written and forgotten.
    assert any(c.startswith("merge(") for c in calls), calls
    # Identity precedes quantization, so the adapter is fitted on the same
    # weights it gets merged into.
    assert calls.index(next(c for c in calls if c.startswith("identity("))) < calls.index(
        next(c for c in calls if c.startswith("quantize("))
    ), calls
    # The decisive assertion: what gets exported is the merged model, never
    # the pre-identity checkpoint.
    assert exported["model_path"] == "/tmp/eullm-test/identity-merged", calls
    assert exported["model_path"] != config.base_model

"""Tests for the pipeline orchestrator and profiles."""

from pathlib import Path

import pytest

from eullm_forge.pipeline import PipelineConfig, load_profile, estimate_target_params
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

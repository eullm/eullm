"""Tests for pruning calibration data loading."""

import pytest


def test_unknown_calibration_dataset_raises_instead_of_silent_fallback(monkeypatch):
    """An explicitly requested calibration dataset that fails to load must fail loudly.

    Falling back to wikitext here would score neuron importance on the wrong
    corpus while logging the requested name, pruning the wrong neurons — damage
    discovered only after days of distillation on the pruned model.
    """
    import sys
    import types
    from unittest.mock import MagicMock

    from eullm_forge import pruning as pruning_module

    def _raise(*args, **kwargs):
        raise FileNotFoundError("Dataset 'no-such-calib' doesn't exist")

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = _raise
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    with pytest.raises(RuntimeError, match="no-such-calib"):
        pruning_module._load_calibration_data("no-such-calib", tokenizer=MagicMock(), num_samples=8)


def test_default_wikitext_is_requested_with_its_config(monkeypatch):
    """The default calibration dataset must be asked for by name AND config.

    `wikitext` declares four configs and marks none of them default, so
    `load_dataset("wikitext")` raises rather than picking one. The fallback
    this PR removes was the only reason PruningConfig's default ever worked.
    Pins the call, so dropping the branch fails here rather than in a job.
    """
    import sys
    import types
    from unittest.mock import MagicMock

    from eullm_forge import pruning as pruning_module

    seen: list[tuple] = []

    def _record(*args, **kwargs):
        seen.append(args)
        return [{"text": "hello"}]

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = _record
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    pruning_module._load_calibration_data("wikitext", tokenizer=MagicMock(), num_samples=1)

    assert seen, "load_dataset was never called"
    assert seen[0][:2] == ("wikitext", "wikitext-2-raw-v1"), (
        f"the default must carry its config, got {seen[0]!r}"
    )


def test_default_dataset_constants_are_shared_with_distill():
    """Both loaders must name the same default dataset from one source of truth.

    #447/#452 showed what happens when the two loaders drift: the default
    worked only by accident through a rescue, and removing the rescue broke
    default runs. The name and config live in distill's constants; pruning
    reuses them, pinned here.
    """
    from eullm_forge import distill as distill_module
    from eullm_forge import pruning as pruning_module

    assert pruning_module.WIKITEXT_DEFAULT == distill_module.WIKITEXT_DEFAULT == "wikitext"
    assert (
        pruning_module.WIKITEXT_DEFAULT_CONFIG
        == distill_module.WIKITEXT_DEFAULT_CONFIG
        == "wikitext-2-raw-v1"
    )

"""Tests for the top-K truncation measurement.

The arithmetic is the part worth pinning. P3 states two thresholds — retained
mass >= 99% and truncation KL < 0.01 nats — as though they were independent,
and they are the same number: KL(truncated-renormalised || full) reduces
exactly to -log(retained mass). A test that fixes this identity stops the
report from presenting one measurement as two agreeing ones.

The model itself is not exercised here; that needs a GPU and belongs to the
job. What is exercised is sampling and the reduction.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).resolve().parents[1]
          / "scripts" / "pilot" / "measure_topk_truncation.py")


def _load():
    spec = importlib.util.spec_from_file_location("p3", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


p3 = _load()


def test_truncation_kl_is_minus_log_retained_mass():
    """The identity P3's two thresholds rest on.

    99% retained is 0.01005 nats, so "mass >= 99%" and "KL < 0.01" are one
    check written twice — and the KL one is very slightly the stricter.
    """
    assert -math.log(0.99) == pytest.approx(0.010050, abs=1e-6)
    # Which means the two thresholds do not even agree at the boundary: 99%
    # retained mass FAILS "KL < 0.01". The mass that clears it is 99.005%,
    # so the KL threshold is the marginally stricter of the two and a run
    # landing between them would score as held on one and falsified on the
    # other. Worth knowing before reporting them side by side.
    assert -math.log(0.99) > 0.01
    assert math.exp(-0.01) == pytest.approx(0.990050, abs=1e-6)


def test_perfect_retention_is_zero_divergence():
    assert -math.log(1.0) == 0.0


def test_load_samples_skips_short_and_malformed(tmp_path):
    data = tmp_path / "val.jsonl"
    data.write_text("\n".join([
        json.dumps({"text": "x" * 300}),
        "{not json",
        json.dumps({"text": "too short"}),
        json.dumps({"other": "y" * 300}),
        json.dumps({"text": "z" * 300}),
    ]), encoding="utf-8")
    assert len(p3.load_samples(data, n=10, seed=1, field="text")) == 2


def test_load_samples_is_deterministic_for_a_seed(tmp_path):
    data = tmp_path / "val.jsonl"
    data.write_text(
        "\n".join(json.dumps({"text": f"{i}" + "x" * 300}) for i in range(50)),
        encoding="utf-8",
    )
    a = p3.load_samples(data, n=5, seed=42, field="text")
    b = p3.load_samples(data, n=5, seed=42, field="text")
    assert a == b
    assert p3.load_samples(data, n=5, seed=43, field="text") != a


def test_reservoir_returns_everything_when_the_corpus_is_small(tmp_path):
    data = tmp_path / "val.jsonl"
    data.write_text(
        "\n".join(json.dumps({"text": f"{i}" + "y" * 300}) for i in range(3)),
        encoding="utf-8",
    )
    assert len(p3.load_samples(data, n=100, seed=1, field="text")) == 3


def test_defaults_span_the_range_the_adr_costs_out():
    """ADR-001 Part 4 prices K = 16 / 32 / 64 / 128; all four get measured."""
    assert p3.DEFAULT_KS == (16, 32, 64, 128)
    # Truncation bites harder as T rises, so a single temperature would not
    # answer the question a distillation schedule actually asks.
    assert p3.DEFAULT_TEMPERATURES == (1.0, 2.0, 4.0)

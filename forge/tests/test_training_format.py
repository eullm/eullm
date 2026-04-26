"""Tests for the training_format module."""

from __future__ import annotations

from eullm_forge.datasets.training_format import (
    DEFAULT_KEEP_FIELDS,
    FormatStats,
    iter_slimmed,
    slim_record,
    split_indices,
)


def test_slim_record_keeps_only_default_fields():
    rec = {
        "text": "Una sentenza di Cassazione.",
        "source_id": "snciv/2023/1",
        "year": 2023,
        "kind": "snciv",
        "chunk_index": 0,
        "chunk_total": 1,
        "metadata": {"anonymization": {"person_ner": 5}},
        "internal_blob": "x" * 5000,
    }
    out = slim_record(rec)
    assert out is not None
    assert set(out.keys()) <= set(DEFAULT_KEEP_FIELDS)
    assert "metadata" not in out
    assert "internal_blob" not in out
    assert out["text"] == rec["text"]


def test_slim_record_returns_none_for_empty_text():
    assert slim_record({"text": ""}) is None
    assert slim_record({"text": "   \n  "}) is None
    assert slim_record({}) is None


def test_slim_record_preserves_text_when_field_missing_from_keep_list():
    """The text field is always preserved, even with a custom keep list."""
    rec = {"text": "ciao", "source_id": "x"}
    out = slim_record(rec, keep_fields=("source_id",))
    assert out is not None
    assert out["text"] == "ciao"
    assert out["source_id"] == "x"


def test_split_indices_is_deterministic():
    a_train, a_val = split_indices(1000, val_ratio=0.05, seed=42)
    b_train, b_val = split_indices(1000, val_ratio=0.05, seed=42)
    assert a_train == b_train
    assert a_val == b_val


def test_split_indices_different_seeds_produce_different_splits():
    _, v1 = split_indices(10000, val_ratio=0.05, seed=1)
    _, v2 = split_indices(10000, val_ratio=0.05, seed=2)
    assert v1 != v2


def test_split_indices_no_overlap_and_full_coverage():
    n = 1000
    train, val = split_indices(n, val_ratio=0.05, seed=7)
    assert set(train).isdisjoint(set(val))
    assert sorted(train + val) == list(range(n))


def test_split_indices_at_least_one_val():
    """Even with 1 record and val_ratio=0.001, we get one val record."""
    train, val = split_indices(1, val_ratio=0.001, seed=0)
    assert len(val) == 1
    assert len(train) == 0


def test_iter_slimmed_increments_stats():
    recs = [
        {"text": "alpha", "source_id": "a"},
        {"text": "", "source_id": "b"},        # skipped
        {"text": "  \n  ", "source_id": "c"},  # skipped
        {"text": "beta", "source_id": "d"},
    ]
    stats = FormatStats()
    out = list(iter_slimmed(recs, stats=stats))
    assert len(out) == 2
    assert stats.seen == 4
    assert stats.skipped_empty == 2

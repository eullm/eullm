"""Tests for the teacher logit cache.

ADR-001 Part 6 separates two questions that must not share a test: whether
the cache faithfully records what the teacher said, and how much truncating
to K loses. This file is the first — pure round-trip and invariant checking,
no model involved. The second needs a live teacher and belongs to the pilot.

The bias is deliberate: nearly every test here is about a failure that would
NOT raise the training loss. An off-by-one, a normaliser from the wrong
temperature, a tokenizer that differs in three merges — each produces a cache
that reads back cleanly and trains something subtly wrong, discovered after
the node-hours are gone.
"""

from __future__ import annotations

import numpy as np
import pytest

from eullm_forge.teacher_cache import (
    CacheProvenance,
    LogitCacheReader,
    LogitCacheWriter,
    check_normalisers,
    fingerprint_vocab,
    from_bf16_bits,
    to_bf16_bits,
    validate_sequence,
)

TEMPS = (1.0, 2.0, 4.0)
K = 8
VOCAB = 512


def provenance(top_k: int = K) -> CacheProvenance:
    return CacheProvenance(
        teacher_model="Qwen/Qwen3-30B-A3B-Base",
        teacher_precision="bf16",
        tokenizer_fingerprint="deadbeef",
        top_k=top_k,
        vocab_size=VOCAB,
        temperatures=TEMPS,
    )


def fake_sequence(length: int = 6, top_k: int = K, seed: int = 0):
    """A sequence whose logZ is the exact full-vocabulary normaliser.

    Built from a real (small) vocabulary rather than made up, so the
    normaliser invariants hold the way they would for a real teacher.
    """
    rng = np.random.default_rng(seed)
    token_ids = rng.integers(0, VOCAB, size=length).astype(np.int32)
    n = length - 1
    # A bf16 teacher's logits already carry exactly bf16 precision, which is
    # the ADR's reason for storing them that way. Generating them already
    # rounded makes "lossless with respect to the source" a claim the
    # round-trip tests can check exactly rather than approximately.
    logits = from_bf16_bits(
        to_bf16_bits(rng.normal(0.0, 4.0, size=(n, VOCAB)).astype(np.float32))
    )

    order = np.argsort(-logits, axis=1)[:, :top_k]
    topk_ids = order.astype(np.uint32)
    topk_logits = np.take_along_axis(logits, order, axis=1).astype(np.float32)

    logz = np.zeros((n, len(TEMPS)), dtype=np.float32)
    for j, t in enumerate(TEMPS):
        s = logits / t
        m = s.max(axis=1, keepdims=True)
        logz[:, j] = (m[:, 0] + np.log(np.exp(s - m).sum(axis=1))).astype(np.float32)
    return token_ids, topk_ids, topk_logits, logz, logits


# ── bf16 ─────────────────────────────────────────────────────────────────

def test_bf16_roundtrip_is_exact_for_representable_values():
    x = from_bf16_bits(to_bf16_bits(np.array([1.0, -2.5, 1024.0], dtype=np.float32)))
    assert np.array_equal(x, np.array([1.0, -2.5, 1024.0], dtype=np.float32))


def test_bf16_rounds_to_nearest_rather_than_truncating():
    """Truncation would bias every logit toward zero, systematically.

    Per value it is tiny; across 700 M positions it is a consistent shift of
    the distribution the student learns.
    """
    rng = np.random.default_rng(7)
    x = rng.normal(0, 10, size=200_000).astype(np.float32)
    err = from_bf16_bits(to_bf16_bits(x)) - x
    # Round-to-nearest keeps the mean error near zero; truncation would make
    # it negative in magnitude terms for every value.
    assert abs(float(err.mean())) < float(np.abs(err).mean()) * 0.1


def test_bf16_relative_error_stays_within_the_format():
    rng = np.random.default_rng(11)
    x = rng.normal(0, 10, size=50_000).astype(np.float32)
    rel = np.abs(from_bf16_bits(to_bf16_bits(x)) - x) / np.maximum(np.abs(x), 1e-6)
    assert rel.max() < 2 ** -8


# ── round trip ───────────────────────────────────────────────────────────

def test_written_sequence_reads_back_identically(tmp_path):
    token_ids, topk_ids, topk_logits, logz, _ = fake_sequence()
    with LogitCacheWriter(tmp_path, provenance()) as w:
        w.add("doc-1", token_ids, topk_ids, topk_logits, logz)

    r = LogitCacheReader(tmp_path)
    assert len(r) == 1
    got = r[0]
    assert got.seq_id == "doc-1"
    assert np.array_equal(got.token_ids, token_ids)
    assert np.array_equal(got.topk_ids, topk_ids)
    assert np.allclose(got.logz, logz)
    # Exactly equal, not merely close: the source is already bf16, so the
    # format loses nothing. If this ever becomes approximate, something is
    # converting through a narrower type than it should.
    assert np.array_equal(got.topk_logits, topk_logits)


def test_sequences_survive_sharding(tmp_path):
    with LogitCacheWriter(tmp_path, provenance(), sequences_per_shard=3) as w:
        for i in range(10):
            w.add(f"doc-{i}", *fake_sequence(length=5 + i, seed=i)[:4])

    r = LogitCacheReader(tmp_path)
    assert len(r) == 10
    assert len(r.manifest["shards"]) == 4          # 3 + 3 + 3 + 1
    assert [s.seq_id for s in r] == [f"doc-{i}" for i in range(10)]
    # Variable lengths must not bleed across the offset boundaries.
    for i, s in enumerate(r):
        assert len(s.token_ids) == 5 + i
        assert s.topk_ids.shape[0] == 4 + i


def test_reader_reports_totals(tmp_path):
    with LogitCacheWriter(tmp_path, provenance()) as w:
        for i in range(4):
            w.add(f"d{i}", *fake_sequence(length=6, seed=i)[:4])
    r = LogitCacheReader(tmp_path)
    assert r.manifest["n_sequences"] == 4
    assert r.manifest["n_positions"] == 4 * 5


# ── alignment, the silent one ────────────────────────────────────────────

def test_targets_are_derived_and_offset_by_one(tmp_path):
    token_ids, topk_ids, topk_logits, logz, _ = fake_sequence(length=6)
    with LogitCacheWriter(tmp_path, provenance()) as w:
        w.add("doc", token_ids, topk_ids, topk_logits, logz)
    s = LogitCacheReader(tmp_path)[0]
    assert np.array_equal(s.targets, token_ids[1:])
    assert len(s.targets) == s.topk_ids.shape[0]


def test_one_distribution_too_many_is_refused():
    """The off-by-one that poisons a cache silently."""
    token_ids, topk_ids, topk_logits, logz, _ = fake_sequence(length=6)
    extra = (np.vstack([topk_ids, topk_ids[-1:]]),
             np.vstack([topk_logits, topk_logits[-1:]]),
             np.vstack([logz, logz[-1:]]))
    with pytest.raises(ValueError, match="alignment"):
        validate_sequence(token_ids, *extra, len(TEMPS))


def test_one_distribution_too_few_is_refused():
    token_ids, topk_ids, topk_logits, logz, _ = fake_sequence(length=6)
    with pytest.raises(ValueError, match="alignment"):
        validate_sequence(token_ids, topk_ids[:-1], topk_logits[:-1],
                          logz[:-1], len(TEMPS))


def test_a_single_token_predicts_nothing():
    with pytest.raises(ValueError, match="at least 2 tokens"):
        validate_sequence(np.array([7], dtype=np.int32),
                          np.zeros((0, K), np.uint32),
                          np.zeros((0, K), np.float32),
                          np.zeros((0, 3), np.float32), 3)


# ── normalisers ──────────────────────────────────────────────────────────

def test_logz_below_the_topk_logsumexp_is_refused():
    """logZ is over the whole vocabulary, so it cannot be the smaller one.

    This catches a normaliser computed over the top-K instead of the full
    vocabulary — a cache that is internally consistent, reads back fine, and
    trains against a distribution that sums to one over 8 tokens.
    """
    _, _, topk_logits, logz, _ = fake_sequence()
    check_normalisers(topk_logits, logz, TEMPS, VOCAB)   # the honest one passes
    with pytest.raises(ValueError, match="below the top-K"):
        check_normalisers(topk_logits, logz - 5.0, TEMPS, VOCAB)


def test_logz_from_the_wrong_temperature_is_refused():
    """T=1's normaliser reused for T=4 — the mistake the field invites."""
    _, _, topk_logits, logz, _ = fake_sequence()
    wrong = logz.copy()
    wrong[:, 2] = wrong[:, 0]
    with pytest.raises(ValueError, match="T=4"):
        check_normalisers(topk_logits, wrong, TEMPS, VOCAB)
    # And the lower bound alone would NOT have caught it: logZ falls as T
    # rises, so T=1's value is too large, not too small. This is the gap
    # the test found.
    check_normalisers(topk_logits, wrong, TEMPS, vocab_size=None)


def test_probabilities_and_residual_mass_are_consistent(tmp_path):
    token_ids, topk_ids, topk_logits, logz, full = fake_sequence(length=8, top_k=K)
    with LogitCacheWriter(tmp_path, provenance()) as w:
        w.add("d", token_ids, topk_ids, topk_logits, logz)
    s = LogitCacheReader(tmp_path)[0]

    for t in TEMPS:
        p = s.probs(t)
        assert (p > 0).all()
        total = p.sum(axis=1)
        assert (total <= 1.0 + 1e-3).all()
        assert np.allclose(s.residual_mass(t), 1.0 - total, atol=1e-6)

    # Against the full vocabulary it started from: the top-K mass is what
    # the truncation actually retains.
    s1 = full / 1.0
    m = s1.max(axis=1, keepdims=True)
    full_p = np.exp(s1 - m) / np.exp(s1 - m).sum(axis=1, keepdims=True)
    retained = np.take_along_axis(full_p, topk_ids.astype(np.int64), axis=1).sum(axis=1)
    assert np.allclose(s.probs(1.0).sum(axis=1), retained, atol=1e-3)


def test_uncached_temperature_is_an_error(tmp_path):
    with LogitCacheWriter(tmp_path, provenance()) as w:
        w.add("d", *fake_sequence()[:4])
    with pytest.raises(KeyError, match="not cached"):
        LogitCacheReader(tmp_path)[0].probs(3.0)


# ── shape and provenance guards ──────────────────────────────────────────

def test_k_mismatch_against_the_manifest_is_refused(tmp_path):
    token_ids, topk_ids, topk_logits, logz, _ = fake_sequence(top_k=K)
    with LogitCacheWriter(tmp_path, provenance(top_k=K + 1)) as w:
        with pytest.raises(ValueError, match="K mismatch"):
            w.add("d", token_ids, topk_ids, topk_logits, logz)


def test_unsorted_topk_is_refused():
    token_ids, topk_ids, topk_logits, logz, _ = fake_sequence()
    shuffled = topk_logits[:, ::-1].copy()
    with pytest.raises(ValueError, match="sorted descending"):
        validate_sequence(token_ids, topk_ids, shuffled, logz, len(TEMPS))


def test_non_finite_values_are_refused():
    token_ids, topk_ids, topk_logits, logz, _ = fake_sequence()
    bad = topk_logits.copy()
    bad[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        validate_sequence(token_ids, topk_ids, bad, logz, len(TEMPS))


def test_provenance_round_trips(tmp_path):
    p = CacheProvenance(
        teacher_model="m", teacher_precision="bf16",
        tokenizer_fingerprint="abc", top_k=K, vocab_size=VOCAB,
        temperatures=TEMPS,
        teacher_adapter="./ckpt", git_commit="cafe123",
    )
    with LogitCacheWriter(tmp_path, p) as w:
        w.add("d", *fake_sequence()[:4])
    got = LogitCacheReader(tmp_path).provenance
    assert got == p
    assert got.temperatures == TEMPS


def test_reading_a_cache_with_no_manifest_says_so(tmp_path):
    (tmp_path / "shard-00000.npz").write_bytes(b"not really")
    with pytest.raises(FileNotFoundError, match="manifest"):
        LogitCacheReader(tmp_path)


# ── tokenizer fingerprint ────────────────────────────────────────────────

def test_same_vocabulary_gives_the_same_fingerprint():
    v = {"a": 0, "b": 1, "c": 2}
    s = {"bos": 0, "eos": 2, "pad": None}
    assert fingerprint_vocab(v, s) == fingerprint_vocab(dict(reversed(v.items())), s)


def test_one_differing_merge_changes_the_fingerprint():
    """Equal sizes, one token different — the case a size check waves through."""
    a = fingerprint_vocab({"a": 0, "b": 1, "c": 2}, {"bos": 0})
    b = fingerprint_vocab({"a": 0, "b": 1, "d": 2}, {"bos": 0})
    assert a != b


def test_a_differing_special_token_id_changes_the_fingerprint():
    v = {"a": 0, "b": 1, "c": 2}
    assert fingerprint_vocab(v, {"eos": 2}) != fingerprint_vocab(v, {"eos": 1})

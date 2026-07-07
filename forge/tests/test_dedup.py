"""Tests for the deduplication module."""

from __future__ import annotations

import pytest

from eullm_forge.datasets.dedup import (
    DedupStats,
    _normalise_for_hash,
    _shingles,
    dedup,
    exact_dedup,
    near_dedup,
)


def test_normalise_collapses_whitespace_and_lowercases():
    assert _normalise_for_hash("  HELLO   World  \n") == "hello world"


def test_shingles_5gram_word_level():
    text = "uno due tre quattro cinque sei sette"
    shingles = _shingles(text, size=5)
    assert "uno due tre quattro cinque" in shingles
    assert "due tre quattro cinque sei" in shingles
    assert "tre quattro cinque sei sette" in shingles
    assert len(shingles) == 3


def test_shingles_short_text_falls_back_to_full_sequence():
    text = "uno due tre"  # < 5 tokens
    shingles = _shingles(text, size=5)
    assert shingles == {"uno due tre"}


def test_shingles_empty_text_returns_empty_set():
    assert _shingles("") == set()
    assert _shingles("   ") == set()


def test_exact_dedup_drops_byte_identical_chunks():
    recs = [
        {"id": "a", "text": "P.Q.M. La Corte dichiara"},
        {"id": "b", "text": "P.Q.M.  La Corte  dichiara"},  # extra spaces
        {"id": "c", "text": "P.Q.M. LA CORTE dichiara"},     # different case
        {"id": "d", "text": "Sentenza di merito diversa"},
    ]
    stats = DedupStats()
    out = list(exact_dedup(recs, stats=stats))
    # All three boilerplate variants normalise to the same string.
    assert len(out) == 2
    assert out[0]["id"] == "a"
    assert out[1]["id"] == "d"
    assert stats.seen == 4
    assert stats.dropped_exact == 2
    assert stats.kept == 2


def test_exact_dedup_skips_empty_text():
    recs = [
        {"id": "a", "text": ""},
        {"id": "b", "text": "   \n  "},
        {"id": "c", "text": "vero contenuto"},
    ]
    out = list(exact_dedup(recs))
    assert len(out) == 1
    assert out[0]["id"] == "c"


def test_near_dedup_catches_paraphrases():
    base = (
        "La Corte ritiene infondata la censura proposta dalla parte "
        "ricorrente, in quanto non si rinviene alcuna violazione di legge "
        "nei termini denunciati con il primo motivo di ricorso. La "
        "questione, come correttamente rilevato dal giudice di merito, "
        "non integra alcuna ipotesi sanzionata dall'ordinamento."
    )
    near = base.replace("infondata", "manifestamente infondata")
    different = (
        "Il ricorso è fondato. La sentenza impugnata va cassata con rinvio "
        "alla corte d'appello in diversa composizione, che si pronuncerà "
        "anche sulle spese del giudizio di legittimità."
    )
    recs = [
        {"id": "a", "text": base},
        {"id": "b", "text": near},
        {"id": "c", "text": different},
    ]
    stats = DedupStats()
    # 'b' shares ~0.77 Jaccard with 'a' on 5-word shingles. MinHash-LSH only
    # *estimates* that (num_perm=128 → ~0.09 std error), so a threshold of 0.7
    # sits inside the estimator's noise band and the catch flips with datasketch
    # internals across versions. 0.6 keeps a comfortable margin below the true
    # similarity so the paraphrase is caught deterministically, while still
    # exercising the aggressive near-dedup path. 'different' (~0 overlap) stays.
    out = list(near_dedup(recs, threshold=0.6, stats=stats))
    ids = [r["id"] for r in out]
    assert "a" in ids
    assert "c" in ids
    # 'b' is a one-word edit of 'a' — comfortably above the 0.6 threshold.
    assert "b" not in ids
    assert stats.dropped_near == 1


def test_near_dedup_keeps_genuinely_different_chunks():
    recs = [
        {"id": str(i), "text": text}
        for i, text in enumerate([
            "primo testo che parla di tributi e cartelle esattoriali",
            "secondo testo sul diritto del lavoro e licenziamenti",
            "terzo testo sul codice della strada e contravvenzioni",
        ])
    ]
    out = list(near_dedup(recs, threshold=0.85))
    assert len(out) == 3


def test_dedup_chains_exact_then_near_no_double_counting():
    base = (
        "La Corte di Cassazione, esaminati gli atti del procedimento e "
        "valutate le argomentazioni svolte dalle parti nelle rispettive "
        "memorie difensive, ritiene infondata la censura proposta dalla "
        "parte ricorrente in quanto non si rinviene alcuna violazione di "
        "legge nei termini denunciati con il primo motivo di ricorso, "
        "nei limiti della cognizione di legittimità riservata a questa "
        "Corte dall'articolo 360 del codice di procedura civile"
    )
    near = base.replace("infondata", "manifestamente infondata")
    recs = [
        {"id": "a", "text": base},
        {"id": "b", "text": base.upper()},      # exact (case-insensitive)
        {"id": "c", "text": near},              # near-dup of a
        {"id": "d", "text": "Sentenza completamente diversa: lavoro e "
                            "contributi previdenziali, riscatto contributivo"},
    ]
    stats = DedupStats()
    out = list(dedup(recs, near_threshold=0.7, stats=stats))
    assert stats.seen == 4
    assert stats.dropped_exact + stats.dropped_near + stats.kept == 4
    assert stats.dropped_exact == 1
    assert stats.dropped_near == 1
    assert stats.kept == 2


def test_dedup_skip_near_short_circuits():
    recs = [
        {"id": "a", "text": "alpha alpha alpha alpha alpha"},
        {"id": "b", "text": "alpha alpha alpha alpha beta"},  # near-dup
        {"id": "c", "text": "alpha alpha alpha alpha alpha"},  # exact-dup
    ]
    stats = DedupStats()
    out = list(dedup(recs, skip_near=True, stats=stats))
    # With skip_near=True, only exact-dup 'c' is dropped; 'b' survives.
    assert {r["id"] for r in out} == {"a", "b"}
    assert stats.dropped_exact == 1
    assert stats.dropped_near == 0


def test_dedup_stats_to_dict_has_expected_keys():
    s = DedupStats(seen=10, kept=6, dropped_exact=3, dropped_near=1)
    d = s.to_dict()
    assert d == {"seen": 10, "kept": 6, "dropped_exact": 3, "dropped_near": 1}


def test_dedup_preserves_record_metadata():
    """Output records must be the SAME dicts as the inputs (no copy/strip)."""
    rec = {
        "source_id": "snciv/2023/12345",
        "chunk_index": 2,
        "text": "Il ricorso è infondato per le ragioni esposte.",
        "metadata": {"foo": "bar"},
    }
    out = list(dedup([rec], skip_near=True))
    assert len(out) == 1
    assert out[0] is rec
    assert out[0]["metadata"] == {"foo": "bar"}

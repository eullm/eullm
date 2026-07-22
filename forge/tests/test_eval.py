"""Tests for the evaluation harness (F0). Pure-Python paths only — no torch."""

from __future__ import annotations

import math

import pytest

from eullm_forge.eval import (
    EvalItem,
    Judgement,
    MockJudge,
    aggregate,
    blind_pairwise,
    build_report,
    exact_match,
    filter_items,
    keyword_coverage,
    load_eval_set,
    load_seed,
    normalize_text,
    perplexity_from_nll,
    save_eval_set,
    score_item,
    spotcheck_markdown,
    to_markdown,
)


# --- text / QA metrics -----------------------------------------------------

def test_normalize_text_strips_accents_case_punct():
    assert normalize_text("Risoluzione, però!") == "risoluzione pero"
    assert normalize_text("  A   B  ") == "a b"


def test_exact_match_is_normalized_and_needs_reference():
    assert exact_match("Sei mesi.", "sei mesi")
    assert not exact_match("qualcosa", "")


def test_keyword_coverage():
    assert keyword_coverage("chiede la risoluzione e il risarcimento", ["risoluzione", "risarcimento"]) == 1.0
    assert keyword_coverage("solo risoluzione", ["risoluzione", "risarcimento"]) == 0.5
    assert keyword_coverage("niente", []) == 1.0  # nothing required → full


def test_score_item_and_aggregate():
    item = EvalItem(id="x", domain="legal", lang="it", question="q",
                    reference="sei mesi", keywords=["sei mesi"])
    perfect = score_item("Sei mesi", item)          # normalized == reference
    assert perfect.exact is True and perfect.keyword_coverage == 1.0
    partial = score_item("sono sei mesi, art. 327", item)  # keyword present, not exact
    assert partial.exact is False and partial.keyword_coverage == 1.0
    miss = score_item("non lo so", item)
    assert miss.exact is False and miss.keyword_coverage == 0.0
    summary = aggregate([perfect, partial, miss])
    assert summary["n"] == 3
    assert summary["exact_match"] == pytest.approx(1 / 3)
    assert summary["keyword_coverage"] == pytest.approx(2 / 3)


def test_aggregate_empty_is_nan():
    s = aggregate([])
    assert s["n"] == 0 and math.isnan(s["exact_match"])


def test_perplexity_from_nll():
    # mean NLL of 0 → perplexity 1; empty → nan
    assert perplexity_from_nll(0.0, 4) == pytest.approx(1.0)
    assert perplexity_from_nll(math.log(2) * 10, 10) == pytest.approx(2.0)
    assert math.isnan(perplexity_from_nll(1.0, 0))


# --- dataset ---------------------------------------------------------------

def test_dataset_roundtrip_and_filter(tmp_path):
    items = [
        EvalItem(id="a", domain="legal", lang="it", question="q1", category="civile"),
        EvalItem(id="b", domain="legal", lang="de", question="q2", category="gdpr"),
    ]
    path = tmp_path / "set.jsonl"
    save_eval_set(items, path)
    loaded = load_eval_set(path)
    assert [i.id for i in loaded] == ["a", "b"]
    assert len(filter_items(loaded, lang="it")) == 1
    assert len(filter_items(loaded, domain="legal")) == 2
    assert filter_items(loaded, category="gdpr")[0].id == "b"


def test_from_dict_ignores_unknown_keys():
    it = EvalItem.from_dict({"id": "z", "domain": "legal", "lang": "it",
                             "question": "q", "bogus": 123})
    assert it.id == "z" and not hasattr(it, "bogus")


def test_loader_rejects_duplicate_and_missing_id(tmp_path):
    dup = tmp_path / "dup.jsonl"
    dup.write_text('{"id":"a","domain":"legal","lang":"it","question":"q"}\n'
                   '{"id":"a","domain":"legal","lang":"it","question":"q"}\n',
                   encoding="utf-8")
    with pytest.raises(ValueError):
        load_eval_set(dup)
    noid = tmp_path / "noid.jsonl"
    noid.write_text('{"domain":"legal","lang":"it","question":"q"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_eval_set(noid)


def test_seed_set_loads():
    items = load_seed()
    assert len(items) >= 8
    assert all(it.domain == "legal" and it.lang == "it" for it in items)
    assert len({it.id for it in items}) == len(items)  # unique ids
    assert all(it.question and it.reference for it in items)


# --- judge / blind A-B -----------------------------------------------------

class _MarkerJudge:
    """Prefers whichever answer contains the marker 'WIN' (position-agnostic)."""

    def compare(self, question, answer_a, answer_b, rubric=""):
        wa, wb = "WIN" in answer_a, "WIN" in answer_b
        if wa and not wb:
            return Judgement("A")
        if wb and not wa:
            return Judgement("B")
        return Judgement("tie")


def _items(n=6):
    return [EvalItem(id=f"i{k}", domain="legal", lang="it", question=f"q{k}") for k in range(n)]


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
def test_blind_pairwise_deblinds_correctly(seed):
    items = _items()
    a = {it.id: "answer WIN" for it in items}   # A always the good one
    b = {it.id: "answer" for it in items}
    out = blind_pairwise(items, a, b, _MarkerJudge(), seed=seed)
    assert out.a_wins == len(items)
    assert out.b_wins == 0 and out.ties == 0
    assert out.win_rate_a == 1.0
    # symmetric: swap which side is good → B wins every time
    out2 = blind_pairwise(items, b, a, _MarkerJudge(), seed=seed)
    assert out2.b_wins == len(items) and out2.a_wins == 0


def test_blind_pairwise_ties_and_skips_missing():
    items = _items(3)
    a = {"i0": "x WIN", "i1": "y WIN"}          # i2 missing on both
    b = {"i0": "z WIN", "i1": "w"}
    out = blind_pairwise(items, a, b, _MarkerJudge(), seed=3)
    assert out.n == 2            # i2 skipped
    assert out.ties == 1        # i0 both have WIN
    assert out.a_wins == 1      # i1 only A has WIN


def test_mock_judge_prefers_rubric_overlap():
    j = MockJudge()
    v = j.compare("q", "contiene risoluzione e risarcimento", "vuoto",
                  rubric="cita risoluzione risarcimento danno")
    assert v.winner == "A"


# --- report ----------------------------------------------------------------

def test_build_report_and_markdown():
    items = load_seed()[:3]
    answers = {it.id: it.reference for it in items}  # perfect answers
    report = build_report(items, answers, model_name="legal-it-7b-test", perplexity=7.9)
    assert report["n_items"] == 3
    assert report["qa"]["keyword_coverage"] == pytest.approx(1.0)
    md = to_markdown(report)
    assert "legal-it-7b-test" in md and "Keyword coverage" in md
    sheet = spotcheck_markdown(items, answers)
    assert sheet.startswith("# Human spot-check sheet")
    assert items[0].id in sheet

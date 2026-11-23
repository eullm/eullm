"""Tests for the evaluation harness (F0). Pure-Python paths only — no torch."""

from __future__ import annotations

import math

import pytest

from eullm_forge.eval import (
    EvalItem,
    Judgement,
    LLMJudge,
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
    assert keyword_coverage("chiede la risoluzione e il risarcimento",
                            ["risoluzione", "risarcimento"]) == 1.0
    assert keyword_coverage("solo risoluzione", ["risoluzione", "risarcimento"]) == 0.5
    assert keyword_coverage("niente", []) == 1.0  # nothing required → full


def test_keyword_coverage_alternatives_are_one_requirement():
    # "60 giorni" and "sessanta giorni" are the same answer, so a group may
    # offer both; otherwise each spelling costs the other half a point.
    kws = ["sessanta giorni|60 giorni", "dalla notifica"]
    assert keyword_coverage("Sessanta giorni dalla notifica", kws) == 1.0
    assert keyword_coverage("60 giorni dalla notifica", kws) == 1.0
    assert keyword_coverage("Trenta giorni dalla notifica", kws) == 0.5
    assert keyword_coverage("Sessanta giorni", kws) == 0.5
    # An empty group is a requirement nothing can satisfy, never a free point,
    # and a trailing separator is ignored rather than matching everything.
    assert keyword_coverage("qualsiasi risposta", ["|", "risoluzione"]) == 0.0
    assert keyword_coverage("qualsiasi risposta", ["risoluzione|"]) == 0.0
    assert keyword_coverage("la risoluzione", ["risoluzione|"]) == 1.0


def test_a_deadline_keyword_is_matched_as_a_number_not_a_substring():
    """The keyword is a plain string, so "20 giorni" is inside "120 giorni".

    A deadline six times too long satisfied the exam's own requirement, and
    the GRPO reward -- which compares the number for exactly this reason --
    disagreed with this headline number about the same answer.
    """
    kws = ["20 giorni|venti giorni"]
    assert keyword_coverage("entro venti giorni dalla notifica", kws) == 1.0
    assert keyword_coverage("entro 20 giorni dalla notifica", kws) == 1.0
    assert keyword_coverage("Il termine è di 30 giorni dalla sentenza.", kws) == 0.0
    assert keyword_coverage("Il termine è di 120 giorni dalla sentenza.", kws) == 0.0
    assert keyword_coverage("Il termine è di centoventi giorni.", kws) == 0.0
    # the repo's own example, in its own docstring
    assert keyword_coverage("entro centosessanta giorni", ["60 giorni|sessanta giorni"]) == 0.0


def test_a_perfect_answer_scores_full_keyword_coverage_on_the_seed():
    """The gate's headline number has to be reachable. An answer identical to
    an item's own reference used to score 0.883, because two seed items listed
    alternative spellings as separate required keywords and one required a term
    its own reference never used."""
    items = load_seed()
    summary = aggregate([score_item(it.reference, it) for it in items])
    assert summary["exact_match"] == 1.0
    assert summary["keyword_coverage"] == 1.0


def test_the_administrative_deadlines_score_the_right_answer_higher():
    """legal-it v0.2 answered 30 days for both deadlines; the metric has to
    punish that, not reward it."""
    items = {it.id: it for it in load_seed()}
    tar, straord = items["legal-it-amm-001"], items["legal-it-amm-002"]
    # The same wrong answer the model gave, on both items.
    wrong_tar = "Entro trenta giorni dalla notifica si propone il ricorso al TAR."
    wrong_straord = ("Entro trenta giorni dalla notifica si propone il ricorso al "
                     "Presidente della Repubblica.")
    assert score_item(wrong_tar, tar).keyword_coverage == 0.0
    assert score_item(wrong_straord, straord).keyword_coverage < score_item(
        straord.reference, straord).keyword_coverage
    # A correct answer in either spelling is fully covered.
    assert score_item("Entro 60 giorni dalla notifica", tar).keyword_coverage == 1.0


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


def test_saving_over_a_set_leaves_the_old_one_intact_when_an_item_is_not_serialisable(
        tmp_path):
    """The destination used to be truncated before anything was serialised.

    A value json cannot represent -- a Path left in an item's metadata is the
    easy one -- aborted on that line, and the exam already on disk was left one
    item shorter with nothing in it to say so. load_eval_set read the result
    without a murmur, so a 3-item set silently became 2.
    """
    from pathlib import Path as _Path

    path = tmp_path / "exam.jsonl"
    good = [EvalItem(id=f"x{i}", domain="legal", lang="it", question=f"q{i}")
            for i in range(1, 4)]
    save_eval_set(good, path)

    bad = good[:2] + [EvalItem(id="x3", domain="legal", lang="it", question="q3",
                              metadata={"src": _Path("x")})]
    with pytest.raises(ValueError, match="x3"):
        save_eval_set(bad, path)

    # the three items are still there, and the reader is happy with them
    assert [i.id for i in load_eval_set(path)] == ["x1", "x2", "x3"]
    # ...and nothing partial is left behind
    assert not list(tmp_path.glob("*.partial"))


def test_saving_leaves_no_partial_file_and_the_writer_is_atomic(tmp_path):
    """os.replace, so a reader never sees a half-written set."""
    path = tmp_path / "set.jsonl"
    items = [EvalItem(id="a", domain="legal", lang="it", question="q")]
    save_eval_set(items, path)
    assert not list(tmp_path.glob("*.partial"))
    assert path.is_file()


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


def test_the_llm_judge_reads_every_compliant_verdict_form():
    def judge(raw):
        return LLMJudge(chat_fn=lambda prompt: raw).compare("q", "A", "B")
    assert judge("Verdict: 1\nFirst is best.").winner == "A"
    assert judge("Verdict: 2\nSecond is best.").winner == "B"
    assert judge("verdict: tie\nEven.").winner == "tie"
    assert judge("verdict=A\nFirst.").winner == "A"
    assert judge("Verdict: B wins.").winner == "B"


def test_a_non_compliant_verdict_is_a_tie_not_a_win():
    """`Verdict: both` matched `b`, `Verdict: abstain` matched `a` and
    `Verdict: 10` matched `1`, scoring three non-compliant replies as wins
    and moving the win rate. A reply that names no winner is unparseable,
    and unparseable is a tie.
    """
    def judge(raw):
        return LLMJudge(chat_fn=lambda prompt: raw).compare("q", "A", "B")
    for raw in ("Verdict: both are good, tie? no, both",
                "Verdict: abstain, cannot decide",
                "Verdict: 10\nFirst is best",
                "Verdict: tiebreaker, call it even"):
        v = judge(raw)
        assert v.winner == "tie", raw
        assert "unparseable" in v.rationale


# --- report ----------------------------------------------------------------

def test_build_report_and_markdown():
    items = load_seed()[:3]
    answers = {it.id: it.reference for it in items}  # perfect answers
    report = build_report(items, answers, model_name="legal-it-4b-test", perplexity=7.9)
    assert report["n_items"] == 3
    assert report["qa"]["keyword_coverage"] == pytest.approx(1.0)
    md = to_markdown(report)
    assert "legal-it-4b-test" in md and "Keyword coverage" in md
    sheet = spotcheck_markdown(items, answers)
    assert sheet.startswith("# Human spot-check sheet")
    assert items[0].id in sheet


def test_markdown_names_the_coverage_denominator_when_items_are_unmeasured():
    """Coverage is over the items with keywords, not over all of them.

    legal_eval.py prints "over K of N items" for the same report; the
    markdown printed "Items: 2" next to "Coverage: 100%" with no denominator,
    reading as the whole set scoring full marks when one item was not
    measured at all.
    """
    from eullm_forge.eval.report import build_report, to_markdown

    items = [
        EvalItem(id="a", domain="legal", lang="it", question="q1",
                 reference="la risoluzione", keywords=["risoluzione"]),
        EvalItem(id="b", domain="legal", lang="it", question="q2",
                 reference="tutto il testo", keywords=[]),
    ]
    md = to_markdown(build_report(items, {"a": "chiede la risoluzione", "b": "x"}))
    assert "- Keyword coverage: **100.0%** (over 1 of 2 items)" in md
    # ...while a fully measured set stays exactly as it was.
    full = to_markdown(build_report(items[:1], {"a": "chiede la risoluzione"}))
    assert "- Keyword coverage: **100.0%**\n" in full

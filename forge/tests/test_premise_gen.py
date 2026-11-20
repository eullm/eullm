"""premise_gen: questions on a false premise, answered by correcting it from the texts shown."""

from __future__ import annotations

import importlib.util
import json
import random
import re
from pathlib import Path

import pytest

from eullm_forge.datasets.premise_gen import (
    build_pairs,
    excluded_articles,
    offence_name,
    penalty_of,
    procedibility_of,
    quote_span,
    retrieved_articles,
    say,
    wrong_deadline,
)
from eullm_forge.eval import EvalItem, NormIndex
from eullm_forge.rl import corrects, score_answer, unverified_ruling

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_premise_pairs.py"
FILLER = " Il presente articolo contiene disposizioni di dettaglio sufficienti." * 3


def rec(code, text, chunk_index=0):
    return {"code": code, "article_num": "", "chunk_index": chunk_index, "text": text}


RECORDS = [
    rec("codice_civile", "Art. 1. \n \n (Fonti). \n \n Sono fonti le leggi." + FILLER),
    rec("codice_civile", "Art. 2. \n \n (Termine di prova). \n \n La domanda si propone "
        "entro sessanta giorni dalla notificazione." + FILLER),
    rec("codice_civile", "Art. 3. \n \n (Opposizione). \n \n L'opposizione si propone "
        "entro dieci giorni dalla comunicazione." + FILLER),
    rec("codice_civile", "Art. 4. \n \n (Abrogato). \n \n Articolo abrogato dalla legge "
        "n. 10 del 2000." + FILLER),
    rec("codice_civile", "Art. 5. \n \n (Contratti). \n \n Il contratto è nullo se manca "
        "la causa." + FILLER),
]


def _index():
    return NormIndex(RECORDS)


def _pairs(fmt="steps", exclude=frozenset(), seed=0, stats=None):
    return build_pairs(_index(), per_code=10, rng=random.Random(seed), fmt=fmt,
                       exclude=exclude, stats=stats)


def _passage(instruction: str, pos: int) -> str:
    return " ".join(instruction.split(f"[{pos}] ", 1)[1].split("\n[")[0].split())


def test_a_wrong_deadline_is_corrected_with_the_article_quoted_where_it_is_shown():
    pairs = [p for p in _pairs() if p.kind == "premessa_termine"]
    assert {p.number for p in pairs} == {"2", "3"}
    for p in pairs:
        right = "60 giorni" if p.number == "2" else "10 giorni"
        assert right not in p.question                       # the premise is wrong
        pos = int(re.search(r"\[(\d)\]", p.output).group(1))
        assert f"art. {p.number}" in _passage(p.instruction, pos).lower()
        quote = re.search(r"«(.+?)»", p.output).group(1)
        assert quote in _passage(p.instruction, pos)         # a literal span of the text shown
        assert p.output.startswith("Verifica della premessa:")
        assert "Norma applicabile:" in p.output and f"Risposta: il termine previsto è di {right}" \
            in p.output
        assert score_answer(p.output, p.kind, p.keywords) == 1.0
        assert "Nota:" not in p.instruction


def test_the_wrong_deadline_is_one_the_article_does_not_state():
    rng = random.Random(1)
    for _ in range(50):
        w = wrong_deadline(30, "giorni", {(30, "giorni"), (60, "giorni")}, rng)
        assert w not in (30, 60)
    assert say(1, "anni") == "1 anno" and say(3, "mesi") == "3 mesi"


def test_an_article_not_in_the_collection_is_attributed_nothing():
    pairs = [p for p in _pairs() if p.kind == "premessa_inesistente"]
    assert pairs
    taken = {"1", "2", "3", "4", "5"}
    for p in pairs:
        assert p.number not in taken
        assert "Nota:" not in p.instruction                  # as RAG Enterprise asks it
        assert f"non c'è l'art. {p.number}" in p.output
        assert score_answer(p.output, p.kind, p.keywords) == 1.0
    assert any(p.number.endswith("-bis") for p in pairs)


def test_an_article_that_repeals_others_is_not_itself_repealed():
    """A codifying decree's final article says "Sono abrogati: ..." about
    other acts and is in force; treating the word as evidence about the
    article itself teaches the inverse of the norm."""
    records = RECORDS + [rec(
        "codice_consumo",
        "Art. 146. \n \n (Abrogazioni). \n \n Dalla data di entrata in vigore del "
        "presente codice sono abrogati: a) il d.P.R. 24 maggio 1988, n. 224; "
        "b) la legge 10 aprile 1991, n. 126." + FILLER)]
    pairs = build_pairs(NormIndex(records), per_code=10, rng=random.Random(0))
    assert "146" not in {p.number for p in pairs if p.kind == "premessa_abrogato"}
    assert "4" in {p.number for p in pairs if p.kind == "premessa_abrogato"}


def test_a_repealed_article_is_said_to_be_repealed_quoting_its_text():
    [p] = [p for p in _pairs() if p.kind == "premessa_abrogato"]
    assert p.number == "4"
    pos = int(re.search(r"\[(\d)\]", p.output).group(1))
    assert re.search(r"«(.+?)»", p.output).group(1) in _passage(p.instruction, pos)
    assert score_answer(p.output, p.kind, []) == 1.0
    assert score_answer("L'art. 4 prevede che il contratto sia nullo.", p.kind, []) == 0.0


def test_the_plain_format_says_the_same_in_one_paragraph():
    for p in _pairs(fmt="plain"):
        assert "\n" not in p.output and "Verifica della premessa" not in p.output
        assert score_answer(p.output, p.kind, p.keywords) == 1.0


def test_excluded_articles_are_never_used_and_an_outside_set_is_read_from_its_questions():
    lexia = EvalItem(id="lexia-1", domain="legal", lang="it", category="x",
                     question="Ai sensi dell'art. 2 del codice civile, entro quando?",
                     reference="", metadata={})
    exam = EvalItem(id="norm-termine-codice_civile-3", domain="legal", lang="it",
                    category="codice_civile", question="Quale termine prevede l'art. 3?",
                    reference="", metadata={"code": "codice_civile", "articolo": "3"})
    exclude = excluded_articles([lexia, exam])
    assert exclude == {("codice_civile", "2"), ("codice_civile", "3")}
    assert not any(p.number in ("2", "3") for p in _pairs(exclude=exclude))


def test_a_long_sentence_is_quoted_around_the_deadline():
    sentence = " ".join(["parola"] * 80) + " entro 60 giorni " + " ".join(["altro"] * 80)
    q = quote_span(sentence, anchor="60")
    assert "60 giorni" in q and len(q) < len(sentence) and q in sentence


def test_the_reward_wants_the_premise_corrected_not_only_the_right_number():
    kw = ["60 giorni|sessanta giorni"]
    assert score_answer("Il termine è di 60 giorni.", "premessa_termine", kw) == 0.0
    assert score_answer("No: il termine è di 60 giorni, non di 30 giorni.",
                        "premessa_termine", kw) == 1.0
    assert score_answer("La premessa è errata: il termine è di 30 giorni.",
                        "premessa_termine", kw) == 0.0
    assert corrects("In realtà l'articolo prevede altro") and not corrects("Non so.")


def _load():
    spec = importlib.util.spec_from_file_location("make_premise_pairs", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_script_splits_sft_and_grpo_by_article_and_prints_counts_only(tmp_path, capsys):
    norms = tmp_path / "legislazione_test.chunks.jsonl"
    norms.write_text("".join(json.dumps(r) + "\n" for r in RECORDS), encoding="utf-8")
    sft, grpo = tmp_path / "sft.jsonl", tmp_path / "grpo.jsonl"
    assert _load().main(["--norms", str(norms), "--per-code", "10", "--out-sft", str(sft),
                         "--out-grpo", str(grpo)]) == 0
    s = [json.loads(ln) for ln in sft.read_text(encoding="utf-8").splitlines()]
    g = [json.loads(ln) for ln in grpo.read_text(encoding="utf-8").splitlines()]
    assert s and g
    assert not {(r["source"], r["key"].split("-", 3)[3]) for r in s} & \
        {(r["code"], r["articolo"]) for r in g}
    assert all(r["prompt"][0]["role"] == "user" and r["tipo"].startswith("premessa_") for r in g)
    out = capsys.readouterr().out
    assert "[premise]" in out and "Verifica" not in out and "giorni" not in out


def test_the_plain_run_writes_the_same_pairs_as_the_steps_run(tmp_path):
    norms = tmp_path / "legislazione_test.chunks.jsonl"
    norms.write_text("".join(json.dumps(r) + "\n" for r in RECORDS), encoding="utf-8")
    steps, plain = tmp_path / "steps.jsonl", tmp_path / "plain.jsonl"
    common = ["--norms", str(norms), "--per-code", "10"]
    assert _load().main(common + ["--out-sft", str(steps), "--out-grpo",
                                  str(tmp_path / "g.jsonl")]) == 0
    assert _load().main(common + ["--format", "plain", "--out-sft", str(plain)]) == 0
    a = [json.loads(ln) for ln in steps.read_text(encoding="utf-8").splitlines()]
    b = [json.loads(ln) for ln in plain.read_text(encoding="utf-8").splitlines()]
    assert [r["key"] for r in a] == [r["key"] for r in b]
    assert [r["instruction"] for r in a] == [r["instruction"] for r in b]
    assert all("Verifica della premessa" not in r["output"] for r in b)


def test_the_script_needs_an_output(tmp_path):
    with pytest.raises(SystemExit):
        _load().main(["--norms", str(tmp_path / "x.jsonl")])


def test_corrections_as_the_model_writes_them_are_recognised():
    # phrasings from the samples of 2026-10-20 the first reward scored 0
    kw = ["30 giorni|trenta giorni"]
    for answer in (
            "L'art. 2197 prevede un termine di trenta giorni. Pertanto, il termine di 20 "
            "giorni non è previsto dal codice civile.",
            "Nei testi non compare alcun riferimento a un termine di 60 giorni. L'art. 45 "
            "prevede un termine perentorio di trenta giorni.",
            "Il termine è di 30 giorni, non 60.",
            "L'articolo non prevede un termine di 60 giorni: il termine è di 30 giorni."):
        assert score_answer(answer, "premessa_termine", kw) == 1.0, answer
    # the right number with the premise left standing is still not a correction
    assert score_answer("Il termine per agire è fissato in trenta giorni.",
                        "premessa_termine", kw) == 0.0
    assert not corrects("È una facoltà, non una semplice possibilità.")


def test_a_true_premise_is_confirmed_and_shares_its_article_with_the_false_one():
    pairs = build_pairs(_index(), per_code=10, rng=random.Random(0), true_share=1.0)
    true = {p.number: p for p in pairs if p.kind == "premessa_vera"}
    false = {p.number: p for p in pairs if p.kind == "premessa_termine"}
    assert true and set(true) == set(false)
    for n, p in true.items():
        right = "60 giorni" if n == "2" else "10 giorni"
        assert right in p.question and right not in false[n].question
        assert "La premessa della domanda è corretta" in p.output
        assert score_answer(p.output, p.kind, p.keywords) == 1.0
        # the false twin's answer, a correction, is wrong here
        assert score_answer(false[n].output, p.kind, p.keywords) == 0.0
    assert not [p for p in build_pairs(_index(), per_code=10, rng=random.Random(0),
                                       true_share=0.0) if p.kind == "premessa_vera"]


def test_the_true_premise_reward_wants_the_deadline_and_no_correction():
    kw = ["60 giorni|sessanta giorni"]
    assert score_answer("Sì, il termine è di 60 giorni [1].", "premessa_vera", kw) == 1.0
    # A contrast inside a confirmation leaves the premise standing: only the
    # denial of the premise itself is a correction.
    assert score_answer("Sì: il termine è di 60 giorni, non di 30.", "premessa_vera",
                        kw) == 1.0
    assert score_answer("La premessa è corretta: non è prevista alcuna proroga, "
                        "il termine è di 60 giorni.", "premessa_vera", kw) == 1.0
    # The gold answer of a generated true pair, quoting a statute that says
    # "non è prevista alcuna proroga", must score its own reward.
    records = [rec("codice_civile", "Art. 2. \n \n (Termine di prova). \n \n La domanda "
                   "si propone entro sessanta giorni e non è prevista alcuna proroga."
                   + FILLER)]
    [p] = [p for p in build_pairs(NormIndex(records), per_code=10,
                                  rng=random.Random(0), true_share=1.0)
           if p.kind == "premessa_vera"]
    assert score_answer(p.output, p.kind, p.keywords) == 1.0
    assert score_answer("No, la premessa è errata: il termine è di 60 giorni.",
                        "premessa_vera", kw) == 0.0
    assert score_answer("Non trovo la disposizione nei testi.", "premessa_vera", kw) == 0.0


PENAL = [
    rec("codice_penale", "Art. 10. \n \n (Truffa semplice). \n \n Chiunque procura a sé un "
        "ingiusto profitto è punito con la reclusione da sei mesi a tre anni e con la multa. "
        "Il delitto è punibile a querela della persona offesa." + FILLER),
    rec("codice_penale", "Art. 11. \n \n (Rapina semplice). \n \n Chiunque si impossessa "
        "della cosa altrui con violenza è punito con la reclusione da tre a dieci anni. "
        "Si procede d'ufficio." + FILLER),
    rec("codice_penale", "Art. 12. \n \n (Disposizione generale). \n \n Le pene si applicano "
        "secondo le norme che seguono." + FILLER),
]


def _penal_pairs(fmt="steps", true_share=1.0):
    return build_pairs(NormIndex(PENAL), per_code=10, rng=random.Random(0), fmt=fmt,
                       true_share=true_share)


def test_a_wrong_maximum_penalty_is_corrected_and_a_right_one_confirmed():
    pairs = _penal_pairs()
    false = {p.number: p for p in pairs if p.kind == "premessa_pena"}
    true = {p.number: p for p in pairs if p.kind == "premessa_vera" and p.key.startswith("pf-pv")}
    assert set(false) == {"10", "11"} and set(true) == {"10", "11"}
    assert "3 anni" in false["10"].output and "6 mesi" in false["10"].output
    assert "3 anni" in true["10"].question and "3 anni" not in false["10"].question
    assert "da 3 anni a 10 anni" in false["11"].output        # the low bound takes the unit
    for n in false:
        assert score_answer(false[n].output, "premessa_pena", false[n].keywords) == 1.0
        assert score_answer(true[n].output, "premessa_vera", true[n].keywords) == 1.0
        assert score_answer(true[n].output, "premessa_pena", false[n].keywords) == 0.0
        assert score_answer(false[n].output, "premessa_vera", true[n].keywords) == 0.0
        pos = int(re.search(r"\[(\d)\]", false[n].output).group(1))
        assert re.search(r"«(.+?)»", false[n].output).group(1) in _passage(false[n].instruction,
                                                                            pos)


def test_the_first_comma_penalty_is_asked_when_the_article_states_several():
    based = ("Art. 14. \n \n (Furto). \n \n Chiunque si impossessa della cosa altrui è "
             "punito con la reclusione da sei mesi a tre anni. \n La pena è della reclusione "
             "da due a sei anni se il fatto è commesso con violenza.")
    assert penalty_of(based)[1:5] == (6, "mesi", 3, "anni")
    assert penalty_of(based)[-1] == ", al primo comma,"
    assert penalty_of(PENAL[0]["text"])[-1] == ""
    # The first sentence states two ranges, or none: no range is the base one.
    assert penalty_of("Art. 15. (Doppia). Chiunque è punito con la reclusione da uno a tre "
                      "anni o con l'arresto da tre a sei mesi. La pena è della reclusione da "
                      "due a cinque anni se il fatto è grave.") is None
    assert penalty_of("Art. 16. (Rinvio). Le pene sono aumentate nei casi seguenti. Si "
                      "applica la reclusione da uno a tre anni. Si applica la reclusione da "
                      "due a sei anni se il fatto è grave.") is None
    pairs = build_pairs(NormIndex([rec("codice_penale", based + FILLER)]), per_code=5,
                        rng=random.Random(0), true_share=1.0)
    false = [p for p in pairs if p.kind == "premessa_pena"]
    true = [p for p in pairs if p.key.startswith("pf-pv")]
    assert false and true
    for p in false + true:
        assert "al primo comma" in p.question and "al primo comma" in p.output
        assert score_answer(p.output, p.kind, p.keywords) == 1.0


def test_the_procedibility_is_read_from_the_article_and_never_from_one_stating_both():
    pairs = _penal_pairs()
    false = {p.number: p for p in pairs if p.kind == "premessa_procedibilita"}
    true = {p.number: p for p in pairs if p.key.startswith("pf-qv")}
    assert false["10"].keywords == ["querela"] and false["11"].keywords == ["d'ufficio"]
    assert "procedibile a querela" in true["10"].question
    assert "procedibile a querela" not in false["10"].question
    for n in false:
        assert score_answer(false[n].output, "premessa_procedibilita", false[n].keywords) == 1.0
        assert score_answer(true[n].output, "premessa_vera", true[n].keywords) == 1.0
        assert score_answer(false[n].output, "premessa_vera", true[n].keywords) == 0.0
    both = rec("codice_penale", "Art. 13. \n \n (Doppia). \n \n Si procede a querela della "
               "persona offesa. Si procede d'ufficio se il fatto è grave." + FILLER)
    assert not [p for p in build_pairs(NormIndex([both]), per_code=5, rng=random.Random(0))
                if p.kind == "premessa_procedibilita"]


def test_an_article_stating_both_with_words_in_between_is_left_out():
    # Art. 624 c.p.: taken for querela alone by the first version.
    assert procedibility_of("Il delitto è punibile a querela della persona offesa. Si "
                            "procede, tuttavia, d'ufficio se ricorre taluna delle "
                            "circostanze.") is None
    assert procedibility_of("Il delitto è punibile a querela della persona offesa. Il "
                            "giudice provvede d'ufficio.") is None
    assert procedibility_of("Si procede, in ogni caso, d'ufficio.")[0] == "d'ufficio"
    assert procedibility_of("Il colpevole è punibile, nei casi indicati, a querela della "
                            "persona offesa.")[0] == "querela"


def test_the_procedibility_reward_takes_the_ways_a_model_corrects_it():
    querela, ufficio = ["querela"], ["d'ufficio"]
    # The 8B's answers of 2026-10-22, scored 0 by the first version.
    for a in ["Pertanto, il reato non è procedibile d'ufficio, ma richiede la querela della "
              "persona offesa [1].",
              "Pertanto, la querela è necessaria [1].",
              "Si procede solo a querela della persona offesa [2].",
              "No, serve la querela della persona offesa."]:
        assert score_answer(a, "premessa_procedibilita", querela) == 1.0, a
    for a in ["Non è necessaria la querela: si procede d'ufficio [1].",
              "Il reato non è procedibile a querela: si procede d'ufficio.",
              "Si procede d'ufficio, senza bisogno di querela."]:
        assert score_answer(a, "premessa_procedibilita", ufficio) == 1.0, a
    # Agreeing with the claim, or the right word without denying the claim.
    assert score_answer("Sì, non serve la querela: si procede d'ufficio.",
                        "premessa_procedibilita", querela) == 0.0
    assert score_answer("Sì, il reato è procedibile a querela.", "premessa_procedibilita",
                        ufficio) == 0.0
    assert score_answer("Il reato è previsto dall'art. 10 e riguarda la querela.",
                        "premessa_procedibilita", querela) == 0.0


def test_the_offence_is_named_by_the_heading_and_only_when_the_heading_names_one():
    assert offence_name("Art. 624. \n (Furto). \n Chiunque si impossessa") == "furto"
    assert offence_name("Art. 624-bis. \n ((Furto in abitazione e furto con strappo)). \n "
                        "Chiunque") == "furto in abitazione e furto con strappo"
    assert offence_name("Art. 61. \n (Circostanze aggravanti comuni). \n Aggravano") is None
    # A repealed article keeps no heading: "abuso d'ufficio" retrieves nothing.
    assert offence_name("Art. 323. \n ((ARTICOLO ABROGATO DALLA L. 9 AGOSTO 2024, "
                        "N. 114))") is None


def test_penal_questions_can_name_the_offence_instead_of_the_article():
    pairs = build_pairs(NormIndex(PENAL), per_code=10, rng=random.Random(0), true_share=1.0,
                        named_share=1.0)
    penal = [p for p in pairs if p.kind in ("premessa_pena", "premessa_procedibilita")
             or p.key.startswith(("pf-pv", "pf-qv"))]
    assert penal
    for p in penal:
        assert "art." not in p.question.lower(), p.question
        assert re.search(r"reato di (truffa|rapina) semplice", p.question), p.question
        # The answer still says which article it is.
        assert f"art. {p.number}" in p.output
        assert score_answer(p.output, p.kind, p.keywords) == 1.0
    numbered = build_pairs(NormIndex(PENAL), per_code=10, rng=random.Random(0),
                           true_share=1.0, named_share=0.0)
    assert all("art. 1" in p.question for p in numbered
               if p.kind in ("premessa_pena", "premessa_procedibilita"))
    # Two articles under one heading: the name would not say which.
    twin = rec("codice_penale", PENAL[0]["text"].replace("Art. 10.", "Art. 13."))
    pairs = build_pairs(NormIndex([*PENAL, twin]), per_code=10, rng=random.Random(0),
                        named_share=1.0)
    assert all("truffa" not in p.question for p in pairs)


def test_an_outside_set_leaves_out_what_its_questions_retrieve_and_its_source():
    items = [{"id": "x", "question": "Quale pena rischia chi commette una rapina semplice?",
              "metadata": {}},
             {"id": "y", "question": "Che cosa prevede la truffa semplice?",
              "metadata": {"code": "codice_civile", "articolo": "2"}}]
    got = retrieved_articles(NormIndex(PENAL), items, k=1)
    assert got == {("codice_penale", "11")}
    assert ("codice_penale", "585") in excluded_articles(
        [{"question": "Quanto rischia chi ferisce?", "metadata": {"source": "art. 585 c.p."}}])


def test_the_plain_format_holds_for_the_penal_kinds_too():
    for p in _penal_pairs(fmt="plain"):
        if p.kind in ("premessa_pena", "premessa_procedibilita", "premessa_vera"):
            assert "\n" not in p.output
            assert score_answer(p.output, p.kind, p.keywords) == 1.0


def test_a_deadline_attributed_to_a_ruling_is_checked_on_the_statute_and_the_ruling_flagged():
    records = [rec("codice_civile", "Art. 2. \n \n (Termine di prova). \n \n La domanda si "
                   "propone entro sessanta giorni dalla notifica." + FILLER)]
    for fmt in ("steps", "plain"):
        pairs = build_pairs(NormIndex(records), per_code=10, rng=random.Random(0), fmt=fmt)
        false = [p for p in pairs if p.kind == "premessa_sentenza"]
        true = [p for p in pairs if p.kind == "premessa_sentenza_vera"]
        assert false and true
        for p in false + true:
            assert re.search(r"sentenza n\. \d+/20\d\d", p.question)
            assert score_answer(p.output, p.kind, p.keywords) == 1.0
        assert score_answer(true[0].output, "premessa_sentenza", true[0].keywords) == 0.0
        assert score_answer(false[0].output, "premessa_sentenza_vera", false[0].keywords) == 0.0
    kw = ["60 giorni|sessanta giorni"]
    # Right deadline, ruling taken on trust: the attribution is the trap.
    assert score_answer("Sì, come ha stabilito la Cassazione il termine è di 60 giorni [1].",
                        "premessa_sentenza_vera", kw) == 0.0
    # "va verificata sulla fonte" says the ruling should be checked, not that
    # it cannot be: the gold answer already says "non c'è ... non posso
    # verificare", so the hedge alone must not satisfy the check.
    trust = ("Sì, come ha stabilito la Cassazione il termine è di 60 giorni [1]. "
             "La sentenza citata va verificata sulla fonte.")
    assert not unverified_ruling(trust)
    assert score_answer(trust, "premessa_sentenza_vera", kw) == 0.0
    assert score_answer("La sentenza citata non è tra i testi riportati, non posso verificarla; "
                        "l'art. 2 prevede comunque proprio 60 giorni [1].",
                        "premessa_sentenza_vera", kw) == 1.0
    assert score_answer("Non posso verificare la sentenza. La premessa è errata: il termine è "
                        "di 60 giorni, non di 30 [1].", "premessa_sentenza", kw) == 1.0
    # Flagging the ruling is not enough: the deadline still has to be right.
    assert score_answer("Non posso verificare la sentenza citata.", "premessa_sentenza",
                        kw) == 0.0


def test_a_ruling_flagged_the_ways_the_model_writes_it_is_recognised():
    # The 8B's answers of 2026-10-23, scored 0 by the first version.
    kw = ["20 giorni|venti giorni"]
    assert score_answer("Verifica della premessa: il testo [1] è l'art. 27 del codice di "
                        "procedura penale, e prevede un termine di 20 giorni. La sentenza n. "
                        "37735/2018 della Cassazione (pen.) non è tra i testi riportati, quindi "
                        "non posso confermarne il contenuto.\nRisposta: il termine previsto è "
                        "di 20 giorni.", "premessa_sentenza_vera", kw) == 1.0
    assert score_answer("La sentenza Cass. civ. n. 4767/2025 non è tra i testi riportati. "
                        "L'art. 27 prevede comunque 20 giorni [1].",
                        "premessa_sentenza_vera", kw) == 1.0
    # Correcting the deadline but saying nothing of the ruling still misses the trap.
    assert score_answer("Il testo [1] prevede un termine di 20 giorni, non di 8 giorni. La "
                        "premessa della domanda è errata.", "premessa_sentenza", kw) == 0.0


def test_sentences_are_never_cut_after_an_abbreviation():
    from eullm_forge.eval.norm_exam import split_sentences
    text = ("Si reputa che il valore dei beni ecceda la cautela se supera di un terzo "
            "l'importo dei crediti iscritti, accresciuto degli accessori a norma dell'art. "
            "2855. Il giudice, sentite le parti, provvede ai sensi del d.lgs. n. 150 del "
            "2011 e della Cass. Sez. Un. n. 4767/2025; si applica l'art. 12 c.c. "
            "Restano ferme le disposizioni speciali.")
    got = [s for _, s in split_sentences(text)]
    assert got[0].endswith("a norma dell'art. 2855.")
    assert got[1].startswith("Il giudice") and got[1].endswith("4767/2025;")
    # after "c.c." it does not end even where it could: a quotation too long is
    # cut by quote_span, one cut short loses the article's number
    assert got[2] == "si applica l'art. 12 c.c. Restano ferme le disposizioni speciali."
    # offsets point at each sentence
    assert all(text[o:o + len(s)] == s for o, s in split_sentences(text))
    # ":" ends a sentence only when asked
    assert len(split_sentences("Sono elementi: il prezzo.", marks=".;:")) == 2
    assert len(split_sentences("Sono elementi: il prezzo.")) == 1


def test_an_answer_quoting_a_text_cut_short_earns_nothing():
    from eullm_forge.rl import answer_reward
    from eullm_forge.rl.rewards import quote_cut_short
    kw = ["60 giorni|sessanta giorni"]
    whole = ("La premessa è errata: il termine è di 60 giorni, non di 30 [1]. Il testo dice: "
             "«La domanda si propone entro sessanta giorni a norma dell'art. 2855».")
    cut = whole.replace("dell'art. 2855»", "dell'art»")
    assert score_answer(whole, "premessa_termine", kw) == 1.0
    assert score_answer(cut, "premessa_termine", kw) == 0.0
    assert quote_cut_short("«…abrogato dal D.LGS»") and quote_cut_short("«…ai sensi del n.»")
    assert not quote_cut_short("«Chiunque cagiona ad altri un danno ingiusto è obbligato.»")
    assert not quote_cut_short("Secondo l'art. 2043 c.c. il danno va risarcito.")
    # a sentence may end on a code's abbreviation
    assert not quote_cut_short("«Il danno si risarcisce ai sensi dell'art. 2043 c.c.»")
    scores = answer_reward([cut, whole], ["pratica", "pratica"], [[], []])
    assert scores == [-1.0, None]


def test_an_offence_with_a_wrong_number_is_corrected_pointed_and_answered():
    # 2026-10-27: the 8B corrected "640-septies" and stopped, with 640-ter in its texts
    from eullm_forge.datasets.premise_gen import unused_number
    assert unused_number("640", {"640", "640-bis", "640-ter", "640-quater",
                                 "640-quinquies"}) == "640-sexies"
    assert unused_number("640-ter", {"640", "640-bis", "640-ter"}) == "640-quater"
    assert unused_number("99", {"99"}) == "99-bis"
    pairs = [p for p in _penal_pairs() if p.kind == "premessa_numero"]
    assert {p.number for p in pairs} == {"10", "11"}       # not the general provision
    p = next(p for p in pairs if p.number == "10")
    assert "10-bis" in p.question and "truffa semplice" in p.question
    assert "art. 10 del codice penale" in p.output and "3 anni" in p.output
    assert p.keywords[0] == "10"
    assert score_answer(p.output, "premessa_numero", p.keywords) == 1.0
    # correcting and stopping, as before, earns nothing
    stopped = ("Verifica della premessa: tra i testi riportati non c'è l'art. 10-bis del codice "
               "penale, e non mi risulta che esista. Se mi indichi l'argomento, posso cercare.")
    assert score_answer(stopped, "premessa_numero", p.keywords) == 0.0
    # nor does answering on the wrong number, or hedging it
    assert score_answer("L'art. 10-bis c.p. prevede la reclusione da sei mesi a tre anni.",
                        "premessa_numero", p.keywords) == 0.0
    assert score_answer("L'art. 10-bis potrebbe non esistere; l'art. 10 prevede da 6 mesi a "
                        "3 anni.", "premessa_numero", p.keywords) == 0.0
    plain = next(q for q in _penal_pairs("plain") if q.kind == "premessa_numero"
                 and q.number == "10")
    assert score_answer(plain.output, "premessa_numero", plain.keywords) == 1.0

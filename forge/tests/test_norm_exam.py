"""The held-out exam drawn from the text of the law.

Built on made-up articles in the three formats the real files hold (see
test_eval_retrieval.py): an empty article_num with the header in the text,
"Art. N." with the XML parser's doubled prefix, and a chunk holding several
articles. The real draw is never in the repository.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from eullm_forge.eval import EvalItem, NormIndex, keyword_coverage
from eullm_forge.eval.norm_exam import (
    CONTENT_RUBRIC,
    _deadline_keyword,
    all_deadlines,
    articles_from_records,
    build_exam,
    retrieval_hits,
    rubric_v2,
    trained_articles,
)
from eullm_forge.eval.retrieval import named_code

FILLER = " Il presente articolo contiene disposizioni di dettaglio sufficienti." * 3


def rec(code, text, article_num="", chunk_index=0):
    return {"code": code, "article_num": article_num, "chunk_index": chunk_index,
            "text": text}


RECORDS = [
    rec("codice_civile", "DISPOSIZIONI GENERALI \n \n Art. 1. \n \n (Fonti). \n \n Sono fonti "
        "le leggi." + FILLER),
    rec("codice_civile", "Art. 2. \n \n (Termine di prova). \n \n La domanda si propone "
        "entro sessanta giorni dalla notificazione." + FILLER),
    rec("codice_civile", "Art. 3. \n \n (Due termini). \n \n Entro dieci giorni si "
        "comunica, entro trenta giorni si decide." + FILLER),
    rec("codice_civile", "Art. 4. \n \n (Abrogato). \n \n Articolo abrogato." + FILLER),
    rec("codice_civile", "Art. 5. \n \n (Lungo). \n \n Prima parte." + FILLER),
    rec("codice_civile", "seconda parte dell'articolo cinque, nel termine di 90 giorni.",
        chunk_index=1),
    rec("legge_procedimento_amministrativo",
        "Art. Art. 2. ((Conclusione del procedimento))\nIl procedimento si conclude "
        "entro trenta giorni." + FILLER, article_num="Art. 2."),
    rec("codice_processo_amministrativo",
        "Art. 29 \n Azione di annullamento \n 1. L'azione si propone nel termine di "
        "decadenza di sessanta giorni." + FILLER + "\n Art. 30 \n Azione di condanna \n "
        "1. Si propone entro centoventi giorni." + FILLER),
    rec("codice_processo_amministrativo",
        "ALLEGATO 2 \n Art. 29 \n Altra norma con lo stesso numero." + FILLER),
]


def test_articles_are_reassembled_from_every_format():
    arts = articles_from_records(RECORDS)
    assert ("codice_civile", "1") in arts and ("codice_civile", "2") in arts
    assert "seconda parte" in arts[("codice_civile", "5")].text
    assert ("legge_procedimento_amministrativo", "2") in arts
    assert ("codice_processo_amministrativo", "30") in arts


def test_a_number_used_twice_in_one_code_is_dropped_as_ambiguous():
    arts = articles_from_records(RECORDS)
    assert ("codice_processo_amministrativo", "29") not in arts


# The real c.p.a. file, as copied from Leonardo into test_eval_retrieval.py: the
# first chunk is the table of contents, one line per article, and the articles
# themselves follow.
CPA_INDEX = rec("codice_processo_amministrativo",
                "INDICE GENERALE \n Art. 27 - Contraddittorio \n Art. 28 - Intervento \n "
                "Art. 29 - Azione di annullamento \n Art. 30 - Azione di condanna \n "
                "Art. 31 - Silenzio")
CPA_ART29 = rec("codice_processo_amministrativo",
                "Art. 29 \n Azione di annullamento \n 1. L'azione di annullamento per "
                "violazione di legge si propone nel termine di decadenza di sessanta "
                "giorni. 2. Il ricorso e' proponibile anche in via amministrativa." + FILLER,
                chunk_index=1)


def test_the_index_of_a_code_is_not_its_articles():
    """Reading the index at face value leaves a stub under every number it
    lists, and the real article then looks like a second occurrence of a number
    that is already there -- so it is dropped as ambiguous, and the flagship
    administrative code contributes nothing to the exam without saying so."""
    arts = articles_from_records([CPA_INDEX, CPA_ART29])
    assert [k[1] for k in arts] == ["29"]
    assert "sessanta giorni" in arts[("codice_processo_amministrativo", "29")].text
    exam = build_exam([CPA_INDEX, CPA_ART29], per_code=2, seed=1)
    assert [it for it in exam if it.metadata["code"] == "codice_processo_amministrativo"]


def test_a_rubrica_called_indice_is_still_an_article():
    one = rec("codice_civile",
              "Art. 5. \n \n (Indice delle materie). \n \n Le materie sono elencate "
              "nell'atto seguente, che qui riportiamo per esteso." + FILLER)
    assert ("codice_civile", "5") in articles_from_records([one])


def test_an_index_longer_than_one_chunk_is_dropped_whole():
    """The real index is longer than a chunk and only its first chunk says
    INDICE; the next one, and one where the index ends and the articles
    begin, must lose their index lines too."""
    first = rec("codice_processo_amministrativo",
                "INDICE GENERALE \n Art. 1 - Effettivita' \n Art. 2 - Giusto processo \n "
                "Art. 3 - Dovere di motivazione")
    second = rec("codice_processo_amministrativo",
                 "Art. 4 - Rinvio esterno \n Art. 5 - Ricorso \n Art. 6 - Consiglio di Stato",
                 chunk_index=1)
    mixed = rec("codice_processo_amministrativo",
                "Art. 7 - Giurisdizione \n Art. 8 - Cognizione incidentale \n "
                "Art. 1 \n Effettivita' \n 1. La giurisdizione amministrativa assicura una "
                "tutela piena ed effettiva secondo i principi della Costituzione." + FILLER,
                chunk_index=2)
    art5 = rec("codice_processo_amministrativo",
               "Art. 5 \n Ricorso \n 1. Il ricorso si propone entro sessanta giorni dalla "
               "notificazione dell'atto." + FILLER, chunk_index=3)
    tail = rec("codice_processo_amministrativo",
               "Art. 9 - Competenza \n Art. 10 - Rilievo dell'incompetenza", chunk_index=4)
    art9 = rec("codice_processo_amministrativo",
               "Art. 9 \n Competenza \n 1. Il difetto di competenza e' rilevato d'ufficio "
               "entro trenta giorni." + FILLER, chunk_index=5)
    arts = articles_from_records([first, second, mixed, art5, tail, art9])
    assert sorted(k[1] for k in arts) == ["1", "5", "9"]
    assert "Rinvio esterno" not in arts[("codice_processo_amministrativo", "1")].text
    assert "sessanta giorni" in arts[("codice_processo_amministrativo", "5")].text


def test_the_heading_is_read_in_both_styles():
    arts = articles_from_records(RECORDS)
    assert arts[("codice_civile", "2")].heading == "Termine di prova"
    assert arts[("legge_procedimento_amministrativo", "2")].heading == \
        "Conclusione del procedimento"


@pytest.fixture
def exam():
    return build_exam(RECORDS, per_code=10, seed=1)


def of_kind(exam, kind):
    return [it for it in exam if it.metadata["tipo"] == kind]


def test_only_articles_with_exactly_one_deadline_become_deadline_questions(exam):
    arts = {(it.metadata["code"], it.metadata["articolo"]) for it in of_kind(exam, "termine")}
    assert ("codice_civile", "2") in arts            # sixty days, once
    assert ("codice_civile", "3") not in arts        # two deadlines: ambiguous
    assert ("codice_civile", "5") in arts            # found in the continuation chunk


def test_a_deadline_question_is_scored_on_the_deadline_in_digits_or_words(exam):
    # `i`, not `it`: naming the variable being assigned in the generator
    # expression made it a free variable of that expression, and the test died
    # with a NameError on every run instead of asserting anything.
    it = next(i for i in of_kind(exam, "termine") if i.metadata["articolo"] == "2"
              and i.metadata["code"] == "codice_civile")
    assert "sessanta giorni" in it.reference
    assert keyword_coverage("Entro 60 giorni.", it.keywords) == 1.0
    assert keyword_coverage("Entro sessanta giorni.", it.keywords) == 1.0
    assert keyword_coverage("Entro trenta giorni.", it.keywords) == 0.0


# A deadline of one unit used to be unwinnable: the unit arrives pluralised, so
# the keyword was "1 anni" and neither "un anno" nor "1 anno" contains it. The
# correct answer scored zero and an invented deadline scored the same.
@pytest.mark.parametrize("n,unit,keyword", [
    (1, "anni", "1 anno|un anno"),
    (1, "giorni", "1 giorno|un giorno"),
    (1, "ore", "1 ora|un ora"),
    (1, "mesi", "1 mese|un mese"),
    (2, "anni", "2 anni|due anni"),
])
def test_a_one_unit_deadline_is_named_the_way_it_is_written(n, unit, keyword):
    assert _deadline_keyword(n, unit) == keyword


def test_the_exam_the_reward_and_the_score_read_deadlines_from_one_copy():
    # The tables were copied into three modules; a number word added to one
    # (an exam that can say "quarantotto ore") would have been scored by the
    # others as no deadline at all.
    from eullm_forge.eval import metrics, norm_exam, paired  # noqa: F401
    from eullm_forge.rl import rewards

    assert norm_exam.NUMBER_WORDS is metrics.NUMBER_WORDS
    assert norm_exam.DEADLINE_UNITS is metrics.DEADLINE_UNITS
    assert norm_exam.ANY_DEADLINE is metrics.ANY_DEADLINE is rewards.ANY_DEADLINE
    assert rewards.number_of is metrics.number_of
    assert rewards.mentioned_deadlines("entro sessanta giorni o 2 anni") == {
        (60, "giorni"), (2, "anni")}


@pytest.mark.parametrize("answer", ["entro un anno", "entro 1 anno"])
def test_a_one_unit_deadline_answer_is_scored(answer):
    assert keyword_coverage(answer, [_deadline_keyword(1, "anni")]) == 1.0


def test_the_elided_feminine_is_scored_too():
    # "un'ora" normalises to "un ora", which is what the keyword looks for.
    assert keyword_coverage("entro un'ora dalla segnalazione",
                            [_deadline_keyword(1, "ore")]) == 1.0


def test_a_one_unit_deadline_item_is_winnable_end_to_end():
    art = rec("codice_consumo", "Art. 7. \n \n (Diritto di recesso). \n \n Il consumatore "
              "puo' esercitare il diritto di recesso entro un anno dalla data di "
              "conclusione del contratto." + FILLER)
    it = next(i for i in build_exam([art], per_code=2, seed=4)
              if i.metadata["tipo"] == "termine")
    assert keyword_coverage("Il diritto di recesso si esercita entro un anno dalla "
                            "conclusione del contratto.", it.keywords) == 1.0
    assert keyword_coverage("Il diritto di recesso si esercita entro trenta giorni.",
                            it.keywords) == 0.0


RATE_RECORD = rec("codice_civile",
                  "Art. 1224. \n \n (Interessi legali) \n \n Gli interessi legali sono "
                  "calcolati al tasso del 6 per cento, salvo quanto disposto per le "
                  "obbligazioni in valuta estera. L'azione giudiziale si esercita entro "
                  "sei mesi dalla maturazione della domanda." + FILLER,
                  article_num="1224")


def test_the_deadline_reference_is_the_sentence_that_states_the_deadline():
    """The number can turn up earlier in the article for another reason — here
    an interest rate — and looking the sentence up by the bare number hands the
    grader that one instead, under a rubric that still asks for the deadline."""
    items = build_exam([RATE_RECORD], per_code=4, seed=1)
    it = next(i for i in items if i.metadata["tipo"] == "termine")
    key = it.reference.split("\n\nTesto integrale")[0]
    assert "sei mesi" in key
    assert "per cento" not in key
    assert keyword_coverage("Entro sei mesi dalla domanda.", it.keywords) == 1.0


def test_a_cross_reference_is_not_the_deadline_answer_key():
    cross = rec("codice_procedura_civile",
                "Art. 750. \n \n (Interpretazione) \n \n Quando la legge rinvia ad altre "
                "disposizioni si applicano le regole dell'articolo 30 del codice "
                "penale. L'istanza si propone entro trenta giorni dalla notifica."
                + FILLER,
                article_num="750")
    items = build_exam([cross], per_code=4, seed=2)
    it = next(i for i in items if i.metadata["tipo"] == "termine")
    key = it.reference.split("\n\nTesto integrale")[0]
    assert "trenta giorni" in key
    assert "articolo 30" not in key


def test_repealed_articles_are_left_out(exam):
    assert not [it for it in exam if it.metadata["articolo"] == "4"
                and it.metadata["code"] == "codice_civile"]


def test_a_nonexistent_article_is_past_the_end_and_must_be_refused(exam):
    fake = [it for it in of_kind(exam, "inesistente") if it.metadata["code"] == "codice_civile"]
    assert fake and all(int(it.metadata["articolo"]) > 5 for it in fake)
    assert keyword_coverage("L'articolo non esiste.", fake[0].keywords) == 1.0
    assert keyword_coverage("Prevede il risarcimento del danno.", fake[0].keywords) == 0.0


def test_the_nonexistent_article_reference_does_not_claim_where_the_code_ends(exam):
    """The builder knows where the corpus it was handed ends, not where the
    code ends: a corpus that stopped at art. 120 produced "la numerazione
    arriva all'art. 120" for an article that exists, and the rubric then
    marked the model describing it as wrong."""
    for it in of_kind(exam, "inesistente"):
        assert "arriva all'art" not in it.reference, it.reference
        # The assumption is kept where it can be seen instead of asserted.
        assert it.metadata["last_article"]


def test_only_saying_the_article_is_absent_scores_on_it(exam):
    """`non contiene` and `non prevede un` were satisfied by an answer that
    invents the article's content and then hedges — which this rubric calls
    wrong — so they scored 1.0 on a made-up article."""
    it = of_kind(exam, "inesistente")[0]
    invented = ("L'articolo tratta l'arricchimento; il legislatore non prevede un "
                "rimedio dedicato.")
    assert keyword_coverage(invented, it.keywords) == 0.0
    assert keyword_coverage("L'articolo non esiste.", it.keywords) == 1.0
    assert keyword_coverage("Non è previsto nel codice.", it.keywords) == 1.0


def test_every_question_names_its_code_so_retrieval_and_readers_can_tell(exam):
    for it in exam:
        assert named_code(it.question) == it.metadata["code"], it.question


def test_the_verticals_are_tagged(exam):
    v = {it.metadata["code"]: it.metadata["vertical"] for it in exam}
    assert v["codice_civile"] == "civile_penale"
    assert v["codice_processo_amministrativo"] == "amministrativo"


def test_the_draw_is_random_unless_seeded():
    a = [it.id for it in build_exam(RECORDS, per_code=1, seed=3)]
    assert a == [it.id for it in build_exam(RECORDS, per_code=1, seed=3)]


def test_retrieval_hits_find_named_articles(exam):
    hits = retrieval_hits(exam, NormIndex(RECORDS))
    assert hits["contenuto"]["top1"] == 1.0
    assert "inesistente" not in hits


def test_retrieval_hits_tally_an_item_with_no_tipo():
    """check_retrieval.py takes eval files as arguments, and the project's own
    seed set carries no `tipo` in any item — the tally was keyed on None and
    died sorting it against a string, before printing a row."""
    untyped = EvalItem(id="u", domain="legal", lang="it", question="art. 1?",
                       reference="r", keywords=[], metadata={})
    typed = EvalItem(id="t", domain="legal", lang="it", question="art. 2?",
                     reference="r", keywords=[],
                     metadata={"tipo": "contenuto", "code": "codice_civile",
                               "articolo": "2"})
    hits = retrieval_hits([untyped, typed], NormIndex(RECORDS))
    assert set(hits) == {"?", "contenuto"}
    assert hits["?"]["n"] == 1


def test_check_retrieval_runs_on_an_eval_file_without_tipos(tmp_path, capsys):
    """End to end, on the file that broke it: the seed set."""
    import json

    spec = importlib.util.spec_from_file_location(
        "check_retrieval", Path(__file__).resolve().parents[1] / "scripts" / "check_retrieval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    norms = tmp_path / "legislazione_x.chunks.jsonl"
    norms.write_text(
        json.dumps({"code": "codice_civile", "article_num": "2043",
                    "text": "Art. 2043. Risarcimento per fatto illecito. " * 4},
                   ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    seed = (Path(__file__).resolve().parents[1] / "eullm_forge" / "eval" / "data"
            / "legal_it_heldout.seed.jsonl")

    assert mod.main([str(seed), "--norms", str(norms)]) == 0
    assert "?" in capsys.readouterr().out


def test_no_answer_can_still_score_on_the_exam(exam):
    """`contenuto` items have no keywords — the reference is the whole
    article, so only the judge can score them. Left in the mean they are a
    free point each, which puts a floor under the number the gate reports:
    an empty answer, a shrug and a wrong essay used to score the same."""
    from eullm_forge.eval import evaluate_qa

    blank = evaluate_qa(exam, {it.id: "" for it in exam})
    shrug = evaluate_qa(exam, {it.id: "Non lo so, non mi risulta." for it in exam})
    essay = evaluate_qa(exam, {it.id: "La materia e regolata dalla legge." for it in exam})
    for report in (blank, shrug, essay):
        assert report["summary"]["keyword_coverage"] == 0.0
        assert report["summary"]["keyword_items"] == sum(1 for it in exam if it.keywords)
    # The judge-graded family is reported as not measured, not as covered.
    unmeasured = [r for r in blank["per_item"] if r["keyword_coverage"] is None]
    assert len(unmeasured) == sum(1 for it in exam if not it.keywords) > 0
    assert not [r for r in unmeasured if r["keyword_coverage"] == 1.0]


def test_the_coverage_mean_ignores_only_the_keyword_less_items(exam):
    from eullm_forge.eval import evaluate_qa

    it = next(i for i in of_kind(exam, "termine"))
    answers = {i.id: "" for i in exam}
    answers[it.id] = "Entro 60 giorni dalla notifica."      # the one right answer
    report = evaluate_qa(exam, answers)
    assert report["summary"]["keyword_coverage"] > 0.0
    assert report["summary"]["keyword_items"] < report["summary"]["n"]


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_norm_exam.py"


def test_the_script_prints_counts_and_refuses_to_redraw(tmp_path, capsys):
    import json

    spec = importlib.util.spec_from_file_location("make_norm_exam", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    norms = tmp_path / "legislazione_x.chunks.jsonl"
    norms.write_text("\n".join(json.dumps(r) for r in RECORDS) + "\n")
    out = tmp_path / "exam.jsonl"
    assert mod.main([str(norms), "--out", str(out), "--check-retrieval"]) == 0
    printed = capsys.readouterr().out
    assert "[exam]" in printed and "[retrieval]" in printed
    assert "Che cosa prevede" not in printed          # counts only, never questions
    assert mod.main([str(norms), "--out", str(out)]) == 1


def test_trained_articles_are_read_from_the_open_book_keys():
    pairs = [{"key": "ob-g-codice_civile-2043"}, {"key": "ob-g-codice_civile-2-bis"},
             {"key": "ob-m-codice_penale-1500-7"}, {"key": "civ-001"}, {}]
    assert trained_articles(pairs) == {("codice_civile", "2043"), ("codice_civile", "2-bis"),
                                       ("codice_penale", "1500")}


def test_raft_and_grpo_training_rows_count_as_trained_articles():
    # make_raft_absent.py appends -absent to the grounded key; GRPO prompts
    # carry no key, only code and articolo. Both were trained on.
    rows = [{"key": "ob-g-codice_civile-1456-absent"}, {"key": "ob-g-codice_civile-2-bis-absent"},
            {"id": "norm-termine-codice_penale-640", "code": "codice_penale", "articolo": "640"},
            {"id": "x", "code": "codice_penale"}]
    assert trained_articles(rows) == {("codice_civile", "1456"), ("codice_civile", "2-bis"),
                                      ("codice_penale", "640")}


def test_a_redraw_leaves_out_what_training_asked_about():
    exclude = {("codice_civile", "2"), ("codice_civile", "3")}
    for seed in range(20):
        items = build_exam(RECORDS, per_code=10, seed=seed, exclude=exclude)
        drawn = {(it.metadata["code"], it.metadata["articolo"]) for it in items}
        assert not drawn & exclude
        assert ("codice_civile", "1") in drawn


def test_a_made_up_number_training_used_is_not_asked_again():
    last = 5  # the last codice_civile article in RECORDS
    exclude = {("codice_civile", str(n)) for n in range(last + 50, last + 900)}
    exclude.discard(("codice_civile", str(last + 900)))
    items = build_exam(RECORDS, per_code=5, seed=4, exclude=exclude)
    fakes = [it for it in items if it.metadata["tipo"] == "inesistente"
             and it.metadata["code"] == "codice_civile"]
    assert fakes and all(it.metadata["articolo"] == str(last + 900) for it in fakes)


def test_two_identical_fake_draws_do_not_lose_an_item():
    """Fake numbers redraw against `exclude` but not against each other, so
    a repeated randint made two items with the same id and setdefault
    silently discarded the second: the exam came back one item short."""
    filler = " Il presente articolo contiene disposizioni di dettaglio sufficienti." * 3
    records = [{"code": "codice_civile", "article_num": "", "chunk_index": 0,
                "text": f"Art. {n}. \n \n (Rubrica {n}). \n \n Disciplina numero {n}."
                        + filler}
               for n in range(1, 61)]
    items = build_exam(records, per_code=30, seed=186)
    fakes = [it for it in items if it.metadata["tipo"] == "inesistente"]
    assert len(fakes) == 6
    assert len({it.id for it in fakes}) == 6


def test_the_script_excludes_trained_articles_and_says_only_how_many(tmp_path, capsys):
    import json

    spec = importlib.util.spec_from_file_location("make_norm_exam", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    norms = tmp_path / "legislazione_x.chunks.jsonl"
    norms.write_text("\n".join(json.dumps(r) for r in RECORDS) + "\n")
    pairs = tmp_path / "openbook-pairs.jsonl"
    pairs.write_text(json.dumps({"key": "ob-g-codice_civile-2"}) + "\n")
    out = tmp_path / "exam.jsonl"
    assert mod.main([str(norms), "--out", str(out), "--exclude-pairs", str(pairs)]) == 0
    printed = capsys.readouterr().out
    assert "1 trained articles left out" in printed and "codice_civile-2" not in printed
    drawn = [json.loads(x)["metadata"] for x in out.read_text().splitlines()]
    assert not [m for m in drawn if (m["code"], m["articolo"]) == ("codice_civile", "2")]


# --- what the development set of 2026-09-28 showed, on the real file format ---

# As the Normattiva files hold them (copied from legislazione_codice_civile
# and legislazione_codice_procedura_civile): footnote markers, a line of
# dashes, then the amendment notes.
NOTED = rec("codice_civile",
            "Art. 5. \n \n (Atti di disposizione del proprio corpo). \n \n Gli atti di "
            "disposizione del proprio corpo sono vietati quando cagionino una diminuzione "
            "permanente della integrita' fisica, o quando siano altrimenti contrari alla "
            "legge, all'ordine pubblico o al buon costume." + FILLER
            + " \n(3a) (15a) (15b) (56a) ((289a)) ------------- AGGIORNAMENTO (3a) La L. 3 "
            "aprile 1957, n. 235 ha disposto (con l'art. 1, commi 1 e 2) che \"E' consentito "
            "il prelievo di parti del cadavere\" e che la presente modifica si applica "
            "decorsi trenta giorni dalla data di entrata in vigore.")
MULTI_NOTE = rec("codice_procedura_civile",
                 "Art. 5. \n \n (Momento determinante della giurisdizione e della "
                 "competenza). \n \n La giurisdizione e la competenza si determinano con "
                 "riguardo alla legge vigente al momento della proposizione della domanda."
                 + FILLER + " \n (67) ((72)) ------------- AGGIORNAMENTO (67) La L. 26 "
                 "novembre 1990, n. 353 ha disposto (con l'art. 92, comma 1) che \"la presente "
                 "legge entra in vigore il 1 gennaio 1993\" ------------- AGGIORNAMENTO (72) La")


def test_amendment_notes_are_not_part_of_the_article():
    arts = articles_from_records([NOTED, MULTI_NOTE])
    for a in arts.values():
        assert "AGGIORNAMENTO" not in a.text
        assert "ha disposto" not in a.text
        assert not a.text.rstrip().endswith(")"), a.text[-40:]
    assert arts[("codice_civile", "5")].text.endswith("dettaglio sufficienti.")


def test_a_deadline_in_a_note_never_becomes_a_question():
    items = build_exam([NOTED], per_code=4, seed=1)
    assert not [it for it in items if it.metadata["tipo"].startswith("termine")]


def test_words_an_amendment_inserted_are_not_a_heading():
    inserted = rec("codice_procedura_penale",
                   "Art. 438. \n \n ((in ogni caso)) L'imputato puo' chiedere il giudizio "
                   "abbreviato entro quindici giorni." + FILLER)
    arts = articles_from_records([inserted, NOTED])
    assert arts[("codice_procedura_penale", "438")].heading == ""
    assert arts[("codice_civile", "5")].heading == "Atti di disposizione del proprio corpo"
    items = build_exam([inserted], per_code=4, seed=1)
    assert not [it for it in items if it.metadata["tipo"] == "termine_argomento"]


def test_the_reference_is_the_whole_article_not_its_first_800_characters():
    tail = " Il recesso e' ammesso per sopravvenuti motivi di pubblico interesse."
    long_art = rec("legge_procedimento_amministrativo",
                   "Art. Art. 11. ((Accordi integrativi o sostitutivi del provvedimento))\n"
                   + "Disposizione iniziale di contenuto generale. " * 30 + tail,
                   article_num="Art. 11.")
    items = build_exam([long_art], per_code=4, seed=1)
    it = next(i for i in items if i.metadata["tipo"] == "contenuto")
    assert len(it.reference) > 800
    assert it.reference.endswith(tail.strip())


def test_an_article_too_long_to_hand_the_grader_whole_is_not_asked():
    huge = rec("codice_civile", "Art. 9. \n \n (Lungo). \n \n " + "Parola. " * 1200)
    assert build_exam([huge], per_code=4, seed=1) == []


def test_consolidated_text_notes_and_inline_markers_are_not_the_article():
    art = rec("codice_consumo",
              "Art. Art. 105. (Presunzione e valutazione di sicurezza) Un prodotto si presume "
              "sicuro. (171) ((173)) Se rifiuta il terzo, il giudice lo condanna." + FILLER
              + " Note all' art. 105: - La direttiva 3 dicembre 2001 n. 95 del Parlamento "
              "europeo e del Consiglio.", article_num="Art. 105.")
    text = articles_from_records([art])[("codice_consumo", "105")].text
    assert "Note all'" not in text and "direttiva" not in text
    assert "(171)" not in text and "((173))" not in text
    assert "si presume sicuro. Se rifiuta il terzo" in text


# The three articles the blind review of 2026-10-03 caught: a second deadline
# in a wording the statute pattern does not know.
HIDDEN_SECOND = [
    "fissa una apposita udienza non oltre sessanta giorni. Tra la data del provvedimento "
    "e l'udienza deve intercorrere un termine non inferiore a venti giorni.",
    "La dichiarazione deve essere fatta non oltre i dieci giorni dalla data del pignoramento "
    "e notificata entro cinque giorni dalla sua data.",
    "Quando sono trascorsi ((cinque)) anni dall'ultima notizia. In nessun caso se non sono "
    "trascorsi nove anni dalla maggiore eta'.",
]


@pytest.mark.parametrize("text", HIDDEN_SECOND)
def test_a_second_deadline_in_any_wording_is_seen(text):
    assert len(all_deadlines(text)) == 2


def test_an_article_with_a_hidden_second_deadline_is_not_asked_about():
    recs = [rec("codice_civile", f"Art. {n}. \n \n (Rubrica {n}). \n \n {t}" + FILLER)
            for n, t in enumerate(HIDDEN_SECOND, start=10)]
    recs.append(rec("codice_civile", "Art. 20. \n \n (Una). \n \n La domanda si propone "
                    "entro sessanta giorni." + FILLER))
    for seed in range(10):
        timed = {it.metadata["articolo"] for it in build_exam(recs, per_code=10, seed=seed)
                 if it.metadata["tipo"] in ("termine", "termine_argomento")}
        assert timed == {"20"}


def test_version_2_rubrics_accept_true_context_and_every_deadline_of_the_article():
    assert rubric_v2("norm-contenuto-codice_civile-1", "old") == CONTENT_RUBRIC
    assert "non contraddicono" in CONTENT_RUBRIC and "contraddice il testo" in CONTENT_RUBRIC
    one = rubric_v2("norm-termine-codice_civile-2",
                    "Corretto solo se indica il termine di 60 giorni.",
                    "La domanda si propone entro sessanta giorni.\n\nTesto integrale "
                    "dell'articolo: La domanda si propone entro sessanta giorni.")
    assert "termine di 60 giorni" in one and "altri termini" not in one
    two = rubric_v2("norm-termine-codice_procedura_penale-554-ter",
                    "Corretto solo se indica il termine di 60 giorni.",
                    "non oltre sessanta giorni\n\nTesto integrale dell'articolo: "
                    + HIDDEN_SECOND[0])
    assert "termine di 60 giorni" in two and "20 giorni" in two
    assert rubric_v2("norm-inesistente-codice_civile-999", "Corretto solo se dice che "
                     "l'articolo non esiste.") == "Corretto solo se dice che l'articolo non esiste."

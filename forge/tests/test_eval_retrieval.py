"""Open-book retrieval and the reference grader."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from eullm_forge.eval import Grade, NormIndex, ReferenceGrader, open_book_prompt
from eullm_forge.eval.retrieval import named_articles, named_code


def rec(code, num, text, chunk=0):
    return {"text": text, "code": code, "article_num": num, "chunk_index": chunk}


RECORDS = [
    rec("codice_civile", "2043", "Art. 2043. Risarcimento per fatto illecito. Qualunque "
        "fatto doloso o colposo, che cagiona ad altri un danno ingiusto, obbliga colui "
        "che ha commesso il fatto a risarcire il danno."),
    rec("codice_civile", "1453", "Art. 1453. Nei contratti con prestazioni corrispettive, "
        "quando uno dei contraenti non adempie le sue obbligazioni, l'altro può a sua "
        "scelta chiedere l'adempimento o la risoluzione del contratto."),
    rec("costituzione", "27", "Art. 27. La responsabilità penale è personale. L'imputato "
        "non è considerato colpevole sino alla condanna definitiva."),
    rec("codice_penale", "27", "Art. 27. Testo di un altro codice con lo stesso numero."),
    rec("codice_procedura_civile", "327", "Art. 327. Decadenza dall'impugnazione. "
        "Indipendentemente dalla notificazione, l'appello non può proporsi dopo "
        "decorsi sei mesi dalla pubblicazione della sentenza."),
]


@pytest.fixture
def index():
    return NormIndex(RECORDS)


@pytest.mark.parametrize("question, code, nums", [
    ("Che cosa prevede l'articolo 2043 del codice civile?", "codice_civile", ["2043"]),
    ("Cosa stabilisce l'art. 27 della Costituzione italiana?", "costituzione", ["27"]),
    ("Cosa dice l'art. 327 c.p.c.?", "codice_procedura_civile", ["327"]),
    ("Cosa dice l'art. 2043-bis c.c.?", "codice_civile", ["2043-bis"]),
])
def test_a_named_article_and_code_are_read_out_of_the_question(question, code, nums):
    assert named_code(question) == code
    assert named_articles(question) == nums


def test_procedura_civile_is_not_read_as_codice_civile():
    assert named_code("art. 327 del codice di procedura civile") == "codice_procedura_civile"


def test_a_named_article_is_looked_up_in_the_named_code_only(index):
    found = index.search("Cosa stabilisce l'art. 27 della Costituzione?", k=1)
    assert found[0]["code"] == "costituzione"


def test_a_question_without_an_article_goes_to_bm25(index):
    found = index.search("Entro quanti mesi si propone l'appello se la sentenza "
                         "non è stata notificata?", k=1)
    assert found[0]["article_num"] == "327"


def test_equal_bm25_scores_keep_index_order():
    """The rule rrf() applies to a fused ranking, applied here too.

    Two chunks the same length with the query term once each have exactly the
    same BM25 score, and scores.sort(reverse=True) on (score, index) tuples
    reversed both keys, so the last record won. The exam prompt was then built
    from the other article, and an article that ties for first measured top1
    = 0.
    """
    filler = " Il presente articolo contiene disposizioni di dettaglio." * 4
    records = [{"code": "codice_civile", "article_num": "", "chunk_index": 0,
                "text": f"Art. {n}. \n (Rimedio giudiziale). \n Quando si chiede "
                        f"la condanna di cui all'articolo {n}{filler}"}
               for n in (1456, 2999)]
    question = ("Nel codice civile, in materia di «Rimedio giudiziale», qual e "
                "il termine previsto?")
    index = NormIndex(records)

    assert index.bm25(question, 2) == records          # not [records[1], records[0]]
    assert index.search(question, 1) == [records[0]]


def test_search_fills_up_to_k_without_repeating_the_looked_up_article(index):
    found = index.search("Che cosa prevede l'articolo 2043 del codice civile sul danno "
                         "ingiusto?", k=3)
    assert found[0]["article_num"] == "2043"
    assert len({id(r) for r in found}) == len(found)


def test_the_prompt_carries_the_text_and_the_question(index):
    found = index.search("Che cosa prevede l'articolo 2043 del codice civile?", k=1)
    p = open_book_prompt("Che cosa prevede l'articolo 2043 del codice civile?", found)
    assert "codice civile, art. 2043" in p and "danno ingiusto" in p
    assert p.endswith("Domanda: Che cosa prevede l'articolo 2043 del codice civile?")
    assert open_book_prompt("Domanda?", []) == "Domanda?"


def test_a_whole_chunked_article_is_shown_whole():
    """The corpus is chunked at --max-chars 3000; a block cut below that
    shows the model half an article and grades the answer on the half."""
    long_article = rec("codice_civile", "1176", "x" * 3000)
    p = open_book_prompt("Q?", [long_article])
    assert "x" * 3000 in p
    assert "[…]" not in p


def test_a_text_that_does_not_fit_is_marked_as_cut():
    long_article = rec("codice_civile", "1176", "y" * 4000)
    p = open_book_prompt("Q?", [long_article], max_chars=1000)
    assert "y" * 1000 in p
    assert "y" * 1001 not in p
    # Otherwise the block just stops, and reads as the end of the article.
    assert p.count("[…]") == 1


def test_from_files_reads_jsonl(tmp_path):
    path = tmp_path / "legislazione_codice_civile.chunks.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in RECORDS[:2]) + "\n")
    assert len(NormIndex.from_files([path]).records) == 2


# --- the grader --------------------------------------------------------------

@pytest.mark.parametrize("raw, label, score", [
    ("Grade: correct\nMatches the reference.", "correct", 1.0),
    ("grade: PARTIAL\nRight article, wrong term.", "partial", 0.5),
    ("Grade: wrong\nThirty days, not one hundred and twenty.", "wrong", 0.0),
    # Decorated labels are still labels: a 30B model puts asterisks on them,
    # and reading that as unreadable turned a formatting difference into a
    # legal error.
    ("Grade: **correct**\nMatches the reference.", "correct", 1.0),
    ("Grade: _wrong_\nThe deadline is wrong.", "wrong", 0.0),
    ("**Grade: partial**\nHalf of it.", "partial", 0.5),
])
def test_grades_are_parsed_and_scored(raw, label, score):
    g = ReferenceGrader.parse(raw)
    assert (g.label, g.score) == (label, score)


@pytest.mark.parametrize("raw", [
    "I think it is fine.",
    "Grade: incorrect",
    "Verdetto: corretto",
    "The answer is correct: it cites art. 2043.",
])
def test_an_unreadable_grade_is_nan_and_not_a_zero(raw):
    """It used to score 0.0, which is the score of a wrong answer, so a mean
    over a run with one unreadable line was quietly too low for a reason
    nothing in it showed."""
    g = ReferenceGrader.parse(raw)
    assert g.label == "unparsed"
    assert math.isnan(g.score)


def test_the_grader_shows_the_reference_to_the_model():
    seen = []
    grader = ReferenceGrader(lambda p: seen.append(p) or "Grade: wrong\nno")
    g = grader.grade("Entro quale termine?", "Centoventi giorni.", "Trenta giorni.")
    assert g.label == "wrong"
    assert "Centoventi giorni." in seen[0] and "Trenta giorni." in seen[0]


# --- the judge script's pure parts --------------------------------------------

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "judge_answers.py"
spec = importlib.util.spec_from_file_location("judge_answers", SCRIPT)
judge_answers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge_answers)


def test_labels_come_from_the_answers_file_name():
    assert judge_answers.label_of(Path("eval/answers-v0.2-open.jsonl")) == "v0.2-open"


def test_the_summary_counts_grades_and_scores_partial_as_half():
    grades = [Grade("correct"), Grade("partial"), Grade("wrong"), Grade("wrong")]
    row = dict(zip(judge_answers.CSV_HEADER, judge_answers.summary_row("m", grades)))
    assert (row["correct"], row["partial"], row["wrong"]) == (1, 1, 2)
    assert row["score"] == "0.375"


def test_grade_file_writes_the_graded_copy(tmp_path):
    src = tmp_path / "answers-x.jsonl"
    src.write_text(json.dumps({"id": "a", "question": "Q?", "answer": "A.",
                               "reference": "R."}) + "\n")
    grades = judge_answers.grade_file(src, ReferenceGrader(lambda p: "Grade: partial\nmeh"))
    assert [g.label for g in grades] == ["partial"]
    line = json.loads((tmp_path / "answers-x.graded.jsonl").read_text())
    assert line["grade"] == "partial" and line["why"].startswith("Grade: partial")


@pytest.mark.parametrize("question, code", [
    ("Cosa prevede l'art. 29 del codice del processo amministrativo?",
     "codice_processo_amministrativo"),
    ("Cosa dice l'art. 29 c.p.a.?", "codice_processo_amministrativo"),
    ("Cosa prevede l'art. 10-bis della legge 241/1990?", "legge_procedimento_amministrativo"),
    ("Cosa dice l'art. 9 del d.P.R. 1199/1971?", "ricorsi_amministrativi"),
])
def test_administrative_norms_are_named_too(question, code):
    assert named_code(question) == code


# --- the formats the legislation files REALLY hold (Leonardo, 2026-09-27) ----
#
# The first version of NormIndex was tested on tidy records with
# article_num="2043". The real files have article_num="" for the codes from
# the Normattiva ZIP, "Art. 1." for the laws from single XML, and chunks of
# the c.p.a. holding several articles or its index. Retrieval by article
# never matched, and the open-book run read unrelated text. These records are
# copied from the real files, text shortened.

REAL = [
    {"source_id": "codice_civile", "kind": "legge", "code": "codice_civile", "article_num": "",
     "chunk_index": 0, "chunk_total": 1,
     "text": "DISPOSIZIONI SULLA LEGGE IN GENERALE \n \n Art. 1. \n \n (Indicazione delle "
             "fonti). \n \n Sono fonti del diritto le leggi, i regolamenti."},
    {"source_id": "codice_civile", "kind": "legge", "code": "codice_civile", "article_num": "",
     "chunk_index": 0, "chunk_total": 1,
     "text": "Art. 2043. \n \n (Risarcimento per fatto illecito). \n \n Qualunque fatto doloso "
             "o colposo, che cagiona ad altri un danno ingiusto, obbliga a risarcire il danno."},
    {"source_id": "codice_civile", "kind": "legge", "code": "codice_civile", "article_num": "",
     "chunk_index": 0, "chunk_total": 1,
     "text": "Art. 2044. \n \n (Legittima difesa). \n \n Non è responsabile chi cagiona il "
             "danno per legittima difesa, ai sensi dell'art. 2043 e seguenti."},
    {"source_id": "costituzione/art_Art. 27.", "kind": "legge", "code": "costituzione",
     "article_num": "Art. 27.", "chunk_index": 0, "chunk_total": 1,
     "text": "Art. Art. 27.\nLa responsabilita' penale e' personale."},
    {"source_id": "legge_procedimento_amministrativo/art_Art. 2.", "kind": "legge",
     "code": "legge_procedimento_amministrativo", "article_num": "Art. 2.",
     "chunk_index": 0, "chunk_total": 2,
     "text": "Art. Art. 2. ((Conclusione del procedimento))\nOve il procedimento consegua..."},
    {"source_id": "legge_procedimento_amministrativo/art_Art. 2.", "kind": "legge",
     "code": "legge_procedimento_amministrativo", "article_num": "Art. 2.",
     "chunk_index": 1, "chunk_total": 2, "text": "...termine di trenta giorni."},
    {"source_id": "codice_processo_amministrativo", "kind": "legge",
     "code": "codice_processo_amministrativo", "article_num": "", "chunk_index": 0,
     "chunk_total": 3,
     "text": "INDICE GENERALE \n Art. 27 - Contraddittorio \n Art. 28 - Intervento \n "
             "Art. 29 - Azione di annullamento \n Art. 30 - Azione di condanna \n "
             "Art. 31 - Silenzio"},
    {"source_id": "codice_processo_amministrativo", "kind": "legge",
     "code": "codice_processo_amministrativo", "article_num": "", "chunk_index": 1,
     "chunk_total": 3,
     "text": "Art. 29 \n Azione di annullamento \n 1. L'azione di annullamento per violazione "
             "di legge si propone nel termine di decadenza di sessanta giorni."},
    {"source_id": "codice_processo_amministrativo", "kind": "legge",
     "code": "codice_processo_amministrativo", "article_num": "", "chunk_index": 2,
     "chunk_total": 3, "text": "continua il testo dell'articolo senza intestazione."},
]


@pytest.fixture
def real():
    return NormIndex(REAL)


@pytest.mark.parametrize("question, first_text", [
    ("Che cosa prevede l'articolo 2043 del codice civile?", "Art. 2043."),
    ("Cosa stabilisce l'art. 27 della Costituzione italiana?", "Art. Art. 27."),
    ("Cosa dice l'art. 2 della legge 241/1990?", "Art. Art. 2."),
    ("Cosa prevede l'art. 29 c.p.a.?", "Art. 29 \n"),
])
def test_the_named_article_is_found_in_the_real_formats(real, question, first_text):
    assert real.search(question, k=1)[0]["text"].startswith(first_text)


def test_a_reference_inside_another_article_is_not_that_article(real):
    hits = real.by_article("Cosa prevede l'art. 2043 del codice civile?")
    assert [h["text"][:10] for h in hits] == ["Art. 2043."]


def test_an_article_comes_before_an_index_that_merely_lists_it(real):
    hits = real.by_article("Cosa prevede l'art. 29 c.p.a.?")
    assert hits[0]["text"].startswith("Art. 29 \n")
    assert hits[-1]["text"].startswith("INDICE GENERALE")


def test_a_chunk_without_a_header_continues_the_article_before_it(real):
    hits = real.by_article("Cosa dice l'art. 2 della legge 241/1990?")
    assert [h["chunk_index"] for h in hits] == [0, 1]


def test_labels_carry_the_article_actually_found(real):
    from eullm_forge.eval.retrieval import label
    assert [label(r) for r in REAL[1:2] + REAL[3:4]] == [
        "codice civile, art. 2043", "costituzione, art. 27"]


def test_bm25_stays_inside_the_code_the_question_names(real):
    found = real.bm25("termine di decadenza di sessanta giorni codice del processo "
                      "amministrativo", k=5, code="codice_processo_amministrativo")
    assert found and {r["code"] for r in found} == {"codice_processo_amministrativo"}


def test_a_named_article_that_is_not_there_is_said_so(real):
    note = real.missing_article_note("Che cosa prevede l'art. 3500 del codice civile?")
    assert "art. 3500" in note and "non è presente" in note
    assert real.missing_article_note("Che cosa prevede l'art. 2043 del codice civile?") == ""
    assert real.missing_article_note("Che cos'è il danno ingiusto?") == ""
    prompt = open_book_prompt("Q?", [], note=note)
    assert prompt.startswith(note) and "nessun testo pertinente" in prompt


# --- the rubrica: read in each real format, and weighted in the ranking ------

def test_the_rubrica_is_read_in_every_real_format():
    from eullm_forge.eval.retrieval import record_heading
    assert record_heading(REAL[1]) == "Risarcimento per fatto illecito"
    assert record_heading(REAL[4]) == "Conclusione del procedimento"
    assert record_heading(REAL[7]) == "Azione di annullamento"
    assert record_heading(REAL[3]) == ""          # no rubrica, the text starts
    assert record_heading(REAL[6]) == ""          # the index is not an article


def test_a_continuation_chunk_keeps_its_articles_rubrica():
    index = NormIndex(REAL)
    assert index._headings[8] == "Azione di annullamento"


def _cpc(num, body):
    return {"code": "codice_procedura_civile", "article_num": "", "chunk_index": 0,
            "text": f"Art. {num}. \n \n {body}"}


# The shape of the failure: the target's rubrica holds the words, its long
# body does not repeat them; a short neighbour repeats them in its body.
CITAZIONE = [
    _cpc(163, "(Contenuto della citazione). \n \n La domanda si propone mediante atto "
              "scritto notificato al convenuto, con l'indicazione del tribunale, delle "
              "parti, dell'oggetto, dei fatti e degli elementi di diritto, e con l'invito "
              "a costituirsi settanta giorni prima dell'udienza." + " Il presidente del "
              "tribunale designa il giudice istruttore e fissa l'udienza." * 6),
    _cpc(164, "(Nullita' dell'atto introduttivo). \n \n Se il contenuto della citazione "
              "e' incompleto, la citazione e' nulla."),
] + [_cpc(n, f"(Disposizione {n}). \n \n Il giudice provvede con ordinanza sulle istanze "
                f"delle parti nel caso numero {n}.") for n in range(200, 206)]


def test_a_topic_question_in_the_rubricas_words_finds_that_article_first():
    """Asked by topic, plain BM25 ranked first the article whose body repeats
    the words, not the one titled with them."""
    q = ("Nel codice di procedura civile, in materia di «Contenuto della citazione», "
         "qual è il termine previsto?")
    plain = NormIndex(CITAZIONE, heading_boost=0).search(q, 1)[0]
    boosted = NormIndex(CITAZIONE, heading_boost=3).search(q, 1)[0]
    assert "Art. 164." in plain["text"]
    assert "Art. 163." in boosted["text"]


def test_the_retrieval_check_prints_counts_per_boost_and_no_question(tmp_path, capsys):
    from eullm_forge.eval import EvalItem, save_eval_set

    norms = tmp_path / "legislazione_cpc.chunks.jsonl"
    norms.write_text("\n".join(json.dumps(r) for r in CITAZIONE) + "\n")
    exam = tmp_path / "dev.jsonl"
    save_eval_set([EvalItem(
        id="norm-termine_argomento-codice_procedura_civile-163", domain="legal", lang="it",
        question="Nel codice di procedura civile, in materia di «Contenuto della citazione», "
                 "qual è il termine previsto?",
        metadata={"tipo": "termine_argomento", "code": "codice_procedura_civile",
                  "articolo": "163"})], exam)
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_retrieval.py"
    spec = importlib.util.spec_from_file_location("check_retrieval", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main([str(exam), "--norms", str(norms), "--boost", "0", "3"]) == 0
    out = capsys.readouterr().out
    lines = {ln.split()[2]: ln for ln in out.splitlines() if "termine_argomento" in ln}
    assert "n=1" in lines["boost=0"] and "top1=0.00" in lines["boost=0"]
    assert "n=1" in lines["boost=3"] and "top1=1.00" in lines["boost=3"]
    assert "Contenuto della citazione" not in out


def test_the_default_ranking_is_unchanged_until_a_boost_is_chosen():
    assert NormIndex(REAL).heading_boost == 0


def test_the_retrieval_check_measures_the_teachers_topic_questions(tmp_path, capsys):
    """Paraphrased, the way people ask: the fairer test of a rubrica weight."""
    norms = tmp_path / "legislazione_cpc.chunks.jsonl"
    norms.write_text("\n".join(json.dumps(r) for r in CITAZIONE) + "\n")
    pairs = tmp_path / "openbook-pairs.jsonl"
    rows = [
        {"task": "openbook_grounded", "named": False, "key": "ob-g-codice_procedura_civile-163",
         "instruction": "Testi normativi di riferimento: ...\n\nDomanda: Nel codice di "
                        "procedura civile, cosa deve contenere la citazione?", "output": "x"},
        {"task": "openbook_grounded", "named": True, "key": "ob-g-codice_procedura_civile-164",
         "instruction": "Domanda: Cosa dice l'art. 164 del codice di procedura civile?",
         "output": "x"},
        {"task": "openbook_missing", "key": "ob-m-codice_procedura_civile-900-1",
         "instruction": "Domanda: art. 900?", "output": "x"},
    ]
    pairs.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_retrieval.py"
    spec = importlib.util.spec_from_file_location("check_retrieval", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert [it.metadata["articolo"] for it in mod.topic_questions(pairs)] == ["163"]
    assert mod.main(["--pairs", str(pairs), "--norms", str(norms), "--boost", "3"]) == 0
    out = capsys.readouterr().out
    assert "argomento_insegnante n=1" in out
    assert "citazione" not in out

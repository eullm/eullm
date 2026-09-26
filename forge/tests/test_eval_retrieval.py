"""Open-book retrieval and the reference grader."""

from __future__ import annotations

import importlib.util
import json
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
    ("I think it is fine.", "unparsed", 0.0),
])
def test_grades_are_parsed_and_scored(raw, label, score):
    g = ReferenceGrader.parse(raw)
    assert (g.label, g.score) == (label, score)


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

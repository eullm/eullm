"""The abstention exam: what counts as abstaining, what as citing from memory,
and the article taken out of the texts by `legal_eval.py --absent`."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from eullm_forge.datasets.openbook_gen import ABSENT_ANSWER
from eullm_forge.eval import EvalItem, NormIndex
from eullm_forge.eval.abstain import ABSTAIN, abstained, cited_articles, unsourced_articles

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_what_the_abstain_rows_teach_is_read_as_abstaining():
    # The answers the OPD rows of make_abstain_prompts.py steer towards.
    assert abstained(ABSENT_ANSWER.format(of="del codice penale"))
    assert abstained("Senza il testo della norma non posso darti una risposta sicura: "
                     "andrebbe consultato il codice civile.")
    assert abstained("Nei testi forniti non è indicato il termine richiesto.")
    assert abstained("Non ho trovato informazioni sufficienti nei documenti.")


def test_a_right_answer_is_not_read_as_abstaining():
    # Phrases a right answer uses: an abstention read here is a lost answer.
    for answer in ("Se il convenuto non è presente all'udienza, il giudice dichiara la "
                   "contumacia.",
                   "Ove non risulta diversamente dal contratto, il termine è di dieci giorni.",
                   "Secondo l'art. 54 c.p. non è punibile chi ha agito per necessità.",
                   "Il giudice non può pronunciare sentenza senza aver sentito le parti.",
                   "Le informazioni rilevanti sono contenute nell'art. 2043 c.c."):
        assert not abstained(answer), answer


@pytest.mark.parametrize("answer", [
    # Real model answers recorded when the reward was fixed: the singular is
    # the same sentence, one text instead of many, and "non risulta alcuna"
    # is the usual phrasing. The exam still knew only "non contengono".
    "Il testo fornito non contiene l'articolo richiesto.",
    "Il brano riportato non contiene la disposizione richiesta.",
    "La raccolta fornita non contiene l'art. 2999 del codice civile.",
    "Dagli atti riportati non risulta alcuna disposizione in merito.",
    "Non risulta nessuna norma che disciplini la materia.",
    "Nel brano riportato non compare la disposizione di cui si chiede.",
    "La raccolta non include l'articolo richiesto.",
    # The three-step answers of the 2026-10 rounds (devbig, nonexistent articles).
    "Verifica della premessa: tra i testi riportati non c'è l'art. 957 del codice di "
    "procedura penale, e non mi risulta che esista: la domanda parte da un riferimento errato.",
    "Risposta: non posso dirti che cosa prevede l'art. 400 della Costituzione.",
])
def test_the_exam_reads_the_abstention_the_reward_already_pays(answer):
    assert abstained(answer), answer
    # the RAG Enterprise copy must agree, and it is the same pattern
    assert _load("rag_enterprise_eval").abstained(answer), answer


def test_rag_enterprise_eval_reads_abstentions_the_same_way():
    assert _load("rag_enterprise_eval").ABSTAIN.pattern == ABSTAIN.pattern


def test_articles_cited_lists_included():
    assert cited_articles("L'articolo 2043 c.c. e gli artt. 1176 e 1375, art. 54-bis, "
                          "art. 9 quater, artt. 3, 4 ed 5") == \
        {"2043", "1176", "1375", "54-bis", "9-quater", "3", "4", "5"}
    assert cited_articles("Il termine è di trenta giorni.") == set()
    # a quantity or a paragraph after the list separator is not an article
    assert cited_articles("ai sensi dell'art. 1453 e 3 mesi dopo") == {"1453"}
    assert cited_articles("l'art. 360, 1° comma, n. 3 c.p.c.") == {"360"}
    assert cited_articles("l'art. 12 e 2 commi") == {"12"}
    assert cited_articles("artt. 2043, 2059 e 2087") == {"2043", "2059", "2087"}


def test_only_articles_from_memory_count():
    # The 2026-10-08 chat: asked about theft, legal-it-8b brought in art. 49.
    answer = "Ai sensi dell'art. 624 c.p. e dell'art. 625, letto con l'art. 49, ..."
    question = "Che cosa prevede l'art. 624 c.p.?"
    assert unsourced_articles(answer, question) == ["49", "625"]
    assert unsourced_articles(answer, question, {"625"}) == ["49"]
    assert unsourced_articles(answer, question, {"625", "49"}) == []


def _rec(code, num, text, chunk=0):
    return {"text": text, "code": code, "article_num": num, "chunk_index": chunk}


def test_absent_takes_the_items_own_article_out_and_says_nothing():
    legal_eval = _load("legal_eval")
    index = NormIndex([
        _rec("codice_penale", "624", "Art. 624. Furto. Chiunque s'impossessa della cosa "
             "mobile altrui sottraendola a chi la detiene è punito con la reclusione."),
        _rec("codice_penale", "625", "Art. 625. Circostanze aggravanti del furto: cosa "
             "mobile altrui, reclusione aumentata."),
        _rec("codice_penale", "626", "Art. 626. Furti punibili a querela: cosa mobile "
             "altrui sottratta per farne uso momentaneo."),
        _rec("codice_civile", "1453", "Art. 1453. Risoluzione del contratto."),
    ])
    it = EvalItem(id="norm-contenuto-codice_penale-624", domain="legal", lang="it",
                  question="Che cosa prevede l'art. 624 c.p. sulla cosa mobile altrui?",
                  metadata={"tipo": "contenuto", "code": "codice_penale", "articolo": "624"})
    found, note = legal_eval.retrieve(index, it, 2)
    assert "624" in index.articles_of(found[0]) and note == ""
    found, note = legal_eval.retrieve(index, it, 2, absent=True)
    assert len(found) == 2 and note == ""
    assert not any("624" in index.articles_of(r) for r in found)
    # An article the collection does not hold: the note stays in the normal run.
    ghost = EvalItem(id="norm-inesistente-codice_penale-9999", domain="legal", lang="it",
                     question="Che cosa prevede l'art. 9999 del codice penale?",
                     metadata={"tipo": "inesistente", "code": "codice_penale",
                               "articolo": "9999"})
    assert "non è presente art. 9999" in legal_eval.retrieve(index, ghost, 2)[1]
    assert legal_eval.retrieve(index, ghost, 2, absent=True)[1] == ""
    # --no-note: the same texts, and no note, as RAG Enterprise asks it
    found, note = legal_eval.retrieve(index, ghost, 2, note=False)
    assert note == "" and found == legal_eval.retrieve(index, ghost, 2)[0]


def test_an_article_a_given_text_refers_to_is_in_hand():
    legal_eval = _load("legal_eval")
    index = NormIndex([_rec("codice_civile", "1218", "Art. 1218. Il debitore risponde "
                            "del danno, salvo quanto previsto dall'art. 1176 e dagli "
                            "artt. 1256 e 1257.")])
    assert legal_eval.articles_in_hand(index, index.records) == {"1218", "1176", "1256",
                                                                  "1257"}


def test_the_summary_counts_by_kind_and_works_out_old_files(tmp_path, capsys):
    mod = _load("abstain_summary")
    new = tmp_path / "answers-x-absent.jsonl"
    new.write_text("".join(json.dumps(r) + "\n" for r in [
        {"id": "norm-contenuto-codice_penale-624", "question": "art. 624?",
         "answer": "Nei testi disponibili non ho trovato la norma.",
         "context": ["codice penale, art. 625"], "abstained": False,
         "unsourced_articles": [], "absent": True},
        {"id": "norm-termine-codice_penale-626", "question": "art. 626?", "answer": "...",
         "context": [], "abstained": False, "unsourced_articles": ["49"], "absent": True}]), encoding="utf-8")
    old = tmp_path / "answers-x.jsonl"         # closed book, written before the fields
    old.write_text(json.dumps({"id": "norm-contenuto-codice_civile-2043",
                               "question": "Che cosa prevede l'art. 2043 c.c.?",
                               "answer": "L'art. 2043, con l'art. 1176, ...",
                               "context": None}) + "\n", encoding="utf-8")
    assert mod.main([str(new), str(old)]) == 0
    out = capsys.readouterr().out
    assert "answers-x-absent.jsonl [absent]: 2 items | abstained 1 (50.0%) | " \
           "citing articles not in hand 1 (50.0%)" in out
    assert "    termine: 1 items | abstained 0 (0.0%)" in out
    assert "answers-x.jsonl [closed]: 1 items | abstained 0 (0.0%) | " \
           "citing articles not in hand n/a (written before the check)" in out
    assert "art. 624?" not in out and "..." not in out   # counts only


def test_the_summary_reads_answers_again_with_the_current_checks(tmp_path, capsys):
    """An abstention the stored flag missed is counted, and with no texts the
    articles cited are read again: "e 3 mesi" is no longer article 3."""
    mod = _load("abstain_summary")
    closed = tmp_path / "answers-y.jsonl"
    closed.write_text(json.dumps({
        "id": "norm-termine-codice_civile-1453", "question": "Che cosa prevede l'art. 1453 c.c.?",
        "answer": "L'art. 1453 e 3 mesi dopo...", "context": None,
        "abstained": False, "unsourced_articles": ["3"]}) + "\n", encoding="utf-8")
    assert mod.main([str(closed)]) == 0
    assert "citing articles not in hand 0 (0.0%)" in capsys.readouterr().out

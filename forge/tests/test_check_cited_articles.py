"""check_cited_articles.py: an article cited that its code does not hold."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from eullm_forge.eval import NormIndex

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_cited_articles.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_cited_articles", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rec(code, num):
    return {"text": f"Art. {num}. Testo.", "code": code, "article_num": str(num),
            "chunk_index": 0}


def _known(mod):
    return mod.corpus_articles(NormIndex([
        _rec("codice_civile", 1453), _rec("codice_civile", 2043), _rec("codice_civile", 2969),
        _rec("codice_civile", 1443), _rec("codice_procedura_civile", 360),
        _rec("codice_procedura_civile", 499), _rec("codice_penale", 624),
        _rec("codice_procedura_penale", "420-quater"), _rec("costituzione", 3)]))


def test_an_article_past_the_code_is_flagged_and_a_real_one_is_not():
    mod = _load()
    known = _known(mod)
    q = "Che cosa prevede l'art. 2043 del codice civile?"
    assert mod.invented("Ai sensi dell'art. 2043 e dell'art. 1453 c.c. ...", q, known) == []
    assert mod.invented("Si veda anche l'art. 3500 c.c.", q, known) == \
        ["codice_civile art. 3500"]
    # the code named right after the citation wins over the question's
    assert mod.invented("e l'art. 360 c.p.c. sul ricorso", q, known) == []
    assert mod.invented("e l'art. 361 c.p.c. sul ricorso", q, known) == \
        ["codice_procedura_civile art. 361"]


def test_the_article_asked_about_is_not_counted_when_the_answer_denies_it():
    mod = _load()
    known = _known(mod)
    q = "Che cosa prevede l'art. 3434 del codice civile?"
    assert mod.invented("L'art. 3434 del codice civile non esiste.", q, known) == []


def test_a_citation_whose_code_cannot_be_told_is_left_out():
    mod = _load()
    assert mod.invented("Vedi l'art. 99999.", "Una domanda senza codice.", _known(mod)) == []
    # an act outside the corpus, named after the article, is not the question's code
    q = "Che cosa prevede l'art. 2043 del codice civile?"
    assert mod.invented("come l'art. 5 del d.lgs. 165/2001", q, _known(mod)) == []
    assert mod.invented("e l'art. 7 della legge n. 300/1970", q, _known(mod)) == []


def test_a_code_in_a_source_list_belongs_to_the_citation_it_opens():
    mod = _load()
    known = _known(mod)
    q = "Che cosa prevede l'art. 2043 del codice civile?"
    assert mod.invented("art. 1443 s.s.; *Fonte: [Codice di procedura civile, art. 499]*",
                        q, known) == []
    assert mod.invented("[Codice civile, art. 2969.txt; Codice penale, art. 624.txt]",
                        q, known) == []
    assert mod.invented("art. 1453] e [artt. 624 Codice Penale]", q, known) == []
    assert mod.invented("[Codice penale, art. 2969]", q, known) == \
        ["codice_penale art. 2969"]
    # a list's heading opens the citations after it
    q = "Che cosa prevede l'art. 360 del codice di procedura civile?"
    assert mod.invented("(artt. 499, 360); * Codice penale e norme correlate (artt. 624)",
                        q, known) == []


def test_a_retrieved_file_name_is_not_checked():
    mod = _load()
    q = "Che cosa prevede l'art. 2043 del codice civile?"
    assert mod.invented("[Codice penale, art. 2969.txt]", q, _known(mod)) == []
    assert mod.invented("art. 392-bis.txt]. Per ottenere dettagli", q, _known(mod)) == []


def test_costituzione_in_mora_is_not_the_constitution():
    mod = _load()
    known = _known(mod)
    q = "Che cosa prevede l'art. 2043 del codice civile?"
    assert mod.invented("art. 1453 - Costituzione in mora.txt", q, known) == []
    assert mod.invented("gli articoli 1443 e 1453 in caso di mancata costituzione entro i termini",
                        q, known) == []
    assert mod.invented("articolo 1443. La costituzione oltre i suddetti termini", q, known) == []
    assert mod.invented("artt. 1443, 1453 (Costituzione), e altri", q, known) == []
    assert mod.invented("l'art. 99 della Costituzione", q, known) == ["costituzione art. 99"]
    assert mod.invented("art. 99 Cost.", q, known) == ["costituzione art. 99"]
    assert mod.invented("oppure Costituzione art. 99", q, known) == ["costituzione art. 99"]


def test_the_first_code_named_wins_and_abbreviations_are_read():
    mod = _load()
    known = _known(mod)
    q = "Che cosa prevede l'art. 2043 del codice civile?"
    assert mod.invented("art. 625 o altri riferimenti al *Codice penale* e *Codice civile*",
                        q, known) == ["codice_penale art. 625"]
    assert mod.invented("Art. 420-quater cpp: il termine è sospeso", q, known) == []
    assert mod.invented("Art. 999 cpp: il termine", q, known) == \
        ["codice_procedura_penale art. 999"]


def test_an_article_cited_without_its_code_counts_only_if_no_code_holds_it():
    mod = _load()
    known = _known(mod)
    q = "Che cosa prevede l'art. 2043 del codice civile?"
    assert mod.invented("l'art. 624 punisce il furto", q, known) == []
    assert mod.invented("l'art. 9999 punisce il furto", q, known) == ["codice_civile art. 9999"]
    assert mod.invented("art. 186-bis R.D. n° 267/40", q, known) == []
    assert mod.invented("articolo 161-quater delle disposizioni per l'attuazione", q, known) == []
    # the codes' own words are not another act
    assert mod.invented("l'art. 9999 sul decreto ingiuntivo", q, known) == \
        ["codice_civile art. 9999"]


def test_only_a_code_next_to_the_citation_is_its_own():
    mod = _load()
    known = _known(mod)
    q = "Che cosa prevede l'art. 2043 del codice civile?"
    # named next to it: a real article under the wrong code is a wrong citation
    assert mod.invented("art. 1453 c.p.c. sul ricorso", q, known) == \
        ["codice_procedura_civile art. 1453"]
    assert mod.invented("art. 9999 del Codice Penale, sostituendo", q, known) == \
        ["codice_penale art. 9999"]
    # named further on, or in the next item: a hint, not its code
    assert mod.invented("artt. 1443, 1453) e del **Codice di procedura penale**", q, known) == []
    assert mod.invented("articolo 1443, 1453) ed elementi del Codice Penale", q, known) == []
    assert mod.invented("art. 1453; *Codice penale*; o norme procedurali", q, known) == []
    assert mod.invented("art. 9999) ed elementi del Codice Penale", q, known) == \
        ["codice_penale art. 9999"]


def test_more_acts_the_corpus_does_not_hold():
    mod = _load()
    known = _known(mod)
    q = "Che cosa prevede l'art. 2043 del codice civile?"
    assert mod.invented("articolo 16-sexies del D.L. 18 ottobre 2012, n. 179", q, known) == []
    assert mod.invented("articolo 19-quater delle disposizioni di coordinamento e transitorie",
                        q, known) == []
    assert mod.invented("artt. 53-bis e 53-quater del codice della nautica da diporto",
                        q, known) == []


def test_the_script_prints_counts_only(tmp_path, capsys):
    mod = _load()
    norms = tmp_path / "legislazione_x.chunks.jsonl"
    norms.write_text("".join(json.dumps(_rec("codice_civile", n)) + "\n" for n in (1, 2, 3)),
                      encoding="utf-8")
    ans = tmp_path / "answers-m.jsonl"
    ans.write_text("".join(json.dumps(r) + "\n" for r in [
        {"question": "Che cosa prevede l'art. 2 del codice civile?",
         "answer": "Art. 2 e art. 9 c.c."},
        {"question": "Che cosa prevede l'art. 3 del codice civile?",
         "answer": "L'art. 3 c.c. ..."}]), encoding="utf-8")
    assert mod.main([str(ans), "--norms", str(norms)]) == 0
    out = capsys.readouterr().out
    assert "answers-m.jsonl: 2 answers | citing an article that does not exist 1 (50.0%) | " \
           "such citations 1" in out
    assert "Art. 2 e art. 9" not in out


def test_show_prints_the_flagged_citation_with_its_context(tmp_path, capsys):
    mod = _load()
    norms = tmp_path / "legislazione_x.chunks.jsonl"
    norms.write_text("".join(json.dumps(_rec("codice_civile", n)) + "\n" for n in (1, 2, 3)),
                      encoding="utf-8")
    ans = tmp_path / "answers-m.jsonl"
    ans.write_text(json.dumps({"question": "Che cosa prevede l'art. 2 del codice civile?",
                               "answer": "Art. 2 e art. 9 c.c. sul punto."}) + "\n",
                      encoding="utf-8")
    assert mod.main([str(ans), "--norms", str(norms), "--show", "5"]) == 0
    assert "    codice_civile art. 9 | art. 9 c.c. sul punto." in capsys.readouterr().out


def test_show_refuses_held_out_answers(tmp_path):
    mod = _load()
    norms = tmp_path / "legislazione_x.chunks.jsonl"
    norms.write_text(json.dumps(_rec("codice_civile", 1)) + "\n", encoding="utf-8")
    ans = tmp_path / "answers-m-heldout.jsonl"
    ans.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit):
        mod.main([str(ans), "--norms", str(norms), "--show", "3"])

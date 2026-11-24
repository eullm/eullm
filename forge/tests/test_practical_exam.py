"""The practical-questions exam (roadmap 1.1): groups, parsing, items, the script."""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

import pytest

from eullm_forge.eval import load_eval_set
from eullm_forge.eval.norm_exam import Article, rubric_v3
from eullm_forge.eval.paired import kind_of
from eullm_forge.eval.practical import (
    PracticalRejected,
    groups,
    heldout,
    messages,
    parse_practical,
    references,
    to_item,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_practical_exam.py"
BODY = " Il testo prosegue con le condizioni e i termini previsti dalla legge per il caso." * 3


def _art(code, num, text):
    return Article(code, str(num), f"Art. {num}.\n{text}{BODY}")


@pytest.fixture
def known():
    arts = [
        _art("codice_civile", 1453, "Nei contratti con prestazioni corrispettive si applica "
             "l'articolo 1455 e, per il danno, l'art. 2043 del codice civile."),
        _art("codice_civile", 1455, "Il contratto non si può risolvere se l'inadempimento ha "
             "scarsa importanza."),
        _art("codice_civile", 2043, "Qualunque fatto doloso o colposo obbliga al risarcimento."),
        _art("codice_procedura_civile", 163, "La domanda si propone con citazione; si veda "
             "l'art. 1453 del codice civile."),
        _art("codice_penale", 624, "Chiunque si impossessa della cosa mobile altrui."),
    ]
    return {(a.code, a.number): a for a in arts}


def test_references_follow_the_law_and_its_codes(known):
    assert references(known[("codice_civile", "1453")], known) == \
        [("codice_civile", "1455"), ("codice_civile", "2043")]
    assert references(known[("codice_procedura_civile", "163")], known) == \
        [("codice_civile", "1453")]
    assert references(known[("codice_penale", "624")], known) == []


def test_groups_hold_an_article_and_what_it_refers_to(known):
    out = groups(known, per_code=10, rng=random.Random(0), neighbours=False)
    heads = {(g[0].code, g[0].number) for g in out}
    assert heads == {("codice_civile", "1453"), ("codice_procedura_civile", "163")}
    assert all(len(g) >= 2 for g in out)
    # an article trained on is in no group, as head or as reference
    out = groups(known, 10, random.Random(0), neighbours=False,
                 exclude=frozenset({("codice_civile", "1455"), ("codice_civile", "2043")}))
    assert {(g[0].code, g[0].number) for g in out} == {("codice_procedura_civile", "163")}


def _reply(**kw):
    obj = {"domanda": "Il mio fornitore non ha consegnato la merce: posso sciogliere il "
                      "contratto e chiedere i danni?",
           "articoli": [1, 3],
           "risposta": "Sì, se l'inadempimento non è di scarsa importanza si può chiedere la "
                       "risoluzione (art. 1453 c.c.) e il risarcimento del danno (art. 2043 "
                       "c.c.), con l'onere di provare il danno.",
           "rubrica": "Deve dire che la risoluzione richiede un inadempimento importante."}
    obj.update(kw)
    return json.dumps(obj)


def test_a_good_reply_is_an_item_graded_by_its_own_rubric(known):
    g = [known[("codice_civile", "1453")], known[("codice_civile", "1455")],
         known[("codice_civile", "2043")]]
    parsed = parse_practical(_reply(), g)
    assert [a.number for a in parsed["articoli"]] == ["1453", "2043"]
    item = to_item(g, parsed, "dev", "t")
    assert item.id == "norm-pratica-codice_civile-1453" and kind_of(item.id) == "pratica"
    assert "art. 1453 c.c.; art. 2043 c.c." in item.rubric
    assert rubric_v3(item.id, item.rubric, item.reference) == item.rubric
    assert item.metadata["articoli"] == ["codice_civile:1453", "codice_civile:2043"]
    assert "Testi degli articoli necessari" in item.reference
    assert "[1] Codice civile, art. 1453" in messages(g)[1]["content"]


@pytest.mark.parametrize("reply,reason", [
    ("niente json", "no_json"),
    (json.dumps({"domanda": ""}), "no_question"),
    (_reply(domanda="Che cosa prevede l'art. 1453 del codice civile in caso di inadempimento?"),
     "names_article"),
    (_reply(articoli=[7]), "bad_articles"),
    (_reply(articoli=[2]), "answer_misses_article"),      # cites neither 1455 nor any declared
    (_reply(risposta="Breve."), "length"),
    (_reply(domanda="Il contratto di [PERSONA_1] può essere risolto per inadempimento?"),
     "placeholder"),
])
def test_unusable_replies_are_refused(known, reply, reason):
    g = [known[("codice_civile", "1453")], known[("codice_civile", "1455")],
         known[("codice_civile", "2043")]]
    with pytest.raises(PracticalRejected) as e:
        parse_practical(reply, g)
    assert e.value.reason == reason


def test_the_split_is_a_hash_not_the_draw(known):
    a = known[("codice_civile", "1453")]
    assert heldout(a, 0.34) == heldout(a, 0.34)
    assert heldout(a, 1.0) and not heldout(a, 0.0)


def _load():
    spec = importlib.util.spec_from_file_location("make_practical_exam", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_script_writes_both_files_and_prints_counts_only(tmp_path, known, monkeypatch,
                                                             capsys):
    mod = _load()
    norms = tmp_path / "legislazione_x.chunks.jsonl"
    norms.write_text("".join(json.dumps({"code": a.code, "article_num": a.number,
                                         "text": a.text, "chunk_index": 0},
                                        ensure_ascii=False) + "\n"
                             for a in known.values()), encoding="utf-8")
    monkeypatch.setattr(mod, "ask", lambda url, msgs, mt, t: _reply(articoli=[1]))
    dev, held = tmp_path / "dev.jsonl", tmp_path / "held.jsonl"
    args = [str(norms), "--url", "http://x", "--out-dev", str(dev), "--out-heldout", str(held),
            "--parallel", "1"]
    assert mod.main(args) == 0
    items = load_eval_set(dev) + load_eval_set(held)
    out = capsys.readouterr().out
    # the c.c. group's answer cites what it needs; the c.p.c. one does not cite art. 163
    assert [it.id for it in items] == ["norm-pratica-codice_civile-1453"]
    assert "answer_misses_article: 1" in out
    assert "fornitore" not in out
    with pytest.raises(SystemExit):        # drawn once
        mod.main(args)


def test_training_prompts_leave_out_every_article_of_a_practical_item(tmp_path, known):
    spec = importlib.util.spec_from_file_location(
        "make_grpo_prompts", SCRIPT.parent / "make_grpo_prompts.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    g = [known[("codice_civile", "1453")], known[("codice_civile", "1455")],
         known[("codice_civile", "2043")]]
    item = to_item(g, parse_practical(_reply(), g), "dev")
    exam = tmp_path / "practical-dev.jsonl"
    from eullm_forge.eval import save_eval_set
    save_eval_set([item], exam)
    assert mod.exam_articles([exam]) == {("codice_civile", "1453"), ("codice_civile", "1455"),
                                         ("codice_civile", "2043")}


def test_an_article_that_refers_to_none_is_paired_with_the_next(known):
    k = dict(known)
    k[("codice_civile", "2044")] = _art("codice_civile", 2044, "Non è responsabile chi cagiona "
                                         "il danno per legittima difesa.")
    out = groups(k, per_code=10, rng=random.Random(0))
    pairs = {tuple(a.number for a in g) for g in out}
    assert ("2043", "2044") in pairs            # 2043 refers to none: its neighbour
    assert ("1453", "1455", "2043") in pairs    # references still make their groups
    # a code's quota takes the reference groups first
    first = groups(k, per_code=1, rng=random.Random(0))
    assert [tuple(a.number for a in g) for g in first if g[0].code == "codice_civile"] == \
        [("1453", "1455", "2043")]


def test_the_needed_articles_are_those_declared_and_cited(known):
    g = [known[("codice_civile", "1453")], known[("codice_civile", "1455")],
         known[("codice_civile", "2043")]]
    # declared 1, 2 and 3; the answer cites 1453 and 2043 only
    parsed = parse_practical(_reply(articoli=[1, 2, 3]), g)
    assert [a.number for a in parsed["articoli"]] == ["1453", "2043"]


def test_a_citation_in_one_code_is_not_a_citation_of_another(known):
    """Art. 2 c.c. and art. 2 Cost. share a number: the code is part of the
    citation, or the rubric demands an article the teacher's answer never
    cited (and the reference includes its text)."""
    civ2 = _art("codice_civile", 2, "La maggiore età è fissata al diciottesimo anno.")
    cost2 = _art("costituzione", 2, "La Repubblica riconosce i diritti inviolabili.")
    g = [civ2, cost2]
    parsed = parse_practical(_reply(
        articoli=[1, 2],
        risposta="La maggiore età è fissata al diciottesimo anno (art. 2 c.c.); "
                 "il fondamento è il riconoscimento dei diritti inviolabili.",
        rubrica="Deve indicare il diciottesimo anno e i diritti inviolabili."), g)
    assert [(a.code, a.number) for a in parsed["articoli"]] == [("codice_civile", "2")]
    item = to_item(g, parsed, "dev")
    assert "art. 2 Cost." not in item.rubric
    assert item.metadata["articoli"] == ["codice_civile:2"]


def test_a_bare_number_shared_by_two_codes_names_neither(known):
    civ2 = _art("codice_civile", 2, "La maggiore età è fissata al diciottesimo anno.")
    cost2 = _art("costituzione", 2, "La Repubblica riconosce i diritti inviolabili.")
    g = [civ2, cost2]
    with pytest.raises(PracticalRejected) as exc:
        parse_practical(_reply(
            articoli=[1, 2],
            risposta="La maggiore età è fissata al diciottesimo anno (art. 2), "
                     "insieme ai diritti inviolabili riconosciuti alla persona "
                     "dalla Repubblica e garantiti nei confronti dei poteri pubblici.",
            rubrica="Deve indicare il diciottesimo anno e i diritti inviolabili."), g)
    assert exc.value.reason == "answer_misses_article"


def _load_prompts():
    spec = importlib.util.spec_from_file_location(
        "make_practical_prompts", SCRIPT.parent / "make_practical_prompts.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_training_prompts_are_judged_rows_beside_the_mixed_ones(tmp_path, known, monkeypatch,
                                                                 capsys):
    from eullm_forge.rl import JUDGED_TYPES
    assert "pratica" in JUDGED_TYPES
    mod = _load_prompts()
    extra = _art("codice_civile", 2044, "Non risponde chi agisce per legittima difesa.")
    arts = {**known, ("codice_civile", "2044"): extra}
    norms = tmp_path / "legislazione_x.chunks.jsonl"
    norms.write_text("".join(json.dumps({"code": a.code, "article_num": a.number,
                                         "text": a.text, "chunk_index": 0},
                                        ensure_ascii=False) + "\n"
                             for a in arts.values()), encoding="utf-8")
    # the c.p.c. group's articles are an exam's: no prompt is written from them
    g = [known[("codice_procedura_civile", "163")], known[("codice_civile", "1453")]]
    exam = tmp_path / "practical-dev.jsonl"
    from eullm_forge.eval import save_eval_set
    save_eval_set([to_item(g, {"domanda": "q", "risposta": "r", "rubrica": "x",
                               "articoli": [g[0]]}, "dev")], exam)
    mix = tmp_path / "prompts-judge.jsonl"
    mix.write_text(json.dumps({"id": "m1", "tipo": "termine", "prompt": []}) + "\n",
                   encoding="utf-8")
    monkeypatch.setattr(mod, "ask", lambda url, msgs, mt, t: _reply(articoli=[1]))
    out = tmp_path / "prompts-practical.jsonl"
    assert mod.main(["--norms", str(norms), "--exclude-exam", str(exam), "--url", "http://x",
                     "--parallel", "1", "--mix", str(mix), "--out", str(out)]) == 0
    rows = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()]
    practical = [r for r in rows if r["tipo"] == "pratica"]
    assert {r["id"] for r in rows} - {r["id"] for r in practical} == {"m1"}
    # 163 and 1453 are an exam's: the one group left is 2043 with its neighbour 2044
    assert [(r["code"], r["articolo"]) for r in practical] == [("codice_civile", "2043")]
    for r in practical:
        assert r["question"] and r["reference"] and r["rubric"]
        assert "Domanda: " in r["prompt"][0]["content"]
    assert "fornitore" not in capsys.readouterr().out

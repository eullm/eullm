"""make_concept_prompts.py: questions by concept, kept only where the texts answer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_concept_prompts.py"
BODY = " Il presente articolo contiene disposizioni di dettaglio sufficienti a una risposta." * 3


def load():
    spec = importlib.util.spec_from_file_location("make_concept_prompts", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rec(n, heading, text):
    return {"code": "codice_penale", "article_num": "", "chunk_index": 0,
            "text": f"Art. {n}. \n \n ({heading}). \n \n {text}{BODY}"}


RECORDS = [
    rec("99", "Recidiva", "Chi, dopo essere stato condannato per un delitto non colposo, ne "
        "commette un altro, puo' essere sottoposto a un aumento di un terzo della pena."),
    rec("52", "Difesa legittima", "Non e' punibile chi ha commesso il fatto per esservi stato "
        "costretto dalla necessita' di difendere un diritto proprio od altrui."),
    rec("1", "Disposizioni generali", "Nessuno puo' essere punito per un fatto che non sia "
        "espressamente preveduto come reato dalla legge."),
]


def test_a_heading_names_a_concept_unless_it_is_generic():
    mod = load()
    assert mod.concept("Recidiva") == "Recidiva"
    assert mod.concept("Disposizioni generali") is None
    assert mod.concept("Ambito di applicazione") is None
    assert mod.concept("ab") is None


def test_questions_by_concept_are_judged_on_the_article_they_retrieve(tmp_path):
    norms = tmp_path / "legislazione_cp.chunks.jsonl"
    norms.write_text("".join(json.dumps(r) + "\n" for r in RECORDS), encoding="utf-8")
    out = tmp_path / "concept.jsonl"
    assert load().main(["--norms", str(norms), "-k", "2", "--out", str(out)]) == 0
    rows = [json.loads(line) for line in out.open()]
    assert sorted(r["articolo"] for r in rows) == ["52", "99"]      # not the generic one
    for r in rows:
        assert r["tipo"] == "contenuto" and r["code"] == "codice_penale"
        assert r["question"] in r["prompt"][0]["content"]
        assert "Chi, dopo essere stato" in r["reference"] or "Non e' punibile" in r["reference"]
        assert "Parziale" in r["rubric"]
        # its own article is among the texts given
        assert f"art. {r['articolo']}" in r["prompt"][0]["content"]


def test_a_question_whose_texts_lack_the_article_is_dropped():
    mod = load()

    class Index:
        def search(self, q, k):
            return [{"code": "codice_civile", "text": "Art. 2. altro"}]

        def articles_of(self, r):
            return ["2"]

    a = type("A", (), {"code": "codice_penale", "number": "99", "text": "x"})()
    assert mod.prompt_row(a, "Che cosa si intende per «Recidiva»?", Index(), 3) is None

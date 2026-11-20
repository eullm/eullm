"""score_closed_book.py: the count of answers that point to texts not given."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_closed_book.py"
spec = importlib.util.spec_from_file_location("score_closed_book", SCRIPT)
scb = importlib.util.module_from_spec(spec)
sys.modules["score_closed_book"] = scb
spec.loader.exec_module(scb)


def test_answers_that_appeal_to_missing_texts_are_told_apart():
    for a in ("Il testo dell'articolo non è tra quelli forniti.",
              "Nei testi forniti non trovo l'art. 12 c.p.",
              "L'art. 2875 c.c. non è presente nella raccolta.",
              "Non dispongo del testo dell'articolo richiesto.",
              "Dai documenti a disposizione non risulta il termine."):
        assert scb.points_to_texts(a), a
    for a in ("L'art. 624 c.p. punisce il furto con la reclusione da sei mesi a tre anni.",
              "L'articolo 2875 del codice civile non esiste.",
              "Il termine è di trenta giorni dalla notifica.", ""):
        assert not scb.points_to_texts(a), a


def test_items_drawn_by_the_old_mistakes_are_left_out(tmp_path):
    import json

    def rec(n, heading, text):
        return {"code": "codice_penale", "article_num": "", "chunk_index": 0,
                "text": f"Art. {n}. \n \n ({heading}). \n \n {text}"}

    norms = tmp_path / "norms.jsonl"
    norms.write_text("".join(json.dumps(r) + "\n" for r in [
        rec("600", "Schiavitu'", "Chiunque riduce una persona in schiavitu' e' punito con la "
            "reclusione da otto a venti anni.\nCOMMA ABROGATO DALLA L. 2 LUGLIO 2010, N. 108."),
        rec("594", "Ingiuria", "Articolo abrogato dal d.lgs. 15 gennaio 2016, n. 7."),
        rec("507", "Boicottaggio", "Chiunque induce altri a non lavorare e' punito con la "
            "reclusione fino a tre anni. Se concorrono fatti di violenza, si applica la "
            "reclusione da due a sei anni."),
        rec("624", "Furto", "Chiunque s'impossessa della cosa mobile altrui e' punito con la "
            "reclusione da sei mesi a tre anni. Il delitto e' punibile a querela della "
            "persona offesa."),
    ]), encoding="utf-8")

    def item(i, kind, n, **meta):
        return i, {"id": i, "keywords": meta.pop("keywords", []),
                   "metadata": {"tipo": kind, "code": "codice_penale", "articolo": n, **meta}}

    items = dict([item("a", "abrogato", "600"), item("b", "abrogato", "594"),
                  item("c", "pena", "507", high=[6, "anni"]),
                  item("d", "pena", "624", high=[3, "anni"]),
                  item("e", "procedibilita", "624", keywords=["querela"]),
                  item("f", "termine", "1")])
    assert scb.stale_items(items, [norms]) == {"a", "c"}

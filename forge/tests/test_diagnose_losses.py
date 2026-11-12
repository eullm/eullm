"""diagnose_losses.py: each loss tagged by what is wrong with it."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_losses.py"
spec = importlib.util.spec_from_file_location("diagnose_losses", SCRIPT)
dl = importlib.util.module_from_spec(spec)
sys.modules["diagnose_losses"] = dl
spec.loader.exec_module(dl)

REF = ("Art. 2861 c.c. Testo integrale dell'articolo: la dichiarazione deve essere fatta "
       "non oltre dieci giorni; il certificato è notificato entro cinque giorni.")
RIGHT = ("Secondo l'art. 2861 c.c. la dichiarazione va fatta entro dieci giorni e il "
         "certificato notificato entro cinque giorni dalla sua data, come dice il testo.")


def row(i, answer, grade, kind="termine", **kw):
    return {"id": f"norm-{kind}-codice_civile-{i}", "answer": answer, "grade": grade,
            "question": f"Quale termine prevede l'art. {i} del codice civile?",
            "reference": REF if kind == "termine" else f"Art. {i}. Testo.",
            "rubric": "termine di 5 giorni" if kind == "termine" else "", "why": "x", **kw}


def test_each_loss_gets_the_tag_of_its_error():
    base = {r["id"]: r for r in (row(1, RIGHT, "correct"), row(2, RIGHT, "correct"),
                                 row(3, RIGHT, "correct"), row(4, RIGHT, "correct"),
                                 row(5, RIGHT, "correct", kind="contenuto"),
                                 row(6, RIGHT, "correct"))}
    model = {r["id"]: r for r in (
        row(1, "Secondo l'art. 2861 c.c. il termine è di dieci giorni.", "wrong"),
        row(2, "Secondo l'art. 2861 c.c. il termine è di quaranta giorni.", "wrong"),
        row(3, "Nei testi forniti non è presente la risposta alla domanda.", "wrong"),
        row(4, "", "wrong"),
        row(5, "La regola è nell'art. 1453 c.c., che disciplina la risoluzione, "
               "come ricordato più volte dalla giurisprudenza in materia.", "wrong",
            kind="contenuto"),
        row(6, RIGHT, "correct"))}
    ids, total, by_kind = dl.diagnose(model, base)
    assert len(ids) == 5
    got = {i: dl.tags(model[i], base[i]) for i in ids}
    assert "altro termine dell'articolo" in got["norm-termine-codice_civile-1"]
    assert "piu' corta" in got["norm-termine-codice_civile-1"]
    assert "termine non nell'articolo" in got["norm-termine-codice_civile-2"]
    assert "si astiene" in got["norm-termine-codice_civile-3"]
    assert got["norm-termine-codice_civile-4"] == ["vuota"]
    assert "articolo non dato" in got["norm-contenuto-codice_civile-5"]
    _, wins, _ = dl.diagnose(model, base, wins=True)
    assert sum(wins.values()) == 0


def test_dump_refuses_the_held_out_exam(tmp_path, capsys):
    d = tmp_path / "norm-exam-answers-v2"
    d.mkdir()
    f = d / "a.graded.jsonl"
    f.write_text(json.dumps(row(1, RIGHT, "correct")) + "\n")
    with pytest.raises(SystemExit):
        dl.main([str(f), "--baseline", str(f), "--dump", str(tmp_path / "x.txt")])
    assert dl.main([str(f), "--baseline", str(f)]) == 0

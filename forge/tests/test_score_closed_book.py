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

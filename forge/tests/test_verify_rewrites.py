"""verify_rewrites.py: planted errors, a checker measured on them, the corpus estimated."""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_rewrites.py"
spec = importlib.util.spec_from_file_location("verify_rewrites", SCRIPT)
vr = importlib.util.module_from_spec(spec)
sys.modules["verify_rewrites"] = vr
spec.loader.exec_module(vr)

ART = ("Art. 1.\n\n(Termine).\n\nIl creditore può chiedere il pagamento entro trenta giorni. "
       "Il debitore non è tenuto oltre.")


def test_each_mutation_changes_one_thing():
    text = "Il creditore può chiedere il pagamento entro 30 giorni, il debitore non paga."
    m = vr.mutations(text, random.Random(0))
    assert set(m) == {"numero", "modale", "negazione", "parte"}
    assert "60 giorni" in m["numero"]
    assert "deve chiedere" in m["modale"]
    assert "debitore paga" in m["negazione"]
    assert m["parte"].startswith("Il debitore può")
    assert vr.mutations("Nessuna parola utile qui.", random.Random(0)) == {}


def test_lexical_flags_see_a_modal_the_article_does_not_have():
    assert vr.lexical_flags("Il creditore deve chiedere il pagamento.", ART)
    assert vr.lexical_flags("Il creditore può chiedere il pagamento.", ART) == []


def test_verdicts_are_parsed_and_unknown_replies_dropped():
    assert vr.verdict_of("VERDETTO: NON FEDELE\nmotivo") == "non fedele"
    assert vr.verdict_of("verdetto: fedele") == "fedele"
    assert vr.verdict_of("non saprei") is None


def test_prepare_check_report_end_to_end(tmp_path):
    norms = tmp_path / "legislazione_codice_civile.chunks.jsonl"
    norms.write_text(json.dumps({"code": "codice_civile", "text": ART}) + "\n")
    rows = tmp_path / "rewrites.jsonl"
    rows.write_text(json.dumps({"id": "codice_civile|1|parafrasi|0", "code": "codice_civile",
                                "articolo": "1", "forma": "parafrasi", "text":
                                "L'art. 1 dice che il creditore può chiedere il pagamento "
                                "entro trenta giorni e che il debitore non è tenuto oltre."})
                    + "\n")
    items = vr.prepare([norms], [rows], per_form=5, seed=0)
    kinds = {it["kind"] for it in items}
    assert "originale" in kinds and {"numero", "modale", "negazione", "parte"} <= kinds

    def perfect(url, msgs):    # a checker that is never fooled
        text = msgs[1]["content"]
        bad = any(k in text for k in ("novanta", "deve chiedere", "debitore è tenuto",
                                      "debitore può chiedere", "il creditore non"))
        return "VERDETTO: NON FEDELE\nx" if bad else "VERDETTO: FEDELE\nx"

    out = tmp_path / "verdicts-perfect.jsonl"
    assert vr.check(items, ["u"], out, "perfect", ask_fn=perfect) == 0
    assert vr.check(items, ["u"], out, "perfect", ask_fn=perfect) == 0   # resumes: nothing new
    assert len(out.read_text().splitlines()) == len(items)
    text = vr.report(items, [out])
    assert "perfect: recall 100%" in text and "flags 0% of 1 corpus rows" in text

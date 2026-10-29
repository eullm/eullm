"""The case-law part of the memory corpus: principles rewritten, checked, chained."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from eullm_forge.caselaw.rewrites import check, messages, parse

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

CARD = {"id": "snciv/2023000001", "court": "snciv", "materia": "responsabilità contrattuale",
        "principi": ["In tema di inadempimento, il creditore deve provare la fonte del suo "
                     "diritto e il termine di scadenza, limitandosi ad allegare "
                     "l'inadempimento della controparte.",
                     "La prescrizione decennale dell'art. 2946 c.c. decorre dal giorno in cui "
                     "il diritto può essere fatto valere."],
        "norme": ["art. 1218 c.c.", "art. 2946 c.c."], "esito": "rigetto"}

GOOD = {"principi": [
    {"riformulazione": "Chi agisce per l'inadempimento prova la fonte del diritto e la scadenza, "
                       "e allega soltanto che la controparte non ha adempiuto (art. 1218 c.c.).",
     "domanda": "Che cosa deve provare il creditore che lamenta un inadempimento?",
     "risposta": "La fonte del diritto e la scadenza; l'inadempimento lo allega soltanto."},
    {"riformulazione": "Il termine decennale dell'art. 2946 c.c. corre da quando il diritto "
                       "può essere esercitato.",
     "domanda": "Da quando decorre la prescrizione ordinaria?",
     "risposta": "Dal giorno in cui il diritto può essere fatto valere, art. 2946 c.c."}]}


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_request_shows_the_principles_and_no_ruling_number():
    content = messages(CARD)[1]["content"]
    assert "1. In tema di inadempimento" in content and "2. La prescrizione" in content
    assert "dalla Corte di cassazione civile" in content and "art. 1218 c.c." in content
    assert "2023000001" not in content


def test_a_rewrite_that_adds_nothing_is_kept_in_two_forms():
    kept, refused = parse(CARD, json.dumps(GOOD))
    assert refused == [] and [(i, f) for i, f, _ in kept] == [
        (0, "riformulazione"), (0, "domanda"), (1, "riformulazione"), (1, "domanda")]
    assert kept[1][2].startswith("D: Che cosa deve provare") and "\nR: La fonte" in kept[1][2]


def test_what_the_principle_does_not_hold_is_refused():
    src = CARD["principi"][1] + " art. 2946 c.c."
    assert check(src, "Il termine decennale dell'art. 2946 c.c. corre dall'esercizio.") is None
    assert check(src, "Il termine dell'art. 2946 c.c., come l'art. 2947 c.c., corre.") \
        == "other_article"
    assert check(src, "Il termine di prescrizione è di 1.000 giorni dall'esercizio.") == "number"
    assert check(src, "La parte [PERSONA_1] ha diritto al risarcimento del danno subito.") \
        == "placeholder"
    bad = json.loads(json.dumps(GOOD))
    bad["principi"][1]["riformulazione"] = "Lo dice anche l'art. 2935 c.c., che va letto insieme."
    kept, refused = parse(CARD, json.dumps(bad))
    assert refused == [(1, "riformulazione:other_article")] and len(kept) == 3
    assert parse(CARD, json.dumps({"principi": GOOD["principi"][:1]})) == ([], [(-1, "count")])
    assert parse(CARD, "non è JSON") == ([], [(-1, "bad_json")])


def test_the_script_asks_the_cards_that_may_leave_and_resumes(tmp_path, monkeypatch, capsys):
    cards = tmp_path / "schede-civ.jsonl"
    sensitive = dict(CARD, id="snciv/2023000002", materia="separazione dei coniugi")
    cards.write_text("".join(json.dumps(c) + "\n" for c in (CARD, sensitive)))
    # the first Consiglio di Stato cards carry no court
    cds = tmp_path / "schede.jsonl"
    cds.write_text(json.dumps({k: v for k, v in CARD.items() if k != "court"}
                              | {"id": "cds/2020000001", "oggetto": "appalti pubblici"}) + "\n")
    (tmp_path / "schede-civ.rejects.jsonl").write_text(json.dumps({"id": "x", "reason": "r"}))
    mod = load("principle_rewrites")
    asked = []

    def fake_ask(url, msgs, max_tokens, temperature, seed, timeout=600.0):
        asked.append(msgs[1]["content"])
        return {"choices": [{"message": {"content": json.dumps(GOOD)},
                             "finish_reason": "stop"}], "usage": {}}

    monkeypatch.setattr(mod, "ask", fake_ask)
    out = tmp_path / "memory" / "principi.jsonl"
    args = ["--cards", str(cards), str(tmp_path / "schede-civ.rejects.jsonl"), str(cds),
            "--courts", "cds,snciv",
            "--out", str(out), "--url", "http://x", "--parallel", "1"]
    assert mod.main(args) == 0
    rows = [json.loads(line) for line in out.open()]
    assert len(asked) == 2 and len(rows) == 8                # the family case is not asked
    assert {r["card"] for r in rows} == {"snciv/2023000001", "cds/2020000001"}
    assert {r["court"] for r in rows} == {"snciv", "cds"} and rows[0]["pv"] == 1
    assert any("dal Consiglio di Stato" in a for a in asked)
    with out.open("a") as f:
        f.write('{"id": "half writ')                         # a sibling's line half written
    capsys.readouterr()
    mod.main(args)
    printed = capsys.readouterr().out
    assert "2 done already" in printed and len(asked) == 2
    assert "sensitive" in printed and "inadempimento" not in printed

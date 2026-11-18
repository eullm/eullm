"""export_clean_cards.py: only cards with nothing of the case leave, under made-up references."""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_clean_cards.py"


def _load():
    spec = importlib.util.spec_from_file_location("export_clean_cards", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _card(rid, oggetto="Appalti pubblici - esclusione", principio=None, **kw):
    return {"id": rid, "year": 2021, "numero": 1234, "nrg": "202100567",
            "sezione": "Sezione Quinta", "data": "2021-05-03", "esito_openga": "Respinto",
            "oggetto": oggetto, "teacher": "qwen",
            "principi": [principio or "L'esclusione dalla gara per grave illecito professionale "
                         "richiede una motivazione specifica sull'affidabilità dell'operatore."],
            "norme": ["art. 80, comma 5, d.lgs. n. 50/2016", "l.r. Regione Lombardia n. 12/2005"],
            "esito": "rigetto", "materia": "appalti pubblici - esclusione dalla gara",
            "domande_ricerca": ["Quando la stazione appaltante può escludere un operatore?"] * 3,
            "domande_esame": [{"domanda": "d", "risposta": "r", "rubrica": "x"}] * 2, **kw}


def test_only_clean_training_cards_leave():
    mod = _load()
    rows = [
        _card("ok-1"),
        _card("ok-1"),                                          # duplicate
        _card("dev-1"),                                         # development ruling
        {"id": "r-1", "reason": "placeholder", "v": 2},         # a refusal row
        _card("s-1", oggetto="Informativa antimafia interdittiva"),
        _card("s-2", oggetto="Rinnovo permesso di soggiorno"),
        _card("s-3", materia="tutela della salute - vaccinazioni"),
        _card("n-1", oggetto=""),
        _card("p-1", principio="Il ricorso di [PERSONA_1] contro il diniego è infondato "
                               "e va respinto."),
        _card("c-1", principio="Il Comune di Bracciano doveva motivare il diniego del permesso "
                               "di costruire in modo specifico."),
        _card("t-1", principio="Il T.A.R. per il Lazio ha correttamente respinto la domanda "
                               "di annullamento dell'aggiudicazione."),
        _card("a-1", principio="La sanzione di euro 12.500 irrogata all'operatore era "
                               "sproporzionata rispetto alla violazione."),
        _card("b-1", principio="La Edilstrade S.r.l. non aveva dichiarato la condanna del suo "
                               "direttore tecnico, il che giustifica l'esclusione."),
    ]
    out, stats = mod.export(rows, {"dev-1"}, random.Random(0))
    assert stats["written"] == 1 and len(out) == 1
    assert stats["duplicate"] == 1 and stats["dev"] == 1 and stats["refusal_rows"] == 1
    assert stats["dropped:sensitive_oggetto"] == 2
    assert stats["dropped:sensitive_materia"] == 1
    assert stats["dropped:no_oggetto"] == 1
    assert stats["dropped:placeholder"] == 1
    assert stats["dropped:named_body"] == 3
    assert stats["dropped:amount"] == 1


def test_what_is_written_points_at_no_real_ruling():
    mod = _load()
    out, _ = mod.export([_card(f"id-{i}") for i in range(50)], {"x"}, random.Random(1))
    for card in out:
        assert set(card) == {"id", "estremi", *mod.CONTENT}
        assert card["id"].startswith("card-")
        e = card["estremi"]
        assert 50000 <= e["numero"] <= 59999 and 2017 <= e["anno"] <= 2024
        assert e["data"].startswith(str(e["anno"]))
        flat = json.dumps(card)
        for real in ("id-", "202100567", "Sezione Quinta", "2021-05-03", "Respinto", "qwen",
                     "Appalti pubblici - esclusione"):
            assert real not in flat
    # a regional law in the norms is the law, not a party
    assert "Regione Lombardia" in json.dumps(out[0], ensure_ascii=False)


def test_the_script_refuses_without_the_dev_split_and_prints_counts_only(tmp_path, capsys):
    mod = _load()
    cards = tmp_path / "schede.jsonl"
    cards.write_text(json.dumps(_card("ok-1"), ensure_ascii=False) + "\n", encoding="utf-8")
    dev = tmp_path / "dev.txt"
    dev.write_text("", encoding="utf-8")
    out = tmp_path / "clean.jsonl"
    try:
        mod.main([str(cards), "--dev-ids", str(dev), "--out", str(out)])
        raise AssertionError("an empty dev split was accepted")
    except SystemExit:
        pass
    dev.write_text("dev-1\n", encoding="utf-8")
    assert mod.main([str(cards), "--dev-ids", str(dev), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "1 written" in printed and "grave illecito" not in printed
    assert len(out.read_text(encoding="utf-8").splitlines()) == 1


def _cass(rid, principio=None, materia="responsabilità contrattuale - onere della prova", **kw):
    card = _card(rid, principio=principio or (
        "In tema di inadempimento contrattuale il creditore deve provare la fonte del suo "
        "diritto, mentre spetta al debitore provare l'esatto adempimento."), **kw)
    card.update(court="snciv", oggetto=None, sezione="", materia=materia,
                norme=["art. 1218 c.c.", "art. 2697 c.c."])
    return card


def test_cassazione_civil_cards_leave_without_oggetto_but_not_the_sensitive_ones():
    mod = _load()
    rows = [
        _cass("snciv/2022/101"),
        _cass("snciv/2022/102", principio="La clausola penale ha funzione di liquidazione "
                                          "preventiva del danno e può essere ridotta dal giudice."),
        _cass("snciv/2022/103", materia="separazione personale dei coniugi - addebito"),
        _cass("snciv/2022/104", principio="Nel risarcimento del danno biologico il giudice deve "
                                          "personalizzare la liquidazione tabellare."),
        _cass("snciv/2022/105", principio="L'affidamento del minore va deciso nel suo esclusivo "
                                          "interesse morale e materiale."),
        _card("cds/2021000001"),                                # another court's card
        {**_cass("snpen/2022/7"), "court": "snpen"},
    ]
    out, stats = mod.export(rows, set(), random.Random(0), "snciv")
    assert stats["written"] == 2 and stats["other_court"] == 2
    assert stats["dropped:sensitive_materia"] == 1 and stats["dropped:sensitive_content"] == 2
    assert all(c["estremi"]["autorita"] == "Cass. civ." for c in out)
    assert all(c["estremi"]["sezione"] in mod.CASS_SECTIONS for c in out)
    assert all(2021 <= c["estremi"]["anno"] <= 2024 and 50000 <= c["estremi"]["numero"] < 60000
               for c in out)
    assert not any("snciv/" in json.dumps(c) for c in out)


def test_criminal_rulings_are_never_exported(tmp_path):
    import pytest
    mod = _load()
    cards = tmp_path / "c.jsonl"
    cards.write_text(json.dumps(_cass("snpen/2022/7")) + "\n", encoding="utf-8")
    dev = tmp_path / "dev.txt"
    dev.write_text("x\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        mod.main([str(cards), "--dev-ids", str(dev), "--kind", "snpen",
                  "--out", str(tmp_path / "o.jsonl")])

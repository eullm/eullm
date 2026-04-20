"""Tests for the anonymisation module."""

from __future__ import annotations

import pytest

from eullm_forge.datasets.anonymize import (
    AnonymiserConfig,
    RedactionStats,
    anonymize_record,
    anonymize_text,
)


def test_redacts_codice_fiscale():
    text = "dell'avvocato COSTANZO GIULIO (CSTGLI71P23F839D) che rappresenta"
    out, stats = anonymize_text(text)
    assert "CSTGLI71P23F839D" not in out
    assert "[CODICE_FISCALE]" in out
    assert stats.codice_fiscale == 1


def test_redacts_partita_iva_only_with_context():
    # With context → redacted
    text1 = "la società con P.IVA 12345678901 ha presentato"
    out1, stats1 = anonymize_text(text1)
    assert "12345678901" not in out1
    assert stats1.partita_iva == 1

    # Without context (just 11 digits) → NOT redacted, to avoid nuking
    # case numbers / protocol IDs.
    text2 = "ricorso iscritto al n. 12345678901 del registro"
    out2, stats2 = anonymize_text(text2)
    assert "12345678901" in out2
    assert stats2.partita_iva == 0


def test_redacts_iban():
    text = "accredito su IBAN IT60X0542811101000000123456 intestato a"
    out, stats = anonymize_text(text)
    assert "IT60X0542811101000000123456" not in out
    assert "[IBAN]" in out
    assert stats.iban == 1


def test_redacts_email():
    text = "contatto: mario.rossi@example.com per info"
    out, stats = anonymize_text(text)
    assert "mario.rossi@example.com" not in out
    assert stats.email == 1


def test_redacts_birth_clause():
    text = (
        "ricorso proposto da CARLOMAGNO FRANCESCO, nato a MONTALBANO JONICO "
        "il 13/03/1972 avverso l'ordinanza"
    )
    out, stats = anonymize_text(text)
    assert "MONTALBANO JONICO" not in out
    assert "13/03/1972" not in out
    assert "[DATI_NASCITA]" in out
    assert stats.birth_clause == 1


def test_redacts_address():
    text = "domiciliato in ROMA, VIA GREGORIO VII n. 16, presso lo studio"
    out, stats = anonymize_text(text)
    assert "GREGORIO VII" not in out
    assert "[INDIRIZZO]" in out
    assert stats.address >= 1


def test_allcaps_name_heuristic_catches_party():
    text = "ricorso proposto da CARLOMAGNO FRANCESCO avverso la sentenza"
    out, stats = anonymize_text(text)
    assert "CARLOMAGNO FRANCESCO" not in out
    assert "[PERSONA]" in out
    assert stats.person_allcaps == 1


def test_allcaps_whitelist_keeps_courts_and_agencies():
    text = (
        "la CORTE SUPREMA di CASSAZIONE pronuncia sentenza; "
        "l'AGENZIA ENTRATE ricorre contro l'ordinanza del TRIBUNALE LIBERTA'"
    )
    out, stats = anonymize_text(text)
    assert "CORTE SUPREMA" in out
    assert "AGENZIA ENTRATE" in out
    assert "TRIBUNALE LIBERTA" in out
    assert stats.person_allcaps == 0


def test_disable_allcaps_via_config():
    text = "CARLOMAGNO FRANCESCO, nato a ROMA il 13/03/1972"
    cfg = AnonymiserConfig(redact_allcaps_names=False, redact_birth=False)
    out, stats = anonymize_text(text, config=cfg)
    assert "CARLOMAGNO FRANCESCO" in out
    assert stats.person_allcaps == 0
    assert stats.birth_clause == 0


def test_anonymize_record_preserves_metadata_and_adds_audit():
    rec = {
        "text": "ricorso di MARIO ROSSI (RSSMRA80A01H501Z) nato a ROMA il 01/01/1980",
        "source": "italgiure",
        "sentence_id": "snpen/2026/1",
        "metadata": {
            "ecli": "ECLI:IT:CASS:2026:1PEN",
            "presidente": "CATENA ROSSELLA",
        },
    }
    out = anonymize_record(rec)
    assert out["sentence_id"] == rec["sentence_id"]
    assert out["metadata"]["ecli"] == rec["metadata"]["ecli"]
    # Presidente is a public official — kept verbatim in metadata.
    assert out["metadata"]["presidente"] == "CATENA ROSSELLA"
    # Audit trail added.
    audit = out["metadata"]["anonymization"]
    assert audit["codice_fiscale"] == 1
    assert audit["birth_clause"] == 1
    # Original record untouched.
    assert "RSSMRA80A01H501Z" in rec["text"]


def test_ner_callback_assigns_stable_tokens():
    text = "Mario Rossi incontra Luca Bianchi. Poi Mario Rossi parla con Luca Bianchi."
    calls: list[str] = []

    def fake_ner(t: str) -> list[tuple[str, int, int]]:
        calls.append(t)
        spans = []
        for name in ("Mario Rossi", "Luca Bianchi"):
            start = 0
            while True:
                idx = t.find(name, start)
                if idx < 0:
                    break
                spans.append((name, idx, idx + len(name)))
                start = idx + len(name)
        return spans

    cfg = AnonymiserConfig(
        use_ner=True,
        redact_allcaps_names=False,
        redact_birth=False,
    )
    out, stats = anonymize_text(text, config=cfg, ner=fake_ner)
    assert "Mario Rossi" not in out
    assert "Luca Bianchi" not in out
    # Stable numbering: the first person in text order is PERSONA_1.
    assert "[PERSONA_1]" in out
    assert "[PERSONA_2]" in out
    # Each unique person appears twice in the source → 2 matches per key.
    assert stats.person_ner == 4
    # NER called exactly once.
    assert len(calls) == 1


def test_redaction_stats_total():
    stats = RedactionStats(codice_fiscale=2, email=1, person_allcaps=3)
    assert stats.total() == 6


def test_empty_text_is_noop():
    out, stats = anonymize_text("")
    assert out == ""
    assert stats.total() == 0


def test_idempotent_on_already_anonymised_text():
    text = "ricorrente [PERSONA_1] nato il [DATI_NASCITA] presso [INDIRIZZO]"
    out, stats = anonymize_text(text)
    # No further redactions should trigger on placeholder tokens.
    assert "[PERSONA_1]" in out
    assert "[DATI_NASCITA]" in out
    assert "[INDIRIZZO]" in out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

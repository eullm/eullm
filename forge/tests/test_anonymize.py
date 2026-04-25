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


def test_redacts_address_ocr_variants():
    # All of these appear in italgiure OCR text with slightly different
    # spacing / abbreviations. None of them should leak through.
    cases = [
        "domiciliato in ROMA P.ZZA ADRIANA 5, presso",
        "con studio in Milano P. ZZA CAVOUR n. 10",
        "ROMA P.le CLODIO 2",
        "Torino V.le RE UMBERTO 8",
        "sede in Napoli C.so UMBERTO I",
        "in loc. SAN MARTINO snc",
    ]
    for text in cases:
        out, _ = anonymize_text(text)
        assert "[INDIRIZZO]" in out, f"address not redacted in: {text!r}"


def test_whitelist_keeps_procedural_all_caps():
    text = (
        "udita in CAMERA di CONSIGLIO del 27/11/2023; "
        "CORTE D'APPELLO SEZ DIST di LECCE pronunciò sentenza; "
        "domiciliazione TELEMATICA presso lo studio"
    )
    out, stats = anonymize_text(text)
    for keep in ("CAMERA", "CONSIGLIO", "D'APPELLO", "LECCE", "TELEMATICA"):
        assert keep in out, f"{keep} wrongly redacted"
    assert stats.person_allcaps == 0


def test_allcaps_name_heuristic_catches_party():
    text = "ricorso proposto da CARLOMAGNO FRANCESCO avverso la sentenza"
    out, stats = anonymize_text(text)
    assert "CARLOMAGNO FRANCESCO" not in out
    assert "[PERSONA_1]" in out
    assert stats.person_allcaps == 1


def test_allcaps_role_aware_avvocato():
    """An ALL-CAPS name preceded by 'avvocato' / 'avv.' becomes
    [AVVOCATO_N] just like the NER layer.
    """
    text = "rappresentata e difesa dall'avvocato PAOLO DOGLIOTTI"
    out, stats = anonymize_text(text)
    assert "PAOLO DOGLIOTTI" not in out
    assert "[AVVOCATO_1]" in out
    assert stats.person_allcaps == 1


def test_allcaps_role_aware_consigliere():
    """An ALL-CAPS name preceded by 'Consigliere Relatore Dott.' becomes
    [CONSIGLIERE_N], not [DOTT_N] — first matching role rule wins.
    """
    text = (
        "dal Consigliere Relatore Dott. LORENZO DELLI PRISCOLI. "
        "Esposti i fatti, il collegio decide."
    )
    out, _stats = anonymize_text(text)
    assert "LORENZO DELLI PRISCOLI" not in out
    assert "[CONSIGLIERE_1]" in out


def test_allcaps_per_doc_stability():
    """The same ALL-CAPS name appearing twice gets the SAME token."""
    text = (
        "il sig. CARLOMAGNO FRANCESCO ricorre contro la sentenza. "
        "CARLOMAGNO FRANCESCO chiede inoltre la sospensione."
    )
    out, stats = anonymize_text(text)
    assert "CARLOMAGNO FRANCESCO" not in out
    assert out.count("[PERSONA_1]") == 2
    assert stats.person_allcaps == 2


def test_allcaps_shares_counter_with_ner():
    """When NER assigns [AVVOCATO_1] and [AVVOCATO_2], a third all-caps
    avvocato in the same doc must be [AVVOCATO_3], not start over at _1.
    """
    text = (
        "difesa dagli avv. Mario Rossi, Luigi Bianchi e dall'avvocato "
        "PAOLO DOGLIOTTI; controricorrente"
    )

    def ner(t: str) -> list[tuple[str, int, int]]:
        spans = []
        for name in ("Mario Rossi", "Luigi Bianchi"):
            idx = t.find(name)
            spans.append((name, idx, idx + len(name)))
        return spans

    cfg = AnonymiserConfig(use_ner=True)
    out, _stats = anonymize_text(text, config=cfg, ner=ner)
    assert "[AVVOCATO_1]" in out
    assert "[AVVOCATO_2]" in out
    assert "[AVVOCATO_3]" in out


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


def test_redacts_cf_azienda_before_phone():
    """11-digit company C.F. starting with 0 must be captured by the C.F.
    regex, not by the phone regex."""
    text = "AGENZIA DELLE ENTRATE (C.F. 06363391001), in persona del Direttore"
    out, stats = anonymize_text(text)
    assert "06363391001" not in out
    assert "[CODICE_FISCALE]" in out
    assert "[TELEFONO]" not in out
    assert stats.codice_fiscale == 1
    assert stats.phone == 0


def test_address_ignores_lowercase_locutions():
    """'in via esclusiva', 'via telematica', 'via breve' are locutions,
    not addresses. They must NOT be redacted to [INDIRIZZO]."""
    cases = [
        "attribuita in via esclusiva al legislatore statale",
        "notificato a mezzo via telematica",
        "la via libera è stata concessa",
        "per via breve e informalmente",
    ]
    for text in cases:
        out, stats = anonymize_text(text)
        assert "[INDIRIZZO]" not in out, f"locution redacted: {text!r}"
        assert stats.address == 0, f"counted: {text!r}"


def test_ner_ignores_allcaps_acronym():
    """Single ALL-CAPS acronyms (ARIF, ENEA, CNEL) are agencies, not
    persons. spaCy tags them as PER — filter them out."""
    text = "l'ente pubblico ARIF ricorre. Anche ENEA partecipa al giudizio."

    def ner(t: str) -> list[tuple[str, int, int]]:
        spans = []
        for name in ("ARIF", "ENEA"):
            idx = t.find(name)
            if idx >= 0:
                spans.append((name, idx, idx + len(name)))
        return spans

    cfg = AnonymiserConfig(use_ner=True, redact_allcaps_names=False)
    out, stats = anonymize_text(text, config=cfg, ner=ner)
    assert "ARIF" in out
    assert "ENEA" in out
    assert stats.person_ner == 0


def test_ner_ignores_institutional_spans():
    """spaCy frequently mis-tags institutional references ("La Corte",
    "Cass.", "Direttore", "Tribunale") as PER. Those must not be redacted.
    """
    text = (
        "La Corte di Cassazione ha stabilito che Cass. 19/08/2020 n. 17313 "
        "si applica. Il Direttore dell'Agenzia ha ricorso."
    )

    def ner(t: str) -> list[tuple[str, int, int]]:
        spans = []
        for name in ("La Corte", "Cass", "Direttore", "Tribunale"):
            idx = t.find(name)
            if idx >= 0:
                spans.append((name, idx, idx + len(name)))
        return spans

    cfg = AnonymiserConfig(use_ner=True, redact_allcaps_names=False)
    out, stats = anonymize_text(text, config=cfg, ner=ner)
    assert "[PERSONA" not in out
    assert stats.person_ner == 0
    assert "La Corte" in out
    assert "Direttore" in out


def test_allcaps_not_redacted_before_company_marker():
    """'BANCA MONTE DEI PASCHI DI SIENA S.P.A.' is a company, not a person.
    The S.P.A. suffix must suppress the all-caps person heuristic.
    """
    cases = [
        "contro BANCA MONTE DEI PASCHI DI SIENA S.P.A., sedente in Siena",
        "la GENERALI ASSICURAZIONI S.p.A. propone ricorso",
        "UNICREDIT BANCA Spa ricorre",
        "contro INTESA SANPAOLO S.R.L. per il tramite",
    ]
    for text in cases:
        out, stats = anonymize_text(text)
        assert "[PERSONA" not in out, f"false positive on: {text!r}"
        assert "[AVVOCATO" not in out, f"false positive on: {text!r}"
        assert "[CONSIGLIERE" not in out, f"false positive on: {text!r}"
        assert "[PRESIDENTE" not in out, f"false positive on: {text!r}"
        assert "[DOTT_" not in out, f"false positive on: {text!r}"
        assert stats.person_allcaps == 0, f"stats wrong on: {text!r}"


def test_allcaps_whitelist_keeps_legal_terms():
    """Phrases like 'ONERE PROVA', 'CAUSA PETENDI' are legal terms, not
    persons. The extended whitelist must keep them.
    """
    text = (
        "un ONERE PROVA censura valut importo; la CAUSA PETENDI non "
        "è mutata; LEGGE FALLIMENTARE; RICORSO INAMMISSIBILE"
    )
    out, stats = anonymize_text(text)
    for keep in ("ONERE PROVA", "CAUSA", "LEGGE", "RICORSO"):
        assert keep in out, f"{keep!r} wrongly redacted in: {out!r}"


def test_ner_ignores_junk_spans():
    """Regression test: spaCy sometimes returns 1-2 char junk spans
    (stopwords, single letters, articles). These must never be used as
    replacement keys — a case-insensitive global replace of "l" would
    nuke every occurrence of that letter in the document.
    """
    text = "La Corte di Cassazione, con ordinanza, dichiara l'inammissibile"

    def junk_ner(t: str) -> list[tuple[str, int, int]]:
        return [
            ("l", 48, 49),
            ("La", 0, 2),
            ("di", 10, 12),
            ("re", 0, 2),
        ]

    cfg = AnonymiserConfig(use_ner=True, redact_allcaps_names=False)
    out, stats = anonymize_text(text, config=cfg, ner=junk_ner)
    assert "[PERSONA" not in out
    assert stats.person_ner == 0
    assert "Cassazione" in out
    assert "ordinanza" in out


def test_ner_respects_word_boundaries():
    """A short surname must not eat substrings inside longer words:
    'Monte' should not redact 'montepremi' or 'Monteverdi'.
    """
    text = "Il signor Monte ricorre; al montepremi si aggiunge Monteverdi."

    def ner(t: str) -> list[tuple[str, int, int]]:
        idx = t.find("Monte")
        return [("Monte", idx, idx + 5)]

    cfg = AnonymiserConfig(
        use_ner=True,
        redact_allcaps_names=False,
        redact_address=False,
    )
    out, stats = anonymize_text(text, config=cfg, ner=ner)
    assert "[PERSONA_1]" in out
    assert "montepremi" in out
    assert "Monteverdi" in out
    assert stats.person_ner == 1


def test_ner_drops_pqm_locution():
    """`P.Q.M. La Corte` (Per Questi Motivi + La Corte) is the dispositive
    formula in Cassazione rulings. spaCy NER bundles it as a single PER
    span. Must not be redacted: P.Q.M. is fixed legalese, La Corte is
    institutional.
    """
    text = (
        "...n. 1778). P. Q. M. La Corte dichiara l'inammissibilità del "
        "ricorso. Condanna l'Agenzia ricorrente al pagamento."
    )

    def ner(t: str) -> list[tuple[str, int, int]]:
        idx = t.find("P. Q. M. La Corte")
        return [("P. Q. M. La Corte", idx, idx + len("P. Q. M. La Corte"))]

    cfg = AnonymiserConfig(use_ner=True, redact_allcaps_names=False)
    out, stats = anonymize_text(text, config=cfg, ner=ner)
    assert "[PERSONA" not in out
    assert "P. Q. M. La Corte" in out
    assert stats.person_ner == 0


def test_ner_drops_org_prefix_spans():
    """Long organisational names like 'Agenzia Regionale per le Attività
    Irrigue e Forestali' must not be redacted as PER, regardless of how
    many tokens they contain.
    """
    text = (
        "L'Agenzia Regionale per le Attività Irrigue e Forestali "
        "(ARIF) il cui rapporto è regolato dal CCNL."
    )

    def ner(t: str) -> list[tuple[str, int, int]]:
        idx = t.find("Agenzia Regionale per le Attività Irrigue e Forestali")
        return [(
            "Agenzia Regionale per le Attività Irrigue e Forestali",
            idx, idx + len(
                "Agenzia Regionale per le Attività Irrigue e Forestali"
            ),
        )]

    cfg = AnonymiserConfig(use_ner=True, redact_allcaps_names=False)
    out, stats = anonymize_text(text, config=cfg, ner=ner)
    assert "[PERSONA" not in out
    assert "Agenzia Regionale" in out
    assert stats.person_ner == 0


def test_ner_drops_acronym_inside_multi_token_span():
    """spaCy occasionally bundles 'ACRONYM lowerword' as a PER span.
    Reject when an all-caps 2-6 char token is glued to a mixed-case one.
    Must NOT reject 'PASQUALE ROSSI' (both all-caps).
    """
    text = "Il signor PASQUALE ROSSI ricorre."

    def junk_ner(t: str) -> list[tuple[str, int, int]]:
        return [
            ("ARIF siaente", 0, 12),
            ("Cass MT", 12, 19),
        ]

    cfg = AnonymiserConfig(use_ner=True, redact_allcaps_names=False)
    out, _stats = anonymize_text(text, config=cfg, ner=junk_ner)
    # Junk spans must never produce a [PERSONA] tag.
    assert "[PERSONA" not in out

    # Sanity check: the lexical filter alone must still let real all-caps
    # surnames through. Run the validator directly.
    from eullm_forge.datasets.anonymize import _is_valid_ner_span
    assert _is_valid_ner_span("PASQUALE ROSSI")
    assert not _is_valid_ner_span("ARIF siaente")
    assert not _is_valid_ner_span("Cass MT")


def test_ner_drops_titlecase_when_acronym_in_doc():
    """If the document contains 'ARIF' (all-caps), a NER span 'Arif'
    elsewhere is the same acronym — never a person.
    """
    text = (
        "L'ARIF è ente pubblico non economico. Il personale "
        "inquadrato in Arif a tempo indeterminato è disciplinato "
        "dal CCNL del comparto."
    )

    def ner(t: str) -> list[tuple[str, int, int]]:
        idx = t.find("Arif")
        return [("Arif", idx, idx + 4)]

    cfg = AnonymiserConfig(use_ner=True, redact_allcaps_names=False)
    out, stats = anonymize_text(text, config=cfg, ner=ner)
    assert "[PERSONA" not in out
    assert "Arif" in out
    assert stats.person_ner == 0


def test_ner_drops_capitalised_gerunds():
    """`Aggiungendosi`, `Avverso`, `Udita`, `Rilevato` start sentences in
    italgiure OCR text and get tagged PER by spaCy. They are participles /
    gerunds, not names.
    """
    cfg = AnonymiserConfig(use_ner=True, redact_allcaps_names=False)
    for word in ("Aggiungendosi", "Avverso", "Udita", "Rilevato"):
        text = f"{word} la sentenza, la Corte ha disposto."

        def ner(t: str, w: str = word) -> list[tuple[str, int, int]]:
            return [(w, 0, len(w))]

        out, _stats = anonymize_text(text, config=cfg, ner=ner)
        assert "[PERSONA" not in out, f"false positive on: {word!r}"
        assert word in out


def test_role_aware_token_avvocato():
    """A NER span preceded by 'avv.' / 'Avv.' / 'avvocato' is mapped to
    `[AVVOCATO_N]`, not `[PERSONA_N]`. Role context is preserved.
    """
    text = (
        "rappresentata e difesa dagli avv. Pasquale Russo, "
        "Guglielmo Fransoni e Francesco Padovani, come da procura"
    )

    def ner(t: str) -> list[tuple[str, int, int]]:
        spans = []
        for name in ("Pasquale Russo", "Guglielmo Fransoni",
                     "Francesco Padovani"):
            idx = t.find(name)
            spans.append((name, idx, idx + len(name)))
        return spans

    cfg = AnonymiserConfig(use_ner=True, redact_allcaps_names=False)
    out, stats = anonymize_text(text, config=cfg, ner=ner)
    assert "[AVVOCATO_1]" in out
    assert "[AVVOCATO_2]" in out
    assert "[AVVOCATO_3]" in out
    assert "[PERSONA_1]" not in out
    assert "Pasquale Russo" not in out
    assert stats.person_ner == 3


def test_role_aware_token_consigliere():
    """`Consigliere Relatore Dott. LORENZO DELLI PRISCOLI` and similar
    phrasings produce `[CONSIGLIERE_N]`.
    """
    text = (
        "udita la relazione svolta dal consigliere Alberto Crivelli, "
        "ha pronunciato la seguente ordinanza"
    )

    def ner(t: str) -> list[tuple[str, int, int]]:
        idx = t.find("Alberto Crivelli")
        return [("Alberto Crivelli", idx, idx + len("Alberto Crivelli"))]

    cfg = AnonymiserConfig(use_ner=True, redact_allcaps_names=False)
    out, stats = anonymize_text(text, config=cfg, ner=ner)
    assert "[CONSIGLIERE_1]" in out
    assert "Alberto Crivelli" not in out
    assert stats.person_ner == 1


def test_role_aware_token_default_is_persona():
    """A NER span without role context falls back to `[PERSONA_N]`."""
    text = "il signor De Simone ha ricorso contro la decisione."

    def ner(t: str) -> list[tuple[str, int, int]]:
        idx = t.find("De Simone")
        return [("De Simone", idx, idx + len("De Simone"))]

    cfg = AnonymiserConfig(use_ner=True, redact_allcaps_names=False)
    out, stats = anonymize_text(text, config=cfg, ner=ner)
    assert "[PERSONA_1]" in out
    assert "De Simone" not in out
    assert stats.person_ner == 1


def test_role_aware_per_role_counters_are_independent():
    """Each role has its own counter: two avvocati and two consiglieri
    produce AVVOCATO_1/AVVOCATO_2 and CONSIGLIERE_1/CONSIGLIERE_2.
    """
    text = (
        "difesa dall'avv. Mario Rossi e dall'avv. Luigi Bianchi; "
        "udita la relazione del consigliere Anna Verdi, "
        "presieduta dal Presidente Carlo Neri"
    )

    def ner(t: str) -> list[tuple[str, int, int]]:
        spans = []
        for name in ("Mario Rossi", "Luigi Bianchi",
                     "Anna Verdi", "Carlo Neri"):
            idx = t.find(name)
            spans.append((name, idx, idx + len(name)))
        return spans

    cfg = AnonymiserConfig(use_ner=True, redact_allcaps_names=False)
    out, _stats = anonymize_text(text, config=cfg, ner=ner)
    assert "[AVVOCATO_1]" in out
    assert "[AVVOCATO_2]" in out
    assert "[CONSIGLIERE_1]" in out
    assert "[PRESIDENTE_1]" in out
    # No cross-contamination of numbering.
    assert "[AVVOCATO_3]" not in out


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

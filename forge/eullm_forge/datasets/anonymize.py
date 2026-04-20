"""GDPR-aware anonymisation for Italian legal text (Cassazione sentences).

The italgiure SentenzeWeb corpus contains personal data in the body text:
party names, birth dates/places, codice fiscale, partita IVA, addresses,
phone numbers, email addresses. Training a language model on this data
without anonymisation would bake those identifiers into model weights and
risk memorisation attacks.

This module provides two layers of redaction:

1. **Regex layer** (always on, no dependencies) — deterministic patterns for
   codice fiscale, partita IVA, IBAN, dates/places of birth, street
   addresses, phone numbers, email addresses.

2. **NER layer** (optional, requires spaCy) — detects PER entities in the
   text and replaces each unique person with a stable per-document token
   (``[PERSONA_1]``, ``[PERSONA_2]``, ...) so coherence inside a ruling is
   preserved (same person → same token throughout).

What is KEPT (not redacted):
    * Presidente / relatore names in metadata — public officials acting in
      their official capacity, their identity is public by law.
    * Court/agency names (Corte di Cassazione, Agenzia delle Entrate, ...).
    * Legal article references, city names not tied to a birth.

What is REDACTED:
    * Party names (ricorrente, controricorrente, imputato, ...).
    * Lawyer names (professional-but-identifying in the context).
    * Codice fiscale, partita IVA, IBAN.
    * Dates and places of birth.
    * Street addresses, phone numbers, email addresses.

The redaction is one-way: original text is not recoverable from the output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Italian codice fiscale: 6 letters + 2 digits + 1 letter + 2 digits + 1 letter
# + 3 digits + 1 letter. Exactly 16 chars.
RE_CF = re.compile(
    r"\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b"
)

# Italian partita IVA: 11 digits. To avoid matching every 11-digit number we
# require an introducing context token ("P.IVA", "P. IVA", "partita iva",
# "VAT", "C.F." when applied to companies).
RE_PIVA = re.compile(
    r"\b(?:P\.?\s*IVA|partita\s+iva|VAT)[:\s]*\d{11}\b",
    re.IGNORECASE,
)

# Italian IBAN: starts with IT + 2 check digits + CIN letter + 5 ABI + 5 CAB
# + 12 account chars. 27 chars total.
RE_IBAN = re.compile(
    r"\bIT\d{2}[A-Z]\d{10}[A-Z0-9]{12}\b"
)

RE_EMAIL = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

# Italian phone: +39 xxx xxxxxxx, 3xx xxxxxxx (mobile), landline with area
# code. Kept conservative to avoid stripping every 9-digit number that might
# be a protocol code.
RE_PHONE = re.compile(
    r"\b(?:\+39[\s.-]?)?(?:3\d{2}|0\d{1,3})[\s.-]?\d{5,8}\b"
)

# "nato a LUOGO il GG/MM/AAAA" or "nata a LUOGO il GG-MM-AAAA" — strongest
# PII signal in the corpus. Captures the whole clause so we can strip both
# the place and the date of birth in one shot.
RE_BIRTH_CLAUSE = re.compile(
    r"\bnat[oa]\s+(?:a|in)\s+[A-ZÀ-ÿ][A-Za-zÀ-ÿ'\s\.-]{1,80}?"
    r"\s+(?:il|in\s+data)\s+\d{1,2}[\s./-]\d{1,2}[\s./-]\d{2,4}",
    re.IGNORECASE,
)

# Street address: "via/viale/piazza/corso/largo/vicolo NAME[, n. NUM]".
# The body must stop at a comma/semicolon/period/newline (or start a civic
# number) so the match doesn't spill into the next clause.
RE_ADDRESS = re.compile(
    r"\b(?:via|viale|v\.le|piazza|p\.?zza|corso|c\.so|largo|vicolo|strada)\s+"
    r"[^,;.\n]{2,80}?"
    r"(?:,?\s*(?:n\.?|numero|nr\.?)\s*\d{1,4}[A-Za-z]?|(?=[,;.\n]))",
    re.IGNORECASE,
)

# Generic dates (fallback, after birth clauses have been handled). We do
# NOT redact plain dates because they are legally meaningful (data decisione,
# data deposito, data sentenza impugnata etc.) — only dates tied to births
# via RE_BIRTH_CLAUSE are considered PII.

# All-caps person names as they appear in the italgiure OCR: "CARLOMAGNO
# FRANCESCO", "LA ROSA FRANCESCO", "DI SPIRITO FABIO". Requires at least
# two consecutive all-caps tokens, each 2+ chars, optionally joined by
# particles (di/de/la/del/della/van/von).
RE_ALLCAPS_NAME = re.compile(
    r"\b(?:[A-ZÀ-Ÿ]{2,}\s+){1,2}"
    r"(?:(?:DI|DE|DEL|DELLA|LA|LO|VAN|VON|DU|DA)\s+)?"
    r"[A-ZÀ-Ÿ]{2,}\b"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class RedactionStats:
    """Counts of redactions made, per category."""

    codice_fiscale: int = 0
    partita_iva: int = 0
    iban: int = 0
    email: int = 0
    phone: int = 0
    birth_clause: int = 0
    address: int = 0
    person_ner: int = 0
    person_allcaps: int = 0

    def total(self) -> int:
        return sum(getattr(self, f.name) for f in self.__dataclass_fields__.values())

    def to_dict(self) -> dict[str, int]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class AnonymiserConfig:
    """Tuning knobs for anonymisation.

    Defaults are conservative: always-on regex layers plus all-caps name
    heuristic (safe on italgiure OCR text). NER is off by default because
    it requires spaCy and its Italian model.
    """

    redact_cf: bool = True
    redact_piva: bool = True
    redact_iban: bool = True
    redact_email: bool = True
    redact_phone: bool = True
    redact_birth: bool = True
    redact_address: bool = True
    redact_allcaps_names: bool = True
    use_ner: bool = False


# Whitelist of ALL-CAPS tokens that must NOT be treated as person names.
_ALLCAPS_WHITELIST = frozenset(
    {
        # Courts and procedural entities
        "CORTE", "CASSAZIONE", "SUPREMA", "APPELLO", "TRIBUNALE", "SEZIONE",
        "SEZIONI", "UNITE", "PENALE", "CIVILE", "LAVORO", "TRIBUTARIA",
        "REPUBBLICA", "ITALIANA", "CONSIGLIERE", "PRESIDENTE", "RELATORE",
        "PUBBLICO", "MINISTERO", "PROCURA", "PROCURATORE", "GENERALE",
        "SOSTITUTO", "AVVOCATO", "AVVOCATI", "DIFENSORE", "DIFENSORI",
        "IMPUTATO", "IMPUTATA", "RICORRENTE", "CONTRORICORRENTE",
        "RICORSO", "SENTENZA", "ORDINANZA", "DECRETO", "MOTIVI", "FATTO",
        "DIRITTO", "CONSIDERATO", "RITENUTO", "RILEVATO", "OSSERVATO",
        "LIBERTA", "LIBERTÀ",
        # Common Italian agencies / public entities
        "AGENZIA", "ENTRATE", "RISCOSSIONE", "EQUITALIA", "INPS", "INAIL",
        "ENEL", "RAI", "POSTE", "ITALIANE", "FERROVIE", "STATO",
        # Company forms
        "SPA", "SRL", "SNC", "SAS", "SCARL", "SCRL", "SPAF",
        # Geographic filler commonly appearing uppercase
        "ROMA", "MILANO", "NAPOLI", "TORINO", "PALERMO", "BARI", "GENOVA",
        "FIRENZE", "BOLOGNA", "CATANIA", "VENEZIA", "VERONA", "PADOVA",
        "PERUGIA", "ITALIA", "ITALY",
        # Misc procedural acronyms
        "ECLI", "IT", "CASS", "PEN", "CIV", "SEZ",
    }
)


def _is_likely_person(match_text: str) -> bool:
    """Heuristic filter for the all-caps name regex — drops obvious false
    positives like 'CORTE SUPREMA' or 'AGENZIA ENTRATE'.
    """
    tokens = [t for t in re.split(r"\s+", match_text.strip()) if t]
    if len(tokens) < 2:
        return False
    # If any token is a whitelisted procedural/geographic word → drop.
    if any(t.upper() in _ALLCAPS_WHITELIST for t in tokens):
        return False
    # Drop if it's just digits-letters mix (codes, not names).
    if any(re.search(r"\d", t) for t in tokens):
        return False
    return True


def anonymize_text(
    text: str,
    *,
    config: Optional[AnonymiserConfig] = None,
    ner: Optional[Callable[[str], list[tuple[str, int, int]]]] = None,
) -> tuple[str, RedactionStats]:
    """Anonymise a piece of Italian legal text.

    Args:
        text: Input text (may contain personal data).
        config: Which layers to apply (defaults to ``AnonymiserConfig()``).
        ner: Optional callable returning a list of ``(entity_text, start,
            end)`` for PERSON entities. Use ``load_spacy_ner()`` to build
            one. If ``None`` and ``config.use_ner`` is True, NER is
            silently skipped.

    Returns:
        ``(anonymised_text, stats)`` tuple. Stats is a RedactionStats with
        per-category counts.
    """
    cfg = config or AnonymiserConfig()
    stats = RedactionStats()

    # 1) NER first (runs on original text so offsets are valid, then we
    #    substitute; we do this before regex so that person names inside
    #    birth clauses get a stable token before the clause is nuked).
    if cfg.use_ner and ner is not None:
        person_map: dict[str, str] = {}
        # Collect entities and dedupe by normalised name.
        entities = ner(text)
        # Sort by start so we get stable numbering in text order.
        entities = sorted(set(entities), key=lambda e: e[1])
        for name, _s, _e in entities:
            key = _normalise_name(name)
            if key and key not in person_map:
                person_map[key] = f"[PERSONA_{len(person_map) + 1}]"
        if person_map:
            # Replace longest names first to avoid partial overlap bugs.
            for key in sorted(person_map, key=len, reverse=True):
                token = person_map[key]
                # Replace with case-insensitive boundary match.
                pattern = re.compile(re.escape(key), re.IGNORECASE)
                new_text, n = pattern.subn(token, text)
                if n:
                    stats.person_ner += n
                    text = new_text

    # 2) Regex layers — order matters: catch structured patterns first.
    if cfg.redact_cf:
        text, n = RE_CF.subn("[CODICE_FISCALE]", text)
        stats.codice_fiscale += n
    if cfg.redact_piva:
        text, n = RE_PIVA.subn("[PARTITA_IVA]", text)
        stats.partita_iva += n
    if cfg.redact_iban:
        text, n = RE_IBAN.subn("[IBAN]", text)
        stats.iban += n
    if cfg.redact_email:
        text, n = RE_EMAIL.subn("[EMAIL]", text)
        stats.email += n
    if cfg.redact_phone:
        text, n = RE_PHONE.subn("[TELEFONO]", text)
        stats.phone += n
    if cfg.redact_birth:
        text, n = RE_BIRTH_CLAUSE.subn("[DATI_NASCITA]", text)
        stats.birth_clause += n
    if cfg.redact_address:
        text, n = RE_ADDRESS.subn("[INDIRIZZO]", text)
        stats.address += n

    # 3) All-caps name heuristic (last, on text with structured PII already
    #    stripped so we don't double-count).
    if cfg.redact_allcaps_names:
        def _sub(m: re.Match[str]) -> str:
            s = m.group(0)
            if _is_likely_person(s):
                stats.person_allcaps += 1
                return "[PERSONA]"
            return s

        text = RE_ALLCAPS_NAME.sub(_sub, text)

    return text, stats


def anonymize_record(
    rec: dict[str, Any],
    *,
    config: Optional[AnonymiserConfig] = None,
    ner: Optional[Callable[[str], list[tuple[str, int, int]]]] = None,
) -> dict[str, Any]:
    """Anonymise the ``text`` field of a record and annotate metadata.

    The record is shallow-copied; the original dict is not mutated. The
    returned record has an ``anonymization`` sub-object inside metadata
    with per-category redaction counts — useful for audit and for spotting
    slices that need extra scrutiny.
    """
    if "text" not in rec:
        return rec
    new_text, stats = anonymize_text(rec["text"], config=config, ner=ner)
    out = dict(rec)
    out["text"] = new_text
    meta = dict(rec.get("metadata") or {})
    meta["anonymization"] = stats.to_dict()
    out["metadata"] = meta
    return out


def _normalise_name(name: str) -> str:
    """Strip trailing punctuation / collapse whitespace for map key."""
    name = re.sub(r"\s+", " ", name).strip(" ,.;:")
    return name


# ---------------------------------------------------------------------------
# Optional NER backend (spaCy)
# ---------------------------------------------------------------------------


def load_spacy_ner(
    model: str = "it_core_news_lg",
) -> Callable[[str], list[tuple[str, int, int]]]:
    """Return a callable that runs spaCy NER and extracts PERSON spans.

    Raises ``RuntimeError`` if spaCy or the model is unavailable. The
    caller can fall back to regex-only mode in that case.
    """
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError(
            "spaCy not installed — run `pip install spacy` and "
            f"`python -m spacy download {model}`"
        ) from exc
    try:
        nlp = spacy.load(model, disable=["parser", "tagger", "lemmatizer"])
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model {model!r} not found — run "
            f"`python -m spacy download {model}`"
        ) from exc

    def _extract(text: str) -> list[tuple[str, int, int]]:
        doc = nlp(text)
        return [
            (ent.text, ent.start_char, ent.end_char)
            for ent in doc.ents
            if ent.label_ == "PER"
        ]

    return _extract

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

# Codice fiscale of entities/companies: "C.F." or "cod. fisc." followed by
# 11 digits. Must be caught BEFORE RE_PHONE, since 11-digit landline-like
# sequences (e.g. '06363391001' for the Agenzia delle Entrate) would
# otherwise be swallowed by the phone regex.
RE_CF_AZIENDA = re.compile(
    r"\b(?:C\.?\s*F\.?|cod(?:ice)?\s*fisc(?:ale)?)[:\s]*\d{11}\b",
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
# Accepts many OCR variants of the street keyword: "V.LE", "V. LE", "P.ZZA",
# "P. ZZA", "P.ZA", "P.LE", "C.SO". The body must stop at a comma/semicolon/
# period/newline (or start a civic number) so the match doesn't spill into
# the next clause.
RE_ADDRESS = re.compile(
    r"\b(?:"
    r"via|viale|v\.\s?le|"
    r"piazza|piazzale|p\.\s?zza|p\.\s?za|p\.\s?le|"
    r"corso|c\.\s?so|"
    r"largo|l\.\s?go|"
    r"vicolo|vico|strada|contrada|loc\.|localit[aà]"
    r")\s+"
    r"[^,;.\n]{2,80}?"
    r"(?:,?\s*(?:n\.?|numero|nr\.?|civico)?\s*\d{1,4}[A-Za-z]?(?=[\s,;.\n]|$)"
    r"|(?=[,;.\n])|$)",
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

# Company-form markers that, when they follow an all-caps run, signal that
# the run is a ragione sociale rather than a person name (e.g.
# "BANCA MONTE DEI PASCHI DI SIENA S.P.A.").
_COMPANY_TAIL_RE = re.compile(
    r"\s*(?:"
    r"S\.?\s?P\.?\s?A\.?|"
    r"S\.?\s?R\.?\s?L\.?|"
    r"S\.?\s?A\.?\s?S\.?|"
    r"S\.?\s?N\.?\s?C\.?|"
    r"S\.?C\.?A\.?R\.?L\.?|"
    r"SCARL|SCRL|SPA|SRL|SAS|SNC|"
    r"Spa|Srl|Sas|Snc|"
    r"GROUP|HOLDING|LTD|LIMITED|GMBH|INC"
    r")\b",
    re.IGNORECASE,
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


# Italian stopwords / particles that spaCy NER occasionally mis-tags as PER.
# Never use these as replacement keys — a 1-2 char key combined with
# case-insensitive global replace nukes the whole document.
_NER_STOPWORDS = frozenset(
    {
        "il", "lo", "la", "i", "gli", "le", "un", "uno", "una",
        "di", "a", "da", "in", "con", "su", "per", "tra", "fra",
        "del", "dello", "della", "dei", "degli", "delle",
        "al", "allo", "alla", "ai", "agli", "alle",
        "dal", "dallo", "dalla", "dai", "dagli", "dalle",
        "nel", "nello", "nella", "nei", "negli", "nelle",
        "sul", "sullo", "sulla", "sui", "sugli", "sulle",
        "e", "o", "ma", "se", "che", "non", "è", "sono",
        "l", "dell", "nell", "sull", "all", "dall",
        "re", "st", "pr", "cd", "ed", "od", "ne", "io", "tu", "lui", "lei",
        "art", "dott", "avv", "sig", "ing", "prof",
        # Legal / procedural abbreviations that spaCy keeps tagging as PER.
        "cost", "ric", "sent", "ord", "cass", "trib", "cod", "cons",
        # Procedural locutions / gerunds in italgiure OCR that come out
        # capitalised and slip past spaCy as PER.
        "pqm", "p.q.m", "p.q.m.", "p. q. m.", "p. q. m",
        "aggiungendosi", "avverso", "udita", "rilevato", "ritenuto",
        "considerato", "premesso", "osservato", "letto",
    }
)


# Organisational prefixes — when a span STARTS with one of these, capitalised,
# it is the name of an entity / public body, never a person ("Agenzia Regionale
# per le Attività Irrigue e Forestali", "Ministero dell'Economia",
# "Direzione Provinciale di Bari", "Tribunale di Lecce", "Avvocatura Generale
# dello Stato", ...). Reject regardless of what follows.
_NER_ORG_PREFIXES = frozenset(
    {
        "agenzia", "ministero", "regione", "comune", "provincia",
        "città", "citta", "direzione", "dipartimento", "ufficio", "servizio",
        "commissione", "autorità", "autorita", "consiglio", "corte",
        "tribunale", "procura", "avvocatura", "consulta", "garante",
        "sovrintendenza", "soprintendenza", "ispettorato", "questura",
        "prefettura", "ente", "istituto", "azienda", "fondazione",
        "associazione", "federazione", "ordine", "albo", "camera",
        "università", "universita", "scuola", "accademia", "centro",
        "osservatorio",
    }
)


# Particles allowed inside multi-token person spans ("DI ROSSI", "DE LUCA").
# When the acronym filter scans a span, all-caps tokens equal to one of these
# are NOT counted as acronyms.
_PARTICLES = frozenset(
    {
        "DI", "DE", "DA", "DEL", "DELLA", "DELLE", "DEGLI", "DEI",
        "LA", "LO", "LE", "LI", "VAN", "VON", "DU", "MAC", "MC", "AL",
    }
)


# Institutional / procedural tokens that spaCy's it_core_news_lg regularly
# mis-tags as PER in legal text ("Cass.", "La Corte", "Direttore",
# "Tribunale", ...). If *every* significant token of a candidate span is
# drawn from this set, we reject the span.
_NER_INSTITUTIONAL = frozenset(
    {
        "cass", "corte", "tribunale", "cassazione", "sezione", "sezioni",
        "camera", "consiglio", "collegio", "udienza", "adunanza",
        "consigliere", "presidente", "relatore", "procura", "procuratore",
        "avvocatura", "avvocato", "ministero", "ministro", "agenzia",
        "commissione", "direttore", "direzione", "ufficio", "servizio",
        "ordine", "albo", "curatore", "commissario", "sindaco", "perito",
        "giudice", "magistrato", "cancelleria", "cancelliere",
        "costituzionale", "costituzione", "suprema", "superiore",
        "regionale", "provinciale", "nazionale", "tributaria", "civile",
        "penale", "amministrativo", "costituente", "consulta",
    }
)


def _is_valid_ner_span(name: str) -> bool:
    """Filter obviously broken NER spans before using them for replacement.

    spaCy's it_core_news_lg occasionally tags isolated characters, particles,
    titles, or punctuation as PER — and feeding those into a regex replace
    erases entire documents. Reject anything we wouldn't want to redact even
    if the tag were correct.
    """
    name = name.strip(" .,;:'\"-–—")
    if len(name) < 3:
        return False
    if not re.search(r"[A-Za-zÀ-ÿ]", name):
        return False
    # All-lowercase tokens are almost never personal names in legal text,
    # which is Title-Case or UPPERCASE for parties. Rejecting lowercase
    # also removes Italian articles/particles.
    if name == name.lower():
        return False
    # Stopword check both on the raw span and on a punctuation-stripped
    # variant ("P. Q. M." -> "pqm" / "p.q.m." / "p. q. m.").
    name_norm = name.lower()
    if name_norm in _NER_STOPWORDS:
        return False
    name_compact = re.sub(r"\s+", "", name_norm)
    if name_compact in _NER_STOPWORDS:
        return False
    # At least one alphabetic token of 3+ chars (rejects "L.", "A.", "J.R.").
    tokens = [t for t in re.split(r"[\s.'-]+", name) if t]
    if not any(len(tok) >= 3 and tok.isalpha() for tok in tokens):
        return False
    # Drop spans where every significant token is an institutional /
    # procedural word ("La Corte", "Cass.", "Direttore", "Tribunale di ...").
    # Filter out single-character tokens too — "P. Q. M. La Corte" tokenises
    # as ['P', 'Q', 'M', 'La', 'Corte']; without this filter the single
    # letters survive as 'significant' and shield the span from rejection.
    significant = [
        t.lower() for t in tokens
        if len(t) >= 2 and t.lower() not in _NER_STOPWORDS
    ]
    if significant and all(t in _NER_INSTITUTIONAL for t in significant):
        return False
    # Drop spans whose first significant token is an organisational prefix
    # ("Agenzia Regionale ...", "Ministero dell'Economia", "Direzione
    # Provinciale di Bari"). spaCy mis-tags long entity names as PER.
    if significant and significant[0] in _NER_ORG_PREFIXES:
        return False
    # Single ALL-CAPS acronym (2-6 chars, no internal whitespace) is almost
    # always an agency/entity code ('ARIF', 'ENEA', 'CNEL', 'CCNL'), not a
    # person. Italian surnames of this form are rare and anyway caught by
    # the all-caps name heuristic when combined with a given name.
    if len(tokens) == 1 and tokens[0].isupper() and 2 <= len(tokens[0]) <= 6:
        return False
    # Multi-token spans that mix an acronym-shaped token with a mixed-case
    # token ("ARIF siaente", "Cass MT", "INPS gestione") are NER bundling
    # bugs, never people. Reject when both are present.
    # ALL-CAPS multi-token spans like "PASQUALE ROSSI" must NOT trip this:
    # we require at least one non-acronym mixed-case token in the same span.
    if len(tokens) >= 2:
        has_acronym = any(
            tok.isupper() and tok.isalpha()
            and 2 <= len(tok) <= 6
            and tok not in _PARTICLES
            for tok in tokens
        )
        has_mixed = any(
            (not tok.isupper())
            and any(c.isalpha() for c in tok)
            for tok in tokens
        )
        if has_acronym and has_mixed:
            return False
    return True


# Whitelist of ALL-CAPS tokens that must NOT be treated as person names.
_ALLCAPS_WHITELIST = frozenset(
    {
        # Courts and procedural entities
        "CORTE", "CASSAZIONE", "SUPREMA", "APPELLO", "D'APPELLO", "TRIBUNALE",
        "SEZIONE", "SEZIONI", "UNITE", "PENALE", "CIVILE", "LAVORO",
        "TRIBUTARIA", "REPUBBLICA", "ITALIANA", "CONSIGLIERE", "PRESIDENTE",
        "RELATORE", "PUBBLICO", "MINISTERO", "PROCURA", "PROCURATORE",
        "GENERALE", "SOSTITUTO", "AVVOCATO", "AVVOCATI", "DIFENSORE",
        "DIFENSORI", "IMPUTATO", "IMPUTATA", "RICORRENTE", "CONTRORICORRENTE",
        "RICORSO", "SENTENZA", "ORDINANZA", "DECRETO", "MOTIVI", "FATTO",
        "DIRITTO", "CONSIDERATO", "RITENUTO", "RILEVATO", "OSSERVATO",
        "LIBERTA", "LIBERTÀ", "CAMERA", "CONSIGLIO", "UDIENZA", "UDITA",
        "PRONUNCIATA", "PRONUNCIATO", "DISTACCATA", "DIST", "COLLEGIO",
        "TELEMATICA", "TELEMATICO",
        # Common Italian agencies / public entities
        "AGENZIA", "ENTRATE", "RISCOSSIONE", "EQUITALIA", "INPS", "INAIL",
        "ENEL", "RAI", "POSTE", "ITALIANE", "FERROVIE", "STATO", "MINISTERO",
        "COMUNE", "PROVINCIA", "REGIONE",
        # Company forms and frequent company-name tokens
        "SPA", "SRL", "SNC", "SAS", "SCARL", "SCRL", "SPAF", "GROUP", "HOLDING",
        "BANCA", "BANCO", "ASSICURAZIONI", "IMMOBILIARE", "COSTRUZIONI",
        "SERVIZI", "COMMERCIO", "INDUSTRIE", "EDIZIONI", "FINANZIARIA",
        # Frequent legal / procedural all-caps phrases that are NOT names
        "ONERE", "PROVA", "CAUSA", "MERITO", "LEGGE", "NORMA", "ARTICOLO",
        "COMMA", "LETTERA", "TITOLO", "CAPO", "LIBRO", "CODICE",
        "PROCEDURA", "PROCESSO", "RICORSO", "APPELLO", "GRAVAME",
        "DOMANDA", "DIFESA", "ECCEZIONE", "CENSURA", "MOTIVAZIONE",
        "RAGIONI", "DECISIONE", "IMPUGNAZIONE", "INAMMISSIBILE",
        "INAMMISSIBILITÀ", "INAMMISSIBILITA", "INFONDATO", "INFONDATA",
        "ACCOLTO", "RIGETTATO", "CASSAZIONE",
        # Geographic filler commonly appearing uppercase (capoluoghi + città
        # grandi; avoids redacting place-of-court mentions).
        "ROMA", "MILANO", "NAPOLI", "TORINO", "PALERMO", "BARI", "GENOVA",
        "FIRENZE", "BOLOGNA", "CATANIA", "VENEZIA", "VERONA", "PADOVA",
        "PERUGIA", "LECCE", "POTENZA", "CATANZARO", "COSENZA", "REGGIO",
        "CALABRIA", "TRIESTE", "CAGLIARI", "BRINDISI", "TARANTO", "SALERNO",
        "MESSINA", "SIRACUSA", "L'AQUILA", "PESCARA", "CHIETI", "TERAMO",
        "CAMPOBASSO", "ANCONA", "PARMA", "MODENA", "VICENZA", "TREVISO",
        "RAVENNA", "FORLI", "RIMINI", "UDINE", "BRESCIA", "BERGAMO",
        "MONZA", "COMO", "VARESE", "SASSARI", "LATINA", "FROSINONE",
        "ITALIA", "ITALY",
        # Misc procedural acronyms
        "ECLI", "IT", "CASS", "PEN", "CIV", "SEZ", "ART", "RG", "PQM",
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

    # Per-role counters shared across the NER and all-caps layers so a name
    # that appears as 'Paolo Dogliotti' in one place and 'PAOLO DOGLIOTTI'
    # elsewhere ends up under a single, consistent [ROLE_N] token.
    role_counters: dict[str, int] = {}

    # 1) NER first (runs on original text so offsets are valid, then we
    #    substitute; we do this before regex so that person names inside
    #    birth clauses get a stable token before the clause is nuked).
    if cfg.use_ner and ner is not None:
        # Pre-scan: collect 2-6 char ALL-CAPS tokens that appear in the doc.
        # If a NER span (typically a 'Title Case' token like 'Arif' that
        # spaCy mis-tags) matches one of these in uppercase, it's the same
        # acronym and must not be redacted as a person.
        doc_acronyms = {
            tok for tok in re.findall(r"\b[A-Z]{2,6}\b", text)
            if tok not in _PARTICLES
        }
        person_map: dict[str, str] = {}
        # Collect entities, drop junk spans, sort by start for stable numbering.
        entities = [e for e in ner(text) if _is_valid_ner_span(e[0])]
        # Drop entities whose uppercase form matches a document acronym
        # ('Arif' when 'ARIF' is in the same doc, 'Inps' when 'INPS' is, ...).
        entities = [
            e for e in entities
            if _normalise_name(e[0]).upper() not in doc_acronyms
        ]
        entities = sorted(set(entities), key=lambda e: e[1])
        # Track the most recent assigned role and the previous span's end so
        # that "avv. A, B e C" chains all three names under AVVOCATO. A
        # subsequent name inherits the role only if (a) its own lookback
        # detected no role, and (b) the gap is purely a list connector.
        last_role: Optional[str] = None
        last_entity_end: int = -1
        for name, span_start, span_end in entities:
            role = _detect_role(text, span_start)
            if (
                role == "PERSONA"
                and last_role is not None
                and last_entity_end >= 0
            ):
                gap = text[last_entity_end:span_start]
                if _ROLE_CONNECTOR_RE.fullmatch(gap):
                    role = last_role
            last_role = role
            last_entity_end = span_end

            key = _normalise_name(name)
            if key and key not in person_map:
                role_counters[role] = role_counters.get(role, 0) + 1
                person_map[key] = f"[{role}_{role_counters[role]}]"
        if person_map:
            # Replace longest names first to avoid partial overlap bugs.
            # Use word-boundary anchors (\w lookaround) so "Mario" doesn't
            # eat the "mar" inside "marzo" and so short surnames don't
            # bleed into longer words.
            for key in sorted(person_map, key=len, reverse=True):
                token = person_map[key]
                pattern = re.compile(
                    r"(?<!\w)" + re.escape(key) + r"(?!\w)",
                    re.IGNORECASE,
                )
                new_text, n = pattern.subn(token, text)
                if n:
                    stats.person_ner += n
                    text = new_text

    # 2) Regex layers — order matters: catch structured patterns first.
    if cfg.redact_cf:
        text, n = RE_CF.subn("[CODICE_FISCALE]", text)
        stats.codice_fiscale += n
    if cfg.redact_piva:
        # Company C.F. (11-digit numeric) before the phone regex, since an
        # 11-digit landline-like sequence starting with 0 would be captured
        # by RE_PHONE otherwise.
        text, n = RE_CF_AZIENDA.subn("[CODICE_FISCALE]", text)
        stats.codice_fiscale += n
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
        def _addr_sub(m: re.Match[str]) -> str:
            body = m.group(0)
            # Split street-keyword off from its name; the first alphabetic
            # character of the name must be uppercase. Locutions like "in
            # via esclusiva", "in via telematica", "via libera" have a
            # lowercase name and must NOT be redacted.
            parts = body.split(None, 1)
            if len(parts) == 2:
                first_alpha = next(
                    (c for c in parts[1] if c.isalpha()), None
                )
                if first_alpha and first_alpha.islower():
                    return body
            stats.address += 1
            return "[INDIRIZZO]"

        text = RE_ADDRESS.sub(_addr_sub, text)

    # 3) All-caps name heuristic (last, on text with structured PII already
    #    stripped so we don't double-count). Role-aware: a match preceded by
    #    "avvocato"/"consigliere"/"presidente"/"dott." gets the matching
    #    [ROLE_N] token instead of the bare [PERSONA_N], sharing counters
    #    with the NER layer for cross-layer consistency.
    if cfg.redact_allcaps_names:
        allcaps_map: dict[str, str] = {}

        def _sub(m: re.Match[str]) -> str:
            s = m.group(0)
            if not _is_likely_person(s):
                return s
            # Suppress when the match is immediately followed by a company
            # marker (S.P.A., S.R.L., S.N.C., S.A.S., SCARL, SPA, ...): the
            # uppercase run is a ragione sociale, not a person name.
            tail = text[m.end():m.end() + 30]
            if _COMPANY_TAIL_RE.match(tail):
                return s
            # Per-doc stability: same all-caps name → same token throughout.
            key = _normalise_name(s).upper()
            if key in allcaps_map:
                stats.person_allcaps += 1
                return allcaps_map[key]
            role = _detect_role(text, m.start())
            role_counters[role] = role_counters.get(role, 0) + 1
            token = f"[{role}_{role_counters[role]}]"
            allcaps_map[key] = token
            stats.person_allcaps += 1
            return token

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


# Role markers detected in the ~30 chars BEFORE a NER person span. Replace
# the generic `[PERSONA_N]` token with a role-specific one so the model
# preserves structural context (who is the lawyer, who is the judge, who is
# the party) without leaking the actual identity.
#
# Order matters: the first matching pattern wins, so longer/more specific
# patterns come first ("consigliere relatore" before "consigliere").
# Pattern matching is done unanchored: the 30-char lookback window already
# bounds how far before the name we look, so requiring an end anchor only
# breaks chains like "Consigliere Relatore Dott. NAME" where the specific
# role marker is shielded from the name by a generic honorific.
_ROLE_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bconsiglier[ei]\b", re.IGNORECASE), "CONSIGLIERE"),
    (re.compile(r"\bcons\.", re.IGNORECASE), "CONSIGLIERE"),
    (re.compile(r"\bpresident[ei]\b", re.IGNORECASE), "PRESIDENTE"),
    (re.compile(r"\bpres\.", re.IGNORECASE), "PRESIDENTE"),
    (re.compile(r"\bavvocat[oi]\b", re.IGNORECASE), "AVVOCATO"),
    (re.compile(r"\bavv\.", re.IGNORECASE), "AVVOCATO"),
    (re.compile(r"\bdifensor[ei]\b", re.IGNORECASE), "AVVOCATO"),
    (re.compile(r"\bdottor[ei]?\b", re.IGNORECASE), "DOTT"),
    (re.compile(r"\bdott\.|\bdr\.", re.IGNORECASE), "DOTT"),
)

_ROLE_LOOKBACK_CHARS = 30

# Connector patterns used to chain multiple names under the same role marker:
# "avv. A, B e C" means all three are avvocati. The regex must match the
# ENTIRE gap between two adjacent NER spans for the inheritance to fire.
_ROLE_CONNECTOR_RE = re.compile(
    r"\s*(?:"
    r"[,;]\s*(?:e[d]?\s+|o\s+)?"
    r"|\s+e[d]?\s+"
    r"|\s+o\s+"
    r")",
    re.IGNORECASE,
)


def _detect_role(text: str, span_start: int) -> str:
    """Look at the chars immediately before ``span_start`` and return the
    matching role tag, or ``"PERSONA"`` if none of the patterns fits.

    When a specific role (avvocato / consigliere / presidente) AND a generic
    honorific (dott. / dr.) are both visible in the lookback window — as in
    "Consigliere Relatore Dott. X" — the specific role wins, since the title
    is a courtesy, not a procedural position.
    """
    left = max(0, span_start - _ROLE_LOOKBACK_CHARS)
    prefix = text[left:span_start]
    matches = [role for pat, role in _ROLE_RULES if pat.search(prefix)]
    if not matches:
        return "PERSONA"
    specific = [r for r in matches if r != "DOTT"]
    if specific:
        return specific[0]
    return matches[0]


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
            if ent.label_ == "PER" and _is_valid_ner_span(ent.text)
        ]

    return _extract

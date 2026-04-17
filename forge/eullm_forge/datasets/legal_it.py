"""Italian legal corpus preparation for EULLM Forge.

Sources (all public domain under Italian law — art. 5, L. 633/1941):
  - normattiva.it  : official Italian legislation XML (Codice Civile, Penale, ecc.)
  - EUR-Lex        : EU regulations in Italian (GDPR, AI Act)

Output: JSONL files with {"text": "..."} records, one article per record.
The pipeline loads them with datasets.load_dataset("json", data_files=path).
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .base import (
    clean_text,
    http_get,
    save_jsonl,
    train_val_split,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source definitions
# ---------------------------------------------------------------------------

@dataclass
class NormaSource:
    id: str
    name: str
    urn: str
    description: str = ""


@dataclass
class EurlexSource:
    id: str
    name: str
    celex: str
    description: str = ""


# Italian legislation from normattiva.it
NORMATTIVA_LAWS: list[NormaSource] = [
    NormaSource(
        id="costituzione",
        name="Costituzione della Repubblica Italiana",
        urn="urn:nir:stato:costituzione:1947-12-27;0",
        description="Costituzione (139 articoli)",
    ),
    NormaSource(
        id="codice_civile",
        name="Codice Civile",
        urn="urn:nir:stato:regio.decreto:1942-03-16;262",
        description="Codice Civile (2969 articoli)",
    ),
    NormaSource(
        id="codice_penale",
        name="Codice Penale",
        urn="urn:nir:stato:regio.decreto:1930-10-19;1398",
        description="Codice Penale (~750 articoli)",
    ),
    NormaSource(
        id="codice_procedura_civile",
        name="Codice di Procedura Civile",
        urn="urn:nir:stato:regio.decreto:1940-10-28;1443",
        description="Codice di Procedura Civile (~840 articoli)",
    ),
    NormaSource(
        id="codice_procedura_penale",
        name="Codice di Procedura Penale",
        urn="urn:nir:stato:decreto.del.presidente.della.repubblica:1988-09-22;447",
        description="Codice di Procedura Penale (~746 articoli)",
    ),
    NormaSource(
        id="codice_consumo",
        name="Codice del Consumo",
        urn="urn:nir:stato:decreto.legislativo:2005-09-06;206",
        description="Codice del Consumo (D.Lgs. 206/2005)",
    ),
]

# EU regulations from EUR-Lex (Italian version)
EURLEX_REGULATIONS: list[EurlexSource] = [
    EurlexSource(
        id="gdpr",
        name="GDPR — Regolamento Generale sulla Protezione dei Dati",
        celex="32016R0679",
        description="Reg. (UE) 2016/679 — 99 articoli",
    ),
    EurlexSource(
        id="ai_act",
        name="AI Act — Regolamento sull'Intelligenza Artificiale",
        celex="32024R1689",
        description="Reg. (UE) 2024/1689 — 113 articoli",
    ),
    EurlexSource(
        id="dsa",
        name="DSA — Digital Services Act",
        celex="32022R2065",
        description="Reg. (UE) 2022/2065 — 93 articoli",
    ),
    EurlexSource(
        id="dma",
        name="DMA — Digital Markets Act",
        celex="32022R1925",
        description="Reg. (UE) 2022/1925 — 54 articoli",
    ),
]

NORMATTIVA_EXPORT_URL = "https://www.normattiva.it/do/atto/export"
EURLEX_HTML_URL = "https://eur-lex.europa.eu/legal-content/IT/TXT/HTML/"

# dati.normattiva.it OpenData — bulk AKN XML download (all codes in one ZIP)
DATI_NORMATTIVA_DOWNLOAD = "https://dati.normattiva.it/download"

# Corte di Cassazione — italgiure.giustizia.it
# italgiure: URL base confermato (con www), database per sezione
ITALGIURE_BASE = "https://www.italgiure.giustizia.it"
ITALGIURE_CANDIDATES = [
    "https://www.italgiure.giustizia.it/",
    "https://italgiure.giustizia.it/",
]
ITALGIURE_SNCASS = ITALGIURE_CANDIDATES[0]

# Codici database italgiure (visibili dalla homepage)
# SN* = testo integrale sentenze; senza SN = solo massime (più accessibili)
_ITALGIURE_DB: dict[str, dict[str, str]] = {
    "civile":  {"sentenze": "snciv",  "massime": "civile"},
    "penale":  {"sentenze": "snpen",  "massime": "penale"},
    "lavoro":  {"sentenze": "snciv",  "massime": "civile"},   # lavoro = sezione civile
    "tributaria": {"sentenze": "snciv", "massime": "civile"},
}

# Fallback: massime pubbliche dal sito ufficiale Cassazione
CASSAZIONE_MASSIME_URL = (
    "https://www.cortedicassazione.it/corte-di-cassazione/it/sentenze.page"
)

# Sentenze Cassazione (uso interno — non pubblicare il dataset grezzo, GDPR)
@dataclass
class CassazioneSource:
    id: str
    name: str
    sezione: str      # "civile", "penale", "lavoro"
    max_sentences: int = 300
    description: str = ""


# Sorgenti gratuite per giurisprudenza italiana:
#   - italgiure.giustizia.it/sncass/ (SentenzeWeb): full-text di tutte le
#     sentenze Cassazione civili+penali dal 2011, accesso pubblico gratuito.
#     Vedi modulo datasets.italgiure (fetch massivo via Solr).
#   - cortedicassazione.it  : sentenze selezionate + massimario (libero)
#   - cortecostituzionale.it: TUTTE le sentenze CC, API pubblica ECLI (libero)
CASSAZIONE_SOURCES: list[CassazioneSource] = [
    CassazioneSource(
        id="cassazione_civile",
        name="Corte di Cassazione — Sentenze Civili",
        sezione="civile",
        max_sentences=300,
        description="Sentenze civili selezionate da cortedicassazione.it",
    ),
    CassazioneSource(
        id="cassazione_penale",
        name="Corte di Cassazione — Sentenze Penali",
        sezione="penale",
        max_sentences=300,
        description="Sentenze penali selezionate da cortedicassazione.it",
    ),
]

# Corte Costituzionale: accesso libero a tutte le sentenze (ECLI API)
CORTECOSTITUZIONALE_BASE = "https://www.cortecostituzionale.it"
CORTECOSTITUZIONALE_SEARCH = (
    f"{CORTECOSTITUZIONALE_BASE}/actionPronuncia.do"
)


# ---------------------------------------------------------------------------
# dati.normattiva.it OpenData — bulk AKN XML ZIP
# ---------------------------------------------------------------------------

def fetch_normattiva_opendata_zip(
    *,
    local_zip: str | None = None,
    vigenza: str = "VIGENTE",
) -> Optional[bytes]:
    """Download or load the dati.normattiva.it bulk AKN XML ZIP for all codici.

    The OPENDATA portal at dati.normattiva.it offers a 'Codici' collection
    downloadable as AKN (Akoma Ntoso) XML — the official structured format.
    This is far more reliable than HTML scraping.

    Args:
        local_zip: Path to a manually downloaded ZIP (skips HTTP download).
            Download at: https://dati.normattiva.it → Collezioni → Codici
            Format: AKN, Vigenza: VIGENTE, then click Download.
        vigenza: "VIGENTE" (current law) or "ORIGINALE" (historical).

    Returns:
        Raw ZIP bytes, or None on failure.
    """
    if local_zip:
        path = Path(local_zip)
        if not path.exists():
            raise FileNotFoundError(f"ZIP not found: {local_zip}")
        logger.info("Loading normattiva.it OpenData ZIP from %s", local_zip)
        return path.read_bytes()

    # Try automatic download
    url = f"{DATI_NORMATTIVA_DOWNLOAD}?collezione=codici&formato=AKN&vigenza={vigenza}"
    logger.info("Downloading normattiva.it OpenData ZIP (codici, %s)...", vigenza)
    try:
        import requests

        from .base import DEFAULT_HEADERS

        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=300, stream=True)
        r.raise_for_status()
        if "zip" not in r.headers.get("Content-Type", "").lower() and len(r.content) < 10000:
            logger.warning(
                "dati.normattiva.it: unexpected response (not a ZIP). "
                "Download manually from https://dati.normattiva.it → Codici → AKN"
            )
            return None
        return r.content
    except Exception as exc:
        logger.warning(
            "dati.normattiva.it download failed (%s). "
            "Download manually from https://dati.normattiva.it → Codici → AKN, "
            "then pass --normattiva-zip path/to/file.zip",
            exc,
        )
        return None


def parse_normattiva_opendata_zip(
    zip_bytes: bytes,
    source_ids: list[str] | None = None,
) -> dict[str, list[dict]]:
    """Parse an AKN XML ZIP from dati.normattiva.it.

    The ZIP may contain nested folders of any depth.  Every .xml entry is
    inspected; the law identity is read from the AKN <FRBRthis> metadata
    inside the file itself — no filename-pattern hardcoding.

    Args:
        zip_bytes: Raw ZIP file bytes.
        source_ids: Optional list of source IDs to filter (e.g. ["codice_civile"]).
            If None, all recognised laws in the ZIP are parsed.

    Returns:
        Dict mapping source_id → list of article records.
    """
    import io
    import zipfile

    results: dict[str, list[dict]] = {}

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            all_entries = zf.namelist()
            xml_files = [n for n in all_entries if n.lower().endswith(".xml")]
            logger.info(
                "OpenData ZIP: %d total entries, %d XML files", len(all_entries), len(xml_files)
            )

            for name in xml_files:
                try:
                    xml_text = zf.read(name).decode("utf-8", errors="replace")
                except Exception as exc:
                    logger.warning("Cannot read %s: %s", name, exc)
                    continue

                # Identify the law from its AKN metadata — no filename guessing
                sid = _detect_source_from_akn(xml_text)
                if sid is None:
                    logger.debug("Unrecognised law in %s — skipping", name)
                    continue
                if source_ids and sid not in source_ids:
                    continue

                try:
                    records = _parse_akn_xml(xml_text, sid)
                    if records:
                        results[sid] = records
                        logger.info("  OpenData [%s] → %d articles  (%s)", sid, len(records), name)
                    else:
                        logger.warning("  OpenData [%s]: 0 articles extracted from %s", sid, name)
                except Exception as exc:
                    logger.warning("Failed to parse %s: %s", name, exc)

    except zipfile.BadZipFile as exc:
        logger.error("Not a valid ZIP file: %s", exc)

    return results


def _detect_source_from_akn(xml_text: str) -> Optional[str]:
    """Identify which NORMATTIVA_LAWS entry this AKN document belongs to.

    Reads the <FRBRthis value="urn:nir:..."/> element from the AKN metadata
    section and matches it against the URNs declared in NORMATTIVA_LAWS.
    Falls back to scanning the raw text for known URN substrings.
    """
    # AKN 3.0: <FRBRthis value="urn:nir:stato:regio.decreto:1942-03-16;262"/>
    # Try both quoted-attribute forms
    m = re.search(r'<FRBRthis\b[^>]+\bvalue="([^"]+)"', xml_text)
    frbrthis = m.group(1).lower() if m else ""

    for law in NORMATTIVA_LAWS:
        urn_lower = law.urn.lower()
        if urn_lower in frbrthis or frbrthis in urn_lower:
            return law.id
        # Match by the date;number tail (e.g. "1942-03-16;262")
        tail = re.search(r":(\d{4}-\d{2}-\d{2};\d+)$", urn_lower)
        if tail and tail.group(1) in frbrthis:
            return law.id

    # Broader fallback: scan the first 4 KB of the file for any known URN fragment
    header = xml_text[:4096].lower()
    for law in NORMATTIVA_LAWS:
        tail = re.search(r":(\d{4}-\d{2}-\d{2};\d+)$", law.urn.lower())
        if tail and tail.group(1) in header:
            return law.id

    return None


def _parse_akn_xml(xml_text: str, source_id: str) -> list[dict]:
    """Parse an AKN (Akoma Ntoso) XML file and extract article records.

    Handles the AKN 3.0 default namespace
    (xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0") by detecting
    the namespace URI from the parsed root element and using it in all
    element lookups — no fragile regex stripping.
    """
    try:
        root = ET.fromstring(xml_text.encode("utf-8"))
    except ET.ParseError as exc:
        logger.warning("AKN parse error for %s: %s", source_id, exc)
        return []

    # Detect the default namespace from the root tag, e.g.:
    # "{http://docs.oasis-open.org/legaldocml/ns/akn/3.0}akomaNtoso"
    ns_uri = ""
    if root.tag.startswith("{"):
        ns_uri = root.tag[1: root.tag.index("}")]
    ns = f"{{{ns_uri}}}" if ns_uri else ""
    logger.debug("AKN [%s]: namespace=%r root=<%s>", source_id, ns_uri, root.tag.split("}")[-1])

    records = []
    art_count = 0
    for art in root.iter(f"{ns}article"):
        art_count += 1
        num_el = art.find(f".//{ns}num")
        num = (num_el.text or "").strip() if num_el is not None else ""

        heading_el = art.find(f".//{ns}heading")
        heading = " ".join(heading_el.itertext()).strip() if heading_el is not None else ""

        # Collect paragraph text.
        # Older "regio decreto" AKN files (1930s–40s) may store text directly
        # as tail text of child elements or in element names other than <p>/<content>.
        # Strategy: try specific elements first, then fall back to itertext().
        paragraphs: list[str] = []
        for pel in art.iter(f"{ns}p"):
            t = clean_text(" ".join(pel.itertext()))
            if t:
                paragraphs.append(t)
        if not paragraphs:
            for cel in art.iter(f"{ns}content"):
                t = clean_text(" ".join(cel.itertext()))
                if t:
                    paragraphs.append(t)
        if not paragraphs:
            # Final fallback: grab all text in the article subtree.
            # Skip elements whose text we already extracted as num/heading so
            # they're not duplicated in the body field.
            skip_ids = {id(num_el), id(heading_el)} - {id(None)}
            parts: list[str] = []
            for el in art.iter():
                if id(el) in skip_ids:
                    continue
                if el.tag in (f"{ns}num", f"{ns}heading"):
                    continue
                if el.text and el.text.strip():
                    parts.append(el.text.strip())
                if el.tail and el.tail.strip():
                    parts.append(el.tail.strip())
            body = clean_text(" ".join(parts))
        else:
            body = " ".join(paragraphs)

        if len(body) < 10:
            continue

        header = f"Art. {num}" if num else ""
        if heading:
            header += f" ({heading})"
        text = f"{header}\n{body}" if header else body

        records.append({
            "text": text,
            "source": source_id,
            "article_num": num,
            "article_title": heading,
        })

    logger.debug("AKN [%s]: %d <article> elements → %d records", source_id, art_count, len(records))

    if art_count < 10:
        from collections import Counter
        tag_counts = Counter(el.tag.split("}")[-1] for el in root.iter())
        top_tags = dict(tag_counts.most_common(20))
        logger.warning("AKN [%s]: only %d <article> found. Element distribution: %s",
                       source_id, art_count, top_tags)

        fallback: list[dict] = []

        # Fallback 1: NIR — <articolo> invece di <article>
        articolo_list = list(root.iter(f"{ns}articolo")) or list(root.iter("articolo"))
        if articolo_list:
            logger.warning("AKN [%s]: %d <articolo> — NIR parser", source_id, len(articolo_list))
            fallback = _parse_nir_articoli(articolo_list, ns, source_id)

        # Fallback 2: documentCollection — ogni articolo è un <doc>
        if len(fallback) < 10:
            doc_elements = list(root.iter(f"{ns}doc")) or list(root.iter("doc"))
            if len(doc_elements) > 10:
                logger.warning("AKN [%s]: %d <doc> — documentCollection",
                               source_id, len(doc_elements))
                fallback = _parse_akn_doc_collection(doc_elements, ns, source_id)

        # Fallback 3: elementi con eId="art_NNN"
        if len(fallback) < 10:
            eId_articles = [
                el for el in root.iter()
                if re.match(r"art[_-]\d", el.get("eId", ""), re.IGNORECASE)
            ]
            if eId_articles:
                logger.warning("AKN [%s]: %d eId='art_...'", source_id, len(eId_articles))
                fallback = _parse_akn_eId_articles(eId_articles, ns, source_id)

        if fallback:
            records = fallback

    return records


def _parse_akn_doc_collection(
    doc_elements: list[ET.Element], ns: str, source_id: str
) -> list[dict]:
    """Estrae articoli da un documentCollection AKN (regio decreto anni 1930-40).

    Struttura: ogni <doc> corrisponde a un articolo. Il numero è ricavato
    dal valore di FRBRthis (es. "urn:nir:...:1942-03-16;262~art_1" → "1").
    Il testo è in mainBody → paragraph → content → p.
    """
    records = []
    for doc in doc_elements:
        # Numero articolo: ricavato da FRBRWork/FRBRthis value="...~art_N"
        num = ""
        frbrthis_el = (
            doc.find(f".//{ns}FRBRWork/{ns}FRBRthis")
            or doc.find(".//FRBRWork/FRBRthis")
            or doc.find(f".//{ns}FRBRthis")
            or doc.find(".//FRBRthis")
        )
        if frbrthis_el is not None:
            val = frbrthis_el.get("value", "")
            m = re.search(r"~art[_-]?(\w+)", val, re.IGNORECASE)
            if m:
                num = m.group(1)

        # Testo: mainBody → paragraph → content → p
        paragraphs: list[str] = []
        for pel in doc.iter(f"{ns}p"):
            t = clean_text(" ".join(pel.itertext()))
            if t:
                paragraphs.append(t)
        if not paragraphs:
            for cel in doc.iter(f"{ns}content"):
                t = clean_text(" ".join(cel.itertext()))
                if t:
                    paragraphs.append(t)

        body = " ".join(paragraphs)
        if len(body) < 10:
            continue

        header = f"Art. {num}" if num else ""
        text = f"{header}\n{body}" if header else body
        records.append({
            "text": text,
            "source": source_id,
            "article_num": num,
            "article_title": "",
        })
    return records


def _parse_nir_articoli(
    elements: list[ET.Element], ns: str, source_id: str
) -> list[dict]:
    """Parse NIR-format <articolo> elements (leggi italiane anni 1930-40).

    NIR (Norme in Rete) usa nomi di elementi in italiano:
      <articolo>  → articolo
      <rubrica>   → titolo/intestazione
      <comma>     → paragrafo numerato
      <alinea>    → testo di apertura del paragrafo
      <lettera>   → sub-item letterale
      <corpo>     → testo del sub-item
    """
    records = []
    for art in elements:
        num_el = art.find(f".//{ns}num") or art.find(".//num")
        num = (num_el.text or "").strip() if num_el is not None else ""

        rub_el = (
            art.find(f".//{ns}rubrica") or art.find(".//rubrica")
            or art.find(f".//{ns}heading") or art.find(".//heading")
        )
        heading = " ".join(rub_el.itertext()).strip() if rub_el is not None else ""

        skip_tags = {
            f"{ns}num", "num",
            f"{ns}rubrica", "rubrica",
            f"{ns}heading", "heading",
        }
        parts: list[str] = []
        for el in art.iter():
            if el.tag in skip_tags:
                continue
            if el.text and el.text.strip():
                parts.append(el.text.strip())
            if el.tail and el.tail.strip():
                parts.append(el.tail.strip())
        body = clean_text(" ".join(parts))

        if len(body) < 10:
            continue

        header = f"Art. {num}" if num else ""
        if heading:
            header += f" ({heading})"
        text = f"{header}\n{body}" if header else body
        records.append({
            "text": text,
            "source": source_id,
            "article_num": num,
            "article_title": heading,
        })
    return records


def _parse_akn_eId_articles(
    elements: list[ET.Element], ns: str, source_id: str
) -> list[dict]:
    """Extract records from AKN elements identified by eId='art_NNN' attribute.

    Used as fallback when <article> elements are absent but the law's articles
    are encoded as <section>/<chapter>/etc. elements with article eId attributes.
    """
    records = []
    for art in elements:
        num_el = art.find(f".//{ns}num")
        num = (num_el.text or "").strip() if num_el is not None else ""
        if not num:
            # Extract number from eId attribute: "art_1" → "1"
            eid = art.get("eId", "")
            m = re.search(r"art[_-](\d+)", eid, re.IGNORECASE)
            num = m.group(1) if m else eid

        heading_el = art.find(f".//{ns}heading")
        heading = " ".join(heading_el.itertext()).strip() if heading_el is not None else ""

        paragraphs: list[str] = []
        for pel in art.iter(f"{ns}p"):
            t = clean_text(" ".join(pel.itertext()))
            if t:
                paragraphs.append(t)
        if not paragraphs:
            for cel in art.iter(f"{ns}content"):
                t = clean_text(" ".join(cel.itertext()))
                if t:
                    paragraphs.append(t)
        if not paragraphs:
            skip_ids = {id(num_el), id(heading_el)} - {id(None)}
            parts: list[str] = []
            for el in art.iter():
                if id(el) in skip_ids:
                    continue
                if el.tag in (f"{ns}num", f"{ns}heading"):
                    continue
                if el.text and el.text.strip():
                    parts.append(el.text.strip())
                if el.tail and el.tail.strip():
                    parts.append(el.tail.strip())
            body = clean_text(" ".join(parts))
        else:
            body = " ".join(paragraphs)

        if len(body) < 10:
            continue

        header = f"Art. {num}" if num else ""
        if heading:
            header += f" ({heading})"
        text = f"{header}\n{body}" if header else body
        records.append({
            "text": text,
            "source": source_id,
            "article_num": num,
            "article_title": heading,
        })
    return records


# ---------------------------------------------------------------------------
# normattiva.it — XML download and parsing
# ---------------------------------------------------------------------------

def make_normattiva_session() -> object:
    """Create and warm up a requests.Session for normattiva.it.

    Visits the main page to establish a JSESSIONID cookie. Reuse this session
    across all normattiva.it downloads to avoid repeated authentication overhead.
    """
    try:
        import requests
    except ImportError:
        raise RuntimeError("requests is required. Install with: pip install requests")

    from .base import DEFAULT_HEADERS

    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    try:
        session.get("https://www.normattiva.it/", timeout=15)
    except Exception as exc:
        logger.warning("Could not warm up normattiva.it session: %s", exc)
    return session


def fetch_normattiva(source: NormaSource, session: object = None) -> Optional[str]:
    """Download law XML from normattiva.it.

    normattiva.it is a Java EE application that requires a JSESSIONID session
    cookie. A cold GET to the export endpoint returns an HTML page instead of XML.

    Pass a session returned by ``make_normattiva_session()`` to share one
    authenticated session across multiple law downloads.

    Returns raw XML string, or None on failure.
    Cached locally at ~/.cache/eullm-forge/raw/ to avoid re-downloading.
    """
    from .base import _cache_load, _cache_save

    cache_key = f"normattiva_{source.id}"

    # Check cache — reject stale HTML responses
    cached = _cache_load(cache_key)
    if cached is not None and not cached.lstrip().startswith("<!"):
        logger.debug("Cache hit: %s", cache_key)
        return cached

    if session is None:
        session = make_normattiva_session()

    url = f"{NORMATTIVA_EXPORT_URL}?urn={source.urn}&includiAllegati=N"
    logger.info("Fetching %s from normattiva.it...", source.name)

    try:
        # Visit the law's viewer page — sets Referer and activates session context
        viewer_url = f"https://www.normattiva.it/uri-res/N2Ls?{source.urn}"
        session.get(viewer_url, timeout=30)
        # Download XML export with Referer set to the viewer page
        response = session.get(
            url,
            timeout=60,
            headers={"Referer": viewer_url},
        )
        response.raise_for_status()
        content = response.text

        # Validate it's actually XML — HTML error pages start with <!DOCTYPE
        # normattiva XML uses <articolo> (NIR) or <article> (AKN) elements
        if content.lstrip().startswith("<!") or not re.search(r"<[Aa]rt", content):
            logger.warning(
                "normattiva.it returned HTML for %s — trying HTML viewer scrape",
                source.id,
            )
            return _scrape_normattiva_html(session, source)

        _cache_save(cache_key, content)
        return content

    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", source.name, exc)
        return None


def _scrape_normattiva_html(session: object, source: NormaSource) -> Optional[str]:
    """Fallback: download the full act via the normattiva.it 'attoCompleto' endpoint.

    The viewer page has an 'esporta/attoCompleto' link that returns the full law
    as a single HTML page (~1–3MB) with all articles pre-rendered:
        <div class="bodyTesto">
            <h2 class="article-num-akn">Art. N</h2>
            <span class="art-just-text-akn">...</span>
        </div>

    This replaces the previous AJAX approach (N individual requests) with a
    single bulk download, avoiding rate limiting entirely.
    """
    try:
        import warnings

        from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    except ImportError:
        return None

    from .base import DEFAULT_HEADERS, _cache_save

    # Use a fresh session for each bulk download — reusing a session that has
    # already fetched a large attoCompleto causes normattiva.it to return a
    # reduced "one article" page on subsequent requests.
    try:
        import requests as _requests
        fresh_session = _requests.Session()
        fresh_session.headers.update(DEFAULT_HEADERS)
        fresh_session.get("https://www.normattiva.it/", timeout=15)
    except Exception:
        fresh_session = session  # fall back to shared session

    # Step 1: load viewer page to find the attoCompleto link
    viewer_url = f"https://www.normattiva.it/uri-res/N2Ls?{source.urn}"
    try:
        resp = fresh_session.get(viewer_url, timeout=60)
        resp.raise_for_status()
        viewer_html = resp.text
    except Exception as exc:
        logger.warning("normattiva.it viewer failed for %s: %s", source.id, exc)
        return None

    soup_v = BeautifulSoup(viewer_html, "html.parser")
    link = soup_v.find("a", href=re.compile(r"attoCompleto"))
    if not link:
        logger.warning("No attoCompleto link found for %s", source.id)
        return None

    # Step 2: download the full act in one request
    full_url = "https://www.normattiva.it" + link["href"]
    logger.info("  Downloading full act for %s...", source.id)
    try:
        r = fresh_session.get(
            full_url,
            timeout=120,
            headers={"Referer": viewer_url},
        )
        r.raise_for_status()
        full_html = r.text
    except Exception as exc:
        logger.warning("normattiva.it attoCompleto failed for %s: %s", source.id, exc)
        return None

    # Step 3: extract articles from the rendered HTML
    soup = BeautifulSoup(full_html, "html.parser")
    xml_parts = ["<atto>"]
    for body in soup.find_all(class_="bodyTesto"):
        num_el = body.find(class_="article-num-akn")
        txt_el = body.find(class_="art-just-text-akn")
        if not txt_el:
            continue
        num = num_el.get_text(strip=True) if num_el else ""
        text = " ".join(txt_el.get_text(separator=" ").split())
        if len(text) < 15:
            continue
        num_s = num.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        txt_s = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        xml_parts.append(
            f"<articolo><num>{num_s}</num><testo>{txt_s}</testo></articolo>"
        )
    xml_parts.append("</atto>")

    if len(xml_parts) <= 2:
        logger.warning("No articles extracted from attoCompleto for %s", source.id)
        return None

    result = "\n".join(xml_parts)
    _cache_save(f"normattiva_{source.id}", result)
    logger.info("  Extracted %d articles for %s", len(xml_parts) - 2, source.id)
    return result


def parse_normattiva_xml(xml_text: str, source_id: str) -> list[dict]:
    """Extract article records from normattiva.it XML.

    Handles NIR (Norme in Rete) XML schema variations by searching for
    article elements recursively. Falls back to regex extraction.

    Returns list of dicts with keys: text, source, article_num, article_title.
    """
    records = []

    # Normalize namespace prefixes — ElementTree is picky about them
    xml_clean = _strip_xml_namespaces(xml_text)

    try:
        root = ET.fromstring(xml_clean)
    except ET.ParseError as exc:
        logger.warning("XML parse error for %s: %s — trying regex fallback", source_id, exc)
        return _parse_normattiva_regex(xml_text, source_id)

    # Find article elements — NIR schema uses various casings
    articles = (
        root.findall(".//articolo")
        or root.findall(".//Articolo")
        or root.findall(".//art")
    )

    if not articles:
        logger.warning(
            "No <articolo> elements found in %s, falling back to text chunking",
            source_id,
        )
        full_text = " ".join(root.itertext())
        return _text_to_records(full_text, source_id)

    for article in articles:
        record = _extract_article_record(article, source_id)
        if record:
            records.append(record)

    logger.info("  Extracted %d articles from %s", len(records), source_id)
    return records


def _strip_xml_namespaces(xml_text: str) -> str:
    """Remove XML namespace declarations and prefixes for simpler parsing."""
    # Remove namespace declarations
    xml_text = re.sub(r'\s+xmlns(?::[a-zA-Z0-9_]+)?="[^"]*"', "", xml_text)
    # Remove namespace prefixes from tags: <nir:Articolo> → <Articolo>
    xml_text = re.sub(r"<([a-zA-Z]+):", "<", xml_text)
    xml_text = re.sub(r"</([a-zA-Z]+):", "</", xml_text)
    return xml_text


def _extract_article_record(article: ET.Element, source_id: str) -> Optional[dict]:
    """Extract text and metadata from a single <articolo> element."""
    # Article number — try multiple element names
    num_elem = (
        article.find(".//num")
        or article.find(".//Num")
        or article.find(".//numeroArticolo")
        or article.find(".//NumeroArticolo")
    )
    num = (num_elem.text or "").strip() if num_elem is not None else ""

    # Article title/rubric
    rubrica_elem = (
        article.find(".//rubrica")
        or article.find(".//Rubrica")
        or article.find(".//intestazione")
    )
    rubrica = ""
    if rubrica_elem is not None:
        rubrica = " ".join(rubrica_elem.itertext()).strip()
        # Remove the article number from the rubrica if it starts with it
        rubrica = re.sub(rf"^{re.escape(num)}\s*[\.\-\s]", "", rubrica).strip()

    # Full text content
    all_text = clean_text(" ".join(article.itertext()))

    if len(all_text) < 30:
        return None

    # Format: "Art. N (Rubrica)\nContent"
    header_parts = []
    if num:
        header_parts.append(f"Art. {num}")
    if rubrica:
        header_parts.append(f"({rubrica})")
    header = " ".join(header_parts)

    full_text = f"{header}\n{all_text}" if header else all_text

    return {
        "text": full_text,
        "source": source_id,
        "article_num": num,
        "article_title": rubrica,
    }


def _parse_normattiva_regex(xml_text: str, source_id: str) -> list[dict]:
    """Fallback: extract article text using regex when XML parsing fails."""
    records = []

    # Try to find article blocks: <articolo ...>...</articolo>
    article_blocks = re.findall(
        r"<[Aa]rticolo[^>]*>(.*?)</[Aa]rticolo>",
        xml_text,
        re.DOTALL,
    )

    if not article_blocks:
        # Last resort: strip all XML tags and chunk plain text
        plain = re.sub(r"<[^>]+>", " ", xml_text)
        return _text_to_records(plain, source_id)

    for i, block in enumerate(article_blocks, 1):
        plain = re.sub(r"<[^>]+>", " ", block)
        text = clean_text(plain)
        if len(text) >= 30:
            records.append({
                "text": text,
                "source": source_id,
                "article_num": str(i),
                "article_title": "",
            })

    logger.info("  Regex fallback: extracted %d articles from %s", len(records), source_id)
    return records


def _text_to_records(text: str, source_id: str, chunk_chars: int = 3000) -> list[dict]:
    """Split a long text into fixed-size chunks as a last-resort fallback."""
    text = clean_text(text)
    records = []
    for i in range(0, len(text), chunk_chars):
        chunk = text[i : i + chunk_chars].strip()
        if len(chunk) >= 50:
            records.append({
                "text": chunk,
                "source": source_id,
                "article_num": str(i // chunk_chars + 1),
                "article_title": "",
            })
    return records


# ---------------------------------------------------------------------------
# EUR-Lex — HTML download and parsing
# ---------------------------------------------------------------------------

def fetch_eurlex(source: EurlexSource) -> Optional[str]:
    """Download EU regulation HTML from EUR-Lex (Italian version).

    EUR-Lex is protected by AWS WAF which requires JavaScript execution.
    Fetching is attempted in order:
      1. Local cache (fast path)
      2. Playwright headless browser (if installed: pip install playwright
         && playwright install chromium)
      3. Plain HTTP GET (works only without WAF — may fail silently)

    Returns raw HTML string, or None on failure.
    """
    from .base import _cache_load, _cache_path, _cache_save

    cache_key = f"eurlex_{source.id}"

    # Check cache — reject stale WAF placeholder responses (~2KB)
    cached = _cache_load(cache_key)
    if cached is not None:
        if _eurlex_content_valid(cached):
            logger.debug("Cache hit: %s", cache_key)
            return cached
        _cache_path(cache_key).unlink(missing_ok=True)

    url = f"{EURLEX_HTML_URL}?uri=CELEX:{source.celex}"
    logger.info("Fetching %s from EUR-Lex...", source.name)

    # Try Playwright first (bypasses AWS WAF JavaScript challenge)
    html = _fetch_eurlex_playwright(url, source.id)
    if html and _eurlex_content_valid(html):
        _cache_save(cache_key, html)
        return html

    # Plain HTTP fallback (works only if WAF is inactive for the IP)
    import time
    for attempt in range(3):
        html = http_get(url, cache_key=None)
        if html and _eurlex_content_valid(html):
            _cache_save(cache_key, html)
            return html
        if html is not None:
            logger.warning(
                "EUR-Lex WAF placeholder for %s (attempt %d/3, len=%d) — "
                "install Playwright to bypass: "
                "pip install playwright && playwright install chromium",
                source.id, attempt + 1, len(html),
            )
        if attempt < 2:
            time.sleep(5 * (attempt + 1))

    logger.warning(
        "Could not fetch %s — EUR-Lex blocks automated access via AWS WAF. "
        "Fix: pip install playwright && playwright install chromium",
        source.name,
    )
    return None


def _fetch_eurlex_playwright(url: str, source_id: str) -> Optional[str]:
    """Fetch a EUR-Lex page using Playwright (headless Chromium).

    Resolves the AWS WAF JavaScript challenge automatically.
    Requires: pip install playwright && playwright install chromium
    """
    try:
        from playwright.sync_api import TimeoutError as PWTimeout
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None  # Playwright not installed — fall through to plain HTTP

    logger.info("  Using Playwright for %s (resolving AWS WAF challenge)...", source_id)
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            ctx = browser.new_context(
                locale="it-IT",
                extra_http_headers={"Accept-Language": "it-IT,it;q=0.9"},
            )
            page = ctx.new_page()
            # Wait for network to be idle so WAF challenge completes
            page.goto(url, wait_until="networkidle", timeout=60_000)
            html = page.content()
            browser.close()
        return html
    except PWTimeout:
        logger.warning("Playwright timeout fetching %s", source_id)
        return None
    except Exception as exc:
        logger.warning("Playwright error fetching %s: %s", source_id, exc)
        return None


def _eurlex_content_valid(html: str) -> bool:
    """Return True if the HTML looks like real regulation content (not a WAF placeholder)."""
    return len(html) > 5000 and "Articolo" in html


def parse_eurlex_html(html: str, source_id: str) -> list[dict]:
    """Extract article records from EUR-Lex HTML.

    Uses a text-split approach: strip all HTML, then split the plain text by
    "Articolo N" boundaries. This is robust against EUR-Lex HTML structure
    variations (different CSS classes, TOC duplicates, nested spans, etc.).

    Returns list of dicts with keys: text, source, article_num, article_title.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise RuntimeError(
            "beautifulsoup4 is required for EUR-Lex parsing. "
            "Install with: pip install beautifulsoup4"
        )

    soup = BeautifulSoup(html, "html.parser")

    # Remove navigation noise before extracting text
    for tag in soup.find_all(["nav", "header", "footer", "script", "style", "noscript"]):
        tag.decompose()

    # Get full plain text — preserving line breaks between elements
    full_text = soup.get_text(separator="\n", strip=True)

    # Split on "Articolo N" boundaries.
    # Pattern captures the article header line (num + optional title on same line).
    # re.split with a capturing group keeps the delimiters in the result list.
    parts = re.split(
        r"\n(Articolo\s+\d+[^\n]*)",
        full_text,
        flags=re.IGNORECASE,
    )
    # parts = [preamble, "Articolo 1 ...", content1, "Articolo 2 ...", content2, ...]

    records: list[dict] = []
    i = 1  # skip preamble at index 0
    while i + 1 < len(parts):
        header = parts[i].strip()
        content = clean_text(parts[i + 1])
        i += 2

        m = re.match(r"Articolo\s+(\d+)\s*(.*)", header, re.IGNORECASE)
        if not m or len(content) < 20:
            continue

        num = m.group(1)
        title = clean_text(m.group(2).lstrip("—–- "))
        full = f"{header}\n{content}"
        records.append({
            "text": full,
            "source": source_id,
            "article_num": num,
            "article_title": title,
        })

    if not records:
        # Fallback: chunk the full plain text
        return _text_to_records(full_text, source_id)

    logger.info("  Extracted %d articles from %s", len(records), source_id)
    return records


# ---------------------------------------------------------------------------
# Giurisprudenza — sorgenti gratuite
#
# Sorgenti libere con full-text:
#   1. italgiure.giustizia.it/sncass/ (SentenzeWeb) — full-text Cassazione
#      civili+penali dal 2011. Vedi datasets.italgiure.fetch_italgiure().
#   2. cortedicassazione.it — sentenze selezionate Cassazione (solo sintesi HTML)
#   3. cortecostituzionale.it — TUTTE le sentenze CC (API pubblica ECLI)
# ---------------------------------------------------------------------------

def fetch_cassazione(source: CassazioneSource) -> list[dict]:
    """Scarica sentenze Cassazione da cortedicassazione.it (accesso libero).

    Il sito ufficiale della Corte di Cassazione pubblica sentenze selezionate
    in HTML senza richiedere registrazione o abbonamento.

    Le sentenze vengono salvate in cache locale.
    Non pubblicare il dataset grezzo (GDPR: contiene dati personali delle parti).

    Returns:
        Lista di record {text, source, sentence_id}.
    """
    import json

    from .base import _cache_load, _cache_save

    cache_key = f"cassazione_{source.id}"
    cached = _cache_load(cache_key)
    if cached:
        try:
            records = json.loads(cached)
            if records:
                logger.debug("Cache hit: %s (%d sentenze)", cache_key, len(records))
                return records
        except Exception:
            pass

    logger.info("Scaricando sentenze %s da cortedicassazione.it...", source.name)
    records = _fetch_cassazione_cortedicassazione(source)

    if records:
        _cache_save(cache_key, json.dumps(records, ensure_ascii=False))
        logger.info("  [%s] %d sentenze", source.id, len(records))
    else:
        logger.warning(
            "Nessuna sentenza per %s da cortedicassazione.it. "
            "Installa Playwright per il fetch headless: "
            "pip install playwright && playwright install chromium",
            source.name,
        )
    return records




    return records


def _fetch_cassazione_cortedicassazione(source: CassazioneSource) -> list[dict]:
    """Scarica sentenze dal sito ufficiale cortedicassazione.it (accesso libero).

    Il sito pubblica sentenze selezionate in HTML senza richiedere login.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return []

    records: list[dict] = []
    # URL sezione-specifica per cortedicassazione.it
    sezione_urls = {
        "civile": (
            "https://www.cortedicassazione.it/corte-di-cassazione/it/"
            "sentenze_civili.page"
        ),
        "penale": (
            "https://www.cortedicassazione.it/corte-di-cassazione/it/"
            "sentenze_penali.page"
        ),
        "lavoro": (
            "https://www.cortedicassazione.it/corte-di-cassazione/it/"
            "sentenze_lavoro.page"
        ),
    }
    url = sezione_urls.get(source.sezione, CASSAZIONE_MASSIME_URL)
    logger.info("  cortedicassazione.it: %s", url)

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(locale="it-IT")
            page.goto(url, wait_until="networkidle", timeout=60_000)

            # Raccoglie link a sentenze individuali
            links = page.eval_on_selector_all(
                "a[href]",
                "els => els.map(e => e.href).filter("
                "h => h && (h.includes('sentenz') || h.includes('ordinanz') "
                "|| h.includes('pronunce') || h.includes('decisioni')))",
            )
            if not links:
                # Diagnostica: mostra tutti i link della pagina
                all_links = page.eval_on_selector_all(
                    "a[href]", "els => els.slice(0,15).map(e => e.href)"
                )
                logger.info("  cortedicassazione.it: 0 link sentenze. "
                            "Sample link pagina: %s", all_links)
            else:
                logger.info("  cortedicassazione.it: %d link trovati", len(links))

            for href in links[: source.max_sentences]:
                full_url = (
                    href if href.startswith("http")
                    else f"https://www.cortedicassazione.it{href}"
                )
                try:
                    page.goto(full_url, wait_until="domcontentloaded", timeout=30_000)
                    rec = _parse_cassazione_page(page.content(), source.id, full_url)
                    if rec:
                        records.append(rec)
                except Exception:
                    continue

            browser.close()
    except Exception as exc:
        logger.warning("cortedicassazione.it errore: %s", exc)

    return records


def _parse_cassazione_page(html: str, source_id: str, url: str) -> Optional[dict]:
    """Estrae il testo da una pagina sentenza di italgiure.

    Le sentenze hanno struttura:
      - Header: sezione, numero, data
      - Sezioni "FATTO", "DIRITTO"/"MOTIVI", "P.Q.M."

    Per il training estraiamo l'intera motivazione (DIRITTO + FATTO).
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return None

    soup = BeautifulSoup(html, "html.parser")

    # Rimuovi nav/script/style
    for tag in soup.find_all(["nav", "header", "footer", "script", "style", "noscript"]):
        tag.decompose()

    full_text = soup.get_text(separator="\n", strip=True)

    # Estratto minimo: deve contenere testo legale sostanziale
    if len(full_text) < 500 or not any(
        kw in full_text for kw in ("Corte di Cassazione", "Cassazione", "ricorso", "motivo")
    ):
        return None

    # Prova a isolare la motivazione (testo dopo "FATTO" o "DIRITTO")
    # Le sentenze italiane hanno sezioni ben marcate in maiuscolo
    body = full_text
    for marker in ("MOTIVI DELLA DECISIONE", "DIRITTO", "FATTO E DIRITTO", "FATTO"):
        idx = full_text.upper().find(marker)
        if idx != -1:
            body = full_text[idx:]
            break

    body = clean_text(body)
    if len(body) < 300:
        body = clean_text(full_text)  # usa tutto se il ritaglio è troppo corto

    if len(body) < 300:
        return None

    # Numero sentenza dall'URL o dall'HTML
    sentence_id = re.search(r"[/=](\d{4,})", url)
    sid = sentence_id.group(1) if sentence_id else url.split("/")[-1]

    return {
        "text": body,
        "source": source_id,
        "sentence_id": sid,
        "article_num": sid,
        "article_title": "",
    }


# ---------------------------------------------------------------------------
# Corte Costituzionale — accesso libero a tutte le sentenze
# ---------------------------------------------------------------------------

def fetch_corte_costituzionale(max_sentences: int = 300) -> list[dict]:
    """Scarica sentenze della Corte Costituzionale (accesso completamente libero).

    Il sito cortecostituzionale.it pubblica TUTTE le sentenze in HTML senza
    richiedere login, abbonamento o registrazione. Sono pubblico dominio.

    La ricerca usa il form pubblico su actionPronuncia.do con parametri GET.

    Returns:
        Lista di record {text, source, sentence_id}.
    """
    import json

    from .base import _cache_load, _cache_save

    cache_key = "corte_costituzionale"
    cached = _cache_load(cache_key)
    if cached:
        try:
            records = json.loads(cached)
            if records:
                logger.debug("Cache hit: corte_costituzionale (%d sentenze)", len(records))
                return records
        except Exception:
            pass

    logger.info("Scaricando sentenze Corte Costituzionale...")
    records = _fetch_cc_playwright(max_sentences)
    if not records:
        records = _fetch_cc_requests(max_sentences)

    if records:
        _cache_save(cache_key, json.dumps(records, ensure_ascii=False))
        logger.info("  [corte_costituzionale] %d sentenze", len(records))
    else:
        logger.warning(
            "Nessuna sentenza Corte Costituzionale. "
            "Installa Playwright: pip install playwright && playwright install chromium"
        )
    return records


def _fetch_cc_playwright(max_sentences: int) -> list[dict]:
    """Scraper Playwright per cortecostituzionale.it."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return []

    records: list[dict] = []
    # Ricerca per anno corrente e precedente — sentenze recenti
    import datetime
    anni = [datetime.date.today().year, datetime.date.today().year - 1]

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(locale="it-IT")

            for anno in anni:
                if len(records) >= max_sentences:
                    break
                # URL ricerca per anno: parametri GET del form pubblico
                search_url = (
                    f"{CORTECOSTITUZIONALE_BASE}/actionPronuncia.do"
                    f"?anno={anno}&tipoatto=S&Submit=Cerca"
                )
                logger.info("  CC: ricerca anno %d", anno)
                # domcontentloaded evita timeout su siti con analytics attivi
                page.goto(search_url, wait_until="domcontentloaded", timeout=45_000)
                page.wait_for_timeout(2000)  # breve attesa per il rendering

                # Raccoglie link alle sentenze
                links = page.eval_on_selector_all(
                    "a[href]",
                    "els => els.map(e => e.href).filter(h =>"
                    " h.includes('actionPronuncia') && h.includes('idAct'))",
                )
                if not links:
                    # prova selettore più ampio
                    links = page.eval_on_selector_all(
                        "a[href*='Pronuncia']",
                        "els => els.map(e => e.href)",
                    )
                logger.info("  CC anno %d: %d link trovati", anno, len(links))

                for href in links[: max_sentences - len(records)]:
                    try:
                        page.goto(href, wait_until="domcontentloaded", timeout=30_000)
                        rec = _parse_cc_page(page.content(), href)
                        if rec:
                            records.append(rec)
                    except Exception:
                        continue

            browser.close()
    except Exception as exc:
        logger.warning("CC Playwright errore: %s", exc)

    return records


def _fetch_cc_requests(max_sentences: int) -> list[dict]:
    """Fallback requests per cortecostituzionale.it (nessun WAF, accesso libero)."""
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        return []

    import datetime

    from .base import DEFAULT_HEADERS

    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    records: list[dict] = []

    for anno in [datetime.date.today().year, datetime.date.today().year - 1]:
        if len(records) >= max_sentences:
            break
        search_url = (
            f"{CORTECOSTITUZIONALE_BASE}/actionPronuncia.do"
            f"?anno={anno}&tipoatto=S&Submit=Cerca"
        )
        logger.info("  CC requests: %s", search_url)
        try:
            r = session.get(search_url, timeout=30)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            # Prova diversi pattern di link
            links = [
                a["href"] for a in soup.find_all("a", href=True)
                if any(k in a["href"] for k in ("idAct", "pronunzia", "Pronuncia"))
            ]
            logger.info("  CC requests anno %d: %d link trovati (HTML len=%d)",
                        anno, len(links), len(r.text))
            if not links:
                # Mostra i primi link per diagnostica
                sample = [a["href"] for a in soup.find_all("a", href=True)][:10]
                logger.info("  CC: sample link dalla pagina: %s", sample)

            for href in links[: max_sentences - len(records)]:
                full_url = (
                    href if href.startswith("http")
                    else f"{CORTECOSTITUZIONALE_BASE}{href}"
                )
                try:
                    resp = session.get(full_url, timeout=20)
                    rec = _parse_cc_page(resp.text, full_url)
                    if rec:
                        records.append(rec)
                except Exception:
                    continue
        except Exception as exc:
            logger.warning("CC requests errore anno %d: %s", anno, exc)
            continue

    return records


def _parse_cc_page(html: str, url: str) -> Optional[dict]:
    """Estrae il testo da una pagina sentenza della Corte Costituzionale."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return None

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["nav", "header", "footer", "script", "style"]):
        tag.decompose()

    full_text = soup.get_text(separator="\n", strip=True)
    if len(full_text) < 500 or "Corte Costituzionale" not in full_text:
        return None

    # Isola la motivazione
    body = full_text
    for marker in ("CONSIDERATO IN DIRITTO", "RITENUTO IN FATTO", "IN DIRITTO"):
        idx = full_text.upper().find(marker)
        if idx != -1:
            body = full_text[idx:]
            break

    body = clean_text(body)
    if len(body) < 300:
        body = clean_text(full_text)
    if len(body) < 300:
        return None

    sid = re.search(r"[/=&](\d{3,})", url)
    sentence_id = sid.group(1) if sid else url.split("/")[-1]

    return {
        "text": body,
        "source": "corte_costituzionale",
        "sentence_id": sentence_id,
        "article_num": sentence_id,
        "article_title": "",
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def prepare_legal_it(
    output_dir: str | Path,
    *,
    sources: list[str] | None = None,
    push_to_hub: bool = False,
    hub_repo: str = "eullm/legal-it-corpus",
    val_ratio: float = 0.05,
    no_cache: bool = False,
    normattiva_zip: str | None = None,
    max_cassazione_sentences: int = 300,
) -> Path:
    """Download and prepare the Italian legal corpus.

    Downloads Italian legislation from normattiva.it, EU regulations from
    EUR-Lex, and Corte di Cassazione sentences from italgiure.giustizia.it.

    Args:
        output_dir: Directory where train.jsonl and val.jsonl will be saved.
        sources: Source IDs to include (None = all). E.g. ["costituzione", "gdpr"].
            Cassazione IDs: cassazione_civile, cassazione_penale, cassazione_lavoro.
        push_to_hub: If True, push dataset to HuggingFace Hub.
            WARNING: non usare con sentenze Cassazione (GDPR — dati personali).
        hub_repo: HuggingFace Hub repo ID for push (default: eullm/legal-it-corpus).
        val_ratio: Fraction of records to put in validation split.
        no_cache: If True, bypass local HTTP cache and re-download all sources.
        normattiva_zip: Path to a locally downloaded AKN ZIP from dati.normattiva.it.
            If provided (or if the automatic download succeeds), this is used instead
            of the HTML scraper for normattiva.it sources.
            Get it at: https://dati.normattiva.it → Collezioni → Codici → AKN → Download
        max_cassazione_sentences: Max sentenze per sezione Cassazione (default 300).

    Returns:
        Path to the output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if no_cache:
        import shutil

        from .base import CACHE_DIR
        logger.info("--no-cache: clearing HTTP cache at %s", CACHE_DIR)
        shutil.rmtree(CACHE_DIR, ignore_errors=True)

    all_records: list[dict] = []
    failed_sources: list[str] = []

    # --- normattiva.it sources ---
    normattiva_laws = [law for law in NORMATTIVA_LAWS if not sources or law.id in sources]

    # Prefer dati.normattiva.it OpenData ZIP (best quality, no scraping needed)
    opendata_results: dict[str, list[dict]] = {}
    if normattiva_laws:
        wanted_ids = [law.id for law in normattiva_laws]
        zip_bytes = fetch_normattiva_opendata_zip(local_zip=normattiva_zip)
        if zip_bytes:
            opendata_results = parse_normattiva_opendata_zip(zip_bytes, wanted_ids)

    # For sources not covered by OpenData, fall back to HTML scraping
    normattiva_session = None
    scrape_laws = [law for law in normattiva_laws if law.id not in opendata_results]
    if scrape_laws:
        normattiva_session = make_normattiva_session()

    for law in normattiva_laws:
        if law.id in opendata_results:
            records = opendata_results[law.id]
        else:
            xml_text = fetch_normattiva(law, session=normattiva_session)
            if xml_text is None:
                failed_sources.append(law.id)
                continue
            records = parse_normattiva_xml(xml_text, law.id)

        if not records:
            logger.warning("No articles extracted from %s", law.name)
            failed_sources.append(law.id)
            continue

        all_records.extend(records)
        logger.info("  [%s] %d articles", law.id, len(records))

    # --- EUR-Lex sources ---
    for reg in EURLEX_REGULATIONS:
        if sources and reg.id not in sources:
            continue

        html = fetch_eurlex(reg)
        if html is None:
            failed_sources.append(reg.id)
            continue

        records = parse_eurlex_html(html, reg.id)
        if not records:
            logger.warning("No articles extracted from %s", reg.name)
            failed_sources.append(reg.id)
            continue

        all_records.extend(records)
        logger.info("  [%s] %d articles", reg.id, len(records))

    # --- Corte Costituzionale ---
    if not sources or "corte_costituzionale" in sources:
        cc_records = fetch_corte_costituzionale(max_sentences=max_cassazione_sentences)
        if cc_records:
            all_records.extend(cc_records)
            logger.info("  [corte_costituzionale] %d sentenze", len(cc_records))
        else:
            failed_sources.append("corte_costituzionale")

    # --- Corte di Cassazione (sentenze selezionate da cortedicassazione.it) ---
    for cass in CASSAZIONE_SOURCES:
        if sources and cass.id not in sources:
            continue

        # Applica il limite personalizzato
        cass_source = CassazioneSource(
            id=cass.id,
            name=cass.name,
            sezione=cass.sezione,
            max_sentences=max_cassazione_sentences,
            description=cass.description,
        )
        records = fetch_cassazione(cass_source)
        if not records:
            logger.warning("Nessuna sentenza per %s", cass.name)
            failed_sources.append(cass.id)
            continue

        all_records.extend(records)
        logger.info("  [%s] %d sentenze", cass.id, len(records))

    if not all_records:
        raise RuntimeError(
            "No records collected. All sources failed. "
            "Check your internet connection and try again."
        )

    if failed_sources:
        logger.warning(
            "Sources that failed (will be absent from dataset): %s",
            ", ".join(failed_sources),
        )

    # Shuffle deterministically (by article order, no random seed needed —
    # sources are interleaved naturally by list order above)
    train_records, val_records = train_val_split(all_records, val_ratio)

    # Save splits
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    meta_path = output_dir / "dataset_info.json"

    save_jsonl(train_records, train_path)
    save_jsonl(val_records, val_path)

    # Write dataset info
    import json
    source_counts: dict[str, int] = {}
    for rec in all_records:
        source_counts[rec["source"]] = source_counts.get(rec["source"], 0) + 1

    info = {
        "name": "eullm/legal-it-corpus",
        "description": (
            "Italian legal corpus: Codice Civile, Penale, Costituzione, GDPR, AI Act, ecc."
        ),
        "language": ["it"],
        "license": "public-domain",
        "total_records": len(all_records),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "sources": source_counts,
        "failed_sources": failed_sources,
    }
    meta_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(
        "Dataset ready: %d train + %d val records → %s",
        len(train_records), len(val_records), output_dir,
    )

    # Optionally push to HuggingFace Hub
    if push_to_hub:
        _push_to_hub(train_path, val_path, hub_repo)

    return output_dir


def _push_to_hub(train_path: Path, val_path: Path, repo_id: str) -> None:
    """Push prepared dataset to HuggingFace Hub."""
    try:
        from datasets import Dataset, DatasetDict
    except ImportError:
        logger.error("datasets package required for Hub push. Install: pip install datasets")
        return

    try:
        train_ds = Dataset.from_json(str(train_path))
        val_ds = Dataset.from_json(str(val_path))
        ds_dict = DatasetDict({"train": train_ds, "validation": val_ds})
        ds_dict.push_to_hub(repo_id, private=False)
        logger.info(
            "Dataset pushed to HuggingFace Hub: https://huggingface.co/datasets/%s", repo_id
        )
    except Exception as exc:
        logger.error("Failed to push to Hub: %s", exc)
        logger.info("Make sure you are logged in: huggingface-cli login")

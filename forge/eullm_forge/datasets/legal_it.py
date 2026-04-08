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
    """Fallback: fetch articles one-by-one via the normattiva.it AJAX endpoint.

    The normattiva.it viewer page embeds all article URLs as JavaScript onclick
    handlers of the form:
        showArticle('/atto/caricaArticolo?...&art.idArticolo=N&...', this)

    We extract all these URLs from the viewer page, then fetch each article via
    the AJAX endpoint. Returns a minimal XML string for parse_normattiva_xml.
    """
    try:
        import warnings

        from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    except ImportError:
        return None

    from .base import _cache_save

    viewer_url = f"https://www.normattiva.it/uri-res/N2Ls?{source.urn}"
    try:
        resp = session.get(viewer_url, timeout=60)
        resp.raise_for_status()
        viewer_html = resp.text
    except Exception as exc:
        logger.warning("normattiva.it viewer failed for %s: %s", source.id, exc)
        return None

    # Extract all article AJAX URLs from onclick handlers
    ajax_urls: list[str] = []
    seen: set[str] = set()
    for m in re.finditer(r"'/atto/caricaArticolo[^']+'", viewer_html):
        url = "https://www.normattiva.it" + m.group(0).strip("'")
        # Deduplicate by art.idArticolo parameter
        art_id = re.search(r"art\.idArticolo=(\d+)", url)
        if art_id:
            key = art_id.group(1)
            if key not in seen:
                seen.add(key)
                ajax_urls.append(url)

    if not ajax_urls:
        logger.warning("No article AJAX URLs found for %s", source.id)
        return None

    logger.info("  Fetching %d articles via AJAX for %s...", len(ajax_urls), source.id)

    # Fetch each article and build XML
    xml_parts = ["<atto>"]
    for url in ajax_urls:
        try:
            r = session.get(url, timeout=30,
                            headers={"X-Requested-With": "XMLHttpRequest",
                                     "Referer": viewer_url})
            if not r.ok:
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            num_el = soup.find(class_="article-num-akn")
            txt_el = soup.find(class_="art-just-text-akn")
            if not txt_el:
                continue
            num = num_el.get_text(strip=True) if num_el else ""
            text = " ".join(txt_el.get_text(separator=" ").split())
            if len(text) < 10:
                continue
            # Escape XML special chars
            num_safe = num.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            txt_safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            xml_parts.append(
                f"<articolo><num>{num_safe}</num><testo>{txt_safe}</testo></articolo>"
            )
        except Exception:
            continue

    xml_parts.append("</atto>")
    if len(xml_parts) <= 2:  # only <atto> + </atto>
        return None

    result = "\n".join(xml_parts)
    _cache_save(f"normattiva_{source.id}", result)
    logger.info("  Fetched %d articles for %s", len(xml_parts) - 2, source.id)
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
) -> Path:
    """Download and prepare the Italian legal corpus.

    Downloads Italian legislation from normattiva.it and EU regulations
    from EUR-Lex, extracts articles, cleans text, and saves as JSONL.

    Args:
        output_dir: Directory where train.jsonl and val.jsonl will be saved.
        sources: Source IDs to include (None = all). E.g. ["costituzione", "gdpr"].
        push_to_hub: If True, push dataset to HuggingFace Hub.
        hub_repo: HuggingFace Hub repo ID for push (default: eullm/legal-it-corpus).
        val_ratio: Fraction of records to put in validation split.
        no_cache: If True, bypass local HTTP cache and re-download all sources.

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

    # --- normattiva.it sources (shared session across all fetches) ---
    normattiva_laws = [law for law in NORMATTIVA_LAWS if not sources or law.id in sources]
    normattiva_session = make_normattiva_session() if normattiva_laws else None

    for law in normattiva_laws:
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

"""Fetch Cassazione sentences from italgiure.giustizia.it (SentenzeWeb).

Public, free, no-registration access to the full text of every civil and
criminal Cassazione ruling **from 2021 onwards**. Earlier years are NOT in
this collection — SentenzeWeb was launched in 2021 and only covers that
window. Pre-2021 sentences exist only as Massimario summaries (``kind:sic``,
~1.4M docs, short abstracts, not full text).

The JSON response already contains the OCR'd full text in the ``ocr`` field
— no PDF download needed.

Backend: Apache Solr behind an ISAPI (hc.dll) handler on IIS.
Endpoint: /sncass/isapi/hc.dll/sn.solr/sn-collection/select?app.query

Coverage (verified via Solr facets, April 2026):
    snciv (civili): 185,994 docs   snpen (penali): 237,310 docs
    2021: 64,968    2022: 87,936    2023: 88,409
    2024: 83,024    2025: 76,917    2026: 22,050 (in progress)
    Total full-text corpus: ~423K sentences, ~2.8 GB JSONL.

Design notes:
  * Records are streamed to a JSONL file, never accumulated in memory.
  * Progress is checkpointed by (kind, anno, start) so runs can be resumed.
  * A warmup GET to /sncass/ establishes the ASP.NET session cookie.
  * Rate-limited (default 1.5s between requests) to be a polite client.

GDPR warning:
    The corpus contains personal data (names of parties, judges, lawyers,
    addresses, tax IDs). The raw corpus must NOT be published. Use only
    for training; anonymise any downstream artefact.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

from .base import clean_text

logger = logging.getLogger(__name__)

BASE_URL = "https://www.italgiure.giustizia.it"
WARMUP_URL = f"{BASE_URL}/sncass/"
SOLR_URL = f"{BASE_URL}/sncass/isapi/hc.dll/sn.solr/sn-collection/select"
PDF_BASE = f"{BASE_URL}/sncass/isapi/hc.dll/download/"

DEFAULT_KINDS = ("snciv", "snpen")
SOLR_MAX_ROWS = 1000  # verified upper bound (response ~11 MB at this size)

REQUIRED_FIELDS = (
    "id,filename,szdec,kind,ssz,tipoprov,numcard,numdec,numdep,datdep,"
    "ecli,anno,datdec,presidente,relatore,requisitoria,testoocr,ocr"
)


@dataclass
class ItalgiureQuery:
    """A single slice of the corpus to download."""

    kind: str           # "snciv" | "snpen"
    anno: int           # four-digit year, e.g. 2026
    sezione: Optional[int] = None  # 1..7, LAVORO is 9, UNITE is 0; None = all

    def lucene(self) -> str:
        parts = [f'kind:"{self.kind}"', f'anno:"{self.anno}"']
        if self.sezione is not None:
            parts.append(f'szdec:"{self.sezione}"')
        return " AND ".join(parts)

    def slug(self) -> str:
        s = f"{self.kind}_{self.anno}"
        if self.sezione is not None:
            s += f"_sez{self.sezione}"
        return s


def make_italgiure_session(verify: bool | str = True):
    """Create a requests.Session warmed up with an ASP.NET session cookie.

    Args:
        verify: Passed to requests. ``True`` verifies TLS certificates using
            the system/certifi CA bundle (default, recommended). A string is
            treated as a path to a custom CA bundle. ``False`` disables
            verification entirely — use only as a last resort on machines
            with a broken CA store; the session attacker can MITM you.
    """
    import requests

    from .base import DEFAULT_HEADERS

    sess = requests.Session()
    sess.verify = verify
    sess.headers.update({
        **DEFAULT_HEADERS,
        "Accept": "application/json,text/plain,*/*",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": WARMUP_URL,
    })
    if verify is False:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        logger.warning(
            "italgiure: TLS verification DISABLED — fix your CA store "
            "(pip install -U certifi) instead of relying on this."
        )
    # Warmup: the Solr endpoint rejects requests without cookiesession1
    try:
        r = sess.get(WARMUP_URL, timeout=30)
        r.raise_for_status()
    except Exception as exc:
        logger.warning("italgiure warmup failed: %s — retrying anyway", exc)
    return sess


def _solr_get(
    session,
    query: ItalgiureQuery,
    *,
    start: int,
    rows: int,
    retries: int = 4,
    backoff: float = 3.0,
) -> Optional[dict]:
    """Execute a single Solr query, with retry/backoff on transient errors."""
    params = {
        "app.query": "",  # flag param that the ISAPI handler expects
        "q": f"({query.lucene()})",
        "rows": str(rows),
        "start": str(start),
        "wt": "json",
        "indent": "off",
        "sort": "pd desc,numdec desc",
        "fl": REQUIRED_FIELDS,
    }
    delay = backoff
    for attempt in range(1, retries + 1):
        try:
            r = session.get(SOLR_URL, params=params, timeout=120)
            # 503 is expected from IP-rate limiters; back off and retry
            if r.status_code == 503:
                logger.warning(
                    "italgiure 503 (attempt %d/%d) — sleeping %.0fs",
                    attempt, retries, delay,
                )
                time.sleep(delay)
                delay *= 2
                continue
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            if attempt >= retries:
                logger.error("italgiure query giving up: %s", exc)
                return None
            logger.warning(
                "italgiure query failed (%s) — retry %d/%d in %.0fs",
                exc, attempt, retries, delay,
            )
            time.sleep(delay)
            delay *= 2
    return None


def _doc_to_record(doc: dict) -> Optional[dict]:
    """Map a Solr doc to our standard record shape.

    Returns None for docs without usable OCR text (drops ~0% in practice but
    shields the pipeline from empty/corrupt entries).
    """
    ocr_list = doc.get("ocr") or []
    if not ocr_list:
        return None
    text = clean_text(ocr_list[0])
    if len(text) < 200:
        return None

    filename_list = doc.get("filename") or []
    filename = filename_list[0] if filename_list else ""
    url_pdf = PDF_BASE + filename.lstrip("./") if filename else ""

    kind = doc.get("kind", "")
    numdec = doc.get("numdec", "")
    anno = doc.get("anno", "")
    szdec = doc.get("szdec", "")
    tipoprov = doc.get("tipoprov", "")

    return {
        "text": text,
        "source": "italgiure",
        "sentence_id": f"{kind}/{anno}/{numdec}",
        "article_num": numdec,
        "article_title": f"Cassazione {kind} Sez. {szdec}, {tipoprov} n.{numdec}/{anno}",
        "url": url_pdf,
        "metadata": {
            "ecli": doc.get("ecli", ""),
            "kind": kind,
            "sezione": szdec,
            "ssz": doc.get("ssz", ""),
            "anno": anno,
            "tipoprov": tipoprov,
            "datdec": doc.get("datdec", ""),
            "datdep": (doc.get("datdep") or [""])[0],
            "presidente": (doc.get("presidente") or [""])[0],
            "relatore": (doc.get("relatore") or [""])[0],
        },
    }


def _iter_query_docs(
    session,
    query: ItalgiureQuery,
    *,
    start: int = 0,
    rows: int = SOLR_MAX_ROWS,
    rate_limit_sec: float = 1.5,
    max_docs: Optional[int] = None,
) -> Iterator[tuple[int, list[dict], int]]:
    """Yield (next_start, batch_records, num_found) tuples for a single query.

    Performs pagination via Solr ``start`` / ``rows``. Stops when ``numFound``
    is reached or ``max_docs`` records have been produced. A sentinel tuple
    ``(start, [], 0)`` is yielded for legitimately empty slices so the caller
    can distinguish "year really has 0 docs" from "server returned nothing".
    """
    cur = start
    emitted = 0
    while True:
        want = rows
        if max_docs is not None:
            remaining = max_docs - emitted
            if remaining <= 0:
                return
            want = min(rows, remaining)

        resp = _solr_get(session, query, start=cur, rows=want)
        if resp is None:
            logger.error("italgiure: aborting %s at start=%d", query.slug(), cur)
            return

        response = resp.get("response") or {}
        num_found = int(response.get("numFound", 0))
        docs = response.get("docs") or []

        # Legit empty slice: emit a sentinel so the caller can mark it complete.
        if num_found == 0 and cur == start:
            yield cur, [], 0
            return

        # numFound > 0 but docs empty at an in-range offset = server glitch.
        # Back off hard and retry once before giving up (leaves progress intact).
        if not docs and cur < num_found:
            backoff = 0.0 if rate_limit_sec <= 0 else max(rate_limit_sec * 5, 10.0)
            logger.warning(
                "italgiure [%s] empty docs at start=%d but numFound=%d — "
                "backing off %.1fs and retrying",
                query.slug(), cur, num_found, backoff,
            )
            time.sleep(backoff)
            resp = _solr_get(session, query, start=cur, rows=want)
            if resp is None:
                return
            response = resp.get("response") or {}
            num_found = int(response.get("numFound", 0))
            docs = response.get("docs") or []
            if not docs:
                logger.error(
                    "italgiure [%s] still empty after retry at start=%d, "
                    "numFound=%d — aborting slice (progress kept for retry)",
                    query.slug(), cur, num_found,
                )
                return

        records: list[dict] = []
        for doc in docs:
            rec = _doc_to_record(doc)
            if rec is not None:
                records.append(rec)

        next_start = cur + len(docs)
        yield next_start, records, num_found
        emitted += len(records)
        cur = next_start

        if cur >= num_found:
            return
        time.sleep(rate_limit_sec)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_italgiure(
    output_dir: str | Path,
    *,
    years: Iterable[int] = range(2021, 2027),
    kinds: Iterable[str] = DEFAULT_KINDS,
    sezione: Optional[int] = None,
    max_docs_per_query: Optional[int] = None,
    rate_limit_sec: float = 1.5,
    rows: int = SOLR_MAX_ROWS,
    session=None,
    verify: bool | str = True,
) -> Path:
    """Download Cassazione sentences from italgiure.giustizia.it.

    The corpus is streamed to one JSONL file per (kind, anno) slice under
    ``output_dir``; resumable via a ``_progress.json`` checkpoint.

    Args:
        output_dir: Directory for italgiure_<kind>_<anno>.jsonl + _progress.json.
        years: Years to download (inclusive). Only 2011+ has full coverage.
        kinds: "snciv" (civile), "snpen" (penale), or both.
        sezione: Restrict to a single section number (1..7, 0=UNITE, 9=LAVORO).
        max_docs_per_query: Safety cap per (kind, anno) slice — mainly for tests.
        rate_limit_sec: Delay between successive Solr calls.
        rows: Rows per Solr page (max 1000 verified).
        session: Optional pre-built requests.Session; a warmed-up session is
            created if not supplied.
        verify: TLS verification for the auto-created session (ignored if
            ``session`` is supplied). ``True`` (default) uses the system CA
            bundle; a string is a custom CA bundle path; ``False`` disables
            verification entirely.

    Returns:
        Path to the output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "_progress.json"
    progress: dict[str, int] = {}
    if progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Progress file corrupt (%s) — starting fresh", exc)
            progress = {}

    if session is None:
        session = make_italgiure_session(verify=verify)

    logger.warning(
        "italgiure download will fetch personal data (names of parties, judges, "
        "lawyers). Do NOT publish the raw corpus — use only for training."
    )

    for kind in kinds:
        for year in years:
            query = ItalgiureQuery(kind=kind, anno=year, sezione=sezione)
            slug = query.slug()
            out_path = output_dir / f"italgiure_{slug}.jsonl"
            start = progress.get(slug, 0)
            if start == -1:
                # Self-heal: if marked complete but the JSONL is empty, a
                # previous run hit a server glitch that was silently swallowed.
                # Reset to offset 0 and retry.
                if out_path.exists() and out_path.stat().st_size == 0:
                    logger.warning(
                        "italgiure [%s] marked complete but file is empty — "
                        "resetting progress and retrying",
                        slug,
                    )
                    start = 0
                    progress.pop(slug, None)
                else:
                    logger.info("italgiure [%s] already complete — skipping", slug)
                    continue

            logger.info("italgiure [%s] downloading from offset %d", slug, start)
            mode = "a" if start > 0 and out_path.exists() else "w"
            written = 0
            had_batch = False
            last_num_found: Optional[int] = None
            final_offset = start
            with open(out_path, mode, encoding="utf-8") as f:
                iterator = _iter_query_docs(
                    session,
                    query,
                    start=start,
                    rows=rows,
                    rate_limit_sec=rate_limit_sec,
                    max_docs=max_docs_per_query,
                )
                for next_start, batch, num_found in iterator:
                    had_batch = True
                    last_num_found = num_found
                    final_offset = next_start
                    for rec in batch:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        written += 1
                    f.flush()
                    progress[slug] = next_start
                    progress_path.write_text(
                        json.dumps(progress, indent=2), encoding="utf-8"
                    )
                    if batch:
                        logger.info(
                            "  [%s] offset=%d/%d (+%d records, %d in this run)",
                            slug, next_start, num_found, len(batch), written,
                        )

            # Mark slice as complete only when the iterator exited naturally
            # after confirming we reached numFound (or numFound==0). If we
            # never got a batch, leave progress intact so the next run retries.
            if max_docs_per_query is None:
                if not had_batch:
                    logger.error(
                        "italgiure [%s] aborted before any response — "
                        "keeping progress=%d for retry",
                        slug, start,
                    )
                elif last_num_found == 0 or final_offset >= (last_num_found or 0):
                    progress[slug] = -1
                    progress_path.write_text(
                        json.dumps(progress, indent=2), encoding="utf-8"
                    )
                    # Legit empty slice (pre-2021): drop the 0-byte file so
                    # the output directory doesn't fill with noise.
                    if written == 0 and out_path.exists() and out_path.stat().st_size == 0:
                        out_path.unlink()
                        logger.info(
                            "italgiure [%s] numFound=0 — removed empty JSONL",
                            slug,
                        )
                else:
                    logger.warning(
                        "italgiure [%s] stopped at offset=%d/%d — "
                        "keeping progress for retry",
                        slug, final_offset, last_num_found,
                    )

    return output_dir


def load_italgiure_jsonl(output_dir: str | Path) -> list[dict]:
    """Load every italgiure_*.jsonl file under ``output_dir`` into a list.

    Intended for small test slices only — for a full corpus, stream through
    ``datasets.load_dataset('json', data_files=...)`` instead.
    """
    output_dir = Path(output_dir)
    records: list[dict] = []
    for path in sorted(output_dir.glob("italgiure_*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records

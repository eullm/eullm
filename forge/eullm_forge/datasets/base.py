"""Shared utilities for dataset preparation: HTTP fetching, text cleaning, JSONL I/O."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Raw download cache — avoids re-downloading sources on re-runs
CACHE_DIR = Path.home() / ".cache" / "eullm-forge" / "raw"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; EULLM-Forge/0.1; "
        "+https://github.com/eullm/eullm)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "it-IT,it;q=0.9,en;q=0.5",
}


def http_get(
    url: str,
    *,
    headers: dict | None = None,
    retries: int = 3,
    delay: float = 2.0,
    cache_key: str | None = None,
) -> Optional[str]:
    """HTTP GET with retry logic and optional local caching.

    Args:
        url: URL to fetch.
        headers: Additional HTTP headers (merged with defaults).
        retries: Number of retry attempts on failure.
        delay: Seconds between retries (doubles on each retry).
        cache_key: If given, cache the raw response under this key.

    Returns:
        Response text, or None if all retries fail.
    """
    try:
        import requests
    except ImportError:
        raise RuntimeError(
            "requests is required for dataset preparation. "
            "Install with: pip install requests"
        )

    # Check local cache first
    if cache_key:
        cached = _cache_load(cache_key)
        if cached is not None:
            logger.debug("Cache hit: %s", cache_key)
            return cached

    merged_headers = {**DEFAULT_HEADERS, **(headers or {})}
    current_delay = delay

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=merged_headers, timeout=30)
            response.raise_for_status()
            text = response.text

            if cache_key:
                _cache_save(cache_key, text)

            return text

        except Exception as exc:
            if attempt < retries - 1:
                logger.warning(
                    "Request failed (%s), retry %d/%d in %.0fs: %s",
                    exc, attempt + 1, retries, current_delay, url,
                )
                time.sleep(current_delay)
                current_delay *= 2
            else:
                logger.error("All retries exhausted for %s: %s", url, exc)

    return None


def clean_text(text: str) -> str:
    """Normalize whitespace and remove common artifacts.

    - Collapse multiple spaces/tabs to a single space
    - Collapse 3+ consecutive newlines to 2
    - Strip leading/trailing whitespace
    """
    # Collapse horizontal whitespace
    text = re.sub(r"[ \t]+", " ", text)
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse excess blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip
    return text.strip()


def strip_html_tags(html: str) -> str:
    """Remove all HTML tags, returning plain text."""
    # Replace block elements with newlines to preserve structure
    for tag in ("p", "div", "br", "h1", "h2", "h3", "h4", "h5", "li"):
        html = re.sub(rf"</?{tag}[^>]*>", "\n", html, flags=re.IGNORECASE)
    # Remove remaining tags
    text = re.sub(r"<[^>]+>", "", html)
    # Decode common HTML entities
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
        .replace("&agrave;", "à")
        .replace("&egrave;", "è")
        .replace("&igrave;", "ì")
        .replace("&ograve;", "ò")
        .replace("&ugrave;", "ù")
    )
    return clean_text(text)


def save_jsonl(records: list[dict], path: Path) -> None:
    """Save a list of dicts as JSONL (one JSON object per line).

    Args:
        records: List of dicts to serialize.
        path: Output file path (created with parent dirs if needed).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Saved %d records → %s", len(records), path)


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file into a list of dicts."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def train_val_split(
    records: list[dict],
    val_ratio: float = 0.05,
) -> tuple[list[dict], list[dict]]:
    """Split records into train and validation sets.

    Deterministic split: last val_ratio fraction goes to validation.
    """
    n_val = max(1, int(len(records) * val_ratio))
    return records[:-n_val], records[-n_val:]


# --- Internal cache helpers ---

def _cache_path(key: str) -> Path:
    safe_key = re.sub(r"[^\w\-]", "_", key)[:120]
    return CACHE_DIR / safe_key


def _cache_load(key: str) -> Optional[str]:
    p = _cache_path(key)
    if p.exists():
        return p.read_text(encoding="utf-8")
    return None


def _cache_save(key: str, content: str) -> None:
    p = _cache_path(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")

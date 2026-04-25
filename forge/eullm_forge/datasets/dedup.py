"""Deduplication for the chunked italgiure corpus.

Two stages, both streaming-friendly:

1. **Exact dedup** — drop chunks whose text is byte-identical (after a
   light normalization: lowercase + collapsed whitespace). Cheap, removes
   the bulk of the redundancy that comes from boilerplate ("RILEVATO CHE",
   "P.Q.M.", standard procedural phrases that appear verbatim).

2. **Near dedup** — MinHash + LSH on word-level shingles. Catches
   passages that say the same thing with minor variations (different
   parties, slightly different formatting). The default 0.85 Jaccard
   threshold is conservative: only very similar chunks get folded
   together, so substantive variants survive.

Both stages preserve the original record's metadata and add a
``dedup_kept`` boolean (True for the canonical kept record, False for
duplicates) when called with ``annotate=True``. By default duplicates are
dropped from the output stream and only kept records are yielded.

Order matters: callers should run exact dedup BEFORE near dedup —
exact catches O(80%) of the redundancy at trivial cost, leaving fewer
chunks for the more expensive MinHash pass.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Optional

# datasketch is a hard runtime dep declared in pyproject.toml; only catch
# the import error so test failures point at the missing package.
try:
    from datasketch import MinHash, MinHashLSH  # type: ignore
except ImportError as _exc:  # pragma: no cover - dep should be installed
    MinHash = None  # type: ignore[assignment]
    MinHashLSH = None  # type: ignore[assignment]
    _DATASKETCH_ERROR = _exc
else:
    _DATASKETCH_ERROR = None


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def _normalise_for_hash(text: str) -> str:
    """Cheap normalisation for exact-dedup hashing: lowercase + collapsed
    whitespace + stripped. Keeps punctuation since legal text uses it
    semantically (e.g. "art. 13" vs "art 13").
    """
    return _WS_RE.sub(" ", text.lower()).strip()


def _shingles(text: str, *, size: int = 5) -> set[str]:
    """Word-level n-gram shingles for MinHash. Italian legal text has
    many short connective tokens (di, della, che, ...), so a 5-gram is
    a good balance: long enough to be distinctive, short enough that
    paraphrases still share several shingles.
    """
    tokens = _WS_RE.split(_normalise_for_hash(text))
    tokens = [t for t in tokens if t]
    if len(tokens) < size:
        # Fall back to the whole token sequence for very short texts.
        return {" ".join(tokens)} if tokens else set()
    return {
        " ".join(tokens[i:i + size])
        for i in range(len(tokens) - size + 1)
    }


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class DedupStats:
    """Per-stage counters for audit and logging."""

    seen: int = 0
    kept: int = 0
    dropped_exact: int = 0
    dropped_near: int = 0
    examples_dropped: list[tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, int]:
        return {
            "seen": self.seen,
            "kept": self.kept,
            "dropped_exact": self.dropped_exact,
            "dropped_near": self.dropped_near,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def exact_dedup(
    records: Iterable[dict[str, Any]],
    *,
    text_field: str = "text",
    stats: Optional[DedupStats] = None,
) -> Iterator[dict[str, Any]]:
    """Stream records dropping byte-identical (post-normalisation) chunks.

    Memory: O(n) — keeps a set of SHA256 digests, ~32 bytes per kept chunk.
    """
    seen: set[str] = set()
    s = stats if stats is not None else DedupStats()
    for rec in records:
        s.seen += 1
        text = rec.get(text_field) or ""
        if not text.strip():
            continue
        digest = hashlib.sha256(
            _normalise_for_hash(text).encode("utf-8")
        ).hexdigest()
        if digest in seen:
            s.dropped_exact += 1
            continue
        seen.add(digest)
        s.kept += 1
        yield rec


def near_dedup(
    records: Iterable[dict[str, Any]],
    *,
    text_field: str = "text",
    threshold: float = 0.85,
    num_perm: int = 128,
    shingle_size: int = 5,
    stats: Optional[DedupStats] = None,
) -> Iterator[dict[str, Any]]:
    """Stream records dropping near-duplicates via MinHash LSH.

    Args:
        records: iterable of dict records with a text field.
        threshold: Jaccard similarity above which two records are treated
            as duplicates. 0.85 is conservative; 0.7 is more aggressive.
        num_perm: number of MinHash permutations. 128 is the standard
            quality/cost tradeoff.
        shingle_size: word n-gram size (5 = good for Italian legal prose).
        stats: optional DedupStats to accumulate into.

    Memory: O(n) MinHashes (~512 bytes each at num_perm=128) plus the
    LSH index, which is comparable in size.
    """
    if MinHash is None or MinHashLSH is None:
        raise RuntimeError(
            f"datasketch is required for near_dedup ({_DATASKETCH_ERROR}). "
            "Install with `pip install datasketch`."
        )
    s = stats if stats is not None else DedupStats()
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    next_key = 0
    for rec in records:
        s.seen += 1
        text = rec.get(text_field) or ""
        shingles = _shingles(text, size=shingle_size)
        if not shingles:
            continue
        m = MinHash(num_perm=num_perm)
        for sh in shingles:
            m.update(sh.encode("utf-8"))
        if lsh.query(m):
            s.dropped_near += 1
            if len(s.examples_dropped) < 5:
                s.examples_dropped.append(
                    (rec.get("source_id", ""), text[:120])
                )
            continue
        key = str(next_key)
        next_key += 1
        lsh.insert(key, m)
        s.kept += 1
        yield rec


def dedup(
    records: Iterable[dict[str, Any]],
    *,
    text_field: str = "text",
    near_threshold: float = 0.85,
    num_perm: int = 128,
    shingle_size: int = 5,
    skip_near: bool = False,
    stats: Optional[DedupStats] = None,
) -> Iterator[dict[str, Any]]:
    """Single-pass exact + near dedup.

    Each input record is checked against (1) the SHA256 set of seen
    chunk hashes, then (2) the MinHash LSH index. The first record of
    a duplicate cluster is emitted; subsequent ones are dropped, and
    the relevant counter (``dropped_exact`` or ``dropped_near``) is
    incremented.

    Set ``skip_near=True`` to bypass the LSH stage for faster runs
    (e.g. while iterating on the chunker or anonymiser).
    """
    if not skip_near and (MinHash is None or MinHashLSH is None):
        raise RuntimeError(
            f"datasketch is required for near dedup ({_DATASKETCH_ERROR}). "
            "Install with `pip install datasketch`, or pass skip_near=True."
        )
    s = stats if stats is not None else DedupStats()
    seen_hashes: set[str] = set()
    lsh = (
        MinHashLSH(threshold=near_threshold, num_perm=num_perm)
        if not skip_near else None
    )
    next_key = 0

    for rec in records:
        s.seen += 1
        text = rec.get(text_field) or ""
        if not text.strip():
            continue

        digest = hashlib.sha256(
            _normalise_for_hash(text).encode("utf-8")
        ).hexdigest()
        if digest in seen_hashes:
            s.dropped_exact += 1
            continue
        seen_hashes.add(digest)

        if lsh is not None:
            shingles = _shingles(text, size=shingle_size)
            if shingles:
                m = MinHash(num_perm=num_perm)
                for sh in shingles:
                    m.update(sh.encode("utf-8"))
                if lsh.query(m):
                    s.dropped_near += 1
                    if len(s.examples_dropped) < 5:
                        s.examples_dropped.append(
                            (rec.get("source_id", ""), text[:120])
                        )
                    continue
                lsh.insert(str(next_key), m)
                next_key += 1

        s.kept += 1
        yield rec

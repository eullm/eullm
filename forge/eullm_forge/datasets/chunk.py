"""Chunking for long legal documents (post-anonymisation).

Cassazione rulings range from ~2k to ~30k chars. To train a 7-14B base
model on commodity GPUs we need to split each ruling into context-fit
chunks. This module produces chunks that:

    * never split a word
    * prefer paragraph boundaries (``\\n\\n`` or ``\\n``) over sentence
      boundaries (``. ``) over arbitrary whitespace
    * carry overlap with the previous chunk for cross-chunk coherence
    * preserve the source record's metadata, plus ``chunk_index`` /
      ``chunk_total`` / ``source_id`` for traceability

The chunker is char-based (no tokenizer dependency) — close enough to
token counts for Italian legal text (~0.75 tokens/char on Qwen tokenizer).
A more accurate token-based chunker can be added once the base model is
locked in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Optional

# Boundary patterns ordered from most to least preferred. Each entry is a
# raw substring; the chunker scans backwards from the soft cut point and
# picks the latest occurrence within the look-back window.
_PREFERRED_BOUNDARIES: tuple[str, ...] = (
    "\n\n",   # paragraph
    "\n",     # line
    ". ",     # end of sentence
    "; ",     # strong clause break
    ", ",     # weak clause break
    " ",      # last-resort word boundary
)


@dataclass(frozen=True)
class ChunkConfig:
    """Knobs for chunking. Defaults are tuned for Italian legal prose
    targeting a 4k-token training context (~3k chars + headroom).
    """

    max_chars: int = 3000
    overlap: int = 200
    min_chars: int = 200
    # When looking for a natural boundary, search backward from the soft
    # cut point by at most this many chars before giving up and cutting
    # mid-stream at the next whitespace.
    boundary_lookback: int = 400


def chunk_text(
    text: str,
    *,
    max_chars: int = 3000,
    overlap: int = 200,
    min_chars: int = 200,
    boundary_lookback: int = 400,
) -> list[str]:
    """Split ``text`` into chunks of at most ``max_chars`` chars.

    The function never splits a word and prefers natural boundaries
    (paragraph > line > sentence > clause > word). Each chunk after the
    first carries ``overlap`` chars from the tail of the previous chunk.
    A trailing fragment shorter than ``min_chars`` is appended to the
    last chunk instead of being kept as its own short chunk.
    """
    if not text:
        return []
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be in [0, max_chars)")

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        soft_end = min(start + max_chars, n)
        if soft_end == n:
            chunks.append(text[start:].strip())
            break
        cut = _find_boundary(text, start, soft_end, boundary_lookback)
        chunks.append(text[start:cut].strip())
        # Compute next start with overlap, clamped to never go backward
        # past the previous start (defensive against pathological inputs).
        next_start = max(cut - overlap, start + 1)
        # Snap to a word boundary to avoid mid-word overlap.
        next_start = _snap_to_word_start(text, next_start)
        start = next_start

    # Clean up: drop empties, merge tiny tail.
    chunks = [c for c in chunks if c]
    if len(chunks) >= 2 and len(chunks[-1]) < min_chars:
        tail = chunks.pop()
        chunks[-1] = (chunks[-1] + "\n\n" + tail).strip()
    return chunks


def _find_boundary(
    text: str, start: int, soft_end: int, lookback: int,
) -> int:
    """Return the cut index closest to ``soft_end`` that lies on a
    natural boundary, looking back at most ``lookback`` chars. If no
    preferred boundary exists in that window, fall back to the nearest
    whitespace before ``soft_end``; if none, return ``soft_end``
    (mid-word cut as last resort — should be unreachable in practice).
    """
    floor = max(start + 1, soft_end - lookback)
    for sep in _PREFERRED_BOUNDARIES:
        idx = text.rfind(sep, floor, soft_end)
        if idx != -1:
            return idx + len(sep)
    # Catch-all: split at any whitespace before soft_end.
    for i in range(soft_end - 1, floor - 1, -1):
        if text[i].isspace():
            return i + 1
    return soft_end


def _snap_to_word_start(text: str, idx: int, *, max_skip: int = 100) -> int:
    """Move ``idx`` forward to the start of the next word, so the
    overlap doesn't begin in the middle of a token.

    To stay robust on pathological inputs with no whitespace (e.g.
    long base64 blobs), give up after ``max_skip`` chars and return the
    original index — accepting a mid-word overlap is better than
    skipping over the rest of the document.
    """
    n = len(text)
    if idx >= n:
        return n
    end = min(n, idx + max_skip)
    j = idx
    while j < end and not text[j].isspace():
        j += 1
    if j >= end:
        return idx
    while j < n and text[j].isspace():
        j += 1
    return j


def chunk_record(
    rec: dict[str, Any],
    *,
    config: Optional[ChunkConfig] = None,
    text_field: str = "text",
    source_id_field: str = "sentence_id",
) -> Iterator[dict[str, Any]]:
    """Yield one record per chunk, preserving the original record's
    fields and adding ``chunk_index`` / ``chunk_total`` / ``source_id``.

    If the record has no text or text is empty, nothing is yielded.
    """
    cfg = config or ChunkConfig()
    text = rec.get(text_field) or ""
    if not text.strip():
        return
    chunks = chunk_text(
        text,
        max_chars=cfg.max_chars,
        overlap=cfg.overlap,
        min_chars=cfg.min_chars,
        boundary_lookback=cfg.boundary_lookback,
    )
    total = len(chunks)
    source_id = rec.get(source_id_field)
    for i, chunk in enumerate(chunks):
        out = dict(rec)
        out[text_field] = chunk
        out["chunk_index"] = i
        out["chunk_total"] = total
        if source_id is not None:
            out["source_id"] = source_id
        yield out

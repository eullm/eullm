"""Final-stage formatter: dedup'd chunks → continued-pretraining JSONL.

Reads ``italgiure_*.dedup.jsonl`` from the corpus directory, mixes the
records across files, splits into train/val, and writes two files in
HuggingFace-compatible format:

    {"text": "...", "source_id": "snciv/2023/12345", "year": 2023,
     "kind": "snciv", "chunk_index": 2, "chunk_total": 5}

The output drops heavy fields (the per-record audit trail produced by
the anonymiser) and keeps only what the trainer / dataloader needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Optional

# Fields kept on every output record. ``text`` is the only one the
# trainer looks at; the others are useful for sampling, weighting,
# debugging and audit.
DEFAULT_KEEP_FIELDS: tuple[str, ...] = (
    "text",
    "source_id",
    "year",
    "kind",
    "chunk_index",
    "chunk_total",
    "sentence_id",
)


@dataclass
class FormatStats:
    seen: int = 0
    written_train: int = 0
    written_val: int = 0
    skipped_empty: int = 0


def slim_record(
    rec: dict[str, Any],
    *,
    keep_fields: Iterable[str] = DEFAULT_KEEP_FIELDS,
    text_field: str = "text",
) -> Optional[dict[str, Any]]:
    """Strip a record down to ``keep_fields``. Returns None if the text
    field is empty (caller should skip).
    """
    text = rec.get(text_field) or ""
    if not text.strip():
        return None
    out = {k: rec[k] for k in keep_fields if k in rec}
    out[text_field] = text
    return out


def split_indices(
    n: int,
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Deterministic train/val index split with shuffling.

    Uses Python's stdlib ``random`` (Mersenne Twister) seeded with
    ``seed`` so the same corpus + same ratio + same seed always produce
    the same split — important for reproducibility of training runs.
    """
    import random
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    n_val = max(1, int(n * val_ratio))
    val_idx = sorted(indices[:n_val])
    train_idx = sorted(indices[n_val:])
    return train_idx, val_idx


def iter_slimmed(
    records: Iterable[dict[str, Any]],
    *,
    keep_fields: Iterable[str] = DEFAULT_KEEP_FIELDS,
    stats: Optional[FormatStats] = None,
) -> Iterator[dict[str, Any]]:
    s = stats if stats is not None else FormatStats()
    for rec in records:
        s.seen += 1
        slim = slim_record(rec, keep_fields=keep_fields)
        if slim is None:
            s.skipped_empty += 1
            continue
        yield slim

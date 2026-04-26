#!/usr/bin/env python3
"""Deduplicate a chunked italgiure corpus.

Reads ``italgiure_*.chunks.jsonl`` from a directory and writes the
parallel ``italgiure_*.dedup.jsonl`` next to each input. Runs exact dedup
(SHA256 over normalised text) followed by near dedup (MinHash LSH, 5-word
shingles, default threshold 0.85).

Cassazione boilerplate (formule procedurali, dispositivi standard) is the
main source of redundancy and gets caught by the exact stage. The near
stage handles substantive paraphrases that share the same legal reasoning
with cosmetic variations.

Usage:
    # Default (exact + near, threshold 0.85):
    python forge/scripts/dedup_corpus.py ~/italgiure_corpus

    # Only exact dedup (much faster, useful for QA on the chunker):
    python forge/scripts/dedup_corpus.py ~/italgiure_corpus --skip-near

    # More aggressive near-dup (catches looser paraphrases):
    python forge/scripts/dedup_corpus.py ~/italgiure_corpus \\
        --near-threshold 0.7
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.datasets.dedup import DedupStats, dedup  # noqa: E402


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                print(
                    f"[WARN] {path.name}:{lineno} malformed JSON: {exc}",
                    file=sys.stderr,
                )


def _dedup_one(
    src: Path,
    dst: Path,
    *,
    near_threshold: float,
    num_perm: int,
    shingle_size: int,
    skip_near: bool,
    dry_run: bool,
) -> DedupStats:
    stats = DedupStats()
    out_f = None if dry_run else dst.open("w", encoding="utf-8")
    try:
        for rec in dedup(
            _iter_jsonl(src),
            near_threshold=near_threshold,
            num_perm=num_perm,
            shingle_size=shingle_size,
            skip_near=skip_near,
            stats=stats,
        ):
            if out_f is not None:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    finally:
        if out_f is not None:
            out_f.close()
    return stats


def _format_stats(stats: DedupStats) -> str:
    seen = stats.seen
    if seen == 0:
        return "no records"
    pct_kept = 100.0 * stats.kept / seen
    pct_exact = 100.0 * stats.dropped_exact / seen
    pct_near = 100.0 * stats.dropped_near / seen
    return (
        f"seen={seen:,}, kept={stats.kept:,} ({pct_kept:.1f}%), "
        f"exact-dup={stats.dropped_exact:,} ({pct_exact:.1f}%), "
        f"near-dup={stats.dropped_near:,} ({pct_near:.1f}%)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "corpus_dir",
        type=Path,
        help="Directory containing italgiure_*.chunks.jsonl files",
    )
    parser.add_argument(
        "--near-threshold",
        type=float,
        default=0.85,
        help="Jaccard similarity threshold for near-dup detection "
        "(default: 0.85; 0.7 is more aggressive).",
    )
    parser.add_argument("--num-perm", type=int, default=128)
    parser.add_argument("--shingle-size", type=int, default=5)
    parser.add_argument(
        "--skip-near",
        action="store_true",
        help="Run only the exact-dup stage (much faster).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute stats without writing the *.dedup.jsonl files",
    )
    args = parser.parse_args(argv)

    if not args.corpus_dir.is_dir():
        parser.error(f"{args.corpus_dir} is not a directory")

    sources = sorted(args.corpus_dir.glob("italgiure_*.chunks.jsonl"))
    if not sources:
        parser.error(
            f"No italgiure_*.chunks.jsonl files under {args.corpus_dir} — "
            "run chunk_corpus.py first."
        )

    total = DedupStats()
    t0 = time.time()

    # Per-file dedup. Cross-file boilerplate (the same procedural
    # formula appearing in every year's slice) survives this stage by
    # design — each slice gets its own canonical copy. A subsequent
    # cross-file pass can be added if downstream analysis shows the
    # boilerplate is overwhelming the dataset.
    for src in sources:
        slug_no_chunks = src.stem.replace(".chunks", "")
        dst = src.with_name(f"{slug_no_chunks}.dedup.jsonl")
        print(f"[work] {src.name} → {dst.name}", file=sys.stderr)
        t = time.time()
        stats = _dedup_one(
            src, dst,
            near_threshold=args.near_threshold,
            num_perm=args.num_perm,
            shingle_size=args.shingle_size,
            skip_near=args.skip_near,
            dry_run=args.dry_run,
        )
        dt = time.time() - t
        print(
            f"  {_format_stats(stats)} in {dt:.1f}s",
            file=sys.stderr,
        )
        total.seen += stats.seen
        total.kept += stats.kept
        total.dropped_exact += stats.dropped_exact
        total.dropped_near += stats.dropped_near

    dt_all = time.time() - t0
    print()
    print("=" * 72)
    print(f"TOTAL: {_format_stats(total)}")
    print(f"Wall time: {dt_all:.1f}s")
    print("=" * 72)
    if args.dry_run:
        print("(dry-run — no .dedup.jsonl files written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

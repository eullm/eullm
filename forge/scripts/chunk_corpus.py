#!/usr/bin/env python3
"""Chunk an anonymised italgiure corpus into training-ready records.

Reads ``italgiure_*.anon.jsonl`` files from a directory and writes the
parallel ``italgiure_*.chunks.jsonl`` next to each input. Each output
record corresponds to one chunk of one source ruling, with
``chunk_index`` / ``chunk_total`` / ``source_id`` fields added so the
training stage can group / weight chunks belonging to the same ruling.

Usage:
    # Default chunking (3000 chars / 200 overlap / 200 min):
    python forge/scripts/chunk_corpus.py ~/italgiure_corpus

    # Tighter chunks for short-context base models:
    python forge/scripts/chunk_corpus.py ~/italgiure_corpus \\
        --max-chars 2000 --overlap 150

    # Inspect what would happen without writing:
    python forge/scripts/chunk_corpus.py ~/italgiure_corpus --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.datasets.chunk import ChunkConfig, chunk_record  # noqa: E402


def _iter_jsonl(path: Path):
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


def _chunk_one(
    src: Path, dst: Path, *, config: ChunkConfig, dry_run: bool,
) -> tuple[int, int, list[int]]:
    """Return (records_in, chunks_out, chunk_size_distribution)."""
    in_count = 0
    out_count = 0
    sizes: list[int] = []
    out_f = None if dry_run else dst.open("w", encoding="utf-8")
    try:
        for rec in _iter_jsonl(src):
            in_count += 1
            for chunked in chunk_record(rec, config=config):
                out_count += 1
                sizes.append(len(chunked.get("text", "")))
                if out_f is not None:
                    out_f.write(json.dumps(chunked, ensure_ascii=False) + "\n")
    finally:
        if out_f is not None:
            out_f.close()
    return in_count, out_count, sizes


def _summary(sizes: list[int]) -> str:
    if not sizes:
        return "no chunks"
    n = len(sizes)
    avg = sum(sizes) // n
    sizes_sorted = sorted(sizes)
    p50 = sizes_sorted[n // 2]
    p95 = sizes_sorted[max(0, int(n * 0.95) - 1)]
    return (
        f"n={n:,}, "
        f"min={sizes_sorted[0]:,}, "
        f"p50={p50:,}, "
        f"p95={p95:,}, "
        f"max={sizes_sorted[-1]:,}, "
        f"avg={avg:,}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "corpus_dir",
        type=Path,
        help="Directory containing italgiure_*.anon.jsonl files",
    )
    parser.add_argument("--max-chars", type=int, default=3000)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--boundary-lookback", type=int, default=400)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute statistics without writing the *.chunks.jsonl files",
    )
    args = parser.parse_args(argv)

    if not args.corpus_dir.is_dir():
        parser.error(f"{args.corpus_dir} is not a directory")

    sources = sorted(args.corpus_dir.glob("italgiure_*.anon.jsonl"))
    if not sources:
        parser.error(
            f"No italgiure_*.anon.jsonl files under {args.corpus_dir} — "
            "run anonymize_italgiure.py first."
        )

    config = ChunkConfig(
        max_chars=args.max_chars,
        overlap=args.overlap,
        min_chars=args.min_chars,
        boundary_lookback=args.boundary_lookback,
    )

    total_in = total_out = 0
    all_sizes: list[int] = []
    t0 = time.time()
    for src in sources:
        slug = src.stem.replace(".anon", "").replace("italgiure_", "")
        dst = src.with_name(f"italgiure_{slug}.chunks.jsonl")
        print(f"[work] {src.name} → {dst.name}", file=sys.stderr)
        t = time.time()
        n_in, n_out, sizes = _chunk_one(
            src, dst, config=config, dry_run=args.dry_run,
        )
        dt = time.time() - t
        total_in += n_in
        total_out += n_out
        all_sizes.extend(sizes)
        print(
            f"  records {n_in:,} → chunks {n_out:,} in {dt:.1f}s "
            f"({_summary(sizes)})",
            file=sys.stderr,
        )

    dt_all = time.time() - t0
    print()
    print("=" * 72)
    print(f"Total records:  {total_in:,}")
    print(f"Total chunks:   {total_out:,} "
          f"(x{total_out / max(total_in, 1):.2f} expansion)")
    print(f"Chunk sizes:    {_summary(all_sizes)}")
    print(f"Wall time:      {dt_all:.1f}s")
    print("=" * 72)
    if args.dry_run:
        print("(dry-run — no .chunks.jsonl files written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

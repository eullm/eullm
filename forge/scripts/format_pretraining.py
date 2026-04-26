#!/usr/bin/env python3
"""Final-stage formatter: dedup'd chunks → continued-pretraining JSONL.

Reads all ``italgiure_*.dedup.jsonl`` files in a corpus directory and
emits two files under ``--output``:

    train.jsonl   (~99% of records, shuffled across years and kinds)
    val.jsonl     (~1% of records, held-out validation set)

Each output record has only the fields the trainer / dataloader needs:

    {"text": "...", "source_id": "snciv/2023/12345", "year": 2023,
     "kind": "snciv", "chunk_index": 2, "chunk_total": 5}

The split is deterministic (same seed → same split) so training runs
are reproducible.

Usage:
    # Default (val 1%, seed 42, output to ~/italgiure_corpus/pretraining):
    python forge/scripts/format_pretraining.py ~/italgiure_corpus

    # Custom output directory and a larger val set:
    python forge/scripts/format_pretraining.py ~/italgiure_corpus \\
        --output ~/datasets/legal_it --val-ratio 0.02

    # Inspect what would happen without writing files:
    python forge/scripts/format_pretraining.py ~/italgiure_corpus --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.datasets.training_format import (  # noqa: E402
    DEFAULT_KEEP_FIELDS,
    FormatStats,
    slim_record,
    split_indices,
)


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


def _infer_year_kind(path: Path) -> tuple[int | None, str | None]:
    """italgiure_snciv_2023.dedup.jsonl -> (2023, "snciv")."""
    parts = path.name.split("_")
    if len(parts) >= 3:
        kind = parts[1]
        try:
            year = int(parts[2].split(".", 1)[0])
        except ValueError:
            year = None
        return year, kind
    return None, None


def _format_stats_human(stats: FormatStats) -> str:
    return (
        f"seen={stats.seen:,}, "
        f"train={stats.written_train:,}, "
        f"val={stats.written_val:,}, "
        f"skipped_empty={stats.skipped_empty:,}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "corpus_dir",
        type=Path,
        help="Directory containing italgiure_*.dedup.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for train.jsonl and val.jsonl "
        "(default: <corpus_dir>/pretraining)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.01,
        help="Fraction of records held out for validation (default: 0.01)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for the train/val split (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute stats without writing the train/val files",
    )
    args = parser.parse_args(argv)

    if not args.corpus_dir.is_dir():
        parser.error(f"{args.corpus_dir} is not a directory")

    sources = sorted(args.corpus_dir.glob("italgiure_*.dedup.jsonl"))
    if not sources:
        parser.error(
            f"No italgiure_*.dedup.jsonl files under {args.corpus_dir} — "
            "run dedup_corpus.py first."
        )

    out_dir = args.output or (args.corpus_dir / "pretraining")
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"

    stats = FormatStats()
    t0 = time.time()

    # Pass 1: stream all records into a memory list. This is the heaviest
    # step: 1M+ chunks, ~3GB total. We hold them in memory for shuffling
    # and indexing — adequate on a 16GB+ workstation; if memory becomes a
    # constraint we can switch to a 2-pass on-disk shuffle.
    print(f"Loading records from {len(sources)} file(s)...", file=sys.stderr)
    all_records: list[dict] = []
    for src in sources:
        year, kind = _infer_year_kind(src)
        n_before = len(all_records)
        for rec in _iter_jsonl(src):
            stats.seen += 1
            slim = slim_record(rec)
            if slim is None:
                stats.skipped_empty += 1
                continue
            # Backfill year/kind from filename if the record didn't carry them.
            if "year" not in slim and year is not None:
                slim["year"] = year
            if "kind" not in slim and kind is not None:
                slim["kind"] = kind
            all_records.append(slim)
        print(
            f"  {src.name}: +{len(all_records) - n_before:,} records",
            file=sys.stderr,
        )

    n = len(all_records)
    if n == 0:
        print("No records to write; nothing to do.", file=sys.stderr)
        return 1

    train_idx, val_idx = split_indices(n, args.val_ratio, args.seed)
    stats.written_train = len(train_idx)
    stats.written_val = len(val_idx)

    print(
        f"\nSplit: {len(train_idx):,} train + {len(val_idx):,} val "
        f"(val ratio {len(val_idx) / n:.4%})",
        file=sys.stderr,
    )

    if not args.dry_run:
        # Shuffle train order (val stays in source order — easier for
        # debugging). Use the same seed so repeated runs give the same
        # train order, which helps reproduce intermediate checkpoints.
        import random
        rng = random.Random(args.seed)
        train_indices_shuffled = list(train_idx)
        rng.shuffle(train_indices_shuffled)

        print(f"Writing {train_path}...", file=sys.stderr)
        with train_path.open("w", encoding="utf-8") as f:
            for i in train_indices_shuffled:
                f.write(json.dumps(all_records[i], ensure_ascii=False) + "\n")
        print(f"Writing {val_path}...", file=sys.stderr)
        with val_path.open("w", encoding="utf-8") as f:
            for i in val_idx:
                f.write(json.dumps(all_records[i], ensure_ascii=False) + "\n")

    dt = time.time() - t0
    print()
    print("=" * 72)
    print(f"DONE in {dt:.1f}s")
    print(_format_stats_human(stats))
    if not args.dry_run:
        print(f"Train: {train_path}")
        print(f"Val:   {val_path}")
        print()
        print("Next step — load with HuggingFace datasets:")
        print('    from datasets import load_dataset')
        print(f'    ds = load_dataset("json", data_files={{')
        print(f'        "train": "{train_path}",')
        print(f'        "validation": "{val_path}",')
        print(f'    }})')
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())

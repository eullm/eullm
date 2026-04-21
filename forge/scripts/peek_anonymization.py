#!/usr/bin/env python3
"""Show BEFORE/AFTER of N random anonymised records for visual inspection.

Reads an italgiure_*.jsonl slice (the raw one), runs the anonymiser on
each record, and prints a side-by-side comparison. Useful for spotting
false positives / false negatives without scrolling through megabytes
of JSONL.

Usage:
    python forge/scripts/peek_anonymization.py \\
        ~/italgiure_corpus/italgiure_snciv_2023.jsonl --n 5

    # Only show chunks of text where *something* was redacted:
    python forge/scripts/peek_anonymization.py \\
        ~/italgiure_corpus/italgiure_snciv_2023.jsonl --n 10 --around 80

    # With NER enabled:
    python forge/scripts/peek_anonymization.py \\
        ~/italgiure_corpus/italgiure_snciv_2023.jsonl --n 5 --ner
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.datasets.anonymize import (  # noqa: E402
    AnonymiserConfig,
    anonymize_record,
    load_spacy_ner,
)

PLACEHOLDER_RE = re.compile(r"\[[A-Z_]+(?:_\d+)?\]")


def _sample_lines(path: Path, n: int, seed: int) -> list[dict]:
    """Reservoir-sample N records from a JSONL file."""
    rng = random.Random(seed)
    picks: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for i, raw in enumerate(f):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if len(picks) < n:
                picks.append(rec)
            else:
                j = rng.randint(0, i)
                if j < n:
                    picks[j] = rec
    return picks


def _context_windows(before: str, after: str, around: int) -> list[tuple[str, str]]:
    """Return (before_chunk, after_chunk) pairs centred on each redaction
    in the anonymised text, with ``around`` chars of context on each side.
    """
    windows: list[tuple[str, str]] = []
    for m in PLACEHOLDER_RE.finditer(after):
        start_a = max(0, m.start() - around)
        end_a = min(len(after), m.end() + around)
        chunk_after = after[start_a:end_a]
        # Best-effort alignment of before: use the same character offset range.
        # (Works well when redactions don't shift text length too much.)
        start_b = max(0, m.start() - around)
        end_b = min(len(before), m.end() + around)
        chunk_before = before[start_b:end_b]
        windows.append((chunk_before, chunk_after))
    return windows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, help="Path to a raw italgiure_*.jsonl")
    parser.add_argument("--n", type=int, default=5, help="Number of records to show")
    parser.add_argument(
        "--around",
        type=int,
        default=0,
        help="If > 0, show only windows of this many chars around each "
        "redaction instead of the full text",
    )
    parser.add_argument("--ner", action="store_true", help="Enable spaCy NER layer")
    parser.add_argument("--ner-model", default="it_core_news_lg")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-allcaps",
        action="store_true",
        help="Disable the all-caps person heuristic (useful when comparing "
        "NER-only vs regex+allcaps)",
    )
    args = parser.parse_args(argv)

    if not args.jsonl.is_file():
        parser.error(f"{args.jsonl} not found")

    config = AnonymiserConfig(
        use_ner=args.ner,
        redact_allcaps_names=not args.no_allcaps,
    )
    ner = None
    if args.ner:
        print(f"Loading spaCy NER model {args.ner_model}...", file=sys.stderr)
        try:
            ner = load_spacy_ner(args.ner_model)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    records = _sample_lines(args.jsonl, args.n, args.seed)
    if not records:
        print(f"No records in {args.jsonl}", file=sys.stderr)
        return 1

    for i, rec in enumerate(records, start=1):
        before = rec.get("text", "")
        out_rec = anonymize_record(rec, config=config, ner=ner)
        after = out_rec.get("text", "")
        stats = out_rec.get("metadata", {}).get("anonymization", {})

        print("=" * 80)
        print(f"[{i}/{len(records)}] {rec.get('sentence_id')} "
              f"({rec.get('article_title', '')})")
        print(f"    length: {len(before):,} → {len(after):,} chars")
        print(f"    redactions: {', '.join(f'{k}={v}' for k, v in stats.items() if v)}")
        print("-" * 80)

        if args.around > 0:
            windows = _context_windows(before, after, args.around)
            if not windows:
                print("(no redactions in this record)")
            for j, (b, a) in enumerate(windows, start=1):
                print(f"\n  --- redaction {j}/{len(windows)} ---")
                print(f"  BEFORE: ...{b}...")
                print(f"  AFTER : ...{a}...")
        else:
            print("BEFORE:")
            print(before[:2000] + ("..." if len(before) > 2000 else ""))
            print()
            print("AFTER:")
            print(after[:2000] + ("..." if len(after) > 2000 else ""))
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

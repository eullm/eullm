#!/usr/bin/env python3
"""The program-written part of the memory corpus (`eullm_forge.datasets.law_corpus`).

    python forge/scripts/make_law_corpus.py --norms $WORK/norms/legislazione_*.chunks.jsonl \\
        --exclude $WORK/eval/closed-book.jsonl --out $WORK/eullm_runs/memory/program.jsonl

Statements only, one JSON row per text (``text``, ``code``, ``articolo``,
``forma``): the article itself, the tables of contents, heading and number,
deadlines, penalties, procedibility, repeals, and the numbers a code does
not have, except those an evaluation set (``--exclude``) asks about. The
teacher-written part is law_rewrites.py. CPU only, a minute; prints counts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.datasets.law_corpus import excluded, program_rows  # noqa: E402
from eullm_forge.eval.norm_exam import articles_from_records  # noqa: E402


def load_articles(paths: list[Path]):
    records = []
    for p in paths:
        with p.open(encoding="utf-8") as f:
            records += [json.loads(line) for line in f if line.strip()]
    return articles_from_records(records)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--norms", nargs="+", type=Path, required=True)
    ap.add_argument("--exclude", nargs="*", type=Path, default=[],
                    help="evaluation sets whose made-up numbers are never stated as absent")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reverse", type=int, default=0,
                    help="heading-first statements an article, against the reversal curse")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    fakes, _ = excluded([p for p in args.exclude if p.is_file()])
    rows = program_rows(load_articles(args.norms), fakes=fakes, seed=args.seed,
                        reverse=args.reverse)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # written aside and renamed: two chains starting together each write the
    # same rows, and neither can read the other's half-written file
    tmp = args.out.with_name(f".{args.out.name}.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, args.out)
    forms = Counter(r["forma"] for r in rows)
    codes = Counter(r["code"] for r in rows)
    chars = sum(len(r["text"]) for r in rows)
    print(f"[law-corpus] {len(rows):,} rows, {chars / 1e6:.1f} M chars -> {args.out}")
    print("[law-corpus] forms: " + ", ".join(f"{k} {v:,}" for k, v in sorted(forms.items())))
    print("[law-corpus] codes: " + ", ".join(f"{k} {v:,}" for k, v in sorted(codes.items())))
    print(f"[law-corpus] {len(fakes)} made-up numbers held out")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())

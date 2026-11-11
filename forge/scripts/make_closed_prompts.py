#!/usr/bin/env python3
"""GRPO prompts with no texts (`eullm_forge.datasets.closed_prompts`), graded by program.

    python forge/scripts/make_closed_prompts.py --norms $D/norms-src/legislazione_*.chunks.jsonl \\
        --exclude $D/eval/norm-exam.jsonl $D/eval/leonardo/*.jsonl ... --out $D/grpo/closed.jsonl

CPU, a minute, on a login node. No article of an ``--exclude`` set is drawn,
nor a made-up number the closed-book suite asks about. Counts only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.datasets.closed_prompts import closed_prompts  # noqa: E402
from eullm_forge.datasets.law_corpus import excluded  # noqa: E402
from eullm_forge.datasets.premise_gen import excluded_articles  # noqa: E402
from eullm_forge.eval.dataset import load_eval_set  # noqa: E402
from eullm_forge.eval.norm_exam import articles_from_records  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--norms", nargs="+", type=Path, required=True)
    ap.add_argument("--exclude", nargs="*", type=Path, default=[])
    ap.add_argument("--per-code", type=int, default=60, help="prompts of each article kind a code")
    ap.add_argument("--rulings", type=int, default=600, help="made-up rulings asked about")
    ap.add_argument("--real-factor", type=int, default=1,
                    help="real articles asked about for each made-up one")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    records = []
    for p in args.norms:
        with p.open(encoding="utf-8") as f:
            records += [json.loads(line) for line in f if line.strip()]
    fakes, asked = excluded([p for p in args.exclude if p.is_file()])
    for p in args.exclude:
        try:
            items = load_eval_set(p)
        except Exception:      # noqa: BLE001 - a set in another shape: its metadata was read above
            continue
        asked |= {(c, n.lower()) for c, n in excluded_articles(items)}
    rows = closed_prompts(articles_from_records(records), per_code=args.per_code,
                          rulings=args.rulings, rng=random.Random(args.seed),
                          excluded=asked, fakes=fakes, real_factor=args.real_factor)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    kinds = Counter(r["tipo"] for r in rows)
    said = ", ".join(f"{k} {v}" for k, v in sorted(kinds.items()))
    print(f"[closed-prompts] {len(rows):,} prompts ({said}), "
          f"{len(asked):,} evaluation articles left out -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""General abilities kept or lost: a model's GGUF on fixed general sets.

    python forge/scripts/regression_check.py --gguf <model.gguf> --label g4-r3 \\
        --sets $CAMPAIGN_DIR/sets/gsm8k.jsonl $CAMPAIGN_DIR/sets/arc-challenge.jsonl \\
        --csv regression.csv

The legal-it roadmap (docs/research/legal-it-roadmap.md, phase 0.4) runs a
small fixed set on every checkpoint from now on, so that a training that
teaches the law and costs the model its maths or its reading is seen. The
sets are ReflexBench's, frozen as JSONL by `campaign.py prefetch` (compute
nodes have no network): GSM8K graded by the final number, ARC by the
letter, with ReflexBench's own graders (`bench/reflexbench/ab_grade.py`).
The first ``--limit`` items of each set, drawn with a fixed seed, so every
model answers the same ones.

The model is served as it ships: its GGUF through EULLM Engine
(`eval.gguf.EullmServer`), each item a `/api/chat` request with thinking
off, greedy. One CSV row a set: label, set, items, right, share.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "forge"))
sys.path.insert(0, str(ROOT / "bench" / "reflexbench"))

import ab_data  # noqa: E402
import ab_grade  # noqa: E402
import ab_methods  # noqa: E402

from eullm_forge.eval.gguf import EullmServer  # noqa: E402


def ask_all(url: str, model: str, items, parallel: int, max_tokens: int,
            timeout: float) -> list[bool]:
    """Whether each item was answered right, in order; a failed request is wrong."""
    def one(item) -> bool:
        try:
            ans = ab_methods.generate(url, item, model, None, timeout, think=False,
                                      max_tokens=max_tokens)
        except Exception:      # noqa: BLE001 - one lost answer costs one item, not the run
            return False
        return bool(ab_grade.correct(item, ans.text))
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        return list(pool.map(one, items))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True, type=Path)
    ap.add_argument("--label", required=True)
    ap.add_argument("--sets", nargs="+", type=Path, required=True,
                    help="frozen ReflexBench sets (campaign.py prefetch)")
    ap.add_argument("--limit", type=int, default=400, help="items a set (0: all)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eullm", default=os.environ.get("EULLM_BIN", "eullm"))
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--csv", type=Path, required=True)
    args = ap.parse_args(argv)

    sets = [ab_data.from_jsonl(p, args.limit, args.seed) for p in args.sets]
    rows = []
    with EullmServer(args.gguf, binary=args.eullm, parallel=args.parallel,
                     ctx_per_slot=4096,
                     log_path=args.csv.with_name(f"regression-{args.label}.server.log")) as srv:
        for ds in sets:
            right = ask_all(srv.url, str(args.gguf.resolve()), ds.items, args.parallel,
                            args.max_tokens, args.timeout)
            share = sum(right) / len(right) if right else 0.0
            rows.append([args.label, ds.name, len(right), sum(right), f"{share:.3f}"])
            print(f"[regression] {args.label} {ds.name}: {sum(right)}/{len(right)} "
                  f"({share:.1%})", flush=True)
    new = not args.csv.exists()
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["label", "set", "items", "right", "share"])
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

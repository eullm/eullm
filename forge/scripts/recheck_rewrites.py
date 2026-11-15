#!/usr/bin/env python3
"""Teacher rewrites already written, checked again by today's `check_rewrite`.

    python forge/scripts/recheck_rewrites.py --norms $WORK/norms/legislazione_*.chunks.jsonl \\
        --rewrites $WORK/eullm_runs/memory/rewrites*.jsonl \\
        --out-dir $WORK/eullm_runs/memory-v2

`law_rewrites.py` checks a rewrite once, when it is written; a check made
stricter later (a new reason, `commi` reading an article differently) never
reaches the rows already kept. This writes, for each input file, a file of
the same name in ``--out-dir`` with the rows the current check still passes,
and prints how many each reason dropped, by style. Rows of an article no
longer in the norms, or no longer rewritable, are dropped too. CPU only;
prints counts, never a text.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.datasets.law_corpus import check_rewrite, rewritable  # noqa: E402
from eullm_forge.eval.norm_exam import articles_from_records  # noqa: E402


def load_articles(paths: list[Path]):
    records = []
    for p in paths:
        with p.open(encoding="utf-8") as f:
            records += [json.loads(line) for line in f if line.strip()]
    return articles_from_records(records)


def recheck(rows, articles) -> tuple[list[dict], Counter]:
    """(rows kept, Counter of (style, reason) dropped)."""
    by_key = {(a.code, a.number.lower()): a for a in articles.values()}
    kept, dropped = [], Counter()
    for r in rows:
        a = by_key.get((r.get("code"), str(r.get("articolo", "")).lower()))
        style = r.get("forma", "")
        if a is None:
            reason = "no_article"
        elif not rewritable(a):
            reason = "not_rewritable"
        else:
            reason = check_rewrite(a, style, r.get("text", ""))
        if reason:
            dropped[(style, reason)] += 1
        else:
            kept.append(r)
    return kept, dropped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--norms", nargs="+", type=Path, required=True)
    ap.add_argument("--rewrites", nargs="+", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args(argv)

    articles = load_articles(a.norms)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for src in a.rewrites:
        if src.name.endswith(".rejects.jsonl"):
            continue
        dst = a.out_dir / src.name
        if dst.resolve() == src.resolve():
            ap.error(f"{dst} is the input: give another --out-dir")
        with src.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        kept, dropped = recheck(rows, articles)
        tmp = dst.with_name(f".{dst.name}.{os.getpid()}")
        with tmp.open("w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, dst)
        print(f"[recheck] {src.name}: {len(kept):,} of {len(rows):,} kept", flush=True)
        for (style, reason), n in sorted(dropped.items(), key=lambda kv: -kv[1]):
            print(f"    {style:12s} {reason:16s} {n:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

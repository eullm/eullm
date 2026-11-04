#!/usr/bin/env python3
"""General Italian text for E1-mini's mix: a sample of the Italian Wikipedia, on a login node.

    python forge/scripts/prefetch_general_it.py --out $FORGE_DATA_DIR/memory/general-it.jsonl

The roadmap's continued pretraining (phase 2) keeps 20% of general Italian in
every batch so that training on the law does not cost the model its language.
Compute nodes have no network: this runs on a login node, downloads one shard
of ``wikimedia/wikipedia`` (``20231101.it``, CC BY-SA 4.0: training data, not
a dependency) into the Hugging Face cache and writes a seeded sample of its
articles as JSONL (``text``, ``title``). An existing output is kept.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = "wikimedia/wikipedia"
SHARD = "20231101.it/train-00003-of-00010.parquet"


def sample_rows(rows: list[dict], n: int, min_chars: int, max_chars: int, seed: int) -> list[dict]:
    """``n`` articles long enough to be prose, each cut at a paragraph near ``max_chars``."""
    rng = random.Random(seed)
    keep = [r for r in rows if len(r.get("text") or "") >= min_chars]
    rng.shuffle(keep)
    out = []
    for r in keep[:n]:
        text = r["text"]
        if len(text) > max_chars:
            cut = text.rfind("\n", 0, max_chars)
            text = text[:cut if cut > max_chars // 2 else max_chars]
        out.append({"title": r.get("title", ""), "text": text.strip()})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", default=SHARD)
    ap.add_argument("--n", type=int, default=40000, help="articles kept")
    ap.add_argument("--min-chars", type=int, default=1500)
    ap.add_argument("--max-chars", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.out.is_file() and args.out.stat().st_size:
        print(f"[general-it] {args.out} exists, kept")
        return 0
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO, args.shard, repo_type="dataset")
    rows = pq.read_table(path, columns=["title", "text"]).to_pylist()
    out = sample_rows(rows, args.n, args.min_chars, args.max_chars, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    chars = sum(len(r["text"]) for r in out)
    print(f"[general-it] {len(out):,} of {len(rows):,} articles, {chars / 1e6:.0f} M chars "
          f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

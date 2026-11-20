#!/usr/bin/env python3
"""The same rewrites through different servers: how far apart the answers are.

    python forge/scripts/compare_rewrite_runs.py --base llama-vanilla \\
        $WORK/eullm_runs/engine-ab/{llama-vanilla,engine4,engine5}

Each directory is one law_rewrites.py run (its rewrites-*.jsonl), all on the
same articles at temperature 0. Against the base run: answers in common,
identical word for word, the median character of the first difference among
the others, the mean length, and how many answers carry a thinking block
(Gemma 4's ``<|channel>``, ``<think>``) that should not be there. And the
speed each run's log printed ("kept in"), when the log is given. Counts only:
no answer is printed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

THINKING = ("<|channel>", "<think>", "<|think|>")


def read_run(d: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for p in sorted(d.glob("rewrites-*.jsonl")):
        if p.name.endswith(".rejects.jsonl"):
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows.setdefault(r["id"], r["text"])
    return rows


def first_difference(a: str, b: str) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def compare(runs: dict[str, dict[str, str]], base: str) -> list[str]:
    ref = runs[base]
    lines = []
    for name, rows in runs.items():
        leaks = sum(any(t in text for t in THINKING) for text in rows.values())
        mean = statistics.mean(len(t) for t in rows.values()) if rows else 0
        line = f"{name:16s} {len(rows):4d} answers, mean {mean:5.0f} chars, thinking in {leaks}"
        if name != base:
            common = sorted(set(rows) & set(ref))
            same = sum(rows[k] == ref[k] for k in common)
            diffs = [first_difference(rows[k], ref[k]) for k in common if rows[k] != ref[k]]
            med = f"{statistics.median(diffs):.0f}" if diffs else "-"
            line += (f"; with {base}: {len(common)} in common, {same} identical, "
                     f"first difference at median char {med}")
        lines.append(line)
    return lines


def speed(log: Path) -> str:
    kept = [x for x in log.read_text(encoding="utf-8", errors="replace").splitlines()
            if "kept in" in x]
    return kept[-1].strip() if kept else "no 'kept in' line"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path, help="one directory per run")
    ap.add_argument("--base", help="the run the others are compared with (default the first)")
    ap.add_argument("--logs", nargs="*", type=Path, default=[],
                    help="job logs, for the speed each printed")
    a = ap.parse_args(argv)
    runs = {d.name: read_run(d) for d in a.runs}
    base = a.base or a.runs[0].name
    if base not in runs:
        ap.error(f"--base {base} is not one of {', '.join(runs)}")
    print("\n".join(compare(runs, base)))
    for log in a.logs:
        print(f"{log.name}: {speed(log)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

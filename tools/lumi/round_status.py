#!/usr/bin/env python3
"""Where a round stands: for each group and engine label, how many of its
points wait, run, are done, failed or are blocked, and the cause of each
failure or block (the first line of its last note).

    $PY tools/lumi/round_status.py --queue $CAMPAIGN_DIR --round r-b12032

`campaign.py status` counts the whole queue, every round and label at once,
and its failures run back to October: the state of one test, old binary
against new, is not in it (03-11-2026).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

STATES = ("todo", "running", "done", "failed", "blocked")


def round_points(queue: str, round_: str) -> list:
    """(state, point) of every point of `round_` in the queue."""
    out = []
    for state in STATES:
        d = os.path.join(queue, state)
        for name in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if not name.endswith(".json") or name.endswith(".owner.json"):
                continue
            try:
                with open(os.path.join(d, name), encoding="utf-8") as f:
                    p = json.load(f)
            except (OSError, ValueError):
                continue
            if p.get("round") == round_:
                out.append((state, p))
    return out


def describe(points: list) -> list:
    if not points:
        return ["no point of this round in the queue"]
    counts = collections.defaultdict(collections.Counter)
    causes = []
    for state, p in points:
        key = (p.get("group", "?"), p.get("engine_label") or "-")
        counts[key][state] += 1
        if state in ("failed", "blocked"):
            note = (((p.get("notes") or [""])[-1]) or "").strip().splitlines()
            causes.append(f"  {state} {key[0]} {key[1]} {p.get('model')}: "
                          f"{(note or ['(no note)'])[0][:100]}")
    lines = [f"{'group':22} {'label':8} " + " ".join(f"{s:>7}" for s in STATES)]
    for (group, label), c in sorted(counts.items()):
        lines.append(f"{group:22} {label:8} " + " ".join(f"{c[s]:>7}" for s in STATES))
    return lines + causes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--queue", default=os.environ.get("CAMPAIGN_DIR"))
    ap.add_argument("--round", required=True)
    args = ap.parse_args(argv)
    if not args.queue:
        ap.error("--queue, or CAMPAIGN_DIR in the environment")
    for line in describe(round_points(args.queue, args.round)):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())

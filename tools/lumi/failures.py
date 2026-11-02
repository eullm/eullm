#!/usr/bin/env python3
"""The failed points of a label, and why: a line per point (when it failed,
its group, model, KV cache and devices, the first line of its last note),
then the end of the last note of the first point of each group and hour that
failed, which ends with the server's log: what made it exit.

    $PY tools/lumi/failures.py --queue $CAMPAIGN_DIR --label v085
    $PY tools/lumi/failures.py --queue $CAMPAIGN_DIR --label v085 --log 40

Read on a login node, it changes nothing; `campaign.py retry` puts points
back once the cause is known. Written because reading this through a block
pasted into the terminal went wrong on 02-11-2026 (the block run was the one
before it).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def failed_points(queue: str, label: str) -> list:
    """(time it failed, point) of every failed point of `label`, oldest first."""
    d = os.path.join(queue, "failed")
    out = []
    for name in os.listdir(d) if os.path.isdir(d) else []:
        if not name.endswith(".json"):
            continue
        path = os.path.join(d, name)
        try:
            with open(path, encoding="utf-8") as f:
                p = json.load(f)
        except (OSError, ValueError):
            continue
        if (p.get("engine_label") or "") == label:
            out.append((os.path.getmtime(path), p))
    return sorted(out, key=lambda tp: tp[0])


def last_note(p: dict) -> list:
    return (((p.get("notes") or [""])[-1]) or "").strip().splitlines()


def when(t: float, fmt: str = "%d-%m %H:%M") -> str:
    return time.strftime(fmt, time.localtime(t))


def describe(points: list, log_lines: int = 20, clusters: int = 4) -> list:
    lines = [f"{len(points)} failed"]
    for t, p in points:
        first = (last_note(p) or [""])[0][:90]
        lines.append(f"{when(t)} {p.get('group', '?'):16} {p.get('model')} {p.get('kv', '')}"
                     f" gcds {p.get('gcds')}: {first}")
    seen = []
    for t, p in points:
        key = (p.get("group"), when(t, "%d-%m %H"))
        if key not in [k for k, _, _ in seen]:
            seen.append((key, t, p))
    for (group, _), t, p in seen[-clusters:]:
        lines.append(f"--- {group} {p.get('model')} {p.get('kv', '')}, {when(t)}: "
                     f"the last {log_lines} lines of its note")
        lines.extend(last_note(p)[-log_lines:])
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--queue", default=os.environ.get("CAMPAIGN_DIR"))
    ap.add_argument("--label", required=True)
    ap.add_argument("--log", type=int, default=20, help="lines of each note shown (20)")
    ap.add_argument("--clusters", type=int, default=4,
                    help="groups and hours whose note is shown, the latest (4)")
    args = ap.parse_args(argv)
    if not args.queue:
        ap.error("--queue, or CAMPAIGN_DIR in the environment")
    for line in describe(failed_points(args.queue, args.label), args.log, args.clusters):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())

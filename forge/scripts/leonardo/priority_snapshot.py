#!/usr/bin/env python3
"""Freeze the scheduler's view of why our jobs are not starting.

`queue_stats.py` answers "where did the calendar go" from `sacct`, which
keeps a few weeks of history. This answers the next question — *why were we
behind* — from `sprio`, `sshare` and `sinfo`, which keep **none at all**.
They report the current instant and nothing else, so a number not written
down when it is read is gone.

That matters because the EuroHPC Final Report has to explain why an
allocation sized for two months of continuous use was not consumed, and
"the cluster was busy" is an assertion. This is the evidence:

    2026-09-22  fairshare 0.751, using a third of our share
                priority 141,963 = QOS 120,000 + fairshare 18,783 + age 3,175
                the partition's top priority that moment: 60,259,060
                one idle node in boost_usr_prod, and it was not responding

An account under-using its share, sitting behind jobs with 1.5x to 424x its
priority, is a condition of the machine rather than a failure of planning —
but only if somebody recorded it on the day.

Three things it captures, and the third is the one people forget:

  * **our own jobs**, with the priority broken into its factors, so a change
    in the mix (age accruing, fairshare decaying) is visible over time
  * **the competition**, as quantiles over every pending job's priority in
    the partition. Not a list of other people's jobs — a distribution, which
    is what a report can quote without naming anyone
  * **the partition's node states**, because "we were behind" and "there was
    nothing free" are different arguments and both get made

Usage:
    python3 priority_snapshot.py --json snapshots/priority_20260922.json
    python3 priority_snapshot.py            # human-readable, to the terminal
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone


def run(cmd: list[str]) -> tuple[str, str]:
    """Return (stdout, error). A missing or failing command is recorded, not
    raised: a snapshot with three of four sections is worth keeping, and the
    fourth being absent is itself a fact about the day."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                           encoding="utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError) as exc:
        return "", f"{type(exc).__name__}: {exc}"
    if p.returncode != 0:
        return p.stdout, f"exit {p.returncode}: {p.stderr.strip()[:200]}"
    return p.stdout, ""


def our_jobs(user: str) -> dict:
    """sprio -l for our jobs, as records rather than a wall of text."""
    out, err = run(["sprio", "-u", user, "-l", "-h"])
    rows = []
    for line in out.splitlines():
        f = line.split()
        # JOBID PARTITION USER ACCOUNT PRIORITY SITE AGE ASSOC FAIRSHARE
        # JOBSIZE PARTITION QOSNAME QOS NICE [TRES]
        if len(f) < 14:
            continue
        rows.append({
            "jobid": f[0], "partition": f[1], "account": f[3],
            "priority": int(f[4]), "age": int(f[6]),
            "fairshare": int(f[8]), "jobsize": int(f[9]),
            "qos_name": f[11], "qos": int(f[12]),
        })
    return {"jobs": rows, "error": err}


def fairshare(user: str) -> dict:
    out, err = run(["sshare", "-U", "-u", user, "-P", "-n"])
    rec = {}
    for line in out.splitlines():
        f = line.split("|")
        if len(f) >= 7:
            rec = {
                "account": f[0].strip(), "user": f[1].strip(),
                "norm_shares": f[3].strip(), "raw_usage": f[4].strip(),
                "effective_usage": f[5].strip(), "fairshare": f[6].strip(),
            }
    return {"share": rec, "error": err}


def competition(partition: str) -> dict:
    """Quantiles over every pending job's priority in the partition.

    A distribution rather than a list: it answers "how far behind were we"
    without recording anybody else's job ids, which is both better evidence
    and better manners.
    """
    out, err = run(["sprio", "-p", partition, "-h", "-o", "%Y"])
    vals = sorted(int(v) for v in out.split() if v.isdigit())
    if not vals:
        return {"n": 0, "error": err or "no pending jobs with a priority"}

    def q(p: float) -> int:
        return vals[min(len(vals) - 1, int(len(vals) * p))]

    return {
        "n": len(vals), "min": vals[0], "max": vals[-1],
        "p25": q(0.25), "median": q(0.50), "p75": q(0.75), "p90": q(0.90),
        "error": err,
    }


def nodes(partition: str) -> dict:
    out, err = run(["sinfo", "-p", partition, "-h", "-o", "%D %T"])
    states: dict[str, int] = {}
    for line in out.splitlines():
        f = line.split()
        if len(f) == 2 and f[0].isdigit():
            states[f[1]] = states.get(f[1], 0) + int(f[0])
    down, derr = run(["sinfo", "-R", "-h", "-o", "%n|%E"])
    drained = [
        {"node": ln.split("|", 1)[0], "reason": ln.split("|", 1)[1]}
        for ln in down.splitlines() if "|" in ln
    ]
    return {
        "by_state": states, "total": sum(states.values()),
        "drained": drained[:50],
        "error": "; ".join(e for e in (err, derr) if e),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--user", default=None, help="default: $USER")
    p.add_argument("--partition", default="boost_usr_prod")
    p.add_argument("--json", default=None, help="write here instead of stdout")
    args = p.parse_args(argv)

    import os
    user = args.user or os.environ.get("USER") or ""
    if not user:
        raise SystemExit("[err] no user: pass --user")

    snap = {
        "taken_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "user": user,
        "partition": args.partition,
        "ours": our_jobs(user),
        "fairshare": fairshare(user),
        "competition": competition(args.partition),
        "nodes": nodes(args.partition),
    }

    if args.json:
        from pathlib import Path
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}", file=sys.stderr)
        return 0

    print(f"# scheduler snapshot {snap['taken_at']}  {user}  {args.partition}\n")
    sh = snap["fairshare"]["share"]
    if sh:
        print(f"fairshare      {sh.get('fairshare')}   "
              f"(norm shares {sh.get('norm_shares')}, "
              f"effective usage {sh.get('effective_usage')})")
    for j in snap["ours"]["jobs"]:
        print(f"  {j['jobid']:>10}  prio {j['priority']:>9,}  "
              f"= qos {j['qos']:,} + fairshare {j['fairshare']:,} "
              f"+ age {j['age']:,}")
    c = snap["competition"]
    if c.get("n"):
        print(f"\npending in {args.partition}: {c['n']}")
        print(f"  median {c['median']:,}   p75 {c['p75']:,}   "
              f"p90 {c['p90']:,}   max {c['max']:,}")
    n = snap["nodes"]
    if n.get("total"):
        busy = ", ".join(f"{k} {v}" for k, v in sorted(n["by_state"].items()))
        print(f"\nnodes ({n['total']}): {busy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

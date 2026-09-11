#!/usr/bin/env python3
"""Where the allocation's calendar went, and whose fault each gap was.

`saldo` and `budget.sh` answer "how many node-hours did we spend". The
EuroHPC Final Report asks something harder: given an allocation sized for two
months of continuous use, **why did you not spend more of it**. That question
has two answers with opposite implications, and reporting them as one number
hides the only part we can act on:

  * **cluster full** — a job was submitted and sat in the queue. Evidence
    about the machine, cited without apology.
  * **queue empty** — nothing was submitted, because nothing was ready. Ours.

Both are measured here, dated, so the report can quote an interval rather
than a total. "7h06m of queue wait" is not citable; "the chain stalled from
2026-09-10 21:07 to 2026-09-11 04:13" is.

Three traps, all of them met on this allocation:

1. **`Start - Submit` is not queue wait.** For a job submitted with
   `--dependency`, most of it is time spent waiting for the predecessor,
   which is by design. On the Phase-1 chain that arithmetic reported 77 hours
   against 7 real ones. Here the clock starts when the predecessor ends.

2. **With parallel jobs, summed durations are not elapsed calendar.** Two
   nodes for an hour is one hour of calendar, not two, so the busy intervals
   are merged before being measured.

3. **`sacct` has a retention window.** These intervals stop being
   recoverable some weeks after the jobs run, which is why `--json` exists:
   it freezes them to disk while they are still queryable.

    python3 queue_stats.py [alloc-start YYYY-MM-DD] [--json report.json]
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime

FMT = "%Y-%m-%dT%H:%M:%S"
STAMP = "%Y-%m-%d %H:%M"
FIELDS = "JobID,JobName,State,Submit,Start,End,ElapsedRaw,NNodes"


def hours(seconds: float) -> str:
    sign = "-" if seconds < 0 else ""
    h, m = divmod(int(abs(seconds)) // 60, 60)
    return f"{sign}{h}h{m:02d}m"


def parse(raw: str, now: datetime) -> list[dict]:
    """Rows of `sacct -P` into jobs, skipping anything that never started."""
    jobs: list[dict] = []
    for line in raw.splitlines():
        f = line.split("|")
        if len(f) < 8:
            continue
        jid, name, state, submit, start, end, elapsed, nodes = f[:8]
        if start in ("Unknown", "None", ""):
            continue            # still queued: there is no wait to close yet
        try:
            t_submit = datetime.strptime(submit, FMT)
            t_start = datetime.strptime(start, FMT)
        except ValueError:
            continue
        # A RUNNING job has no End. It is occupying the machine right now, so
        # it counts as busy up to this instant.
        try:
            t_end = datetime.strptime(end, FMT)
        except ValueError:
            t_end = now
        jobs.append({
            "id": jid, "name": name, "state": state,
            "submit": t_submit, "start": t_start, "end": t_end,
            "nodes": int(nodes or 1),
            "node_seconds": int(elapsed or 0) * int(nodes or 1),
        })
    return sorted(jobs, key=lambda j: j["start"])


def gate_of(job: dict, jobs: list[dict]) -> datetime:
    """The instant from which the job *could* have started.

    For a link in a chain that is the predecessor's end, not the submission:
    before that moment it was held back by our own `--dependency`, and
    charging that to the scheduler is how a 7-hour wait gets reported as 77.
    The predecessor is the last job of the same name to finish before this
    one started — SLURM does not record the dependency itself in `sacct`, and
    same-name-and-ordered is what a chain actually looks like.
    """
    prev_end = max(
        (j["end"] for j in jobs
         if j["name"] == job["name"] and j["id"] != job["id"]
         and j["end"] <= job["start"]),
        default=None,
    )
    return max(job["submit"], prev_end) if prev_end else job["submit"]


def merge(intervals: list[tuple]) -> list[tuple]:
    """Union of intervals, so parallel jobs do not double-count calendar."""
    out: list[tuple] = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def gaps_of(busy: list[tuple], jobs: list[dict], alloc_start: datetime,
            now: datetime) -> list[dict]:
    """Every stretch with nothing running, each attributed to a cause.

    A gap during which some job was already submitted and had not yet started
    is the cluster being full; a gap with nothing submitted is ours.
    """
    out: list[dict] = []
    cursor = alloc_start
    for s, e in busy + [(now, now)]:
        if s > cursor:
            waiting = [j["id"] for j in jobs
                       if j["submit"] <= cursor and j["start"] >= s]
            out.append({
                "from": cursor, "to": s,
                "seconds": (s - cursor).total_seconds(),
                "cause": "cluster full" if waiting else "queue empty",
                "waiting": waiting,
            })
        cursor = max(cursor, e)
    return out


def summarise(jobs: list[dict], alloc_start: datetime, now: datetime) -> dict:
    """Everything the report needs, computed once for print and for JSON."""
    waits = []
    wait_res = wait_dep = 0.0
    for j in jobs:
        gate = gate_of(j, jobs)
        r = (j["start"] - gate).total_seconds()
        wait_res += r
        wait_dep += (j["start"] - j["submit"]).total_seconds() - r
        if r >= 60:
            waits.append({"job": j["id"], "from": gate,
                          "to": j["start"], "seconds": r})

    busy = merge([(j["start"], j["end"]) for j in jobs])
    gaps = gaps_of(busy, jobs, alloc_start, now)
    busy_h = sum((e - s).total_seconds() for s, e in busy) / 3600
    used_h = sum(j["node_seconds"] for j in jobs) / 3600
    calendar_h = (now - alloc_start).total_seconds() / 3600
    full_h = sum(g["seconds"] for g in gaps if g["cause"] == "cluster full") / 3600
    empty_h = sum(g["seconds"] for g in gaps if g["cause"] == "queue empty") / 3600
    return {
        "resource_waits": waits, "idle_gaps": gaps,
        "node_hours_used": used_h, "calendar_hours": calendar_h,
        "busy_hours": busy_h,
        "idle_cluster_full_hours": full_h, "idle_queue_empty_hours": empty_h,
        "mean_nodes_when_busy": (used_h / busy_h) if busy_h else 0.0,
        "resource_wait_seconds": wait_res, "dependency_wait_seconds": wait_dep,
    }


def report(jobs: list[dict], s: dict, alloc_start: datetime) -> None:
    print("JOBS")
    print(f"{'job':>11} {'name':<22} {'state':<10} {'start':<16} "
          f"{'end':<16} {'elapsed':>8} {'nodes':>5}")
    for j in jobs:
        print(f"{j['id']:>11} {j['name'][:22]:<22} {j['state'][:10]:<10} "
              f"{j['start']:{STAMP}} {j['end']:{STAMP}} "
              f"{hours(j['node_seconds'] / j['nodes']):>8} {j['nodes']:>5}")

    print()
    print("QUEUE WAIT  (from the predecessor's end, or from submission)")
    print(f"{'job':>11} {'from':<16} {'to':<16} {'waited':>8}")
    for w in s["resource_waits"]:
        print(f"{w['job']:>11} {w['from']:{STAMP}} {w['to']:{STAMP}} "
              f"{hours(w['seconds']):>8}")
    if not s["resource_waits"]:
        print("            (nothing waited more than a minute)")

    print()
    print("CALENDAR IDLE")
    print(f"{'from':<16} {'to':<16} {'duration':>9}  cause")
    for g in s["idle_gaps"]:
        who = f"  (job {', '.join(g['waiting'])})" if g["waiting"] else ""
        print(f"{g['from']:{STAMP}} {g['to']:{STAMP}} "
              f"{hours(g['seconds']):>9}  {g['cause']}{who}")

    cal = s["calendar_hours"]
    print()
    print(f"  node-hours used             {s['node_hours_used']:9.1f}")
    print(f"  calendar since {alloc_start:%Y-%m-%d}    {cal:9.1f} h")
    print(f"    at least one job running  {s['busy_hours']:9.1f} h"
          f"   ({100 * s['busy_hours'] / cal if cal else 0:.1f} %)")
    print(f"    idle, cluster full        {s['idle_cluster_full_hours']:9.1f} h")
    print(f"    idle, queue empty         {s['idle_queue_empty_hours']:9.1f} h")
    print(f"  mean nodes while busy       {s['mean_nodes_when_busy']:9.2f}")
    print()
    print(f"  queue wait, total           {hours(s['resource_wait_seconds']):>11}")
    print(f"  dependency wait (by design) {hours(s['dependency_wait_seconds']):>11}")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    out_json = None
    if "--json" in argv:
        i = argv.index("--json")
        try:
            out_json = argv[i + 1]
        except IndexError:
            print("--json needs a path", file=sys.stderr)
            return 2
        del argv[i:i + 2]
    alloc_start = datetime.strptime(argv[0] if argv else "2026-09-02", "%Y-%m-%d")
    now = datetime.now()

    raw = subprocess.run(
        ["sacct", "-S", alloc_start.strftime("%Y-%m-%d"), "-X", "-P",
         "--noheader", f"--format={FIELDS}"],
        capture_output=True, text=True, check=True,
    ).stdout
    jobs = parse(raw, now)
    if not jobs:
        print(f"no jobs since {alloc_start:%Y-%m-%d}")
        return 1

    s = summarise(jobs, alloc_start, now)
    report(jobs, s, alloc_start)

    if out_json:
        def iso(o):
            return o.strftime(FMT) if isinstance(o, datetime) else o
        payload = {
            "captured_at": now.strftime(FMT),
            "allocation_start": alloc_start.strftime("%Y-%m-%d"),
            "jobs": [{k: iso(v) for k, v in j.items()} for j in jobs],
            "resource_waits": [{k: iso(v) for k, v in w.items()}
                               for w in s["resource_waits"]],
            "idle_gaps": [{k: iso(v) for k, v in g.items()}
                          for g in s["idle_gaps"]],
            "totals": {k: v for k, v in s.items()
                       if k not in ("resource_waits", "idle_gaps")},
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nwrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

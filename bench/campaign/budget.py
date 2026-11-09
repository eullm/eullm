"""How much of the allocation is spent, and how far behind the calendar.

LUMI-G billing, from docs.lumi-supercomputer.eu/runjobs/lumi_env/billing/:

    standard-g        GPU-hours = 4 x nodes x hours           (whole nodes)
    small-g, dev-g    GPU-hours = 0.5 x hours x
                        max(ceil(cores / 8), ceil(memory / 64 GB), GCDs)

and one node-hour is 4 GPU-hours. `lumi-allocations` is the authority; this
recomputes the same figure from `sacct` so the pace can be checked against
the calendar in one place, and so the two can be compared when they differ.
"""

from __future__ import annotations

import datetime as dt
import math
import re
import subprocess

GPU_HOURS_PER_NODE_HOUR = 4
SMALL_G_MEM_UNIT_MIB = 64 * 1024

SACCT_FIELDS = "JobID,JobName,Partition,State,ElapsedRaw,NNodes,AllocTRES,Start,End"


def mem_mib(text: str) -> float:
    """'480G', '64000M', '1T', '512000K' → MiB."""
    m = re.fullmatch(r"([\d.]+)([KMGT]?)", text.strip())
    if not m:
        return 0.0
    value, unit = float(m.group(1)), m.group(2) or "M"
    return value * {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}[unit]


def parse_tres(text: str) -> dict:
    out = {}
    for part in text.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def billed_gpu_hours(partition: str, elapsed_s: float, nodes: int, tres: dict) -> float:
    hours = elapsed_s / 3600
    partition = partition.split(",")[0]
    if partition == "standard-g":
        return GPU_HOURS_PER_NODE_HOUR * nodes * hours
    if partition in ("small-g", "dev-g"):
        gcds = int(tres.get("gres/gpu", 0) or 0)
        cores = int(tres.get("cpu", 0) or 0)
        mem = mem_mib(tres.get("mem", "0"))
        units = max(math.ceil(cores / 8), math.ceil(mem / SMALL_G_MEM_UNIT_MIB), gcds)
        return 0.5 * hours * units
    return 0.0  # CPU partitions spend a different budget


def _when(text: str):
    """A sacct time ('2026-10-10T21:53:25'), or None for 'Unknown' and the like."""
    try:
        return dt.datetime.fromisoformat(text.strip())
    except ValueError:
        return None


def parse_sacct(text: str) -> list:
    """`sacct -X -n -P -o SACCT_FIELDS` → one dict per allocation: what it
    billed, at what rate (GPU-hours per hour), and when it ran (`start`,
    `end`: None while it runs, or for output without the two fields)."""
    jobs = []
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) not in (7, 9):
            continue
        jobid, name, partition, state, elapsed, nodes, tres = parts[:7]
        if not elapsed.isdigit():
            continue
        t = parse_tres(tres)
        jobs.append({
            "job": jobid,
            "name": name,
            "partition": partition,
            "state": state,
            "elapsed_s": int(elapsed),
            "gpu_hours": billed_gpu_hours(partition, int(elapsed), int(nodes or 0), t),
            "rate": billed_gpu_hours(partition, 3600, int(nodes or 0), t),
            "start": _when(parts[7]) if len(parts) == 9 else None,
            "end": _when(parts[8]) if len(parts) == 9 else None,
        })
    return jobs


def spent_between(jobs: list, since: dt.datetime, now: dt.datetime) -> float:
    """GPU-hours billed between `since` and `now`: each job's rate over the
    part of its run in that window, a running job's up to now."""
    total = 0.0
    for j in jobs:
        if j.get("start") is None:
            continue
        end = j["end"] or now
        overlap = (min(end, now) - max(j["start"], since)).total_seconds()
        if overlap > 0:
            total += j["rate"] * overlap / 3600
    return total


def per_day(jobs: list, first: dt.date, now: dt.datetime) -> dict:
    """GPU-hours billed on each day from `first` to today: each job's rate over
    the part of its run that fell on that day, a running job's up to now."""
    days, d = {}, first
    while d <= now.date():
        start = dt.datetime.combine(d, dt.time())
        days[d] = spent_between(jobs, start, min(start + dt.timedelta(days=1), now))
        d += dt.timedelta(days=1)
    return days


ALLOCATIONS_GPU = re.compile(r"([\d.]+)\s*/\s*([\d.]+)\s*\(\s*[\d.]+%\)\s*gpu/hours", re.I)
ALLOCATIONS_UPDATED = re.compile(r"Data updated:\s*(\S.*?)\s*$", re.M)


def parse_allocations(text: str, project=None):
    """The GPU-hours `lumi-allocations` bills `project` (the first project
    listed when None): {"used", "allocated", "updated"}, or None when its
    table has no such line. This is the figure the allocation is charged."""
    for line in text.splitlines():
        if project and project not in line:
            continue
        m = ALLOCATIONS_GPU.search(line)
        if m:
            updated = ALLOCATIONS_UPDATED.search(text)
            return {"used": float(m.group(1)), "allocated": float(m.group(2)),
                    "updated": updated.group(1) if updated else None}
    return None


def burning(jobs: list) -> float:
    """GPU-hours per hour the running jobs bill right now."""
    return sum(j["rate"] for j in jobs if j["state"] == "RUNNING")


def recovery(budget_node_hours: float, start: dt.date, end: dt.date, behind: float,
             burn_node_hours_per_day: float) -> dict:
    """How fast the gap to the straight line closes at the current burn: the
    line's own rate, and the days until the gap is made up (None when it
    does not close, 0 when there is none)."""
    days = (end - start).days or 1
    line = budget_node_hours / days
    if behind <= 0:
        catch_up = 0.0
    elif burn_node_hours_per_day > line:
        catch_up = round(behind / (burn_node_hours_per_day - line), 1)
    else:
        catch_up = None
    return {"line_node_hours_per_day": round(line, 1), "catch_up_days": catch_up}


def runs_out(left_node_hours: float, rate_node_hours_per_day: float, now: dt.datetime,
             end: dt.date) -> dict:
    """When what is left is spent at a rate a day: the date, and how many days
    before the end date that is (None for both when nothing is spent at that
    rate, or when it lasts past the end). Spending the allocation by need
    rather than by the straight line makes this the date by which more
    hours have to be asked for."""
    if rate_node_hours_per_day <= 0:
        return {"date": None, "days_early": None}
    when = now + dt.timedelta(days=max(left_node_hours, 0.0) / rate_node_hours_per_day)
    early = (dt.datetime.combine(end, dt.time()) - when).total_seconds() / 86400
    if early <= 0:
        return {"date": None, "days_early": None}
    return {"date": when.date().isoformat(), "days_early": round(early, 1)}


def margin(budget_node_hours: float, start: dt.date, end: dt.date, now: dt.datetime,
           spent_node_hours: float) -> dict:
    """How far the spend is ahead of the straight line from the start to the
    end of `end` (negative: behind it), in node-hours and in days of that
    line's rate: the days without a job (a maintenance, no way to log in)
    the allocation can take before an hour of it goes unspent at the end.
    The PI's rule since 04-11-2026: always ahead, never on the line."""
    begin = dt.datetime.combine(start, dt.time())
    span = (dt.datetime.combine(end + dt.timedelta(days=1), dt.time()) - begin).total_seconds()
    done = min(max((now - begin).total_seconds() / span, 0.0), 1.0) if span > 0 else 1.0
    ahead = spent_node_hours - budget_node_hours * done
    rate = budget_node_hours / (span / 86400) if span > 0 else 0.0
    return {"ahead_node_hours": round(ahead, 1), "line_node_hours_per_day": round(rate, 1),
            "days": round(ahead / rate, 1) if rate else None}


def read_sacct(start: str, account=None) -> list:
    cmd = ["sacct", "-X", "-n", "-P", "-S", start, "-E", "now", "-o", SACCT_FIELDS]
    if account:
        cmd += ["-A", account, "--allusers"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                         encoding="utf-8", errors="replace")
    if out.returncode != 0:
        raise RuntimeError(f"sacct failed: {out.stderr.strip()}")
    return parse_sacct(out.stdout)


def pace(budget_node_hours: float, start: dt.date, end: dt.date, now: dt.datetime,
         spent_node_hours: float) -> dict:
    """Where the spend stands against a straight line from start to end."""
    total_h = (dt.datetime.combine(end, dt.time()) - dt.datetime.combine(start, dt.time()))
    total_h = total_h.total_seconds() / 3600
    elapsed_h = (now - dt.datetime.combine(start, dt.time())).total_seconds() / 3600
    elapsed_h = min(max(elapsed_h, 0.0), total_h)
    left_h = total_h - elapsed_h
    target = budget_node_hours * elapsed_h / total_h if total_h else 0.0
    remaining = budget_node_hours - spent_node_hours
    per_day = remaining / (left_h / 24) if left_h > 0 else None
    return {
        "budget_node_hours": budget_node_hours,
        "spent_node_hours": round(spent_node_hours, 1),
        "spent_pct": round(100 * spent_node_hours / budget_node_hours, 1),
        "calendar_pct": round(100 * elapsed_h / total_h, 1) if total_h else None,
        "target_to_date": round(target, 1),
        "behind_node_hours": round(target - spent_node_hours, 1),
        "days_left": round(left_h / 24, 1),
        "needed_node_hours_per_day": round(per_day, 1) if per_day is not None else None,
        # Nodes kept busy around the clock from now on to spend the rest.
        "needed_nodes_continuous": round(per_day / 24, 2) if per_day is not None else None,
    }

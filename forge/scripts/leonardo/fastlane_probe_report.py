#!/usr/bin/env python3
"""Read the Fast Lane's first-day probes back and say what they mean.

`fastlane_probe.sh` submits one probe per job shape (3 GPUs or whole nodes,
2 to 24 hours, 1 to 8 nodes, every QOS the project may use) and writes their
ids to `manifest.tsv`. This reads the scheduler's answer for each:

* **started**, and after how long: that shape is usable;
* **pending because of a limit** (a reason naming a QOS or association
  limit): that shape is not allowed, however long one waits;
* **pending for resources or priority**: allowed, but the machine has not
  made room yet. At the last report that is the answer too: it did not start
  within the window.

It also says whether the two 16-GPU probes held on purpose (`hold` > 0) ran
at the same time: the 16 + 16 question.

Run by `sbatch_fastlane_probe_report.slurm` at 1, 6 and 24 hours; the last
run passes ``--final`` and cancels the probes still waiting, so nothing is
left in the queue. Writes ``report.md`` and ``report.json`` in the probe
directory, overwriting the previous report.

Usage:
    python3 fastlane_probe_report.py DIR [--final]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

#: Pending reasons that mean a configured limit, not a busy machine. Slurm
#: spells them QOSMaxGRESPerUser, AssocGrpGRES, QOSMaxWallDurationPerJobLimit,
#: QOSMaxNodePerJobLimit and so on.
LIMIT = re.compile(r"^(QOS|Assoc|Partition)\w*(Limit|Max|Grp|PerUser|PerJob|PerAccount)\w*"
                   r"|^(QOS|Assoc)(Max|Grp)\w*")
WAITING = {"Resources", "Priority", "ReqNodeNotAvail", "Reservation", "None", "BeginTime"}


def read_manifest(path: Path) -> list[dict]:
    """The probes as the launcher wrote them: one per line, tab-separated,
    with a header. Rejected submissions have no id but a note saying why."""
    lines = path.read_text(encoding="utf-8").splitlines()
    head = lines[0].split("\t")
    return [dict(zip(head, line.split("\t"))) for line in lines[1:] if line.strip()]


def parse_time(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.strip())
    except ValueError:  # "Unknown", "None", "N/A", ""
        return None


def parse_sacct(text: str) -> dict[str, dict]:
    """``sacct -X -n -P -o JobID,State,Submit,Start,End,Reason``."""
    out = {}
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) < 6:
            continue
        jid, state, submit, start, end, reason = parts[:6]
        out[jid] = {"state": state.split()[0] if state else "", "submit": parse_time(submit),
                    "start": parse_time(start), "end": parse_time(end), "reason": reason}
    return out


def parse_squeue(text: str) -> dict[str, dict]:
    """``squeue -h -o %i|%T|%r|%S``: the reason and the planned start of what
    is still queued, which sacct does not know."""
    out = {}
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue
        jid, state, reason, start = parts[:4]
        out[jid] = {"state": state, "reason": reason, "planned": parse_time(start)}
    return out


def classify(state: str, reason: str) -> str:
    """started / limit / waiting / other, from the job's state and reason."""
    if state in {"RUNNING", "COMPLETED", "COMPLETING", "TIMEOUT"}:
        return "started"
    if state == "PENDING":
        if LIMIT.match(reason or ""):
            return "limit"
        if (reason or "None") in WAITING:
            return "waiting"
    return "other"


def hours(delta_seconds: float | None) -> str:
    if delta_seconds is None:
        return ""
    if delta_seconds < 120:
        return f"{int(delta_seconds)} s"
    if delta_seconds < 7200:
        return f"{delta_seconds / 60:.0f} min"
    return f"{delta_seconds / 3600:.1f} h"


def overlap(a: dict, b: dict) -> bool | None:
    """Whether two probes ran at the same time; None while either has not
    started. A probe still running has no end yet, which counts as now."""
    if not a.get("start") or not b.get("start"):
        return None
    now = datetime.now()
    a_end = a.get("end") or now
    b_end = b.get("end") or now
    return a["start"] < b_end and b["start"] < a_end


def build(manifest: list[dict], sacct: dict, squeue: dict, now: datetime) -> list[dict]:
    rows = []
    for probe in manifest:
        jid = probe.get("id", "")
        acct = sacct.get(jid, {})
        queued = squeue.get(jid, {})
        state = queued.get("state") or acct.get("state") or ("REJECTED" if not jid else "?")
        reason = queued.get("reason") or acct.get("reason", "")
        submit = acct.get("submit") or parse_time(probe.get("submitted", ""))
        start = acct.get("start")
        verdict = "rejected" if not jid else classify(state, reason)
        if start and submit:
            waited = (start - submit).total_seconds()
        elif submit:
            waited = (now - submit).total_seconds()
        else:
            waited = None
        rows.append({**probe, "state": state, "reason": reason, "verdict": verdict,
                     "waited_s": waited, "start": start, "end": acct.get("end"),
                     "planned": queued.get("planned")})
    return rows


def concurrency(rows: list[dict]) -> str:
    held = [r for r in rows if r.get("hold", "0") not in {"", "0"} and r.get("id")]
    if len(held) < 2:
        return "non provato (meno di due probe trattenuti)"
    a, b = held[:2]
    both = overlap(a, b)
    gpus = sum(int(r.get("nodes", 0)) * int(r.get("gpus", 0)) for r in (a, b))
    if both is True:
        return f"SÌ: i due job sono girati insieme ({gpus} GPU contemporanee)"
    if both is False:
        return ("non dimostrato: sono partiti uno dopo l'altro; se nessuno dei due ha un "
                "motivo di limite, era la macchina piena e la prova va ripetuta")
    blocked = [r for r in (a, b) if r["verdict"] == "limit"]
    if blocked:
        return f"NO, limite: {blocked[0]['reason']}"
    return "non ancora: almeno uno dei due non è partito (" + \
        ", ".join(f"{r['name']} {r['state']} {r['reason']}" for r in (a, b)) + ")"


def render(rows: list[dict], final: bool, now: datetime) -> str:
    out = [f"# Probe Fast Lane — {now:%Y-%m-%d %H:%M}" + (" (finale)" if final else ""), "",
           "| probe | nodi × GPU | ore | QOS | esito | attesa | stato / motivo |",
           "|---|---|---|---|---|---|---|"]
    label = {"started": "partito", "limit": "**limite**", "waiting": "in coda",
             "rejected": "**rifiutato**", "other": "?"}
    for r in rows:
        why = r["reason"] if r["verdict"] != "rejected" else r.get("note", "")
        if r["verdict"] == "waiting" and r.get("planned"):
            why += f", partenza prevista {r['planned']:%d/%m %H:%M}"
        waited = hours(r["waited_s"])
        if r["verdict"] != "started" and waited:
            waited = "da " + waited
        shape = f"{r.get('nodes', '')} × {r.get('gpus', '')}"
        out.append(f"| {r['name']} | {shape} | {r.get('time', '')} | {r.get('qos', '')} "
                   f"| {label[r['verdict']]} | {waited} | {r['state']} {why} |")
    out += ["", f"**16 + 16 in parallelo:** {concurrency(rows)}", ""]
    started = [r for r in rows if r["verdict"] == "started"]
    if started:
        biggest = max(started, key=lambda r: (int(r.get("nodes", 0)) * int(r.get("gpus", 0)),
                                              r.get("time", "")))
        out.append(f"**Forma più grande partita:** {biggest['name']} "
                   f"({biggest['nodes']} × {biggest['gpus']} GPU, {biggest['time']}) "
                   f"in {hours(biggest['waited_s'])}.")
    limits = sorted({r["reason"] for r in rows if r["verdict"] == "limit"})
    if limits:
        out.append(f"**Limiti incontrati:** {', '.join(limits)}.")
    if final:
        out.append("")
        out.append("Report finale: i probe ancora in coda sono stati cancellati.")
    return "\n".join(out) + "\n"


def run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"[flp] {' '.join(cmd[:2])} failed: {exc}", file=sys.stderr)
        return ""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("dir", type=Path)
    p.add_argument("--final", action="store_true",
                   help="cancel the probes still queued after writing the report")
    a = p.parse_args(argv)

    manifest = read_manifest(a.dir / "manifest.tsv")
    ids = [m["id"] for m in manifest if m.get("id")]
    if not ids:
        print("[flp] no submitted probe in the manifest", file=sys.stderr)
        return 1
    sacct = parse_sacct(run(["sacct", "-X", "-n", "-P", "-j", ",".join(ids),
                             "-o", "JobID,State,Submit,Start,End,Reason"]))
    squeue = parse_squeue(run(["squeue", "--me", "-h", "-o", "%i|%T|%r|%S"]))
    now = datetime.now()
    rows = build(manifest, sacct, squeue, now)

    text = render(rows, a.final, now)
    (a.dir / "report.md").write_text(text, encoding="utf-8")
    (a.dir / "report.json").write_text(json.dumps(rows, default=str, indent=1), encoding="utf-8")
    print(text)

    if a.final:
        pending = [r["id"] for r in rows if r.get("id") and r["state"] == "PENDING"]
        if pending:
            run(["scancel", *pending])
            print(f"[flp] cancelled {len(pending)} probes still queued: {' '.join(pending)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

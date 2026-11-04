#!/usr/bin/env bash
# LUMI-G node-hours at a glance, as tools/jupiter/nodi.sh on JUPITER: each of
# the last 7 days, the average a day, what is left and what a day must spend
# to use it by the plan's end, and what runs and waits now. One node-hour is
# 4 GPU-hours: a standard-g node for an hour; small-g bills 0.5 a GCD-hour
# (bench/campaign/budget.py).
#
#     bash tools/lumi/nodi.sh            # or `nodi`, with the alias:
#     echo "alias nodi='bash $PWD/tools/lumi/nodi.sh'" >> ~/.bashrc
#
# The days come from Slurm's accounting (sacct, every user of the project,
# today included); what is left from lumi-allocations, which is what the
# allocation is charged, and from Slurm. The plan ends on 09-03-2027, three
# days before the allocation (12-03-2027). `campaign.py budget` has the
# long form.
#
#   LUMI_PLAN_END=2027-03-09 LUMI_ALLOC_END=2027-03-12 LUMI_START=2026-09-12
#   LUMI_NODE_HOURS=4500       what the allocation holds

set -uo pipefail
# shellcheck source-path=SCRIPTDIR source=campaign_env.sh
source "$(dirname "$0")/campaign_env.sh"
ACCOUNT="${SBATCH_ACCOUNT:-project_465003366}"

ALLOC=$(lumi-allocations 2>/dev/null)
RUN=$(squeue --me -h -t RUNNING -o '%j/%P' 2>/dev/null | sort | uniq -c | awk '{ printf "%s%s x%d", (NR > 1 ? ", " : ""), $2, $1 }')
WAIT=$(squeue --me -h -t PENDING -o '%j/%P' 2>/dev/null | sort | uniq -c | awk '{ printf "%s%s x%d", (NR > 1 ? ", " : ""), $2, $1 }')

ALLOC="$ALLOC" RUN="$RUN" WAIT="$WAIT" "$PY" - "$EULLM_REPO/bench/campaign" "$ACCOUNT" \
    "${LUMI_NODE_HOURS:-4500}" "${LUMI_START:-2026-09-12}" "${LUMI_PLAN_END:-2027-03-09}" \
    "${LUMI_ALLOC_END:-2027-03-12}" <<'PY'
import datetime as dt
import math
import os
import sys

sys.path.insert(0, sys.argv[1])
import budget  # noqa: E402

account, total = sys.argv[2], float(sys.argv[3])
start, plan_end, alloc_end = (dt.date.fromisoformat(a) for a in sys.argv[4:7])
per = budget.GPU_HOURS_PER_NODE_HOUR
now = dt.datetime.now()
try:
    jobs = budget.read_sacct(start.isoformat(), account)
except (RuntimeError, OSError) as e:
    sys.exit(f"sacct: {e}")

today = now.date()
days = budget.per_day(jobs, today - dt.timedelta(days=6), now)
unit = max(1, math.ceil(max(days.values(), default=0) / per / 40))
print(f"LUMI-G node-hours (1 = {per} GPU-hours), {account}, last 7 days (# = {unit}):")
for d, gpu_h in days.items():
    label = "today, so far" if d == today else d.strftime("%a %d-%m")
    print(f"  {d}  {label:<14} {gpu_h / per:6.1f}  {'#' * round(gpu_h / per / unit)}")
week = budget.spent_between(jobs, now - dt.timedelta(days=7), now) / per
spent = budget.spent_between(jobs, dt.datetime.combine(start, dt.time()), now) / per
elapsed = (now - dt.datetime.combine(start, dt.time())).total_seconds() / 86400
print(f"  {'last 7 days, to now':<26} {week:6.1f}  ({week / 7:.1f} a day)")
print(f"  {'since ' + start.isoformat():<26} {spent:6.1f}  ({spent / elapsed:.1f} a day over "
      f"{elapsed:.0f} days)")

billed = budget.parse_allocations(os.environ.get("ALLOC", ""), account)
left_slurm = total - spent
print()
if billed:
    left = (billed["allocated"] - billed["used"]) / per
    print(f"Left: {left:,.0f} node-hours billed by lumi-allocations (updated "
          f"{billed['updated'] or '?'}), {left_slurm:,.0f} by Slurm, of {billed['allocated'] / per:,.0f}")
else:
    left = left_slurm
    print(f"Left: {left:,.0f} node-hours by Slurm, of {total:,.0f} (lumi-allocations gave no figure)")
for label, end in (("to the plan's end", plan_end), ("to the allocation's end", alloc_end)):
    to_go = (dt.datetime.combine(end + dt.timedelta(days=1), dt.time()) - now).total_seconds() / 86400
    need = left / to_go
    print(f"  {label} {end}: {to_go:5.1f} days, {need:5.1f} node-hours a day "
          f"({need / 24:.2f} nodes busy around the clock)")
# Days without a job are covered only by spending ahead of the line (PI, 04-11-2026).
alloc_total = billed["allocated"] / per if billed else total
m = budget.margin(alloc_total, start, plan_end, now, alloc_total - left)
if m["ahead_node_hours"] >= 0:
    print(f"  ahead of the straight line {start} to {plan_end}: {m['ahead_node_hours']:,.0f} "
          f"node-hours, {m['days']:.1f} days without a job before an hour is left over")
else:
    print(f"  BEHIND the straight line {start} to {plan_end} by {-m['ahead_node_hours']:,.0f} "
          f"node-hours ({-m['days']:.1f} days at its {m['line_node_hours_per_day']:.1f} a day): "
          f"a day without a job now leaves hours unspent at the end")
if week > 0:
    out = budget.runs_out(left, week / 7, now, plan_end)
    if out["date"]:
        print(f"  at the last 7 days' {week / 7:.1f} a day the hours run out on {out['date']}, "
              f"{out['days_early']:.0f} days before {plan_end}: more are asked for by then")
    else:
        print(f"  at the last 7 days' {week / 7:.1f} a day they last past {plan_end}")
print()
print(f"Running: {os.environ.get('RUN') or 'nothing'}")
print(f"Waiting: {os.environ.get('WAIT') or 'nothing'}")
PY

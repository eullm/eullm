#!/usr/bin/env bash
# Where the allocation stands, in one screen: spend against the calendar,
# the campaign jobs in Slurm, the queue, and an alarm when no whole-node job
# is running or waiting — the one state in which budget is being lost.
#
#   bash tools/lumi/status.sh

set -uo pipefail
# shellcheck source-path=SCRIPTDIR source=campaign_env.sh
source "$(dirname "$0")/campaign_env.sh"

# lumi-allocations is what the allocation is charged; sacct is our own
# recount, and the only source for the last 24 hours. Both are shown.
ALLOC=""
if command -v lumi-allocations >/dev/null 2>&1; then
    ALLOC=$(mktemp) && lumi-allocations > "$ALLOC" 2>&1
fi
echo "=== budget: EHPC-DEV-2026D09-278, 4,500 node-h, 12-09-2026 → 12-03-2027 ==="
$CAMPAIGN budget ${ALLOC:+--allocations "$ALLOC"} || echo "  (sacct unavailable: run on a LUMI login node)"
if [ -n "$ALLOC" ]; then
    echo
    echo "--- lumi-allocations (the authority; sacct above is our own recount) ---"
    sed 's/^/  /' "$ALLOC"
    rm -f "$ALLOC"
fi

echo
echo "=== campaign jobs ==="
JOBS=$(squeue --me -h -n eullm-campaign -o '%i %t %M %L %R' 2>/dev/null)
if [ -n "$JOBS" ]; then
    echo "  job        state elapsed    left       where/why"
    echo "$JOBS" | awk '{printf "  %-10s %-5s %-10s %-10s %s\n", $1, $2, $3, $4, $5}'
fi
RUNNING=$(echo "$JOBS" | awk '$2=="R"' | grep -c . || true)
PENDING=$(echo "$JOBS" | awk '$2=="PD"' | grep -c . || true)
echo "  running $RUNNING, pending $PENDING"
if [ "$RUNNING" -eq 0 ] && [ "$PENDING" -eq 0 ]; then
    echo
    echo "  !!! NO WHOLE-NODE JOB RUNNING OR QUEUED — every hour from now is budget lost."
    echo "  !!! bash $EULLM_REPO/tools/lumi/submit_campaign.sh"
elif [ "$PENDING" -le 1 ]; then
    echo "  (chains are nearly used up: TOP_UP=1 bash $EULLM_REPO/tools/lumi/submit_campaign.sh N" \
         "keeps N of them)"
fi

echo
echo "=== campaign queue: $CAMPAIGN_DIR ==="
$CAMPAIGN status --queue "$CAMPAIGN_DIR"
TODO=$($CAMPAIGN status --queue "$CAMPAIGN_DIR" | awk '/^queue:/{print $3}')
if [ "${TODO:-0}" = "0" ]; then
    echo "  !!! nothing left to run: plan the next campaign (campaign.py plan)"
fi

LAST=$(find "$CAMPAIGN_DIR/results" -maxdepth 1 -name '*.summary.json' -printf '%T@ %p\n' \
    2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
if [ -n "$LAST" ]; then
    echo
    echo "=== last finished job: $(basename "$LAST" .summary.json) ==="
    $PY - "$LAST" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"  {d['elapsed_s'] / 3600:.1f} h on {d['host']}: {d['points']}")
frac = d.get("assigned_fraction", {})
print("  share of the job each GCD had a point: "
      + "  ".join(f"{k}:{v:.0%}" for k, v in sorted(frac.items())))
PY
    JOB=$(basename "$LAST" .summary.json)
    [ -f "$CAMPAIGN_DIR/logs/summary-$JOB.txt" ] &&
        echo "  what it measured: cat $CAMPAIGN_DIR/logs/summary-$JOB.txt"
fi

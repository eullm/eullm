#!/usr/bin/env bash
# Where the allocation actually stands, including what is still in flight.
#
# `saldo -b` only counts jobs that have FINISHED. A run that has been on a node
# for ten hours shows as zero until it ends, so the one command everybody
# reaches for is a lagging indicator — and the lag is exactly as large as the
# thing you most want to see. On a four-job chain the gap between what saldo
# says and what is really committed can be most of a phase.
#
# Three numbers, and the third is the one that answers "how much can I still
# promise to something else":
#
#   settled    what saldo reports — completed jobs only
#   in flight  running jobs, elapsed x nodes x 32 core-hours per node-hour
#   committed  settled + in flight + the walltime every queued and running job
#              could still consume. The worst case, known now rather than
#              discovered at the end.
#
# Usage:
#   bash forge/scripts/leonardo/budget.sh
#   bash forge/scripts/leonardo/budget.sh --reserve 200   # node-hours to keep
#
# CINECA bills a whole node: 1 node-hour = 32 core-hours on Booster, whatever
# the job does with the GPUs.

set -uo pipefail

CORES_PER_NODE="${CORES_PER_NODE:-32}"
RESERVE_NODE_H=0
if [ "${1:-}" = "--reserve" ]; then
    RESERVE_NODE_H="${2:?--reserve needs a number of node-hours}"
fi

# ── settled, from saldo ──────────────────────────────────────────────────
# Columns on the data row: 1 account, 2 start, 3 end, 4 total, 5 localCluster,
# 6 totConsumed, 7 pct, 8 monthTotal, 9 monthConsumed. Guarded by NF so a
# change in saldo's output degrades to a clear message instead of nonsense.
read -r TOTAL SETTLED MONTH_TOTAL MONTH_USED <<<"$(
    saldo -b 2>/dev/null |
    awk 'NF >= 9 && $4 ~ /^[0-9]+$/ && $2 ~ /^[0-9]{8}$/ {print $4, $6, $8, $9; exit}'
)"
if [ -z "${TOTAL:-}" ]; then
    echo "[err] could not parse 'saldo -b' — run it by hand and check the columns" >&2
    exit 2
fi

# ── in flight and committed, from sacct/squeue ───────────────────────────
# ElapsedRaw is seconds, which avoids parsing SLURM's D-HH:MM:SS by hand.
IN_FLIGHT_CH="$(
    sacct -X -n -s RUNNING --format=ElapsedRaw,NNodes 2>/dev/null |
    awk -v c="$CORES_PER_NODE" '{s += $1 * $2 * c} END {printf "%.0f", s/3600}'
)"
IN_FLIGHT_CH="${IN_FLIGHT_CH:-0}"

# %L is the remaining time limit for running jobs and the full limit for
# pending ones — i.e. exactly what each job may still burn.
REMAINING_CH="$(
    squeue --me -h -o '%L %D' 2>/dev/null |
    awk -v c="$CORES_PER_NODE" '
        {
            n = split($1, p, "-")            # optional  D-HH:MM:SS
            days = (n == 2) ? p[1] : 0
            t = (n == 2) ? p[2] : $1
            m = split(t, q, ":")
            sec = (m == 3) ? q[1]*3600 + q[2]*60 + q[3] : q[1]*60 + q[2]
            s += (days*86400 + sec) * $2 * c
        }
        END {printf "%.0f", s/3600}'
)"
REMAINING_CH="${REMAINING_CH:-0}"

COMMITTED_CH=$(( SETTLED + IN_FLIGHT_CH + REMAINING_CH ))
FREE_CH=$(( TOTAL - COMMITTED_CH ))
RESERVE_CH=$(( RESERVE_NODE_H * CORES_PER_NODE ))

nodeh() { awk -v v="$1" -v c="$CORES_PER_NODE" 'BEGIN {printf "%.1f", v/c}'; }
pct()   { awk -v v="$1" -v t="$TOTAL" 'BEGIN {printf "%.1f", (t ? 100*v/t : 0)}'; }

printf '\n  allocation      %8s core-h  = %8s node-h\n' "$TOTAL" "$(nodeh "$TOTAL")"
printf '  ─────────────────────────────────────────────────────\n'
printf '  settled         %8s core-h  = %8s node-h   %5s%%   (saldo: finished jobs only)\n' \
    "$SETTLED" "$(nodeh "$SETTLED")" "$(pct "$SETTLED")"
printf '  in flight       %8s core-h  = %8s node-h   %5s%%   (running, not yet billed)\n' \
    "$IN_FLIGHT_CH" "$(nodeh "$IN_FLIGHT_CH")" "$(pct "$IN_FLIGHT_CH")"
printf '  still bookable  %8s core-h  = %8s node-h   %5s%%   (walltime queued+running may use)\n' \
    "$REMAINING_CH" "$(nodeh "$REMAINING_CH")" "$(pct "$REMAINING_CH")"
printf '  ─────────────────────────────────────────────────────\n'
printf '  COMMITTED       %8s core-h  = %8s node-h   %5s%%\n' \
    "$COMMITTED_CH" "$(nodeh "$COMMITTED_CH")" "$(pct "$COMMITTED_CH")"
printf '  free to promise %8s core-h  = %8s node-h   %5s%%\n' \
    "$FREE_CH" "$(nodeh "$FREE_CH")" "$(pct "$FREE_CH")"
printf '\n  month           %8s core-h quota, %s used\n' "$MONTH_TOTAL" "$MONTH_USED"

if [ "$RESERVE_CH" -gt 0 ]; then
    printf '\n'
    if [ "$FREE_CH" -lt "$RESERVE_CH" ]; then
        printf '  [!!] reserve of %s node-h NOT covered: %s node-h free.\n' \
            "$RESERVE_NODE_H" "$(nodeh "$FREE_CH")"
        printf '       Cancel a queued job to release its walltime, or shorten\n'
        printf '       #SBATCH --time on the ones not yet started.\n\n'
        exit 1
    fi
    printf '  [ok] reserve of %s node-h covered, %s node-h free beyond it.\n\n' \
        "$RESERVE_NODE_H" "$(nodeh $(( FREE_CH - RESERVE_CH )))"
fi

# "still bookable" is a worst case on purpose: a job that finishes early gives
# its remaining walltime back. Treat it as the ceiling to plan against, not as
# a forecast of what will be spent.

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
#   consumed        every job since the allocation opened, finished and
#                   running alike, elapsed x nodes x 32 core-hours per node-hour
#   still bookable  the walltime every queued and running job could yet use
#   committed       the two together. The worst case, known now rather than
#                   discovered at the end.
#
# Usage:
#   bash forge/scripts/leonardo/budget.sh
#   bash forge/scripts/leonardo/budget.sh --reserve 200   # node-hours to keep
#
# Calendar, rather than node-hours, is the binding constraint on this
# allocation — see docs/leonardo-allocation-plan.md. `queue_stats.py` in this
# directory measures that side: where the calendar went and whose fault each
# idle stretch was.
#
# CINECA bills 8 local hours per GPU-hour on Booster (32 a node-hour): a
# 1-GPU job is a quarter of a node, not a node. Until 2026-11-20 this script
# counted every job as a whole node and read 1,767 node-hours consumed where
# saldo had about 650: the Fast Lane looked ahead of its pace while it was
# far behind it.

set -uo pipefail

CORES_PER_NODE="${CORES_PER_NODE:-32}"
CORES_PER_GPU="${CORES_PER_GPU:-8}"
PARTITION="${PARTITION:-boost_usr_prod}"
RESERVE_NODE_H=0
if [ "${1:-}" = "--reserve" ]; then
    RESERVE_NODE_H="${2:?--reserve needs a number of node-hours}"
fi

# ── the allocation's own numbers, from saldo ─────────────────────────────
# Columns on the data row: 1 account, 2 start, 3 end, 4 total, 5 localCluster,
# 6 totConsumed, 7 pct, 8 monthTotal, 9 monthConsumed. Guarded by NF so a
# change in saldo's output degrades to a clear message instead of nonsense.
# Only an allocation still open: from 2026-11-02 saldo lists the ended
# Playground (AIFAC_P02_1147) before the Fast Lane, and its first row made the
# Fast Lane read -191% free on 2026-11-20. $EULLM_ACCOUNT picks among several
# open ones; otherwise the first open one.
read -r TOTAL ALLOC_START SETTLED MONTH_TOTAL MONTH_USED ALLOC_END <<<"$(
    saldo -b 2>/dev/null |
    awk -v acct="${EULLM_ACCOUNT:-}" -v today="$(date +%Y%m%d)" '
        NF >= 9 && $4 ~ /^[0-9]+$/ && $2 ~ /^[0-9]{8}$/ && $3 >= today {
            row = $4 " " $2 " " $6 " " $8 " " $9 " " $3
            if ($1 == acct) { print row; found = 1; exit }
            if (first == "") first = row
        }
        END { if (!found && first != "") print first }'
)"
if [ -z "${TOTAL:-}" ]; then
    echo "[err] could not parse 'saldo -b' — run it by hand and check the columns" >&2
    exit 2
fi
# saldo prints the allocation start as YYYYMMDD; sacct wants YYYY-MM-DD.
SINCE="${ALLOC_START:0:4}-${ALLOC_START:4:2}-${ALLOC_START:6:2}"

# ── consumed, from sacct ─────────────────────────────────────────────────
# NOT from saldo. On 2026-09-11 saldo's totConsumed read 907 core-hours while
# the jobs since the allocation opened added up to 1,675: a job that ENDED
# hours earlier is neither still running nor yet billed, and falls through the
# gap between the two columns. Summing sacct over the whole allocation counts
# finished and running work alike, and needs no reconciliation.
#
# ElapsedRaw is seconds, which avoids parsing SLURM's D-HH:MM:SS by hand.
# Restricted to one partition because billing is per-partition: a one-core
# job on the serial queue reports NNodes=1 and would otherwise be charged a
# whole 32-core Booster node.
# Billed by GPU (gres/gpu in AllocTRES, 8 cores each); a job with no GPU
# there is billed by its cores.
CONSUMED_CH="$(
    sacct -X -n -P -S "$SINCE" -r "$PARTITION" --format=ElapsedRaw,AllocTRES,AllocCPUS 2>/dev/null |
    awk -F'|' -v g="$CORES_PER_GPU" '
        { gpus = 0
          if (match($2, /gres\/gpu=[0-9]+/)) gpus = substr($2, RSTART + 9, RLENGTH - 9)
          s += $1 * (gpus > 0 ? gpus * g : $3) }
        END {printf "%.0f", s/3600}'
)"
CONSUMED_CH="${CONSUMED_CH:-0}"

# %L is the remaining time limit for running jobs and the full limit for
# pending ones — i.e. exactly what each job may still burn.
REMAINING_CH="$(
    squeue --me -h -p "$PARTITION" -o '%L %D %b %C' 2>/dev/null |
    awk -v g="$CORES_PER_GPU" '
        {
            n = split($1, p, "-")            # optional  D-HH:MM:SS
            days = (n == 2) ? p[1] : 0
            t = (n == 2) ? p[2] : $1
            m = split(t, q, ":")
            sec = (m == 3) ? q[1]*3600 + q[2]*60 + q[3] : q[1]*60 + q[2]
            gpus = 0                         # %b: gres per node, "gres/gpu:3" or "gpu:3"
            if (match($3, /gpu(:[a-zA-Z0-9_]+)?:[0-9]+/)) { k = split(substr($3, RSTART, RLENGTH), x, ":"); gpus = x[k] }
            s += (days*86400 + sec) * (gpus > 0 ? gpus * g * $2 : $4)
        }
        END {printf "%.0f", s/3600}'
)"
REMAINING_CH="${REMAINING_CH:-0}"

COMMITTED_CH=$(( CONSUMED_CH + REMAINING_CH ))
FREE_CH=$(( TOTAL - COMMITTED_CH ))
RESERVE_CH=$(( RESERVE_NODE_H * CORES_PER_NODE ))

nodeh() { awk -v v="$1" -v c="$CORES_PER_NODE" 'BEGIN {printf "%.1f", v/c}'; }
pct()   { awk -v v="$1" -v t="$TOTAL" 'BEGIN {printf "%.1f", (t ? 100*v/t : 0)}'; }

printf '\n  allocation      %8s core-h  = %8s node-h\n' "$TOTAL" "$(nodeh "$TOTAL")"
printf '  ─────────────────────────────────────────────────────\n'
printf '  consumed        %8s core-h  = %8s node-h   %5s%%   (sacct, %s onward, finished + running)\n' \
    "$CONSUMED_CH" "$(nodeh "$CONSUMED_CH")" "$(pct "$CONSUMED_CH")" "$SINCE"
printf '  still bookable  %8s core-h  = %8s node-h   %5s%%   (walltime queued+running may use)\n' \
    "$REMAINING_CH" "$(nodeh "$REMAINING_CH")" "$(pct "$REMAINING_CH")"
printf '  ─────────────────────────────────────────────────────\n'
printf '  COMMITTED       %8s core-h  = %8s node-h   %5s%%\n' \
    "$COMMITTED_CH" "$(nodeh "$COMMITTED_CH")" "$(pct "$COMMITTED_CH")"
printf '  free to promise %8s core-h  = %8s node-h   %5s%%\n' \
    "$FREE_CH" "$(nodeh "$FREE_CH")" "$(pct "$FREE_CH")"
# The straight line: where an even spend would be today, and the pace the
# rest needs. The end date is saldo's (column 3).
read -r PACE_LINE PACE_NEED DAYS_LEFT <<<"$(awk -v t="$TOTAL" -v c="$CONSUMED_CH" \
    -v s="$(date -d "$SINCE" +%s)" -v e="$(date -d "${ALLOC_END:0:4}-${ALLOC_END:4:2}-${ALLOC_END:6:2}" +%s)" \
    -v now="$(date +%s)" 'BEGIN { d = (e - s) / 86400; done = (now - s) / 86400; left = d - done
        printf "%.0f %.0f %.0f", t * done / d, (left > 0 ? (t - c) / left : 0), left }')"
printf '\n  straight line   %8s core-h  = %8s node-h   (an even spend, today)\n' \
    "$PACE_LINE" "$(nodeh "$PACE_LINE")"
printf '  ahead (+) / behind (-) %+.0f node-h; the rest needs %s node-h a day (%s GPUs round the clock) for %s days\n' \
    "$(awk -v a="$CONSUMED_CH" -v b="$PACE_LINE" -v c="$CORES_PER_NODE" 'BEGIN {print (a-b)/c}')" \
    "$(nodeh "$PACE_NEED")" "$(awk -v v="$PACE_NEED" -v g="$CORES_PER_GPU" 'BEGIN {printf "%.0f", v/g/24}')" "$DAYS_LEFT"
printf '\n  month           %8s core-h quota, %s used\n' "$MONTH_TOTAL" "$MONTH_USED"
printf '  saldo settled   %8s core-h              (lags: cross-check only)\n' "$SETTLED"

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

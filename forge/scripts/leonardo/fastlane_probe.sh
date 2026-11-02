#!/bin/bash
# The Fast Lane's first test, run once from a login node as soon as the new
# project is active. Before a single training job, it answers three questions
# whose answers decide how the 30,000 GPU-hours can be spent at all:
#
#   1. What the project may ask for: its QOS, the wall-time and GPU limits,
#      the priority (sacctmgr, scontrol, saldo), written to limits.txt.
#   2. Which job shapes start, and how soon: 3 GPUs or whole nodes, 1 to 8
#      nodes, 2 to 24 hours, under every QOS the project has. On the
#      Playground (1147) only 3-GPU jobs of at most 2 hours started; if the
#      Fast Lane is the same, the 326 GPU-hours a day it needs are out of
#      reach, and CINECA must be asked for something else now, not in December.
#   3. Whether two jobs of 16 GPUs can run at the same time (16 + 16): two
#      4-node probes that stay up 15 minutes each.
#
# Every probe but the 16 + 16 pair exits as soon as it starts, so the whole
# test costs about ten GPU-hours. Reports are written by jobs on the serial
# partition at +1 h, +6 h and +24 h (report.md in the probe directory); the
# last one cancels the probes still waiting. Nothing to watch, nothing to
# clean up by hand.
#
# Usage (login node, after `module load python/3.11.7`):
#   export EULLM_ACCOUNT=<new project account, from saldo -b>
#   bash fastlane_probe.sh
#   cat <dir>/limits.txt            # at once
#   cat <dir>/report.md             # from one hour on
#
# Environment: FLP_OUT (default /leonardo_work/$EULLM_ACCOUNT/eullm_runs/
# fastlane-probe), FLP_HOLD (seconds the 16 + 16 pair stays up, default 900),
# FLP_FINAL_HOURS (default 24), FLP_FORCE=1 to run again while probes of an
# earlier run are still queued.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PROBE="$HERE/sbatch_fastlane_probe.slurm"
REPORT="$HERE/sbatch_fastlane_probe_report.slurm"

: "${EULLM_ACCOUNT:?set EULLM_ACCOUNT to the new project account (saldo -b)}"
HOLD="${FLP_HOLD:-900}"
FINAL_HOURS="${FLP_FINAL_HOURS:-24}"
BASE="${FLP_OUT:-/leonardo_work/$EULLM_ACCOUNT/eullm_runs/fastlane-probe}"

for f in "$PROBE" "$REPORT" "$HERE/fastlane_probe_report.py" "$HERE/priority_snapshot.py"; do
    [ -f "$f" ] || { echo "[flp] FATAL: $f is missing" >&2; exit 1; }
done

if ! sacctmgr -n -P show assoc user="$USER" format=account | grep -ix "$EULLM_ACCOUNT" >/dev/null; then
    echo "[flp] FATAL: $USER has no association with account '$EULLM_ACCOUNT'." >&2
    echo "[flp] Accounts it has: $(sacctmgr -n -P show assoc user="$USER" format=account | sort -u | xargs)" >&2
    exit 1
fi

if squeue --me -h -o %j | grep '^eullm-flp' >/dev/null && [ "${FLP_FORCE:-0}" != 1 ]; then
    echo "[flp] probes of an earlier run are still queued (squeue --me -n ...)." >&2
    echo "[flp] Read that run's report.md, or FLP_FORCE=1 to submit a second set." >&2
    exit 1
fi

DIR="$BASE/$(date +%Y%m%d_%H%M)"
mkdir -p "$DIR"
echo "[flp] probe directory $DIR"

# --- 1. What the project may ask for ----------------------------------------
{
    echo "== $(date -Is)  account $EULLM_ACCOUNT  user $USER"
    echo; echo "== saldo -b"; saldo -b 2>&1 || true
    echo; echo "== associations (account, partition, QOS, limits)"
    sacctmgr -P show assoc user="$USER" \
        format=account,partition,qos,defaultqos,grptres,maxtres,maxtrespu,maxjobs,maxsubmit,maxwall 2>&1 || true
    echo; echo "== QOS (priority, wall time, limits)"
    sacctmgr -P show qos \
        format=name,priority,maxwall,grptres,maxtres,maxtrespu,maxjobspu,maxsubmitpu,mintres 2>&1 || true
    echo; echo "== partition boost_usr_prod"
    scontrol show partition boost_usr_prod 2>&1 || true
    echo; echo "== nodes of boost_usr_prod by state"
    sinfo -p boost_usr_prod -h -o "%T %D" 2>&1 || true
} > "$DIR/limits.txt"
echo "[flp] limits written: $DIR/limits.txt"

# QOS the project may use besides normal and the debug one (30 minutes, not
# a production shape).
mapfile -t EXTRA_QOS < <(
    sacctmgr -n -P show assoc user="$USER" format=account,qos |
    awk -F'|' -v a="$EULLM_ACCOUNT" 'tolower($1) == tolower(a) {print $2}' |
    tr ',' '\n' | grep -v -e '^$' -e '^normal$' -e 'dbg' | sort -u)
echo "[flp] other QOS of the project: ${EXTRA_QOS[*]:-none}"

# --- 2 and 3. The probes ----------------------------------------------------
printf 'id\tname\tnodes\tgpus\ttime\tqos\thold\tsubmitted\tnote\n' > "$DIR/manifest.tsv"

submit() {  # name nodes gpus time qos hold
    local name="$1" nodes="$2" gpus="$3" time="$4" qos="$5" hold="$6"
    local cpus=$(( gpus * 8 )) mem="$(( gpus * 113 ))G" out id note=""
    [ "$gpus" -eq 4 ] && mem=450G
    local args=(--parsable --account="$EULLM_ACCOUNT" --job-name="eullm-flp-$name"
                --nodes="$nodes" --ntasks-per-node=1 --cpus-per-task="$cpus"
                --gres="gpu:$gpus" --mem="$mem" --time="$time"
                --output="$DIR/%x-%j.out" --export="ALL,PROBE_HOLD=$hold")
    [ "$qos" != normal ] && args+=(--qos="$qos")
    if out="$(sbatch "${args[@]}" "$PROBE" 2>&1)"; then
        id="${out%%;*}"
    else
        id=""; note="$(echo "$out" | tr '\t\n' '  ')"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "$name" "$nodes" "$gpus" "$time" \
        "$qos" "$hold" "$(date +%Y-%m-%dT%H:%M:%S)" "$note" >> "$DIR/manifest.tsv"
    echo "[flp] ${id:-REJECTED} $name ($nodes x $gpus GPU, $time, $qos)${note:+: $note}"
}

#       name     nodes gpus time      qos    hold
submit g1-2h     1     1    02:00:00  normal 0
submit g3-2h     1     3    02:00:00  normal 0
submit g3-4h     1     3    04:00:00  normal 0
submit g3-8h     1     3    08:00:00  normal 0
submit g3-24h    1     3    24:00:00  normal 0
submit n1-2h     1     4    02:00:00  normal 0
submit n1-8h     1     4    08:00:00  normal 0
submit n1-24h    1     4    24:00:00  normal 0
submit n2-8h     2     4    08:00:00  normal 0
submit n4-2h     4     4    02:00:00  normal 0
submit n4-8h     4     4    08:00:00  normal 0
submit n4-24h    4     4    24:00:00  normal 0
submit n8-2h     8     4    02:00:00  normal 0
submit par-a     4     4    00:30:00  normal "$HOLD"
submit par-b     4     4    00:30:00  normal "$HOLD"
for q in "${EXTRA_QOS[@]}"; do
    submit "$q-n1-8h"  1 4 08:00:00 "$q" 0
    submit "$q-n4-8h"  4 4 08:00:00 "$q" 0
    submit "$q-g3-24h" 1 3 24:00:00 "$q" 0
done

# --- The reports, written by the scheduler itself ---------------------------
for h in 1 6 "$FINAL_HOURS"; do
    final=0; [ "$h" = "$FINAL_HOURS" ] && final=1
    sbatch --parsable --account="$EULLM_ACCOUNT" --begin="now+${h}hours" \
        --output="$DIR/report-${h}h-%j.out" \
        --export="ALL,FLP_DIR=$DIR,FLP_REPO=$REPO,FLP_FINAL=$final" "$REPORT" >/dev/null &&
        echo "[flp] report at +${h} h queued$([ $final = 1 ] && echo ' (final: cancels what is still waiting)')"
done

echo
echo "[flp] Done. Now: cat $DIR/limits.txt"
echo "[flp] From one hour on: cat $DIR/report.md"

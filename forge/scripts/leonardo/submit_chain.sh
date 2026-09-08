#!/usr/bin/env bash
# Submit N chained copies of an sbatch script (afterany dependencies).
#
# Leonardo's boost_usr_prod partition caps jobs at 24 h, so a multi-day
# phase is a chain: when job k dies (walltime or otherwise), job k+1
# starts and the launchers (train.sh / distill.sh) resume from the
# latest checkpoint. A count of 1 is a plain submission with the logs
# dir prepared.
#
# Usage:
#   bash submit_chain.sh <script.slurm> [count] [extra sbatch args...]
#
# Examples:
#   bash submit_chain.sh sbatch_smoke.slurm
#   bash submit_chain.sh sbatch_phase2.slurm 7
#
# Run from $EULLM_RUN_DIR (the scripts' #SBATCH --output is relative to
# the submission directory). The account is picked up from
# SBATCH_ACCOUNT — export EULLM_ACCOUNT and source env.sh first.

set -euo pipefail

SCRIPT="${1:?Usage: $0 <script.slurm> [count] [extra sbatch args...]}"
COUNT="${2:-1}"
shift
if [ $# -gt 0 ]; then shift; fi

[ -f "$SCRIPT" ] || { echo "[err] sbatch script not found: $SCRIPT" >&2; exit 1; }
case "$COUNT" in
    ''|*[!0-9]*) echo "[err] count must be a positive integer, got '$COUNT'" >&2; exit 1;;
esac

if [ -z "${SBATCH_ACCOUNT:-}" ]; then
    echo "[warn] SBATCH_ACCOUNT not set — export EULLM_ACCOUNT and source" >&2
    echo "[warn] forge/scripts/leonardo/env.sh, or pass -A <account>" >&2
fi

mkdir -p logs

prev=""
for i in $(seq 1 "$COUNT"); do
    if [ -z "$prev" ]; then
        jid=$(sbatch --parsable "$@" "$SCRIPT")
    else
        jid=$(sbatch --parsable --dependency=afterany:"$prev" "$@" "$SCRIPT")
    fi
    jid="${jid%%;*}"   # --parsable may append ';cluster'
    echo "[ok] submitted $jid ($i/$COUNT)${prev:+ — after $prev}"
    prev="$jid"
done

echo "[ok] monitor with: squeue --me   |   logs in $(pwd)/logs/"

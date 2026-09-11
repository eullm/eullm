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
#   bash submit_chain.sh [--after <jobid>] <script.slurm> [count] [extra sbatch args...]
#
# Examples:
#   bash submit_chain.sh sbatch_smoke.slurm
#   bash submit_chain.sh sbatch_phase2.slurm 7
#   bash submit_chain.sh --after 56912622 sbatch_phase2.slurm 7
#
# `--after <jobid>` holds the FIRST link until that job finishes cleanly
# (afterok), so a phase can be queued days ahead of the one it depends on and
# the machine never sits idle between them. It exists because passing
# --dependency through the extra args does NOT work and fails dangerously:
# those args are appended after the chain's own --dependency, sbatch takes the
# last occurrence, and every link would then wait on the same external job
# instead of on its predecessor. Seven Phase-2 jobs would start at once, all
# resuming from the same checkpoint. Extra args containing --dependency are
# therefore rejected outright.
#
# Run from $EULLM_RUN_DIR (the scripts' #SBATCH --output is relative to
# the submission directory). The account is picked up from
# SBATCH_ACCOUNT — export EULLM_ACCOUNT and source env.sh first.

set -euo pipefail

AFTER=""
if [ "${1:-}" = "--after" ]; then
    AFTER="${2:?--after needs a job id}"
    shift 2
fi

SCRIPT="${1:?Usage: $0 [--after <jobid>] <script.slurm> [count] [extra sbatch args...]}"
COUNT="${2:-1}"
shift
if [ $# -gt 0 ]; then shift; fi

[ -f "$SCRIPT" ] || { echo "[err] sbatch script not found: $SCRIPT" >&2; exit 1; }

# See the header: a --dependency in the extra args overrides the chain's own
# and every link would wait on the same job rather than on its predecessor.
for arg in "$@"; do
    case "$arg" in
        -d|-d=*|--dependency|--dependency=*)
            echo "[err] do not pass $arg — it would override each link's own" >&2
            echo "[err] dependency and start the whole chain at once." >&2
            echo "[err] Use: $0 --after <jobid> $SCRIPT $COUNT" >&2
            exit 1
            ;;
    esac
done
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
    dep=()
    if [ -z "$prev" ]; then
        # afterok, not afterany: a first link that starts on a FAILED
        # predecessor would resume from a checkpoint that phase never wrote.
        [ -n "$AFTER" ] && dep=(--dependency=afterok:"$AFTER")
    else
        # afterany within the chain: TIMEOUT is how a 24 h link is meant to
        # end, and afterok would stop the chain on every one of them.
        dep=(--dependency=afterany:"$prev")
    fi
    # ${dep[@]+...} rather than a bare "${dep[@]}": under `set -u` an empty
    # array counts as unbound on bash before 4.4, and this should not depend
    # on which node's shell it happens to run under.
    jid=$(sbatch --parsable ${dep[@]+"${dep[@]}"} "$@" "$SCRIPT")
    jid="${jid%%;*}"   # --parsable may append ';cluster'
    echo "[ok] submitted $jid ($i/$COUNT)${prev:+ — after $prev}${prev:+}"
    [ -z "$prev" ] && [ -n "$AFTER" ] && echo "[ok]   held until $AFTER completes"
    prev="$jid"
done

echo "[ok] monitor with: squeue --me   |   logs in $(pwd)/logs/"

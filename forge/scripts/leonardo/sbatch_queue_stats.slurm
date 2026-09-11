#!/bin/bash
# A daily snapshot of where the allocation's calendar is going, that keeps
# itself alive.
#
# Leonardo has no user cron (`crontab` is not permitted) and `scrontab` is
# disabled cluster-wide, so there is no scheduler to hang this off. What does
# work is a job that re-submits itself with `--begin=now+1day` before
# exiting: the successor sits PENDING for a day, which costs nothing, then
# does the same. It runs until the allocation ends and then stops on its own.
#
# It lives on the serial partition and takes about three seconds of one core.
# Putting it on boost_usr_prod would reserve four A100s to run some Python.
#
#   sbatch forge/scripts/leonardo/sbatch_queue_stats.slurm
#   squeue --me -n eullm-queue-stats     # must always show exactly one PENDING
#
# The chain's one failure mode is a re-submission that does not happen — a
# full queue, an expired allocation, a typo in a path. Nothing announces it;
# the job simply stops existing. Check for that one PENDING row now and then,
# and if it is gone, submit this again.
#
#SBATCH --job-name=eullm-queue-stats
#SBATCH --partition=lrd_all_serial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=%x-%j.out

set -uo pipefail

# Absolute, because a batch job inherits none of the login shell's exports and
# $WORK does not exist here. Override when the project path differs.
BASE="${EULLM_BASE:-/leonardo_work/AIFAC_P02_1147}"
REPO="${EULLM_REPO:-$BASE/eullm-v11}"
OUT="$BASE/eullm_runs/qstats"
ALLOC_START="${EULLM_ALLOC_START:-2026-09-02}"
ALLOC_END="${EULLM_ALLOC_END:-20261102}"

mkdir -p "$OUT"

# `|| true` on purpose: an instrument must never fail the thing it measures,
# and here that thing includes its own re-submission below.
python3 "$REPO/forge/scripts/leonardo/queue_stats.py" "$ALLOC_START" \
    --json "$OUT/queue_stats_$(date +%Y%m%d_%H%M).json" || true

# Re-arm, unless the allocation is over. Without this guard the chain would
# keep trying past the end date and fill the directory with failures.
if [ "$(date +%Y%m%d)" -lt "$ALLOC_END" ]; then
    cd "$OUT" || exit 0
    sbatch --begin=now+1day "$REPO/forge/scripts/leonardo/sbatch_queue_stats.slurm"
fi

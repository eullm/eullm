#!/bin/bash
# One probe of the Fast Lane's first day: does a job of THIS shape start, and
# how soon? Submitted many times by fastlane_probe.sh, each time with a
# different shape given on the sbatch command line (which overrides the
# header below). Run alone, it is the smallest shape there is.
#
# It does nothing but say where it landed and leave, so a probe that starts
# costs seconds of its nodes. The exception is the concurrency pair
# (PROBE_HOLD > 0): two jobs of 16 GPUs each that stay up a few minutes, so the
# report can see whether both ran at once. That costs about five GPU-hours and
# answers whether 16 + 16 is allowed.
#
#SBATCH --job-name=eullm-flp
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

set -uo pipefail

echo "[probe] job      ${SLURM_JOB_ID:-?} (${SLURM_JOB_NAME:-?})"
echo "[probe] qos      ${SLURM_JOB_QOS:-?}"
echo "[probe] nodes    ${SLURM_JOB_NUM_NODES:-?}: ${SLURM_JOB_NODELIST:-?}"
echo "[probe] gpus     ${SLURM_GPUS_ON_NODE:-?} per node"
echo "[probe] started  $(date -Is)"

HOLD="${PROBE_HOLD:-0}"
if [ "$HOLD" -gt 0 ] 2>/dev/null; then
    echo "[probe] holding  ${HOLD}s, so a sibling can be seen running beside it"
    sleep "$HOLD"
fi

echo "[probe] done     $(date -Is)"

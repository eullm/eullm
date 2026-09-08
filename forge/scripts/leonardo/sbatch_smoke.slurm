#!/bin/bash
# Smoke test on Leonardo Booster: 1 GPU (= 1/4 node of budget), debug
# QOS for fast scheduling. Validates the whole offline wiring — venv,
# HF cache, dataset, LLaMA-Factory, checkpoint write — before any
# node-hours go to the production phases.
#
# Submit from $EULLM_RUN_DIR (submit_chain.sh does the right thing):
#   bash "$EULLM_REPO/forge/scripts/leonardo/submit_chain.sh" \
#       "$EULLM_REPO/forge/scripts/leonardo/sbatch_smoke.slurm"
#
# The account comes from SBATCH_ACCOUNT (exported by env.sh from
# EULLM_ACCOUNT), so no -A flag is needed.
#
#SBATCH --job-name=eullm-smoke
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=110G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

EULLM_REPO="${EULLM_REPO:-$WORK/eullm}"
# shellcheck disable=SC1091
source "$EULLM_REPO/forge/scripts/leonardo/env.sh"
cd "$EULLM_RUN_DIR"

python "$EULLM_REPO/forge/scripts/check_training_env.py" \
    --offline --data-dir "$EULLM_DATA_DIR"

bash "$EULLM_REPO/forge/scripts/train.sh" \
    "$EULLM_REPO/forge/training/configs/smoke_qwen3_1.7b.yaml" \
    "$EULLM_DATA_DIR"

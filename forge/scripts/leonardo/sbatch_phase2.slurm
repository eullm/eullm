#!/bin/bash
# Phase 2 — distillation Qwen3-32B-legal-it → Qwen3-7B on one Leonardo
# Booster node. Single process: the frozen BF16 teacher is sharded
# across the 4 GPUs (accelerate device_map), the student trains on
# cuda:0. See the Leonardo distill YAML header for the memory budget.
#
# Expect 5-7 days of wall time — submit a chain of 24 h jobs; each one
# resumes from the latest checkpoint:
#   bash "$EULLM_REPO/forge/scripts/leonardo/submit_chain.sh" \
#       "$EULLM_REPO/forge/scripts/leonardo/sbatch_phase2.slurm" 7
#
#SBATCH --job-name=eullm-p2-distill
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=450G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

EULLM_REPO="${EULLM_REPO:-$WORK/eullm}"
# shellcheck disable=SC1091
source "$EULLM_REPO/forge/scripts/leonardo/env.sh"
cd "$EULLM_RUN_DIR"

# Phase 1 must have produced the teacher adapter this YAML points at.
ADAPTER="./checkpoints/qwen3_32b_legal_it_continued_pt"
[ -f "$ADAPTER/adapter_config.json" ] || {
    echo "[err] Phase-1 adapter not found at $EULLM_RUN_DIR/$ADAPTER" >&2
    echo "[err] run sbatch_phase1.slurm to completion first" >&2
    exit 1
}

python "$EULLM_REPO/forge/scripts/check_training_env.py" \
    --offline --expect-gpus 4 --data-dir "$EULLM_DATA_DIR"

bash "$EULLM_REPO/forge/scripts/distill.sh" \
    "$EULLM_REPO/forge/training/configs/leonardo/distill_qwen3_30b_a3b_to_4b_leonardo.yaml" \
    "$EULLM_DATA_DIR"

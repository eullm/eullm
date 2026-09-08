#!/bin/bash
# Phase 1 — continued pre-training of the Qwen3-30B-A3B teacher on one
# Leonardo Booster node (4x A100 64 GB), DeepSpeed ZeRO-3 via
# LLaMA-Factory. See the Leonardo YAML header for the memory budget.
#
# The 24 h walltime cap makes preemption certain for a full epoch —
# submit a chain; every job resumes from the latest checkpoint:
#   bash "$EULLM_REPO/forge/scripts/leonardo/submit_chain.sh" \
#       "$EULLM_REPO/forge/scripts/leonardo/sbatch_phase1.slurm" 2
#
#SBATCH --job-name=eullm-p1-cpt30ba3b
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

python "$EULLM_REPO/forge/scripts/check_training_env.py" \
    --offline --multi-gpu --expect-gpus 4 --data-dir "$EULLM_DATA_DIR" \
    --tokenizer-model Qwen/Qwen3-30B-A3B-Base

# Report on ourselves: LLaMA-Factory and DeepSpeed are silent through
# checkpoint load and ZeRO-3 init, which is where the long waits are.
# shellcheck disable=SC1091
source "$EULLM_REPO/forge/scripts/leonardo/heartbeat.sh"
heartbeat_start

# LLaMA-Factory launches through torchrun with one rank per GPU.
export FORCE_TORCHRUN=1
export NPROC_PER_NODE=4
export OMP_NUM_THREADS=8

bash "$EULLM_REPO/forge/scripts/train.sh" \
    "$EULLM_REPO/forge/training/configs/leonardo/continued_pt_qwen3_30b_a3b_leonardo.yaml" \
    "$EULLM_DATA_DIR"

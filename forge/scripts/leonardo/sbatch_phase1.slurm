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

# Report on ourselves BEFORE the pre-flight, not after. The pre-flight
# imports transformers, which on a cold node cache reads thousands of small
# files off Lustre and takes minutes in complete silence — so starting the
# heartbeat after it left the job mute through the very first phase someone
# would want to watch. Ten minutes of apparent nothing on job 56775410.
# shellcheck disable=SC1091
source "$EULLM_REPO/forge/scripts/leonardo/heartbeat.sh"
heartbeat_start

python "$EULLM_REPO/forge/scripts/check_training_env.py" \
    --offline --multi-gpu --expect-gpus 4 --data-dir "$EULLM_DATA_DIR" \
    --tokenizer-model Qwen/Qwen3-30B-A3B-Base

# LLaMA-Factory launches through torchrun with one rank per GPU. No
# DeepSpeed config is passed, so this is plain DDP and every rank holds the
# whole quantized base — see the training YAML header for why.
export FORCE_TORCHRUN=1
export NPROC_PER_NODE=4
export OMP_NUM_THREADS=8

# Let the caching allocator grow segments instead of pre-carving fixed ones.
# Three OOMs on this model asked for a few hundred MiB while the allocator
# held gigabytes it could not hand back because of fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Python auto-imports `sitecustomize` from PYTHONPATH, which is the only
# hook into LLaMA-Factory's worker processes we get without forking it. The
# probe prints peak allocated/reserved per GPU at exit and the allocator
# summary on a CUDA OOM — the breakdown the exception itself never carries.
export PYTHONPATH="$EULLM_REPO/forge/scripts/leonardo/memprobe${PYTHONPATH:+:$PYTHONPATH}"

# Run identity, in the log rather than in someone's memory. Comparing two
# runs a week apart is impossible without knowing which commit and which
# quantization produced each.
CFG="$EULLM_REPO/forge/training/configs/leonardo/continued_pt_qwen3_30b_a3b_leonardo.yaml"
echo "=================== RUN ==================="
echo "  job          : ${SLURM_JOB_ID:-?} on ${SLURMD_NODENAME:-?}"
echo "  commit       : $(git -C "$EULLM_REPO" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "  config       : $(basename "$CFG")"
for k in quantization_bit cutoff_len per_device_train_batch_size \
         gradient_accumulation_steps lora_rank lora_target num_train_epochs; do
    printf '  %-13s: %s\n' "$k" "$(grep -E "^${k}:" "$CFG" | head -1 | cut -d: -f2- | xargs)"
done
echo "  deepspeed    : $(grep -qE '^deepspeed:' "$CFG" && echo yes || echo 'no (plain DDP)')"
echo "==========================================="

bash "$EULLM_REPO/forge/scripts/train.sh" \
    "$EULLM_REPO/forge/training/configs/leonardo/continued_pt_qwen3_30b_a3b_leonardo.yaml" \
    "$EULLM_DATA_DIR"

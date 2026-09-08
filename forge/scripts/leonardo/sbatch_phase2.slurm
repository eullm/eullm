#!/bin/bash
# Phase 2 — distillation Qwen3-30B-A3B-legal-it → Qwen3-4B on one Leonardo
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
ADAPTER="./checkpoints/qwen3_30b_a3b_legal_it_continued_pt"
[ -f "$ADAPTER/adapter_config.json" ] || {
    echo "[err] Phase-1 adapter not found at $EULLM_RUN_DIR/$ADAPTER" >&2
    echo "[err] run sbatch_phase1.slurm to completion first" >&2
    exit 1
}

CFG="$EULLM_REPO/forge/training/configs/leonardo/distill_qwen3_30b_a3b_to_4b_leonardo.yaml"

# Report on ourselves BEFORE the pre-flight, not after. The pre-flight
# imports transformers, which on a cold node cache reads thousands of small
# files off Lustre and takes minutes in complete silence — so starting the
# heartbeat after it left the job mute through the very first phase someone
# would want to watch. Ten minutes of apparent nothing on job 56775410.
# shellcheck disable=SC1091
source "$EULLM_REPO/forge/scripts/leonardo/heartbeat.sh"
heartbeat_start

python "$EULLM_REPO/forge/scripts/check_training_env.py" \
    --offline --expect-gpus 4 --data-dir "$EULLM_DATA_DIR" \
    --tokenizer-model Qwen/Qwen3-4B-Base \
    --assert-frozen-base "$CFG"

# Gate: the teacher is quantized, and in distillation its distribution IS
# the target, so a quantization error is trained into the student rather
# than merely degrading the run. Measure it before spending the budget.
#
# Cached deliberately. The check loads both teachers at once and takes
# tens of minutes; running it again in every job of an afterany chain would
# re-pay that for a result that cannot have changed. The report records
# which bit width was validated, so switching 8-bit to 4-bit invalidates it.
TEACHER_BITS="$(grep -qE '^teacher_load_in_8bit:[[:space:]]*true' "$CFG" && echo 8 \
    || { grep -qE '^teacher_load_in_4bit:[[:space:]]*true' "$CFG" && echo 4 || echo 0; })"
REPORT="$EULLM_RUN_DIR/quant_validation_${TEACHER_BITS}bit.json"

if [ "$TEACHER_BITS" = "0" ]; then
    echo "[..] teacher is not quantized — no logit validation needed"
elif [ -f "$REPORT" ]; then
    echo "[ok] teacher already validated at ${TEACHER_BITS}-bit ($REPORT)"
else
    echo "[..] validating the ${TEACHER_BITS}-bit teacher against bf16"
    python "$EULLM_REPO/forge/scripts/validate_quantized_teacher.py" \
        --model "$(grep -E '^teacher_model:' "$CFG" | cut -d: -f2- | xargs)" \
        --bits "$TEACHER_BITS" \
        --adapter "$(grep -E '^teacher_adapter:' "$CFG" | cut -d: -f2- | xargs)" \
        --data "$EULLM_DATA_DIR/val.jsonl" \
        --report "$REPORT"
fi

bash "$EULLM_REPO/forge/scripts/distill.sh" "$CFG" "$EULLM_DATA_DIR"

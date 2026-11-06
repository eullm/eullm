#!/bin/bash
# Line A2-mini of the Fast Lane (docs/cineca/fastlane-plan.md): a hashed
# n-gram memory trained on a frozen model (ngram_train.py), one GPU,
# two-hour links of a chain.
#
#   bash forge/scripts/leonardo/submit_chain.sh \
#       forge/scripts/leonardo/sbatch_ngram_train.slurm 12 -J eullm-a2-ngram
#
# The LoRA control, same tokens in the same order, as a second chain:
#   NG_MODE=lora bash forge/scripts/leonardo/submit_chain.sh \
#       forge/scripts/leonardo/sbatch_ngram_train.slurm 12 -J eullm-a2-lora
#
# Every link resumes from the last save with the data where it stopped, and
# saves and exits after 110 minutes; a link that finds the run finished
# (<out>/DONE) exits at once, so spare links cost seconds. The answers then
# come from sbatch_norm_exam.slurm with EXAM_MEMORY / EXAM_ADAPTER.
#
# One GPU, 8 cores, 120 GB: on 4 Nov 2026 only one-GPU jobs were being
# placed (forge/CLAUDE.md, the shape of a job that starts). An 8B in bf16
# with a 0.4B memory and its optimiser fits in one A100 64GB.
#
# Settable:
#   NG_MODEL  the frozen model (default r2l merged, on the fast scratch)
#   NG_DATA   memory corpus files (default $WORK/eullm_runs/memory/program.jsonl
#             and rewrites*.jsonl there)
#   NG_MODE   ngram (default) or lora
#   NG_OUT    run directory (default $WORK/eullm_runs/lineA/a2-r2l-$NG_MODE)
#   NG_ARGS   extra ngram_train.py options (default: 3 epochs, a 0.41B table:
#             16 heads x 400,000 rows x 64 dims; --embed-dim is the 16 heads
#             together, as ple_embed_dim in Qwen3.8-Flash-Next)
#   NG_REPO   checkout to run (default: the one this file was submitted from)
#
#SBATCH --job-name=eullm-a2-ngram
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail

SELF="$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null |
        sed -n 's/.*Command=\([^ ]*\).*/\1/p' | head -1)"
if [ -n "${NG_REPO:-}" ]; then
    REPO="$NG_REPO"
elif [ -n "$SELF" ] && [ -f "$SELF" ]; then
    REPO="$(cd "$(dirname "$SELF")/../../.." && pwd)"
else
    REPO="${EULLM_REPO:-$WORK/eullm}"
fi
# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/env.sh"

MODE="${NG_MODE:-ngram}"
MODEL="${NG_MODEL:-/leonardo_scratch/fast/$(basename "$WORK")/grpo-merged/r2l-steps-8b-quote}"
OUT="${NG_OUT:-$WORK/eullm_runs/lineA/a2-r2l-$MODE}"
MEMDIR="$WORK/eullm_runs/memory"
# shellcheck disable=SC2206
DATA=(${NG_DATA:-$MEMDIR/program.jsonl $MEMDIR/rewrites*.jsonl})
ARGS="${NG_ARGS:---epochs 3 --rows-per-head 400000 --embed-dim 1024}"

echo "[ng] repo  $REPO ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?'))"
echo "[ng] mode  $MODE, model $MODEL"
echo "[ng] out   $OUT"
if [ -f "$OUT/DONE" ]; then
    echo "[ng] finished already: $(cat "$OUT/DONE")"
    exit 0
fi
if [ -f "$OUT/DIVERGED" ]; then  # the rest of the chain costs seconds, not a link each
    echo "[ng] diverged: $(cat "$OUT/DIVERGED"); start again with a lower --lr in a new NG_OUT"
    exit 0
fi
fail=0
case "$MODE" in ngram|lora) ;; *) echo "[ng] NG_MODE is ngram or lora, not $MODE" >&2; fail=1 ;; esac
[ -f "$MODEL/config.json" ] || { echo "[ng] no model at $MODEL" >&2; fail=1; }
for f in "${DATA[@]}"; do [ -s "$f" ] || { echo "[ng] missing $f" >&2; fail=1; }; done
[ "$fail" -eq 0 ] || exit 1
echo "[ng] data  ${DATA[*]}"

# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/heartbeat.sh"
heartbeat_start

# 6600 s from the start: the save and the exit fit in the two hours.
# shellcheck disable=SC2086
python "$REPO/forge/scripts/ngram_train.py" --model "$MODEL" --data "${DATA[@]}" \
    --out "$OUT" --mode "$MODE" --stop-after 6600 $ARGS
status=$?
[ -f "$OUT/train-log.jsonl" ] && echo "[ng] last log: $(tail -1 "$OUT/train-log.jsonl")"
exit $status

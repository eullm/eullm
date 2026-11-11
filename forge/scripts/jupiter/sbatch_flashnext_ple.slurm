#!/bin/bash
# What Qwen3.8-Flash-Next's n-gram tables know of Italian law: the closed-book
# suite (716 questions, no text) asked twice in one job, the native model and
# the same model with its PLE layers switched off (legal_eval.py --ple-off),
# then scored side by side by program. On Leonardo it is the first in the
# columns of knowledge (number from heading, heading from number, terms,
# penalties: docs/legal-it-experiments.md, section 14), and the only model
# measured with hashed n-gram tables (51 B parameters, docs/research/
# flash-next-ngram.md); whether the tables are why is what this answers.
#
#     sbatch --account e-aif-2026pg01-1434 \
#         --output=$PROJECT/$USER/forge/runs/logs/flashnext-ple-%j.out \
#         forge/scripts/jupiter/sbatch_flashnext_ple.slurm
#
# One node: the model in BF16 over the four GH200 (eval/placement.py), the
# table in the Grace memory. Answers and the score go to
# $FORGE_RUN_DIR/flashnext-ple/, readable by the user alone; the log holds
# counts only. A pass already written whole is not asked again.
#
# Settable: FN_MODEL (default Qwen/Qwen3.8-Flash-Next, offline cache),
# FN_ITEMS (default $FORGE_DATA_DIR/eval/leonardo/closed-book.jsonl),
# FN_MAX_NEW (default 1000, as the Leonardo GGUF run quoted), FN_BATCH (16).

#SBATCH --job-name=forge-flashnext-ple
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00

set -uo pipefail
EULLM_REPO="${EULLM_REPO:-$SLURM_SUBMIT_DIR}"
# flash-linear-attention needs the Triton of libs-qwen35 on Hopper (job 2686808)
export FORGE_EXTRA_LIBS="${FORGE_EXTRA_LIBS:-$PROJECT/$USER/forge/libs-qwen35}"
# shellcheck source=env.sh
source "$EULLM_REPO/forge/scripts/jupiter/env.sh"
MODEL="${FN_MODEL:-Qwen/Qwen3.8-Flash-Next}"
ITEMS="${FN_ITEMS:-$FORGE_DATA_DIR/eval/leonardo/closed-book.jsonl}"
OUT="$FORGE_RUN_DIR/flashnext-ple"
echo "=== Flash-Next n-gram on/off: job ${SLURM_JOB_ID:-local} on $(hostname), $(date -Is) ==="
python -c "import sys; from transformers import AutoConfig; AutoConfig.from_pretrained(sys.argv[1], local_files_only=True)" "$MODEL" ||
    { echo "[!!] $MODEL is not in the offline cache: snapshot_download on a login node"; exit 1; }
[ -s "$ITEMS" ] || { echo "[!!] missing $ITEMS"; exit 1; }
mkdir -p "$OUT" && chmod 700 "$OUT"
WANT=$(grep -c . "$ITEMS")
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RC=0
for pass in on off; do
    label="cb-flashnext-bf16$([ $pass = off ] && echo -pleoff)"
    answers="$OUT/answers-$label.jsonl"
    if [ -s "$answers" ] && [ "$(grep -c . "$answers")" -eq "$WANT" ]; then
        echo "  $label: answered already"; continue
    fi
    extra=(); [ $pass = off ] && extra=(--ple-off)
    t0=$(date +%s)
    python "$EULLM_REPO/forge/scripts/legal_eval.py" "$MODEL" --label "$label" --items "$ITEMS" \
        --max-new-tokens "${FN_MAX_NEW:-1000}" --batch-size "${FN_BATCH:-16}" \
        --answers "$answers" --quiet "${extra[@]}" || RC=1
    echo "  $label: exit $RC, $(( $(date +%s) - t0 )) s"
done
echo "--- grades, by kind"
python "$EULLM_REPO/forge/scripts/score_closed_book.py" "$ITEMS" "$OUT"/answers-cb-flashnext-bf16*.jsonl \
    | tee "$OUT/score.txt"
echo "--- answers pointing to texts not given, by kind"
python "$EULLM_REPO/forge/scripts/score_closed_book.py" "$ITEMS" --texts "$OUT"/answers-cb-flashnext-bf16*.jsonl \
    | tee -a "$OUT/score.txt"
echo "=== done $(date -Is) (exit $RC) ==="
exit "$RC"

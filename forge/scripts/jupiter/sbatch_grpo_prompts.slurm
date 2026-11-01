#!/bin/bash
# New GRPO prompts made on JUPITER, for rounds whose models have learnt what
# Leonardo's prompts teach (the third rounds gained nothing, 29/31-10-2026):
#
#   1. practical questions written by a teacher from the law
#      (make_practical_prompts.py, Leonardo's roadmap 1.2): Qwen3.6-27B, not
#      the judge's model, one llama-server on each of GPUs 1-3; the practical
#      prompts gave the largest gains of every round so far;
#   2. by-topic deadline questions asked twice, with their article among the
#      texts and without it (make_grpo_prompts.py --absent-share 1.0): the
#      wrong-article answers are the commonest mistake left.
#
# Both retrieve as the exams ask (BM25 + Qwen3 embedder + reranker, on
# GPU 0), and leave out every article of the held-out sets; the private
# ones are left out again when a round filters its prompts
# (filter_grpo_prompts.py). Each file is written once: a second submission
# keeps it. Counts only, never a question.
#
#     eullm && sbatch --account e-aif-2026pg01-1434 \
#         --output="$PROJECT/$USER/forge/runs/logs/grpo-prompts-%j.out" \
#         forge/scripts/jupiter/sbatch_grpo_prompts.slurm
#
# Settable: GP_PER_CODE (practical groups a code, default 400), GP_SEED
# (default 7; Leonardo's were 1), GP_TEACHER (a model of the engine's store,
# default qwen3.6-27b), GP_OUT_DIR (default $FORGE_DATA_DIR/grpo/jupiter).

#SBATCH --job-name=forge-grpo-prompts
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00

set -uo pipefail
EULLM_REPO="${EULLM_REPO:-$SLURM_SUBMIT_DIR}"
# shellcheck source=env.sh
source "$EULLM_REPO/forge/scripts/jupiter/env.sh"
export EULLM_MODELS_DIR="${EULLM_MODELS_DIR:-$PROJECT/$USER/eullm-models}"
D="$FORGE_DATA_DIR" P="${FORGE_PRIVATE_EVAL:-$FORGE_HOME/private-eval}"
OUT="${GP_OUT_DIR:-$D/grpo/jupiter}"
SEED="${GP_SEED:-7}"
SERVER="${LLAMA_SERVER_BIN:-$PROJECT/$USER/campaign/runtimes/llama-server}"
TEACHER=$(python -c "import sys; sys.path.insert(0, sys.argv[1]); from point import store_gguf; print(store_gguf(sys.argv[2]) or '')" \
          "$EULLM_REPO/bench/campaign" "${GP_TEACHER:-qwen3.6-27b}")
read -r -a NORMS <<<"$(ls "$D"/norms-src/legislazione_*.chunks.jsonl | grep -v eurlex | tr '\n' ' ')"
EXAMS=("$D/eval/trap-test.jsonl" "$P/practical-dev.jsonl")
for f in "$D"/eval/norm-exam*.jsonl; do EXAMS+=("$f"); done
RETR=(--embedder Qwen/Qwen3-Embedding-0.6B --reranker Qwen/Qwen3-Reranker-0.6B
      --retrieval-cache "$FORGE_RUN_DIR/retrieval-cache")
JOB="${SLURM_JOB_ID:-local}"
mkdir -p "$OUT" "$FORGE_RUN_DIR/power" "$FORGE_RUN_DIR/retrieval-cache"

echo "=== GRPO prompts on JUPITER: job $JOB on $(hostname), $(date -Is) ==="
fail=0
[ -s "$TEACHER" ] || { echo "[!!] no ${GP_TEACHER:-qwen3.6-27b} in $EULLM_MODELS_DIR"; fail=1; }
[ -x "$SERVER" ] || { echo "[!!] no llama-server at $SERVER"; fail=1; }
for f in "${NORMS[@]}" "${EXAMS[@]}"; do [ -s "$f" ] || { echo "[!!] missing $f"; fail=1; }; done
[ "$fail" = 0 ] || exit 1
echo "  teacher $TEACHER on GPUs 1-3, retrieval on GPU 0, seed $SEED -> $OUT"

POWER="$FORGE_RUN_DIR/power/grpo-prompts-$JOB.csv"
nvidia-smi --query-gpu=timestamp,index,power.draw,utilization.gpu,memory.used \
    --format=csv,noheader,nounits -l 5 >"$POWER" 2>/dev/null &
SAMPLER=$!
RC=0

ABSENT="$OUT/prompts-absent-s$SEED.jsonl"
if [ -s "$ABSENT" ]; then
    echo "[ok] $ABSENT holds $(wc -l <"$ABSENT") prompts, kept"
else
    CUDA_VISIBLE_DEVICES=0 python "$EULLM_REPO/forge/scripts/make_grpo_prompts.py" \
        --norms "${NORMS[@]}" --exclude-exam "${EXAMS[@]}" --per-code "${GP_ABSENT_PER_CODE:-300}" \
        --absent-share 1.0 --inesistente-share 0.1 --seed "$SEED" "${RETR[@]}" \
        --out "$ABSENT" || RC=$?
fi

PRACTICAL="$OUT/prompts-practical-s$SEED.jsonl"
if [ -s "$PRACTICAL" ]; then
    echo "[ok] $PRACTICAL holds $(wc -l <"$PRACTICAL") prompts, kept"
else
    python "$EULLM_REPO/forge/scripts/make_practical_prompts.py" \
        --norms "${NORMS[@]}" --exclude-exam "${EXAMS[@]}" --per-code "${GP_PER_CODE:-400}" \
        --serve "$TEACHER" --llama-server "$SERVER" --gpus 0,1,2,3 --parallel 8 \
        --seed "$SEED" "${RETR[@]}" --out "$PRACTICAL" || RC=$?
fi

kill "$SAMPLER" 2>/dev/null
wait "$SAMPLER" 2>/dev/null
python "$EULLM_REPO/forge/scripts/jupiter/energy.py" "$POWER" --interval 5
echo "=== done $(date -Is) (exit $RC) ==="
exit "$RC"

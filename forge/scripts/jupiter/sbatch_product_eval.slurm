#!/bin/bash
# The candidates as the product runs them (roadmap docs/research/legal-it-roadmap.md,
# phase 0.2): each model's GGUF Q4_K_M served by EULLM Engine, asked open
# book with the hybrid retrieval and no note, on every set JUPITER holds:
# the trap test and its rewrites, the LexIA traps, the practical exam, the
# held-out exam. One model a GPU, up to four at once; the judge grades them
# after (sbatch_judge.slurm with JUDGE_ANSWERS_DIR, submitted behind this job).
#
#     eullm && sbatch --account e-aif-2026pg01-1434 \
#         --output="$PROJECT/$USER/forge/runs/logs/product-eval-%j.out" \
#         forge/scripts/jupiter/sbatch_product_eval.slurm
#
# PE_MODELS: "label=<merged dir>=<gguf> ..."; by default the 9B and 4B
# candidates (g9r-r2, g4-r3) and the 9B they started from (dqp d-q9b). The
# merged directory gives the tokenizer. Answers go to the teacher eval's
# directory as answers-<set>-<label>-q4e.jsonl, readable by the user alone
# (the sets are private), totals only in the log.

#SBATCH --job-name=forge-product-eval
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=03:00:00

set -uo pipefail
EULLM_REPO="${EULLM_REPO:-$SLURM_SUBMIT_DIR}"
# shellcheck source=env.sh
source "$EULLM_REPO/forge/scripts/jupiter/env.sh"
export EULLM_MODELS_DIR="${EULLM_MODELS_DIR:-$PROJECT/$USER/eullm-models}"
EULLM_BIN="${EULLM_BIN:-$PROJECT/$USER/eullm-target/v0.8.5/release/eullm}"
R="$FORGE_RUN_DIR" D="$FORGE_DATA_DIR" P="${FORGE_PRIVATE_EVAL:-$FORGE_HOME/private-eval}"
G="$R/gguf"
read -r -a MODELS <<<"${PE_MODELS:-g9r-r2=$R/grpo/g9r-r2/merged=$G/g9r-r2/g9r-r2-q4_k_m.gguf g4-r3=$R/grpo/g4-r3/merged=$G/g4-r3/g4-r3-q4_k_m.gguf dqp-d-q9b=$R/sft-dqp/d-q9b/merged=$G/d-q9b/d-q9b-q4_k_m.gguf}"
SETS="trap-test=$D/eval/trap-test.jsonl trap-test-para=$D/eval/trap-test-para.jsonl lexia-trappole=$P/lexia-trappole.jsonl practical-dev=$P/practical-dev.jsonl norm-exam=$D/eval/norm-exam.jsonl"
NORMS="$(ls "$D"/norms-src/legislazione_*.chunks.jsonl | grep -v eurlex | tr '\n' ' ')"
OUT="$R/teacher-eval"
JOB="${SLURM_JOB_ID:-local}"
mkdir -p "$OUT" "$R/power" "$R/retrieval-cache"
umask 077
chmod 700 "$OUT"

echo "=== the candidates as the product runs them: job $JOB on $(hostname), $(date -Is) ==="
echo "  engine $("$EULLM_BIN" --version 2>&1 | head -1) ($EULLM_BIN)"
bad=0
[ "${#MODELS[@]}" -le 4 ] || { echo "[!!] four models a node"; bad=1; }
for m in "${MODELS[@]}"; do
    IFS='=' read -r label merged gguf <<<"$m"
    [ -f "$merged/config.json" ] || { echo "[!!] $label: no merged model at $merged"; bad=1; }
    [ -s "$gguf" ] || { echo "[!!] $label: no GGUF at $gguf"; bad=1; }
done
for s in $SETS; do [ -s "${s#*=}" ] || { echo "[!!] missing ${s#*=}"; bad=1; }; done
[ "$bad" = 0 ] || exit 1

POWER="$R/power/product-eval-$JOB.csv"
nvidia-smi --query-gpu=timestamp,index,power.draw,utilization.gpu,memory.used \
    --format=csv,noheader,nounits -l 5 >"$POWER" 2>/dev/null &
SAMPLER=$!

ask() {   # ask <gpu> <label> <merged> <gguf>: every set, one after the other
    local g=$1 label=$2 merged=$3 gguf=$4 s set items ans t0
    for s in $SETS; do
        set="${s%%=*}" items="${s#*=}" ans="$OUT/answers-$set-$label-q4e.jsonl"
        [ -s "$ans" ] && { echo "  $label on $set: answered already"; continue; }
        t0=$(date +%s)
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES=$g taskset -c "$((72 * g))-$((72 * g + 71))" \
            python "$EULLM_REPO/forge/scripts/legal_eval.py" "$merged" --label "$set-$label-q4e" \
            --items "$items" --norms $NORMS --embedder Qwen/Qwen3-Embedding-0.6B \
            --reranker Qwen/Qwen3-Reranker-0.6B --retrieval-cache "$R/retrieval-cache" --no-note \
            --gguf "$gguf" --runtime eullm --eullm "$EULLM_BIN" \
            --server-parallel 16 --server-ctx 8192 --max-new-tokens 400 \
            --answers "$ans" --csv "$OUT/product-eval.csv" --quiet \
            >"$OUT/product-eval-$JOB-$label-$set.log" 2>&1
        echo "  $label on $set: exit $?, $(( $(date +%s) - t0 )) s, $(wc -l <"$ans" 2>/dev/null || echo 0) answers"
    done
}

PIDS=() g=0
for m in "${MODELS[@]}"; do
    IFS='=' read -r label merged gguf <<<"$m"
    ask "$g" "$label" "$merged" "$gguf" &
    PIDS+=($!)
    g=$((g + 1))
done
for p in "${PIDS[@]}"; do wait "$p"; done

kill "$SAMPLER" 2>/dev/null
wait "$SAMPLER" 2>/dev/null
python "$EULLM_REPO/forge/scripts/jupiter/energy.py" "$POWER" --interval 5
echo "=== done $(date -Is) ==="

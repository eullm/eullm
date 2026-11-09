#!/bin/bash
# General abilities of each model, as shipped (legal-it roadmap, phase 0.4):
# its GGUF Q4_K_M through EULLM Engine on GSM8K, ARC-Easy and ARC-Challenge,
# the sets campaign.py froze in $CAMPAIGN_DIR/sets (forge/scripts/regression_check.py).
# One model a GPU, up to four at once; one CSV row a model and set in
# $FORGE_RUN_DIR/regression/regression.csv. Run on every checkpoint from now
# on, beside the legal exams.
#
#     eullm && sbatch --account e-aif-2026pg01-1434 \
#         --output="$PROJECT/$USER/forge/runs/logs/regression-%j.out" \
#         forge/scripts/jupiter/sbatch_regression.slurm
#
# RG_MODELS: "label=<gguf> ..." (default the candidates and their start);
# RG_LIMIT: items a set (default 400). Every answer is kept in
# $FORGE_RUN_DIR/regression/answers/<label>-<set>.jsonl (public sets).

#SBATCH --job-name=forge-regression
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00

set -uo pipefail
EULLM_REPO="${EULLM_REPO:-$SLURM_SUBMIT_DIR}"
# shellcheck source=env.sh
source "$EULLM_REPO/forge/scripts/jupiter/env.sh"
EULLM_BIN="${EULLM_BIN:-$PROJECT/$USER/eullm-target/v0.8.5/release/eullm}"
SETS_DIR="${RG_SETS_DIR:-$PROJECT/$USER/campaign/sets}"
G="$FORGE_RUN_DIR/gguf"
read -r -a MODELS <<<"${RG_MODELS:-g9r-r2=$G/g9r-r2/g9r-r2-q4_k_m.gguf g4-r3=$G/g4-r3/g4-r3-q4_k_m.gguf dqp-d-q9b=$G/d-q9b/d-q9b-q4_k_m.gguf}"
SETS=("$SETS_DIR/gsm8k.jsonl" "$SETS_DIR/arc-easy.jsonl" "$SETS_DIR/arc-challenge.jsonl")
OUT="$FORGE_RUN_DIR/regression"
JOB="${SLURM_JOB_ID:-local}"
mkdir -p "$OUT"

echo "=== general abilities: job $JOB on $(hostname), $(date -Is) ==="
echo "  engine $("$EULLM_BIN" --version 2>&1 | head -1)"
bad=0
[ "${#MODELS[@]}" -le 4 ] || { echo "[!!] four models a node"; bad=1; }
for m in "${MODELS[@]}"; do [ -s "${m#*=}" ] || { echo "[!!] no GGUF at ${m#*=}"; bad=1; }; done
for s in "${SETS[@]}"; do [ -s "$s" ] || { echo "[!!] missing $s (campaign.py prefetch)"; bad=1; }; done
[ "$bad" = 0 ] || exit 1

PIDS=() g=0
for m in "${MODELS[@]}"; do
    CUDA_VISIBLE_DEVICES=$g taskset -c "$((72 * g))-$((72 * g + 71))" \
        python "$EULLM_REPO/forge/scripts/regression_check.py" --gguf "${m#*=}" --label "${m%%=*}" \
        --sets "${SETS[@]}" --limit "${RG_LIMIT:-400}" --eullm "$EULLM_BIN" \
        --csv "$OUT/regression-$JOB-${m%%=*}.csv" --answers-dir "$OUT/answers" >"$OUT/regression-$JOB-${m%%=*}.log" 2>&1 &
    PIDS+=($!)
    g=$((g + 1))
done
for p in "${PIDS[@]}"; do wait "$p"; done
grep -h '^\[regression\]' "$OUT"/regression-"$JOB"-*.log | sed 's/^/  /'
cat "$OUT"/regression-"$JOB"-*.csv 2>/dev/null | grep -v '^label' >>"$OUT/regression.csv"
echo "=== done $(date -Is) ==="

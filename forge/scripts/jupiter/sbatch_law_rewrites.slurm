#!/bin/bash
# Memory corpus, teacher part (law_rewrites.py), on JUPITER: the EU acts in
# Italian (norms-src/legislazione_eurlex.chunks.jsonl), which the legal-it
# roadmap (docs/research/legal-it-roadmap.md, phase 2) needs before E1 grows
# past the nine codes Leonardo rewrote. Legislation only: nothing here comes
# from a ruling.
#
# One teacher llama-server a GPU, four GPUs: Qwen3-30B-A3B-Instruct-2507
# Q8_0, the teacher Leonardo's rewrites used. Each link carries on from the
# rewrites written and stops asking 20 minutes before its end; a link with
# nothing left exits at once. Two chains share the work from both ends:
#
#     eullm && for c in a b; do prev=""; for i in 1 2 3 4; do \
#         prev=$(LR_OUT=$PROJECT/$USER/forge/datasets/memory/eurlex/rewrites-$c.jsonl \
#                LR_ARGS="$([ $c = b ] && echo --reverse)" \
#                sbatch --parsable --account e-aif-2026pg01-1434 ${prev:+--dependency=afterany:$prev} \
#                --output=$PROJECT/$USER/forge/runs/logs/law-rewrites-%j.out \
#                forge/scripts/jupiter/sbatch_law_rewrites.slurm); echo "$c$i $prev"; done; done
#
# Styles: paraphrase and elements by default. The questions-and-answers
# style waits for the closed-book suite to be on JUPITER, so that no article
# it asks about is rewritten as questions (law_rewrites.py --exclude); added
# later, only the missing rewrites are asked. The program part
# (make_law_corpus.py) waits for the same reason.
#
# Settable: LR_OUT, LR_NORMS (default the EU acts), LR_STYLES (default
# parafrasi,elementi), LR_EXCLUDE (held-out sets), LR_ARGS.

#SBATCH --job-name=forge-law-rewrites
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00

set -uo pipefail
EULLM_REPO="${EULLM_REPO:-$SLURM_SUBMIT_DIR}"
# shellcheck source=env.sh
source "$EULLM_REPO/forge/scripts/jupiter/env.sh"
export EULLM_MODELS_DIR="${EULLM_MODELS_DIR:-$PROJECT/$USER/eullm-models}"
D="$FORGE_DATA_DIR"
OUT="${LR_OUT:-$D/memory/eurlex/rewrites-a.jsonl}"
read -r -a NORMS <<<"${LR_NORMS:-$D/norms-src/legislazione_eurlex.chunks.jsonl}"
read -r -a EXCLUDE <<<"${LR_EXCLUDE:-}"
SERVER="${LLAMA_SERVER_BIN:-$PROJECT/$USER/campaign/runtimes/llama-server}"
TEACHER="${LR_TEACHER:-$(ls "$EULLM_MODELS_DIR"/*30b-a3b-instruct-2507*/*.gguf 2>/dev/null | grep -iv mmproj | head -1)}"
JOB="${SLURM_JOB_ID:-local}"
T0=$(date +%s)

echo "=== memory corpus, teacher part: job $JOB on $(hostname), $(date -Is) ==="
fail=0
[ -s "$TEACHER" ] || { echo "[!!] no teacher GGUF (Qwen3-30B-A3B-Instruct-2507) in $EULLM_MODELS_DIR"; fail=1; }
[ -x "$SERVER" ] || { echo "[!!] no llama-server at $SERVER"; fail=1; }
for f in "${NORMS[@]}" "${EXCLUDE[@]}"; do [ -s "$f" ] || { echo "[!!] missing $f"; fail=1; }; done
[ "$fail" = 0 ] || exit 1
mkdir -p "$(dirname "$OUT")" "$FORGE_RUN_DIR/power"
echo "  teacher $TEACHER, one server a GPU; ${NORMS[*]} -> $OUT"

LIMIT_S=$(job_time_limit_s)
STOP=$(( ${LIMIT_S:-21600} - ($(date +%s) - T0) - 1200 ))
[ "$STOP" -gt 600 ] || { echo "[!!] no time left ($STOP s)"; exit 1; }

POWER="$FORGE_RUN_DIR/power/law-rewrites-$JOB.csv"
nvidia-smi --query-gpu=timestamp,index,power.draw,utilization.gpu,memory.used \
    --format=csv,noheader,nounits -l 5 >"$POWER" 2>/dev/null &
SAMPLER=$!
EX=()
[ "${#EXCLUDE[@]}" -gt 0 ] && EX=(--exclude "${EXCLUDE[@]}")
# shellcheck disable=SC2086
python "$EULLM_REPO/forge/scripts/law_rewrites.py" --norms "${NORMS[@]}" "${EX[@]}" \
    --out "$OUT" --styles "${LR_STYLES:-parafrasi,elementi}" --teacher "$(basename "$TEACHER" .gguf)" \
    --serve "$TEACHER" --gpus 0,1,2,3 --llama-server "$SERVER" \
    --stop-after "$STOP" ${LR_ARGS:-}
RC=$?
kill "$SAMPLER" 2>/dev/null
wait "$SAMPLER" 2>/dev/null
python "$EULLM_REPO/forge/scripts/jupiter/energy.py" "$POWER" --interval 5
for f in "$(dirname "$OUT")"/rewrites-*.jsonl; do [ -s "$f" ] && echo "  $(wc -l <"$f") lines in $f"; done
echo "=== done $(date -Is) (exit $RC) ==="
exit "$RC"

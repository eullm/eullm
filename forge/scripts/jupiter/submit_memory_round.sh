#!/usr/bin/env bash
# One E1-mini run from a login node, measured to the end without anyone
# waiting on it: the training links (sbatch_memory_cpt.slurm), then the GGUF
# (sbatch_vertical.slurm), then the closed book, the seven open-book sets as
# the product runs them and the general-abilities regression, all behind the
# GGUF. The judge is left to the caller, so that several rounds share one
# judge chain (two judges on one directory would grade the same files).
#
#     eullm && MC_PREPARE_ARGS="--epochs 5" \
#         bash forge/scripts/jupiter/submit_memory_round.sh e1-g9-e5 $PROJECT/$USER/forge/runs/grpo/g9r-r2/merged
#
# MC_PREPARE_ARGS / MC_TRAIN_ARGS reach the training (memory_cpt.py prepare /
# train); MR_LINKS training links (default 2) of MR_LINK_H hours (default 6);
# MR_AFTER a job the first link waits for. Nothing is submitted when a check
# fails, or when the round is already queued. The last line printed is
# "OPEN <job>", the open-book evaluation, for the judge to wait on.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"
# shellcheck source=sbatch_retry.sh
source "$HERE/sbatch_retry.sh"
REPO="$(cd "$HERE/../../.." && pwd)"
NAME="${1:?usage: submit_memory_round.sh NAME MODEL}"
MODEL="${2:?usage: submit_memory_round.sh NAME MODEL}"
R="$FORGE_RUN_DIR" D="$FORGE_DATA_DIR" P="${FORGE_PRIVATE_EVAL:-$FORGE_HOME/private-eval}"
G="$R/gguf" OUT="$R/memory/$NAME" LOGS="$R/logs" S=forge/scripts/jupiter
bad=0
[ -f "$MODEL/config.json" ] || [ -n "${MR_AFTER:-}" ] || { echo "[!!] no model at $MODEL"; bad=1; }
for f in "$D/memory/general-it.jsonl" "$D/openbook/pairs.jsonl" "$D/eval/leonardo/closed-book.jsonl" \
         "$D/eval/leonardo/norm-exam-devbig.jsonl" "$P/practical-heldout.jsonl"; do
    [ -s "$f" ] || { echo "[!!] missing $f"; bad=1; }
done
[ "$bad" = 0 ] || { echo "[!!] nothing submitted"; exit 1; }
wait_for_slurm || exit 1
if [ -n "$(squeue -h -u "$USER" -n "forge-mem-$NAME" -o %i)" ]; then
    echo "[ok] $NAME already queued"; exit 0
fi
mkdir -p "$LOGS"
cd "$REPO" || exit 1
sub() { local id; id=$(sbatch_retry --parsable --account "$SBATCH_ACCOUNT" "$@") || exit 1; echo "${id%%;*}"; }

prev="${MR_AFTER:-}"
for i in $(seq 1 "${MR_LINKS:-2}"); do
    prev=$(MC_MODEL="$MODEL" MC_OUT="$OUT" sub --job-name="forge-mem-$NAME" \
        --time="${MR_LINK_H:-6}:00:00" ${prev:+--dependency=afterany:$prev} \
        --output="$LOGS/memory-cpt-%j.out" $S/sbatch_memory_cpt.slurm) || exit 1
    echo "[ok] training link $i: $prev"
done
v=$(VERT_SFT="$R/memory" VERT_ARMS="$NAME" sub --time=04:00:00 --dependency="afterany:$prev" \
    --output="$LOGS/vertical-%j.out" $S/sbatch_vertical.slurm) || exit 1
M="$NAME=$OUT/merged=$G/$NAME/$NAME-q4_k_m.gguf"
cb=$(PE_MODELS="$M" PE_CLOSED=1 PE_SETS="closed-book=$D/eval/leonardo/closed-book.jsonl" \
    sub --time=01:30:00 --dependency="afterany:$v" --output="$LOGS/product-eval-%j.out" \
    $S/sbatch_product_eval.slurm) || exit 1
pe=$(PE_MODELS="$M" PE_SETS="trap-test=$D/eval/trap-test.jsonl trap-test-para=$D/eval/trap-test-para.jsonl lexia-trappole=$P/lexia-trappole.jsonl practical-dev=$P/practical-dev.jsonl norm-exam=$D/eval/norm-exam.jsonl norm-exam-devbig=$D/eval/leonardo/norm-exam-devbig.jsonl practical-heldout=$P/practical-heldout.jsonl" \
    sub --time=04:00:00 --dependency="afterany:$v" --output="$LOGS/product-eval-%j.out" \
    $S/sbatch_product_eval.slurm) || exit 1
rg=$(RG_MODELS="$NAME=$G/$NAME/$NAME-q4_k_m.gguf" sub --dependency="afterany:$v" \
    --output="$LOGS/regression-%j.out" $S/sbatch_regression.slurm) || exit 1
echo "[ok] $NAME: GGUF $v, closed book $cb, open book $pe, regression $rg"
echo "OPEN $pe"

#!/bin/bash
# Questions on a false premise (make_premise_pairs.py), as on Leonardo
# (forge/scripts/leonardo/sbatch_premise_pairs.slurm): a wrong deadline or
# penalty, an article that does not exist, a repealed one taken for one in
# force, each answered by correcting the premise from the retrieved text. The
# premise is false by construction, so the answers need no teacher. Built
# from the legislation alone, no court decision. SFT pairs only, in the three
# checked steps of Leonardo's best rounds (r2b-steps): no GRPO here.
#
#     eullm && sbatch --account e-aif-2026pg01-1434 \
#         --output="$PROJECT/$USER/forge/runs/logs/premise-%j.out" \
#         forge/scripts/jupiter/sbatch_premise_pairs.slurm
#
# No article of the held-out exam, the practical exam or the LexIA traps is
# used (PP_EXCLUDE): they stay tests. The log carries counts only, never a
# question. Writes $FORGE_DATA_DIR/premise/sft-steps.jsonl; an existing file
# is kept, so a second submission ends at once. PP_GRPO_OUT=<file> writes
# GRPO prompts instead (every pair a prompt, graded by the rewards on its
# kind): the false premises, made-up rulings among them, that the GRPO
# prompts copied from Leonardo hold too few of (10 of 2,184, 09-11-2026). One GPU for the retrieval
# models of a whole node: minutes.

#SBATCH --job-name=forge-premise
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00

set -uo pipefail
EULLM_REPO="${EULLM_REPO:-$SLURM_SUBMIT_DIR}"
# shellcheck source=env.sh
source "$EULLM_REPO/forge/scripts/jupiter/env.sh"
OUT="${PP_GRPO_OUT:-${PP_OUT:-$FORGE_DATA_DIR/premise/sft-steps.jsonl}}"
if [ -n "${PP_GRPO_OUT:-}" ]; then
    MODE=(--grpo-share 1 --out-grpo "$OUT")
else
    MODE=(--grpo-share 0 --out-sft "$OUT")
fi
P="${FORGE_PRIVATE_EVAL:-$FORGE_HOME/private-eval}"
read -r -a NORMS <<<"${PP_NORMS:-$(ls "$FORGE_DATA_DIR"/norms-src/legislazione_*.chunks.jsonl | grep -v eurlex | tr '\n' ' ')}"
read -r -a EXCLUDE <<<"${PP_EXCLUDE:-$(ls "$FORGE_DATA_DIR"/eval/norm-exam*.jsonl | tr '\n' ' ') $P/practical-dev.jsonl $P/lexia-trappole.jsonl}"

echo "=== Forge premise pairs: job ${SLURM_JOB_ID:-local} on $(hostname), $(date -Is) ==="
if [ -s "$OUT" ]; then
    echo "[ok] $OUT holds $(wc -l <"$OUT") pairs, kept"
    exit 0
fi
fail=0
for f in "${NORMS[@]}" "${EXCLUDE[@]}"; do
    [ -s "$f" ] || { echo "[!!] missing $f"; fail=1; }
done
[ "$fail" = 0 ] || exit 1
echo "  ${#NORMS[@]} norm files, ${#EXCLUDE[@]} sets left out"
mkdir -p "$(dirname "$OUT")" "$FORGE_RUN_DIR/retrieval-cache"
CUDA_VISIBLE_DEVICES=0 python "$EULLM_REPO/forge/scripts/make_premise_pairs.py" \
    --norms "${NORMS[@]}" --exclude-exam "${EXCLUDE[@]}" \
    --per-code "${PP_PER_CODE:-80}" --seed "${PP_SEED:-0}" --format steps \
    --embedder Qwen/Qwen3-Embedding-0.6B --reranker Qwen/Qwen3-Reranker-0.6B \
    --retrieval-cache "$FORGE_RUN_DIR/retrieval-cache" "${MODE[@]}"
RC=$?
[ -s "$OUT" ] && echo "[ok] $(wc -l <"$OUT") rows in $OUT" || { echo "[!!] no pairs written (exit $RC)"; RC=1; }
echo "=== done $(date -Is) (exit $RC) ==="
exit "$RC"

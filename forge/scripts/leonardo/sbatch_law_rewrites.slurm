#!/bin/bash
# Memory corpus, teacher part (law_rewrites.py): every article rewritten,
# one teacher llama-server per GPU, three GPUs, two-hour links of a chain.
#
#   bash forge/scripts/leonardo/submit_chain.sh \
#       forge/scripts/leonardo/sbatch_law_rewrites.slurm 6 -J eullm-rewrites
#
# A second chain from the other end of the list, sharing the work:
#   LR_OUT=$WORK/eullm_runs/memory/rewrites-b.jsonl LR_ARGS=--reverse \
#   bash forge/scripts/leonardo/submit_chain.sh \
#       forge/scripts/leonardo/sbatch_law_rewrites.slurm 6 -J eullm-rewrites-b
#
# Every link carries on from the last rewrite written and stops taking new
# articles after 105 minutes; a link that finds nothing left exits at once.
# The program-written part (make_law_corpus.py) needs no GPU and is run
# here first, once, when its output is missing.
#
# Three GPUs, 24 cores, 340 GB: the shape that gets placed (forge/CLAUDE.md).
#
# Settable:
#   LR_OUT      rewrites JSONL (default $WORK/eullm_runs/memory/rewrites.jsonl)
#   LR_NORMS    legislation files (default $WORK/norms/legislazione_*.chunks.jsonl)
#   LR_EXCLUDE  evaluation sets held out (default $WORK/eval/closed-book.jsonl)
#   LR_TEACHER  teacher GGUF (default the judge's Q8_0)
#   LR_GPUS     --gpus as law_rewrites.py takes it (default one server per GPU);
#               "0+1+2" serves a teacher too big for one GPU over all three
#   LR_ARGS     extra law_rewrites.py options, e.g. "--samples 2" or "--reverse"
#   LR_REPO     checkout to run (default: the one this file was submitted from)
#
#SBATCH --job-name=eullm-rewrites
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:3
#SBATCH --mem=340G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail

SELF="$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null |
        sed -n 's/.*Command=\([^ ]*\).*/\1/p' | head -1)"
if [ -n "${LR_REPO:-}" ]; then
    REPO="$LR_REPO"
elif [ -n "$SELF" ] && [ -f "$SELF" ]; then
    REPO="$(cd "$(dirname "$SELF")/../../.." && pwd)"
else
    REPO="${EULLM_REPO:-$WORK/eullm}"
fi
# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/env.sh"

OUT="${LR_OUT:-$WORK/eullm_runs/memory/rewrites.jsonl}"
TEACHER="${LR_TEACHER:-$WORK/gguf/qwen3-30b-a3b-instruct-2507/qwen3-30b-a3b-instruct-2507-q8_0.gguf}"
# shellcheck disable=SC2206
NORMS=(${LR_NORMS:-$WORK/norms/legislazione_*.chunks.jsonl})
# shellcheck disable=SC2206
EXCLUDE=(${LR_EXCLUDE:-$WORK/eval/closed-book.jsonl})
SERVER="${LCPP_SERVER:-$LCPP_DIR/build-cuda/bin/llama-server}"
GPUS="$(nvidia-smi -L 2>/dev/null | wc -l)"

fail=0
[ -s "$TEACHER" ] || { echo "[lr] no teacher GGUF at $TEACHER" >&2; fail=1; }
[ -x "$SERVER" ] || { echo "[lr] no llama-server at $SERVER" >&2; fail=1; }
for f in "${NORMS[@]}"; do [ -s "$f" ] || { echo "[lr] missing $f" >&2; fail=1; }; done
for f in "${EXCLUDE[@]}"; do [ -s "$f" ] || { echo "[lr] missing held-out set $f" >&2; fail=1; }; done
[ "$GPUS" -ge 1 ] || { echo "[lr] no GPU visible" >&2; fail=1; }
[ "$fail" -eq 0 ] || exit 1

if [ -s "$(dirname "$SERVER")/../modules.txt" ] && command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC2046
    module load $(cat "$(dirname "$SERVER")/../modules.txt") 2>/dev/null || true
fi
DEVS="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((GPUS - 1)))}"
echo "[lr] repo $REPO ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?')), GPUs $DEVS"

PROGRAM="$(dirname "$OUT")/program.jsonl"
if [ ! -s "$PROGRAM" ]; then
    python "$REPO/forge/scripts/make_law_corpus.py" --norms "${NORMS[@]}" \
        --exclude "${EXCLUDE[@]}" --out "$PROGRAM" ||
        echo "[lr] the program part failed; the rewrites go on" >&2
fi

# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/heartbeat.sh"
heartbeat_start

# shellcheck disable=SC2086
python "$REPO/forge/scripts/law_rewrites.py" --norms "${NORMS[@]}" --exclude "${EXCLUDE[@]}" \
    --out "$OUT" --serve "$TEACHER" --gpus "${LR_GPUS:-$DEVS}" --llama-server "$SERVER" \
    --stop-after 6300 ${LR_ARGS:-}

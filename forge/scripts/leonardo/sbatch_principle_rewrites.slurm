#!/bin/bash
# Memory corpus, case-law part (principle_rewrites.py): every card's
# principles rewritten, one teacher llama-server per GPU, two-hour links.
#
#   bash forge/scripts/leonardo/submit_chain.sh \
#       forge/scripts/leonardo/sbatch_principle_rewrites.slurm 8 -J eullm-principi
#
# More chains share the list as the rewrites' do: PR_OUT=.../principi-b.jsonl
# PR_ARGS=--reverse, and --start 0.5 (with or without --reverse) for a third
# and a fourth. Every link carries on from the last card done and stops
# taking new ones after 105 minutes; a link with nothing left exits at once.
#
# Three GPUs, 24 cores, 340 GB: the shape that gets placed (forge/CLAUDE.md);
# one GPU, 8 cores, 120 GB passed on the command line works the same.
#
# Settable:
#   PR_OUT      rows JSONL (default $WORK/eullm_runs/memory/principi.jsonl)
#   PR_CARDS    card files (default the Consiglio di Stato's and the civil ones)
#   PR_COURTS   courts asked (default cds,snciv)
#   PR_TEACHER  teacher GGUF (default the judge's Q8_0)
#   PR_ARGS     extra principle_rewrites.py options, e.g. "--reverse"
#   PR_REPO     checkout to run (default: the one this file was submitted from)
#
#SBATCH --job-name=eullm-principi
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
if [ -n "${PR_REPO:-}" ]; then
    REPO="$PR_REPO"
elif [ -n "$SELF" ] && [ -f "$SELF" ]; then
    REPO="$(cd "$(dirname "$SELF")/../../.." && pwd)"
else
    REPO="${EULLM_REPO:-$WORK/eullm}"
fi
# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/env.sh"

OUT="${PR_OUT:-$WORK/eullm_runs/memory/principi.jsonl}"
TEACHER="${PR_TEACHER:-$WORK/gguf/qwen3-30b-a3b-instruct-2507/qwen3-30b-a3b-instruct-2507-q8_0.gguf}"
# shellcheck disable=SC2206
CARDS=(${PR_CARDS:-$WORK/eullm_runs/cds/schede*.jsonl $WORK/eullm_runs/cass/schede-civ*.jsonl})
SERVER="${LCPP_SERVER:-$LCPP_DIR/build-cuda/bin/llama-server}"
GPUS="$(nvidia-smi -L 2>/dev/null | wc -l)"

fail=0
[ -s "$TEACHER" ] || { echo "[pr] no teacher GGUF at $TEACHER" >&2; fail=1; }
[ -x "$SERVER" ] || { echo "[pr] no llama-server at $SERVER" >&2; fail=1; }
for f in "${CARDS[@]}"; do [ -s "$f" ] || { echo "[pr] missing $f" >&2; fail=1; }; done
[ "$GPUS" -ge 1 ] || { echo "[pr] no GPU visible" >&2; fail=1; }
[ "$fail" -eq 0 ] || exit 1

if [ -s "$(dirname "$SERVER")/../modules.txt" ] && command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC2046
    module load $(cat "$(dirname "$SERVER")/../modules.txt") 2>/dev/null || true
fi
DEVS="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((GPUS - 1)))}"
echo "[pr] repo $REPO ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?')), GPUs $DEVS"

# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/heartbeat.sh"
heartbeat_start

# shellcheck disable=SC2086
python "$REPO/forge/scripts/principle_rewrites.py" --cards "${CARDS[@]}" \
    --courts "${PR_COURTS:-cds,snciv}" --out "$OUT" --serve "$TEACHER" --gpus "$DEVS" \
    --llama-server "$SERVER" --stop-after 6300 ${PR_ARGS:-}

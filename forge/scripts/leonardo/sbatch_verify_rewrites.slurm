#!/bin/bash
# One checker model over the verification items of the memory corpus
# (verify_rewrites.py check): three GPUs, two-hour links of a chain, resumable.
#
#   VR_CHECKER=<checker.gguf> VR_LABEL=q30b bash forge/scripts/leonardo/submit_chain.sh \
#       forge/scripts/leonardo/sbatch_verify_rewrites.slurm 3 -J eullm-verify-q30b
#
# The items (the sample, and its copies made wrong on purpose) are written
# once by `verify_rewrites.py prepare` on a login node; every checker reads
# the same file, so their recalls are comparable. A link that finds every
# item checked exits at once.
#
# Settable:
#   VR_CHECKER  checker GGUF (required)
#   VR_LABEL    checker label, the verdicts file's name (required)
#   VR_GPUS     --gpus: "0,1,2" one server a GPU (default), "0+1+2" one over three
#   VR_ITEMS    items (default $WORK/eullm_runs/verify/items.jsonl)
#   VR_REPO     checkout to run (default: the one this file was submitted from)
#
#SBATCH --job-name=eullm-verify
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
if [ -n "${VR_REPO:-}" ]; then
    REPO="$VR_REPO"
elif [ -n "$SELF" ] && [ -f "$SELF" ]; then
    REPO="$(cd "$(dirname "$SELF")/../../.." && pwd)"
else
    REPO="${EULLM_REPO:-$WORK/eullm}"
fi
# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/env.sh"

ITEMS="${VR_ITEMS:-$WORK/eullm_runs/verify/items.jsonl}"
CHECKER="${VR_CHECKER:-}"
LABEL="${VR_LABEL:-}"
SERVER="${LCPP_SERVER:-$LCPP_DIR/build-cuda/bin/llama-server}"
GPUS="$(nvidia-smi -L 2>/dev/null | wc -l)"
fail=0
[ -s "$ITEMS" ] || { echo "[vr] no items at $ITEMS: verify_rewrites.py prepare first" >&2; fail=1; }
[ -s "$CHECKER" ] || { echo "[vr] no checker GGUF at '$CHECKER'" >&2; fail=1; }
[ -n "$LABEL" ] || { echo "[vr] VR_LABEL is required" >&2; fail=1; }
[ -x "$SERVER" ] || { echo "[vr] no llama-server at $SERVER" >&2; fail=1; }
[ "$GPUS" -ge 1 ] || { echo "[vr] no GPU visible" >&2; fail=1; }
[ "$fail" -eq 0 ] || exit 1
if [ -s "$(dirname "$SERVER")/../modules.txt" ] && command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC2046
    module load $(cat "$(dirname "$SERVER")/../modules.txt") 2>/dev/null || true
fi
DEVS="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((GPUS - 1)))}"
OUT="$(dirname "$ITEMS")/verdicts-$LABEL.jsonl"
echo "[vr] checker $LABEL ($CHECKER) on GPUs ${VR_GPUS:-$DEVS} -> $OUT"

# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/heartbeat.sh"
heartbeat_start

python "$REPO/forge/scripts/verify_rewrites.py" check --items "$ITEMS" --out "$OUT" \
    --label "$LABEL" --serve "$CHECKER" --gpus "${VR_GPUS:-$DEVS}" --llama-server "$SERVER" \
    --stop-after 6300

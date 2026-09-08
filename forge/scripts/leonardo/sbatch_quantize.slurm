#!/bin/bash
# Phase 3 — quantize the distilled student to GGUF Q4_K_M. CPU-only, so
# it runs on the serial partition and costs no GPU budget. llama.cpp
# must already be cloned at $LCPP_DIR (setup.sh does it — this node has
# no outbound network, hence LCPP_SKIP_UPDATE=1).
#
#   bash "$EULLM_REPO/forge/scripts/leonardo/submit_chain.sh" \
#       "$EULLM_REPO/forge/scripts/leonardo/sbatch_quantize.slurm"
#
#SBATCH --job-name=eullm-p3-gguf
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

EULLM_REPO="${EULLM_REPO:-$WORK/eullm}"
# shellcheck disable=SC1091
source "$EULLM_REPO/forge/scripts/leonardo/env.sh"
cd "$EULLM_RUN_DIR"

# The LoRA student is exported merged (full HF weights) by distill.py.
HF_DIR="$EULLM_RUN_DIR/checkpoints/qwen3_7b_legal_it_distilled/merged"
[ -f "$HF_DIR/config.json" ] || {
    echo "[err] merged student not found at $HF_DIR" >&2
    echo "[err] Phase 2 must run to completion first (the merge happens" >&2
    echo "[err] in its final save)" >&2
    exit 1
}

LCPP_SKIP_UPDATE=1 bash "$EULLM_REPO/forge/scripts/quantize_to_gguf.sh" \
    "$HF_DIR" \
    "$EULLM_RUN_DIR/gguf/legal-it-7b"

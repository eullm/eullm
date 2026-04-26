#!/usr/bin/env bash
# Universal LLaMA-Factory training launcher.
#
# Handles three things on top of `llamafactory-cli train`:
#   1. Copies forge/training/configs/dataset_info.json next to the
#      dataset files so LLaMA-Factory can discover the corpus.
#   2. Auto-detects the most recent checkpoint inside the YAML's
#      output_dir and passes --resume_from_checkpoint, so a re-run
#      after a preemption picks up where it left off.
#   3. Logs the resolved command before launching, so you can copy/
#      paste it later for debugging.
#
# Usage:
#   bash forge/scripts/train.sh <config.yaml> [data-dir]
#
# If <data-dir> is omitted, defaults to:
#   - $TRAINING_DATA_DIR if set
#   - ~/italgiure_corpus/pretraining (smoke-test default)
#
# Examples:
#   # Smoke test on the local 5070 Ti workstation
#   bash forge/scripts/train.sh \\
#       forge/training/configs/smoke_qwen3_1.7b.yaml
#
#   # Production continued PT on a rented 96 GB instance
#   bash forge/scripts/train.sh \\
#       forge/training/configs/continued_pt_qwen3_32b.yaml \\
#       ~/datasets/legal_it
#
#   # Resume after a preemption — same command, no special flag needed
#   bash forge/scripts/train.sh \\
#       forge/training/configs/continued_pt_qwen3_32b.yaml \\
#       ~/datasets/legal_it

set -euo pipefail

CONFIG="${1:?Usage: $0 <config.yaml> [data-dir]}"
DATA_DIR="${2:-${TRAINING_DATA_DIR:-$HOME/italgiure_corpus/pretraining}}"

err() { printf '\033[31m[err]\033[0m %s\n' "$*" >&2; exit 1; }
ok()  { printf '\033[32m[ok]\033[0m  %s\n' "$*"; }
log() { printf '\033[34m[..]\033[0m  %s\n' "$*"; }

[ -f "$CONFIG" ]    || err "config not found: $CONFIG"
[ -d "$DATA_DIR" ]  || err "data dir not found: $DATA_DIR"
[ -f "$DATA_DIR/train.jsonl" ] || err "missing $DATA_DIR/train.jsonl"
[ -f "$DATA_DIR/val.jsonl" ]   || err "missing $DATA_DIR/val.jsonl"
command -v llamafactory-cli >/dev/null || \
    err "llamafactory-cli not on PATH — run forge/scripts/install_training_deps.sh"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET_INFO_SRC="$REPO_ROOT/forge/training/configs/dataset_info.json"
DATASET_INFO_DST="$DATA_DIR/dataset_info.json"

[ -f "$DATASET_INFO_SRC" ] || err "missing $DATASET_INFO_SRC"

# Copy the LLaMA-Factory dataset registration next to the dataset
# files. cp is unconditional so a doc-update on dataset_info.json
# always propagates.
cp "$DATASET_INFO_SRC" "$DATASET_INFO_DST"
ok "dataset_info.json copied to $DATASET_INFO_DST"

# Extract output_dir from the YAML to look for an existing checkpoint
# WITHOUT requiring yaml/python deps.
OUTPUT_DIR=$(grep -E '^output_dir:' "$CONFIG" \
    | head -1 | sed -E 's/output_dir:[[:space:]]*//; s/[[:space:]]+#.*$//; s/^["'\'']//; s/["'\'']$//')

if [ -z "$OUTPUT_DIR" ]; then
    err "could not extract output_dir from $CONFIG"
fi

RESUME_ARG=()
if [ -d "$OUTPUT_DIR" ]; then
    LATEST_CKPT=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' \
        -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | awk 'NR==1{print $2}')
    if [ -n "$LATEST_CKPT" ]; then
        log "found existing checkpoint: $LATEST_CKPT — resuming"
        RESUME_ARG=(--resume_from_checkpoint "$LATEST_CKPT")
    else
        log "$OUTPUT_DIR exists but no checkpoint inside — starting fresh"
    fi
else
    log "no $OUTPUT_DIR yet — starting fresh"
fi

CMD=(llamafactory-cli train "$CONFIG"
     --dataset_dir "$DATA_DIR"
     "${RESUME_ARG[@]}")

echo
log "Launching:"
printf '   %s\n' "${CMD[*]}"
echo

exec "${CMD[@]}"

#!/usr/bin/env bash
# Wrapper around forge/scripts/distill.py with the same resume-friendly
# semantics as forge/scripts/train.sh: pass the YAML, optionally a
# data dir, and a re-run will pick up the most recent checkpoint inside
# the YAML's output_dir.
#
# Usage:
#   bash forge/scripts/distill.sh <config.yaml> [data-dir]
#
# Example:
#   bash forge/scripts/distill.sh \
#       forge/training/configs/distill_qwen3_32b_to_7b.yaml \
#       ~/datasets/legal_it

set -euo pipefail

CONFIG="${1:?Usage: $0 <config.yaml> [data-dir]}"
DATA_DIR="${2:-${TRAINING_DATA_DIR:-$HOME/datasets/legal_it}}"

err() { printf '\033[31m[err]\033[0m %s\n' "$*" >&2; exit 1; }
ok()  { printf '\033[32m[ok]\033[0m  %s\n' "$*"; }
log() { printf '\033[34m[..]\033[0m  %s\n' "$*"; }

[ -f "$CONFIG" ]    || err "config not found: $CONFIG"
[ -d "$DATA_DIR" ]  || err "data dir not found: $DATA_DIR"
[ -f "$DATA_DIR/train.jsonl" ] || err "missing $DATA_DIR/train.jsonl"
[ -f "$DATA_DIR/val.jsonl" ]   || err "missing $DATA_DIR/val.jsonl"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$REPO_ROOT/forge/scripts/distill.py"
[ -f "$SCRIPT" ] || err "missing $SCRIPT"

# Pull the output_dir out of the YAML (one shell-only sed, no Python
# imports needed here — distill.py will load and validate the full
# config itself).
OUTPUT_DIR=$(grep -E '^output_dir:' "$CONFIG" \
    | head -1 | sed -E 's/output_dir:[[:space:]]*//; s/[[:space:]]+#.*$//; s/^["'\'']//; s/["'\'']$//')
OUTPUT_DIR="${OUTPUT_DIR/#\~/$HOME}"
[ -n "$OUTPUT_DIR" ] || err "could not extract output_dir from $CONFIG"

RESUME_ARG=()
if [ -d "$OUTPUT_DIR" ]; then
    LATEST_CKPT=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' \
        -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | awk 'NR==1{print $2}')
    if [ -n "$LATEST_CKPT" ]; then
        log "found existing checkpoint: $LATEST_CKPT — resuming"
        RESUME_ARG=(--resume-from "$LATEST_CKPT")
    fi
fi

CMD=(python "$SCRIPT"
     --config "$CONFIG"
     --dataset-dir "$DATA_DIR"
     "${RESUME_ARG[@]}")

echo
log "Launching:"
printf '   %s\n' "${CMD[*]}"
echo

exec "${CMD[@]}"

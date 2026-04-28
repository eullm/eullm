#!/usr/bin/env bash
# Universal LLaMA-Factory training launcher.
#
# Handles three things on top of `llamafactory-cli train`:
#   1. Copies forge/training/configs/dataset_info.json next to the
#      dataset files so LLaMA-Factory can discover the corpus.
#   2. Materialises a temporary YAML with __DATASET_DIR__ replaced by
#      the runtime data dir, and (when an existing checkpoint is
#      found) appends `resume_from_checkpoint: <path>`. LLaMA-Factory
#      0.9.5 rejects extra CLI args when a YAML is given, so we keep
#      everything in the YAML.
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
#   bash forge/scripts/train.sh \
#       forge/training/configs/smoke_qwen3_1.7b.yaml
#
#   # Production continued PT on a rented 96 GB instance
#   bash forge/scripts/train.sh \
#       forge/training/configs/continued_pt_qwen3_32b.yaml \
#       ~/datasets/legal_it
#
#   # Resume after a preemption — same command, no special flag needed
#   bash forge/scripts/train.sh \
#       forge/training/configs/continued_pt_qwen3_32b.yaml \
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
cp "$DATASET_INFO_SRC" "$DATASET_INFO_DST"
ok "dataset_info.json copied to $DATASET_INFO_DST"

# Materialise a runtime YAML with __DATASET_DIR__ filled in (and
# resume_from_checkpoint appended if applicable). The temp file is
# cleaned up on exit, including on Ctrl-C.
TMP_YAML=$(mktemp --suffix=.yaml)
trap 'rm -f "$TMP_YAML"' EXIT

# Use a delimiter that can't appear in a Unix path (|) and an absolute
# DATA_DIR so the substitution is unambiguous.
ABS_DATA_DIR="$(cd "$DATA_DIR" && pwd)"
sed "s|__DATASET_DIR__|$ABS_DATA_DIR|g" "$CONFIG" > "$TMP_YAML"

# Extract output_dir from the resolved YAML to look for an existing
# checkpoint.
OUTPUT_DIR=$(grep -E '^output_dir:' "$TMP_YAML" \
    | head -1 | sed -E 's/output_dir:[[:space:]]*//; s/[[:space:]]+#.*$//; s/^["'\'']//; s/["'\'']$//')

if [ -z "$OUTPUT_DIR" ]; then
    err "could not extract output_dir from $CONFIG"
fi

# Resolve ~ expansion
OUTPUT_DIR="${OUTPUT_DIR/#\~/$HOME}"

if [ -d "$OUTPUT_DIR" ]; then
    LATEST_CKPT=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' \
        -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | awk 'NR==1{print $2}')
    if [ -n "$LATEST_CKPT" ]; then
        log "found existing checkpoint: $LATEST_CKPT — resuming"
        echo "resume_from_checkpoint: $LATEST_CKPT" >> "$TMP_YAML"
    else
        log "$OUTPUT_DIR exists but no checkpoint inside — starting fresh"
    fi
else
    log "no $OUTPUT_DIR yet — starting fresh"
fi

echo
log "Resolved YAML at $TMP_YAML — launching:"
printf '   llamafactory-cli train %s\n' "$TMP_YAML"
echo

exec llamafactory-cli train "$TMP_YAML"

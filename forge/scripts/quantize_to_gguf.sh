#!/usr/bin/env bash
# Phase 3 — Convert the distilled student to GGUF Q4_K_M and smoke-test it.
#
# Inputs:
#   1) HF model directory (the final --output-dir of distill.py),
#      containing config.json, tokenizer.json, and a .safetensors weight
#      file. Must be in standard HuggingFace format (not LoRA adapter).
#   2) Output directory for the GGUF file.
#
# Steps:
#   * clone llama.cpp (CPU build, no GPU needed for conversion +
#     quantization) into ~/llama.cpp on first run
#   * build the conversion + quantize binaries
#   * convert HF → GGUF F16 (full precision), ~14 GB for a 7B model
#   * quantize F16 → Q4_K_M, ~4.5 GB (Q4_K_M is the sweet spot for
#     legal text quality vs file size)
#   * run a smoke prompt through llama-cli to confirm the GGUF loads
#
# Usage:
#   bash forge/scripts/quantize_to_gguf.sh \
#       <hf-model-dir> [output-dir]
#
# Example:
#   bash forge/scripts/quantize_to_gguf.sh \
#       ~/checkpoints/qwen3_7b_legal_it_distilled \
#       ~/gguf/legal-it-7b
#
# After this completes, the file at <output-dir>/legal-it-7b-q4_k_m.gguf
# can be loaded by the EULLM Engine, Ollama, or any llama.cpp-compatible
# runtime.

set -euo pipefail

HF_DIR="${1:?Usage: $0 <hf-model-dir> [output-dir]}"
OUT_DIR="${2:-$HF_DIR/gguf}"
LCPP_DIR="${LCPP_DIR:-$HOME/llama.cpp}"
LCPP_REPO="${LCPP_REPO:-https://github.com/ggerganov/llama.cpp.git}"
GGUF_NAME="${GGUF_NAME:-legal-it-7b}"
QUANT_TYPE="${QUANT_TYPE:-q4_k_m}"

err() { printf '\033[31m[err]\033[0m %s\n' "$*" >&2; exit 1; }
ok()  { printf '\033[32m[ok]\033[0m  %s\n' "$*"; }
log() { printf '\033[34m[..]\033[0m  %s\n' "$*"; }

[ -d "$HF_DIR" ]                || err "HF model dir not found: $HF_DIR"
[ -f "$HF_DIR/config.json" ]    || err "missing $HF_DIR/config.json"

mkdir -p "$OUT_DIR"

# ---------------------------------------------------------------------------
# 1. Clone or update llama.cpp
# ---------------------------------------------------------------------------

if [ -d "$LCPP_DIR/.git" ]; then
    log "llama.cpp already at $LCPP_DIR — pulling latest"
    git -C "$LCPP_DIR" pull --quiet --ff-only
else
    log "cloning llama.cpp into $LCPP_DIR"
    git clone --depth 1 "$LCPP_REPO" "$LCPP_DIR"
fi

# ---------------------------------------------------------------------------
# 2. Build (CPU only is enough for conversion + quantization + smoke)
# ---------------------------------------------------------------------------

if [ ! -f "$LCPP_DIR/build/bin/llama-quantize" ] || \
   [ ! -f "$LCPP_DIR/build/bin/llama-cli" ]; then
    log "building llama.cpp (this takes 2-5 min on first run)"
    cmake -S "$LCPP_DIR" -B "$LCPP_DIR/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_CURL=OFF \
        >/dev/null
    cmake --build "$LCPP_DIR/build" --config Release \
        --target llama-quantize llama-cli -j \
        >/dev/null
    ok "llama.cpp built"
fi

# Make sure the conversion script has its Python deps
log "ensuring HF→GGUF Python deps installed"
python3 -m pip install --quiet --upgrade \
    "transformers>=4.45" "sentencepiece" "gguf>=0.10" "protobuf>=4" \
    "torch>=2.1" "numpy"

# ---------------------------------------------------------------------------
# 3. Convert HF → GGUF F16 (full precision)
# ---------------------------------------------------------------------------

F16_FILE="$OUT_DIR/${GGUF_NAME}-f16.gguf"
if [ -f "$F16_FILE" ]; then
    log "F16 GGUF already at $F16_FILE — skipping conversion"
else
    log "converting HF → GGUF F16 ($F16_FILE)"
    python3 "$LCPP_DIR/convert_hf_to_gguf.py" "$HF_DIR" \
        --outfile "$F16_FILE" \
        --outtype f16
    ok "F16 GGUF written ($(du -h "$F16_FILE" | cut -f1))"
fi

# ---------------------------------------------------------------------------
# 4. Quantize F16 → Q4_K_M
# ---------------------------------------------------------------------------

QUANT_FILE="$OUT_DIR/${GGUF_NAME}-${QUANT_TYPE}.gguf"
if [ -f "$QUANT_FILE" ]; then
    log "$QUANT_TYPE GGUF already at $QUANT_FILE — skipping quantization"
else
    log "quantizing F16 → ${QUANT_TYPE} ($QUANT_FILE)"
    "$LCPP_DIR/build/bin/llama-quantize" \
        "$F16_FILE" "$QUANT_FILE" "${QUANT_TYPE}"
    ok "${QUANT_TYPE} GGUF written ($(du -h "$QUANT_FILE" | cut -f1))"
fi

# ---------------------------------------------------------------------------
# 5. Smoke prompt — confirm the model loads and replies in Italian
# ---------------------------------------------------------------------------

log "smoke prompt to verify the GGUF loads correctly"
"$LCPP_DIR/build/bin/llama-cli" \
    -m "$QUANT_FILE" \
    -p "Articolo 2086 del codice civile italiano: " \
    -n 128 -t 4 --temp 0.7 --top-p 0.95 --no-display-prompt \
    2>/dev/null | head -20 \
    || err "smoke prompt failed — the GGUF is malformed"

cat <<EOF

================================================================================
 Phase 3 done.
   F16 GGUF:    $F16_FILE
   ${QUANT_TYPE} GGUF:  $QUANT_FILE

 Next: load into the EULLM Engine, or push to HuggingFace Hub:
   huggingface-cli upload eullm/legal-it-7b "$QUANT_FILE" \\
       legal-it-7b-${QUANT_TYPE}.gguf
================================================================================
EOF

#!/usr/bin/env bash
# Install LLaMA-Factory and the training stack into the active Python env.
#
# Idempotent: if LLaMA-Factory is already installed at $LF_DIR, it just
# pulls the latest changes and re-runs `pip install -e .` which is a
# no-op if nothing changed.
#
# Usage:
#   bash forge/scripts/install_training_deps.sh
#
# After running, you can verify with:
#   llamafactory-cli version

set -euo pipefail

LF_DIR="${LF_DIR:-$HOME/LLaMA-Factory}"
LF_REPO="${LF_REPO:-https://github.com/hiyouga/LLaMA-Factory.git}"

err() { printf '\033[31m[err]\033[0m %s\n' "$*" >&2; exit 1; }
ok()  { printf '\033[32m[ok]\033[0m  %s\n' "$*"; }
log() { printf '\033[34m[..]\033[0m  %s\n' "$*"; }

PYTHON="${PYTHON:-python3}"
log "Using $($PYTHON --version)"
$PYTHON -m pip install --quiet --upgrade pip wheel

# 1. Clone or update LLaMA-Factory
if [ -d "$LF_DIR/.git" ]; then
    log "LLaMA-Factory at $LF_DIR — pulling latest"
    git -C "$LF_DIR" pull --quiet --ff-only
else
    log "Cloning LLaMA-Factory into $LF_DIR"
    git clone --depth 1 "$LF_REPO" "$LF_DIR"
fi

# 2. Install LLaMA-Factory in editable mode. The optional extras
#    [torch,metrics,bitsandbytes] were renamed/removed in LLaMA-Factory
#    0.9.5+, so we install the base package and add the missing
#    runtime deps explicitly afterwards.
log "Installing LLaMA-Factory (base)"
(cd "$LF_DIR" && $PYTHON -m pip install --quiet -e ".")

# 3. Add the runtime deps the YAMLs assume:
#    rouge-score + nltk — used by the eval metrics; transformers
#                         warns loudly without them.
# Note: we deliberately do NOT install bitsandbytes / torchao here.
# The YAMLs use plain `optim: adamw_torch` so neither is needed; if
# we ever switch to an 8-bit/4-bit optimizer we'll add the dep AND
# the install line together to avoid the version-skew mess we hit
# with `adamw_torch_8bit` on transformers 4.5x.
log "Installing eval runtime deps (rouge-score, nltk)"
$PYTHON -m pip install --quiet rouge-score nltk

# 3. Sanity check.
if command -v llamafactory-cli >/dev/null; then
    ok "LLaMA-Factory installed: $(llamafactory-cli version 2>&1 | head -1)"
else
    err "llamafactory-cli not on PATH after install"
fi

# 4. Optional: wandb for live training metrics.
if [ "${INSTALL_WANDB:-1}" = "1" ]; then
    log "Installing wandb (optional, run 'wandb disabled' to opt out)"
    $PYTHON -m pip install --quiet wandb
fi

cat <<EOF

================================================================================
 Training stack installed.
   LLaMA-Factory: $LF_DIR

 Next:
   1) Run the pre-flight check (deps + GPU + dataset + tokenizer):
        python forge/scripts/check_training_env.py

   2) When the pre-flight is green, launch the smoke test:
        bash forge/scripts/train.sh \\
            forge/training/configs/smoke_qwen3_1.7b.yaml

 See forge/training/README.md for the full flow.
================================================================================
EOF

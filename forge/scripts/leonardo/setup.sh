#!/usr/bin/env bash
# One-time setup for the legal-it-7b pipeline on Leonardo (CINECA).
# Run ON A LOGIN NODE (internet access) after cloning the repo to $WORK:
#
#   git clone https://github.com/eullm/eullm.git "$WORK/eullm"
#   cd "$WORK/eullm"
#   bash forge/scripts/leonardo/setup.sh
#
# For the dataset download step, export first (skipped when the corpus
# is already at $EULLM_DATA_DIR, e.g. rsync'ed from the workstation):
#   HF_TOKEN    — HF token with read access to the private dataset repo
#   HF_DATASET  — e.g. primoco/legal_it_pretraining
#
# Idempotent: re-running skips whatever is already in place.

set -euo pipefail

err() { printf '\033[31m[err]\033[0m %s\n' "$*" >&2; exit 1; }
ok()  { printf '\033[32m[ok]\033[0m  %s\n' "$*"; }
log() { printf '\033[34m[..]\033[0m  %s\n' "$*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

# -----------------------------------------------------------------------------
# 1. Python venv on $WORK
# -----------------------------------------------------------------------------

if [ ! -f "$EULLM_VENV/bin/activate" ]; then
    log "creating venv at $EULLM_VENV"
    PYBIN="$(command -v python3.11 || command -v python3.10 || command -v python3)"
    "$PYBIN" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
        || err "need Python >= 3.10 — try 'module avail python' and load one"
    "$PYBIN" -m venv "$EULLM_VENV"
    # shellcheck disable=SC1091
    source "$EULLM_VENV/bin/activate"
else
    ok "venv already at $EULLM_VENV"
fi
log "using $(python --version) at $(command -v python)"

# -----------------------------------------------------------------------------
# 2. Training stack: LLaMA-Factory + deps (into the venv, LF on $WORK)
# -----------------------------------------------------------------------------

LF_DIR="${LF_DIR:-$WORK/LLaMA-Factory}" PYTHON=python \
    bash "$EULLM_REPO/forge/scripts/install_training_deps.sh"

# Torch built for CUDA 12, not the CUDA 13 wheels PyPI now serves by
# default. Leonardo's driver reports 12020 (r535, CUDA 12.2) and a CUDA
# 13 build refuses to initialise on it — the first setup here installed
# torch 2.14.0+cu130 and every job failed the pre-flight with "CUDA not
# available: The NVIDIA driver on your system is too old". cu126 carries
# the same torch version, and CUDA minor-version compatibility means a
# 12.x build runs on any r525+ driver. This is the same driver floor
# that forced the engine's data-center binary from CUDA 13.1 to 12.4;
# it is a property of the machine, not an accident.
log "installing torch for CUDA 12 (Leonardo driver is r535 / CUDA 12.2)"
python -m pip install --quiet --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu126 \
    torch torchvision torchaudio
ok "torch installed from the cu126 index"

# Multi-GPU stack for the ZeRO-3 Phase-1 config + distill extras.
#
# peft is pinned, not floored: LLaMA-Factory 0.9.6 requires
# peft<=0.18.1,>=0.18.0, and "peft>=0.12" resolved to 0.20.0, which pip
# installed while printing a dependency-conflict error that the script
# then ignored. Widen this only together with the LLaMA-Factory pin.
log "installing multi-GPU / distillation deps"
python -m pip install --quiet --upgrade \
    "deepspeed>=0.19.6" \
    "bitsandbytes>=0.43"
python -m pip install --quiet "peft==0.18.1"
ok "deepspeed + peft + bitsandbytes installed"

# -----------------------------------------------------------------------------
# 3. Prefetch models into $HF_HOME (compute nodes are offline)
# -----------------------------------------------------------------------------

python "$EULLM_REPO/forge/scripts/leonardo/prefetch_models.py"

# -----------------------------------------------------------------------------
# 4. Dataset onto $WORK
# -----------------------------------------------------------------------------

if [ -f "$EULLM_DATA_DIR/train.jsonl" ] && [ -f "$EULLM_DATA_DIR/val.jsonl" ]; then
    ok "dataset already at $EULLM_DATA_DIR"
elif [ -n "${HF_TOKEN:-}" ] && [ -n "${HF_DATASET:-}" ]; then
    log "downloading $HF_DATASET → $EULLM_DATA_DIR"
    mkdir -p "$EULLM_DATA_DIR"
    export TARBALL_NAME="${TARBALL_NAME:-legal_it_pretraining.tar.gz}"
    python - <<PY
import os
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id=os.environ["HF_DATASET"],
    filename=os.environ.get("TARBALL_NAME", "legal_it_pretraining.tar.gz"),
    repo_type="dataset",
    local_dir=os.environ["EULLM_DATA_DIR"],
)
print(f"   downloaded {path}")
PY
    tar -xzf "$EULLM_DATA_DIR/$TARBALL_NAME" -C "$EULLM_DATA_DIR"
    rm -f "$EULLM_DATA_DIR/$TARBALL_NAME"
    ok "dataset extracted to $EULLM_DATA_DIR"
else
    err "no dataset at $EULLM_DATA_DIR and HF_TOKEN/HF_DATASET not set —
     either export them and re-run, or copy the corpus manually:
       rsync -av --progress train.jsonl val.jsonl \\
           <user>@login.leonardo.cineca.it:$EULLM_DATA_DIR/"
fi

# -----------------------------------------------------------------------------
# 5. llama.cpp for Phase 3 (clone + CPU build now; jobs run it offline)
# -----------------------------------------------------------------------------

if [ ! -d "$LCPP_DIR/.git" ]; then
    log "cloning llama.cpp into $LCPP_DIR (Phase 3 uses it offline)"
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "$LCPP_DIR"
else
    ok "llama.cpp already at $LCPP_DIR"
fi

# -----------------------------------------------------------------------------
# 6. Login-node pre-flight (no GPU here — jobs re-check with GPUs)
# -----------------------------------------------------------------------------

python "$EULLM_REPO/forge/scripts/check_training_env.py" \
    --cpu-only --multi-gpu --offline --data-dir "$EULLM_DATA_DIR"

cat <<EOF

================================================================================
 Leonardo setup complete. Next (see docs/leonardo-runbook.md):

   export EULLM_ACCOUNT=<project account from 'saldo -b'>
   cd "\$EULLM_RUN_DIR"

   # 1. Smoke test (1 GPU, debug QOS, ~15 min):
   bash "\$EULLM_REPO/forge/scripts/leonardo/submit_chain.sh" \\
       "\$EULLM_REPO/forge/scripts/leonardo/sbatch_smoke.slurm"

   # 2. Phase 1 (1 node / 4 GPUs, chained over the 24 h walltime cap):
   bash "\$EULLM_REPO/forge/scripts/leonardo/submit_chain.sh" \\
       "\$EULLM_REPO/forge/scripts/leonardo/sbatch_phase1.slurm" 2
================================================================================
EOF

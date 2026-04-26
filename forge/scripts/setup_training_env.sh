#!/usr/bin/env bash
# Bootstrap a fresh rented GPU instance for the legal-it-7b training run.
#
# Run this on a fresh Linux instance with NVIDIA drivers + CUDA already
# installed (most rented GPU images ship that way). The script clones
# the repo, sets up a conda/uv env, installs deps, downloads the
# anonymized training corpus from a private HF Hub dataset, and prints
# the launch command for the next phase.
#
# Required env vars (set BEFORE running):
#   HF_TOKEN        — HuggingFace access token with repo read permission.
#   HF_DATASET      — Private dataset repo, e.g. 'primoco/legal_it_pretraining'.
#   WANDB_API_KEY   — (optional) for live training metrics.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/eullm/eullm/feat/legal-it/forge/scripts/setup_training_env.sh \
#       | HF_TOKEN=hf_xxx HF_DATASET=primoco/legal_it_pretraining bash
#
# Idempotent: re-running it will pick up where the previous run left off.

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/eullm/eullm.git}"
BRANCH="${BRANCH:-feat/legal-it}"
WORKDIR="${WORKDIR:-$HOME/eullm}"
DATA_DIR="${DATA_DIR:-$HOME/datasets/legal_it}"
TARBALL_NAME="${TARBALL_NAME:-legal_it_pretraining.tar.gz}"

err() { printf '\033[31m[err]\033[0m %s\n' "$*" >&2; exit 1; }
ok()  { printf '\033[32m[ok]\033[0m  %s\n' "$*"; }
log() { printf '\033[34m[..]\033[0m  %s\n' "$*"; }

# -----------------------------------------------------------------------------
# 0. Sanity checks
# -----------------------------------------------------------------------------

[ -n "${HF_TOKEN:-}" ]   || err "HF_TOKEN must be set (HF read-permission token)"
[ -n "${HF_DATASET:-}" ] || err "HF_DATASET must be set (e.g. primoco/legal_it_pretraining)"

command -v nvidia-smi >/dev/null || err "nvidia-smi not found — wrong instance image?"
log "GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

# -----------------------------------------------------------------------------
# 1. Clone or update the repo
# -----------------------------------------------------------------------------

if [ -d "$WORKDIR/.git" ]; then
    log "Repo already cloned at $WORKDIR — pulling latest of $BRANCH"
    git -C "$WORKDIR" fetch --quiet origin "$BRANCH"
    git -C "$WORKDIR" checkout --quiet "$BRANCH"
    git -C "$WORKDIR" pull --quiet --ff-only origin "$BRANCH"
else
    log "Cloning $REPO_URL ($BRANCH) into $WORKDIR"
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$WORKDIR"
fi
ok "repo at $WORKDIR ($(git -C "$WORKDIR" rev-parse --short HEAD))"

# -----------------------------------------------------------------------------
# 2. Python env
# -----------------------------------------------------------------------------

# Most rented instances expose either a system Python or a conda. We do
# NOT create a venv to keep things simple — the instance is single-purpose.
PYTHON="${PYTHON:-python3}"
log "Using $($PYTHON --version)"
$PYTHON -m pip install --quiet --upgrade pip
log "Installing eullm-forge[legal] from local checkout"
$PYTHON -m pip install --quiet -e "$WORKDIR/forge[legal]"

# Training-time deps that are heavier and version-sensitive: pin via the
# install command instead of dragging them into pyproject.toml.
log "Installing training stack (transformers, peft, bitsandbytes, accelerate, datasets)"
$PYTHON -m pip install --quiet --upgrade \
    "transformers>=4.45" \
    "peft>=0.12" \
    "accelerate>=0.34" \
    "bitsandbytes>=0.43" \
    "datasets>=2.20" \
    "trl>=0.10"

ok "deps installed"

# -----------------------------------------------------------------------------
# 3. HuggingFace login
# -----------------------------------------------------------------------------

log "Authenticating to HuggingFace Hub"
$PYTHON -c "
from huggingface_hub import login
import os
login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)
print('   logged in')
"

if [ -n "${WANDB_API_KEY:-}" ]; then
    log "Configuring wandb"
    $PYTHON -m pip install --quiet wandb
    wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 || true
    ok "wandb authenticated"
fi

# -----------------------------------------------------------------------------
# 4. Pull the training corpus
# -----------------------------------------------------------------------------

mkdir -p "$DATA_DIR"

if [ -f "$DATA_DIR/train.jsonl" ] && [ -f "$DATA_DIR/val.jsonl" ]; then
    ok "dataset already present at $DATA_DIR — skipping download"
else
    log "Downloading $HF_DATASET → $DATA_DIR"
    $PYTHON -c "
from huggingface_hub import hf_hub_download
import os
path = hf_hub_download(
    repo_id=os.environ['HF_DATASET'],
    filename=os.environ.get('TARBALL_NAME', 'legal_it_pretraining.tar.gz'),
    repo_type='dataset',
    local_dir=os.environ['DATA_DIR'],
    local_dir_use_symlinks=False,
)
print(f'   downloaded {path}')
" TARBALL_NAME="$TARBALL_NAME" DATA_DIR="$DATA_DIR"

    log "Extracting tarball"
    tar -xzf "$DATA_DIR/$TARBALL_NAME" -C "$DATA_DIR"
    rm -f "$DATA_DIR/$TARBALL_NAME"
    ok "dataset extracted"
fi

train_n=$(wc -l < "$DATA_DIR/train.jsonl" 2>/dev/null || echo "?")
val_n=$(wc -l < "$DATA_DIR/val.jsonl" 2>/dev/null || echo "?")
ok "dataset ready: $train_n train + $val_n val records under $DATA_DIR"

# -----------------------------------------------------------------------------
# 5. Done — print next-step hint
# -----------------------------------------------------------------------------

cat <<EOF

================================================================================
 Setup complete.
 Repo:    $WORKDIR
 Dataset: $DATA_DIR

 Next step depends on which phase you are running. See:
     $WORKDIR/docs/legal-it-7b-strategy.md
 for the pipeline overview.

 Phase 1 (continued pre-training of the teacher) launcher will live at
     $WORKDIR/forge/scripts/train_continued_pt.sh
 once it has been authored.
================================================================================
EOF

# Shared environment for EULLM work on Leonardo (CINECA) — EuroHPC
# allocation EHPC-AIF-2026PG01-1147 (Leonardo Booster, 4x A100 64 GB
# per node).
#
# Source this from login shells AND from sbatch scripts:
#
#   EULLM_REPO="${EULLM_REPO:-$WORK/eullm}"
#   source "$EULLM_REPO/forge/scripts/leonardo/env.sh"
#
# Everything lives on $WORK (project storage, 1 TB default quota):
# $HOME is only 50 GB and $SCRATCH is purged periodically. Override any
# EULLM_* variable before sourcing to relocate.
#
# Set once in ~/.bashrc on Leonardo (value from `saldo -b`):
#   export EULLM_ACCOUNT=<your_project_account>

: "${WORK:?WORK is not set — this file is meant to be sourced on a Leonardo node}"

export EULLM_REPO="${EULLM_REPO:-$WORK/eullm}"
export EULLM_VENV="${EULLM_VENV:-$WORK/eullm_venv}"
export EULLM_RUN_DIR="${EULLM_RUN_DIR:-$WORK/eullm_runs/legal_it}"
export EULLM_DATA_DIR="${EULLM_DATA_DIR:-$WORK/datasets/legal_it}"

# HF model cache on project storage — prefetch_models.py fills it from a
# login node; jobs read it offline.
export HF_HOME="${HF_HOME:-$WORK/hf_home}"

# llama.cpp checkout for Phase 3 (cloned/built on a login node).
export LCPP_DIR="${LCPP_DIR:-$WORK/llama.cpp}"

# Compute nodes have no outbound network: force the HF stack offline
# inside jobs so a cache miss fails fast instead of hanging on retries.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
fi

# sbatch reads SBATCH_ACCOUNT, so no per-submission -A flag is needed.
if [ -n "${EULLM_ACCOUNT:-}" ]; then
    export SBATCH_ACCOUNT="$EULLM_ACCOUNT"
fi

# Lmod: newest python module for the venv's interpreter (best effort —
# adjust after checking `module avail python` if the default is too old).
if command -v module >/dev/null 2>&1; then
    module try-load python 2>/dev/null || true
fi

if [ -f "$EULLM_VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$EULLM_VENV/bin/activate"
fi

mkdir -p "$EULLM_RUN_DIR/logs"

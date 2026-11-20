#!/bin/bash
# The GPU check of a new Forge venv (venv_smoke.py), before it replaces the
# one every job uses. One A100, about 15-25 minutes, most of it loading the
# 30B-A3B MoE in 8 bits from Lustre (line A of the Fast Lane plan).
#
#   EULLM_VENV=$WORK/eullm_venv-latest sbatch -A $EULLM_ACCOUNT \
#       forge/scripts/leonardo/sbatch_venv_smoke.slurm
#
# The log ends with SMOKE PASSED or SMOKE FAILED and the list of checks.
#   SMOKE_MODELS  the HF cache with the models (default: the project's fast scratch)
#   SMOKE_SKIP    checks to skip, comma-separated (e.g. moe)
#
#SBATCH --job-name=eullm-venv-smoke
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out

set -uo pipefail

BASE="${EULLM_BASE:-/leonardo_work/AIFAC_F02_1400}"
VENV="${EULLM_VENV:?set EULLM_VENV to the venv to check}"
REPO="${EULLM_REPO:-$BASE/eullm}"
FAST="${SMOKE_FAST:-/leonardo_scratch/fast/AIFAC_F02_1400}"

[ -f "$VENV/bin/activate" ] || { echo "[smoke] FATAL: no venv at $VENV" >&2; exit 1; }
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_CACHE="${SMOKE_MODELS:-$FAST/models/hub}"

echo "[smoke] node $(hostname), job ${SLURM_JOB_ID:-?}, venv $VENV"
nvidia-smi | head -4

ARGS=(--small Qwen/Qwen3-4B-Instruct-2507
      --qwen35 "$FAST/models/Qwen3.5-4B"
      --moe Qwen/Qwen3-30B-A3B-Instruct-2507
      --skip "${SMOKE_SKIP:-}")
LIBS="${GRPO_LIBS:-$BASE/pylibs/grpo}"
[ -d "$LIBS" ] && ARGS+=(--trl-path "$LIBS")

python "$REPO/forge/scripts/venv_smoke.py" "${ARGS[@]}"

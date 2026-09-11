#!/bin/bash
# P1 — can vLLM score the corpus with the teacher, and how fast?
#
# The first pilot job for the v1.1 pipeline, and the one that decides whether
# design A of ADR-001 is buildable at all. It runs beside a Phase-1 chain
# without touching it: different checkout, different run directory, and a
# venv of its own.
#
# **The venv is not a detail.** vLLM pins its own PyTorch. Installing it into
# the venv a running chain uses would change that chain's torch the moment its
# next link restarts, which is the most efficient way to lose days of compute.
# Prepare it on a login node, where there is internet:
#
#   python -m venv "$WORK/v11_venv"
#   "$WORK/v11_venv/bin/pip" install --upgrade pip
#   "$WORK/v11_venv/bin/pip" install 'vllm==0.28.0'
#
#   cd "$WORK/eullm_runs/v11_pilot"
#   sbatch "$WORK/eullm-v11/forge/scripts/leonardo/sbatch_p1_logprob_probe.slurm"
#
# Exits non-zero when P1 is falsified — which is a result, not a malfunction.
# The report JSON carries the numbers either way.
#
#SBATCH --job-name=eullm-p1-logprobs
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=450G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail

# Spelled out rather than taken from EULLM_REPO: env.sh sets that to whatever
# tree a frozen run is pinned to, and sbatch exports the submitting shell's
# environment. A pilot job must run from the pilot checkout.
BASE="${EULLM_BASE:-$WORK}"
export EULLM_REPO="${EULLM_PILOT_REPO:-$BASE/eullm-v11}"
export EULLM_RUN_DIR="${EULLM_PILOT_RUN_DIR:-$BASE/eullm_runs/v11_pilot}"
PILOT_VENV="${EULLM_PILOT_VENV:-$BASE/v11_venv}"

# shellcheck disable=SC1091
source "$EULLM_REPO/forge/scripts/leonardo/env.sh"
mkdir -p "$EULLM_RUN_DIR/logs"
cd "$EULLM_RUN_DIR" || exit 1

if [ ! -x "$PILOT_VENV/bin/python" ]; then
    echo "[p1] FATAL: no venv at $PILOT_VENV — see the header of this file." >&2
    exit 1
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# shellcheck disable=SC1091
source "$EULLM_REPO/forge/scripts/leonardo/heartbeat.sh"
heartbeat_start

echo "[p1] commit $(git -C "$EULLM_REPO" rev-parse --short HEAD)"
"$PILOT_VENV/bin/python" -c 'import vllm, torch; print(f"[p1] vllm {vllm.__version__}, torch {torch.__version__}")'

"$PILOT_VENV/bin/python" "$EULLM_REPO/forge/scripts/pilot/probe_teacher_logprobs.py" \
    --model Qwen/Qwen3-30B-A3B-Base \
    --data "$EULLM_DATA_DIR/val.jsonl" \
    --samples 32 \
    --top-k 64 \
    --seq-len 1024 \
    --tensor-parallel-size 4 \
    --report "$EULLM_RUN_DIR/p1_prompt_logprobs_k64.json"
STATUS=$?

# An instrument must not fail the job it measures.
python3 "$EULLM_REPO/forge/scripts/leonardo/queue_stats.py" 2026-09-02 \
    --json "$BASE/eullm_runs/qstats/queue_stats_$(date +%Y%m%d_%H%M).json" || true

# Pass the probe's verdict up: non-zero means P1 was falsified, which the
# report records as a finding rather than as a broken run.
exit "$STATUS"

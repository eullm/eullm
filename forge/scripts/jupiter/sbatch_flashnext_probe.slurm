#!/bin/bash
# Qwen3.8-Flash-Next on one node (forge/scripts/flashnext_probe.py): loading
# with the n-gram table in the Grace memory, three answers, a few steps
# training the table alone. One hour; needs the model in the offline cache
# (snapshot_download on a login node, ~360 GB).
#
#     eullm && sbatch --account e-aif-2026pg01-1434 \
#         --output=$PROJECT/$USER/forge/runs/logs/flashnext-probe-%j.out \
#         forge/scripts/jupiter/sbatch_flashnext_probe.slurm

#SBATCH --job-name=forge-flashnext-probe
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00

set -uo pipefail
EULLM_REPO="${EULLM_REPO:-$SLURM_SUBMIT_DIR}"
# Its linear attention runs on flash-linear-attention, which refuses the
# module's Triton on Hopper (wrong gradients, fla #640; job 2686808): the
# Triton 3.8 of libs-qwen35 (setup_qwen35_libs.sh) goes first.
export FORGE_EXTRA_LIBS="${FORGE_EXTRA_LIBS:-$PROJECT/$USER/forge/libs-qwen35}"
# shellcheck source=env.sh
source "$EULLM_REPO/forge/scripts/jupiter/env.sh"
MODEL="${FN_MODEL:-Qwen/Qwen3.8-Flash-Next}"
TEXTS="${FN_TEXTS:-$FORGE_DATA_DIR/memory/codes/program.jsonl}"
echo "=== Flash-Next probe: job ${SLURM_JOB_ID:-local} on $(hostname), $(date -Is) ==="
python -c "import sys; from transformers import AutoConfig; AutoConfig.from_pretrained(sys.argv[1], local_files_only=True)" "$MODEL" ||
    { echo "[!!] $MODEL is not in the offline cache: snapshot_download on a login node"; exit 1; }
[ -s "$TEXTS" ] || { echo "[!!] missing $TEXTS"; exit 1; }
free -g | head -2
# shellcheck disable=SC2086
python "$EULLM_REPO/forge/scripts/flashnext_probe.py" --model "$MODEL" --texts "$TEXTS" ${FN_ARGS:-}
RC=$?
echo "=== done $(date -Is) (exit $RC) ==="
exit "$RC"

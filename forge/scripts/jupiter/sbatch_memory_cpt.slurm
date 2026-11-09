#!/bin/bash
# E1-mini on JUPITER (legal-it roadmap, phase 2): the articles of the nine
# codes trained into the backbone, continued pretraining through LoRA with a
# replay of the post-training and general Italian (forge/scripts/memory_cpt.py).
# One node: the data prepared once on the CPU, then LoRA on the four GPUs
# (accelerate, one process a GPU); a link stops 20 minutes before its end and
# the next resumes from the last checkpoint. The link that finishes merges
# the adapter into $OUT/merged, for the vertical (GGUF) and the closed book.
#
#     eullm && prev=""; for i in 1 2 3; do prev=$(sbatch --parsable \
#         --account e-aif-2026pg01-1434 ${prev:+--dependency=afterany:$prev} \
#         --output=$PROJECT/$USER/forge/runs/logs/memory-cpt-%j.out \
#         forge/scripts/jupiter/sbatch_memory_cpt.slurm); echo "$i $prev"; done
#
# MC_SMOKE=1 (a 40-minute job) prepares the data, trains 30 steps apart and
# prints the seconds a step and the hours the run will take; the links go
# behind it with afterok.
#
# Settable: MC_MODEL (default the 9B candidate g9r-r2, merged), MC_OUT
# (default $FORGE_RUN_DIR/memory/e1-g9), MC_CORPUS, MC_REPLAY, MC_GENERAL
# (files, space-separated), MC_PREPARE_ARGS (memory_cpt.py prepare: epochs,
# shares) and MC_TRAIN_ARGS (train: rank, lr, grad-accum). Needs
# general-it.jsonl from prefetch_general_it.py, run once on a login node.
# Counts only in the log, never a text.

#SBATCH --job-name=forge-memory-cpt
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00

set -uo pipefail
EULLM_REPO="${EULLM_REPO:-$SLURM_SUBMIT_DIR}"
export FORGE_EXTRA_LIBS="${FORGE_EXTRA_LIBS:-$PROJECT/$USER/forge/libs-qwen35}"
# shellcheck source=env.sh
source "$EULLM_REPO/forge/scripts/jupiter/env.sh"
R="$FORGE_RUN_DIR" D="$FORGE_DATA_DIR"
MODEL="${MC_MODEL:-$R/grpo/g9r-r2/merged}"
OUT="${MC_OUT:-$R/memory/e1-g9}"
read -r -a CORPUS <<<"${MC_CORPUS:-$(ls "$D"/memory/codes/program.jsonl "$D"/memory/codes/rewrites*.jsonl 2>/dev/null | grep -v rejects | tr '\n' ' ')}"
read -r -a REPLAY <<<"${MC_REPLAY:-$D/openbook/pairs.jsonl $D/premise/sft-steps.jsonl}"
read -r -a GENERAL <<<"${MC_GENERAL:-$D/memory/general-it.jsonl}"
JOB="${SLURM_JOB_ID:-local}"
T0=$(date +%s)

echo "=== E1-mini, memory by continued pretraining: job $JOB on $(hostname), $(date -Is) ==="
if [ -f "$OUT/merged/config.json" ]; then echo "[ok] merged already: $OUT/merged"; exit 0; fi
bad=0
[ -f "$MODEL/config.json" ] || { echo "[!!] no backbone at $MODEL"; bad=1; }
[ "${#CORPUS[@]}" -gt 0 ] || { echo "[!!] no memory corpus in $D/memory/codes"; bad=1; }
for f in "${CORPUS[@]}" "${REPLAY[@]}" "${GENERAL[@]}"; do
    [ -s "$f" ] || { echo "[!!] missing $f"; bad=1; }
done
python -c "import torch, peft, transformers" 2>/dev/null ||
    { echo "[!!] torch, peft or transformers does not import (module load PyTorch?)"; bad=1; }
[ "$bad" = 0 ] || exit 1
mkdir -p "$OUT" "$R/power"
echo "  backbone $MODEL"
echo "  corpus ${#CORPUS[@]} files, replay ${#REPLAY[@]}, general ${#GENERAL[@]} -> $OUT"

# shellcheck disable=SC2086
python "$EULLM_REPO/forge/scripts/memory_cpt.py" prepare --model "$MODEL" --out "$OUT" \
    --corpus "${CORPUS[@]}" --replay "${REPLAY[@]}" --general "${GENERAL[@]}" \
    ${MC_PREPARE_ARGS:-} || { echo "[!!] prepare failed"; exit 1; }

POWER="$R/power/memory-cpt-$JOB.csv"
nvidia-smi --query-gpu=timestamp,index,power.draw,utilization.gpu,memory.used \
    --format=csv,noheader,nounits -l 5 >"$POWER" 2>/dev/null &
SAMPLER=$!
LIMIT_S=$(job_time_limit_s)
TRAIN_S=$(( ${LIMIT_S:-21600} - ($(date +%s) - T0) - 1200 ))
RC=0
if [ "${MC_SMOKE:-0}" = 1 ]; then
    # shellcheck disable=SC2086
    accelerate launch --num_processes 4 --num_machines 1 --mixed_precision bf16 \
        --dynamo_backend no "$EULLM_REPO/forge/scripts/memory_cpt.py" \
        train --model "$MODEL" --out "$OUT" --max-steps 30 ${MC_TRAIN_ARGS:-}
    RC=$?
    kill "$SAMPLER" 2>/dev/null
    wait "$SAMPLER" 2>/dev/null
    python "$EULLM_REPO/forge/scripts/jupiter/energy.py" "$POWER" --interval 5
    echo "=== smoke test done $(date -Is) (exit $RC) ==="
    exit "$RC"
fi
if [ ! -f "$OUT/adapter/adapter_config.json" ]; then
    [ "$TRAIN_S" -gt 600 ] || { echo "[!!] no time left to train ($TRAIN_S s)"; kill "$SAMPLER"; exit 1; }
    echo "  training at most $TRAIN_S s"
    # shellcheck disable=SC2086
    timeout --signal=TERM "$TRAIN_S" accelerate launch --num_processes 4 --num_machines 1 \
        --mixed_precision bf16 --dynamo_backend no "$EULLM_REPO/forge/scripts/memory_cpt.py" \
        train --model "$MODEL" --out "$OUT" ${MC_TRAIN_ARGS:-}
    RC=$?
fi
kill "$SAMPLER" 2>/dev/null
wait "$SAMPLER" 2>/dev/null
python "$EULLM_REPO/forge/scripts/jupiter/energy.py" "$POWER" --interval 5
if [ ! -f "$OUT/adapter/adapter_config.json" ]; then
    if [ "$RC" = 124 ]; then
        echo "[ok] time is up: the next link resumes from $(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)"
        exit 0
    fi
    echo "[!!] memory_cpt.py train: exit $RC, no adapter"
    exit "$RC"
fi
python - "$MODEL" "$OUT/adapter" "$OUT/merged" <<'PY' || { echo "[!!] merge failed"; exit 1; }
import sys
from eullm_forge.identity import merge_identity_adapter
merge_identity_adapter(sys.argv[1], sys.argv[2], sys.argv[3])
PY
echo "[ok] merged into $OUT/merged ($(( ($(date +%s) - T0) / 60 )) min)"
echo "=== done $(date -Is) ==="

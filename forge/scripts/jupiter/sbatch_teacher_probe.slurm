#!/bin/bash
# The first Forge numbers on GH200, before any corpus is on JUPITER
# (teacher_probe.py has the details), on one node:
#
#   1. two distillation micro-batch loops side by side, the shape of
#      deliverable 3: GPUs 0-1 with cores 0-143, GPUs 2-3 with cores 144-287
#      (probe, 14-10-2026: GPU i sits by cores 72i-72i+71), the teacher in
#      BF16 alone on the first GPU of each pair (deliverable 2);
#   2. the teacher generating with plain Transformers on GPU 0, batches of
#      1, 8 and 32 (deliverable 5's baseline);
#   3. the node's GPU power sampled every second throughout, to a CSV: Slurm
#      keeps no energy here, so this is how deliverable 9 is measured.
#
# Also the first job to run the Forge venv on a compute node: a broken venv
# fails it in the first minute.
#
#     eullm && sbatch --account e-aif-2026pg01-1434 \
#         --output="$PROJECT/$USER/forge/runs/logs/teacher-probe-%j.out" \
#         forge/scripts/jupiter/sbatch_teacher_probe.slurm
#
# PROBE_TEACHER and PROBE_STUDENT probe another pair (deliverable 7), e.g.
#
#     eullm && PROBE_TEACHER=Qwen/Qwen3.5-35B-A3B-Base PROBE_STUDENT=Qwen/Qwen3.5-4B-Base \
#         sbatch --account e-aif-2026pg01-1434 \
#         --output="$PROJECT/$USER/forge/runs/logs/teacher-probe-%j.out" \
#         forge/scripts/jupiter/sbatch_teacher_probe.slurm
#
# Whole node: 4 GPU-hours an hour. Two teacher loads at once from GPFS, 30
# micro-batches each, then three generations: 20-35 minutes, 1.5-2.5 GPU-hours.

#SBATCH --job-name=forge-teacher-probe
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:50:00

set -uo pipefail
# sbatch runs a copy of this file from Slurm's spool: the checkout is where
# the job was submitted from.
EULLM_REPO="${EULLM_REPO:-$SLURM_SUBMIT_DIR}"
# shellcheck source=env.sh
source "$EULLM_REPO/forge/scripts/jupiter/env.sh"
PROBE="$EULLM_REPO/forge/scripts/jupiter/teacher_probe.py"
OUT="$FORGE_RUN_DIR/teacher-probe-${SLURM_JOB_ID:-local}"
MODELS=()
[ -n "${PROBE_TEACHER:-}" ] && MODELS+=(--teacher "$PROBE_TEACHER")
[ -n "${PROBE_STUDENT:-}" ] && MODELS+=(--student "$PROBE_STUDENT")
mkdir -p "$OUT"

echo "=== Forge teacher probe: job ${SLURM_JOB_ID:-local} on $(hostname), $(date -Is) ==="
echo "  python $(command -v python) ($(python --version 2>&1)), HF_HOME $HF_HOME, offline ${HF_HUB_OFFLINE:-0}"
nvidia-smi --query-gpu=index,name,memory.total,power.limit --format=csv,noheader | sed 's/^/  /'

# What the probe imports, before a GPU is used: a venv that does not import
# stops the job here, with the path Python searched.
if ! python -c "import transformers, peft, accelerate, datasets; print('  imports: transformers', transformers.__version__, 'peft', peft.__version__)"; then
    echo "[!!] the venv does not import on this node; PYTHONPATH:"
    tr ':' '\n' <<<"${PYTHONPATH:-}" | sed 's/^/    /'
    exit 1
fi
echo "  pair: ${PROBE_TEACHER:-the default teacher} -> ${PROBE_STUDENT:-the default student}"
# Qwen3.5's linear-attention layers have fast kernels only with these
# packages; without them Transformers falls back to plain torch, slower.
python -c "import fla" 2>/dev/null && echo "  flash-linear-attention: present" || echo "  flash-linear-attention: absent"
python -c "import causal_conv1d" 2>/dev/null && echo "  causal-conv1d: present" || echo "  causal-conv1d: absent"

# Power, every second, all four GPUs, until the end of the job.
nvidia-smi --query-gpu=timestamp,index,power.draw,utilization.gpu,memory.used \
    --format=csv,noheader,nounits -l 1 >"$OUT/power.csv" 2>/dev/null &
SAMPLER=$!
phase() { echo "$(date +%s) $1" >>"$OUT/phases.txt"; echo; echo "=== $1 ($(date +%T)) ==="; }

phase "1. two micro-batch loops side by side"
CUDA_VISIBLE_DEVICES=0,1 taskset -c 0-143 python "$PROBE" step "${MODELS[@]}" >"$OUT/step-a.log" 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=2,3 taskset -c 144-287 python "$PROBE" step "${MODELS[@]}" >"$OUT/step-b.log" 2>&1 &
B=$!
wait "$A"; RA=$?
wait "$B"; RB=$?
for x in a b; do
    echo "--- pair $x"
    grep -E '^RESULT|^torch|fast path|newly initialized|MISSING|UNEXPECTED' "$OUT/step-$x.log" | sort -u | sed 's/^/  /'
done
if [ "$RA" -ne 0 ] || [ "$RB" -ne 0 ]; then
    echo "[!!] a loop failed (exit $RA and $RB): the end of its log"
    for x in a b; do tail -15 "$OUT/step-$x.log" | sed 's/^/    /'; done
fi

phase "2. the teacher generating, plain Transformers, GPU 0"
CUDA_VISIBLE_DEVICES=0 taskset -c 0-71 python "$PROBE" generate "${MODELS[@]}" >"$OUT/generate.log" 2>&1 ||
    { echo "[!!] generation failed: the end of its log"; tail -15 "$OUT/generate.log" | sed 's/^/    /'; }
grep -E '^RESULT' "$OUT/generate.log" | sed 's/^/  /'

phase "end"
kill "$SAMPLER" 2>/dev/null
wait "$SAMPLER" 2>/dev/null
echo
echo "=== energy by phase, from $OUT/power.csv (all four GPUs, Wh) ==="
python - "$OUT/power.csv" "$OUT/phases.txt" <<'PY'
import csv, sys
from datetime import datetime

samples = []
for row in csv.reader(open(sys.argv[1], encoding="utf-8")):
    try:
        t = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
        samples.append((t, float(row[2])))
    except (ValueError, IndexError):
        continue
phases = [(int(l.split()[0]), l.split(maxsplit=1)[1].strip())
          for l in open(sys.argv[2], encoding="utf-8")]
for (start, name), (end, _) in zip(phases, phases[1:]):
    ws = [w for t, w in samples if start <= t < end]  # one sample per GPU per second
    print(f"  {name:<46} {(end - start) / 60:5.1f} min  {sum(ws) / 3600:7.1f} Wh"
          f"  (mean {sum(ws) / max(1, end - start):6.0f} W for the node's GPUs)")
PY
echo
echo "=== done $(date -Is) ==="

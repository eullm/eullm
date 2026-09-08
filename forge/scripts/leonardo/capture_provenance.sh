#!/usr/bin/env bash
# Snapshot everything needed to reproduce a run, into the run's own directory.
#
# A job log is not an artifact. It lives in a directory nobody backs up, it is
# named after a SLURM id that means nothing in six months, and the pieces that
# matter — the resolved config, the library versions, the commit — are
# scattered through thousands of lines of framework chatter. This copies them
# next to the checkpoints, where the model they describe is.
#
# Run it while the job is alive, or at least before the log is rotated away.
#
#   bash forge/scripts/leonardo/capture_provenance.sh 56803262 \
#       ./checkpoints/qwen3_30b_a3b_legal_it_continued_pt
#
# The second argument defaults to the Phase-1 output directory.

set -uo pipefail

JOBID="${1:?Usage: $0 <jobid> [output_dir]}"
OUT="${2:-./checkpoints/qwen3_30b_a3b_legal_it_continued_pt}"
REPO="${EULLM_REPO:-$WORK/eullm}"
RUN_DIR="${EULLM_RUN_DIR:-$PWD}"

DEST="$OUT/provenance"
mkdir -p "$DEST" || { echo "cannot write $DEST" >&2; exit 2; }

LOG="$(compgen -G "$RUN_DIR/logs/*-${JOBID}.out" | head -1 || true)"
[ -n "$LOG" ] || echo "[warn] no log found for job $JOBID" >&2

{
    echo "# Provenance for job $JOBID"
    echo "captured_at: $(date -Is)"
    echo "captured_on: $(hostname)"
    echo
    echo "## Git"
    echo "commit: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo '?')"
    echo "describe: $(git -C "$REPO" describe --always --dirty 2>/dev/null || echo '?')"
    echo "dirty_files:"
    git -C "$REPO" status --porcelain 2>/dev/null | sed 's/^/  /' || true
    echo
    echo "## SLURM"
    sacct -j "$JOBID" \
        --format=JobID%16,JobName%22,State,ExitCode,Elapsed,Start,End,NodeList%20,AllocTRES%40 \
        2>/dev/null || echo "sacct unavailable"
    echo
    echo "## Versions"
    python - <<'PY' 2>/dev/null || echo "python probe failed"
import importlib
for name in ("torch", "transformers", "peft", "accelerate", "deepspeed",
             "bitsandbytes", "datasets", "llamafactory"):
    try:
        m = importlib.import_module(name)
        print(f"{name}: {getattr(m, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"{name}: not importable ({type(exc).__name__})")
try:
    import torch
    print(f"cuda_runtime: {torch.version.cuda}")
    print(f"cudnn: {torch.backends.cudnn.version()}")
    print(f"gpus: {torch.cuda.device_count()} x "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'}")
except Exception:
    pass
PY
    echo
    echo "## Driver"
    nvidia-smi --query-gpu=driver_version,name,memory.total \
        --format=csv 2>/dev/null | head -5 || echo "nvidia-smi unavailable (login node)"
} > "$DEST/provenance.txt"

# The resolved config, not the template: train.sh substitutes paths and may
# append resume_from_checkpoint, so the file on disk is not what ran.
for cfg in "$REPO"/forge/training/configs/leonardo/*.yaml; do
    cp -f "$cfg" "$DEST/" 2>/dev/null || true
done

if [ -n "$LOG" ]; then
    cp -f "$LOG" "$DEST/job-${JOBID}.log" 2>/dev/null || true
    # The measurements, extracted so they are readable without the log.
    {
        echo "# Measurements for job $JOBID"
        echo
        echo "## Run banner"
        sed -n '/=================== RUN ===================/,/^===========================================$/p' "$LOG"
        echo
        echo "## Trainable parameters"
        grep -m1 "trainable params:" "$LOG" || true
        echo
        echo "## Peak memory (allocated is the real number; reserved is the allocator's pool)"
        grep "memprobe" "$LOG" | grep -E "peak_allocated" || echo "(no memprobe output yet)"
        echo
        echo "## Loss (first and last 20 points)"
        grep -o "{'loss'[^}]*}" "$LOG" | head -20
        echo "..."
        grep -o "{'loss'[^}]*}" "$LOG" | tail -20
        echo
        echo "## Evaluation"
        grep -o "{'eval_loss'[^}]*}" "$LOG" || echo "(no eval yet)"
        echo
        echo "## Heartbeat, every 10th sample"
        grep '^\[hb\]' "$LOG" | awk 'NR % 10 == 1'
    } > "$DEST/measurements.txt"
fi

echo "[ok] provenance written to $DEST"
ls -la "$DEST"

#!/bin/bash
# Writes the Fast Lane probe report (fastlane_probe_report.py) for the probe
# directory FLP_DIR, plus a priority snapshot and `saldo -b` beside it.
# Submitted by fastlane_probe.sh three times with --begin (1, 6, 24 hours);
# the last one has FLP_FINAL=1 and cancels the probes still waiting.
#
# Serial partition: a few seconds of one core, nothing from the GPU budget.
#
#SBATCH --job-name=eullm-flp-report
#SBATCH --partition=lrd_all_serial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=%x-%j.out

set -uo pipefail

: "${FLP_DIR:?FLP_DIR is not set}"
: "${FLP_REPO:?FLP_REPO is not set}"
TOOLS="$FLP_REPO/forge/scripts/leonardo"

# The system python3 is 3.6 on this cluster; take the first one that is not
# (same reasoning as sbatch_queue_stats.slurm).
PY=""
for candidate in ${FLP_PYTHON:-} python3.12 python3.11 python3.10 python3.9 python3; do
    if command -v "$candidate" >/dev/null 2>&1 &&
       "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 7) else 1)' 2>/dev/null; then
        PY="$candidate"
        break
    fi
done
if [ -z "$PY" ]; then
    echo "[flp] FATAL: no python >= 3.7; 'module load python/3.11.7' before fastlane_probe.sh" >&2
    exit 1
fi

STAMP="$(date +%Y%m%d_%H%M)"
"$PY" "$TOOLS/priority_snapshot.py" --json "$FLP_DIR/priority_$STAMP.json" ||
    echo "[flp] priority snapshot failed" >&2
saldo -b > "$FLP_DIR/saldo_$STAMP.txt" 2>&1 || true

if [ "${FLP_FINAL:-0}" = 1 ]; then
    "$PY" "$TOOLS/fastlane_probe_report.py" "$FLP_DIR" --final
else
    "$PY" "$TOOLS/fastlane_probe_report.py" "$FLP_DIR"
fi

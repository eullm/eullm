#!/usr/bin/env bash
# What a campaign job measured, written to $CAMPAIGN_DIR/logs/summary-<job>.txt:
# its points by group (`campaign.py report --job`, failures included), each
# configuration against its last earlier measurement (`compare --job`), how
# each point's VRAM moved while it ran (vram_slopes.py), and, for servers with
# llama.cpp's log on, how long reading and copying their weights took (patch
# 0012's load_all_data line, c11's load-* groups), and what the server logs say
# of the expert cache, the VRAM pool and ROCm errors (log_checks.py).
# sbatch_campaign.slurm runs it when its runner ends; summary_after.sh queues
# it behind jobs that were submitted before that.
#
#   bash tools/lumi/job_summary.sh 22981849 [...]

set -uo pipefail
# shellcheck source-path=SCRIPTDIR source=campaign_env.sh
source "$(dirname "$0")/campaign_env.sh"
[ $# -gt 0 ] || { echo "usage: $0 JOB [JOB ...]" >&2; exit 2; }

for JOB in "$@"; do
    SUMMARY="$CAMPAIGN_DIR/logs/summary-$JOB.txt"
    {
        echo "=== what job $JOB measured ==="
        $CAMPAIGN report --queue "$CAMPAIGN_DIR" --job "$JOB"
        echo
        echo "=== against the last earlier measurement of each configuration ==="
        $CAMPAIGN compare --queue "$CAMPAIGN_DIR" --job "$JOB"
        echo
        echo "=== VRAM of each point while it ran ==="
        "$PY" "$EULLM_REPO/tools/lumi/vram_slopes.py" --queue "$CAMPAIGN_DIR" "$JOB"
        echo
        echo "=== the expert cache's statistics, llama.cpp's pool and errors (server logs) ==="
        "$PY" "$EULLM_REPO/tools/lumi/log_checks.py" --queue "$CAMPAIGN_DIR" "$JOB"
        # The engine's log drops llama.cpp's "load_all_data: " in front of the
        # line; the point's environment (RUST_LOG left out) says which way it
        # loaded.
        LOADS=$(for f in $(grep -l "weights, .* read into memory" \
                    "$CAMPAIGN_DIR/results/logs/$JOB"/*.log 2>/dev/null); do
            p=$(basename "$f"); p=${p%.server*}
            env=$("$PY" - "$CAMPAIGN_DIR" "$p.$JOB.json" <<'PY'
import glob, json, os, sys
paths = glob.glob(os.path.join(sys.argv[1], "results", "*", sys.argv[2]))
env = json.load(open(paths[0], encoding="utf-8"))["params"].get("env") or {} if paths else {}
print(" ".join(f"{k}={v}" for k, v in sorted(env.items()) if k != "RUST_LOG") or "-")
PY
)
            grep -h "weights, .* read into memory" "$f" |
                sed -E 's|^.* ([0-9]+ weights, .*waiting for a copy [0-9.]+ s).*$|\1|' |
                while read -r line; do printf '%s (%s): %s\n' "$p" "$env" "$line"; done
        done)
        # The expert cache's choice where no flag made it (0.7.94 and later).
        MOE=$(grep -H "MoE: .* of the experts do not fit" "$CAMPAIGN_DIR/results/logs/$JOB"/*.log \
                2>/dev/null | sed -E 's|^.*/([^/]+)\.server([0-9]+)\.log:.*MoE: |\1 server\2: |')
        if [ -n "$MOE" ]; then
            echo
            echo "=== where each MoE server put its experts (no --moe-cache flag) ==="
            echo "$MOE"
        fi
        if [ -n "$LOADS" ]; then
            echo
            echo "=== how each server read and copied its weights (llama.cpp's load_all_data) ==="
            echo "$LOADS"
        fi
    } > "$SUMMARY" 2>&1
    echo "summary: $SUMMARY"
done

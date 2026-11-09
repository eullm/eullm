#!/bin/bash
# MTP on two engine builds, side by side on one GH200 node: each build with
# and without --mtp 2, one server a GPU, the four at once, as the reference
# PC measured the llama.cpp bump (docs/llama-bump-b11925.md on
# chore/llama-cpp-b11925, 28-10-2026: with --mtp 2 the new llama.cpp kept
# about 5 points fewer of its drafts, every other figure level or better).
#
#     eullm && MTP_BINS="old=$PROJECT/$USER/eullm-target/<label>/release/eullm \
#                        new=$PROJECT/$USER/eullm-target/llama-b11925/release/eullm" \
#         sbatch --account e-aif-2026pg01-1434 \
#         --output="$PROJECT/$USER/campaign/logs/mtp-compare-%j.out" \
#         tools/jupiter/sbatch_mtp_compare.slurm
#
# On each server bench/speed_check.py, three times on each of a story, a
# piece of code and a 16,000-token prompt read: tokens a second, the share
# of drafts kept (the server's own count) and the answer's hash. The model
# is Qwen3.5-9B-MTP Q4_K_M, the PC's, pulled on a login node:
#     $EULLM_BIN pull hf.co/unsloth/Qwen3.5-9B-MTP-GGUF:Q4_K_M
# MTP drafts with one request at a time: --batch-size 1 on every server, the
# ones without MTP too, so the four differ by the build and the flag only.
# Whole node, billed whole: minutes.

#SBATCH --job-name=eullm-mtp-compare
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:45:00

set -uo pipefail
EULLM_REPO="${EULLM_REPO:-$SLURM_SUBMIT_DIR}"
# shellcheck source=env.sh
source "$EULLM_REPO/tools/jupiter/env.sh"
read -r -a BINS <<<"${MTP_BINS:?MTP_BINS: \"old=<eullm> new=<eullm>\"}"
RUNS="${MTP_RUNS:-3}"
JOB="${SLURM_JOB_ID:-local}"
OUT="$CAMPAIGN_DIR/logs/mtp-compare-$JOB"
CODE="Write a Python function that reads a CSV file and returns the mean of every numeric column, with type hints and a docstring."
mkdir -p "$OUT"

echo "=== MTP on two builds: job $JOB on $(hostname), $(date -Is) ==="
MODEL=""
for b in "${BINS[@]}"; do
    [ -x "${b#*=}" ] || { echo "[!!] no binary at ${b#*=}"; exit 1; }
    echo "  ${b%%=*}: $("${b#*=}" --version 2>&1 | head -1) (${b#*=})"
    MODEL=$("${b#*=}" list 2>/dev/null | awk 'tolower($1) ~ /qwen3\.5-9b-mtp/ && tolower($1) ~ /q4_k_m/ && !/file missing/ { print $1; exit }')
done
[ -n "$MODEL" ] || { echo "[!!] Qwen3.5-9B-MTP Q4_K_M is not in $EULLM_MODELS_DIR: pull it on a login node"; exit 1; }
echo "  model $MODEL"

# old-mtp0 on GPU 0, old-mtp2 on GPU 1, new-mtp0 on GPU 2, new-mtp2 on GPU 3.
CASES=() g=0
for b in "${BINS[@]}"; do
    for m in 0 2; do CASES+=("$g ${b%%=*}-mtp$m ${b#*=} $m"); g=$((g + 1)); done
done
[ "$g" -le 4 ] || { echo "[!!] four servers a node: two builds at most"; exit 1; }

measure() {  # GPU LABEL BIN MTP
    local gpu=$1 label=$2 bin=$3 mtp=$4 port=$((18900 + ${SLURM_JOB_ID:-0} % 50 * 4 + $1)) pid i
    CUDA_VISIBLE_DEVICES=$gpu EULLM_AUDIT_DIR="$OUT/audit-$label" taskset -c "$((72 * gpu))-$((72 * gpu + 71))" \
        "$bin" serve --port "$port" --batch-size 1 --ctx-size 20480 --mtp "$mtp" --mmap \
        >"$OUT/$label.server.log" 2>&1 &
    pid=$!
    for _ in $(seq 1 300); do
        curl -sf "http://127.0.0.1:$port/api/version" >/dev/null 2>&1 && break
        kill -0 "$pid" 2>/dev/null || { echo "[!!] $label: the server stopped"; return 1; }
        sleep 1
    done
    for i in $(seq 1 "$RUNS"); do
        python3 "$EULLM_REPO/bench/speed_check.py" --url "http://127.0.0.1:$port/v1" --model "$MODEL" \
            --prompt-tokens 16000 --answer-file "$OUT/$label.story.$i.txt" >"$OUT/$label.story.$i.log" 2>&1
        python3 "$EULLM_REPO/bench/speed_check.py" --url "http://127.0.0.1:$port/v1" --model "$MODEL" \
            --prompt-tokens 16000 --write-prompt "$CODE" --answer-file "$OUT/$label.code.$i.txt" \
            >"$OUT/$label.code.$i.log" 2>&1
    done
    kill "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
}

PIDS=()
for c in "${CASES[@]}"; do
    # shellcheck disable=SC2086
    measure $c &
    PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait "$p"; done

# One block a server: every speed_check run's lines, the reading included.
for c in "${CASES[@]}"; do
    read -r _ label _ _ <<<"$c"
    for t in story code; do
        echo "--- $label, $t"
        for i in $(seq 1 "$RUNS"); do
            grep -hiE "writes|reads|kept|answer" "$OUT/$label.$t.$i.log" | sed "s/^/  run $i: /"
        done
    done
done
echo "logs and answers: $OUT"
echo "=== done $(date -Is) ==="

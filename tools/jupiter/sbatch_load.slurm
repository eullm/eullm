#!/bin/bash
# How long a model takes to load on a GH200 node, EuLLM against llama-server,
# the same file read the two ways llama.cpp knows: mapped (mmap) and read in
# (load mode "none"). One GPU, one server at a time, so that no load competes
# with another for the file system.
#
# Why: job 2326571 (14-10-2026) reported 107-184 s for EuLLM's loads and
# 2-12 s for llama-server's, but the two numbers are not the same thing. The
# runner times the warm-up request after the server answers its readiness
# check; `eullm serve` starts with no model and loads it in that request,
# llama-server loads it before /health answers, so its load was not in the
# 2-12 s at all. Here every case is timed from the start of the process to
# the end of its first answer (8 tokens).
#
# The hypothesis to check: the engine reads a model that goes to the GPUs
# whole into memory when its file is on a network file system
# (fit::read_in_whole_on_gpu, from LUMI's Lustre), $PROJECT is probably GPFS
# to statfs (the header below prints what it is), and llama.cpp's read-in
# path is synchronous 1 MiB reads through four pinned buffers, one thread.
# The cases, in this order:
#
#   1 eullm            the default: read in, if the rule fires (first load of
#                      the file on this node)
#   2 eullm-mmap       --mmap: the file mapped, as llama-server does
#   3 llama-auto       llama-server's default (auto: mapped)
#   4 llama-none       llama-server --load-mode none: EuLLM's read-in path
#                      without EuLLM
#   5 llama-dio        llama-server --load-mode dio: direct I/O, the cache
#                      bypassed
#   6 eullm-again      the default again, the file now read once on this node
#
# 1 against 6 says what the first read of the file costs; 2 against 6, and
# 3 against 4, what reading in costs against mapping, in either server.
#
#     eullm && source tools/jupiter/env.sh &&
#         export EULLM_BIN=$EULLM_TARGETS/next10/release/eullm &&
#         sbatch --account "$SBATCH_ACCOUNT" \
#             --output="$CAMPAIGN_DIR/logs/eullm-load-%j.out" \
#             --export=ALL tools/jupiter/sbatch_load.slurm
#
# EULLM_LOAD_MODEL picks the model (default qwen3.6-27b, about 17 GB).
# A booster node is billed whole: 4 GPU-hours an hour, for the time used.
# Six loads of up to three minutes each: 10-20 minutes, 0.7-1.3 GPU-hours.
#
# EULLM_LOAD_CASES=together measures the other half of the rule instead:
# four EuLLM servers, one per GPU, loading one file at once, first mapped
# (--mmap), then read in (the default). On LUMI's Lustre, servers mapping one
# file at once were the slow case (c09 load-together: four of gpt-oss-120b,
# 1,789 s mapped against 53 s read in), which is why the rule reads in on
# every network file system; this says whether GPFS does the same. Mapped
# goes first, on a file this node has not read: the worst case for mapping.
# Its default model is qwen3.6-35b-a3b (22 GB), loaded alone in 166 s read in
# by job 2326571.

#SBATCH --job-name=eullm-load
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00

set -uo pipefail
# shellcheck source=env.sh
source "${EULLM_REPO:?submit with --export=ALL after sourcing tools/jupiter/env.sh}/tools/jupiter/env.sh"
: "${EULLM_BIN:?set EULLM_BIN to the CUDA eullm binary}"
CASES="${EULLM_LOAD_CASES:-six}"
if [ "$CASES" = together ]; then
    MODEL="${EULLM_LOAD_MODEL:-qwen3.6-35b-a3b}"
else
    MODEL="${EULLM_LOAD_MODEL:-qwen3.6-27b}"
fi
OUT="$CAMPAIGN_DIR/logs/eullm-load-${SLURM_JOB_ID:-local}"
PORT=$((18400 + ${SLURM_JOB_ID:-0} % 300))
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=0
# Every request leaves an audit line: here, not in $HOME.
export EULLM_AUDIT_DIR="$OUT/audit"
mkdir -p "$EULLM_AUDIT_DIR"

GGUF=$("$PY" -c "import sys; sys.path.insert(0, '$EULLM_REPO/bench/campaign')
from point import store_gguf; print(store_gguf('$MODEL') or '')")
[ -n "$GGUF" ] || { echo "[!!] $MODEL is not in $EULLM_MODELS_DIR"; exit 1; }

echo "=== model load on GH200: job ${SLURM_JOB_ID:-local} on $(hostname), $(date -Is) ==="
printf '  %-12s %s\n' engine "$("$EULLM_BIN" --version 2>&1 | head -1)" \
    repo "${EULLM_BENCH_REV:-$(git -C "$EULLM_REPO" rev-parse --short HEAD 2>/dev/null || echo ?)}" \
    llama-server "${LLAMA_SERVER_BIN:-none}" model "$MODEL" file "$GGUF" \
    size "$(du -h --apparent-size "$GGUF" | cut -f1)" \
    filesystem "$(stat -f -c '%T (magic 0x%t)' "$GGUF")"
awk '/^MemAvailable:/ { printf "  %-12s %.0f GiB\n", "available", $2 / 1048576 }' /proc/meminfo
CG="/sys/fs/cgroup$(sed -n 's/^0:://p' /proc/self/cgroup)"
[ -r "$CG/memory.max" ] && printf '  %-12s %s\n' "cgroup max" "$(cat "$CG/memory.max")"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/  /'
echo

wait_ready() {  # PATH PID [PORT]: up to ten minutes for the server to answer, while it lives
    for _ in $(seq 1 1200); do
        curl -sf "localhost:${3:-$PORT}$1" >/dev/null 2>&1 && return 0
        kill -0 "$2" 2>/dev/null || return 1
        sleep 0.5
    done
    return 1
}

ROWS=() LABELS=()
case_() {  # LABEL RUNTIME [ARGS...]: start, time the first answer, stop
    local label=$1 runtime=$2 ready path body server t0 t1 t2 load
    shift 2
    LABELS+=("$label")
    local log="$OUT/$label.log" answer="$OUT/$label.answer.json"
    t0=$(date +%s.%N)
    if [ "$runtime" = llama-server ]; then
        # -lv 4: llama.cpp's own information lines (load_mode, buffer sizes)
        # are verbosity 4, and llama-server shows 3; job 2326571's logs had none.
        ready=/health path=/completion body='{"prompt": "Ciao.", "n_predict": 8}'
        "$LLAMA_SERVER_BIN" --host 127.0.0.1 --port "$PORT" -m "$GGUF" --alias "$MODEL" \
            -c 16384 -np 1 -ngl 999 -lv 4 --log-timestamps --log-prefix "$@" >"$log" 2>&1 &
    else
        ready=/api/version path=/api/generate
        body="{\"model\": \"$MODEL\", \"prompt\": \"Ciao.\", \"stream\": false, \"think\": false, \"options\": {\"num_predict\": 8}}"
        RUST_LOG=eullm=info "$EULLM_BIN" serve --port "$PORT" --batch-size 1 \
            --ctx-size 16384 "$@" >"$log" 2>&1 &
    fi
    server=$!
    if ! wait_ready "$ready" "$server"; then
        echo "[!!] $label: the server did not answer $ready; the end of $log:"
        tail -5 "$log" | sed 's/^/    /'
        kill "$server" 2>/dev/null
        wait "$server" 2>/dev/null
        ROWS+=("$(printf '%-16s %8s %8s %8s %s' "$label" - - - failed)")
        return
    fi
    t1=$(date +%s.%N)
    curl -s -m 900 -H 'Content-Type: application/json' -d "$body" \
        "localhost:$PORT$path" >"$answer" 2>/dev/null
    t2=$(date +%s.%N)
    kill "$server" 2>/dev/null
    wait "$server" 2>/dev/null
    # EuLLM's own figure for the load, from its answer.
    load=$("$PY" -c "import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
    print(f\"{d['load_duration'] / 1e9:.1f}\" if d.get('load_duration') else '-')
except Exception:
    print('-')" "$answer")
    ROWS+=("$(awk -v l="$label" -v a="$t0" -v b="$t1" -v c="$t2" -v d="$load" \
        'BEGIN { printf "%-16s %8.1f %8.1f %8.1f %8s", l, b - a, c - b, c - a, d }')")
    echo "  $label done: $(( ${t2%.*} - ${t0%.*} )) s"
    sleep 3
}

together() {  # LABEL [ARGS...]: four EuLLM servers, one per GPU, loading at once
    local label=$1 i t0 t1 load
    shift
    local servers=() clients=()
    for i in 0 1 2 3; do
        LABELS+=("$label.$i")
        CUDA_VISIBLE_DEVICES=$i RUST_LOG=eullm=info "$EULLM_BIN" serve \
            --port $((PORT + i)) --batch-size 1 --ctx-size 16384 "$@" \
            >"$OUT/$label.$i.log" 2>&1 &
        servers+=($!)
    done
    for i in 0 1 2 3; do
        wait_ready /api/version "${servers[$i]}" $((PORT + i)) ||
            echo "[!!] $label.$i: the server did not start; tail $OUT/$label.$i.log"
    done
    local body="{\"model\": \"$MODEL\", \"prompt\": \"Ciao.\", \"stream\": false, \"think\": false, \"options\": {\"num_predict\": 8}}"
    t0=$(date +%s.%N)
    for i in 0 1 2 3; do
        (curl -s -m 1500 -H 'Content-Type: application/json' -d "$body" \
            "localhost:$((PORT + i))/api/generate" >"$OUT/$label.$i.answer.json" 2>/dev/null
         date +%s.%N >"$OUT/$label.$i.end") &
        clients+=($!)
    done
    wait "${clients[@]}"
    for i in 0 1 2 3; do
        kill "${servers[$i]}" 2>/dev/null
        wait "${servers[$i]}" 2>/dev/null
        t1=$(cat "$OUT/$label.$i.end" 2>/dev/null || echo "$t0")
        load=$("$PY" -c "import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
    print(f\"{d['load_duration'] / 1e9:.1f}\" if d.get('load_duration') else '-')
except Exception:
    print('-')" "$OUT/$label.$i.answer.json")
        ROWS+=("$(awk -v l="$label.$i" -v a="$t0" -v c="$t1" -v d="$load" \
            'BEGIN { printf "%-16s %8s %8.1f %8s %8s", l, "-", c - a, "-", d }')")
    done
    echo "  $label done: $(( ${t1%.*} - ${t0%.*} )) s"
    sleep 3
}

if [ "$CASES" = together ]; then
    echo "=== four servers loading at once, one per GPU: mapped, then read in ==="
    together together-mmap --mmap
    together together-read
else
    echo "=== the six loads, one after the other, on GPU 0 ==="
    case_ eullm eullm
    case_ eullm-mmap eullm --mmap
    if [ -x "${LLAMA_SERVER_BIN:-}" ]; then
        case_ llama-auto llama-server
        case_ llama-none llama-server --load-mode none
        case_ llama-dio llama-server --load-mode dio
    else
        echo "[!!] no llama-server: bash tools/jupiter/install_runtimes.sh on a login node"
    fi
    case_ eullm-again eullm
fi

echo
echo "=== seconds from the start of the process ==="
printf '%-16s %8s %8s %8s %8s\n' case ready answer total "eullm load"
printf '%s\n' "${ROWS[@]}"
echo "(ready: until the server answers; answer: the first request, 8 tokens,"
echo " for 'together' from the moment the four were sent; eullm load:"
echo " load_duration in EuLLM's answer)"

echo
echo "=== the log lines that say how each one read the file ==="
for label in "${LABELS[@]}"; do
    echo "--- $label"
    grep -E 'into memory instead of mapping|Loading model:|Allocating context|Model swap complete|load_mode =|model buffer size|model loaded|server is listening' \
        "$OUT/$label.log" | cut -c1-260 | sed 's/^/  /'
done
echo
echo "=== done $(date -Is) ==="

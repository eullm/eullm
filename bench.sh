#!/usr/bin/env bash
# Multi-sequence batching benchmark
# Usage: ./bench.sh <base_url> <model_name>
# Example (Ollama):  ./bench.sh http://localhost:11435 qwen3.5:9b
# Example (EULLM):   ./bench.sh http://localhost:11434 Qwen3.5-9B-Q8_0

set -uo pipefail  # no -e, background jobs may fail

BASE_URL="${1:?Usage: ./bench.sh <base_url> <model_name>}"
MODEL="${2:?Usage: ./bench.sh <base_url> <model_name>}"
PROMPT="Write a detailed essay about the history of Rome from its founding to the fall of the Western Roman Empire. Include key events, important figures, political changes, military campaigns, cultural achievements, and the reasons for its decline. Be thorough and comprehensive."
NUM_PREDICT=150
ENDPOINT="${BASE_URL}/api/chat"

echo "Benchmark: ${MODEL} @ ${BASE_URL}"
echo "num_predict=${NUM_PREDICT}"
echo "=========================================="
echo ""

TMPDIR_BENCH=$(mktemp -d)
trap 'rm -rf "$TMPDIR_BENCH"' EXIT

run_single() {
    local id=$1
    local outfile="${TMPDIR_BENCH}/req${id}.txt"
    local start end elapsed

    start=$(date +%s%N)

    # Send both top-level num_predict (EULLM) and options.num_predict (Ollama)
    # Also send think:false for Qwen models
    local payload
    payload=$(cat <<EOJSON
{
  "model": "${MODEL}",
  "messages": [{"role": "user", "content": "${PROMPT}"}],
  "stream": false,
  "think": false,
  "num_predict": ${NUM_PREDICT},
  "options": {"num_predict": ${NUM_PREDICT}}
}
EOJSON
)

    local response
    response=$(curl -s --max-time 180 -H "Content-Type: application/json" "$ENDPOINT" -d "$payload" 2>&1)

    end=$(date +%s%N)
    elapsed=$(( (end - start) / 1000000 ))

    # Parse response — works for both Ollama and EULLM response format
    local eval_count eval_duration toks_per_sec
    eval_count=$(echo "$response" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('eval_count', 0))
except:
    print(0)
" 2>/dev/null)
    eval_duration=$(echo "$response" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('eval_duration', 0))
except:
    print(0)
" 2>/dev/null)

    toks_per_sec="0"
    if [ "${eval_count:-0}" -gt 0 ] && [ "${eval_duration:-0}" -gt 0 ] 2>/dev/null; then
        toks_per_sec=$(python3 -c "print(f'{${eval_count} / (${eval_duration} / 1e9):.1f}')")
    fi

    # If no eval_duration from API, compute from wall time
    if [ "$toks_per_sec" = "0" ] && [ "${eval_count:-0}" -gt 0 ] && [ "$elapsed" -gt 0 ]; then
        toks_per_sec=$(python3 -c "print(f'{${eval_count} / (${elapsed} / 1000):.1f}')")
        echo "  req${id}: ${eval_count} tokens, ${elapsed}ms wall, ~${toks_per_sec} tok/s (wall)" | tee "$outfile"
    else
        echo "  req${id}: ${eval_count} tokens, ${elapsed}ms wall, ${toks_per_sec} tok/s" | tee "$outfile"
    fi
}

for N in 1 2 4 8 16; do
    echo "=== ${N} concurrent request(s) ==="

    total_start=$(date +%s%N)

    pids=()
    for i in $(seq 1 "$N"); do
        run_single "$i" &
        pids+=($!)
    done

    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done

    total_end=$(date +%s%N)
    total_ms=$(( (total_end - total_start) / 1000000 ))

    # Summary
    total_tokens=0
    for f in "${TMPDIR_BENCH}"/req*.txt; do
        [ -f "$f" ] || continue
        t=$(grep -oP '\d+ tokens' "$f" | grep -oP '\d+' || echo 0)
        total_tokens=$((total_tokens + t))
    done
    rm -f "${TMPDIR_BENCH}"/req*.txt

    if [ "$total_ms" -gt 0 ]; then
        throughput=$(python3 -c "print(f'{${total_tokens} / (${total_ms} / 1000):.1f}')")
    else
        throughput="0"
    fi

    echo "  --- wall: ${total_ms}ms | total tokens: ${total_tokens} | throughput: ${throughput} tok/s ---"
    echo ""
done

echo "Done."

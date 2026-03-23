#!/usr/bin/env bash
# Multi-sequence batching benchmark
# Usage: ./bench.sh <base_url> <model_name>
# Example: ./bench.sh http://localhost:11435 qwen3.5:9b

set -uo pipefail  # no -e, background jobs may fail

BASE_URL="${1:?Usage: ./bench.sh <base_url> <model_name>}"
MODEL="${2:?Usage: ./bench.sh <base_url> <model_name>}"
PROMPT="List the 5 largest cities in Europe. Be brief."
NUM_PREDICT=150  # cap output tokens to keep runs short
ENDPOINT="${BASE_URL}/api/generate"

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

    local response
    response=$(curl -s --max-time 180 "$ENDPOINT" \
        -d "{\"model\":\"${MODEL}\",\"prompt\":\"${PROMPT}\",\"stream\":false,\"options\":{\"num_predict\":${NUM_PREDICT}}}" 2>&1)

    end=$(date +%s%N)
    elapsed=$(( (end - start) / 1000000 ))

    # Parse response
    local eval_count eval_duration toks_per_sec
    eval_count=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('eval_count',0))" 2>/dev/null || echo 0)
    eval_duration=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('eval_duration',0))" 2>/dev/null || echo 0)

    toks_per_sec="0"
    if [ "$eval_duration" -gt 0 ] 2>/dev/null; then
        toks_per_sec=$(python3 -c "print(f'{${eval_count} / (${eval_duration} / 1e9):.1f}')")
    fi

    echo "  req${id}: ${eval_count} tokens, ${elapsed}ms wall, ${toks_per_sec} tok/s" | tee "$outfile"
}

for N in 1 2 4 8; do
    echo "=== ${N} concurrent request(s) ==="

    total_start=$(date +%s%N)

    pids=()
    for i in $(seq 1 "$N"); do
        run_single "$i" &
        pids+=($!)
    done

    # Wait for all, don't fail on individual errors
    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done

    total_end=$(date +%s%N)
    total_ms=$(( (total_end - total_start) / 1000000 ))

    # Summary: total tokens across all requests
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

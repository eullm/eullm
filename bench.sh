#!/usr/bin/env bash
# Multi-sequence batching benchmark
# Usage: ./bench.sh <base_url> <model_name>
# Example: ./bench.sh http://localhost:11434 qwen3.5:9b

set -euo pipefail

BASE_URL="${1:?Usage: ./bench.sh <base_url> <model_name>}"
MODEL="${2:?Usage: ./bench.sh <base_url> <model_name>}"
PROMPT="Explain quantum computing in exactly 100 words."
ENDPOINT="${BASE_URL}/api/generate"

echo "Benchmark: ${MODEL} @ ${BASE_URL}"
echo "=========================================="
echo ""

run_single() {
    local id=$1
    local start end elapsed
    start=$(date +%s%N)

    local response
    response=$(curl -s --max-time 120 "$ENDPOINT" \
        -d "{\"model\":\"${MODEL}\",\"prompt\":\"${PROMPT}\",\"stream\":false}")

    end=$(date +%s%N)
    elapsed=$(( (end - start) / 1000000 )) # ms

    # Parse token counts and durations from response
    local eval_count eval_duration prompt_eval_count
    eval_count=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('eval_count',0))" 2>/dev/null || echo 0)
    eval_duration=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('eval_duration',0))" 2>/dev/null || echo 0)
    prompt_eval_count=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('prompt_eval_count',0))" 2>/dev/null || echo 0)

    local toks_per_sec="0"
    if [ "$eval_duration" -gt 0 ]; then
        toks_per_sec=$(python3 -c "print(f'{${eval_count} / (${eval_duration} / 1e9):.1f}')")
    fi

    echo "  req${id}: ${eval_count} tokens, ${elapsed}ms wall, ${toks_per_sec} tok/s"
}

for N in 1 2 4 8; do
    echo "=== ${N} concurrent request(s) ==="

    total_start=$(date +%s%N)

    pids=()
    for i in $(seq 1 "$N"); do
        run_single "$i" &
        pids+=($!)
    done

    for pid in "${pids[@]}"; do
        wait "$pid"
    done

    total_end=$(date +%s%N)
    total_ms=$(( (total_end - total_start) / 1000000 ))

    echo "  --- total wall time: ${total_ms}ms ---"
    echo ""
done

echo "Done."

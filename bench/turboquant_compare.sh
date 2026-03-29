#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# TurboQuant KV Cache Comparison Orchestrator
#
# Starts the EULLM engine with each KV cache type (f16, tq4_0, tq3_0),
# runs the benchmark, collects results, and produces a comparison table.
#
# Usage:
#   ./bench/turboquant_compare.sh <model_path> [extra bench args...]
#
# Examples:
#   ./bench/turboquant_compare.sh ./models/qwen3-14b-q4_k_m.gguf
#   ./bench/turboquant_compare.sh ./models/qwen3-14b-q4_k_m.gguf --concurrency 1,2,4 --tokens 100
#
# Environment variables:
#   EULLM_BIN       Path to eullm binary (default: ./target/release/eullm-engine)
#   EULLM_PORT      Port for the engine (default: 11434)
#   EULLM_HOST      Host to bind (default: 127.0.0.1)
#   BENCH_ROUNDS    Rounds per concurrency level (default: 3)
#   BENCH_WARMUP    Warmup requests (default: 1)
#   CTX_SIZE        Context size (default: engine default)
#   BATCH_SIZE      Batch size / concurrent slots (default: 16)
#   HEALTH_TIMEOUT  Seconds to wait for engine health (default: 120)
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_PATH="${1:?Usage: $0 <model_path> [extra bench args...]}"
shift

EULLM_BIN="${EULLM_BIN:-./eullm-tq}"
EULLM_PORT="${EULLM_PORT:-11434}"
EULLM_HOST="${EULLM_HOST:-127.0.0.1}"
EULLM_URL="http://${EULLM_HOST}:${EULLM_PORT}"

BENCH_ROUNDS="${BENCH_ROUNDS:-3}"
BENCH_WARMUP="${BENCH_WARMUP:-1}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-120}"
CTX_SIZE="${CTX_SIZE:-}"
BATCH_SIZE="${BATCH_SIZE:-16}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_SCRIPT="${SCRIPT_DIR}/turboquant_bench.py"

RESULTS_DIR="${SCRIPT_DIR}/results/turboquant_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

MODEL_NAME="$(basename "${MODEL_PATH}" .gguf)"

CACHE_TYPES=("f16" "tq4_0" "tq3_0")

ENGINE_PID=""

# ── Cleanup trap ──────────────────────────────────────────────────────────────

cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ -n "$ENGINE_PID" ] && kill -0 "$ENGINE_PID" 2>/dev/null; then
        echo "  Stopping engine (PID $ENGINE_PID)..."
        kill "$ENGINE_PID" 2>/dev/null || true
        # Give it a moment to shut down gracefully
        for _ in $(seq 1 5); do
            if ! kill -0 "$ENGINE_PID" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        # Force kill if still running
        if kill -0 "$ENGINE_PID" 2>/dev/null; then
            echo "  Force-killing engine..."
            kill -9 "$ENGINE_PID" 2>/dev/null || true
        fi
        echo "  Engine stopped."
    fi
}

trap cleanup EXIT INT TERM

# ── Helper functions ──────────────────────────────────────────────────────────

wait_for_health() {
    local url="$1"
    local timeout="$2"
    local elapsed=0

    echo -n "  Waiting for engine health at ${url}..."
    while [ "$elapsed" -lt "$timeout" ]; do
        # Check if engine process is still alive
        if [ -n "$ENGINE_PID" ] && ! kill -0 "$ENGINE_PID" 2>/dev/null; then
            echo ""
            echo "  ENGINE DIED (PID $ENGINE_PID exited)"
            local log="${RESULTS_DIR}/${CURRENT_CACHE_TYPE}_engine.log"
            if [ -f "$log" ]; then
                # Detect common failure causes
                if grep -qi "out of memory\|CUDA error\|OOM\|alloc\|GGML_ASSERT\|SIGABRT\|VRAM" "$log" 2>/dev/null; then
                    echo "  CAUSE: Out of memory / VRAM insufficient"
                elif grep -qi "Failed to create context\|Failed to load" "$log" 2>/dev/null; then
                    echo "  CAUSE: Context creation failed (likely OOM)"
                fi
                echo "  Last 5 lines of log:"
                tail -5 "$log" | sed 's/^/    /'
            fi
            ENGINE_PID=""
            return 1
        fi
        if curl -sf "${url}/api/version" > /dev/null 2>&1; then
            echo " ready (${elapsed}s)"
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
        echo -n "."
    done

    echo " TIMEOUT after ${timeout}s"
    return 1
}

# Track current cache type for log file detection in wait_for_health
CURRENT_CACHE_TYPE=""

start_engine() {
    local cache_type="$1"
    local ctx="${2:-$CTX_SIZE}"
    local log_file="${RESULTS_DIR}/${cache_type}_engine.log"

    echo "  Starting engine with --cache-type-k ${cache_type} --cache-type-v ${cache_type} ctx=${ctx:-default}"
    echo "  Log: ${log_file}"

    local engine_args=(
        run "$MODEL_PATH"
        --port "$EULLM_PORT"
        --cache-type-k "$cache_type"
        --cache-type-v "$cache_type"
        --batch-size "$BATCH_SIZE"
    )
    if [ -n "$ctx" ]; then
        engine_args+=(--ctx-size "$ctx")
    fi

    "$EULLM_BIN" "${engine_args[@]}" > "$log_file" 2>&1 &

    ENGINE_PID=$!
    echo "  Engine PID: $ENGINE_PID"
}

# Probe the maximum ctx_size a cache type can handle by trying descending
# values until the engine starts successfully. Returns the working ctx_size
# via the PROBED_CTX global variable.
probe_max_ctx() {
    local cache_type="$1"
    local start_ctx="$2"

    # Try: start_ctx, 75%, 50%, 25%, 12.5%, and some common values
    local try_sizes=()
    local c="$start_ctx"
    while [ "$c" -ge 4096 ]; do
        try_sizes+=("$c")
        c=$((c * 3 / 4))            # 75% of previous
        # Round down to nearest 1024
        c=$(( (c / 1024) * 1024 ))
    done
    try_sizes+=(4096)

    echo "  Probing max ctx_size for ${cache_type}..."
    echo "  Will try: ${try_sizes[*]}"

    PROBED_CTX=""
    for try_ctx in "${try_sizes[@]}"; do
        CURRENT_CACHE_TYPE="${cache_type}_probe_${try_ctx}"
        local probe_log="${RESULTS_DIR}/${cache_type}_probe_${try_ctx}.log"

        echo -n "  Trying ctx_size=${try_ctx}..."
        local probe_args=(
            run "$MODEL_PATH"
            --port "$EULLM_PORT"
            --cache-type-k "$cache_type"
            --cache-type-v "$cache_type"
            --batch-size "$BATCH_SIZE"
            --ctx-size "$try_ctx"
        )
        "$EULLM_BIN" "${probe_args[@]}" > "$probe_log" 2>&1 &
        ENGINE_PID=$!

        # Wait up to 60s for health
        local elapsed=0
        local ok=false
        while [ "$elapsed" -lt 60 ]; do
            if ! kill -0 "$ENGINE_PID" 2>/dev/null; then
                echo " FAILED (engine crashed)"
                ENGINE_PID=""
                break
            fi
            if curl -sf "${EULLM_URL}/api/version" > /dev/null 2>&1; then
                ok=true
                break
            fi
            sleep 2
            elapsed=$((elapsed + 2))
        done

        if $ok; then
            echo " OK! Max ctx_size for ${cache_type} = ${try_ctx}"
            PROBED_CTX="$try_ctx"
            # Engine is running — stop it, we'll restart properly later
            stop_engine
            return 0
        else
            # Make sure it's dead
            if [ -n "$ENGINE_PID" ] && kill -0 "$ENGINE_PID" 2>/dev/null; then
                kill -9 "$ENGINE_PID" 2>/dev/null || true
                sleep 1
            fi
            ENGINE_PID=""
        fi
    done

    echo "  Could not find a working ctx_size for ${cache_type}"
    PROBED_CTX=""
    return 1
}

stop_engine() {
    if [ -n "$ENGINE_PID" ] && kill -0 "$ENGINE_PID" 2>/dev/null; then
        echo "  Stopping engine (PID $ENGINE_PID)..."
        kill "$ENGINE_PID" 2>/dev/null || true
        # Wait for graceful shutdown
        for _ in $(seq 1 10); do
            if ! kill -0 "$ENGINE_PID" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        if kill -0 "$ENGINE_PID" 2>/dev/null; then
            kill -9 "$ENGINE_PID" 2>/dev/null || true
            sleep 1
        fi
        echo "  Engine stopped."
    fi
    ENGINE_PID=""
}

# ── Banner ────────────────────────────────────────────────────────────────────

echo "================================================================"
echo "  TurboQuant KV Cache Comparison"
echo "================================================================"
echo "  Model:       ${MODEL_PATH}"
echo "  Model name:  ${MODEL_NAME}"
echo "  Engine:      ${EULLM_BIN}"
echo "  URL:         ${EULLM_URL}"
echo "  Cache types: ${CACHE_TYPES[*]}"
echo "  Ctx size:    ${CTX_SIZE:-engine default}"
echo "  Batch size:  ${BATCH_SIZE}"
echo "  Rounds:      ${BENCH_ROUNDS}"
echo "  Results:     ${RESULTS_DIR}"
echo "  Extra args:  $*"
echo "================================================================"
echo ""

# ── Validate prerequisites ────────────────────────────────────────────────────

if [ ! -f "$EULLM_BIN" ]; then
    echo "ERROR: EULLM binary not found at ${EULLM_BIN}"
    echo "  Set EULLM_BIN to the correct path or build with: cargo build --release -p eullm-engine"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model file not found at ${MODEL_PATH}"
    exit 1
fi

if [ ! -f "$BENCH_SCRIPT" ]; then
    echo "ERROR: Benchmark script not found at ${BENCH_SCRIPT}"
    exit 1
fi

if ! python3 -c "import aiohttp" 2>/dev/null; then
    echo "ERROR: aiohttp is required. Install with: pip install aiohttp"
    exit 1
fi

# ── Run benchmarks ────────────────────────────────────────────────────────────

RESULT_FILES=()

for cache_type in "${CACHE_TYPES[@]}"; do
    echo ""
    echo "================================================================"
    echo "  Cache type: ${cache_type}"
    echo "================================================================"

    CURRENT_CACHE_TYPE="$cache_type"
    # Map cache type to label for the bench script
    CACHE_LABEL="${cache_type^^}"  # uppercase: f16 -> F16, tq4_0 -> TQ4_0
    OUTPUT_FILE="${RESULTS_DIR}/${cache_type}.json"

    # Determine ctx_size for this cache type.
    # For non-TQ types (f16, q8_0, etc.) with large ctx: probe to find max.
    effective_ctx="$CTX_SIZE"
    if [ -n "$CTX_SIZE" ] && [[ "$cache_type" != tq* ]]; then
        echo "  Non-TurboQuant type with explicit ctx_size — probing VRAM fit..."
        if probe_max_ctx "$cache_type" "$CTX_SIZE"; then
            effective_ctx="$PROBED_CTX"
            if [ "$effective_ctx" != "$CTX_SIZE" ]; then
                CACHE_LABEL="${CACHE_LABEL} (ctx=${effective_ctx})"
                echo "  NOTE: Reduced ctx_size from ${CTX_SIZE} to ${effective_ctx} for ${cache_type}"
            fi
        else
            echo "  SKIPPING ${cache_type} — cannot fit in VRAM even at ctx_size=4096"
            continue
        fi
    fi

    RESULT_FILES+=("$OUTPUT_FILE")

    # Start engine with the effective ctx
    start_engine "$cache_type" "$effective_ctx"

    # Wait for health
    if ! wait_for_health "$EULLM_URL" "$HEALTH_TIMEOUT"; then
        echo "  ERROR: Engine failed to start with cache type ${cache_type}"
        echo "  Check log: ${RESULTS_DIR}/${cache_type}_engine.log"
        stop_engine
        continue
    fi

    # Run benchmark
    echo ""
    python3 "$BENCH_SCRIPT" collect \
        --url "$EULLM_URL" \
        --model "$MODEL_NAME" \
        --cache-label "$CACHE_LABEL" \
        --rounds "$BENCH_ROUNDS" \
        --warmup "$BENCH_WARMUP" \
        --output "$OUTPUT_FILE" \
        "$@"

    # Stop engine before next cache type
    echo ""
    stop_engine

    echo "  Done with ${cache_type} (ctx=${effective_ctx:-default})."
    echo ""
done

# ── Compare results ───────────────────────────────────────────────────────────

# Filter to only files that exist (in case some cache types failed)
EXISTING_FILES=()
for f in "${RESULT_FILES[@]}"; do
    if [ -f "$f" ]; then
        EXISTING_FILES+=("$f")
    fi
done

if [ ${#EXISTING_FILES[@]} -lt 1 ]; then
    echo "ERROR: No benchmark results were collected."
    exit 1
fi

echo ""
echo "================================================================"
echo "  Comparison"
echo "================================================================"
echo ""

python3 "$BENCH_SCRIPT" compare "${EXISTING_FILES[@]}"

echo ""
echo "  Markdown version:"
echo ""
python3 "$BENCH_SCRIPT" compare "${EXISTING_FILES[@]}" --markdown

echo ""
echo "================================================================"
echo "  Results saved to: ${RESULTS_DIR}"
for f in "${EXISTING_FILES[@]}"; do
    echo "    - ${f}"
done
echo "================================================================"

#!/usr/bin/env bash
# Compare EULLM vs Ollama with the real stress test
#
# Usage:
#   ./bench/compare.sh <eullm_model> <ollama_model>
#   ./bench/compare.sh Qwen3.5-9B-Q8_0 qwen3.5:9b
#   ./bench/compare.sh Qwen3.5-9B-Q8_0 qwen3.5:9b --concurrency 1,2,4,8,16 --tokens 150

set -euo pipefail

EULLM_MODEL="${1:?Usage: ./bench/compare.sh <eullm_model> <ollama_model> [extra args...]}"
OLLAMA_MODEL="${2:?Usage: ./bench/compare.sh <eullm_model> <ollama_model> [extra args...]}"
shift 2

EULLM_URL="${EULLM_URL:-http://localhost:11434}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11435}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STRESS_TEST="${SCRIPT_DIR}/stress_test.py"

RESULTS_DIR="${SCRIPT_DIR}/results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "================================================================"
echo "  EULLM vs Ollama — Stress Test Comparison"
echo "================================================================"
echo "  EULLM:  ${EULLM_URL} / ${EULLM_MODEL}"
echo "  Ollama: ${OLLAMA_URL} / ${OLLAMA_MODEL}"
echo "  Results: ${RESULTS_DIR}"
echo "  Extra args: $*"
echo "================================================================"
echo ""

# Check servers are reachable
echo "Checking servers..."
if ! curl -sf "${EULLM_URL}/api/version" > /dev/null 2>&1; then
    echo "ERROR: EULLM not reachable at ${EULLM_URL}"
    exit 1
fi
echo "  EULLM:  OK"

if ! curl -sf "${OLLAMA_URL}/api/version" > /dev/null 2>&1; then
    echo "ERROR: Ollama not reachable at ${OLLAMA_URL}"
    exit 1
fi
echo "  Ollama: OK"
echo ""

# Run EULLM benchmark
echo "Running EULLM benchmark..."
echo ""
python3 "$STRESS_TEST" \
    --url "$EULLM_URL" \
    --model "$EULLM_MODEL" \
    --label "EULLM" \
    --warmup \
    --json "${RESULTS_DIR}/eullm.json" \
    "$@"

echo ""
echo ""

# Run Ollama benchmark
echo "Running Ollama benchmark..."
echo ""
python3 "$STRESS_TEST" \
    --url "$OLLAMA_URL" \
    --model "$OLLAMA_MODEL" \
    --label "Ollama" \
    --warmup \
    --json "${RESULTS_DIR}/ollama.json" \
    "$@"

echo ""
echo "================================================================"
echo "  Results saved to: ${RESULTS_DIR}"
echo "  - ${RESULTS_DIR}/eullm.json"
echo "  - ${RESULTS_DIR}/ollama.json"
echo "================================================================"

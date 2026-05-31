#!/usr/bin/env bash
# Run the KV cache quality benchmark across 5 arms (F16, Q8_0, Q4_0, TQ4_0, TQ3_0)
# against the same model with the same prompts and seed.
#
# Usage:
#   ./bench/run_quality_arms.sh <model.gguf> [<eullm-binary>] [<model-name>]
#
# Defaults:
#   eullm-binary = ./target/release/eullm (or 'eullm' from PATH if missing)
#   model-name   = qwen3-14b
#   ctx-size     = 8192
#   port         = 11434
#
# Requires the engine to be built with --features turboquant_native for TQ arms.

set -uo pipefail

MODEL_PATH="${1:?Usage: ./bench/run_quality_arms.sh <model.gguf> [<eullm-bin>] [<model-name>]}"
EULLM_BIN="${2:-./target/release/eullm}"
MODEL_NAME="${3:-qwen3-14b}"
CTX_SIZE="${CTX_SIZE:-8192}"
PORT="${PORT:-11434}"
TEMPERATURE="${TEMPERATURE:-0.0}"

if [[ ! -x "$EULLM_BIN" ]]; then
  if command -v eullm >/dev/null 2>&1; then
    EULLM_BIN="$(command -v eullm)"
    echo "Using eullm from PATH: $EULLM_BIN"
  else
    echo "ERROR: eullm binary not found at '$EULLM_BIN' and not in PATH." >&2
    exit 1
  fi
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$REPO_ROOT/bench/results"
LOG_DIR="$RESULTS_DIR/quality_arm_logs"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# Arms: (label, cache-type-k, cache-type-v, requires-turboquant)
ARMS=(
  "F16:f16:f16:0"
  "Q8_0:q8_0:q8_0:0"
  "Q4_0:q4_0:q4_0:0"
  "TQ4_0:tq4_0:tq4_0:1"
  "TQ3_0:tq3_0:tq3_0:1"
)

cleanup() {
  if [[ -n "${ENGINE_PID:-}" ]] && kill -0 "$ENGINE_PID" 2>/dev/null; then
    echo "Stopping engine (pid $ENGINE_PID)..."
    kill -TERM "$ENGINE_PID" 2>/dev/null || true
    sleep 1
    kill -KILL "$ENGINE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

wait_for_engine() {
  local url="http://localhost:${PORT}/api/tags"
  for _ in $(seq 1 60); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "ERROR: engine did not come up on port $PORT within 60s." >&2
  return 1
}

echo "Quality benchmark — 5 arms"
echo "  Binary:  $EULLM_BIN"
echo "  Model:   $MODEL_PATH ($MODEL_NAME)"
echo "  Ctx:    $CTX_SIZE  Port: $PORT  Temp: $TEMPERATURE"
echo "  Results: $RESULTS_DIR/quality_<arm>.json"
echo

for entry in "${ARMS[@]}"; do
  IFS=':' read -r LABEL CK CV NEEDS_TQ <<<"$entry"
  echo "==================================================================="
  echo "ARM: $LABEL  (cache-type-k=$CK, cache-type-v=$CV)"
  echo "==================================================================="

  ENGINE_LOG="$LOG_DIR/engine_${LABEL}.log"
  OUT_JSON="$RESULTS_DIR/quality_${LABEL}.json"

  # Start engine
  "$EULLM_BIN" run "$MODEL_PATH" \
      --port "$PORT" \
      --ctx-size "$CTX_SIZE" \
      --cache-type-k "$CK" \
      --cache-type-v "$CV" \
      > "$ENGINE_LOG" 2>&1 &
  ENGINE_PID=$!
  echo "  engine pid=$ENGINE_PID  log=$ENGINE_LOG"

  if ! wait_for_engine; then
    echo "  Tail of engine log:"
    tail -n 30 "$ENGINE_LOG" || true
    if [[ "$NEEDS_TQ" == "1" ]]; then
      echo "  SKIP — TurboQuant arm; engine likely not built with --features turboquant_native."
      kill -TERM "$ENGINE_PID" 2>/dev/null || true
      wait "$ENGINE_PID" 2>/dev/null || true
      continue
    else
      exit 1
    fi
  fi

  # Run the prompts
  python3 "$REPO_ROOT/bench/turboquant_quality.py" collect \
      --url "http://localhost:${PORT}" \
      --model "$MODEL_NAME" \
      --label "$LABEL" \
      --temperature "$TEMPERATURE" \
      --output "$OUT_JSON"

  # Stop engine
  kill -TERM "$ENGINE_PID" 2>/dev/null || true
  wait "$ENGINE_PID" 2>/dev/null || true
  unset ENGINE_PID
  echo
done

echo "All arms complete. Regenerating charts..."
python3 "$REPO_ROOT/bench/generate_quality_charts.py" "$RESULTS_DIR/quality_*.json"

echo
echo "Markdown summary:"
python3 "$REPO_ROOT/bench/turboquant_quality.py" compare \
    "$RESULTS_DIR"/quality_*.json --markdown | tail -n 20

echo
echo "Done. Send to repo:"
echo "  bench/results/quality_*.json"
echo "  bench/results/chart_quality_comparison.png"
echo "  bench/results/chart_quality_radar.png"

#!/bin/bash
# Replication (data parallelism) on LUMI-G: one whole copy of the model per GCD,
# one server per copy, requests spread across them.
#
# WHY THIS EXISTS. sbatch_bench.slurm measures what the engine does today with
# several devices: llama.cpp's layer split, which cuts the model by layer and
# walks a token through the devices IN SEQUENCE. That is a capacity mechanism —
# it exists to fit a model too big for one device — and it cannot add
# throughput. Measured on 12 Sep 2026 with a 29.3 GiB model on MI250X GCDs:
# 97.8 tok/s on one GCD, 96.3 on four. Flat, as the mechanism predicts.
#
# When the model fits on one device, the way to use four is to run four of it.
# No tensors are split, so there is nothing to communicate between devices at
# any point: the scaling limit is the host, the filesystem and the request
# source, not an interconnect. This script measures that.
#
#     export SBATCH_ACCOUNT=project_465003366
#     export EULLM_BIN=$PWD/eullm-rocm
#     export EULLM_MODELS_DIR=/scratch/$SBATCH_ACCOUNT/$USER/eullm-models
#     export MODEL=qwen3.8-27b-ud-q8_k_xl
#
#     # Like for like: the same 16 requests as the single-GCD row, but served by
#     # 4 replicas (4 requests each) instead of one server at batch 16.
#     BATCH=4 CTX=16384 CONCURRENCY=16 \
#         sbatch --gpus-per-node=4 --cpus-per-task=28 tools/lumi/sbatch_replicas.slurm
#
#     # Four times the load: each replica gets the same 16 concurrent requests
#     # the single-GCD run handled, so this is the throughput ceiling of a node.
#     BATCH=16 CTX=65536 CONCURRENCY=64 \
#         sbatch --gpus-per-node=4 --cpus-per-task=28 tools/lumi/sbatch_replicas.slurm
#
# Both are worth having and they answer different questions: the first asks
# whether replication beats layer split on an identical workload, the second
# asks how much a whole node can actually serve.
#
# Each replica loads the full model, so N GCDs need N copies in HBM — fine for
# 29.3 GiB on 64 GiB devices, and the reason this approach stops working the
# moment the model no longer fits one device. There the answer is the hybrid:
# the smallest model split that fits the weights, replicated over what is left.
#
# Prints one `BENCH_RESULT {...}` line with `"mode": "replicas"`, the same shape
# sbatch_bench.slurm emits with `"mode": "layer-split"`, so a sweep collects both
# with one grep and the two are directly comparable.

#SBATCH --job-name=eullm-replicas
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=28
#SBATCH --time=01:00:00
#SBATCH --output=eullm-replicas-%j.out

set -uo pipefail

EULLM_BIN="${EULLM_BIN:?set EULLM_BIN to the eullm binary}"
export MODEL="${MODEL:?set MODEL to the model id (see: eullm list)}"
export EULLM_MODELS_DIR="${EULLM_MODELS_DIR:-$HOME/.eullm/models}"

export BATCH="${BATCH:-16}"
export CTX="${CTX:-65536}"
export CACHE_K="${CACHE_K:-f16}"
export CACHE_V="${CACHE_V:-f16}"
export CONCURRENCY="${CONCURRENCY:-64}"
export NUM_PREDICT="${NUM_PREDICT:-150}"
PROMPT="${PROMPT:-Scrivi una breve storia sul mare.}"

export REPLICAS="${REPLICAS:-${SLURM_GPUS_ON_NODE:-1}}"
export GCDS="${SLURM_GPUS_ON_NODE:-1}"
export MODE="replicas"
PORT_BASE=$((11434 + SLURM_JOB_ID % 900))
OUT_PREFIX="eullm-replicas"

# Per-device utilisation, sampled by the job itself. Catching the timed window
# by hand from a login node does not work: it lasts seconds, arrives minutes
# after submission, and job 21982159 had finished before the first manual
# sample landed. This is also the measurement objective (3) actually needs —
# not "was a GPU busy" but "were ALL of them busy, at once".
UTIL_LOG="$OUT_PREFIX-$SLURM_JOB_ID.util.log"
start_sampler() {
    ( while :; do
        echo "T $(date +%s)"
        rocm-smi --showuse 2>/dev/null | grep -E 'GPU use \(%\)'
        sleep 1
      done ) > "$UTIL_LOG" 2>&1 &
    SAMPLER_PID=$!
}
stop_sampler() { kill "${SAMPLER_PID:-0}" 2>/dev/null; }

summarise_util() {
    python3 - "$UTIL_LOG" <<'PYU'
import re, sys
from collections import defaultdict
per = defaultdict(list)
for line in open(sys.argv[1], errors="replace"):
    m = re.search(r"GPU\[(\d+)\].*?GPU use \(%\):\s*(\d+)", line)
    if m:
        per[int(m.group(1))].append(int(m.group(2)))
if not per:
    print("  (no utilisation samples captured)")
    raise SystemExit(0)
print("  device   samples   mean%   max%")
for dev in sorted(per):
    v = per[dev]
    print(f"  GPU[{dev}]   {len(v):7d}   {sum(v)/len(v):5.1f}   {max(v):4d}")
busy = [d for d, v in per.items() if sum(v)/len(v) > 50]
print(f"  devices averaging over 50% busy: {len(busy)} of {len(per)}")
PYU
}

echo "=== configuration ==="
printf '  %-14s %s\n' node "$(hostname)" gcds "$GCDS" replicas "$REPLICAS" \
    model "$MODEL" batch "$BATCH" ctx "$CTX" kv "$CACHE_K/$CACHE_V" \
    concurrency "$CONCURRENCY" num_predict "$NUM_PREDICT"
rocm-smi --showproductname 2>&1 | sed 's/^/  /'

# One server per GCD. ROCR_VISIBLE_DEVICES is the ROCm equivalent of
# CUDA_VISIBLE_DEVICES and is what pins a process to a single device: without
# it every replica would find all four and layer-split across them, which is
# the arrangement this script exists to avoid.
SERVER_PIDS=()
cleanup() {
    for pid in "${SERVER_PIDS[@]:-}"; do kill "$pid" 2>/dev/null; done
    stop_sampler
    [ -n "${WORK:-}" ] && rm -rf "$WORK"
    return 0
}
trap cleanup EXIT
trap 'cleanup; exit 143' TERM INT

echo
echo "=== starting $REPLICAS replicas, one per GCD ==="
for r in $(seq 0 $((REPLICAS - 1))); do
    ROCR_VISIBLE_DEVICES="$r" "$EULLM_BIN" serve --port $((PORT_BASE + r)) \
        --batch-size "$BATCH" --ctx-size "$CTX" \
        --cache-type-k "$CACHE_K" --cache-type-v "$CACHE_V" \
        > "eullm-replicas-$SLURM_JOB_ID.server$r.log" 2>&1 &
    SERVER_PIDS+=("$!")
    echo "  replica $r → GCD $r, port $((PORT_BASE + r)), pid ${SERVER_PIDS[-1]}"
done

# Every replica reads the same GGUF at once. On Lustre that is the one part of
# this design that does not scale for free, and the warm-up below times it.
echo
echo "=== waiting for all replicas ==="
for r in $(seq 0 $((REPLICAS - 1))); do
    ready=0
    for _ in $(seq 1 150); do
        curl -sf "http://127.0.0.1:$((PORT_BASE + r))/api/version" >/dev/null 2>&1 && { ready=1; break; }
        sleep 2
    done
    if [ "$ready" -ne 1 ]; then
        echo "ERROR: replica $r never became reachable"
        tail -30 "eullm-replicas-$SLURM_JOB_ID.server$r.log"
        exit 1
    fi
    echo "  replica $r up"
done

WORK=$(mktemp -d)

# $1 = request id, $2 = replica index
request() {
    curl -s --max-time 1800 -H "Content-Type: application/json" \
        "http://127.0.0.1:$((PORT_BASE + $2))/api/generate" -d @- > "$WORK/resp.$1" <<EOJSON
{
  "model": "$MODEL",
  "prompt": "$PROMPT",
  "stream": false,
  "think": false,
  "num_predict": $NUM_PREDICT,
  "options": {"num_predict": $NUM_PREDICT}
}
EOJSON
}

# Warm-up: every replica loads its own copy. Untimed, as in sbatch_bench.slurm.
echo
echo "=== warm-up: one request per replica (untimed, loads $REPLICAS copies) ==="
WARM_START=$(date +%s%N)
WARM_PIDS=()
for r in $(seq 0 $((REPLICAS - 1))); do request "warmup$r" "$r" & WARM_PIDS+=("$!"); done
wait "${WARM_PIDS[@]}"
echo "  warm-up completed in $(( ($(date +%s%N) - WARM_START) / 1000000 )) ms"
grep -iE "fit:|found .* ROCm devices" "eullm-replicas-$SLURM_JOB_ID.server0.log" | sed 's/^/  /'

echo
echo "=== timed: $CONCURRENCY requests spread round-robin over $REPLICAS replicas ==="
start_sampler
START=$(date +%s%N)
PIDS=()
for i in $(seq 1 "$CONCURRENCY"); do
    request "$i" $(( (i - 1) % REPLICAS )) &
    PIDS+=("$!")
done
wait "${PIDS[@]}"
ELAPSED_MS=$(( ($(date +%s%N) - START) / 1000000 ))
stop_sampler

echo
echo "=== device utilisation during the timed window ==="
summarise_util

python3 - "$WORK" "$ELAPSED_MS" <<'PY'
import json, os, sys
work, elapsed_ms = sys.argv[1], int(sys.argv[2])
tok = prompt_tok = 0
ok = failed = 0
for name in sorted(os.listdir(work)):
    if not name.startswith("resp.") or name.startswith("resp.warmup"):
        continue
    try:
        d = json.load(open(os.path.join(work, name)))
    except Exception:
        failed += 1
        continue
    if "error" in d:
        print("  request error:", str(d["error"])[:200])
        failed += 1
        continue
    tok += d.get("eval_count") or 0
    prompt_tok += d.get("prompt_eval_count") or 0
    ok += 1
secs = elapsed_ms / 1000
print(f"  requests: {ok} ok, {failed} failed")
print(f"  prompt tokens: {prompt_tok} | generated tokens: {tok}")
print(f"  wall clock: {secs:.1f} s")
if tok and secs:
    print(f"  AGGREGATE THROUGHPUT: {tok/secs:.1f} tok/s")
result = {
    "site": "lumi-g", "mode": os.environ.get("MODE"),
    "gcds": int(os.environ.get("GCDS", 0)),
    "replicas": int(os.environ.get("REPLICAS", 0)),
    "model": os.environ.get("MODEL"), "batch": int(os.environ.get("BATCH", 0)),
    "ctx": int(os.environ.get("CTX", 0)),
    "kv": f"{os.environ.get('CACHE_K')}/{os.environ.get('CACHE_V')}",
    "concurrency": ok, "failed": failed,
    "generated_tokens": tok, "wall_clock_s": round(secs, 2),
    "aggregate_tok_s": round(tok / secs, 1) if tok and secs else None,
    "job": os.environ.get("SLURM_JOB_ID"),
}
print("BENCH_RESULT " + json.dumps(result))
PY

echo
echo "=== devices after the run (all $REPLICAS should have been busy) ==="
rocm-smi --showmeminfo vram --showuse 2>&1 | sed 's/^/  /'

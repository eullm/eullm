#!/bin/bash
# Throughput benchmark on LUMI-G, reproducing the method used on CINECA
# Leonardo so the two sets of numbers can be put side by side.
#
# That method, from docs/cineca/leonardo.md, and every part of it matters:
# aggregate output tokens summed across N concurrent /api/generate requests
# with the same prompt, divided by the wall-clock time of the whole batch,
# with a warm-up request sent first and excluded from the timed window.
# Change any of it and the comparison stops meaning anything.
#
# One configuration per job, because context size, batch size and KV cache type
# are launch flags: changing them means restarting the server, and a job that
# quietly restarted it four times would attribute four results to one
# allocation. Submit it once per row of the matrix.
#
#     export SBATCH_ACCOUNT=project_465003366
#     export EULLM_BIN=$PWD/eullm-rocm
#     export EULLM_MODELS_DIR=/scratch/$SBATCH_ACCOUNT/$USER/eullm-models
#     export MODEL=<the model id `eullm list` shows>
#
#     # Leonardo's four rows, as LUMI jobs. GCDs, not GPUs: a LUMI node is
#     # 4 MI250X = 8 GCDs, and software sees each GCD as a device — so 4 GCDs
#     # is the same DEVICE count as Leonardo's 4 A100, and 8 GCDs is the same
#     # NODE. Both are worth having; they answer different questions.
#     BATCH=1  CTX=4096  sbatch --gpus-per-node=1 --cpus-per-task=7  tools/lumi/sbatch_bench.slurm
#     BATCH=16 CTX=65536 sbatch --gpus-per-node=1 --cpus-per-task=7  tools/lumi/sbatch_bench.slurm
#     BATCH=16 CTX=65536 sbatch --gpus-per-node=4 --cpus-per-task=28 tools/lumi/sbatch_bench.slurm
#     BATCH=16 CTX=65536 CACHE_K=q8_0 CACHE_V=q8_0 \
#              sbatch --gpus-per-node=4 --cpus-per-task=28 tools/lumi/sbatch_bench.slurm
#
#     # Full node: 8 GCDs is a whole node, so standard-g rather than small-g,
#     # and a node-hour there costs 4 GPU-hours.
#     BATCH=16 CTX=65536 sbatch --partition=standard-g --gpus-per-node=8 \
#              --cpus-per-task=56 tools/lumi/sbatch_bench.slurm
#
# The engine has no --tensor-split or --split-mode today, so it uses every
# device Slurm makes visible, split by layer — which is exactly what produced
# Leonardo's 1.87x from one card to four. The GCD count in the allocation IS
# the experiment variable; there is nothing to pass to the binary.
#
# Prints one JSON line, `BENCH_RESULT {...}`, so a sweep's outputs can be
# collected with `grep -h BENCH_RESULT eullm-bench-*.out`.

#SBATCH --job-name=eullm-bench
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --time=01:00:00
#SBATCH --output=eullm-bench-%j.out

set -uo pipefail  # no -e: the server runs in the background and must be reaped

EULLM_BIN="${EULLM_BIN:?set EULLM_BIN to the eullm binary}"
export MODEL="${MODEL:?set MODEL to the model id (see: eullm list)}"
export EULLM_MODELS_DIR="${EULLM_MODELS_DIR:-$HOME/.eullm/models}"

# The experiment variables. Defaults are Leonardo's single-stream row.
#
# Exported, every one of them, because the result JSON below is assembled by a
# python block that reads them from the environment. A value left as a plain
# shell variable reaches the human-readable summary and NOT the machine-readable
# record — so a sweep would print correct numbers on screen while filing
# `"gcds": 0, "kv": "None/None"` into the file the final report is built from.
# Silently wrong data is worse than no data.
export BATCH="${BATCH:-1}"
export CTX="${CTX:-4096}"
export CACHE_K="${CACHE_K:-f16}"
export CACHE_V="${CACHE_V:-f16}"
export CONCURRENCY="${CONCURRENCY:-16}"
export NUM_PREDICT="${NUM_PREDICT:-150}"
# Leonardo's prompt, kept verbatim: a different one tokenizes differently and
# generates a different length, which moves the number being compared.
PROMPT="${PROMPT:-Scrivi una breve storia sul mare.}"

# One request at batch size 1 — concurrency against a single slot would just
# queue, measuring the queue rather than the engine.
[ "$BATCH" -le 1 ] && export CONCURRENCY=1

export GCDS="${SLURM_GPUS_ON_NODE:-1}"
# Names what several devices are doing, so a result from this script and one
# from sbatch_replicas.slurm can sit in the same file and still be told apart:
# here the devices hold one model between them, there they hold one each.
export MODE="layer-split"
PORT=$((11434 + SLURM_JOB_ID % 1000))
OUT_PREFIX="eullm-bench"
LOG="$OUT_PREFIX-$SLURM_JOB_ID.server.log"

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
printf '  %-14s %s\n' node "$(hostname)" gcds "$GCDS" model "$MODEL" batch "$BATCH" \
    ctx "$CTX" kv "$CACHE_K/$CACHE_V" concurrency "$CONCURRENCY" num_predict "$NUM_PREDICT"
rocm-smi --showproductname 2>&1 | sed 's/^/  /'

echo
echo "=== starting server ==="
"$EULLM_BIN" serve --port "$PORT" \
    --batch-size "$BATCH" --ctx-size "$CTX" \
    --cache-type-k "$CACHE_K" --cache-type-v "$CACHE_V" > "$LOG" 2>&1 &
SERVER_PID=$!

# TERM and INT as well as EXIT, with the limits established by test rather than
# assumed, because bash's trap semantics are not what they look like:
#
#   - a trap does NOT run while the shell sits in a foreground command; it is
#     deferred until that command returns, so a SIGTERM during a `curl` or the
#     readiness `sleep` waits for it;
#   - a trap DOES interrupt `wait`, which is where this script spends the whole
#     timed window — and there the cleanup runs and the server really goes
#     (absent from /proc afterwards, not left as a zombie).
#
# So this makes cleanup prompt in the phase that matters and not everywhere.
# It does NOT claim to fix the Slurm CG (completing) state seen after a scancel
# on 12 Sep 2026: scancel signals every process in the job step, so the server
# receives SIGTERM from Slurm whatever this trap does, and what actually kept
# those four nodes completing was never established.
cleanup() {
    kill "$SERVER_PID" 2>/dev/null
    stop_sampler
    [ -n "${WORK:-}" ] && rm -rf "$WORK"
    return 0
}
trap cleanup EXIT
trap 'cleanup; exit 143' TERM INT

for _ in $(seq 1 90); do
    curl -sf "http://127.0.0.1:$PORT/api/version" >/dev/null 2>&1 && break
    sleep 2
done
if ! curl -sf "http://127.0.0.1:$PORT/api/version" >/dev/null 2>&1; then
    echo "ERROR: server never became reachable"; tail -40 "$LOG"; exit 1
fi

WORK=$(mktemp -d)   # cleanup() above removes it, on a signal as well as on exit

request() {
    curl -s --max-time 1800 -H "Content-Type: application/json" \
        "http://127.0.0.1:$PORT/api/generate" -d @- > "$WORK/resp.$1" <<EOJSON
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

# Warm-up: loads the model and pays the first-token costs. Excluded from the
# timed window, as on Leonardo — timing it would measure the 29 GiB read.
echo
echo "=== warm-up (untimed: loads the model) ==="
WARM_START=$(date +%s%N)
request warmup
WARM_S=$(( ($(date +%s%N) - WARM_START) / 1000000 ))
echo "  warm-up completed in ${WARM_S} ms"
grep -iE "gpu|offload|device|layers" "$LOG" | head -12 | sed 's/^/  /'

echo
echo "=== timed: $CONCURRENCY concurrent requests ==="
start_sampler
START=$(date +%s%N)
# `wait` with no arguments waits for EVERY background child, and the server is
# one of them — it never exits, so a bare `wait` hangs after the requests have
# finished, holding the allocation until the walltime kills it. Found live: four
# jobs sat idle at GPU 0% for eleven minutes past their last completed request.
# Wait for the request PIDs and nothing else.
PIDS=()
for i in $(seq 1 "$CONCURRENCY"); do request "$i" & PIDS+=("$!"); done
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
    if not name.startswith("resp.") or name == "resp.warmup":
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
echo "=== device after the run ==="
rocm-smi --showmeminfo vram --showuse 2>&1 | sed 's/^/  /'

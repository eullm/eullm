#!/bin/bash
# Phase 0 for diffusion LLMs on LUMI-G: measure, do not integrate.
#
#     export SBATCH_ACCOUNT=project_465003366
#     export DIFFUSION_BIN=$PWD/.../bin/llama-diffusion-cli   # build_diffusion_cli.sh
#     export TOKENIZE_BIN=$PWD/.../bin/llama-tokenize
#     export LLADA_GGUF=/scratch/$SBATCH_ACCOUNT/$USER/models/LLaDA-8B-Instruct-Q4_K_M.gguf
#     export DREAM_GGUF=/scratch/$SBATCH_ACCOUNT/$USER/models/Dream-v0-Instruct-7B-Q4_K_M.gguf
#     sbatch tools/lumi/sbatch_diffusion.slurm
#
# WHAT THIS ANSWERS, and what it deliberately does not. The question is not
# "does it run" — llama.cpp has carried `dream`, `llada`, `llada-moe` and `rnd1`
# since before the commit we vendor. The question is how far the UNCACHED
# baseline sits from autoregressive decode on the same silicon, because that is
# the number that decides whether porting the 408-line loop into the Rust engine
# is worth doing now or worth waiting for.
#
# Uncached is not an oversight, it is what this code is: there is not one line
# of KV-cache handling in examples/diffusion/diffusion.cpp. Every step is a
# whole-sequence `llama_decode`. For Dream and LLaDA 1.x that is inherent —
# bidirectional attention over a sequence whose token values change each step
# means every position's K/V changes too. It is NOT inherent for LLaDA 2.x,
# which is block diffusion: block-causal attention, confirmed blocks frozen, so
# past blocks cache exactly. That family is the one where a cache pays, and it
# is also the one this llama.cpp cannot load — hence measuring 1.x here and
# treating the result as a floor, not as the ceiling of the approach.
#
# The comparable AR number comes from the script we already have. Run
# `CONCURRENCY=1 BATCH=1 CTX=4096 sbatch --gpus-per-node=1 tools/lumi/sbatch_bench.slurm`
# and compare `aggregate_tok_s` against `tok_s` here. Do not compare against the
# batched rows: diffusion serves one sequence per context by construction
# (`llama_batch_init(max_length, 0, 1)`), so a batch-16 AR figure is a different
# question.
#
# ms_per_step is reported because it is the unit the upstream work is quoted in
# (ggml-org/llama.cpp discussion #22972 reports ~1000 ms/step for the uncached
# MVP and ~190 ms/step with block KV, on LLaDA2 mini Q4_0, block 32, 256 steps,
# on Arc B60 under SYCL). Those are a different model, a different quant and the
# least-tuned llama.cpp backend, so the numbers are not directly comparable —
# but ms/step at block 32 / 256 steps is the axis along which they can at least
# be argued about, and nobody has published the CUDA or HIP side of it.
#
# Emits one `BENCH_RESULT {...}` line per point with `"mode": "diffusion"`, the
# same shape sbatch_bench.slurm and sbatch_replicas.slurm emit, so one grep
# collects every arrangement we have measured.

#
# IF `sbatch` REFUSES THE JOB, read which limit it names — the two look alike
# and have nothing in common:
#
#   AssocMaxSubmitJobLimit         no account resolved, so the job went to the
#                                  default association, whose MaxSubmit is 0.
#                                  `squeue -u $USER` is EMPTY in this case,
#                                  which is what makes the message misleading:
#                                  nothing is queued, the limit is simply zero.
#                                  `sbatch` reads SBATCH_ACCOUNT at submission
#                                  and a fresh shell does not have it — pass
#                                  --account=project_465003366 explicitly instead of
#                                  trusting the environment. Same root cause as
#                                  the three-variable note in sbatch_smoke.slurm,
#                                  different symptom.
#
#   QOSMaxWallDurationPerJobLimit  --time exceeds what this QoS grants. Lower
#                                  the time, or leave the debug QoS and queue
#                                  behind the long jobs.
#
# Neither costs allocation: both are refused before the job queues.

#SBATCH --job-name=eullm-diffusion
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --time=01:00:00
#SBATCH --output=eullm-diffusion-%j.out

set -uo pipefail

DIFFUSION_BIN="${DIFFUSION_BIN:?set DIFFUSION_BIN — see tools/lumi/build_diffusion_cli.sh}"
TOKENIZE_BIN="${TOKENIZE_BIN:-}"
PROMPT="${PROMPT:-Spiega in tre frasi cosa rende un supercomputer diverso da un cluster ordinario.}"
ALGORITHM="${ALGORITHM:-4}"     # 4 = confidence-based, the CLI default
TEMP="${TEMP:-0.2}"
SEED="${SEED:-42}"
DREAM_EPS="${DREAM_EPS:-0.001}"
LLADA_BLOCK="${LLADA_BLOCK:-32}"

# One point per line: label | gguf | schedule_flag | max_length | steps
# max_length is passed as -ub: `diff_params.max_length = params.n_ubatch`
# (diffusion-cli.cpp:205), so the output length IS the physical micro-batch.
# That ceiling is the single most restrictive property of this code path and the
# sweep is built to show where it starts to hurt.
build_sweep() {
    [ -n "${LLADA_GGUF:-}" ] && cat <<EOP
llada-8b|$LLADA_GGUF|--diffusion-block-length $LLADA_BLOCK|256|256
llada-8b|$LLADA_GGUF|--diffusion-block-length $LLADA_BLOCK|256|128
llada-8b|$LLADA_GGUF|--diffusion-block-length $LLADA_BLOCK|512|256
EOP
    [ -n "${DREAM_GGUF:-}" ] && cat <<EOP
dream-7b|$DREAM_GGUF|--diffusion-eps $DREAM_EPS|256|256
dream-7b|$DREAM_GGUF|--diffusion-eps $DREAM_EPS|256|128
dream-7b|$DREAM_GGUF|--diffusion-eps $DREAM_EPS|512|256
EOP
}

SWEEP="$(build_sweep)"
# Say which one is missing, and what it was actually set to. An unset variable
# and one set to an empty string look identical from here, and a shell that ate
# a placeholder — `<...>` is a redirection, not a placeholder — leaves exactly
# the second case.
if [ -z "$SWEEP" ] || [ ! -x "$DIFFUSION_BIN" ]; then
    echo "ERROR: cannot build a sweep. What this job can see:"
    printf '  %-14s %s\n' \
        DIFFUSION_BIN "${DIFFUSION_BIN:-<unset>}$([ -n "${DIFFUSION_BIN:-}" ] && { [ -x "$DIFFUSION_BIN" ] && echo '  (ok)' || echo '  (NOT EXECUTABLE)'; })" \
        LLADA_GGUF    "${LLADA_GGUF:-<unset or empty>}" \
        DREAM_GGUF    "${DREAM_GGUF:-<unset or empty>}"
    echo "  At least one of LLADA_GGUF / DREAM_GGUF must name a readable .gguf."
    echo "  Derive them from what actually landed rather than typing them:"
    echo "    export LLADA_GGUF=\$(ls \$MODELS/*LLaDA*Q4_K_M*.gguf | head -1)"
    exit 1
fi

echo "=== node and device ==="
hostname
rocm-smi --showproductname 2>&1 | sed 's/^/  /'
echo "  binary: $DIFFUSION_BIN"

# Prompt length decides how many of the max_length positions are actually
# generated. The CLI never prints it on the success path, and `n_generated` it
# returns is max_length — the prompt included. Reporting that as tok/s would
# inflate every row by the prompt, so count the prompt properly or say we could
# not. Silently-wrong numbers have cost this project a benchmark already.
N_INPUT=""
if [ -n "$TOKENIZE_BIN" ] && [ -x "$TOKENIZE_BIN" ]; then
    FIRST_GGUF=$(printf '%s\n' "$SWEEP" | head -1 | cut -d'|' -f2)
    # `--show-count` prints one line, "Total number of tokens: N". Parsing that
    # beats counting the `--ids` array: no bracket/comma handling, and a change
    # in how ids are formatted cannot silently return a wrong count.
    TOK_OUT=$("$TOKENIZE_BIN" -m "$FIRST_GGUF" -p "$PROMPT" --show-count 2>&1)
    TOK_RC=$?
    N_INPUT=$(printf '%s\n' "$TOK_OUT" \
        | sed -n 's/^Total number of tokens: *\([0-9]\{1,\}\).*/\1/p' | tail -1)
    [ "${N_INPUT:-0}" -gt 0 ] 2>/dev/null || N_INPUT=""
    # Keep the reason. Throwing stderr away here once cost a run whose tok/s
    # could only be reported as an upper bound, with nothing on record saying
    # why — and the candidates are not alike: a missing --show-count in this
    # build, a binary that cannot start because libcuda.so.1 is absent, or a
    # model this tokenizer refuses. Each has a different fix.
    if [ -z "$N_INPUT" ]; then
        echo "  prompt count failed (exit $TOK_RC) — tok/s will be an upper bound. First lines:"
        printf '%s\n' "$TOK_OUT" | head -3 | sed 's/^/      /'
    fi
fi
if [ -n "$N_INPUT" ]; then
    echo "  prompt: $N_INPUT tokens"
else
    echo "  prompt: NOT COUNTED (set TOKENIZE_BIN) — tok/s will be reported as an upper bound"
fi

UTIL_LOG="eullm-diffusion-$SLURM_JOB_ID.util.log"
start_sampler() {
    ( while :; do rocm-smi --showuse 2>/dev/null | grep -E 'GPU use \(%\)'; sleep 1; done ) \
        > "$UTIL_LOG" 2>&1 &
    SAMPLER_PID=$!
}
stop_sampler() { kill "${SAMPLER_PID:-0}" 2>/dev/null; }
trap 'stop_sampler' EXIT
trap 'stop_sampler; exit 143' TERM INT

while IFS='|' read -r LABEL GGUF SCHED MAXLEN STEPS; do
    [ -n "${LABEL:-}" ] || continue
    echo
    echo "=== $LABEL | $SCHED | max_length=$MAXLEN | steps=$STEPS ==="

    if [ ! -f "$GGUF" ]; then echo "  SKIP: no such file: $GGUF"; continue; fi

    # Preflight the asserts rather than discovering them after the model has
    # loaded. diffusion.cpp aborts on both of these, and on this partition a
    # failed point costs the load time of an 8B model for nothing.
    if [[ "$SCHED" == --diffusion-block-length* ]]; then
        BLK="${SCHED##* }"
        if (( MAXLEN % BLK != 0 )); then
            echo "  SKIP: max_length $MAXLEN is not a multiple of block $BLK (GGML_ASSERT would abort)"; continue
        fi
        NBLOCKS=$(( MAXLEN / BLK ))
        if (( STEPS % NBLOCKS != 0 )); then
            echo "  SKIP: steps $STEPS is not a multiple of $NBLOCKS blocks (GGML_ASSERT would abort)"; continue
        fi
    fi
    if [ -n "$N_INPUT" ] && (( MAXLEN <= N_INPUT )); then
        echo "  SKIP: max_length $MAXLEN <= prompt $N_INPUT — diffusion_generate returns immediately"; continue
    fi

    LOG="eullm-diffusion-$SLURM_JOB_ID.$LABEL.$MAXLEN.$STEPS.log"
    start_sampler
    WALL_START=$(date +%s%N)
    "$DIFFUSION_BIN" -m "$GGUF" -p "$PROMPT" \
        -ngl 999 -c "$MAXLEN" -b "$MAXLEN" -ub "$MAXLEN" \
        --diffusion-steps "$STEPS" $SCHED \
        --diffusion-algorithm "$ALGORITHM" \
        --temp "$TEMP" --seed "$SEED" \
        > "$LOG" 2>&1
    RC=$?
    WALL_MS=$(( ($(date +%s%N) - WALL_START) / 1000000 ))
    stop_sampler

    if [ "$RC" -ne 0 ]; then
        echo "  FAILED (exit $RC) — last lines:"; tail -15 "$LOG" | sed 's/^/    /'; continue
    fi

    LABEL="$LABEL" SCHED="$SCHED" MAXLEN="$MAXLEN" STEPS="$STEPS" \
    N_INPUT="${N_INPUT:-}" WALL_MS="$WALL_MS" UTIL_LOG="$UTIL_LOG" \
    python3 - "$LOG" <<'PY'
import json, os, re, sys
from collections import defaultdict

log = open(sys.argv[1], errors="replace", encoding="utf-8").read()

# diffusion.cpp logs exactly one line with all three figures:
#   total time: 12345.67ms, time per step: 48.22ms, sampling time per step: 1.10ms
m = re.search(r"total time:\s*([\d.]+)ms,\s*time per step:\s*([\d.]+)ms,"
              r"\s*sampling time per step:\s*([\d.]+)ms", log)
total_ms = ms_step = ms_sample = None
if m:
    total_ms, ms_step, ms_sample = (float(m.group(i)) for i in (1, 2, 3))
else:
    print("  WARNING: no timing line in the log — the run may have aborted early")

maxlen = int(os.environ["MAXLEN"])
steps  = int(os.environ["STEPS"])
n_in   = os.environ.get("N_INPUT") or ""
n_in   = int(n_in) if n_in.isdigit() else None

# Generated tokens are the positions the prompt does not occupy. Without a
# prompt count we can only report max_length/time, which overstates tok/s —
# label it rather than quietly publish it.
generated = (maxlen - n_in) if n_in is not None else maxlen
secs = (total_ms or float(os.environ["WALL_MS"])) / 1000.0
tok_s = round(generated / secs, 2) if secs else None

per = defaultdict(list)
for line in open(os.environ["UTIL_LOG"], errors="replace", encoding="utf-8"):
    g = re.search(r"GPU\[(\d+)\].*?GPU use \(%\):\s*(\d+)", line)
    if g:
        per[int(g.group(1))].append(int(g.group(2)))
util = {f"gpu{d}_mean": round(sum(v)/len(v), 1) for d, v in sorted(per.items()) if v}

print(f"  total: {total_ms} ms | per step: {ms_step} ms | sampling/step: {ms_sample} ms")
print(f"  generated: {generated} tok ({'prompt counted' if n_in is not None else 'PROMPT NOT COUNTED — upper bound'})")
print(f"  THROUGHPUT: {tok_s} tok/s")
if util:
    print(f"  gpu util: {util}")

sched = os.environ["SCHED"]
result = {
    "site": "lumi-g", "mode": "diffusion", "gcds": 1,
    "model": os.environ["LABEL"],
    "schedule": "block" if "block-length" in sched else "timestep",
    "schedule_arg": sched,
    "steps": steps, "max_length": maxlen,
    "n_input": n_in, "generated_tokens": generated,
    "prompt_counted": n_in is not None,
    "total_ms": total_ms, "ms_per_step": ms_step, "sampling_ms_per_step": ms_sample,
    "wall_clock_s": round(int(os.environ["WALL_MS"]) / 1000.0, 2),
    "tok_s": tok_s, "gpu_util": util,
    "job": os.environ.get("SLURM_JOB_ID"),
}
print("BENCH_RESULT " + json.dumps(result))
PY
done <<< "$SWEEP"

echo
echo "=== device after the sweep ==="
rocm-smi --showmeminfo vram --showuse 2>&1 | sed 's/^/  /'

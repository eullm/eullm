#!/bin/bash
# AutoBench on LUMI-G: `"model": "auto"` with cost ratios a 16 GB card cannot
# hold — a small model beside a 27B or 32B one on a single 64 GiB GCD (L4 of
# docs/reflex-mvp3-plan.md).
#
# WHY HERE. On the 5070 Ti the pairs are 4B against 8B or 14B: what routing
# saves is bounded by how little the large model costs there. With a 32B as
# the large model, every request routed small saves several times more GPU
# time, and whether the router's accuracy holds at that gap is the question.
# One GCD, so both models and the decision model share its 64 GiB, as they
# would share one card in a deployment.
#
#     export SBATCH_ACCOUNT=project_465003366
#     export EULLM_BIN=$PWD/eullm-rocm
#     export EULLM_MODELS_DIR=/scratch/$SBATCH_ACCOUNT/$USER/eullm-models
#     export REFLEXBENCH_CACHE=/scratch/$SBATCH_ACCOUNT/$USER/reflexbench-cache
#     export DECISION=/scratch/$SBATCH_ACCOUNT/$USER/jev-style-0.8b.gguf
#
#     SMALL=qwen3-4b LARGE=qwen3-32b sbatch tools/lumi/sbatch_autobench.slurm
#     SMALL=qwen3-8b LARGE=qwen3.6-27b-q8_0 sbatch tools/lumi/sbatch_autobench.slurm
#
# The sets are downloaded on first use into $REFLEXBENCH_CACHE, at pinned
# revisions, and checked; put it on scratch, where a second job finds them.
# Stage 1's answers are kept beside the job's output (ANSWERS, by default per
# pair), so a job repeated with another decision model only routes.
#
# Prints the AutoBench tables and one `BENCH_RESULT {...}` line with
# `"mode": "autobench"`: per set, the calls Reflex avoids and the accuracy it
# keeps against always-large, the router's latency, and end to end, per
# concurrency, the share of answers identical to the chosen model's alone and
# the time to the first token.

#SBATCH --job-name=eullm-autobench
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --time=03:00:00
#SBATCH --output=eullm-autobench-%j.out

set -uo pipefail

# LUMI's /usr/bin/python3 is 3.6 and ab_data.py needs 3.8+ (a walrus at line
# 242 is a parse-time SyntaxError there). campaign_env.sh picks the newest
# interpreter on PATH, then cray-python, as every other LUMI job here does.
# shellcheck source=campaign_env.sh
source "${EULLM_REPO:-$PWD}/tools/lumi/campaign_env.sh" || exit 1

EULLM_BIN="${EULLM_BIN:?set EULLM_BIN to the eullm binary}"
DECISION="${DECISION:?set DECISION to the decision model GGUF}"
export SMALL="${SMALL:?set SMALL to the small model id (see: eullm list)}"
export LARGE="${LARGE:?set LARGE to the large model id}"
export EULLM_MODELS_DIR="${EULLM_MODELS_DIR:-$HOME/.eullm/models}"
export CTX="${CTX:-8192}"
export SETS="${SETS:-gsm8k,arc-easy,arc-challenge,mmlu}"
export LIMIT="${LIMIT:-200}"
export E2E_LIMIT="${E2E_LIMIT:-50}"
export CONCURRENCY="${CONCURRENCY:-1,4,16}"
EMBED="${EMBED:-}"
ANSWERS="${ANSWERS:-$PWD/eullm-autobench-$SMALL-$LARGE.answers.jsonl}"
BENCH="${BENCH:-$(dirname "$(realpath "$0")")/../../bench/reflexbench/autobench.py}"
# sbatch runs a copy of this script from the spool directory: the bench is
# found from where the job was submitted instead.
[ -f "$BENCH" ] || BENCH="$SLURM_SUBMIT_DIR/bench/reflexbench/autobench.py"
PORT=$((11434 + SLURM_JOB_ID % 900))
URL="http://127.0.0.1:$PORT"
PREFIX="eullm-autobench-$SLURM_JOB_ID"
export ROCR_VISIBLE_DEVICES=0

UTIL_LOG="$PREFIX.util.log"
start_sampler() {
    ( while :; do
        echo "T $(date +%s)"
        rocm-smi --showuse --showmeminfo vram 2>/dev/null | grep -E 'GPU use \(%\)|VRAM Total Used'
        sleep 5
      done ) > "$UTIL_LOG" 2>&1 &
    SAMPLER_PID=$!
}

SERVER_PID=
cleanup() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
    kill "${SAMPLER_PID:-0}" 2>/dev/null
    return 0
}
trap cleanup EXIT
trap 'cleanup; exit 143' TERM INT

echo "=== configuration ==="
printf '  %-12s %s\n' node "$(hostname)" small "$SMALL" large "$LARGE" \
    decision "$DECISION" embed "${EMBED:-none}" ctx "$CTX" sets "$SETS" limit "$LIMIT" \
    e2e "$E2E_LIMIT items at $CONCURRENCY"
rocm-smi --showproductname 2>&1 | sed 's/^/  /'

# An audit trail of the job's own, beside its output: every route and every
# routed answer, for whoever checks the report against it.
EULLM_AUDIT_DIR="$PWD/$PREFIX.audit" "$EULLM_BIN" serve --port "$PORT" \
    --max-loaded-models 2 --ctx-size "$CTX" \
    --auto-model "$SMALL=Short everyday requests, simple facts and quick rewrites" \
    --auto-model "$LARGE=Multi-step reasoning, maths, code, analysis and long answers" \
    --decision-model "$DECISION" ${EMBED:+--embedding-model "$EMBED"} \
    > "$PREFIX.server.log" 2>&1 &
SERVER_PID=$!

echo
echo "=== waiting for the server and its warm-up (both candidates loaded) ==="
WARM_START=$(date +%s)
for _ in $(seq 1 900); do
    grep -q "Auto routing warm-up:" "$PREFIX.server.log" && break
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "ERROR: the server exited"; tail -30 "$PREFIX.server.log"; exit 1; }
    sleep 2
done
sed 's/\x1b\[[0-9;]*m//g' "$PREFIX.server.log" | grep -E "Auto routing warm-up|fit:|Loading model" | sed 's/^/  /'
echo "  ready after $(( $(date +%s) - WARM_START )) s (model loads from Lustre included)"

echo
echo "=== AutoBench ==="
start_sampler
"$PY" "$BENCH" --url "$URL" --small "$SMALL" --large "$LARGE" \
    ${EMBED:+--embed-model "$EMBED"} --sets "$SETS" --limit "$LIMIT" \
    --e2e-limit "$E2E_LIMIT" --concurrency "$CONCURRENCY" --answers "$ANSWERS" \
    --out "$PREFIX.json" --details "$PREFIX.jsonl"
STATUS=$?
kill "${SAMPLER_PID:-0}" 2>/dev/null

"$PY" - "$PREFIX.json" "$UTIL_LOG" "$STATUS" <<'PY'
import json, os, re, sys
report_path, util_path, status = sys.argv[1], sys.argv[2], int(sys.argv[3])
try:
    report = json.load(open(report_path, encoding="utf-8"))
except (OSError, ValueError):
    report = {}
use = [int(m) for m in re.findall(r"GPU use \(%\):\s*(\d+)", open(util_path, encoding="utf-8", errors="replace").read())]
vram = [int(m) for m in re.findall(r"VRAM Total Used Memory \(B\):\s*(\d+)", open(util_path, encoding="utf-8", errors="replace").read())]
sets = {}
for row in report.get("results", []):
    if row["router"] != "reflex (own)":
        continue
    m = row["metrics"]
    sets[row["set"]] = {
        "calls_avoided": m.get("calls_avoided"),
        "delta": m.get("delta"),
        "delta_ci95": m.get("delta_ci95"),
        "auroc": m.get("auroc"),
        "decision_ms": m.get("decision_ms"),
        "reasons": m.get("reasons"),
    }
e2e = [
    {
        "set": r["set"],
        "concurrency": r["concurrency"],
        "same_answer": r["metrics"]["same_answer"],
        "ttft_ms": r["metrics"]["ttft_ms"],
        "ttft_large_alone_ms": r["metrics"]["ttft_large_alone_ms"],
        "errors": r["metrics"]["error_count"],
    }
    for r in report.get("end_to_end", [])
]
result = {
    "site": "lumi-g", "mode": "autobench", "status": status,
    "small": os.environ.get("SMALL"), "large": os.environ.get("LARGE"),
    "decision_models": report.get("decision_models"),
    "ctx": int(os.environ.get("CTX", 0)), "limit": report.get("limit"),
    "reflex": sets, "kill_criterion": report.get("kill_criterion"), "end_to_end": e2e,
    "generation_evictions": report.get("generation_evictions_during_run"),
    "gpu_use_mean": round(sum(use) / len(use), 1) if use else None,
    "vram_max_gib": round(max(vram) / 2**30, 2) if vram else None,
    "job": os.environ.get("SLURM_JOB_ID"),
}
print("BENCH_RESULT " + json.dumps(result))
PY
exit "$STATUS"

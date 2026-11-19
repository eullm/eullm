#!/bin/bash
# law_rewrites.py with a teacher too large for one node: llama.cpp's RPC
# backend spreads it over several (Mistral Large 4, 1T parameters, 605 GB in
# UD-Q4_K_XL: four nodes of 4x A100 64 GB).
#
#   RR_TEACHER=<first part of the GGUF> RR_OUT=<rewrites.jsonl> RR_ARGS="--styles caso --limit 200" \
#       bash forge/scripts/leonardo/submit_chain.sh \
#       forge/scripts/leonardo/sbatch_rpc_rewrites.slurm 4 -J eullm-rpc-pilot
#
# The first node runs llama-server and holds four GPUs itself; every other
# node runs one rpc-server per GPU, and llama-server sends each its share of
# the layers at load. Needs the RPC build (sbatch_build_llama_server.slurm
# with LCPP_RPC=1, in build-cuda-rpc). Each link loads the model again
# (minutes, most of them the transfer to the other nodes), then asks the
# rewrites until its two hours are nearly up; law_rewrites.py resumes from
# its output, so the next link carries on.
#
# rpc-server has no authentication: it listens on the cluster's internal
# network for the job's two hours only, on ports of its own, and is stopped
# with the job. Never run it outside a job.
#
# Whole nodes, by construction: listed in tests/test_leonardo_job_shape.py's
# WHOLE_NODE. Submit it in the evening: on 2026-11-04 whole nodes asked for at
# 20:37 started between 06:05 and 06:54 (forge/CLAUDE.md, rule 1).
#
# Settable:
#   RR_TEACHER  first part of the teacher GGUF (required)
#   RR_OUT      rewrites JSONL (required)
#   RR_ARGS     more law_rewrites.py options (--styles, --limit, --teacher, ...)
#   RR_PARALLEL requests in flight (default 8)
#   RR_CTX      context per request (default 8192)
#   RR_REPO     checkout to run (default: the one this file was submitted from)
#
#SBATCH --job-name=eullm-rpc-rewrites
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=480G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail

SELF="$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null |
        sed -n 's/.*Command=\([^ ]*\).*/\1/p' | head -1)"
if [ -n "${RR_REPO:-}" ]; then
    REPO="$RR_REPO"
elif [ -n "$SELF" ] && [ -f "$SELF" ]; then
    REPO="$(cd "$(dirname "$SELF")/../../.." && pwd)"
else
    REPO="${EULLM_REPO:-$WORK/eullm}"
fi
# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/env.sh"

TEACHER="${RR_TEACHER:-}"
OUT="${RR_OUT:-}"
BIN="${LCPP_RPC_BIN:-$LCPP_DIR/build-cuda-rpc/bin}"
PARALLEL="${RR_PARALLEL:-8}"
CTX="${RR_CTX:-8192}"
# shellcheck disable=SC2206
NORMS=(${LR_NORMS:-$WORK/norms/legislazione_*.chunks.jsonl})
# shellcheck disable=SC2206
EXCLUDE=(${LR_EXCLUDE:-$WORK/eval/closed-book.jsonl})
PORT=$((20000 + ${SLURM_JOB_ID:-0} % 20000))
RPC_PORT=$((PORT + 1))
fail=0
[ -s "$TEACHER" ] || { echo "[rr] no teacher GGUF at '$TEACHER'" >&2; fail=1; }
[ -n "$OUT" ] || { echo "[rr] RR_OUT is required" >&2; fail=1; }
for b in llama-server rpc-server; do
    [ -x "$BIN/$b" ] || { echo "[rr] no $b in $BIN: sbatch_build_llama_server.slurm with LCPP_RPC=1" >&2; fail=1; }
done
[ "$fail" -eq 0 ] || exit 1
if [ -s "$BIN/../modules.txt" ] && command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC2046
    module load $(cat "$BIN/../modules.txt") 2>/dev/null || true
fi

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname)}")
LOGDIR="$(dirname "$OUT")/rpc-logs"; mkdir -p "$LOGDIR"
echo "[rr] nodes ${NODES[*]}, teacher $TEACHER"

# One rpc-server per GPU on every node but the first.
RPC=()
for node in "${NODES[@]:1}"; do
    srun --nodes=1 --ntasks=1 --nodelist="$node" --gres=gpu:4 --overlap bash -c "
        for g in 0 1 2 3; do
            CUDA_VISIBLE_DEVICES=\$g '$BIN/rpc-server' -H 0.0.0.0 -p \$(($RPC_PORT + g)) \
                > '$LOGDIR/rpc-$node-\$g-${SLURM_JOB_ID:-0}.log' 2>&1 &
        done
        wait" &
    for g in 0 1 2 3; do RPC+=("$node:$((RPC_PORT + g))"); done
done
for addr in "${RPC[@]}"; do
    host=${addr%:*}; port=${addr#*:}
    for _ in $(seq 120); do (exec 3<>"/dev/tcp/$host/$port") 2>/dev/null && break; sleep 5; done
    (exec 3<>"/dev/tcp/$host/$port") 2>/dev/null || { echo "[rr] rpc-server $addr never answered" >&2; exit 1; }
done
echo "[rr] ${#RPC[@]} rpc-servers up"

RPC_LIST=$(IFS=,; echo "${RPC[*]}")
"$BIN/llama-server" -m "$TEACHER" ${RPC_LIST:+--rpc "$RPC_LIST"} -ngl 999 -np "$PARALLEL" \
    -c $((PARALLEL * CTX)) --host 127.0.0.1 --port "$PORT" --no-webui \
    > "$LOGDIR/server-${SLURM_JOB_ID:-0}.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null; kill $(jobs -p) 2>/dev/null' EXIT
for _ in $(seq 720); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
    kill -0 $SERVER_PID 2>/dev/null || { echo "[rr] llama-server died:" >&2; tail -20 "$LOGDIR/server-${SLURM_JOB_ID:-0}.log" >&2; exit 1; }
    sleep 5
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "[rr] the model did not load in an hour" >&2; exit 1; }
echo "[rr] teacher loaded after ${SECONDS} s"

# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/heartbeat.sh"
heartbeat_start

# shellcheck disable=SC2086
python "$REPO/forge/scripts/law_rewrites.py" --norms "${NORMS[@]}" --exclude "${EXCLUDE[@]}" \
    --out "$OUT" --url "http://127.0.0.1:$PORT" --parallel "$PARALLEL" \
    --stop-after $((6600 - SECONDS)) ${RR_ARGS:-}

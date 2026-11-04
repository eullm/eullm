#!/bin/bash
# The held-out exam (make_norm_exam.py) put to several models, on one GPU.
#
#   sbatch --export=ALL,EXAM_MODELS="v0.2=/path/merged qwen3-4b-instruct=/path/snapshot" \
#       forge/scripts/leonardo/sbatch_norm_exam.slurm
#
# One answers file per model in EXAM_OUT, then grade them with
#   sbatch --dependency=afterok:<this job> \
#       --export=ALL,JUDGE_DIR=$EXAM_OUT,JUDGE_QUIET=1 sbatch_judge.slurm
#
# QUIET THROUGHOUT. legal_eval.py runs with --quiet, so this log holds totals
# and never a question: the exam is only held out as long as nobody who
# improves the models reads it, and logs get pasted into conversations.
#
# One GPU, like stage-3 training: a 4B model in bf16 answers the whole exam,
# 16 questions at a time, in minutes, where the CPU took an hour for ten.
#
# Settable:
#   EXAM_MODELS  space-separated label=path pairs (required)
#   EXAM_ITEMS   the exam (default $WORK/eval/norm-exam.jsonl)
#   EXAM_OUT     answers directory (default $WORK/eval/norm-exam-answers)
#   EXAM_NORMS   legislation files: set to ALSO ask each model open book,
#                as "<label>-open" (default: closed book only)
#   EXAM_OPEN_ONLY  1 = with EXAM_NORMS, ask open book only
#   EXAM_EMBEDDER / EXAM_RERANKER  open book with BM25 fused with this
#                embedding model and reordered by this reranker
#                (eullm_forge.eval.dense) instead of BM25 alone
#   EXAM_OPEN_TAG  suffix of the open-book label (default "open"); give the
#                hybrid runs another one ("hyb") so both sit in one directory
#                and compare_graded.py sets them side by side
#   EXAM_OPEN_ARGS  extra legal_eval.py options for the open-book run only:
#                "--absent" with EXAM_OPEN_TAG=absent is the abstention exam
#                (each item's own article taken out of the texts); the
#                closed-book run of the same job is its no-text half.
#                abstain_summary.py counts both
#   EXAM_GGUF    space-separated label=file.gguf pairs: those labels of
#                EXAM_MODELS answer with the GGUF through llama-server, the
#                model's path giving the tokenizer (legal_eval.py --gguf), so
#                "grpo-v04-q4=<merged>" with "grpo-v04-q4=<its q4_k_m.gguf>"
#                is the shipped file asked exactly like the bf16 "grpo-v04".
#                Needs the CUDA llama-server (sbatch_build_llama_server.slurm)
#   EXAM_MEMORY  space-separated label=dir pairs: those labels of EXAM_MODELS
#                answer with this n-gram memory grafted on (ngram_train.py,
#                <out>/memory; legal_eval.py --ngram-memory), so
#                "r2l-mem=<merged r2l>" with "r2l-mem=<run>/memory" is the
#                same model asked with its memory, beside a plain "r2l"
#   EXAM_ADAPTER space-separated label=dir pairs: a LoRA adapter merged first
#                (the control of ngram_train.py --mode lora, <out>/adapter)
#   EXAM_REPO    checkout to run (default: the one this file was submitted from)
#
#SBATCH --job-name=eullm-exam
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail

SELF="$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null |
        sed -n 's/.*Command=\([^ ]*\).*/\1/p' | head -1)"
if [ -n "${EXAM_REPO:-}" ]; then
    REPO="$EXAM_REPO"
elif [ -n "$SELF" ] && [ -f "$SELF" ]; then
    REPO="$(cd "$(dirname "$SELF")/../../.." && pwd)"
else
    REPO="${EULLM_REPO:-$WORK/eullm}"
fi
# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/env.sh"

ITEMS="${EXAM_ITEMS:-$WORK/eval/norm-exam.jsonl}"
OUT="${EXAM_OUT:-$WORK/eval/norm-exam-answers}"
echo "[exam] repo   $REPO"
echo "[exam] commit $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "[exam] items  $ITEMS ($(wc -l < "$ITEMS" 2>/dev/null || echo 0) lines)"
echo "[exam] out    $OUT"

[ -s "$ITEMS" ] || { echo "[exam] no exam at $ITEMS — run make_norm_exam.py" >&2; exit 1; }
[ -n "${EXAM_MODELS:-}" ] || { echo "[exam] EXAM_MODELS is required" >&2; exit 1; }
fail=0
for pair in $EXAM_MODELS; do
    path="${pair#*=}"
    [ -f "$path/config.json" ] || { echo "[exam] no model at $path (${pair%%=*})" >&2; fail=1; }
done
pair_of() {  # pair_of LIST LABEL: the value LIST gives this label, or nothing
    local p
    for p in $1; do [ "${p%%=*}" = "$2" ] && { echo "${p#*=}"; return; }; done
}
gguf_of() { pair_of "${EXAM_GGUF:-}" "$1"; }
for pair in ${EXAM_MEMORY:-}; do
    [ -f "${pair#*=}/config.json" ] || { echo "[exam] no n-gram memory at ${pair#*=} (${pair%%=*})" >&2; fail=1; }
done
for pair in ${EXAM_ADAPTER:-}; do
    [ -f "${pair#*=}/adapter_config.json" ] || { echo "[exam] no LoRA adapter at ${pair#*=} (${pair%%=*})" >&2; fail=1; }
done
SERVER="${LCPP_SERVER:-$LCPP_DIR/build-cuda/bin/llama-server}"
for pair in ${EXAM_GGUF:-}; do
    [ -f "${pair#*=}" ] || { echo "[exam] no GGUF at ${pair#*=} (${pair%%=*})" >&2; fail=1; }
    [ -x "$SERVER" ] || { echo "[exam] no llama-server at $SERVER -- sbatch_build_llama_server.slurm" >&2; fail=1; }
done
[ "$fail" -eq 0 ] || exit 1
# The CUDA build records the modules it was compiled with; its libraries
# come from them.
if [ -n "${EXAM_GGUF:-}" ] && [ -s "$(dirname "$SERVER")/../modules.txt" ] &&
   command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC2046
    module load $(cat "$(dirname "$SERVER")/../modules.txt") 2>/dev/null || true
fi
mkdir -p "$OUT"

# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/heartbeat.sh"
heartbeat_start

EVAL=(python "$REPO/forge/scripts/legal_eval.py" --items "$ITEMS" --quiet
      --csv "$OUT/keywords.csv")
# An answers file with one line per item is a finished run: legal_eval.py
# writes the file only after the last answer, so it is either whole or absent.
# Skipping those makes the job a link of a chain (sbatch_exam_round.slurm,
# ROUND_EXAM_LINKS): a 2 h link that runs out of time leaves what it finished,
# and the next link does the rest instead of starting over.
WANT="$(grep -c . "$ITEMS")"
done_already() { [ -s "$1" ] && [ "$(grep -c . "$1")" -eq "$WANT" ]; }
# EXAM_OPEN_ONLY=1 asks open book only: the closed-book run is a second full
# pass that the open-book decisions do not use.
TAG="${EXAM_OPEN_TAG:-open}"
RETRIEVAL=()
if [ -n "${EXAM_EMBEDDER:-}" ]; then
    RETRIEVAL=(--embedder "$EXAM_EMBEDDER" --retrieval-cache "${EXAM_RETRIEVAL_CACHE:-$WORK/eval/retrieval-cache}")
    [ -z "${EXAM_RERANKER:-}" ] || RETRIEVAL+=(--reranker "$EXAM_RERANKER")
fi
status=0
for pair in $EXAM_MODELS; do
    label="${pair%%=*}"; path="${pair#*=}"
    echo "[exam] === $label ==="
    EXTRA=()
    g="$(gguf_of "$label")"
    [ -z "$g" ] || EXTRA=(--gguf "$g" --llama-server "$SERVER")
    m="$(pair_of "${EXAM_MEMORY:-}" "$label")"
    [ -z "$m" ] || EXTRA+=(--ngram-memory "$m")
    m="$(pair_of "${EXAM_ADAPTER:-}" "$label")"
    [ -z "$m" ] || EXTRA+=(--adapter "$m")
    if [ "${EXAM_OPEN_ONLY:-0}" != "1" ]; then
        if done_already "$OUT/answers-$label.jsonl"; then
            echo "[exam] $label: answered already, skipped"
        else
            "${EVAL[@]}" "$path" --label "$label" "${EXTRA[@]}" \
                --answers "$OUT/answers-$label.jsonl" || status=1
        fi
    fi
    if [ -n "${EXAM_NORMS:-}" ]; then
        if done_already "$OUT/answers-$label-$TAG.jsonl"; then
            echo "[exam] $label-$TAG: answered already, skipped"
        else
            # shellcheck disable=SC2086
            "${EVAL[@]}" "$path" --label "$label-$TAG" --norms $EXAM_NORMS "${RETRIEVAL[@]}" "${EXTRA[@]}" \
                ${EXAM_OPEN_ARGS:-} --answers "$OUT/answers-$label-$TAG.jsonl" || status=1
        fi
    fi
done
exit "$status"

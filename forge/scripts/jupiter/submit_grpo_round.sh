#!/usr/bin/env bash
# One GRPO round on JUPITER, start to finish, in one submission from a login
# node: Leonardo's r2l recipe (submit_grpo_round.sh there) on a model of
# ours. Nothing to time by hand afterwards.
#
#     eullm && bash forge/scripts/jupiter/submit_grpo_round.sh NAME START [LINKS] [MAX_STEPS]
#
#   NAME       the round: $FORGE_RUN_DIR/grpo/NAME, jobs forge-grpo-NAME
#   START      merged model the policy starts from
#   LINKS      GRPO links of GRPO_LINK_H hours (default 2 of 6)
#   MAX_STEPS  grpo_train.py --max-steps (default 300)
#
# What it does, each step waiting on the one it needs:
#
#   0. on this login node, the prompts: Leonardo's (GRPO_PROMPTS, copied to
#      $FORGE_DATA_DIR/grpo/leonardo) without the articles of the tests
#      drawn here and of the private sets, the note taken out of 70% of
#      them as on Leonardo (filter_grpo_prompts.py), into NAME/prompts.jsonl;
#   1. a smoke test, three steps on 48 prompts with the judge, into
#      NAME-smoke (40 minutes at most); the training waits for it to end
#      well (afterok), so a broken setup costs one short job;
#   2. LINKS GRPO links (sbatch_grpo.slurm), each resuming from the last
#      checkpoint; the one that finds the adapter merges it (NAME/merged);
#   3. the merged model asked open book, no note, as the product asks
#      (sbatch_teacher_eval.slurm with EVAL_SETS): the trap test and its
#      rewrites, the LexIA traps, the practical exam, the held-out exam;
#      and START on whichever of those it has no answers for yet;
#   4. the judge on all of them (sbatch_judge.slurm, JUDGE_ANSWERS_DIR).
#
# Settable:
#   GRPO_PROMPTS     prompts files (default $FORGE_DATA_DIR/grpo/leonardo/*.jsonl)
#   GRPO_LR          learning rate (default 1e-5, as on Leonardo)
#   GRPO_EXTRA       more grpo_train.py options
#   GRPO_LINK_H      hours a link (default 6)
#   GRPO_SMOKE=0     no smoke test
#   GRPO_START_LABEL START's label in the teacher eval (default dqp-d-q9b)
#   GRPO_START_SETS  the sets START may be asked (default all five; those it
#                    has answers for are skipped): narrower when some are
#                    being asked of it by a job already queued
#   GRPO_AFTER       a job id: the round waits for it to end (afterany), and
#                    START may not exist yet, the merged model of a round
#                    still running (give the id of its last GRPO link). No
#                    smoke test, and START is not asked: its own round asks it.
#                    Prompts files that job writes (sbatch_grpo_prompts.slurm)
#                    may not exist yet either: the first GRPO link filters
#                    them then, with the command left in NAME/filter.sh
#
# Idempotent: a round with jobs in the queue, or merged already, is not
# submitted again; the prompts once written are kept.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ $# -ge 2 ] || { sed -n '2,16p' "$0" >&2; exit 2; }
NAME=$1 START=$2 LINKS=${3:-2} STEPS=${4:-300}
# shellcheck source=env.sh
source "$HERE/env.sh"
# shellcheck source=sbatch_retry.sh
source "$HERE/sbatch_retry.sh"
REPO="$(cd "$HERE/../../.." && pwd)"
R="$FORGE_RUN_DIR" D="$FORGE_DATA_DIR" LOGS="$FORGE_RUN_DIR/logs"
P="${FORGE_PRIVATE_EVAL:-$FORGE_HOME/private-eval}"
OUT="$R/grpo/$NAME"
LIBS="${FORGE_EXTRA_LIBS:-$FORGE_HOME/libs-qwen35}"
read -r -a PROMPTS <<<"${GRPO_PROMPTS:-$(ls "$D"/grpo/leonardo/*.jsonl 2>/dev/null | tr '\n' ' ')}"
LR="${GRPO_LR:-1e-5}"
LINK_H="${GRPO_LINK_H:-6}"
START_LABEL="${GRPO_START_LABEL:-dqp-d-q9b}"
SETS="trap-test=$D/eval/trap-test.jsonl trap-test-para=$D/eval/trap-test-para.jsonl lexia-trappole=$P/lexia-trappole.jsonl practical-dev=$P/practical-dev.jsonl norm-exam=$D/eval/norm-exam.jsonl"

bad=0
case "$LINKS$STEPS" in *[!0-9]*) echo "[!!] LINKS and MAX_STEPS are numbers"; bad=1 ;; esac
[ -n "${GRPO_AFTER:-}" ] || [ -f "$START/config.json" ] || { echo "[!!] no start model at $START"; bad=1; }
case "${GRPO_AFTER:-0}" in *[!0-9]*) echo "[!!] GRPO_AFTER is a job id"; bad=1 ;; esac
[ "${#PROMPTS[@]}" -gt 0 ] || { echo "[!!] no prompts in $D/grpo/leonardo: copy Leonardo's there"; bad=1; }
DEFER=0
for f in "${PROMPTS[@]}"; do
    [ -s "$f" ] && continue
    if [ -n "${GRPO_AFTER:-}" ]; then DEFER=1; echo "[..] $f not written yet: filtered by the first link"
    else echo "[!!] missing $f"; bad=1; fi
done
for s in $SETS; do [ -s "${s#*=}" ] || { echo "[!!] missing ${s#*=}"; bad=1; }; done
[ -d "$FORGE_HOME/libs-grpo" ] || { echo "[!!] no $FORGE_HOME/libs-grpo: setup_grpo_libs.sh"; bad=1; }
[ -d "$LIBS" ] || { echo "[!!] no $LIBS: setup_qwen35_libs.sh"; bad=1; }
[ "$bad" = 0 ] || { echo "[!!] nothing submitted"; exit 1; }
wait_for_slurm || exit 1
if [ -n "$(squeue -h -u "$USER" -n "forge-grpo-$NAME" -o %i)" ]; then
    echo "[ok] $NAME: GRPO already queued, nothing submitted"; exit 0
fi
if [ -f "$OUT/merged/config.json" ]; then
    echo "[ok] $NAME: already merged ($OUT/merged), nothing submitted"; exit 0
fi
mkdir -p "$OUT" "$LOGS"
cd "$REPO" || exit 1

if [ ! -s "$OUT/prompts.jsonl" ]; then
    EXCL=()
    for s in $SETS; do EXCL+=("${s#*=}"); done
    for f in "$D"/eval/norm-exam*.jsonl; do [ -s "$f" ] && EXCL+=("$f"); done
    FILTER=(python "$REPO/forge/scripts/filter_grpo_prompts.py" "${PROMPTS[@]}"
            --exclude-exam "${EXCL[@]}" --out "$OUT/prompts.jsonl")
    if [ "$DEFER" = 1 ]; then
        printf '%q ' "${FILTER[@]}" >"$OUT/filter.sh" && echo >>"$OUT/filter.sh"
        echo "[ok] prompts to be filtered by the first link: $OUT/filter.sh"
    else
        "${FILTER[@]}" || { echo "[!!] nothing submitted"; exit 1; }
    fi
fi
printf '%s\n' "$START" >"$OUT/start"

sub() {    # sub <sbatch options...>: the job id, or a failure said aloud
    local j
    j=$(sbatch_retry --parsable --account "$SBATCH_ACCOUNT" "$@") || { echo "[!!] sbatch failed: $*" >&2; return 1; }
    echo "${j%%;*}"
}
export FORGE_EXTRA_LIBS="$LIBS"
dep=""
if [ -n "${GRPO_AFTER:-}" ]; then
    dep="--dependency=afterany:$GRPO_AFTER"
    echo "[ok] after job $GRPO_AFTER, which writes $START"
elif [ "${GRPO_SMOKE:-1}" != 0 ] && [ ! -f "$OUT-smoke/adapter/adapter_config.json" ]; then
    smoke=$(GRPO_MODEL="$START" GRPO_PROMPTS="$OUT/prompts.jsonl" GRPO_OUT="$OUT-smoke" GRPO_NO_MERGE=1 \
            GRPO_ARGS="--lr $LR --max-steps 3 --limit 48" \
            sub -J "forge-grpo-$NAME" --time=00:40:00 --output="$LOGS/grpo-%j.out" \
                forge/scripts/jupiter/sbatch_grpo.slurm) || exit 1
    echo "[ok] smoke test: job $smoke"
    dep="--dependency=afterok:$smoke"
fi
last=""
for i in $(seq 1 "$LINKS"); do
    # shellcheck disable=SC2086
    last=$(GRPO_MODEL="$START" GRPO_PROMPTS="$OUT/prompts.jsonl" GRPO_OUT="$OUT" \
           GRPO_ARGS="--lr $LR --max-steps $STEPS ${GRPO_EXTRA:-}" \
           sub -J "forge-grpo-$NAME" --time="$LINK_H:00:00" --output="$LOGS/grpo-%j.out" $dep \
               forge/scripts/jupiter/sbatch_grpo.slurm) || exit 1
    echo "[ok] GRPO link $i/$LINKS: job $last"
    dep="--dependency=afterany:$last"
done

ids=""
ev=$(EVAL_MODEL="$OUT/merged" EVAL_LABEL="grpo-$NAME" EVAL_SETS="$SETS" \
     sub --time=03:00:00 --dependency="afterany:$last" --output="$LOGS/teacher-eval-%j.out" \
         forge/scripts/jupiter/sbatch_teacher_eval.slurm) || exit 1
ids=":$ev"
echo "[ok] $NAME asked the five sets: job $ev"
missing=""
for s in $SETS; do
    [ -z "${GRPO_AFTER:-}" ] || break
    case " ${GRPO_START_SETS:-${s%%=*}} " in *" ${s%%=*} "*) ;; *) continue ;; esac
    [ -s "$R/teacher-eval/answers-${s%%=*}-$START_LABEL.jsonl" ] || missing="$missing $s"
done
if [ -n "$missing" ]; then
    j=$(EVAL_MODEL="$START" EVAL_LABEL="$START_LABEL" EVAL_SETS="${missing# }" \
        sub --time=02:00:00 --output="$LOGS/teacher-eval-%j.out" \
            forge/scripts/jupiter/sbatch_teacher_eval.slurm) || exit 1
    ids="$ids:$j"
    echo "[ok] $START_LABEL on$missing: job $j"
fi
j=$(JUDGE_ANSWERS_DIR="$R/teacher-eval" \
    sub --dependency="afterany$ids" --output="$LOGS/judge-%j.out" forge/scripts/jupiter/sbatch_judge.slurm) || exit 1
echo "[ok] graded by job $j"
echo "     then: python forge/scripts/score_trap_test.py $D/eval/trap-test.jsonl $R/teacher-eval/answers-trap-test-*.jsonl"

#!/usr/bin/env bash
# Follow a Leonardo job and say something the moment it ends.
#
# `tail -f` on a job log is not enough: when the job dies the tail just sits
# there, and the difference between "loading a 61 GB checkpoint" and "died
# twenty minutes ago" is invisible. That happened, and the failure went
# unnoticed for twenty minutes. This waits for the job to appear, follows the
# interesting lines while it runs, and when the job leaves the queue prints
# its final state from sacct plus the error that killed it.
#
# Usage:
#   bash forge/scripts/leonardo/watch_job.sh 56775410
#   bash forge/scripts/leonardo/watch_job.sh 56775410 --all   # unfiltered
#
# Run it from the directory whose logs/ holds the job. $EULLM_RUN_DIR is the
# fallback, not the override — see the comment on RUN_DIR below.

set -uo pipefail

JOBID="${1:?Usage: $0 <jobid> [--all]}"
MODE="${2:-}"
POLL="${WATCH_POLL_SECONDS:-30}"

# The directory you are standing in wins, when it actually holds this job's
# log. $EULLM_RUN_DIR used to take precedence, and because `env.sh` sets it to
# whichever run is pinned, `cd`-ing into the pilot's directory and asking to
# watch a pilot job silently followed Phase 1's directory instead: it sat
# there polling for a log that was never going to appear, while the job it was
# asked about ran elsewhere. Two runs sharing a login shell is now normal, so
# the environment cannot be the one that decides.
RUN_DIR="${EULLM_RUN_DIR:-$PWD}"
if compgen -G "$PWD/logs/*-${JOBID}.out" >/dev/null 2>&1; then
    RUN_DIR="$PWD"
elif [ -d "$PWD/logs" ] && ! compgen -G "$RUN_DIR/logs/*-${JOBID}.out" >/dev/null 2>&1; then
    # Neither has the log yet — the job may still be queued. Prefer where the
    # user is standing, since that is where they expect it to land.
    RUN_DIR="$PWD"
fi
cd "$RUN_DIR" || { echo "no such directory: $RUN_DIR" >&2; exit 2; }

# Lines worth seeing while the job runs. Deliberately not `error`: the corpus
# is Italian legal text and every other page contains the word "errore",
# which matched thousands of times the first time this filter was written.
# Unanchored on purpose. The heartbeat writes from a background subshell
# while the training process writes its own lines, and with no synchronisation
# between the two a [hb] line regularly lands mid-line. A '^\[hb\]' anchor
# therefore drops most of them: measured on a 10h run, the log held 645
# heartbeat lines and the anchored pattern showed none.
KEEP='\[hb\]|pre-flight|trainable params|'"'"'loss'"'"':|eval_loss|Traceback|OutOfMemoryError|RuntimeError|ValueError|AssertionError|out of memory|Killed'

# Patterns that identify the actual cause in the post-mortem.
CAUSE='Traceback|OutOfMemoryError|RuntimeError|ValueError|KeyError|TypeError|AssertionError|out of memory|Killed|CUDA error'

echo "[watch] job $JOBID, polling every ${POLL}s, logs in $RUN_DIR/logs/"

# 1. Wait for the log to appear (the job may still be queued).
while ! compgen -G "logs/*-${JOBID}.out" >/dev/null; do
    if ! squeue -j "$JOBID" -h >/dev/null 2>&1; then
        echo "[watch] job $JOBID is not in the queue and wrote no log — never started?"
        break
    fi
    sleep "$POLL"
done

LOG="$(compgen -G "logs/*-${JOBID}.out" | head -1 || true)"
if [ -z "$LOG" ]; then
    echo "[watch] no log file for $JOBID" >&2
    exit 1
fi
echo "[watch] following $LOG"

# 2. Follow it, and stop the tail when the job leaves the queue. `tail -f`
#    would otherwise outlive the job and look like it is still working.
if [ "$MODE" = "--all" ]; then
    tail -f -n +1 "$LOG" &
else
    tail -f -n +1 "$LOG" | grep --line-buffered -E "$KEEP" &
fi
TAIL_PGID=$!
trap 'kill -- -$$ 2>/dev/null' EXIT INT TERM

while squeue -j "$JOBID" -h -o '%T' 2>/dev/null | grep -qE 'PENDING|RUNNING|COMPLETING'; do
    sleep "$POLL"
done
sleep 3            # let the last writes land
kill "$TAIL_PGID" 2>/dev/null

# 3. Post-mortem, loudly.
echo
echo "==================== job $JOBID ENDED ===================="
sacct -j "$JOBID" --format=JobID%16,State,ExitCode,Elapsed,Start,End 2>/dev/null |
    head -4
STATE="$(sacct -n -j "$JOBID" --format=State 2>/dev/null | head -1 | tr -d ' ')"
if [ "${STATE:-}" != "COMPLETED" ]; then
    echo
    echo "---- last matching cause lines in $LOG ----"
    grep -nE "$CAUSE" "$LOG" | tail -15
    echo "---- last 15 lines of the log ----"
    tail -15 "$LOG"
fi
echo "=========================================================="

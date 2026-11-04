#!/bin/bash
# A daily snapshot of where the allocation's calendar is going, that keeps
# itself alive.
#
# Leonardo has no user cron (`crontab` is not permitted) and `scrontab` is
# disabled cluster-wide, so there is no scheduler to hang this off. What does
# work is a job that re-submits itself with `--begin=now+1day` before
# exiting: the successor sits PENDING for a day, which costs nothing, then
# does the same. It runs until the allocation ends and then stops on its own.
#
# It lives on the serial partition and takes about three seconds of one core.
# Putting it on boost_usr_prod would reserve four A100s to run some Python.
#
# Submit it from the directory the snapshots should land in, since --output
# is relative to where sbatch was run:
#
#   cd "$WORK/eullm_runs/qstats"
#   sbatch "$WORK/eullm-v11/forge/scripts/leonardo/sbatch_queue_stats.slurm"
#   squeue --me -n eullm-queue-stats     # must always show exactly one PENDING
#
# The chain's one failure mode is a re-submission that does not happen — a
# full queue, an expired allocation, a path that moved. There is no next run
# to notice it, so every such case here exits non-zero and says why: a break
# shows as FAILED in `sacct` instead of vanishing. Check for that one PENDING
# row now and then, and if it is gone, submit this again.
#
#SBATCH --job-name=eullm-queue-stats
#SBATCH --partition=lrd_all_serial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=%x-%j.out

set -uo pipefail

# Absolute paths: $WORK does not exist inside a batch job.
BASE="${EULLM_BASE:-/leonardo_work/AIFAC_P02_1147}"
OUT="$BASE/eullm_runs/qstats"
ALLOC_START="${EULLM_ALLOC_START:-2026-09-02}"
ALLOC_END="${EULLM_ALLOC_END:-20261102}"

# Deliberately NOT $EULLM_REPO, even though that is the obvious name.
# `env.sh` sets EULLM_REPO to the checkout a running chain is pinned to, and
# `sbatch` exports the submitting shell's environment by default — so the
# first version of this file, submitted from a shell that had sourced env.sh,
# went looking for itself inside the *frozen* tree, found nothing, and died on
# its first run. Its `|| true` swallowed the message. A variable this script
# owns cannot be hijacked by one that means something else.
REPO="${EULLM_QSTATS_REPO:-$BASE/eullm-v11}"
SELF="$REPO/forge/scripts/leonardo/sbatch_queue_stats.slurm"
TOOL="$REPO/forge/scripts/leonardo/queue_stats.py"

mkdir -p "$OUT"

# `python3` is not a fixed thing on this cluster. RHEL 8's system interpreter
# is 3.6, which cannot even parse `from __future__ import annotations`, while
# `module load python/3.11.7` puts a modern one ahead of it on PATH. A job
# that inherits the module from its submitter works; the same job submitted
# from a clean shell dies with a SyntaxError. That happened interactively on
# a different login node, and the batch chain had the same latent fault.
#
# So the interpreter is chosen here rather than assumed, and a failure to
# find one is loud.
usable_python() {
    command -v "$1" >/dev/null 2>&1 &&
        "$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 7) else 1)' \
            >/dev/null 2>&1
}

PY=""
# An explicit override is still checked. Pointing it at a path that does not
# exist, or at the system 3.6, should say so here rather than fail somewhere
# less obvious a few lines later.
if [ -n "${EULLM_QSTATS_PYTHON:-}" ]; then
    if usable_python "$EULLM_QSTATS_PYTHON"; then
        PY="$EULLM_QSTATS_PYTHON"
    else
        echo "[qstats] FATAL: EULLM_QSTATS_PYTHON='$EULLM_QSTATS_PYTHON' is missing" >&2
        echo "[qstats] or older than 3.7 — the daily chain stops here." >&2
        exit 1
    fi
else
    for candidate in python3.12 python3.11 python3.10 python3.9 python3; do
        if usable_python "$candidate"; then
            PY="$candidate"
            break
        fi
    done
fi
if [ -z "$PY" ]; then
    echo "[qstats] FATAL: no python >= 3.7 on PATH — the daily chain stops here." >&2
    echo "[qstats] Try 'module load python/3.11.7' before submitting, or set" >&2
    echo "[qstats] EULLM_QSTATS_PYTHON to an interpreter." >&2
    exit 1
fi
echo "[qstats] python $("$PY" -c 'import sys; print(sys.version.split()[0])') ($PY)"

# Check before acting. A self-perpetuating job that cannot find itself has no
# second chance: there is no next run to notice the problem.
for f in "$SELF" "$TOOL"; do
    if [ ! -f "$f" ]; then
        echo "[qstats] FATAL: $f does not exist — the daily chain stops here." >&2
        echo "[qstats] Point EULLM_QSTATS_REPO at the checkout to run from," >&2
        echo "[qstats] then submit this script again." >&2
        exit 1
    fi
done

# `|| true` only covers a genuine runtime failure of a tool we have just
# confirmed exists — never a missing file, which is what it used to hide.
STAMP="$(date +%Y%m%d_%H%M)"
if ! "$PY" "$TOOL" "$ALLOC_START" \
        --json "$OUT/queue_stats_$STAMP.json"; then
    echo "[qstats] snapshot failed; re-arming anyway so one bad day is not fatal" >&2
fi

# The budget, frozen RAW and unparsed.
#
# `saldo -b` prints the one number nothing else on the system knows: the
# monthly quota. On 2026-09-22 it read monthTotal 19672 against a total of
# 40000, which is exactly 40000 x 30/61 — a static proration of the
# allocation over a 30-day window, not a recomputation of what is left over
# the months that remain. The distinction decides whether three weeks of
# under-use can be made up in October or are simply gone, and a single
# reading cannot tell them apart: inside the first month `monthConsumed` and
# `totConsumed` are the same number.
#
# The answer arrives at the October boundary and only then. If monthTotal is
# still 19672 the unspent quota is lost; if it has risen, it carried over.
# `saldo` keeps no history, so a day not captured is a day that never
# existed — the same reason priority_snapshot.py exists.
#
# Stored as raw text on purpose. The column layout is the vendor's and may
# change; parsing it now would turn a formatting change into a silent wrong
# number, where a text file stays readable whatever happens. Parse it later,
# when there is something specific to ask it.
if command -v saldo >/dev/null 2>&1; then
    if saldo -b > "$OUT/saldo_$STAMP.txt" 2>&1; then
        echo "[qstats] budget snapshot written"
    else
        echo "[qstats] saldo failed — continuing" >&2
    fi
else
    echo "[qstats] no saldo on PATH — skipping the budget snapshot" >&2
fi

# The priority snapshot, taken here because it is the one that cannot be
# taken later.
#
# `queue_stats` reads `sacct`, which keeps weeks of history: a missed day is
# recoverable. `sprio`, `sshare` and `sinfo` keep nothing — they report the
# current instant, so a day not captured is a day that never existed. Asking
# a person to remember a daily command is how that gap gets made.
#
# Failure here never stops the chain: the calendar snapshot above has already
# been written, and one absent file is better than a broken re-arm.
PRIO="$REPO/forge/scripts/leonardo/priority_snapshot.py"
if [ -f "$PRIO" ]; then
    if ! "$PY" "$PRIO" --json "$OUT/priority_$STAMP.json"; then
        echo "[qstats] priority snapshot failed — continuing" >&2
    fi
else
    echo "[qstats] no priority_snapshot.py at $PRIO — skipping (update the checkout)" >&2
fi

# The queue as it is now, raw: node states and our own jobs with their
# reasons and planned starts. The "nothing starts" days are told by these.
{ date -Is; sinfo -p boost_usr_prod -h -o "%T %D"; echo "--";
  squeue --me -h -o "%i|%j|%T|%b|%l|%V|%S|%r" ; } > "$OUT/queue_$STAMP.txt" 2>&1 || true

# Re-arm, unless the allocation is over. Without this guard the chain would
# keep trying past the end date and fill the directory with failures.
if [ "$(date +%Y%m%d)" -lt "$ALLOC_END" ]; then
    cd "$OUT" || exit 1
    # QSTATS_EVERY: how often (sbatch --begin syntax, e.g. 3hours). Daily by
    # default; sprio and sinfo keep no history, so on days the queue misbehaves
    # (the Fast Lane's 4-11-2026) a snapshot every few hours is the evidence.
    EVERY="${QSTATS_EVERY:-1day}"
    if sbatch --begin="now+$EVERY" "$SELF"; then
        echo "[qstats] re-armed: next run in $EVERY"
    else
        # Exit non-zero so the break shows as FAILED in sacct. Silence is what
        # made the first failure cost a day.
        echo "[qstats] FATAL: re-submission failed — the daily chain stops here." >&2
        exit 1
    fi
else
    echo "[qstats] allocation ended ($ALLOC_END); not re-arming."
fi

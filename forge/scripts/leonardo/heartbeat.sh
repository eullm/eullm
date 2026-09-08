# Heartbeat for long Leonardo jobs — source this, then `heartbeat_start`.
#
# Why this exists: LLaMA-Factory and DeepSpeed go silent for the two longest
# phases of a job. Reading a 61 GB checkpoint off Lustre and building the
# ZeRO-3 partitions for a MoE with ~18,400 expert tensors take tens of
# minutes each and print nothing, and transformers' progress bars are
# suppressed anyway because tqdm disables itself when stderr is not a TTY —
# which it never is under sbatch. The result is a log that stops for half an
# hour while four A100s sit at 0%, with no way to tell a working job from a
# deadlocked one without ssh-ing around /proc from a login node.
#
# So the job reports on itself: one line a minute with the three numbers that
# actually distinguish the states.
#
#   [hb] +0480s  gpu util 0,0,0,0 %  mem 16780,16780,16780,16780 MiB  rss 203.6 GiB  load 20.33
#
#   * GPU util 0 and rss CLIMBING      → reading the checkpoint, be patient
#   * GPU util 0 and rss FLAT          → ZeRO-3 init, or a hang; compare loads
#   * GPU util 90-100                  → training
#
# Usage, in an sbatch script, after the environment is set up:
#
#     source "$EULLM_REPO/forge/scripts/leonardo/heartbeat.sh"
#     heartbeat_start
#     ... the long-running command ...
#
# heartbeat_start installs an EXIT trap that stops the background loop, so
# the caller does not have to remember to.

HEARTBEAT_INTERVAL="${HEARTBEAT_INTERVAL:-60}"

# Sum the resident memory of this job's python workers, in GiB. Reported as
# a total rather than per-rank because the useful signal is "is it still
# growing", and one number is easier to eyeball in a log than four.
_heartbeat_rss_gib() {
    local pids
    # Anchor on the python interpreter, not on the word alone: a bare
    # 'llamafactory|distill' also matches the shell that happens to carry
    # those words in its own command line — a `tail` on the log, or this
    # very script — and silently reports their RSS as the job's.
    pids="$(pgrep -u "$(id -u)" -d, -f 'python[^ ]* .*(llamafactory|distill)' \
            2>/dev/null)" || true
    [ -z "$pids" ] && { printf '?'; return; }
    ps -o rss= -p "$pids" 2>/dev/null |
        awk '{s += $1} END {if (s) printf "%.1f", s / 1048576; else printf "?"}'
}

_heartbeat_gpu() {
    command -v nvidia-smi >/dev/null 2>&1 || { printf 'n/a n/a'; return; }
    local util mem
    util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits \
            2>/dev/null | paste -sd, -)"
    mem="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
           2>/dev/null | paste -sd, -)"
    printf '%s %s' "${util:-n/a}" "${mem:-n/a}"
}

_heartbeat_loop() {
    local t0=$SECONDS
    while true; do
        # shellcheck disable=SC2046  # deliberate word split: util and mem
        set -- $(_heartbeat_gpu)
        printf '[hb] +%05ds  gpu util %s %%  mem %s MiB  rss %s GiB  load %s\n' \
            "$((SECONDS - t0))" "$1" "$2" "$(_heartbeat_rss_gib)" \
            "$(cut -d' ' -f1 /proc/loadavg)"
        sleep "$HEARTBEAT_INTERVAL"
    done
}

heartbeat_start() {
    _heartbeat_loop &
    _HEARTBEAT_PID=$!
    # Kill the loop however the job ends — normal exit, error under `set -e`,
    # or SIGTERM from SLURM at the walltime cap. Without the signal traps a
    # preempted job would leave the loop running until the step is reaped.
    trap '[ -n "${_HEARTBEAT_PID:-}" ] && kill "$_HEARTBEAT_PID" 2>/dev/null' \
        EXIT INT TERM
    echo "[hb] heartbeat every ${HEARTBEAT_INTERVAL}s (pid $_HEARTBEAT_PID)"
}

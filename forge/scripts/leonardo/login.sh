# The working environment on Leonardo, ready at every login.
#
# Sourced from ~/.bashrc, for interactive shells only:
#
#   [[ $- == *i* ]] && source /leonardo_work/AIFAC_F02_1400/eullm/forge/scripts/leonardo/login.sh
#
# Interactive only, because scp, rsync and the datamovers open non-interactive
# shells too, and anything printed there breaks the transfer.
#
# What it does: loads Python, sources env.sh (the venv, HF_HOME, the offline
# switches inside jobs, SBATCH_ACCOUNT from EULLM_ACCOUNT), goes to the
# checkout, and prints one screen: the budget, the queue, the space. Nothing
# is submitted, installed or pulled. The account and the paths stay in
# ~/.bashrc (EULLM_ACCOUNT, EULLM_VENV, EULLM_REPO, EULLM_RUN_DIR, ...).
#
# Shortcuts it defines:
#   coda     the queue, one line per job
#   spazio   space used in $WORK, the fast scratch and $HOME
#   budget   budget.sh: GPU-hours spent against the allocation, from sacct
#   carico   how loaded this login node is now, also written to the load log
#
# Every login also appends one line, in the background, to
# $WORK/eullm_runs/login-load.tsv: time, node, users, load, cores, processes
# waiting on the disk (state D), CPU split from vmstat (user, system, idle,
# I/O wait), available memory and swap used. On 2026-11-04, after the
# storage outage, the login nodes ran at load 76-96 with 19-27% I/O wait;
# the Fast Lane's final report describes the infrastructure with a series
# of these numbers, not with screenshots.

module load python/3.11.7 2>/dev/null

if [ -n "${EULLM_REPO:-}" ] && [ -f "$EULLM_REPO/forge/scripts/leonardo/env.sh" ]; then
    # shellcheck disable=SC1091
    source "$EULLM_REPO/forge/scripts/leonardo/env.sh"
    cd "$EULLM_REPO" || true
else
    echo "[login] EULLM_REPO is not set, or has no forge/scripts/leonardo/env.sh: environment not loaded"
fi

EULLM_FAST="${EULLM_FAST:-/leonardo_scratch/fast/${EULLM_ACCOUNT:-}}"
export EULLM_FAST

alias coda='squeue --me -o "%.10i %.28j %.2t %.10M %.11l %.5D %.12b %.22R"'
alias spazio='_eullm_space'
alias budget='bash "$EULLM_REPO/forge/scripts/leonardo/budget.sh"'

_eullm_space() {
    local label path
    for label in WORK FAST HOME; do
        case "$label" in
            WORK) path="$WORK" ;; FAST) path="$EULLM_FAST" ;; HOME) path="$HOME" ;;
        esac
        [ -d "$path" ] || continue
        df -h --output=used,size,pcent "$path" 2>/dev/null | tail -1 |
            awk -v l="$label" '{printf "== space  %-5s %6s of %-6s (%s)\n", l, $1, $2, $3}'
    done
}

EULLM_LOAD_LOG="${EULLM_LOAD_LOG:-$WORK/eullm_runs/login-load.tsv}"
_eullm_login_load() {
    local cpu users dstate mem
    cpu="$(vmstat 1 2 2>/dev/null | awk 'NR == 2 {for (i = 1; i <= NF; i++) c[$i] = i}
        NR == 4 {printf "%s\t%s\t%s\t%s", $c["us"], $c["sy"], $c["id"], $c["wa"]}')"
    users="$(who 2>/dev/null | awk '{print $1}' | sort -u | wc -l)"
    dstate="$(ps -eo stat= 2>/dev/null | grep -c '^D')"
    mem="$(free -g 2>/dev/null | awk '/^Mem:/ {m = $7} /^Swap:/ {s = $3} END {printf "%s\t%s", m, s}')"
    [ -s "$EULLM_LOAD_LOG" ] ||
        printf 'time\thost\tusers\tload1\tcores\tproc_D\tus\tsy\tid\twa\tmem_avail_G\tswap_used_G\n' > "$EULLM_LOAD_LOG"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$(hostname -s)" "$users" \
        "$(cut -d' ' -f1 /proc/loadavg)" "$(nproc)" "$dstate" "${cpu:-?\t?\t?\t?}" "${mem:-?\t?}" \
        >> "$EULLM_LOAD_LOG"
}
alias carico='_eullm_login_load; column -t "$EULLM_LOAD_LOG" | sed -n "1p;\$p"'
[ -d "$(dirname "$EULLM_LOAD_LOG")" ] && { _eullm_login_load >/dev/null 2>&1 & disown; } 2>/dev/null

_eullm_login_summary() {
    local venv="${VIRTUAL_ENV:-none}" branch commit running pending
    branch="$(git -C "$EULLM_REPO" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
    commit="$(git -C "$EULLM_REPO" log -1 --format='%h %cd' --date=format:'%d/%m %H:%M' 2>/dev/null || echo '?')"
    echo "== EuLLM  account ${EULLM_ACCOUNT:-?}  venv ${venv##*/}  repo ${EULLM_REPO##*/} ($branch, $commit)"
    # saldo -b columns: account start end total localCluster totConsumed % monthTotal monthConsumed
    saldo -b 2>/dev/null | awk -v a="${EULLM_ACCOUNT:-}" 'toupper($1) == toupper(a) {
        printf "== budget %d of %d GPU-h (%s%%), month %d of %d, until %s\n",
               $6 / 8, $4 / 8, $7, $9 / 8, $8 / 8, $3 }'
    running="$(squeue --me -h -t R -o %i 2>/dev/null | wc -l)"
    pending="$(squeue --me -h -t PD -o %i 2>/dev/null | wc -l)"
    echo "== queue  $running running, $pending pending   (coda: the list)"
    _eullm_space
    [ "$running" -gt 0 ] || echo "== [!!] no job is running: the GPUs of the allocation are idle"
}
_eullm_login_summary

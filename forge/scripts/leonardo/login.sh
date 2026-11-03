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

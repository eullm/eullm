#!/bin/bash
# Copies directories of an expired project's $WORK into the new project's,
# inside Leonardo (Lustre to Lustre, nothing leaves the cluster), and checks
# the copy itself.
#
# Written for the move from the Playground AIFAC_P02_1147 to the Fast Lane
# AIFAC_F02_1400 (November 2026). Long rsyncs on the login nodes were killed
# by their CPU-time limit in October, so the copy runs as a job on the serial
# partition, where nothing is billed to the GPU budget.
#
#   COPY_DIRS   directories to copy, relative to COPY_FROM (required)
#   COPY_FROM   default /leonardo_work/AIFAC_P02_1147
#   COPY_TO     default /leonardo_work/AIFAC_F02_1400
#   COPY_DRY=1  measure only: what would be copied and how much, per directory
#   COPY_HEAVY=1  also copy the heavy weights (checkpoints, merged models,
#               safetensors, .bin, .pt); by default they are left behind
#
# Measure first, then copy, then the log says TUTTO COPIATO or what is still
# different:
#
#   sbatch -A AIFAC_F02_1400 --export=ALL,COPY_DRY=1,COPY_DIRS="eullm_runs eval" \
#       forge/scripts/leonardo/sbatch_copy_project.slurm
#   sbatch -A AIFAC_F02_1400 --export=ALL,COPY_DIRS="eullm_runs eval" \
#       forge/scripts/leonardo/sbatch_copy_project.slurm
#
# A copy cut short by the time limit is finished by submitting the same
# command again: rsync copies only what is missing.
#
#SBATCH --job-name=eullm-copy
#SBATCH --partition=lrd_all_serial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

set -uo pipefail

FROM="${COPY_FROM:-/leonardo_work/AIFAC_P02_1147}"
TO="${COPY_TO:-/leonardo_work/AIFAC_F02_1400}"
: "${COPY_DIRS:?set COPY_DIRS to the directories to copy, relative to $FROM}"

EXCLUDES=()
if [ "${COPY_HEAVY:-0}" != 1 ]; then
    EXCLUDES=(--exclude='checkpoint-*' --exclude='merged*' --exclude='model*.safetensors'
              --exclude='pytorch_model*.bin' --exclude='*.pt' --exclude='.remerge-*')
fi

command -v rsync >/dev/null || { echo "[copy] FATAL: no rsync on this node" >&2; exit 1; }
[ -d "$FROM" ] || { echo "[copy] FATAL: $FROM does not exist or is not readable" >&2; exit 1; }
[ -d "$TO" ] || { echo "[copy] FATAL: $TO does not exist" >&2; exit 1; }
for d in $COPY_DIRS; do
    [ -e "$FROM/$d" ] || { echo "[copy] FATAL: $FROM/$d does not exist" >&2; exit 1; }
done

echo "[copy] from $FROM"
echo "[copy] to   $TO"
echo "[copy] dirs $COPY_DIRS"
echo "[copy] heavy weights $([ "${COPY_HEAVY:-0}" = 1 ] && echo included || echo left behind)"
df -h "$TO" | tail -1 | awk '{print "[copy] space at the destination: " $4 " free of " $2}'
echo

if [ "${COPY_DRY:-0}" = 1 ]; then
    total=0
    for d in $COPY_DIRS; do
        stats="$(cd "$FROM" && rsync -a --relative -n --stats "${EXCLUDES[@]}" "./$d" "$TO/")" ||
            { echo "[copy] FATAL: cannot measure $d" >&2; exit 1; }
        bytes="$(echo "$stats" | awk -F': ' '/Total transferred file size/ {gsub(/[^0-9]/, "", $2); print $2}')"
        bytes="${bytes:-0}"
        total=$(( total + bytes ))
        awk -v b="$bytes" -v d="$d" 'BEGIN {printf "[copy] %-40s %8.1f GB\n", d, b / 1e9}'
    done
    awk -v b="$total" 'BEGIN {printf "[copy] %-40s %8.1f GB to copy\n", "TOTAL", b / 1e9}'
    exit 0
fi

start=$(date +%s)
failed=0
for d in $COPY_DIRS; do
    echo "[copy] $d ..."
    if ! (cd "$FROM" && rsync -a --relative "${EXCLUDES[@]}" "./$d" "$TO/"); then
        echo "[copy] rsync of $d ended with an error, see above" >&2
        failed=1
    fi
done
echo "[copy] copied in $(( ($(date +%s) - start) / 60 )) min"

# The copy checks itself: a second, dry rsync lists every file still
# different. Directories alone do not count.
check_failed=0
diff_list="$(cd "$FROM" && for d in $COPY_DIRS; do
    rsync -an --relative --out-format='%n' "${EXCLUDES[@]}" "./$d" "$TO/" || echo "CHECK FAILED: $d"
    done | grep -v '/$')"
case "$diff_list" in *"CHECK FAILED"*) check_failed=1 ;; esac
if [ "$failed" = 0 ] && [ "$check_failed" = 0 ] && [ -z "$diff_list" ]; then
    echo "[copy] TUTTO COPIATO"
elif [ "$check_failed" = 1 ] || [ "$failed" = 1 ]; then
    echo "[copy] NOT VERIFIED: rsync failed (see above); nothing here proves the copy"
    echo "$diff_list" | head -20
    exit 1
else
    echo "[copy] still different: $(echo "$diff_list" | wc -l) files, the first ones:"
    echo "$diff_list" | head -20
    echo "[copy] submit the same command again to finish"
fi
du -sh "$TO" 2>/dev/null | awk '{print "[copy] destination now holds " $1}'

#!/usr/bin/env bash
# Detect which logical CPUs are Cortex-A720 ("big"/"medium") vs Cortex-A520
# ("little") on an Armv9.2-A tri-cluster SoC (written for the CIX P1 /
# Radxa Orion O6, but the MIDR-based method is generic to any big.MEDIUM.LITTLE
# Cortex-A720/A520 design).
#
# Why detection instead of a hardcoded core range: on this board, core
# numbering is firmware-dependent and non-contiguous by tier. A SystemReady
# UEFI firmware exposes only 8 cores (big+medium A720, little A520 disabled
# outright); Radxa's own BSP firmware exposes all 12 across 5 cpufreq
# policies. A real capture on one firmware showed cpu0 = big, cpu1-4 =
# little, cpu5-8 = medium, cpu9-11 = big — i.e. NOT a contiguous block per
# tier. Never assume a fixed range; always detect on the actual board.
#
# Method: read MIDR_EL1 per core (bits [15:4] = primary part number) —
# Cortex-A720 = 0xd81, Cortex-A520 = 0xd80 (confirmed against the Linux
# kernel's arch/arm64/include/asm/cputype.h and independently against
# pytorch/cpuinfo's src/arm/uarch.c). Cortex-A720 cores are then split into
# "big" vs "medium" tiers by cpufreq's cpuinfo_max_freq, since both tiers
# share the same MIDR part number and only differ in clock ceiling.
set -euo pipefail

A720_PART="0xd81"
A520_PART="0xd80"

declare -a a720_cores=()
declare -a a520_cores=()
declare -a unknown_cores=()

for cpu_path in /sys/devices/system/cpu/cpu[0-9]*; do
    [ -d "$cpu_path" ] || continue
    n=$(basename "$cpu_path" | sed 's/cpu//')
    midr_file="$cpu_path/regs/identification/midr_el1"
    if [ ! -r "$midr_file" ]; then
        echo "cpu$n: cannot read $midr_file (need root, or kernel too old to expose it)" >&2
        continue
    fi
    midr=$(cat "$midr_file")
    part_hex=$(printf '0x%03x' "$(( (midr >> 4) & 0xFFF ))")
    case "$part_hex" in
        "$A720_PART") a720_cores+=("$n") ;;
        "$A520_PART") a520_cores+=("$n") ;;
        *) unknown_cores+=("$n ($part_hex)") ;;
    esac
done

if [ "${#a520_cores[@]}" -gt 0 ]; then
    mapfile -t a520_cores < <(printf '%s\n' "${a520_cores[@]}" | sort -n)
fi
echo "Cortex-A520 (little): ${a520_cores[*]:-none detected}"

if [ "${#unknown_cores[@]}" -gt 0 ]; then
    echo "Unrecognized MIDR part number on: ${unknown_cores[*]} (not A720/A520 - different SoC/silicon rev?)" >&2
fi

if [ "${#a720_cores[@]}" -eq 0 ]; then
    echo "No Cortex-A720 cores detected - check that this is actually running on the target board." >&2
    exit 1
fi

# Split A720 cores into big/medium by max frequency. If every A720 core
# reports the same cpuinfo_max_freq (some firmware/BIOS configs may not
# expose the distinction), everything falls into "big" as a safe fallback.
declare -A freq_of
for n in "${a720_cores[@]}"; do
    f="/sys/devices/system/cpu/cpu$n/cpufreq/cpuinfo_max_freq"
    freq_of[$n]=$( [ -r "$f" ] && cat "$f" || echo 0 )
done

max_freq=0
for n in "${a720_cores[@]}"; do
    [ "${freq_of[$n]}" -gt "$max_freq" ] && max_freq=${freq_of[$n]}
done

big=() medium=()
for n in "${a720_cores[@]}"; do
    if [ "${freq_of[$n]}" -eq "$max_freq" ]; then
        big+=("$n")
    else
        medium+=("$n")
    fi
done

# `cpu[0-9]*` globs (and thus every array above) come out in shell
# lexicographic order (cpu10/cpu11 before cpu2) on 10+ core boards like this
# one - sort numerically so the reported core lists read sanely.
if [ "${#big[@]}" -gt 0 ]; then
    mapfile -t big < <(printf '%s\n' "${big[@]}" | sort -n)
fi
if [ "${#medium[@]}" -gt 0 ]; then
    mapfile -t medium < <(printf '%s\n' "${medium[@]}" | sort -n)
fi

echo "Cortex-A720 big    (max ~$((max_freq / 1000)) MHz): ${big[*]}"
echo "Cortex-A720 medium: ${medium[*]:-none distinguished (same max freq as big - treated as big above)}"

# What to actually run.
#
# The rule is "exclude A520, use every A720" — NOT "use only the fastest
# A720". big and medium are the same microarchitecture with the same ISA
# extensions (i8mm, SVE2) and differ only in clock ceiling, so dropping the
# medium ones throws away real throughput to chase a few percent of clock.
# A520 is a different, far weaker core, and ggml puts a barrier after every
# operation, so one A520 in the pool does gate the whole batch — that is the
# case pinning exists for.
#
# Measured on this board (2026-08-11, 8 A720 visible, no A520): going from
# --threads 4 to --threads 8 gave 1.71-1.80x on prefill across three models,
# i.e. 95% of the theoretical `n_threads x slowest_clock` ceiling. Pinning to
# the 2 cores at the single highest clock would have cost about 3x. See
# docs/arm-cix-p1-cpu-profile.md § 7.2.
a720_csv=$(IFS=,; echo "${a720_cores[*]}")
echo ""
if [ "${#a520_cores[@]}" -gt 0 ]; then
    echo "Recommended for this run (pin away from the A520 cores):"
    echo "  taskset -c $a720_csv eullm run <model> --no-ui --threads ${#a720_cores[@]} < /dev/null > server.log 2>&1 &"
else
    echo "Recommended for this run (no A520 present - every core is an A720,"
    echo "so pinning would only remove cores; use them all):"
    echo "  eullm run <model> --no-ui --threads ${#a720_cores[@]} < /dev/null > server.log 2>&1 &"
fi

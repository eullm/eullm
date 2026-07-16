# CIX P1 (Armv9.2-A) CPU build profile — POSCAR WP4

CPU-only baseline for running EULLM Engine on the CIX P1 SoC (Radxa Orion O6
mini-ITX board, also sold as the smaller Orion O6N Nano-ITX with the same
SoC) — no GPU, no NPU. This covers WP4's build profile, runtime verification,
thread pinning, and the T4.1 benchmark baseline. NPU offload is explicitly
out of scope here; see the roadmap for where that would land separately.

## 1. Hardware summary

CIX P1 (silkscreened CD8180, CD8160 on early boards — functionally
identical) is a 12-core Armv9.2-A tri-cluster SoC:

| Tier | Cores | Max clock |
|---|---|---|
| Cortex-A720 "big" | 4 | ~2.6-2.8 GHz (see caveat below) |
| Cortex-A720 "medium" | 4 | ~2.4 GHz |
| Cortex-A520 "little" | 4 | ~1.8 GHz |

12 MB shared L3, single DSU. ISA extensions relevant to this profile: SVE2,
BF16, I8MM, DotProd.

**Caveats (flagging what's confirmed vs. not):**
- Early CIX P1 silicon targeted 2.8 GHz on the big cores; later production
  parts appear capped at 2.6 GHz. Don't hardcode a specific number — read
  `cpuinfo_max_freq` on the actual unit (the detection script below does
  this).
- I have not read CIX's own datasheet/TRM directly, only secondary coverage
  confirming one exists (published Dec 2025). If you need authoritative
  numbers beyond what's here, pull it from CIX's developer portal.
- Sources: [CNX Software board announcement](https://www.cnx-software.com/2024/12/18/radxa-orion-o6-mini-itx-motherboard-is-powered-by-cix-p1-12-core-armv9-soc-with-a-30-tops-ai-accelerator/), [Radxa product page](https://radxa.com/products/orion/o6/), [SBCwiki CD8180/P1](https://sbcwiki.com/docs/soc-manufacturers/cix/cd8180-p1/), [CNX Software review](https://www.cnx-software.com/2025/01/29/radxa-orion-o6-review-unboxing-debian-12-installation-and-first-benchmarks/).

## 2. Build profile

### Toolchain requirement — read this before building

`armv9.2-a` as a `-march` value needs **GCC 13+ or Clang 14+**. GCC's own
release notes confirm armv9.1-a/9.2-a/9.3-a landed in GCC 13 (GCC 12 only
has plain `armv9-a`); Clang added it in 14.0.0. **Ubuntu 22.04's default
`gcc-aarch64-linux-gnu` package is 11.2.0 and cannot parse this flag at
all** — it predates even `armv9-a` (GCC 12). If you're cross-compiling from
an Ubuntu 22.04 host (as `release-engine.yml`'s generic `aarch64-unknown-linux-gnu`
job does today, with no `-march` override), install a newer cross-toolchain
first: the `ubuntu-toolchain-r/test` PPA, the Arm GNU Toolchain from
developer.arm.com, or build from a newer host (Ubuntu 24.04's default is
new enough — verified empirically in this profile's own testing: GCC
13.3.0 from a 24.04-based cross-toolchain accepted the flag and produced a
working cross-compiled binary end to end).

Note the `+sve2+bf16+i8mm+dotprod` suffixes are technically redundant on a
new-enough compiler — GCC's own AArch64-Options table shows `armv9.1-a`
already implies `+sve+sve2+bf16+i8mm`, and dotprod is inherited further
back from `armv8.4-a`. They're kept explicit here anyway: harmless, and it
makes the intent self-documenting instead of relying on an implication
chain the reader has to look up.

### How the flag is wired

`engine/vendor/llama-cpp-rs/llama-cpp-sys-2/build.rs` already had a
cross-compile branch for generic `aarch64-unknown-linux-gnu` that hardcoded
`GGML_CPU_ARM_ARCH=armv8-a` (the lowest common denominator — correct for a
generic ARM64 release binary, but it never gave the CIX P1 its actual ISA
extensions). It now reads a `LLAMA_GGML_CPU_ARM_ARCH` env var and falls back
to the same `armv8-a` default when unset, so every existing build (CI's
generic arm64 job included) is unaffected unless it opts in.

`GGML_CPU_ARM_ARCH` itself is upstream `ggml`'s own CMake option
(`ggml/CMakeLists.txt`) — this profile does not patch the vendored
`llama.cpp` submodule at all, only how our own build script drives an
option it already exposes.

### Prebuilt binary from CI

Every tagged release now also builds `eullm-linux-arm64-cix-p1` (job
`build-arm-cix-p1` in `release-engine.yml`), on GitHub's native
`ubuntu-24.04-arm` runners — its default `gcc` is already 13.x, so no
toolchain upgrade step is needed there the way a cross-compile job would.
Grab it from the [latest release](https://github.com/eullm/eullm/releases/latest)
instead of building it yourself if a tagged version is recent enough.
**This is a separate artifact from the generic `eullm-linux-arm64`
binary on purpose** — it will SIGILL on any ARM64 host that lacks these
exact ISA extensions (Raspberry Pi, Graviton, etc.), so it's not listed in
the README's general platform table.

To iterate on the job itself without cutting a real tag (and without
paying for the other 5 platform builds, or creating a GitHub Release):
Actions tab → *Release Engine* → *Run workflow* (`workflow_dispatch`) on
this branch. That trigger runs `build-arm-cix-p1` only — every other job,
and the release step itself, are skipped on manual dispatch.

### Cross-compiling for the CIX P1

```bash
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
export CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc
export CXX_aarch64_unknown_linux_gnu=aarch64-linux-gnu-g++
export LLAMA_GGML_CPU_ARM_ARCH="armv9.2-a+sve2+bf16+i8mm+dotprod"
cargo build --release --target aarch64-unknown-linux-gnu -p eullm-engine
```

Verified in this profile's own testing: the resulting `CMakeCache.txt` shows
`GGML_CPU_ARM_ARCH:STRING=armv9.2-a+sve2+bf16+i8mm+dotprod` and
`GGML_NATIVE:BOOL=OFF`, and the actual `ggml-cpu` target's `flags.make`
carries `-march=armv9.2-a+sve2+bf16+i8mm+dotprod -fopenmp` through to the
real compile command — not just the CMake cache variable.

### Building natively, on the Orion itself

If you build directly on the board instead of cross-compiling, set
`GGML_NATIVE`'s automatic detection aside and use the same override — native
`-mcpu=native` probing runs on the actual CPU so it can work, but pinning the
exact profile explicitly is more reproducible for a WP4 baseline than
relying on what the compiler's `-mcpu=native` probe happens to detect:

```bash
LLAMA_GGML_CPU_ARM_ARCH="armv9.2-a+sve2+bf16+i8mm+dotprod" cargo build --release -p eullm-engine
```

## 3. Verify the ISA path is actually active at runtime

Start the engine and look at the `system_info:` line it prints at startup
(`common_params_get_system_info` in llama.cpp, called from `cmd_run`). It
lists every compiled-in feature the CPU backend detected as present, e.g.:

```
system_info: n_threads = 8 / 12 | CPU : NEON = 1 | ARM_FMA = 1 | FP16_VA = 1 | MATMUL_INT8 = 1 | SVE = 1 | SVE_CNT = 16 | DOTPROD = 1 | LLAMAFILE = 1 |
```

`MATMUL_INT8 = 1` is i8mm, `SVE = 1` (+ `SVE_CNT`, the vector length in
bytes) confirms SVE2, `DOTPROD = 1` confirms dotprod. If any of these are
missing, the `-march` override either wasn't picked up (check
`LLAMA_GGML_CPU_ARM_ARCH` was actually exported before the build, and that
`CMakeCache.txt` in the build directory shows it) or the toolchain silently
fell back to a lower baseline (check the compiler actually recognized the
flag — the GCC 13+/Clang 14+ requirement above).

## 4. Confirm Q4_0 quantization triggers the i8mm repack path

Q4_0 tensors get repacked at model-load time into an interleaved layout
matched to whatever the CPU backend detected (`ggml/src/ggml-cpu/repack.cpp`,
`get_tensor_traits`): SVE 8x8 if `SVE_CNT` happens to equal 32 bytes, else
i8mm-backed `q4_0_4x8` if i8mm is present (this is the CIX P1's expected
path — a 256-bit SVE vector length isn't confirmed for this core, and even
if present the codepath still needs the exact 32-byte match), else
dotprod-only `q4_0_4x4`, else the plain unpacked path.

This repack step logs at `GGML_LOG_DEBUG` — and EULLM does not suppress
logs until *after* the inference context is created (`scheduler.rs` calls
`backend.void_logs()` only once the context is up, specifically so
model-load diagnostics like this stay visible). No special verbosity flag
is needed: just check the model-load output.

```bash
eullm run <model> --no-ui --threads 4 < /dev/null 2>&1 | grep -i "repack tensor"
# or, if you started it backgrounded into server.log:
grep -i "repack tensor" server.log
```

Expect lines like:

```
ggml_backend_cpu_x86_repack_buffer_type_get_extra_bufts?: repack tensor blk.0.attn_q.weight with q4_0_4x8
```

(exact function-name prefix depends on the call site — the load-bearing
part is `with q4_0_4x8`, confirming the i8mm/smmla kernel was selected, vs
`q4_0_4x4` for dotprod-only or `q4_0_8x8` for the SVE path). Only Q4_0
tensors with `ne[1] % 4 == 0` are eligible — true for essentially every
real weight matrix, but worth knowing if a specific tensor is missing from
the log.

## 5. Thread pinning — big cores only

**Do not hardcode a core range.** Core numbering on this board is
firmware-dependent and non-contiguous by tier: a SystemReady UEFI firmware
exposes only 8 cores (big+medium A720; the 4 little A520 cores are disabled
outright), while Radxa's own BSP firmware exposes all 12 across 5 cpufreq
policies. A real capture on one firmware showed `cpu0` = big, `cpu1-4` =
little, `cpu5-8` = medium, `cpu9-11` = big — the 4 big cores are
non-contiguous even within one boot. Always detect on the actual unit.

`bench/detect_arm_big_cores.sh` does this: it reads each online core's
`MIDR_EL1` (Cortex-A720 = part `0xd81`, Cortex-A520 = part `0xd80`,
cross-checked against the Linux kernel's `cputype.h` and independently
against `pytorch/cpuinfo`), then splits the A720 cores into big/medium by
`cpuinfo_max_freq` (both tiers share the same MIDR part number). Run it on
the Orion itself:

```bash
sudo bash bench/detect_arm_big_cores.sh
```

It prints a ready-to-use `taskset` command. Combine with `--threads` set to
the detected big-core count so the scheduler doesn't also spin up worker
threads for cores it isn't pinned to:

```bash
taskset -c <big-core-list> eullm run <model> --no-ui --threads <big-core-count> < /dev/null > server.log 2>&1 &
```

`taskset` pins the whole process (and everything it spawns) regardless of
whether the build uses OpenMP or ggml's own thread pool, so it's the
recommended default. This build is OpenMP-enabled by default
(`llama-cpp-2`'s `openmp` feature, `-fopenmp` visible in the build's own
`flags.make` — see § 2), so `OMP_PROC_BIND=close` /
`OMP_PLACES="{<big-core-list>}"` work as a complementary/alternative
mechanism if you'd rather not wrap the whole process in `taskset`.

Native low-level pinning via ggml's own `ggml_threadpool_params` cpumask
(what upstream's `--cpu-mask`/`--cpu-range` CLI flags use) is not wired
through our Rust bindings at all today — only a plain thread *count*
(`--threads`) is. Wiring that through would mean new `wrapper_common.cpp` +
Rust binding surface, which is a real option later if `taskset` proves
insufficient, but wasn't pursued here as a first cut, since it touches
inference-pipeline plumbing on a change this profile can't verify itself
(no ARM hardware in the build/dev environment used to write it).

## 6. Power estimation

**Not confirmed on this board.** I checked Radxa's official hardware-info
docs, the product brief, multiple independent reviews, and the community
forum, and found no documented onboard INA-family hwmon sensor, vendor power
CLI tool, or PMIC telemetry exposed to Linux for the Orion O6. All power
figures in published reviews (idle ~11-15 W, load ~24-55+ W depending on
workload) appear to be external wall/USB meter readings, not board-reported
values.

`bench/arm_cpu_bench.py --power-sysfs-path <path>` is the documented hook:
if you find a real hwmon node on your actual unit — check
`ls /sys/class/hwmon/*/name` on the board itself, since this wasn't
confirmed remotely — point the flag at it (plus
`--power-scale-to-watts` if it's not in microwatts) and the benchmark
will sample it during the decode run and report average watts and
tok/s-per-watt. Until then, treat power/perf-per-watt as requiring an
external USB power meter around the board.

## 7. T4.1 baseline benchmark

`bench/arm_cpu_bench.py` measures prefill and decode tok/s separately
(client-side TTFT/generation-time split — the server doesn't yet report
these as separate durations; see roadmap item 0.7-B), sweeping prompt
length for prefill and repeating a fixed generation for decode:

```bash
pip install aiohttp
python bench/arm_cpu_bench.py \
    --url http://localhost:11434 --model <model> \
    --notes "4 big A720 cores, taskset 0,9,10,11, --threads 4" \
    --json t4.1-baseline-$(date +%Y%m%d).json
```

Re-run with the same flags after any build/config change and diff the JSON
to track regressions or gains against this baseline. `--notes` exists
specifically so the pinning/thread config used for a given run travels with
its numbers instead of being tribal knowledge.

## Known open items

- Exact big-core clock ceiling (2.6 vs 2.8 GHz) varies by silicon revision —
  read it on the actual unit, don't assume.
- Power telemetry is unconfirmed; treat as absent until verified with
  `ls /sys/class/hwmon/*/name` on real hardware.
- Native ggml threadpool cpumask pinning (beyond `taskset`) isn't exposed
  through our Rust bindings; see § 5.
- None of this document's runtime claims (system_info output, repack log
  lines, actual tok/s) have been executed on real CIX P1 hardware — this
  environment has no ARM device. The build itself was verified end to end
  by cross-compiling in this environment (GCC 13.3.0 cross-toolchain,
  confirmed via the resulting `CMakeCache.txt` and `flags.make`); the
  runtime behavior described in §§ 3-4 follows directly from reading the
  vendored `ggml`/`llama.cpp` source, not from an actual run. Please
  confirm on the Orion and report back if anything here doesn't match.

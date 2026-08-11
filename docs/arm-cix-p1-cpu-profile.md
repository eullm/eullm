# CIX P1 (Armv9.2-A) CPU build profile — WP4

CPU-only baseline for running EULLM Engine on the CIX P1 SoC (Radxa Orion O6
mini-ITX board, also sold as the smaller Orion O6N Nano-ITX with the same
SoC) — no GPU, no NPU. This covers WP4's build profile, runtime verification,
thread pinning, and the T4.1 benchmark baseline.

Both on-chip accelerators were since investigated and closed: the Zhouyi NPU
does not run autoregressive language models at all, and the integrated GPU
shares the CPU's DRAM and measures slower than this CPU build on both
prefill and decode. See § 9.5 for the evidence, and § 9.3 for the one thing
that does move the needle on this board, which is a discrete card in its
PCIe slot.

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

**Correction (from the first real run on an Orion, 2026-07-17):** an
earlier version of this section pointed at a `system_info:` startup line
that this binary does not actually print — `common_params_get_system_info`
/ `llama_print_system_info` is never called anywhere in `engine/src/` or
the Rust bindings. That was an unverified claim carried over from generic
llama.cpp CLI behavior; eullm's own startup banner is different and
doesn't include it. Confirmed by grepping a real run's log and finding
nothing. The repack log below (§4) is the verification method that
actually works, and doubles as proof the ISA path is active — repacking
to an i8mm-gated tier can't happen unless `ggml_cpu_has_matmul_int8()`
(and `ggml_cpu_has_neon()`) returned true, so a `_8x8` tier in that log
**is** the runtime confirmation, not just an indirect signal.

If a dedicated feature-flags line is wanted later, `llama_print_system_info()`
is a plain `llama.h` C API (not gated behind the `common` feature) that
isn't wired into eullm today — a few lines to add if this indirect check
ever isn't enough.

## 4. Confirm quantized tensors trigger the i8mm repack path

Repacking happens at model-load time, into an interleaved layout matched
to whatever the CPU backend detected (`ggml/src/ggml-cpu/repack.cpp`,
`get_tensor_traits`). The exact gating differs per quant type — read from
the source, not by analogy across types:

| Quant type | `_8x8` tier requires | Fallback tiers |
|---|---|---|
| Q4_K | `ggml_cpu_has_neon() \|\| ggml_cpu_has_matmul_int8()` (both, per the `&&`) | `q4_K_8x4` (NEON+dotprod) |
| Q6_K | same: NEON + i8mm | (no lower ARM tier) |
| Q4_0 | AVX2, **or** `ggml_cpu_has_sve() && ggml_cpu_has_matmul_int8() && ggml_cpu_get_sve_cnt() == 32` (SVE2 8x8 tier — needs a 256-bit SVE implementation specifically, not just SVE2 present) | `q4_0_4x8` (NEON+i8mm), then `q4_0_4x4` (NEON+dotprod) |

Most real models are Q4_K_M (a Q4_K/Q6_K mix — llama.cpp's own default
quant, and this engine's registry default too), not plain Q4_0, so **the
Q4_K/Q6_K row is what you'll actually see day to day.** `_8x8` for either
of those is unambiguous i8mm confirmation on ARM (no AVX2 alternative
path to confuse it with, unlike Q4_0's `_8x8`).

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

**Confirmed working on real CIX P1 hardware (Orion, 2026-07-17)**, running
`qwen3-14b` (Q4_K_M): every tensor repacked to `q4_K_8x8` or `q6_K_8x8`,
e.g. `repack: repack tensor blk.0.attn_q.weight with q4_K_8x8` — i8mm
confirmed active per the table above. Only tensors meeting the tier's
`ne[1] % N == 0` shape requirement are eligible — true for essentially
every real weight matrix, but worth knowing if a specific tensor is missing from
the log.

## 5. Thread pinning — big cores only

**Do not hardcode a core range.** Core numbering on this board is
firmware-dependent and non-contiguous by tier: a SystemReady UEFI firmware
exposes only 8 cores (big+medium A720; the 4 little A520 cores are disabled
outright), while Radxa's own BSP firmware exposes all 12 across 5 cpufreq
policies. A real capture on one firmware showed `cpu0` = big, `cpu1-4` =
little, `cpu5-8` = medium, `cpu9-11` = big — the 4 big cores are
non-contiguous even within one boot. Always detect on the actual unit.

**Confirmed on a real Orion (2026-07-17):** `htop` showed exactly 8 cores
(0-7), consistent with the SystemReady UEFI firmware case above — no A520
little cores visible, so on that unit every visible core is already a
"fast" tier and pinning is moot unless the firmware is switched to
Radxa's BSP build (which exposes all 12).

`bench/detect_arm_big_cores.sh` does this: it reads each online core's
`MIDR_EL1` (Cortex-A720 = part `0xd81`, Cortex-A520 = part `0xd80`,
cross-checked against the Linux kernel's `cputype.h` and independently
against `pytorch/cpuinfo`), then splits the A720 cores into big/medium by
`cpuinfo_max_freq` (both tiers share the same MIDR part number). Run it on
the Orion itself:

```bash
sudo bash bench/detect_arm_big_cores.sh
```

**The rule is "exclude A520, use every A720" — not "use only the fastest
A720".** Both A720 tiers are the same microarchitecture with the same ISA
extensions and differ only in clock ceiling (12% apart on the unit measured
in § 7.2), so excluding the medium ones trades away half the cores to chase
a few percent of clock. A520 is the case pinning exists for: a different,
much weaker core, and with ggml's per-operation barrier one of them in the
pool gates the whole batch.

On a unit with no A520 visible — which is what the SystemReady UEFI
firmware gives — that means **no `taskset` at all**, and `--threads` set to
the full core count:

```bash
eullm run <model> --no-ui --threads <all-A720-count> < /dev/null > server.log 2>&1 &
```

Only when A520 cores are present does pinning earn its keep, and then it is
to all the A720s, not to the fastest ones:

```bash
taskset -c <all-A720-list> eullm run <model> --no-ui --threads <A720-count> < /dev/null > server.log 2>&1 &
```

`bench/detect_arm_big_cores.sh` prints whichever of the two applies. It
recommended the fastest-tier-only form until 2026-08-11, when measuring it
showed that costs about 3× on this board — see § 7.2.

**A core list carried over from another firmware silently pins to the
intersection, not to an error.** `taskset -c 0,9,10,11` on a unit exposing
cpu0-cpu7 runs happily on core 0 alone. Always derive the list on the unit.

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

### 7.1 A/B against the generic arm64 binary — the profile is worth ~2-3×

Measured 11 August 2026 on the Orion O6, `qwen3-4b` (Q4_K_M), CPU-only,
`--threads 4`, `--ctx-size 4096`, both builds from the same release, same
GGUF file, same sweep (`--prefill-word-targets 128 512 --decode-repeats 1`):

| | generic `arm64` | `arm64-cix-p1` | gain |
|---|---:|---:|---:|
| prefill, 145 prompt tokens | 8.8 tok/s | **27.1 tok/s** | **3.1×** |
| prefill, 580 prompt tokens | 11.4 tok/s | **25.5 tok/s** | **2.2×** |
| decode, 128 tokens | 6.9 tok/s | **12.2 tok/s** | **1.8×** |

The prefill gain is the expected one and lands where predicted. The decode
gain is *not* what §7's note below predicts, and the discrepancy is
informative rather than contradictory: that note was written from a dense
14B, where the weight set is far larger than any cache and DRAM bandwidth
genuinely sets the ceiling. A 4B at Q4 is ~2.5 GB — small enough that
decode is no longer purely bandwidth-starved on this SoC, so the faster
int8 kernels (and the `q4_K_8x8` repack they enable) show up in decode too.

**These are 4-thread numbers on an 8-core board.** The A/B ratio is
unaffected — both sides carried the same handicap — but the absolute
figures are roughly half of what the board delivers: `--threads 8` is worth
another 1.71-1.83× on prefill (§ 7.2). Read this table as a comparison
between builds, not as this SoC's ceiling.

Practical reading: **use the `cix-p1` binary on this board, always**. It is
the same engine with the same defaults; the only difference is that the CPU
kernels are compiled for the ISA this SoC actually has. The generic binary
exists so ARM64 boards *without* these extensions do not SIGILL, not as an
equivalent alternative.

Still open, and worth one run: the same A/B on the dense 14B, to see how
much of the decode gain survives when the model no longer fits the caches.
If it collapses toward 1.0× there, the note below is right for large models
and this table is right for small ones — which is a more useful statement
than either alone.

**Why decode tok/s alone can look unchanged even with i8mm confirmed
active** (observed on the first real run, on a dense 14B): single-token decode is
typically memory-bandwidth-bound on CPU, not compute-bound — every token
requires streaming the *entire* quantized weight set through the core
once, regardless of how fast the multiply-accumulate itself runs. i8mm/
SVE2 speed up the arithmetic, which mostly shows up in **prefill**
(batched matmul over many tokens at once, genuinely compute-bound) and
barely moves decode, which is bounded by DRAM bandwidth instead. Don't
judge the profile's effect from the REPL's end-of-turn tok/s line alone
(that's decode-dominated when the prompt is short) — run the sweep above
and look at the `prefill` numbers specifically.

### 7.2 Thread count: use every A720 core, and do not pin

Measured 11 August 2026 on the Orion O6, same binary and flags as § 9,
`--threads 4` vs `--threads 8`, nothing else changed:

| model | 580 tok prefill | 2349 tok prefill | decode |
|---|---:|---:|---:|
| `qwen3-8b` dense | 13.8 → 24.9 (**1.80×**) | 8.6 → 15.7 (**1.83×**) | 6.9 → 7.8 (1.13×) |
| `qwen3.5-9b` hybrid | 15.5 → 27.1 (**1.75×**) | 13.3 → 22.7 (**1.71×**) | 5.6 → 6.3 (1.13×) |

Doubling the thread count buys 1.71-1.83× on prefill and ~1.13× on decode.
The split is the expected one: prefill is compute-bound and scales with
cores, decode is bandwidth-bound and does not (§ 7).

**1.83× is the theoretical ceiling here, and the dense model hits it
exactly.** ggml puts a barrier after every operation, so a batch runs at
`n_threads × slowest_clock`. This board's 8 cores run at 2500, 2500, 2400,
2400, 2300, 2300, 2200, 2200 MHz; the kernel had placed the 4-thread run on
the four fastest (0, 5, 6, 7), so the ratio to beat was
`8 × 2200 / 4 × 2400` = 1.83. Getting 95-100% of that means the prefill
path on this SoC is **not** yet memory-bandwidth-limited — added cores turn
almost entirely into work. The hybrid falls slightly short (1.71-1.75×)
because the Gated DeltaNet recurrence carries a sequential dependency along
the token axis that parallelizes worse than a plain attention matmul.

**Do not pin to the "big" cores on a board with no A520.** Until this
measurement `bench/detect_arm_big_cores.sh` recommended
`taskset -c <highest-clock cores> --threads <that many>`, which on this unit
means 2 cores out of 8 — roughly a 3× loss. The heuristic is right only when
the excluded cores are Cortex-A520: those are a different, far weaker
microarchitecture, and with a per-operation barrier one of them in the pool
gates the whole batch. A720 "big" and "medium" are the same core with a
different clock ceiling (12% apart here), so excluding the medium ones
trades 4 cores for 12% of clock. The script now recommends
`--threads <all A720>` with no `taskset` when no A520 is present.

**Two traps that cost real measurements on this board**, both worth a check
before trusting any number:

- **A `taskset` core list copied from another firmware silently pins to one
  core.** `taskset -c 0,9,10,11` on a unit that exposes cpu0-cpu7 does not
  fail: `sched_setaffinity` drops the non-existent bits and keeps the
  intersection, here `{0}`. The symptom is 4 threads at ~25% each and one
  core at 100% in `htop`. It produced 4.3 tok/s where the same run
  unpinned gives 15.0. Always derive the list on the unit
  (`bench/detect_arm_big_cores.sh`, or `lscpu -e=CPU,MAXMHZ`), never carry
  one over from a document.
- **A stale server keeps burning a core.** Up to 0.6.80 the engine does not
  exit on SIGTERM (the `#[tokio::main]` runtime drop waits on the blocking
  inference task — fixed in 0.6.81), so `pkill -f eullm` leaves the process
  alive with its ggml threads spinning, ~100% of one core and the model
  still resident. Use `pkill -9`, and confirm with `pgrep -c -f eullm`
  before starting a measurement. Note that `htop` shows one row per
  *thread*: four rows with consecutive PIDs are one process, not four.

Wait for the server to be ready rather than sleeping a fixed interval — a
21 GB MoE takes minutes to load from cold page cache, and the bench fails
with a connection error if it starts first:

```bash
until curl -sf localhost:11434/api/version >/dev/null; do sleep 5; done; echo ready
```

## 8. Hybrid/recurrent MoE models (Qwen3.5/3.6) on this profile

`qwen3.6-35b-a3b` (~21GB Q4_K_M-ish GGUF, `general.architecture =
qwen35moe`, ~3B active params/token via MoE routing) was tested live on the
Orion as a throughput case study, since MoE's lower active-param count
should suit a bandwidth-bound CPU decode path better than a dense model of
similar size. Two separate things were found and are documented here in
full because both required tracing into upstream llama.cpp source and its
issue tracker to actually resolve, not just local experimentation.

### 8.1 Decode throughput is genuinely good, and matches upstream's own numbers

Observed: ~10.2 tok/s decode vs. 2.8-3.3 tok/s for the dense `qwen3-14b` on
the same hardware. This is consistent with CPU decode being
memory-bandwidth-bound (§7) — only the ~3B active-expert weights need to
stream per token for MoE, vs. the full parameter count for a dense model.
This is a genuinely good WP4 result and needs no further work.

### 8.2 KV-cache prefix reuse does not work on this architecture — root cause, and why `--rs-seq` is not the fix

Multi-turn conversations on this model showed a small, unstable
longest-common-prefix match on reuse (e.g. 31/326, 322/608, 29/671 tokens
across separate test turns) and repeated `reused prefill failed ... likely
a recurrent/hybrid model architecture` warnings — i.e. reuse barely
engages and falls back to a full re-prefill almost every turn, unlike the
~97-99% reuse confirmed working on the dense `qwen3-14b` (§ above). Two
hypotheses were checked directly against eullm's own code and ruled out:

- **Client-side think-block stripping was suspected, then ruled out by
  reading the code.** `interactive_chat()` in `engine/src/main.rs` and
  `build_chatml()` in `engine/src/chat_template.rs` both store and resend
  every past turn's raw text verbatim, with no stripping of `<think>`
  blocks. `think_mode` only affects how the *current* turn's assistant-open
  tag renders, not past messages. This isn't the cause of the small match.
- **Wrong-architecture-clamp was suspected, then ruled out with a direct
  GGUF metadata check.** `general.architecture = qwen35moe` maps to
  `LLM_ARCH_QWEN35MOE`, which *is* on llama.cpp's
  `llm_arch_supports_rs_rollback` allow-list (confirmed in
  `src/llama-arch.cpp`) — so the reuse rejection isn't an unsupported-arch
  clamp; `n_rs_seq` really is being applied and really is insufficient at
  the values tried.

What the research (below) actually established: full re-prefill on every
turn is the current, upstream-acknowledged ceiling for this whole
architecture class on llama.cpp, independent of eullm. llama.cpp's own
server hits the identical condition and logs the near-identical message
(`tools/server/server-context.cpp`): *"forcing full prompt re-processing
due to lack of cache data (likely due to SWA or hybrid/recurrent memory,
see https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)"*.
Multiple open upstream issues track this for the Qwen3-Next/Qwen3.5/3.6
family specifically: checkpoints repeatedly invalidated
(`llama.cpp#19794`, `#24055`), multi-minute turns at moderate context
(`llama.cpp#22384`, `#20225`), and a request to at least mitigate it via
conversation truncation, closed not-planned (`llama.cpp#19838`). A
narrower, more promising upstream direction exists but wasn't ready to use
at research time: `llama.cpp#24785` ("server: add recurrent state
shrink/expand for prompt cache") explicitly targets this model family,
noting ~75% of Qwen3.6's layers carry recurrent state.

**Why `--rs-seq` (the eullm flag added to try to fix this) is not the
answer**, confirmed directly against upstream source
(`common/common.cpp`, `tools/server/server-context.cpp`,
`src/llama-memory-recurrent.cpp`):

1. Upstream's own server never uses `n_rs_seq` for prompt/conversation
   caching. It's derived exclusively from speculative-decoding draft
   length (`need_n_rs_seq()` — single digits to low teens) and is
   explicitly zeroed everywhere else (`cparams_dft.n_rs_seq = 0`). Using
   it as a general rollback window for chat history, as we tried, is
   using a speculative-decoding primitive outside its intended domain.
2. Real-hardware testing confirmed this is actively unsafe at useful
   values: recurrent-state tensors scale by `(1 + n_rs_seq)`
   (`n_rows = mem_size * (1 + n_rs_seq)`, confirmed in source). At
   `--rs-seq 64`, RES grew from ~21GB to ~44.6GB with heavy swap
   thrashing and near-zero throughput; at `--rs-seq 512`, the engine
   crashed (`ggml_new_object: not enough space in the context's memory
   pool`, a `GGML_ASSERT(obj_new)` failure inside `graph_reserve`) — the
   same failure signature as a previously-fixed ubatch-scaling bug on
   Qwen3-Next (`llama.cpp#17578`, fixed by `#17794`, "graph size scaling
   with something other than a static reservation"), now recurring for
   `n_rs_seq` specifically and, as far as could be found, not yet
   reported upstream.
3. The feature is simply new and untested at this scale: its only
   upstream test coverage (`llama.cpp#25758`) merged the day before this
   was written, against a small synthetic Qwen3.5 model — not anything
   near 35B.

**Conclusion (this is the actual finding, not a punted investigation):**
`--rs-seq` remains available in eullm (default 0) as an experimental
escape hatch, but the correct guidance is to leave it at 0 for this model
class. Full re-prefill per turn is a real, sourced, upstream architectural
limitation shared by llama.cpp's own reference server — not an eullm gap.

**Implemented (v0.6.24): `--ctx-checkpoints`/`--checkpoint-min-step`.**
A bounded pool of full-state snapshots (`state_seq_get_data_ext`/
`state_seq_set_data_ext`, the same primitives llama.cpp server uses for
`server_prompt_checkpoint`), taken at the end of each clean turn and
LRU-evicted once the pool fills. Implemented and verified correct at the
FFI level (checkpoint → restore into a fresh sequence → byte-identical
continuation, on a real TinyLlama GGUF) and safe on memory (no blowup,
unlike `n_rs_seq`). **But on real Orion hardware against the real 35B
model, it turned out not to be what fixed the problem** — structurally,
a checkpoint taken at a turn boundary has the exact same content as the
live idle slot at that instant, so for the very next turn it can never do
better than the live slot's own match. It remains useful for a different
scenario (several conversations competing for few slots, where the live
slot has been overwritten since an older checkpoint was taken), not for
a single growing conversation, which is what actually needed fixing here.

**What actually fixed it (v0.6.25 + v0.6.26), confirmed end to end on real
Orion hardware against the real hybrid 35B model:**

1. **v0.6.25 — stop retokenizing history every turn.** The real root
   cause of the small/unstable match (31/326, 322/608, 29/671 across
   separate turns) wasn't the rollback window being too small — a
   substantial match (661/1394) was rejected outright too, no partial
   credit. `build_chatml` is deterministic string concatenation, so the
   shared prefix of turn N and N+1's prompts is byte-identical *as
   text*; the bug was that eullm retokenized that shared text from
   scratch every turn, and retokenizing the same text twice isn't
   guaranteed by BPE to produce the same token ids. Fixed by matching on
   exact text prefix before tokenizing anything, reusing the already-known
   tokens for the matched portion and tokenizing only the new suffix —
   see the README's "The actual root cause of small/unstable reuse"
   section for the full writeup.
2. **v0.6.26 — `/no_think` was corrupting history reconstruction.**
   A second, independent bug with the same symptom: `eullm run --cli`'s
   sticky `/no_think` toggle injects `<think>\n</think>\n\n` right before
   generation — text the model actually decodes as part of that turn's
   resident state — but this was never re-added when reconstructing that
   turn for later history, so every `/no_think` turn permanently diverged
   from what was truly resident, compounding turn over turn. Confirmed on
   real hardware to be the *dominant* cause of the degradation in
   practice (disabling `/no_think` alone, no other change, restored ~99%
   reuse even before the v0.6.25 fix existed). Fixed by exposing
   `ChatTemplate::think_suppression_prefix()` and re-applying it when
   storing a suppressed turn into history.

**Result, real Orion hardware, real `qwen3.6-35b-a3b`, both fixes
together:** `reused N from cache` matched the *entire* previous turn's
resident length across 6+ consecutive turns spanning multiple topics
(fractals → Italian parliament), at both 4096 and 16384-token context,
with F16 and Q8_0 KV cache — zero `reused prefill failed` warnings.
Decode held at ~9-11 tok/s throughout, unaffected by growing conversation
length since prefill cost stopped scaling with it. This is the confirmed
WP4 headline result — see the README's proof-of-concept callout at the
top.

### 8.3 Quantization on ARM: smaller file, but *slower* — tested and confirmed

§4 established that ggml's ARM online-repack fast-GEMM path covers
**Q4_0, Q4_K, Q5_K, Q6_K, IQ4_NL, MXFP4, and Q8_0** (confirmed by reading
`ggml-cpu/repack.cpp`'s type-dispatch directly) — not every quant type.
Tested `unsloth/Qwen3.6-35B-A3B-GGUF`'s `UD-IQ4_NL` (18.04 GB, imatrix
calibrated) against the Q4_K_M file already in use (each with a freshly
truncated log to avoid cross-run contamination), comparing the loader's
own `- type X: N tensors` breakdown and `file size` line:

| | Q4_K_M | UD-IQ4_NL |
|---|---|---|
| Shared (both files) | 361 × f32, 251 × q8_0 | 361 × f32, 251 × q8_0 |
| Variable | 80 × Q4_K + 37 × Q5_K + 4 × Q6_K | 37 × IQ4_NL + **80 × IQ3_S** + 4 × Q6_K |
| File size | 20.60 GiB (5.11 BPW) | 16.79 GiB (4.16 BPW) |
| ARM-accelerated variable tensors | 121/121 (100%) | 41/121 (34%) |

**Result: the smaller file is slower in practice** (measured ~7.6-9.1
tok/s vs ~9.5-10.7 tok/s on identical prompts). The size reduction comes
from Unsloth's dynamic per-tensor quantization pushing the majority of
variable tensors to **IQ3_S** — confirmed absent from `repack.cpp`'s
type list, so it runs the unaccelerated generic path — instead of the
fully-accelerated `Q4_K`/`Q5_K`/`Q6_K` mix Q4_K_M uses. `IQ4_NL` itself
*is* ARM-accelerated (confirmed, same file), but only 37 of the 121
variable tensors actually used it here; the naming ("UD-IQ4_NL") reflects
the target average bit-rate, not a guarantee that IQ4_NL blocks dominate
the file. **Recommendation for CPU-only ARM: prefer Q4_K_M (or any
quantization where the variable tensors are entirely on the accelerated
list) over a smaller "IQ4_NL-named" dynamic quant** — file size and ARM
decode speed are not the same axis for this class of quant.

## 9. Prefill scaling by architecture: dense vs hybrid vs hybrid-MoE

The question this answers: **for a long prompt — RAG, document Q&A, a
stuffed context — does a hybrid (linear-attention) model actually hold its
throughput where a dense one collapses, and by how much?** The claim is
routine in model cards. It had not been measured here, and the answer turns
out to decide which model belongs on this board.

Measured 11 August 2026, Radxa Orion O6, `eullm 0.6.80`
`linux-arm64-cix-p1`, CPU only, all three models Q4_K_M, `--threads 8`, no
`taskset`, `--ctx-size 8192 --batch-size 0` (sequential engine, so all
three run the same code path — a model carrying an mmproj is otherwise
forced out of continuous batching), `--cache-type-k f16 --cache-type-v f16`,
`bench/arm_cpu_bench.py`. One model resident at a time, verified with
`pgrep -c`. No thermal throttling: every busy core held its `lscpu` maximum
for the whole run.

| | `qwen3-8b` | `qwen3.5-9b` | `qwen3.6-35b-a3b` |
|---|---:|---:|---:|
| architecture | dense | hybrid | hybrid MoE |
| full-attention layers | 36/36 | 8/32 | 1 in 4 |
| active params/token | 8B | 9B | ~3B |
| file size | 5.0 GB | 5.7 GB | 21 GB |
| **prefill, 580 tok** | 24.9 tok/s | 27.1 tok/s | **39.9 tok/s** |
| **prefill, 2349 tok** | 15.7 tok/s | 22.7 tok/s | **29.4 tok/s** |
| loss over that range | **-37%** | **-16%** | **-26%** |
| **decode, 128 tok** | 7.8 tok/s | 6.3 tok/s | **10.8 tok/s** |

The same A/B at `--threads 4`, where the two 5-6 GB models were swept over
four prompt lengths rather than two:

| prompt | `qwen3-8b` dense | `qwen3.5-9b` hybrid | ratio |
|---:|---:|---:|---:|
| 145 tok | 16.2 | 16.0 | 0.99× |
| 580 tok | 13.8 | 15.5 | 1.12× |
| 2349 tok | 8.6 | 13.3 | 1.55× |
| 4727 tok | 5.6 | 11.1 | **1.98×** |
| decode | 6.9 | 5.6 | 0.81× |

**The two start level and diverge with length.** At 145 tokens the dense is
marginally ahead — it is the smaller model, and at that length nothing else
matters. The crossover is around 500-600 prompt tokens. From there the gap
widens monotonically to 2× at 4727 tokens and is still opening. The hybrid
is the *larger* model of the two (9B vs 8B, 5.7 GB vs 5.0 GB), so this is
not a size effect: it is the 24 of 32 layers that do not pay a quadratic
cost in sequence length.

### 9.1 Fitting the curve, and what each coefficient means

Two points per model are enough to fit `time = a·n + b·n²` — `a` is
everything that scales linearly with token count (FFN, linear-attention
layers, embeddings), `b` is the full-attention term. At `--threads 8`:

| | `a` (s/token) | `b` (s/token²) |
|---|---:|---:|
| `qwen3-8b` dense | 0.0324 | **1.33e-5** |
| `qwen3.5-9b` hybrid | 0.0346 | 4.02e-6 |
| `qwen3.6-35b-a3b` MoE | **0.0221** | 5.07e-6 |

Each number says exactly what the architecture claims. The two hybrids have
a quadratic term **2.6-3.3× smaller** than the dense model's, which is the
linear-attention layers not being there. The MoE has the smallest linear
term of the three despite being by far the largest model, which is ~3B
active parameters per token instead of 8-9B.

It also corrects a reading that the raw percentages invite: the MoE's -26%
looks worse than the hybrid's -16%, but that is a percentage of a much
higher starting point. Its quadratic coefficient is only 26% above the
hybrid's and less than half the dense model's.

Extrapolating to 4727 tokens at `--threads 8`:

| | predicted | vs. measured at 4 threads × the measured thread ratio |
|---|---:|---|
| `qwen3-8b` dense | 10.5 tok/s (TTFT 451 s) | 10.2 (+3%) |
| `qwen3.5-9b` hybrid | 18.6 tok/s (TTFT 254 s) | 19.1 (-3%) |
| `qwen3.6-35b-a3b` MoE | 21.7 tok/s (TTFT 218 s) | not measured |

The two checkable predictions land within 3% of independently measured
numbers, which is why the third is worth quoting at all.

**One prediction worth testing:** the MoE starts higher but rises faster,
so the two hybrid curves cross at roughly **11,900 prompt tokens**, beyond
which `qwen3.5-9b` should be the faster of the two at prefill. That is an
extrapolation from two points each and should be treated as a hypothesis;
confirming it needs `--ctx-size 16384` and a point at ~8192 words.

### 9.2 Decode is a flat memory wall at ~37 GB/s

Decode tells a completely different story from prefill, and the numbers
converge on one explanation:

| | bytes streamed per token | decode | effective bandwidth |
|---|---:|---:|---:|
| `qwen3-8b` dense | ~5.0 GB (all weights) | 7.8 tok/s | **39.0 GB/s** |
| `qwen3.5-9b` hybrid | ~5.7 GB (all weights) | 6.3 tok/s | **35.8 GB/s** |

Two models of different size and different architecture land within 9% of
each other on effective DRAM bandwidth. That is the memory wall, visible
directly: decode on this board is not computing, it is waiting for weights.
It also explains the one place the hybrid loses — it is the bigger file, so
it streams more per token, and architecture cannot help with that.

Inverting the same relation for the MoE gives ~3.4 GB moved per token at
10.8 tok/s, consistent with ~3B active parameters at Q4 plus the
always-resident attention and embedding weights. The MoE wins decode by
moving less memory, not by being faster.

**This is the number to beat with any accelerator.** Anything that would
speed up decode on this board has to pull more than ~37 GB/s out of the
same LPDDR5 — an integrated GPU shares that bus with the CPU, so the
ceiling is shared too. Before investing in a Vulkan or OpenCL build for the
onboard Immortalis-G720, measure what the memory subsystem will actually
give (a STREAM-style test), because the honest headroom is the gap between
37 GB/s and the board's achievable peak, and nothing more.

### 9.3 The same board with a discrete GPU in its PCIe slot

Everything above is the CPU-only configuration. The same unit has also been
run with an RTX 3060 12GB in its PCIe slot, which makes this a controlled
comparison rather than a cross-machine one: same CPU, same RAM, same OS,
same binary family, one variable.

| model | Orion O6, CPU only | Orion O6 + RTX 3060 12GB |
|---|---:|---:|
| `qwen3-14b` dense (9.0 GB) | 3.0 tok/s | **33 tok/s** |
| `qwen3.6-35b-a3b` MoE (21 GB) | 10.8 tok/s | **35.6 tok/s** ¹ |

¹ with `--n-cpu-moe 24 --cache-type-k q8_0 --cache-type-v q8_0`, 10.7 GB
VRAM — the expert tensors do not fit in 12 GB, so most of them stay in
system RAM and the card carries the rest.

**The same bandwidth relation explains both columns**, which is why this is
arithmetic rather than a benchmark result to be argued with. The 3060 reads
its GDDR6 at 360 GB/s; 33 tok/s on a 9.0 GB model is 297 GB/s of effective
traffic, 82% of the card's peak — the same 82-97% band the CPU hits against
its own 40.1 GB/s (§ 9.2). Neither side is leaving anything on the table.
The 9× difference in outcome is a 9× difference in memory bandwidth, and
nothing else.

Two consequences worth stating plainly, because they are the ones that
decide how to deploy this hardware:

- **No software change closes this gap.** The CPU path is at 88-97% of what
  the board's DRAM can deliver (§ 9.2), the integrated GPU shares that same
  DRAM, and the NPU does not run autoregressive models at all (§ 9.5).
  There is no remaining inefficiency to remove.
- **The board does not have to be replaced to cross it.** It has the slot.
  The measured 11× on the dense 14B came from adding a mid-range consumer
  card to this exact unit, not from moving to a different class of machine.

### 9.4 What to run on this board

- **Long prompts, RAG, document Q&A → `qwen3.6-35b-a3b`.** Fastest at
  every prompt length measured, fastest at decode, and by a wide margin the
  strongest model of the three. 2349 tokens of retrieved context cost 80
  seconds to first token. That is not interactive, but it is usable for
  asynchronous or semi-interactive work, which the 4.5 minutes the dense
  model needs at 4727 tokens is not. Needs ~21 GB resident, so this is a
  64 GB-board recommendation.
- **Short prompts, chat, small memory → `qwen3-8b`.** Below ~600 prompt
  tokens the prefill difference vanishes and it decodes 24% faster than
  `qwen3.5-9b`.
- **`qwen3.5-9b`** sits between the two: the right choice when 21 GB is not
  available but prompts are long, and the only one of the three that reads
  images.
- The absolute numbers still say **no interactive RAG on this board**. What
  changed with this measurement is the margin: the right model is 2-4×
  faster than the wrong one at the lengths RAG actually uses, and the
  ranking is not the one file size would suggest.

### 9.5 The two on-chip accelerators, and why neither is the answer

The SoC is marketed with a 30 TOPS AI accelerator and an Immortalis-G720
GPU, so "use the NPU" and "use the GPU" are the first two suggestions
anyone makes on seeing the CPU numbers. Both were investigated and both are
closed; this section exists so they are not reopened from the marketing
sheet every six months.

**The NPU (Arm China Zhouyi) does not run autoregressive language models.**
It is a static-graph engine: an ONNX graph with fixed shapes, compiled
ahead of time, in per-tensor or per-channel INT8. An LLM is the opposite
profile — shapes change every token, there is a KV cache that grows, and
llama.cpp's K-quants (per-block scales inside superblocks) are not a format
the NPU speaks. There is no ggml backend for it, and writing one is a
project against ggml internals plus a vendor SDK, with a licensing question
attached that matters under this project's Apache-2.0-only rule. It is a
good accelerator for the workload it was designed for, and there is a
working example of exactly that (object detection under Frigate on this
board). Radxa's own llama.cpp documentation for the O6 lists CPU only.
Independently confirmed by the community running this hardware: *"The
Zhouyi NPU doesn't know how to LLM. It's an agentic memory and embeddings
processor. It straight-up doesn't do autoregressive LLM decode."*

**The integrated GPU shares the CPU's DRAM, so it inherits the same
ceiling.** § 9.2 puts the CPU path at 88-97% of the board's 40.1 GB/s, and
an integrated GPU has no memory of its own to do better with. The one
public Vulkan recipe for this board reports 9.9 tok/s on
`qwen2.5-3b-instruct-q5_k_m` (2.44 GB), against 4.3 tok/s for its own CPU
baseline. We ran the same model and quant on the same board, CPU only,
`--threads 8`:

| | prefill (145 / 580 tok) | decode | effective bandwidth | % of 40.1 GB/s |
|---|---:|---:|---:|---:|
| public result, CPU baseline | not reported | 4.3 tok/s | 10.5 GB/s | 26% |
| public result, Vulkan G720 | 15.0-16.5 tok/s | 9.9 tok/s | 24.1 GB/s | 60% |
| **eullm, CPU, cix-p1 build** | **49.9 / 46.1 tok/s** | **14.5 tok/s** | **35.4 GB/s** | **88%** |

The Vulkan path is 1.46× slower than this CPU build at decode and about 3×
slower at prefill. Its own CPU baseline, at 26% of the board's bandwidth,
is roughly one core's worth of throughput — the likely cause is thread
placement (§ 5): with the BSP firmware exposing all 12 cores, a default
thread count includes the four A520s, and ggml's per-operation barrier lets
the slowest core gate every batch. So the published 2.3× "Vulkan wins" is
measured against a configuration problem, not against this SoC's CPU.

None of this rules out the GPU winning on some workload — prefill is
compute-bound (§ 7.2) and is where an accelerator could in principle
contribute. It rules out the *reason* usually given for reaching for it,
which is decode, and it sets the bar any future attempt has to clear: 35-39
GB/s effective, and 46 tok/s of prefill on a 2.44 GB model.

**What does cross the gap is a discrete card with its own memory** (§ 9.3),
and the board has the slot for one.

## Known open items

- ~~Hybrid/recurrent MoE models (Qwen3.5/3.6) don't get ordinary KV-cache
  prefix reuse~~ — **resolved (v0.6.25 + v0.6.26)**, confirmed on real
  Orion hardware against the real 35B model: 100% exact-match reuse
  across 6+ consecutive turns, multiple topics, both 4096/16384 context
  and F16/Q8_0 KV cache. Root cause was retokenization instability +
  a `/no_think` history-corruption bug, not the rollback window size —
  see § 8.2. `--ctx-checkpoints` remains implemented and safe but turned
  out not to be the fix for this particular scenario.
- ~~`qwen3.6-35b-a3b` running Q4_K_M vs switching to UD-IQ4_NL~~ —
  **tested, resolved**: UD-IQ4_NL is measurably *slower* despite being
  smaller (34% of its variable tensors are ARM-accelerated vs 100% for
  Q4_K_M) — see § 8.3. Q4_K_M is the better choice for this hardware, not
  a fallback.
- eullm's chat template resends historical `<think>` blocks verbatim when
  thinking is *on*, unlike the official Qwen3.6 template's default
  (strip past reasoning unless `preserve_thinking` is set). This is a
  separate, still-open, lower-priority deviation from official behavior —
  distinct from the (now-fixed) `/no_think`-suppression persistence bug,
  and doesn't block the reuse result above (verbatim resend is actually
  what makes exact-text-prefix matching work reliably when thinking is
  on).
- Exact big-core clock ceiling (2.6 vs 2.8 GHz) varies by silicon revision —
  read it on the actual unit, don't assume. The unit measured in § 9 reports
  2500/2500/2400/2400/2300/2300/2200/2200 MHz across its 8 visible A720
  cores, i.e. four clock tiers, not two.
- **The `qwen3.5-9b` vs `qwen3.6-35b-a3b` prefill crossover at ~11,900
  tokens (§ 9.1) is an extrapolation from two points each, not a
  measurement.** Confirming it needs `--ctx-size 16384` and a prompt of
  roughly 8192 words. Worth doing: it decides which model to put behind a
  long-context workload on this board.
- **Whether the onboard Immortalis-G720 can beat the CPU at decode is
  unmeasured, and the bar is ~37 GB/s of effective DRAM bandwidth (§ 9.2),
  not a tok/s figure.** An integrated GPU shares the same LPDDR5 as the CPU
  cluster, so the question is purely how much of the achievable peak each
  path extracts. Measure the board's real memory bandwidth first (a
  STREAM-style test); only if it is well above 37 GB/s is a Vulkan or
  OpenCL build worth building and testing. Prefill, being compute-bound
  (§ 7.2), is the more likely place for a GPU to win.
- A third data point at 4727 tokens for `qwen3.6-35b-a3b` at `--threads 8`
  would replace the one extrapolated figure in § 9.1's table with a
  measurement. Roughly four minutes of runtime.
- Power telemetry is unconfirmed; treat as absent until verified with
  `ls /sys/class/hwmon/*/name` on real hardware.
- Native ggml threadpool cpumask pinning (beyond `taskset`) isn't exposed
  through our Rust bindings; see § 5.
- eullm doesn't print a CPU-feature-flags line of its own (§3) — the repack
  log is the only confirmed-working runtime verification today.
- **Resolved, unrelated bug found along the way**: the first real Orion run
  showed no `tracing::info!` output at all (no "reused N from cache" line,
  no scheduler/context startup info — only `println!`/raw ggml C++ logs).
  Root cause: `Cargo.toml` declares `[[bin]] name = "eullm"` but the default
  `EnvFilter` fallback (used whenever `RUST_LOG` isn't set) was
  `"eullm_engine=info"` — never matching the actual `eullm::...` targets,
  silently disabling all engine-side logging by default, on every platform.
  Fixed by correcting the fallback to `"eullm=info"`; reproduced and
  verified fixed on x86 first, unrelated to ARM/i8mm.
- **Now resolved thanks to the above fix**: with logging actually visible,
  a real multi-turn `qwen3-14b` conversation confirmed KV-cache prefix
  reuse (roadmap 0.7-A) *is* engaging correctly on the `--cli` path —
  turn 2 (`prompt=504, reused=485, decoded 19 fresh`) and turn 3
  (`prompt=1081, reused=1042, decoded 39 fresh`) each reused essentially
  the entire prior turn's resident history, decoding only the new
  suffix. The earlier "stalled at 100% CPU for a long time" observation
  on the very first (pre-fix) run was therefore not a reuse failure —
  most likely plain prefill latency with no progress indicator, or an
  unrelated one-off on that specific run.

**Confirmed on real CIX P1 hardware (Orion, 2026-07-17):** the build
profile itself works — `q4_K_8x8`/`q6_K_8x8` in the repack log on a real
`qwen3-14b` (Q4_K_M) run is direct proof i8mm is active (§4), and the
logging fix above let the same run confirm KV-cache reuse is also working
correctly on this build. Everything else in this document was written
from reading the vendored `ggml`/`llama.cpp` source and a cross-compile
smoke test (no ARM hardware in the environment that wrote it).

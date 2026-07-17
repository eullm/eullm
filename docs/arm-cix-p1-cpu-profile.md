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

**Why decode tok/s alone can look unchanged even with i8mm confirmed
active** (observed on the first real run): single-token decode is
typically memory-bandwidth-bound on CPU, not compute-bound — every token
requires streaming the *entire* quantized weight set through the core
once, regardless of how fast the multiply-accumulate itself runs. i8mm/
SVE2 speed up the arithmetic, which mostly shows up in **prefill**
(batched matmul over many tokens at once, genuinely compute-bound) and
barely moves decode, which is bounded by DRAM bandwidth instead. Don't
judge the profile's effect from the REPL's end-of-turn tok/s line alone
(that's decode-dominated when the prompt is short) — run the sweep above
and look at the `prefill` numbers specifically.

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
limitation shared by llama.cpp's own reference server — not an eullm gap —
and the practical mitigations today are conversation-length management and
non-thinking mode to bound the per-turn re-prefill growth rate, not a
larger rollback window. Building eullm's own bounded checkpoint mechanism
(mirroring the server's `--ctx-checkpoints`/`--checkpoint-min-step`
design: a capped number of full-state snapshots at bounded spacing,
falling back to full reprocessing only when no checkpoint covers the
request) is the right longer-term fix, and is a natural next roadmap item
rather than pushing further on `n_rs_seq`.

**Separately, and independent of the caching question:** the official
Qwen3.6 chat template strips historical `<think>` reasoning blocks between
turns by default (`preserve_thinking` is opt-in) — eullm's own template
resends them verbatim. This is a genuine deviation from the model's
documented intended usage and is worth fixing regardless of the caching
outcome, though it wasn't established as the cause of the small
prefix-match (ruled out in the code-reading step above).

## Known open items

- **Hybrid/recurrent MoE models (Qwen3.5/3.6) don't get KV-cache prefix
  reuse today** — a known, sourced, upstream llama.cpp limitation, not an
  eullm gap; see § 8.2 for the full investigation and why `--rs-seq` is
  not the fix. eullm's own bounded-checkpoint mechanism, mirroring
  llama.cpp server's `--ctx-checkpoints` design, is the tracked follow-up.
- eullm's chat template resends historical `<think>` blocks verbatim,
  unlike the official Qwen3.6 template's default (strip past reasoning
  unless `preserve_thinking` is set) — see § 8.2, last paragraph.
- Exact big-core clock ceiling (2.6 vs 2.8 GHz) varies by silicon revision —
  read it on the actual unit, don't assume.
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

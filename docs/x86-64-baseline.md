# x86_64 CPU-only baseline: why `target-cpu=x86-64-v3`

Applies to the three CPU-only x86_64 release binaries: `eullm-linux-x64`,
`eullm-macos-x64`, `eullm-windows-x64.exe` (not the ARM64 builds, not the
CUDA builds — see § below on scope).

## The problem, verified

Rust's default target features for the x86_64 triples used in
`release-engine.yml` do not include AVX or AVX2:

```
$ rustc --print cfg --target x86_64-apple-darwin        | grep target_feature
target_feature="cmpxchg16b" "fxsr" "sse" "sse2" "sse3" "sse4.1" "ssse3"

$ rustc --print cfg --target x86_64-unknown-linux-gnu    | grep target_feature
target_feature="fxsr" "sse" "sse2"

$ rustc --print cfg --target x86_64-pc-windows-msvc      | grep target_feature
target_feature="cmpxchg16b" "fxsr" "sse" "sse2" "sse3"
```

`engine/vendor/llama-cpp-rs/llama-cpp-sys-2/build.rs` only turns on
`GGML_AVX`/`GGML_AVX2`/etc. in llama.cpp's CMake config when those features
are actually present in `CARGO_CFG_TARGET_FEATURE` — it reads what rustc
enabled, it doesn't probe the build host's real CPU. With no `RUSTFLAGS`/
`target-cpu` set anywhere in `release-engine.yml` (confirmed: none of the
three CPU-only jobs set it), every published binary is compiled without
AVX2, running llama.cpp's slower, far-less-exercised no-AVX2 CPU kernels —
**regardless of the actual CPU of whoever downloads it**, new or old. This
is baked in at compile time on the GitHub Actions runner, not detected at
runtime on the user's machine.

## Real-world evidence (issue #140)

A 12-core Intel Core i9-8950HK (MacBook Pro 2018/2019, Coffee Lake —
supports AVX2 since it's newer than Haswell) running `eullm-macos-x64`
0.6.31 (already CPU-only, Metal removed separately for an unrelated bug on
the same issue) logged:

```
eval_count: 131, eval_duration: 112805000000ns  →  ~1.16 tok/s
```

on a `qwen3-4b` Q4_K_M model — for comparison, a Raspberry Pi 5 (ARM
Cortex-A76, far weaker silicon) got ~3.4-3.7 tok/s CPU-only in the same
issue thread. A 12-core desktop-class CPU running slower than a Pi 5 is the
symptom of a missing SIMD acceleration path, not a hardware limitation.

## The fix

`rustflags: "-C target-cpu=x86-64-v3"` added to the `eullm-linux-x64` and
`eullm-macos-x64` matrix entries (`build` job), and the same as a static
`RUSTFLAGS` env on the `build-windows` job's build step. Verified this
actually flips the relevant `CARGO_CFG_TARGET_FEATURE` flags on:

```
$ rustc --print cfg --target x86_64-unknown-linux-gnu -C target-cpu=x86-64-v3 | grep target_feature
avx, avx2, bmi1, bmi2, cmpxchg16b, f16c, fma, fxsr, lzcnt, movbe, popcnt,
sse, sse2, sse3, sse4.1, sse4.2, ssse3, xsave
```

which is exactly the set `build.rs` maps to `GGML_AVX`/`GGML_AVX2`/etc.

## The trade-off (chosen deliberately, not a side effect)

`x86-64-v3` is the x86-64 psABI microarchitecture level requiring AVX2 +
FMA + BMI1/2 + F16C + LZCNT + MOVBE, supported by:

- **Intel**: Haswell (mid-2013) onward for mainstream Core i3/i5/i7/i9.
  Some budget Atom/Celeron/Pentium SKUs lagged behind this for a few years
  after 2013 — the date is reliable for the Core line, not an absolute
  guarantee across every Intel part number sold since.
- **AMD**: reliably from Zen/Ryzen (2017) onward. Excavator (2015) had
  partial AVX2 in some mobile APUs, not a dependable baseline.

Below this floor, the binary does not run slower — it crashes on the first
unsupported instruction (`SIGILL`), because the object code for that
instruction literally is not there below `x86-64-v2` vs. being present but
unused. This is a real regression for anyone still on that hardware today
(a decade-old office PC, a cheap thin client, pre-2017 AMD) — accepted
deliberately in exchange for fixing the crippling slowdown for everyone
else, rather than shipping N separate ISA-level binaries per platform (the
approach upstream llama.cpp itself uses for its own releases) or adopting
llama.cpp's `GGML_CPU_ALL_VARIANTS` runtime-dispatch mechanism (rejected
here because it requires `GGML_BACKEND_DL` + `BUILD_SHARED_LIBS` — shipping
separate backend `.so`/`.dll` files alongside the executable, which
conflicts with the project's single-binary distribution principle).

A middle-ground `x86-64-v2` (SSE4.2 + POPCNT, no AVX2) was considered and
rejected: it would not actually fix this bug, since `GGML_AVX2` gates
specifically on the `avx2` feature being present, which `v2` doesn't
include. There is no safe partial step here — it's `v3` or nothing.

## Scope: what this does and doesn't touch

- **Not applied to ARM64 builds** (`eullm-linux-arm64`, `eullm-macos-arm64`):
  NEON is part of the ARM64 baseline itself, not an optional extension the
  way AVX is on x86 — this class of bug doesn't apply there. Consistent
  with every ARM/Apple Silicon report in issue #140 showing correct,
  reasonable performance.
- **Not applied to the CUDA x64 build jobs**: left as-is deliberately, to
  avoid introducing the same SIGILL compatibility cliff for GPU users where
  the GPU carries the bulk of inference compute anyway. Whether any
  CPU-side code path in those builds (tokenization, sampling, CPU-offloaded
  MoE experts) also benefits from or needs this is not evaluated here.

## Verification status

**Confirmed the fix reached the actual published binary.** Beyond
`rustc --print cfg` and reading `build.rs`'s logic (which only proves what
*should* happen), the released `eullm-macos-x64` v0.6.32 asset was
downloaded and disassembled directly (`llvm-objdump -d`): it contains
thousands of AVX2-only integer-vector instructions (`vpaddd`, `vpermq`,
`vpbroadcastd`, `vpmulld`, ...) and FMA instructions (`vfmadd*`), and zero
AVX-512 — exactly the `x86-64-v3` profile, in the real shipped machine
code, not just in a CI log or a local reproduction on different hardware.

**Not confirmed to fix the reported symptom.** Real hardware in issue #140
(MacBook Pro 2018, Intel i9-8950HK — fully AVX2-capable) showed no
measurable speedup after this landed: ~1.16 tok/s on v0.6.31 (no AVX2) vs.
~1.33 tok/s on v0.6.32 (AVX2 confirmed present) — within run-to-run noise,
not the 5-10x a real AVX2 vs. scalar-fallback gap should produce. One
`mac_mini_2018_intel` (Intel i7-8700B) shows unchanged `@@@@` garbage
output across three different engine configurations (Metal / CPU no-AVX2 /
CPU+AVX2) — the same deterministic symptom regardless of GPU backend or
SIMD level strongly suggests a numerical correctness bug (e.g. NaN/Inf
propagating through the compute path) independent of both, not something
this fix addresses. **Conclusion: this was a real, worthwhile fix (removes
a genuine, universal performance cliff on all x86_64 CPU-only binaries),
but it was not the root cause of the issue #140 reports that motivated it.**
See `warn_if_logits_corrupt` (`inference/scheduler.rs`) and
`cpu_features_summary` (`inference/mod.rs`), added specifically to get a
real answer next time instead of re-running this same investigation.

**Operational lesson, hit while verifying this:** a local `cargo test`/
`cargo check` run with no explicit `RUSTFLAGS` can still report AVX2 as
present if an earlier build in the same `target/` directory populated the
CMake cache with it — `llama-cpp-sys-2`'s `build.rs` does not declare
`cargo:rerun-if-env-changed=RUSTFLAGS`, so a RUSTFLAGS change alone doesn't
reliably force CMake to reconfigure once `CMakeCache.txt` already exists.
This does not affect CI (every release build starts on a fresh runner with
no prior `target/`), but it means **local before/after RUSTFLAGS
comparisons must use a clean `target/` directory** (or a fresh clone) to
be trustworthy — otherwise a stale cache can silently mask a real change.

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

Confirmed via `rustc --print cfg` (above) and by reading `build.rs`'s
feature-to-CMake-define mapping directly. **Not yet confirmed on real
hardware against Peter's exact failing case** (issue #140, MacBook Pro
2018 / Mac mini 2018) — that requires the next tagged release and a
re-test, since this environment has no x86_64 Haswell/Coffee-Lake-class
hardware to verify locally. Update this section once that comes back.

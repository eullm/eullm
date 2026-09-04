# CINECA Leonardo — build-from-source notes and multi-GPU benchmarks

**Cluster:** CINECA Leonardo (RHEL 8.7, SLURM 25.05). Tested 2026-09-04 on the
Booster module (Atos Bull Sequana X2135): 4× NVIDIA A100-SXM-64GB, 32-core
Ice Lake, 512 GB RAM per node. Account used: `AIFAC_P02_1147`.

This documents why the prebuilt GitHub release binaries don't run on this
cluster, the exact recipe that builds a working one from source, and the
multi-GPU/batching/KV-quantization benchmarks run on it. Every number and
every fix here came from a real, live session — nothing in this file is
theoretical.

## Why the prebuilt release binaries don't work here

Two independent, unrelated reasons — both are hard blockers, not tuning:

1. **glibc.** RHEL 8.7 ships glibc 2.28 and never moves past it for the life
   of the release (RHEL freezes the major glibc version for ABI stability).
   Our GitHub Actions release binaries are built on Ubuntu 22.04 runners
   (glibc ~2.35). Running `eullm-linux-x64` (the CPU-only build — same
   applies to the CUDA one) fails immediately:
   ```
   ./eullm-linux-x64: /lib64/libc.so.6: version `GLIBC_2.29' not found
   ./eullm-linux-x64: /lib64/libc.so.6: version `GLIBC_2.30' not found
   ...
   ```
   This is permanent — no flag or env var works around it. **Any binary run
   on Leonardo has to be compiled on Leonardo.**
2. **CUDA architecture.** `release-engine.yml` compiles the CUDA build for
   `CMAKE_CUDA_ARCHITECTURES: "86;89;120"` (RTX 3000/4000/5000 consumer
   Ampere/Ada/Blackwell) to keep the binary small (see the nvprune work in
   `CHANGELOG.md` 0.7.3). The A100 is `sm_80` — data-center Ampere, a
   *different* compute capability than sm_86, and *older* than the lowest
   architecture in that list, so it doesn't even benefit from PTX
   forward-compatibility. The official CUDA binary would not run on an A100
   even if glibc were not a problem.

Building from source sidesteps both: compiling directly on the target node
picks up its own glibc, and CMake's `native` CUDA-architecture detection (or
an explicit `CMAKE_CUDA_ARCHITECTURES=80`) targets the A100 correctly.

## Build recipe that works

Run entirely on `login05` (no SLURM allocation needed — compiling is a
normal login-node activity, and it doesn't need a GPU present; more on that
below).

```bash
# Rust toolchain (not preinstalled)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

git clone --recursive https://github.com/eullm/eullm.git
cd eullm

module load gcc/12.2.0      # see "why gcc/12.2.0" below
module load cuda/12.2       # only 12.2 / 12.3 / 12.6 available, no CUDA 13
export CC=gcc
export CXX=g++

export LIBCLANG_PATH=/leonardo/prod/spack/06/install/0.22/linux-rhel8-icelake/gcc-8.5.0/llvm-14.0.6-24mj73cub5kwvjmkwmpnolugquneqkyl/lib
export BINDGEN_EXTRA_CLANG_ARGS="-I$(dirname $(dirname $(which gcc)))/lib/gcc/x86_64-pc-linux-gnu/12.2.0/include"

export RUSTFLAGS="-L $CUDA_HOME/targets/x86_64-linux/lib -L $CUDA_HOME/targets/x86_64-linux/lib/stubs"

cargo rustc --release --features cuda -p eullm-engine --bin eullm -- -C link-arg=-no-pie
```

The `llvm/14.0.6...` path above is Spack's hashed install path on this
cluster and **will differ on a re-image or a different Spack stack** — find
it fresh with `module load llvm/14.0.6--gcc--12.2.0-cuda-12.2` (the only
`llvm` module currently on `module av llvm`) and then
`echo $LD_LIBRARY_PATH | tr ':' '\n' | xargs -I{} find {} -maxdepth 1 -iname "libclang*"`.

No `CMAKE_CUDA_ARCHITECTURES` override needed: with nothing set, ggml's
CMakeLists defaults to `native`, which probes the actual GPU present at
build time — since this build runs on Leonardo, it correctly picks up
`sm_80` on its own (verified: `Device 0: NVIDIA A100-SXM-64GB, compute
capability 8.0` in the resulting binary's own startup log).

Building does **not** need a GPU allocated. `nvcc` cross-compiles device
code without a physical device present; a `srun --gres=gpu:N` allocation is
only needed to *run* the result. Compiling on the login node (shared by ~80
users) costs nothing in GPU-hours and doesn't wait in the SLURM queue.

### Every blocker hit, in the order they appeared, and why

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | `error while loading shared libraries: GLIBC_2.29 not found` (prebuilt binary) | RHEL 8.7's frozen glibc 2.28 vs. the CI runner's newer glibc | Build from source (see above) |
| 2 | `error: linking with 'cc' failed`, `unable to find library -lcuda` (well before this, or `-lcudart_static`) | see rows 3-4 | — |
| 3 | `could not find native static library 'cudart_static'` | `libcudart_static.a` lives under `$CUDA_HOME/targets/x86_64-linux/lib/`, not `$CUDA_HOME/lib64` (the path the module puts in `LIBRARY_PATH`) on this Spack-packaged CUDA install — logged as backlog item **H4-H** for a real upstream fix in `find_cuda_helper`/`build.rs` | `RUSTFLAGS="-L $CUDA_HOME/targets/x86_64-linux/lib"` |
| 4 | `unable to find library -lcuda` | `libcuda.so` (the *driver* lib) isn't the toolkit's job to provide; the toolkit ships a link-time-only stub for exactly this situation | add `-L $CUDA_HOME/targets/x86_64-linux/lib/stubs` too |
| 5 | `relocation R_X86_64_32 cannot be used against local symbol; recompile with -fPIC` | `nvcc`-compiled device code isn't PIC by default; Rust binaries link as PIE (ASLR) by default — incompatible | `-C link-arg=-no-pie` on the final binary link (not full CUDA recompile) |
| 6 | Same `-no-pie` flag broke *other*, unrelated crates (`displaydoc`, `yoke-derive`, `zerofrom-derive`, `zerovec-derive` — proc-macros/dylibs) with `undefined symbol: main` | `RUSTFLAGS` applies to *every* compilation unit in the graph, including proc-macros that get built as `.so` and don't want `-no-pie` at all | `cargo rustc -p eullm-engine --bin eullm -- -C link-arg=-no-pie` instead of `cargo build` — `cargo rustc`'s trailing flags apply only to the final target, not its dependency graph |
| 7 | `std::filesystem::__cxx11::path::_List::_List(...)` and friends undefined, `__kmpc_dispatch_init_4` (OpenMP) undefined | System `/usr/bin/gcc` is RHEL's own patched GCC 8.5.0. Pre-GCC-9, `std::filesystem` lived in a separate `libstdc++fs`; combined with the RHEL patching, linking against it produced ABI mismatches on internal (non-exported-inline) classes rather than a clean missing-symbol error | Switch the whole build to `module load gcc/12.2.0` (Spack's GCC, not the system one) via `CC=gcc CXX=g++`, where `std::filesystem` is in the main libstdc++, no separate lib, no ABI drift |
| 8 | `fatal error: 'stdbool.h' file not found` (bindgen/libclang) | Switching to `gcc/12.2.0` meant `libclang` (from the separately-loaded `llvm/14.0.6` module) no longer found the new GCC's own bundled C headers automatically | `BINDGEN_EXTRA_CLANG_ARGS="-I<gcc-12.2.0 prefix>/lib/gcc/x86_64-pc-linux-gnu/12.2.0/include"` (a bare `--gcc-toolchain=` driver flag did **not** work — bindgen drives libclang via its C API, not the `clang` executable, and doesn't honour every driver-level flag the same way) |

Net result: `cargo rustc --release --features cuda -p eullm-engine --bin eullm -- -C link-arg=-no-pie` with the env above builds cleanly, and the
resulting binary correctly detects and uses all 4 A100s.

### SLURM specifics for this account

- Compute nodes have **no outbound internet** (`git clone`/`wget` fail with
  `Network is unreachable`). `$HOME` is shared between login and compute
  nodes, so `git clone`, `rustup`, model downloads, and the build itself all
  happen on `login05`; only *running* the binary needs a compute-node
  allocation.
- `AIFAC_P02_1147` has budget on `boost_usr_prod` (Booster/GPU) but **not**
  on `dcgp_usr_prod` (the CPU-only DataCentric module) — `srun` there fails
  with `invalid account or expired budget`. A CPU-only baseline, if wanted
  later, would need to run on a Booster node without `--gres=gpu`, not on
  DCGP, under this account.
- Interactive test allocation used throughout:
  ```bash
  srun --partition=boost_usr_prod --qos=boost_qos_dbg \
       --account=AIFAC_P02_1147 \
       --gres=gpu:N --ntasks=1 --cpus-per-task=8 --mem=64G \
       --time=00:30:00 --pty bash
  ```
  SLURM only bills actual elapsed time, not the `--time` ceiling — exiting
  early stops the meter (confirmed via `sacct -j <id> --format=Elapsed`).

## Runtime findings

Model: `qwen3.8-27b-ud-q8_k_xl` (Qwen3.8 27B, Q8_K_XL, dense — not MoE,
~29.3 GiB). All numbers are aggregate `output_tokens` summed across 16
concurrent `/api/generate` requests (`Scrivi una breve storia sul mare.`),
divided by wall-clock time from `time (... ; wait)`. Warm-up request sent
first and excluded from the timed window in every case.

| Configuration | GPUs | Batch | Total ctx (per-slot) | KV cache | Aggregate throughput |
|---|---|---|---|---|---|
| Single request, no concurrency | 1 | 1 | 4096 | f16 | ~31-34 tok/s |
| Concurrent, small context | 1 | 16 | 65536 (4096) | f16 | 54.7 tok/s |
| Concurrent, small context | 4 | 16 | 65536 (4096) | f16 | 102.1 tok/s |
| Concurrent, small context, **quantized KV** | 4 | 16 | 65536 (4096) | q8_0 / q8_0 | **128.1 tok/s** |
| Concurrent, huge context | 4 | 16 | 1,048,576 (65536) | q8_0 (K auto-raised) / q4_0 | 15.8 tok/s |

### Conclusions

- **Multi-GPU layer-split scaling is real but sub-linear.** 1→4 GPUs at
  equal batch/context gave ~1.87× (54.7 → 102.1 tok/s), not 4×. Expected:
  eullm/llama-cpp-2 default to `LlamaSplitMode::Layer` (pipeline split), so
  every layer boundary that crosses a GPU costs a PCIe hop. A single
  sequential request sees **no** benefit from extra GPUs at all (decode is
  strictly sequential layer-to-layer) — the benefit only shows up under
  concurrent load, which is what these tests exercise. True tensor-parallel
  (`split-mode row`) would scale differently but isn't exposed as a CLI flag
  today.
- **Batching alone is worth almost as much as the 3 extra GPUs.** Going
  from 1 request to 16 concurrent ones on the *same single GPU* was a 1.6×
  gain; adding 3 more GPUs on top of that batching was another ~1.87×. For
  a "many concurrent users" deployment, tuning `--batch-size`/`--ctx-size`
  is worth doing before reaching for more hardware.
- **KV cache quantization is a net win here, not a cost.** `q8_0`/`q8_0` at
  the *same* context size was 128.1 tok/s vs. 102.1 tok/s for `f16` — 25%
  **faster**, isolated with a controlled single-variable test after an
  earlier confound (see next point). Decode is memory-bandwidth-bound, so
  moving less data per step outweighs the small dequantization compute
  cost.
- **A huge allocated context tanks throughput, independent of
  quantization.** The 15.8 tok/s row was originally (wrongly) attributed to
  KV quantization; re-testing with quantized KV at the *original* small
  context (128.1 tok/s row) proved quantization itself is free or better,
  and isolated the real cause to the 16× larger allocated context
  (1,048,576 vs. 65536 total tokens) — a **6.4× regression** from that one
  change alone, apparently from overhead that scales with allocated
  capacity, not actually-used length. Don't combine "test long context" and
  "test throughput" in one run — they pull in opposite directions and the
  result of combining them isn't informative about either axis alone.
- **`--fit`'s conservative estimate can hold back a layer even when real
  free VRAM is abundant.** With a very large KV reservation, `--fit`
  offloaded only 42/65 (then 64/65) layers to GPU despite ~48-50 GiB free
  per card — `--no-fit` (trusting `--gpu-layers -1` literally) recovered
  full GPU residency once we'd independently confirmed from the printed
  memory breakdown that the headroom was real.
- **64k-per-slot context at 16 concurrent slots is the best point found so
  far**, not a hard ceiling — nobody has yet tried 32-64 slots at smaller
  per-slot context, a genuinely wider `--ctx-size` sweep at the *good*
  (small-context, quantized-KV) configuration, or `split-mode row` if it
  becomes available. Worth revisiting.

## Open items

- Backlog **H4-H** (`docs/backlog-fix-e-hardening.md`) tracks a real fix for
  the `cudart_static` search-path gap so future Spack-packaged CUDA builds
  don't need the manual `RUSTFLAGS` workaround.
- No CPU-only baseline was collected on Leonardo (the login-node attempt was
  killed by the shared node's resource limits, and this account has no
  `dcgp_usr_prod` budget for a proper isolated CPU run).

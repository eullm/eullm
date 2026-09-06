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

> **Status as of 0.7.5-rc3 (6 Sep 2026): reasons 1 and 2 are fixed in CI.**
> `eullm-linux-x64` now builds on a Rocky Linux 8 base and was verified
> running on a Leonardo login node — building from source is no longer
> required for the CPU binary, which is the one that matters here anyway
> (compute nodes have no outbound network, so model downloads run from the
> login node). A data-center CUDA artifact targeting sm_80/sm_90 now exists
> too. Reason 3 below was found *while* validating that work and is the
> one still open at the time of writing.

Three independent, unrelated reasons — all hard blockers, not tuning:

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
   No flag or env var works around it from the running side — but it is
   fixable at build time, and now is: since 0.7.5-rc3 the CPU and both CUDA
   x64 builds compile on a Rocky Linux 8 base (glibc 2.28), and glibc is
   backward-compatible, so those binaries run here unchanged. The Vulkan and
   ARM64 Linux artifacts are still built on Ubuntu and still fail this way.
2. **CUDA architecture.** `release-engine.yml` compiles the CUDA build for
   `CMAKE_CUDA_ARCHITECTURES: "86;89;120"` (RTX 3000/4000/5000 consumer
   Ampere/Ada/Blackwell) to keep the binary small (see the nvprune work in
   `CHANGELOG.md` 0.7.3). The A100 is `sm_80` — data-center Ampere, a
   *different* compute capability than sm_86, and *older* than the lowest
   architecture in that list, so it doesn't even benefit from PTX
   forward-compatibility. The official CUDA binary would not run on an A100
   even if glibc were not a problem. Addressed since 0.7.5-rc1 by a separate
   `-datacenter` artifact built for `80;90`.
3. **CUDA driver version.** Found on 6 Sep 2026, testing the datacenter
   artifact on a Booster node. The binary starts, prints a normal banner
   claiming `GPU backend: CUDA` and `GPU layers: all`, and then runs
   entirely on CPU. The only sign is one line early in the output:
   ```
   ggml_cuda_init: failed to initialize CUDA: CUDA driver version is insufficient for CUDA runtime version
   ```
   with the memory breakdown showing a `Host` row and no device row. A 27B
   Q8 model then decodes at CPU speed, which looks like a hang rather than a
   fallback. The cause: that artifact was built with CUDA 13.1, which needs
   driver r580 (August 2025). Leonardo's newest toolkit module is
   `cuda/12.6`, so its driver predates that. Fixed by rebuilding the
   data-center artifact on CUDA 12.4 (driver floor r550) — see the matrix
   comment in `release-engine.yml` for why 12.4 rather than 12.6.

   Two lessons worth keeping: check `module avail cuda` on any new site
   before assuming a CUDA build will run there — the newest toolkit offered
   bounds the driver — and treat a silent CPU fallback as a reporting bug,
   because a banner that says `CUDA` while running on CPU costs real GPU
   allocation before anyone notices.

Building from source sidesteps all three: compiling directly on the target
node picks up its own glibc, its own driver-compatible CUDA, and CMake's
`native` architecture detection (or an explicit
`CMAKE_CUDA_ARCHITECTURES=80`). It is no longer the *only* way in, but it
remains the fallback when a published artifact does not match the site.

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

## Storage: `$WORK` vs the fast scratch tier (6 Sep 2026)

Model loading felt slow — around a minute for a 29.3 GiB GGUF — so we
measured instead of guessing. Two plausible explanations turned out to be
wrong, and the one that mattered was never about tuning.

Raw sequential read of the same 31.46 GB file, login node, cold:

| Path | Time | Throughput |
|---|---|---|
| `$WORK` (`/leonardo_work/AIFAC_P02_1147`) | 48.99 s | ~640 MB/s |
| `/leonardo_scratch/fast/AIFAC_P02_1147` | **6.54 s** | **~4.8 GB/s** |

**7.5× faster on the fast tier**, and it holds for the real workload: model
load on a compute node is visibly quicker from scratch.

What did *not* help, both tested and both dead ends worth not repeating:

- **Lustre striping.** `$WORK` applies a progressive layout (PFL) by
  default: `stripe_count 1` for the first 10 GiB, 2 up to 100 GiB, 4 beyond
  — so a third of this file came off a single OST. Restriping the whole file
  to `lfs setstripe -c 8` gained **7%** (48.99 s → 46.74 s), not the several
  times expected. Striping helps parallel or large-block I/O; a model load
  is one sequential stream and barely engages the extra targets.
- **Larger read blocks.** `dd bs=16M` was 570 MB/s, marginally *slower* than
  `cat`'s default. Request size is not the limit either.

The page-cache confound was ruled out explicitly rather than assumed:
reading the 31 GB `$WORK` copy evicts the scratch copy from cache, and a
re-read of scratch immediately afterwards still returned 6.5 s.

**Why this is worth the trouble:** the model load happens on the compute
node, so those ~45 saved seconds are billed A100 time on every single
allocation. Ten loads is roughly eight minutes of GPU allocation not spent.

Recommended layout, given scratch is purged of inactive files while `$WORK`
is permanent — keep the canonical copy on `$WORK` and a working copy on
scratch, pointing eullm at the fast one only where speed matters:

```bash
export EULLM_MODELS_DIR=/leonardo_scratch/fast/AIFAC_P02_1147/models
```

Leave `~/.eullm` symlinked to `$WORK/.eullm` as the safe default: if scratch
is cleaned, nothing breaks and re-copying takes about a minute.

Still unmeasured: the same comparison run *from a compute node* rather than
the shared login node. A 7.5× gap is unlikely to invert, but the login
node's storage connectivity is not the one that matters at run time.

## What the published 0.7.5-rc4 binaries actually do here (6 Sep 2026)

| Artifact | Result |
|---|---|
| `eullm-linux-x64` (CPU) | ✅ runs on the login node — the Rocky Linux 8 rebuild fixed the glibc failure. This is the one that matters for downloads, since compute nodes have no outbound network |
| `eullm-linux-x64-cuda-12.4-datacenter` | ✅ initialises CUDA on an A100-SXM-64GB (`compute capability 8.0`), `--fit` offloads all layers automatically |
| same, built on CUDA 13.1 (rc1–rc3) | ❌ silent CPU fallback — see reason 3 above |

Single-stream decode of `qwen3.8-27b-ud-q8_k_xl`, 1 GPU, all layers on GPU:
**32.4 tok/s** — inside the ~31-34 tok/s band measured before the llama.cpp
bump from `b10405` to `b10818`, so that bump carries no performance
regression on an existing architecture. That closes the real-hardware
validation the bump still owed.

Two things worth fixing that this surfaced, neither blocking:

- The banner prints `GPU backend: CUDA` and `GPU layers: all` even when
  `ggml_cuda_init` has failed and everything is running on CPU. The single
  warning line scrolls past and the only reliable tell is a memory breakdown
  with a `Host` row and no device row. A banner that reports the *requested*
  configuration rather than the achieved one costs real GPU allocation
  before anyone notices.
- The model download path never logs its destination directory, and eullm
  defaults to `~/.eullm/models` — which on this system is a 50 GB quota. A
  30 GB model landed there unnoticed. One log line naming the target would
  have made it obvious.

## Open items

- Backlog **H4-H** (`docs/backlog-fix-e-hardening.md`) tracks a real fix for
  the `cudart_static` search-path gap so future Spack-packaged CUDA builds
  don't need the manual `RUSTFLAGS` workaround.
- No CPU-only baseline was collected on Leonardo (the login-node attempt was
  killed by the shared node's resource limits, and this account has no
  `dcgp_usr_prod` budget for a proper isolated CPU run).

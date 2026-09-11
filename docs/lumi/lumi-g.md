# LUMI-G — the allocation, the ROCm build, and what the engine is missing

> **Nothing in this file has been run on LUMI yet.** That is the exact opposite
> of [`../cineca/leonardo.md`](../cineca/leonardo.md), where every number and
> every fix came from a live session. What follows is the plan to execute at
> first login, assembled from the accepted proposal, the source of the vendored
> llama.cpp, and LUMI's own documentation. Correct it in place against what the
> machine actually says, and delete this banner once the recipe below has
> produced a binary that really runs on a compute node.

## The allocation

**EHPC-DEV-2026D09-278** — *Development and Optimisation of EULLM Engine for
Heterogeneous EuroHPC Architectures*. EuroHPC Development Access, six months,
PI Francesco Marchetti (I3K Technologies).

The application requested two partitions: 4,500 node-hours on Leonardo Booster
**and** 4,500 on LUMI-G. **Only LUMI-G was assessed and awarded.** The technical
assessment contains a single `Partition details #1: LUMI-G` block, and the
Puhuri plan shows one compute component — 4,500 node-hours, alongside 90,000
TB-hours of storage. Leonardo is not part of this project: the only Leonardo
access we hold is the separate AI-Factory allocation EHPC-AIF-2026PG01-1147,
which runs to 02/11/2026 and exists for legal-it-4b training (see
[`../leonardo-allocation-plan.md`](../leonardo-allocation-plan.md)).

That asymmetry matters, because the proposal promises a CUDA↔ROCm comparison —
objective (4) of five — and the granted hardware is AMD only. The NVIDIA half
has to come from measurements we already hold or can still take elsewhere.

**The allocation window is not recorded here on purpose.** The application asked
for a 01-10-2026 start, but access was opened earlier; the authoritative dates
come from `lumi-allocations` on a login node. Fill them in when known — the
budget arithmetic below depends entirely on them.

### What the project committed to measuring

Five technical questions, from the accepted proposal:

1. how model size, quantisation level and context length affect GPU memory
   allocation and inference throughput;
2. how continuous batching behaves as concurrency increases;
3. how execution can be extended across multiple GPUs without disproportionate
   communication overhead;
4. how performance and memory-management strategies differ between NVIDIA and
   AMD accelerators;
5. which runtime and model-loading optimisations most help workloads that must
   later run on smaller on-premises systems.

The metrics named in the application, which the final report will be read
against: time-to-first-token, prompt-processing throughput, generation
throughput, memory/HBM utilisation, model-loading time, scaling efficiency, and
stability under sustained load — from single device to full node, on dense and
Mixture-of-Experts models.

## Why not Vulkan

We ship a Vulkan binary and it covers AMD consumer hardware, so the question
"can we skip the AMD build and use Vulkan on LUMI?" is a fair one. The answer is
no, for three independent reasons, in descending order of how decisive they are.

1. **It is not what was accepted.** The technical assessment reads *"A HIP
   implementation is part of the project"*, and the application states that
   "ROCm/HIP will be used for GPU execution on LUMI-G". Benchmarking the Vulkan
   path instead would not be the project that was assessed.
2. **The driver is probably absent.** MI250X modules are compute-only, with no
   display engine. The kernel `amdgpu` driver is present for ROCm, but a Vulkan
   ICD (RADV or AMDVLK) is a separate graphics userspace component that Cray
   compute-node images typically do not carry. Without one, the binary starts,
   reports a GPU backend, and runs entirely on CPU.
3. **Even if it ran, it would measure the wrong thing.** RADV exposes no
   cooperative-matrix path on CDNA 2, so the MFMA units — the whole point of
   this card — would sit idle during prompt processing, and llama.cpp's Vulkan
   multi-GPU support is well behind its ROCm equivalent, on a node that has
   eight devices.

Note that glibc is *not* a reason here, unlike on Leonardo: LUMI's login nodes
run SLES 15 SP6 (glibc 2.38), so the Ubuntu-built release artifacts are not
blocked the way they are by RHEL 8.7's glibc 2.28.

Vulkan on gfx90a is still worth one measurement as a portability data point for
the final report — ROCm vs Vulkan on identical hardware is exactly the kind of
comparison the proposal is about. It is a result, not a route. Settle whether it
is even possible in ten seconds: `ls /usr/share/vulkan/icd.d` and
`vulkaninfo --summary`.

## The machine

- **Node**: 4× MI250X, each a multi-chip module of 2 Graphics Compute Dies, so
  **8 GCDs per node** as far as Slurm and software are concerned. 110 compute
  units and 64 GB of HBM2E per GCD (128 GB per module). One 64-core AMD EPYC
  "Trento", 512 GB in 4 NUMA domains.
- **Architecture**: `gfx90a`, CDNA 2. This is the string the build needs.
- **No local storage on compute nodes.** Everything is on Lustre; model staging
  and cold-start measurements have to account for that.
- **Login nodes have no GPU**, which is precisely why the build needs an
  explicit architecture (see below).
- **ROCm 6.3.4** is the system default since the January 2026 update, installed
  at `/opt/rocm` — which happens to be exactly where our build script looks by
  default. The `amdgpu` driver in use is compatible with ROCm userland 6.1 to
  7.0.
- **Billing**: on `standard-g` a node-hour costs 4 GPU-hours (whole nodes are
  allocated); `small-g` and `dev-g` bill 0.5 per GCD-hour, so single-device
  experiments are cheap. Storage bills the *allocated quota* × time: LUMI-P at
  1×, LUMI-F (flash) at **3×**, LUMI-O (object) at 0.25×.

### The budget, in the only terms that matter

4,500 node-hours across a six-month window (~4,370 hours) is **about one node
occupied continuously for the entire allocation**. The constraint is the
calendar, not the hours — the same arithmetic, and the same trap, as
[`../leonardo-allocation-plan.md`](../leonardo-allocation-plan.md): an idle
queue is budget that evaporates, and an allocation returned 80% unused is
recorded in the final report and remembered at the next call.

The practical consequence is a two-track plan. Single-GCD work on `small-g` and
`dev-g` costs 0.125 node-hours per hour and is where iteration belongs; it will
never consume the allocation. Full-node `standard-g` campaigns are what actually
spends it, and they have to be queued deliberately rather than as an
afterthought.

## Build recipe (unverified)

The engine already has the backend: `--features rocm` → `llama-cpp-2/rocm` →
`GGML_HIP=ON`, linking `amdhip64`, `rocblas` and `hipblas`. What it did not have
until now is a way to say *which* AMD GPU to compile for.

`ggml-hip/CMakeLists.txt` forwards `AMDGPU_TARGETS` → `GPU_TARGETS` →
`CMAKE_HIP_ARCHITECTURES`, and if none of the three is set it leaves the choice
to `enable_language(HIP)`, which resolves the architecture from the GPUs present
on the **build host**. On a LUMI login node there are none. The build script now
accepts `EULLM_AMDGPU_TARGETS` (with `AMDGPU_TARGETS`/`GPU_TARGETS` as aliases)
and passes it through as `GPU_TARGETS` + `CMAKE_HIP_ARCHITECTURES`; without it,
it prints a cargo warning rather than silently producing a binary with no device
code.

There are two routes onto the machine, and the first costs nothing to try.

**The published artifact.** `release-engine.yml` builds
`eullm-linux-x64-rocm-gfx90a` inside a `rocm/dev-ubuntu-22.04:6.3.4` container.
6.3.4 is chosen to match LUMI's own system ROCm, not to be current: ROCm's
libraries are soname-versioned and this binary resolves them from the site, so
building against a newer stack would produce something LUMI cannot load. Same
reasoning that produced the CUDA 12.4 data-centre artifact after 13.1 refused to
start on Leonardo. The job also runs on `workflow_dispatch`, so a fresh binary
can be built from the Actions tab without cutting a release.

**Building on the login node**, which picks up whatever ROCm the machine has:

```bash
bash tools/lumi/build_engine.sh     # EULLM_AMDGPU_TARGETS=gfx90a by default
```

It refuses to start rather than fail late — ROCm, a CMake new enough for
`enable_language(HIP)` (3.21), cargo, and the llama.cpp submodule are all
checked before anything compiles — and then verifies what it produced.
Underneath it is just:

```bash
export ROCM_PATH=/opt/rocm          # build.rs falls back to this anyway
export EULLM_AMDGPU_TARGETS=gfx90a  # MI250X / CDNA 2
cargo build --release --features rocm -p eullm-engine
```

Either way, prove it against a real device before trusting it:

```bash
eullm pull qwen3-8b                   # from a login node; compute nodes have no network
sbatch tools/lumi/sbatch_smoke.slurm  # ~0.03 node-hours on dev-g
```

That job reads `rocm-smi` before and after one request, because the startup
banner reports the *compiled* backend and would print `ROCm` just the same from
a run that fell back to CPU. HBM in use is the measurement that cannot lie.

Things to confirm on the first attempt, each of which has a known failure mode:

- **CMake ≥ 3.21** is required for the HIP language, and `find_package(hip)`
  fails the build below ROCm 6.1. Both should be satisfied by the system stack.
- **A Rust toolchain** is not part of the LUMI software stack; install rustup
  into `$HOME` or `$PROJECT_SCRATCH` from a login node (which has outbound
  network — compute nodes do not).
- **Verify the device code is actually in the binary**, the ROCm analogue of the
  `ggml_vk_` symbol check `release-engine.yml` runs for Vulkan. `roc-obj-ls` or
  `llvm-objdump --offloading` on `target/release/eullm` should list a `gfx90a`
  code object. A binary that builds cleanly and carries no `gfx90a` object is
  the exact Leonardo trap: it runs, it claims a GPU, and it bills node-hours at
  CPU speed.
- **A fully static ROCm binary is not possible** — `ggml-hip` makes `GGML_STATIC`
  a fatal error. The binary will depend on the site's ROCm shared libraries.

HIP build options worth knowing about, all off the same CMake file and all
candidate experiment axes rather than defaults to change blindly:
`GGML_HIP_MMQ_MFMA` (on by default — it is what uses the matrix cores),
`GGML_HIP_RCCL` (off; links RCCL for multi-device collectives),
`GGML_HIP_GRAPHS`, `GGML_HIP_NO_VMM`, `GGML_CUDA_FA_ALL_QUANTS`.

## First-login reconnaissance

Run these before planning anything; each one can invalidate a paragraph above.

```bash
lumi-allocations                       # real window, real remaining budget
sinfo -o "%P %l %D"                    # partitions and walltime limits
ls -d /opt/rocm*; hipconfig --version  # ROCm version actually installed
module avail rocm cmake                # what the module system adds
ls /usr/share/vulkan/icd.d 2>/dev/null; vulkaninfo --summary  # closes the Vulkan question
srun -p dev-g --gpus 1 -t 5 rocm-smi   # what a compute node reports
ldd --version                          # glibc, for the prebuilt artifacts
```

At runtime, `ROCR_VISIBLE_DEVICES` is the ROCm equivalent of
`CUDA_VISIBLE_DEVICES` and is how a single-GCD or 2/4/8-GCD scaling sweep gets
built out of one node.

## What the engine is missing for this project

Four gaps, found by reading the code against the five objectives. None of them
needs allocation time to close.

1. **The banner reports the compiled backend, not the live one.**
   `inference::mod.rs` picks the string from `cfg!(feature = "rocm")`, so a build
   that fails to initialise a device still prints `GPU backend: ROCm` and then
   runs on CPU. This is not hypothetical: it is precisely what cost real A100
   time on Leonardo, documented in [`../cineca/leonardo.md`](../cineca/leonardo.md).
   The information is already available — `ggml_backend_dev_count()` and
   `ggml_backend_dev_memory()` are called a hundred lines further down — so this
   is a reporting fix, not a capability one. On a machine billed by the
   node-hour it is the cheapest possible insurance.
2. **There is no multi-GPU control at all.** `RuntimeOpts` exposes `gpu_layers`
   and nothing else: no `--tensor-split`, no `--main-gpu`, no `--split-mode`.
   llama.cpp will spread layers across every visible device by default, but we
   can neither direct nor vary it — which is objective (3) in its entirety, on a
   node with eight devices. The vendored bindings already expose `main_gpu`,
   `split_mode` and `tensor_split` in `model/params.rs`, so this is plumbing
   into `RuntimeOpts`, not new bindings work.
3. **The collective-communication question is unasked.** The CUDA build sets
   `GGML_CUDA_NCCL=OFF` deliberately, with the reasoning recorded in
   `build.rs`: EuLLM targets single-GPU inference and a hard dependency on
   `libnccl.so.2` breaks consumer machines. The HIP equivalent, `GGML_HIP_RCCL`,
   is likewise off. That default is right for the product and is exactly the
   kind of assumption this project exists to measure.
4. **There is no benchmark harness that produces comparable output across
   sites.** `bench/arm_cpu_bench.py` already measures TTFT and prefill/decode
   throughput against `/api/generate` with streaming, and `bench.sh` drives
   concurrency; what is missing is model-loading time, HBM utilisation, a
   1→2→4→8 device scaling sweep, and a single JSON schema that a LUMI run and a
   CUDA run both emit. Without that, the cross-platform comparison the proposal
   promises has to be assembled by hand at the end.

## Open questions

- **The allocation window.** Everything about pacing depends on it.
- **Where the CUDA half of the comparison comes from**, now that Leonardo was
  not awarded. `../cineca/leonardo.md` already holds real A100 numbers from
  04-09-2026 (1 GPU single request ~31-34 tok/s; 1 GPU at batch 16, 54.7 tok/s;
  4 GPUs at batch 16, 102.1 tok/s; 128.1 tok/s with q8_0 KV; 1→4 GPU scaling of
  1.87×). They were measured by hand rather than by a common harness, and the
  AI-Factory allocation that produced them closes 02/11/2026.
- **Storage tier and quota.** The application asked for 4 TB; the plan grants
  90,000 TB-hours, which at LUMI-P rates is roughly 20 TB held for six months.
  Quota is billed whether or not it is used, and flash costs 3×.
- **The licence stated in the application is Apache-2.0**, in three places. The
  repository has been AGPL-3.0-or-later since August 2026. Neither affects
  eligibility — both are open source — but the final report should not
  contradict the repository.

## Sources

LUMI documentation, current at the time of writing:
[LUMI-G hardware](https://docs.lumi-supercomputer.eu/hardware/lumig/),
[billing policy](https://docs.lumi-supercomputer.eu/runjobs/lumi_env/billing/),
[AI software environment](https://docs.lumi-supercomputer.eu/laif/software/ai-environment/),
[changes after the January 2026 update](https://lumi-supercomputer.github.io/LUMI-training-materials/User-Updates/Update-202601/).
The HIP build behaviour is read from `ggml/src/ggml-hip/CMakeLists.txt` at the
commit this repository's llama.cpp submodule pins.

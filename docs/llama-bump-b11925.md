# llama.cpp b11925: what moved, and how to check it before it lands

Branch `chore/llama-cpp-b11925`, from `main` at be0cf9e (release 0.7.95). The
llama.cpp submodule moves from b86d2f0 (7 October) to b11925 (27 October, 443
commits), the latest tag on the `eullm/llama.cpp` mirror that day. `llama-cpp-rs`
stays at 0.1.159: 0.1.160 only adds WebGPU, Emscripten, PowerPC and Android
build changes, none of which EuLLM builds.

Merged into `main` and released as 0.8.0 on 28 October, after the reference
PC (section 3) and without waiting for the sites, at the maintainer's call.
LUMI-G's round came after (section 4) and found no obstacle; JUPITER and
Leonardo are still to report.

## 1. What changed

### Patches (`engine/vendor/llama-cpp-rs/llama-cpp-sys-2/patches`)

| Patch | At b11925 |
|---|---|
| `0001` pin host memory | applies as it was |
| `0002` pin the host experts, step statistics | rewritten for the cache across several GPUs (#30112): the pages are pinned once, through the backend of the first GPU that caches experts, which registers them as portable for every GPU of that backend |
| `0003` prefetch on a second stream | the scheduler is now made in `llama_context::sched_new`, which sets the prefetch; the scheduler's new realloc callback kept, beside the prefetch's placement check |
| `0004` prefetch reads the cache | the same move; the prefetch still turns itself off with more than one GPU, so the multi-GPU cache never meets it |
| `0005`, `0006`, `0010` | apply as they were |
| `0007` one `ssm_out` product | **dropped**: upstream does the same (#30823) |
| `0008` + `0009` eviction | **one patch**, `0008-moe-cache-evicts-by-pass-and-decaying-counts`, replacing upstream's new 50/50 SLRU (#30637). Replayed through `bench/moe_cache_sim` (the SLRU added to it), the SLRU finds what the LRU found: the same where a step fits, and 0% of the experts in VRAM where it overflows the cache (Qwen3-235B at 32 sequences, Qwen3-Coder-480B at 8 and 32), where this patch finds 73%, 60% and 33% |
| `0011` pool keeps VRAM free | **halved**: upstream now drops a context's CUDA graphs when its pool frees memory (`invalidate_cuda_graphs`), so the patch's own per-device flush counter is gone and its threshold flush calls upstream's function |
| `0012` meta split loads | the loader's buffers are a vector now: one line |
| `0013` top-k MoE group mask on HIP | **new**: upstream's grouped top-k MoE fusion (#29181) uses `__ballot_sync`, which ROCm 6.x declares only with `HIP_ENABLE_WARP_SYNC_BUILTINS`, so ggml-hip did not compile for the MI250X with ROCm 6.3.4 (found by the LUMI build, 28 October; the RDNA release build uses ROCm 7.2.4 and does not see it). On HIP the mask is read through a shuffle; checked with ROCm 6.3.4's clang for gfx90a, gfx1100 and gfx1201 |

### The wrapper and the Rust side

- **The chat output parser** (llama.cpp #30210) is a C++ object with no saved
  form. It used to cross to Rust as a string and be rebuilt on every parse; it
  now crosses as an owned handle, `llama_cpp_2::model::ChatParser`, freed with
  its last clone. Whether reasoning is split out of the content is now decided
  when the prompt is rendered (`reasoning_format = AUTO`, as before), no longer
  at parse time.
- **The rendered prompt** is a list of parts (#30389), template text and user
  input; the wrapper joins them (`str()`), which is what the prompt was before.
  llama.cpp marks the user's parts so that special tokens inside them are not
  parsed; EuLLM does not use that yet (a later change, not this bump).
- `llama_perf_context_data` has two new counters; `engine/src/llama_archs.rs`
  regenerated: eight more architectures (`beam`, `deepseek41`, `flint`,
  `kolibri1`, `maion-coder`, `muse-spark`, `sensenova_u1`, `vllm-decision2`).

### What upstream added that EuLLM can use next (not in this branch)

- Rejection sampling for speculative drafts (#30095, #30213, #30663) in
  `common_speculative`, which `--mtp` goes through.
- n-gram speculation types and a saved n-gram cache (#22055).

## 2. Checked here (cloud container, 4 cores, no GPU)

- CPU build; `cargo test`: 756 unit tests and the 11 `llama_patches` tests pass.
- `cargo clippy --all-targets`: no new warning. `cargo check --features multimodal`: clean.
- The two CUDA files the patches change (`ggml-cuda.cu`, `mmvf.cu`) compile
  with nvcc 13.4 for sm_89, CUDA graphs on.
- The real-model tests (`--ignored real_model`, release build, CPU): 35 of 35.
  Generation on Gemma-3-270M, MTP on Qwen3.5-0.8B-MTP (drafts do not change
  the answer), the hybrid state on Qwen3.5-0.8B, the reasoning block and the
  chat parser on Spark-X2.5-1.7B, decisions on Qwen3-0.6B Q8_0 and
  stories260K F32, embeddings on Qwen3-Embedding-0.6B, finetune on stories260K
  F32. Two decision tests need what their own comments say: an F32 model (on
  Q8_0 the batched and separate results differ by rounding) and a process of
  their own (the backend is initialised once per process).

Not checked here: any GPU run, the full CUDA, ROCm, Vulkan and Metal builds,
Windows.

## 3. On the reference PC (RTX 5070 Ti)

Build the branch next to the installed 0.7.95:

```bash
git fetch origin && git checkout chore/llama-cpp-b11925
git submodule update --init --recursive
cd engine && cargo build --release --features cuda
```

Then the same model and flags on each binary, one server at a time:

1. **Speed and the same answer**: start each binary with the model and run
   `python3 bench/speed_check.py --url http://127.0.0.1:11434/v1` against it.
   Writing and reading speeds, and the answer's hash. Hashes may differ where
   llama.cpp's numerics moved; the text should still read the same.
2. **The expert cache**: `bench/prefetch_check.sh` with each binary on
   Qwen3.8-Flash-Next IQ2_XS (`SETTINGS="0 4"`), and one run with
   `LLAMA_MOE_CACHE_STATS=64`: the share of experts in VRAM should be the
   0.7.95 one or better.
3. **MTP**: Qwen3.5/3.6 with `--mtp 2`, the share of drafts kept in the log.
4. **Chat with reasoning and tools** (the rewritten parser): a reasoning model
   (Qwen3, Spark) and one tool call through Open WebUI or `/v1/chat/completions`
   with `tools`: reasoning in `reasoning_content`, the call in `tool_calls`.

### Measured, 28 October

The 0.7.95 release binary (be0cf9e) against this branch (f49a78e), built
with `--features cuda`, one server at a time, `bench/speed_check.py` at
temperature 0. Tokens per second.

| Model, setting | 0.7.95 writes / reads | b11925 writes / reads |
|---|---:|---:|
| Qwen3.6-35B-A3B Q4 (22 GB, expert cache), 16K prompt | 134.4 / 5,065 | 137.0 / 5,510 |
| Qwen3.8-Flash-Next IQ2_XS (67 GB), 16K prompt | 66.5 / 1,737 | 66.8 / 1,826 |
| Flash-Next, `prefetch_check` 33K, prefetch off | 66.5 / 1,057 | 69.3 / 1,151 |
| Flash-Next, `prefetch_check` 33K, prefetch 4 | 64.8 / 1,634 | 66.4 / 1,766 |
| Qwen3.5-9B MTP Q4_K_M, no MTP, story / code | 123.9 / 124.4 | 125.1 / 125.4 |
| the same, `--mtp 2`, story | 178.2, 55% of drafts kept | 171.9, 50% |
| the same, `--mtp 2`, code | 228.2, 86% | 223.8, 81% |
| the same, `--mtp 2`, 16K prompt read | 4,738-4,758 | 5,513-5,531 |

- **The expert cache**: Qwen3.6-35B-A3B found 97.57% of its experts in VRAM
  at decode on 0.7.95 and 97.72% on b11925, and copied 7,225 and 6,789 MiB;
  the pinning (18.22 GiB, loaded without mmap) and the statistics work on
  the rewritten `0002` and `0008`. The prefetch reads 23% of a micro-batch's
  experts from the cache's slots in VRAM, as before (`0004`).
- **The same answer**: Flash-Next gave e23bd058 on both binaries and both
  prefetch settings. The other models' greedy answers differ between the two
  binaries, as llama.cpp's CUDA kernels moved; Qwen3.5-9B's story is now the
  same with and without `--mtp` (1083b386), where 0.7.95 gave two.
- **MTP**: the only loss. Every other figure is level or better, but with
  `--mtp 2` the model keeps about 5 points fewer of its drafts on both texts
  (three runs each, every run the same figure), and writes 2-3.5% slower.
  llama.cpp's MTP drafting is unchanged between the two pins (only the batch
  call that carries the hidden state was renamed), and so is Qwen3.5's graph
  once EuLLM's old `0007` is counted; the target's numerics moved, and the MTP
  head drafts against them. Not general: LUMI-G's round on the six
  Qwen3.5/3.6 MTP models (section 4) keeps as many drafts on b11925 as on
  0.7.95, so the loss belongs to this card's CUDA kernels, not to the bump.
- **Tools and reasoning**: Qwen3.6-35B-A3B answered a `tools` request with an
  empty `content`, the reasoning in `reasoning_content` and
  `meteo({"citta": "Roma"})` in `tool_calls` on both binaries.

## 4. On the three sites

Each site builds this branch into a build directory of its own, so that no
running job's binary changes, and runs it beside the current engine:

- **LUMI-G**: `bash tools/lumi/build_engine.sh` from a checkout of this branch,
  with `CARGO_TARGET_DIR` set to a new directory; then the smoke job and one
  round of the usual campaign. ROCm is where `0011` came from: a long run should
  show the pool staying under the VRAM (`GGML_CUDA_POOL_KEEP_FREE_MB` lines in
  the log) and no NaN.
- **JUPITER**: `bash tools/jupiter/build_engine.sh llama-b11925` (the label
  names the build directory); `sbatch_compare.slurm` against the current build.
- **Leonardo**: the jobs use the released binary; this branch is built from
  source there (`docs/cineca/leonardo.md`), or tested once it is released.

The branch reaches each site's own branch through `main`, once the PC and the
sites agree.

### LUMI-G, measured 29-31 October

`feat/model-libraries` at 4cf28c5 (0.8.0, b11925) against 0.7.95 (be0cf9e),
MI250X (gfx90a) with ROCm 6.3, one 5.5-hour job per binary. Build and smoke
job pass (with `0013`; without it ggml-hip did not compile here).

- **MTP**: `--mtp 2` on the six Qwen3.5/3.6 MTP models, gsm8k at temperature
  0. The share of drafts kept is the same on both binaries within one point
  (87.4-90.5%), and writing is level to 3% faster on b11925. The 5-point loss
  of the reference PC does not show here.
- **The expert cache** on Qwen3-235B: the same share of experts in VRAM on
  both binaries, 92% at 8 requests and 56% at 32 requests. `0008` holds where
  upstream's SLRU would find none (section 1).
- **Long runs**: four 5-hour soaks at 32 requests. No server died, VRAM stayed
  flat, and the logs hold no HSA error, no out-of-memory and no NaN.
- **Not seen**: `0011`'s pool flush never fired, as the VRAM never came near
  full, so it is still unchecked on b11925. A run that fills the VRAM (a
  larger context or more requests) is what shows it.

No obstacle from LUMI-G to the merge.

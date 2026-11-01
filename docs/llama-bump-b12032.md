# llama.cpp b12032: what moved, and what it is worth to EuLLM

Branch `chore/llama-cpp-b12032`, from `main` after release 0.8.5. The llama.cpp
submodule moves from b11925 (27 October) to b12032 (1 November, 107
commits), upstream `master` and the `eullm/llama.cpp` mirror that day.
Upstream now also tags releases `vX.Y.Z` (v0.8.0 on 28 October, 35 commits
after b11925); the `bNNNNN` tags go on as before, and are what EuLLM pins.
`llama-cpp-rs` stays at 0.1.159: crates.io has nothing after 0.1.160
(15 October), which only adds builds EuLLM does not make.

Of the 107 commits, about half are for backends EuLLM does not build (SYCL,
OpenCL, Hexagon, OpenVINO, WebGPU) or for llama-server's own web UI. What
follows is the rest, by what it changes for EuLLM.

## 1. What EuLLM gains

### Speed

- **MoE prompts read faster on NVIDIA** (#31188): the gate and up expert
  products and the GLU between them run as one quantized kernel (MMQ) for
  `MUL_MAT_ID`, as they already did for dense models. It applies where MMQ
  runs, so to reading prompts and to large batches, not to writing one token
  at a time (MMVQ). Qwen3.6-35B-A3B, Qwen3.8-Flash-Next and every MoE with
  separate gate and up experts are concerned. On the reference PC, with the
  expert cache on, it gained nothing measurable (section 4).
- **The top-k MoE fusion covers 384 experts** (#31288): Kimi-sized routers
  now take the fused path on CUDA.
- **KV rotation constants stay on the GPU** (#31073): no upload per step for
  the models that rotate their KV (DeepSeek 4.1, Mistral Large 4, MiniMax-M3,
  GLM-DSA, dflash).
- **Sparse attention on CDNA** (#31259): the DSA/QSA attention of the
  GLM-5 and DeepSeek 4 families now runs on the MI250X (LUMI-G).
- **Metal**: faster MXFP4 matrix-vector products (#31075).

### Memory

- **Gemma 3n and Gemma 4 no longer hold the token table twice** where input
  and output share a buffer, as on CPU or `-ngl 0`: gemma-4-E2B Q4_K_M takes
  216 MiB less (#31394).
- **Model loading reuses one staging buffer** instead of allocating one per
  weight, and reads direct-IO files with no bounce buffer (#31321). This is
  the path of every buffer that has no async upload; EuLLM's `0012` keeps its
  own overlapped read for meta buffers (`--split-mode tensor`) and now uses
  this one for everything else.

### Models and formats

- Three more architectures: `lfm3` (LiquidAI LFM3, its VL, Audio and Omni
  versions), `limite` (Limite 1B) and `mistral-large4` (Mistral's "le
  chonk"); `engine/src/llama_archs.rs` regenerated.
- **FP8 E4M3 weights** (OCP FP8, #27512): a new GGUF type,
  `LLAMA_FTYPE_MOSTLY_F8_E4M3`, so FP8 checkpoints convert without a
  detour through BF16.
- Gemma 4 images are resized without letterbox padding (#30922), InternVL's
  CLS token is in the right place (#30886), and the rotation of layers that
  reuse another layer's KV is fixed for Gemma 3n and 4 (#31406).

### Correctness to watch

- **Recurrent memory refuses rollbacks its last batch cannot honour**
  (#31368): a hybrid model (Qwen3.5/3.6) now rejects a `seq_rm` deeper than
  the snapshots the last micro-batch wrote. EuLLM's `--mtp` drafts 2 tokens
  inside `n_rs_seq`, so it should not meet it; the MTP real-model test and a
  Qwen3.5/3.6 `--mtp 2` run say whether it does.

## 2. Patches

| Patch | At b12032 |
|---|---|
| `0001`-`0006`, `0008`, `0010`, `0011` | apply as they were |
| `0012` meta split loads | **regenerated**: upstream's #31321 rewrote the loader's staging read the patch replaced. The overlapped read for meta buffers is unchanged; every other buffer without async uploads now takes upstream's read (one reused buffer, direct IO with no bounce), timed for the patch's log line |
| `0013` top-k MoE group mask on HIP | **kept**: upstream now maps `__ballot_sync` to `__ballot` below ROCm 7 (#31259), which compiles on ROCm 6.3, but on a 64-wide wavefront that ballot spans two of the fusion's 32-lane warps; the patch's shuffle gives each warp its own mask on every HIP version |

`cargo test --test llama_patches`: 11 of 11.

## 3. Checked here (cloud container, 4 cores, no GPU)

- CPU build with no warning; `cargo test`: 764 unit tests and the 11
  `llama_patches` tests pass; `cargo clippy --all-targets`: nothing new.
- The real-model tests (`--ignored real_model`, release build, CPU): 35 of
  35, with the models of b11925's round. The MTP test (Qwen3.5-0.8B-MTP,
  drafts do not change the answer) and the hybrid-state test (Qwen3.5-0.8B)
  pass, so #31368's stricter rollbacks do not reach `--mtp 2`. The two
  decision tests ran as their comments ask, on stories260K F32 and in a
  process of their own; three API tests failed first on a full disk and
  passed once it had room.

Not checked here: any GPU run, the CUDA, ROCm, Vulkan and Metal builds.

## 4. On the reference PC (RTX 5070 Ti), 1 November

The 0.8.5 release binary (a3bbe56, b11925) against this branch (09c7975,
b12032) built with `--features cuda`, one server at a time,
`bench/speed_check.py` at temperature 0, 16K-token prompt. Tokens per second.

| Model, setting | 0.8.5 writes / reads | b12032 writes / reads | Answer |
|---|---:|---:|---|
| Qwen3.6-35B-A3B Q4 (22 GB, expert cache 9 GiB, 39% of experts in RAM) | 143.7 / 5,624 | 145.4 / 5,634 | same (831d8d6c) |
| Qwen3-Next-80B-A3B Q4_K_M (48.5 GB), `prefetch_check` 33K, prefetch off | 107.7 / 1,704 | 106.8 / 1,692 | same (9e584b3b) |
| the same, prefetch 4 | 99.7 / 1,470 | 99.5 / 1,465 | same |
| Qwen3.5-9B MTP Q4_K_M, `--mtp 2` | 174.2 / 5,477, 50% of drafts kept | 173.9 / 5,464, 50% | same (1083b386) |

- **Level everywhere**, within 1.2%, and every answer the same bit for bit
  on both binaries. #31188's fused MoE kernel gains nothing measurable on
  these two MoE models: with the expert cache on, the experts' products do
  not take the path it fuses, which the identical answers also suggest. A
  MoE held whole in VRAM is where it would show; none was at hand.
- **MTP**: as many drafts kept as on 0.8.5, the same answer, and no rollback
  error in the server's log: #31368 does not reach `--mtp 2`.
- **The prefetch costs on Qwen3-Next-80B**, on both binaries alike: its four
  420 MiB slots take 1.75 GiB from the expert cache (10.25 GiB to 8.50), and
  reading falls 14%, writing 7%. Not the bump's; on this model
  `--moe-prefetch 0` is the faster setting.

Nothing here holds the merge back. Still to run: LUMI-G's smoke job and an
MTP round, for `0013` beside upstream's ballot and the sparse attention now
enabled on CDNA.

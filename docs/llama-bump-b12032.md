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
  separate gate and up experts are concerned. Not measured yet: section 3.
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

## 3. To check before it lands

On the reference PC (RTX 5070 Ti), the 0.8.5 release binary against this
branch built with `--features cuda`, one server at a time, as in
`docs/llama-bump-b11925.md` section 3:

1. `python3 bench/speed_check.py` with Qwen3.6-35B-A3B Q4 and a 16K prompt:
   the reading speed is where #31188 shows, with and without `--moe-cache`.
2. `bench/prefetch_check.sh` on Flash-Next (`SETTINGS="0 4"`): the expert
   cache with the fused kernel, the same answer hash on both binaries.
3. Qwen3.5-9B MTP with `--mtp 2`: drafts kept and speed, for #31368.
4. A Gemma 4 model with an image, and one reasoning model with a tool call.

LUMI-G: the smoke job and one MTP round, for `0013` beside upstream's
ballot and for the sparse attention now enabled on CDNA.

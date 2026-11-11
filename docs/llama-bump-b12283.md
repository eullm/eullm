# llama.cpp b12283: what moved, and what it is worth to EuLLM

Branch `chore/llama-cpp-b12283`, from `main` after release 0.8.8. The llama.cpp
submodule moves from b12032 (1 November) to b12283 (10 November, 251
commits). `llama-cpp-rs` moves from 0.1.159 to 0.1.161, by hand: its one change
EuLLM builds is `RopeType::IMRope`; its `with_tensor_split` and the MTP
wrapper's own `common_batch` were in EuLLM's vendored copy already, and its
build script's WebAssembly/WebGPU, Android and PowerPC work is left out.

**Before CI can check this branch out**, the `eullm/llama.cpp` mirror needs
b12283: on 11 November its `master` was at #32004 (10 November) and its last
tag b12246. It syncs every night at 03:17 UTC (`mirror-sync.yml`), or at
once from the Actions tab (Run workflow).

About half of the 251 commits are for backends EuLLM does not build (SYCL,
OpenCL, Hexagon, WebGPU, CANN) or for llama-server's web UI. What follows is
the rest, by what it changes for EuLLM.

## 1. What EuLLM gains

### The expert cache, rewritten upstream

- **ClassLRU eviction** (#31233): each expert's uses, decayed with a
  half-life of 128 tokens, rank it in one of eight classes; a miss takes,
  among the oldest slot of each class, the one with the largest
  age/2^class. It replaces the 50/50 SLRU, which found no expert in VRAM
  once a step wanted more experts than the slots. EuLLM's `0008` now builds
  on it instead of replacing it (section 2).
- **Misses computed on the CPU** (#30966): a share of the misses can be
  computed from the host experts instead of uploaded
  (`moe_cache_cpu_share`, llama-server's `--moe-cache-cpu-share`), and
  `scripts/moe-cache-tune.py` fits the upload and CPU speeds of a machine to
  choose the cache size and the share. EuLLM does not expose the share yet:
  it is 0, every miss uploaded, as before. It is worth measuring on the
  reference PC, whose 16-core CPU sits idle while the bus copies.
- Its CPU part no longer waits for the cached experts (#31888), and a draft
  model gets no cache (#31497).
- Upstream writes `LLAMA_MOE_CACHE_TRACE` itself now ("MCT1", every group of
  layers); `bench/moe_cache_sim` reads it, and `0008`'s own trace is gone.

### Speed

- **MoE data laid out linearly for CUDA's MMQ/MMF** (#31047), and NVFP4
  gate/up and GLU fused in MMQ (#31296).
- **Gemma 4 E2B/E4B read prompts much faster** (#31356): the graph prunes
  what the KV-shared layers do not need.
- **Unified KV cache with several sequences** (#29510): attention reads only
  the cells of each sequence (`flash_attn_ext_rows`), on by default where the
  cache is unified. EuLLM's `--kv-unified` measured 2-7% slower with sixteen
  requests and 38-45% slower with 32k-token contexts on an MI250X; worth
  measuring again.
- The scheduler no longer uploads inputs nothing reads (#31419); batched
  tensor reads are gathered into mapped host memory on CUDA (#31391).
- MTP reads the target's hidden states from a device buffer (#31927) and
  saves and restores them with a checkpoint (#30141).

### Memory and stability

- On-device state copies of a quantized KV cache are fixed (#30541): with
  `q8_0` or `q4_0` they copied part of each range. EuLLM's checkpoints copy
  a sequence's state through host memory (`PARTIAL_ONLY` or no flag, never
  `ON_DEVICE`), which this did not reach.
- The CUDA VMM pool halves its reservation and retries when the address
  space cannot be reserved (#30542); CUDA graphs about to be used are not
  evicted (#30662). Neither is what `0011` does (the legacy pool, the only
  one on HIP, keeping VRAM free), which stays.
- KV shifting fixed and tested per model (#31683).

### Models and formats

- **FP8 KV cache** (#28392). Not offered by `--cache-type-k/-v` yet.
- Five more architectures: `kimi-k31` (Kimi-K3.1-Pro), `metis16`,
  `clover_moe`, `ricursive` (Ripple) and `xing4_0`;
  `engine/src/llama_archs.rs` regenerated (172).
- Granite and Qwen2 NVFP4 scales, LFM3 in NVFP4, Chatterbox and Drama-3 TTS
  (mtmd), external per-layer embedding files (#31185).

## 2. Patches

| Patch | At b12283 |
|---|---|
| `0001`, `0005`, `0006`, `0010`-`0014` | apply as they were |
| `0002` pinning and statistics | **regenerated** over the new cache: the destructor closes upstream's trace as well, and the totals at exit keep upstream's count of misses computed on the CPU |
| `0003` prefetch on a second stream | **regenerated**: upstream copies a split's host weights in a second pass over its inputs, after its own prefetch of the next split's small inputs (#30966); the staged lookup moved into that pass, a weight the split does not read is not copied (#31419), and `moe_prefetch_slots` follows upstream's new `moe_cache_cpu_share` |
| `0004` prefetch from the cache | the member the cache's lookup reads is `lru`, no longer `slru`; an expert the cache does not hold reads slot `n_slots` or later in upstream's map now (#30966), no longer -1, and the lookup turns that into -1 (section 4) |
| `0008` eviction | **rewritten** as `0008-moe-cache-classlru-among-stale-slots`: ClassLRU's victim among the slots unused for a whole pass, and the former rule (lowest decayed count among the slots used earlier in the pass) when every slot was used within it. `LLAMA_MOE_CACHE_EVICT=classlru` for ClassLRU alone; `=lru` and `=pass` are gone |

`0008`, replayed through `bench/moe_cache_sim` on the routing calibrated to
LUMI-G (its README has the commands and ten more settings), experts in VRAM:

| model, sequences | ClassLRU | former `0008` | new `0008` |
|---|---:|---:|---:|
| Qwen3-235B, 8 | 91.7% | 91.1% | 91.7% |
| Qwen3-235B, 32 | 65.9% | 73.1% | 73.2% |
| Qwen3-Coder-480B, 1 | 81.4% | 80.8% | 81.4% |
| Qwen3-Coder-480B, 8 | 28.9% | 46.5% | 46.8% |
| Qwen3-Coder-480B, 32 | 0% | 21.1% | 21.8% |

One C API change reached EuLLM's own wrapper: `common_fit_params` takes a
list of model files (#31185), and `llama_rs_fit_params` passes the one GGUF.

`cargo test --test llama_patches`: 11 of 11.

## 3. Checked here (cloud container, 4 cores, no GPU)

- CPU build with no warning, and `--features multimodal`; `cargo test`: 772
  unit tests and the 11 `llama_patches` tests pass; `cargo clippy --no-deps
  -- -D warnings`: nothing.
- The CUDA files the patches touch, compiled alone with nvcc 13.5 for sm_89:
  `ggml-cuda.cu` (`0001`, `0011`, CUDA graphs on), `mmvf.cu` (`0006`),
  `topk-moe.cu` (`0013`'s CUDA side). Not the HIP side of `0011` and `0013`:
  no ROCm here.
- The real-model tests (`--ignored real_model`, release build, CPU): 36 of
  36, with the models of b12032's round (stories260K F32, Qwen3-0.6B Q8_0 and
  its embedding model, Qwen3.5-0.8B with and without MTP, Spark-X2.5-1.7B).
  The MTP test (drafts do not change the answer) and the hybrid-state test
  pass. The two decision tests that load a backend of their own ran in a
  process of their own each, as their comments ask; in the shared run the
  temperature test fails with `BackendAlreadyInitialized`, as before.

Not checked here: any GPU run; the CUDA, ROCm, Vulkan and Metal builds as a
whole.

## 4. To measure before release

On the reference PC (RTX 5070 Ti), the 0.8.8 release against this branch,
as for b12032:

- Qwen3.6-35B-A3B with its expert cache (16K prompt): writing and reading,
  and the cache's hits with `LLAMA_MOE_CACHE_STATS=64`, with the new `0008`
  and with `LLAMA_MOE_CACHE_EVICT=classlru`.
- Qwen3-Next-80B or Qwen3.8-Flash-Next (most experts in RAM): the same, and
  `prefetch_check` with `--moe-prefetch 4` (`0003`, `0004` regenerated).
- DeepSeek-Coder-V2-Lite held whole in VRAM (#31047's MoE layout).
- Qwen3.5-9B MTP with `--mtp 2`: drafts kept and the same answer (#31927,
  #30141).

On LUMI-G (MI250X, ROCm 6.3): the build (`0011` and `0013` on HIP), the
smoke job, Qwen3-235B's cache at 8 and 32 requests with both evictions, and
a soak with the pool's flush (`0011`). The traces of 16-10-2026, replayed
through the simulator with the new policy, would confirm section 2 on real
routing.

## 5. Measured on the reference PC (11 November)

RTX 5070 Ti 16 GB, the 0.8.8 release binary (`eullm-linux-x64-cuda-13.1`)
against this branch built there.

Qwen3.6-35B-A3B with its expert cache, a 16K-token prompt,
`LLAMA_MOE_CACHE_STATS=64`, two runs each:

| binary | writing, tokens/s | reading, tokens/s |
|---|---:|---:|
| 0.8.8 | 143.9, 144.0 | 5,670, 5,630 |
| b12283 | 146.8, 147.6 | 6,846, 6,734 |

The cache's hits are the same in both (92.3% of the experts in VRAM, then
99.1%). The answers part mid-sentence ("shockwave up the" against "shockwave
up through the foundation"), both coherent: upstream's kernels moved the
numbers.

`bench/prefetch_check.sh` (33,200-token prompt, `--moe-prefetch` 0 and 4,
each setting asked twice, greedy), with `0004` fixed (below):

| model | binary | prefetch | reading | writing | from the cache's slots | answer |
|---|---|---:|---:|---:|---:|---|
| Qwen3-Next-80B-A3B Q4_K_M | 0.8.8 | 0 | 1,696.9 | 107.4 | | `9e584b3b` |
| | 0.8.8 | 4 | 1,460.3 | 99.0 | 20% | `9e584b3b` |
| | b12283 | 0 | 1,788.7 | 112.8 | | `98260681` |
| | b12283 | 4 | 1,449.2 | 102.3 | 20% | `98260681` |
| Qwen3.8-Flash-Next IQ2_XS | 0.8.8 | 0 | 1,160.7 | 70.1 | | `e23bd058` |
| | 0.8.8 | 4 | 1,773.9 | 67.9 | 23% | `b2941bfb` |
| | b12283 | 0 | 1,241.2 | 75.9 | | `b2941bfb` |
| | b12283 | 4 | 1,794.2 | 73.6 | 23% | `b2941bfb` |

b12283 writes 3-8% faster than 0.8.8, and reads 5-7% faster without the
prefetch and as fast with it. Every b12283 setting answers the same, and on
Flash-Next with the answer of October's rounds (`b2941bfb`), which 0.8.8 gave
only with the prefetch.

On Qwen3-Next the prefetch reads 19% slower than no prefetch (14% on 0.8.8):
its four slots take 1.75 GB from the expert cache (8.50 GB against 10.25 GB),
and at Q4_K_M a micro-batch's copy does not cover the GPU's computing as it
does at IQ2_XS. Not this bump's doing; a reason to look at when `--moe-prefetch`
should switch itself off.

The first round, before the fix, read 3,764.8 (Qwen3-Next) and 2,239.0
(Flash-Next) tokens/s with the prefetch and said 100% from the cache's slots:
`0004` took every expert of upstream's map for a cached one (an uncached
expert reads slot `n_slots` or later since #30966, no longer -1), copied the
uncached ones from past the end of the cache's banks instead of over the bus,
and answered wrong (on Qwen3-Next, two answers to the same request). The
lookup now turns those slots into -1.

DeepSeek-Coder-V2-Lite Q4_K_M held whole in VRAM (`CTX=10240
PROMPT_TOKENS=8000 MOE_CACHE=off SETTINGS=0 bench/prefetch_check.sh`; at the
script's 40,960 tokens its KV cache, 16 heads of 192+128 values over 27
layers, does not fit beside the model on either binary):

| binary | reading | writing | answer, two runs |
|---|---:|---:|---|
| 0.8.8 | 5,733.1 | 265.7 | `ea6c5d76` both |
| b12283 | 6,339.0 | 268.9 | `592ef83a` both |

Reading +10.6%, against b12032's +3.1% over 0.8.5: #31047's MoE layout for
MMQ, on a model whose experts all sit in VRAM. Writing level.

Qwen3.5-9B MTP Q4_K_M (`MTP_SETTINGS="0 2" bench/mtp_sweep.sh`):

| binary | `--mtp` | story | code | drafts kept | answers, story/code |
|---|---:|---:|---:|---:|---|
| 0.8.8 | 0 | 127.9 | 127.4 | | `1083b386`/`7210b8f2` |
| 0.8.8 | 2 | 172.9 | 215.0 | 61% | `1083b386`/`7210b8f2` |
| b12283 | 0 | 128.4 | 128.3 | | `e58b246d`/`bbc626da` |
| b12283 | 2 | 183.5 | 228.1 | 68% | `e58b246d`/`d24be5ff` |

With drafts b12283 keeps 68% of them against 61% and writes 6% faster
(#31927, #30141). Its code answer with `--mtp 2` parts from the one without
at byte 595 of 1,041 (line 22), on the wording of a comment ("Try parsing
with time component first" against "Try parsing with time first"); both go
on as valid Python (one falls back to `strptime`, the other to
`fromisoformat`), and the server's log has no error or rollback. A near tie
that the three-token check's kernel and the one-token kernel round
differently, not a state restored wrong: that would part early, or go on
wrong.

Nothing measured on the reference PC holds the merge back. The HIP sides of
`0011` and `0013` are compiled by the release workflow's ROCm build, which
the Actions tab runs alone on this branch (Run workflow, `job: rocm`)
without making a release.

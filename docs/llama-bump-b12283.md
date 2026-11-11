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
| `0004` prefetch from the cache | the member the cache's lookup reads is `lru`, no longer `slru` |
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

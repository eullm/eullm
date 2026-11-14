# EuLLM's patches to llama.cpp

Changes EuLLM needs in llama.cpp before upstream has them. The build script
applies every `*.patch` here, in name order, to a copy of the `llama.cpp`
submodule under `OUT_DIR` and builds from that copy (`../llama_patches.rs`);
the submodule itself is never modified.

Each file is a `git diff` against the commit the submodule is pinned to, with a
description of the change above the first `diff --git` line. To make one:

```bash
cd engine/vendor/llama-cpp-rs/llama-cpp-sys-2/llama.cpp
# edit, build and test the change in place, then:
{ printf 'what the change does, and why\n\n'; git diff -- <files>; } > ../patches/NNNN-short-name.patch
git checkout -- <files>        # the submodule goes back to the pinned commit
```

A later patch may change a file an earlier one changed; it is then a diff
against the tree with the earlier patches applied.

Keep new declarations out of the headers every backend's sources include
(`ggml.h`, `ggml-backend.h`, `ggml-impl.h`): a patch that changes one rebuilds
all of them, the CUDA kernels among them, on every machine that builds it and
in every release build, where sccache finds none of them cached. `0003`
declares its one function in `ggml-backend-prefetch.h`, a header of its own,
for that reason.

When the submodule moves, `cargo test --test llama_patches` in `engine/` says
which patches still apply. Regenerate the ones that do not against the new
commit, and delete the ones upstream has taken.

| Patch | What it does |
|---|---|
| `0001-cuda-pin-host-memory-on-request.patch` | CUDA procs to pin and unpin host memory on request (read-only registration), reporting why pinning failed |
| `0002-moe-cache-pin-host-experts-and-step-statistics.patch` | The MoE expert cache pins the experts it uploads from (`LLAMA_MOE_CACHE_PIN=0` to compare), and `LLAMA_MOE_CACHE_STATS=N` reports where a decode step goes; regenerated on b12283 over upstream's ClassLRU and CPU share (#31233, #30966) |
| `0003-moe-prefetch-experts-on-a-second-stream.patch` | With `llama_context_params.moe_prefetch_slots` set (EuLLM's `--moe-prefetch`), the experts in host memory of a MUL_MAT_ID of 512 tokens or more are copied into that many VRAM slots, 2 to 8, on a second stream of the GPU while the splits before it compute (phase 6 of `docs/moe-offload-plan.md`); one GPU of the CUDA backend (CUDA, or ROCm since 14-10-2026), experts in pinned host memory (a model loaded without mmap), slots only where they leave VRAM free for the compute; off unless set. A staged copy skips the copy callback of `llama-context`; since b12283 the staged copies are looked up in upstream's second pass over a split's inputs, the host weights (#30966) |
| `0004-moe-prefetch-reads-cached-experts-from-vram.patch` | The prefetch copies the experts the MoE cache holds from its banks in VRAM, on the GPU's own stream, and only the others over the bus (`llama_moe_cache::sched_lookup`, which since b12283 turns upstream's "slot `n_slots` or later" for an uncached expert into -1; `LLAMA_MOE_PREFETCH_FROM_CACHE=0` to compare); a long prompt reads about 9% faster (phase 6b) |
| `0005-penalties-by-index-on-the-whole-vocabulary.patch` | The repeat, frequency and presence penalties change the logits of the recent tokens at their own index when the candidates are the whole vocabulary in order, as `llama_sampler_sample` builds them: `penalty_last_n` lookups instead of one per token of the vocabulary (1,196 to 2 µs per token on 248,320 tokens), the same logits; any other array is scanned whole, as before |
| `0006-cuda-one-warp-per-row-for-short-bf16-matrix-vector-products.patch` | One-token bf16 matrix-vector products with at most 512 columns and at least 4,096 rows (the low-rank "up" projection of the hyper-connections, 320 to 10,240) give each row to a warp instead of a block of 160 threads with a shared-memory reduction: 17 to 9 µs, twice a layer. Same fp32 arithmetic, the sum in another order; `GGML_CUDA_MMVF_SHORT_ROWS=0` restores the generic kernel |
| `0008-moe-cache-classlru-among-stale-slots.patch` | Keeps upstream's ClassLRU eviction (#31233, b12283) and adds the pass to it: the victim is ClassLRU's among the slots whose expert went unused for a whole pass over the layers; when every slot was used within the pass (a step wants more experts than the slots), the lowest decayed count among the slots used earlier in this pass. ClassLRU alone evicts slots used within the pass there, the experts the next layers want. Replayed through `bench/moe_cache_sim`: ClassLRU's hits where a step fits; 73.2% of the experts in VRAM instead of 65.9% for Qwen3-235B at 32 sequences, 46.8% instead of 28.9% for Qwen3-Coder-480B at 8, 21.8% instead of 0% at 32. `LLAMA_MOE_CACHE_EVICT=classlru` for ClassLRU alone; the trace is upstream's (`MCT1`) since b12283. Was the pass and decaying counts on the SLRU until b12032 (`0008`, before that `0008` and `0009`) |
| `0010-meta-split-weights-copied-as-gathered-rows.patch` | With `--split-mode tensor`, a weight split along axis 0 (`attn_output`, `ffn_down`, `ffn_down_exps`: part of every row on each device) is gathered row by row into a host buffer and copied to each device in pieces of 64 MiB, instead of one 2D copy from pageable memory; on LUMI-G's MI250X tensor loads ran at 100-230 MB/s, 10-50 times slower than layer (`GGML_META_GATHER_ROWS=0` restores the 2D copy) |
| `0011-cuda-pool-keeps-vram-free-and-graphs-follow-its-flushes.patch` | The legacy CUDA/HIP pool (the only one on HIP) frees its cached buffers before a new allocation would leave less than `GGML_CUDA_POOL_KEEP_FREE_MB` of VRAM free (512 by default, 0 for the old behaviour), unless a graph is being captured; that flush drops the context's CUDA graphs, which would otherwise replay into freed memory (upstream does the same since b11925 for the flush after a failed allocation). On LUMI-G's MI250X the pool grew 1.3-2.5 GiB an hour under continuous batching until the GCD was full, then NaN logits, `hipGraphInstantiate` out of memory or memory faults |
| `0012-meta-split-loads-copy-to-every-device-at-once-while-the-next-weight-is-read.patch` | With `--split-mode tensor`, a model read into memory (the meta buffer has no async uploads) has each weight read by 16 threads at once in positioned reads of 16 MiB, into the same two buffers reused from weight to weight, the next one read while the last is copied; the meta buffer copies a weight of 16 MiB or more to each device on a thread of its own, gathering rows into buffers kept from call to call; `load_all_data` logs how long reading, copying and waiting took. On LUMI-G the reading was most of a tensor load (Qwen3-235B: 209-272 s, copying 13-53 s, against 130 s for a whole layer-split load); with 16 threads Qwen3-235B, Qwen3-Coder-480B and gpt-oss-120b load in 140, 121 and 59 s (124, 162 and 69 by layer). `LLAMA_LOAD_READ_THREADS=1`, `LLAMA_LOAD_OVERLAP=0` and `GGML_META_SET_THREADS=0` turn each part off |
| `0013-cuda-topk-moe-group-mask-without-ballot-on-hip.patch` | The grouped top-k MoE fusion builds its group mask with `__ballot_sync`, which ROCm 6.x declares only with `HIP_ENABLE_WARP_SYNC_BUILTINS`: ggml-hip did not compile for the MI250X (gfx90a) with ROCm 6.3.4, the version of LUMI and of EuLLM's gfx90a release build. On HIP each lane reads whether its group is kept from that group's lane by a shuffle instead; the same mask. CUDA keeps the ballot |
| `0014-model-loader-reads-tensors-in-file-order-without-mmap.patch` | Without mmap the tensors are read sorted by file and offset, one pass through the file, instead of in the order llama.cpp creates them (before b12032) or biggest-first (#31321, b12032), and the reused staging buffer only grows, to the largest tensor. On LUMI-G's Lustre, cold (`c16-load-order`): Qwen3.8-27B Q8 on one GCD 44 s instead of 95-130, Qwen3-32B split by tensor 31 instead of 101-107, Qwen3-235B Q8_0 261 instead of 650-690, Qwen3-Coder-480B 149 instead of 183-185; Qwen3-235B Q4_K_M level |
| `0015-cuda-few-row-mmq-can-be-turned-off.patch` | `GGML_CUDA_MMQ_FEW=0` sends quantized products of 2 to 24 tokens back to MMVQ and MMQ instead of llama.cpp 1.0's few-row kernel (#31533). On the reference PC v1.0.0 kept 61% of `--mtp 2`'s drafts against b12283's 68% and wrote 4-7% slower; the switch tells whether this kernel is the cause, and keeps the old path where it measures better |

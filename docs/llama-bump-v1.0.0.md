# llama.cpp v1.0.0: what moved, and what it is worth to EuLLM

Branch `chore/llama-cpp-v1.0.0`, from `main` after release 0.8.9. The llama.cpp
submodule moves from b12283 (10 November) to the tag `v1.0.0` (b12359,
`b1b770d`, 13 November): 76 commits. Upstream now tags releases with semantic
versions besides the `bNNNNN` build numbers; v1.0.0 is the first major one.
The 17 commits after it on 13 November are left for the next bump, among them
#32157, which moves the MoE cache's code out of `llama-graph.cpp` where `0003`
patches it, and a retuning of HIP's MMQ for GCN (#30021) worth measuring on
the MI250X then.

## 1. What llama.cpp 1.0 removed

Two pruning PRs, "llama : prune" (#30565) and "ggml : prune" (#29801), drop
what was marked deprecated: 81 functions of `llama.h`, `ggml.h`, `common.h` and
`src/llama-ext.h`, which is gone (its NextN and speculative API moved to
`llama.h`, `llama_set_embd_state_src` renamed `llama_set_ctx_state_src`), and
the `defrag_thold` field of `llama_context_params` with `--defrag-thold`.

EuLLM called ten of them, all in the vendored `llama-cpp-2`; none in the
engine, the C++ wrappers or the patches:

| File | Removed | Now calls |
|---|---|---|
| `model.rs` | `llama_load_model_from_file`, `llama_free_model`, `llama_new_context_with_model`, `llama_n_ctx_train`, `llama_n_embd` | `llama_model_load_from_file`, `llama_model_free`, `llama_init_from_model`, `llama_model_n_ctx_train`, `llama_model_n_embd` |
| `context/session.rs` | `llama_get_state_size`, `llama_copy_state_data`, `llama_set_state_data`, `llama_load_session_file`, `llama_save_session_file` | `llama_state_get_size`, `llama_state_get_data` (no size limit, as the old call), `llama_state_set_data` (bounded by the slice), `llama_state_load_file`, `llama_state_save_file` |
| `context/params/get_set.rs` | the `defrag_thold` field | `with_defrag_thold` and `defrag_thold` removed; the engine used neither |

The Rust methods keep their names, so nothing above `llama-cpp-2` changed.
Upstream llama-cpp-rs had not followed: its `main` tracks llama.cpp v0.9.0
(4 November) and now generates its bindings ahead of time instead of in the
build, a change EuLLM does not take with this bump.

## 2. What EuLLM gains

- **Few-row MMQ** (#31533): quantized products of 2 to 24 tokens get an int8
  MMA kernel of their own (Q8_0, Q3-Q6_K, IQ3_S, IQ4_NL, IQ4_XS; NVIDIA from
  Ampere on, not HIP). That is continuous batching with 2-24 requests and the
  check of `--mtp`'s drafts.
- **BF16 with few rows on MMVF** (#31959) at batch 2-8: the "down" projections
  of the hyper-connections. `0006` does their "up" projections at batch 1; the
  two do not overlap and `0006` stays.
- MoE: MMQ's stream-k fixup skips empty tiles (#32067); upstream's own
  prefetch reads only results of earlier splits, so never an expert the cache
  is still uploading, and the cache turns itself off where its device cannot
  run the matmul (#32100).
- ROCm 7 and newer keep one BLAS workspace (#31838): the RDNA and Windows ROCm
  builds; not LUMI-G (ROCm 6.3).
- Graph results reused across reserves (#32089), unused graph metadata pages
  left untouched (#32043), CUDA graphs re-instantiated when a kernel's cluster
  dims change.
- OCP FP8 on CUDA (#28413), the CPU's flash attention accumulating V in F32
  (#29810), Vulkan flash attention for Intel at prefill (#29357).

## 3. Patches

All twelve apply to v1.0.0 as they were (`cargo test --test llama_patches`:
11 of 11). Upstream touched none of the code they change.

## 4. Checked here (cloud container, 4 cores, no GPU)

- CPU release build with no warning; `cargo test`: 772 unit tests and the 11
  `llama_patches` tests; `cargo clippy --no-deps -- -D warnings`: nothing.
- `engine/src/llama_archs.rs` regenerated: the same 172 architectures.
- The real-model tests (`--ignored real_model`, release build, CPU): 35 of
  35, with stories260K F32 for generation, Qwen3-0.6B Q8_0 and its embedding
  model, Qwen3.5-0.8B with and without MTP, and Spark-X2.5-1.7B. The two
  decision tests that load a backend of their own ran in a process of their
  own each; the equivalence of the decision modes on the F32 model, as its
  message asks (on Qwen3-0.6B Q8_0 batched and separate evaluation differ by
  up to 0.41 in a log-probability, quantized rounding).

Not checked here: any GPU run, and the CUDA, ROCm, Vulkan and Metal builds.

## 5. To measure before release 0.9.0

On the reference PC (RTX 5070 Ti), 0.8.9 against this branch, as for b12283:
Qwen3.6-35B-A3B with its expert cache, `prefetch_check` on Qwen3-Next-80B and
Flash-Next, DeepSeek-Coder-V2-Lite whole in VRAM, Qwen3.5-9B with `--mtp 2`,
and, for #31533, a dense model answering four and eight requests at once
(`--batch-size 4`/`8`).

On LUMI-G: `tools/lumi/build_engine.sh` and the smoke job.

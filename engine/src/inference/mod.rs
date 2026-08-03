//! Inference engine powered by llama.cpp via llama-cpp-2 bindings.
//!
//! Supports two modes:
//!
//! - **Sequential** (`InferenceEngine`): one request at a time, simple mutex.
//!   Good for single-user CLI usage.
//! - **Continuous batching** (`BatchScheduler`): multiple concurrent requests
//!   decoded in parallel on a single context. Good for API server / RAG
//!   workloads with many parallel requests.

pub(crate) mod output;
pub(crate) mod sampling;
pub mod scheduler;

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::pin::pin;

// Re-export KvCacheType for use in CLI and API.
pub use llama_cpp_2::context::params::KvCacheType;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use parking_lot::Mutex;
use tokio::sync::mpsc;

/// Whether this binary was compiled with any GPU backend at all.
pub const fn has_gpu_backend() -> bool {
    cfg!(feature = "cuda")
        || cfg!(feature = "rocm")
        || cfg!(feature = "vulkan")
        || cfg!(feature = "metal")
}

/// Report the binary's GPU capability and return the layer count that may
/// actually be handed to llama.cpp.
///
/// **This return value is not advisory.** A binary compiled without any GPU
/// backend must ask for zero offloaded layers, and until v0.6.39 it did not:
/// it printed "All inference will run on CPU" and then passed
/// `n_gpu_layers = 1000`. On Linux and Windows that was harmless, because
/// without a GPU feature there is no backend for llama.cpp to offload *to*.
/// On macOS it was not harmless — ggml compiles the Metal backend by default
/// on every Apple target, independent of our cargo features, so llama.cpp
/// dutifully offloaded all 29 layers onto whatever Metal device it found.
///
/// Confirmed in the field on two Intel Macs in issue #140, on a build we had
/// been describing as CPU-only since v0.6.30: a Mac mini offloading to an
/// Intel UHD 630 and a MacBook Pro offloading to an AMD Radeon Pro 560X.
/// Metal on non-Apple GPUs is a documented source of wrong results upstream
/// (ggml-org/llama.cpp#19563, #4004), and that machine produced NaN logits on
/// every single request. The build-side half of the fix is in
/// `llama-cpp-sys-2`'s build.rs, which now honours the `metal` feature
/// instead of always enabling the backend on Apple; this half makes the
/// runtime request match what the binary claims to support, on every platform
/// and regardless of how it was built.
#[must_use]
pub fn check_gpu_support(gpu_layers: i32) -> i32 {
    let has_gpu = has_gpu_backend();

    if gpu_layers != 0 && !has_gpu {
        eprintln!();
        eprintln!("╔══════════════════════════════════════════════════════════════╗");
        eprintln!("║  WARNING: GPU requested but this binary has no GPU support  ║");
        eprintln!("║                                                              ║");
        eprintln!("║  All inference will run on CPU (very slow for large prompts) ║");
        eprintln!("║                                                              ║");
        eprintln!("║  Rebuild with GPU support:                                   ║");
        eprintln!("║    cargo build --release --features cuda    # NVIDIA         ║");
        eprintln!("║    cargo build --release --features rocm    # AMD            ║");
        eprintln!("║    cargo build --release --features vulkan  # Cross-platform ║");
        eprintln!("║    cargo build --release --features metal   # Apple Silicon  ║");
        eprintln!("║                                                              ║");
        eprintln!("║  Docker: use the engine-gpu service or build with:           ║");
        eprintln!("║    docker build --build-arg FEATURES=cuda -t eullm .         ║");
        eprintln!("╚══════════════════════════════════════════════════════════════╝");
        eprintln!();
        tracing::warn!(
            "No GPU backend compiled: requested gpu_layers={gpu_layers}, forcing 0. \
             Rebuild with --features cuda/rocm/vulkan/metal for GPU acceleration."
        );
        return 0;
    }

    if has_gpu {
        let backend_name = if cfg!(feature = "cuda") {
            "CUDA"
        } else if cfg!(feature = "rocm") {
            "ROCm"
        } else if cfg!(feature = "vulkan") {
            "Vulkan"
        } else {
            "Metal"
        };
        tracing::info!("GPU backend: {backend_name}");
    }

    gpu_layers
}

/// Summary of the CPU SIMD features this binary was actually compiled
/// with (AVX, AVX2, FMA, NEON, etc.), via llama.cpp's own
/// `llama_print_system_info()`. A pure report of compile-time flags baked
/// into this binary — it does not probe the running CPU's own
/// capabilities, and does not require the backend or a model to be
/// loaded first.
///
/// Printed once at startup (unconditionally, not gated behind
/// `RUST_LOG=debug`) so a bug report never has to guess what instruction
/// set a binary actually contains. Before this existed, confirming
/// whether a release binary really had AVX2 compiled in required
/// downloading and disassembling it by hand — see
/// docs/x86-64-baseline.md for the investigation that hit exactly that
/// wall.
pub fn cpu_features_summary() -> String {
    let ptr = unsafe { llama_cpp_sys_2::llama_print_system_info() };
    if ptr.is_null() {
        return "unknown (llama_print_system_info returned null)".to_string();
    }
    unsafe { std::ffi::CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

/// Vision encoders use NON-causal attention, which requires the entire image
/// to land in a single micro-batch: `n_ubatch >= image_tokens`. Sized to the
/// image token budget (`EULLM_IMAGE_MAX_TOKENS`) so a higher budget genuinely
/// raises resolution instead of hard-aborting (GGML_ASSERT "non-causal
/// attention requires n_ubatch >= n_tokens"). Deliberately NOT derived from
/// `config.n_batch` (the text prefill batch, 2048 by default): that batch size
/// has nothing to do with how many tokens one image encodes to (Gemma 4's clip
/// output is ~256-300 tokens per slice — verified against a real load's
/// `n_tokens_batch` log line), and reserving a compute buffer sized for a
/// 2048-token micro-batch by default squeezed the KV cache — and so n_ctx —
/// far more than any single image actually required. 512 is the floor,
/// comfortably above a typical single-slice image. Shared between
/// `generate_multimodal`'s real context and `probe_and_shrink_context`'s
/// probe, which must agree — a probe built from a smaller batch than the
/// request that follows it proves nothing.
fn multimodal_batch_size() -> u32 {
    let img_budget = std::env::var("EULLM_IMAGE_MAX_TOKENS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(0);
    img_budget.max(512)
}

/// Build context params with flash attention, n_batch, and KV cache types applied.
pub(crate) fn build_ctx_params(
    config: &InferenceConfig,
    ctx_size: NonZeroU32,
) -> LlamaContextParams {
    build_ctx_params_with_cache(config, ctx_size, config.cache_type_k, config.cache_type_v)
}

/// Build context params, allowing KV cache types to be overridden (used for
/// automatic fallback from quantized → F16 when the GPU doesn't support
/// Flash Attention with quantized V cache).
pub(crate) fn build_ctx_params_with_cache(
    config: &InferenceConfig,
    ctx_size: NonZeroU32,
    cache_type_k: KvCacheType,
    cache_type_v: KvCacheType,
) -> LlamaContextParams {
    // n_ubatch is the *physical* micro-batch the GPU actually processes in one
    // pass during prefill — distinct from n_batch, which is just the logical
    // ceiling on how many tokens can be queued. Left unset, llama.cpp defaults
    // it to 512 regardless of n_batch, so prefill never used the larger batch
    // this server configures. 1024 is a deliberately conservative bump (not
    // n_batch's full 2048) since a bigger micro-batch needs a bigger compute
    // buffer, and fit.rs's COMPUTE_BUFFER_RESERVE_BYTES hasn't been
    // recalibrated against real GPU measurements at this value yet. Must never exceed
    // n_batch (llama.cpp requirement), hence the min().
    let n_ubatch = config.n_batch.min(1024);
    let mut params = LlamaContextParams::default()
        .with_n_ctx(Some(ctx_size))
        .with_n_batch(config.n_batch)
        .with_n_ubatch(n_ubatch)
        .with_n_threads(config.threads as i32)
        .with_n_threads_batch(config.threads as i32)
        .with_n_rs_seq(config.rs_seq);

    if cache_type_k != KvCacheType::F16 || cache_type_v != KvCacheType::F16 {
        params = params.with_type_k(cache_type_k).with_type_v(cache_type_v);
    }

    // The default flash_attn_type in llama.cpp is AUTO (-1), not DISABLED (0).
    // We must always set it explicitly — if we skip this when flash_attn=false,
    // the default AUTO mode stays active, which can enable FA unexpectedly.
    let fa_policy: i32 = if config.flash_attn {
        // AUTO (-1): llama.cpp tests whether the GPU supports FA for the
        // current KV cache types.  If not → disables FA and uses standard
        // attention.  ENABLED (1) would skip this check.
        -1
    } else {
        // DISABLED (0): explicitly turn off FA.
        // Note: llama.cpp requires flash_attn for quantized V cache; if V is
        // quantized and FA is disabled, context creation returns an error and
        // our fallback logic switches to F16/F16.
        0
    };
    params = params.with_flash_attention_policy(fa_policy);
    params
}

/// Sampling seed to use when the client doesn't specify one.
///
/// Ollama's real default is a fresh, effectively random seed per request
/// (-1), not a fixed value — confirmed against Ollama's own docs/source
/// (see ollama/ollama#7691). Wall-clock nanoseconds give enough variety for
/// sampling without pulling in a `rand` dependency for this. `fallback` is
/// only used in the practically-impossible case the system clock predates
/// the Unix epoch.
pub(crate) fn random_seed_fallback(fallback: u32) -> u32 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(fallback)
}

pub use scheduler::{BatchScheduler, SchedulerConfig, SchedulerHandle};

/// Configuration for the inference engine.
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Path to the GGUF model file.
    pub model_path: PathBuf,
    /// Number of GPU layers to offload (-1 = all).
    pub gpu_layers: i32,
    /// Total context window size (shared across batch slots in scheduler mode).
    pub context_size: u32,
    /// Number of threads for CPU inference.
    pub threads: u32,
    /// Enable flash attention (reduces memory bandwidth, faster decode).
    pub flash_attn: bool,
    /// Prompt processing batch size (how many tokens per eval during prefill).
    pub n_batch: u32,
    /// KV cache data type for keys.  Lower precision = less VRAM.
    /// Default: F16 (maximum GPU compatibility).
    pub cache_type_k: KvCacheType,
    /// KV cache data type for values.  Lower precision = less VRAM.
    /// Default: F16 (maximum GPU compatibility).
    pub cache_type_v: KvCacheType,
    /// Optional path to a multimodal projector (mmproj) GGUF file. When
    /// present AND the `multimodal` cargo feature is enabled, the engine
    /// loads an `MtmdContext` alongside the text model so it can accept
    /// image / audio input. When `None` or the feature is off, the engine
    /// runs text-only exactly as before.
    pub mmproj_path: Option<PathBuf>,
    /// Keep MoE expert tensors (`*.ffn_(up|down|gate)_exps`) on CPU RAM
    /// while everything else — attention, embeddings, shared/dense layers,
    /// and the KV cache — loads onto GPU as usual. Equivalent to llama.cpp's
    /// `--cpu-moe`. For MoE models the expert tensors are the bulk of the
    /// file but only a handful fire per token, so this trades a small
    /// compute-bandwidth cost (CPU matmul for the routed experts) for a much
    /// smaller VRAM footprint than offloading whole layers via `gpu_layers`.
    /// No effect on dense (non-MoE) models — the tensor pattern simply
    /// matches nothing.
    pub cpu_moe: bool,
    /// Keep MoE expert tensors on CPU RAM for only the first `n_cpu_moe`
    /// transformer layers (0 = disabled), leaving the rest on GPU. Finer
    /// grained than `cpu_moe`: lets a model that doesn't fully fit in VRAM
    /// offload just enough expert weight to close the gap instead of moving
    /// every expert tensor to CPU. Equivalent to llama.cpp's `--n-cpu-moe`.
    /// Mutually exclusive with `cpu_moe` (blanket override wins if both are
    /// set — checked at the CLI layer).
    pub n_cpu_moe: u32,
    /// Recurrent-state rollback window (llama.cpp's `n_rs_seq`) for hybrid/
    /// recurrent architectures (Mamba/Gated-DeltaNet-style SSM layers, e.g.
    /// Qwen3.5/3.6's hybrid attention+SSM design). `0` (default, matching
    /// `llama_context_default_params()`, strongly recommended) means
    /// KV-cache prefix reuse can never roll back that architecture's
    /// recurrent state at all — every reused-prefill attempt with
    /// `reuse_len > 0` is rejected and the scheduler falls back to a full
    /// re-prefill (see `prefill_sequence` in `inference/scheduler.rs`).
    ///
    /// This is NOT a general conversation/KV-reuse knob, despite the
    /// naming suggesting otherwise. Confirmed against upstream llama.cpp
    /// source (`tools/server/server-context.cpp`, `common/common.cpp`):
    /// the official server only ever derives `n_rs_seq` from
    /// speculative-decoding draft length (`need_n_rs_seq()`, single-digit
    /// to low-teens) and hard-zeroes it on non-speculative paths
    /// (`cparams_dft.n_rs_seq = 0`). The server's own mechanism for
    /// cross-turn prompt reuse on hybrid/recurrent architectures is a
    /// different, bounded feature — periodic full-state
    /// `server_prompt_checkpoint` snapshots (`--ctx-checkpoints`, default
    /// 32, `--checkpoint-min-step`, default 8192 tokens) with a graceful
    /// full-reprocessing fallback when no checkpoint covers the request.
    /// eullm does not yet implement an equivalent bounded-checkpoint
    /// mechanism; that is the correct future direction for this problem,
    /// not raising `rs_seq`.
    ///
    /// Setting this nonzero allows rollback up to that many positions
    /// back, but recurrent-state tensors scale by a factor of
    /// `(1 + n_rs_seq)` (confirmed in `llama-memory-recurrent.cpp`:
    /// `n_rows = mem_size * (1 + n_rs_seq)`), so even moderate values (64)
    /// have been observed to multiply resident memory by 2x+ on a 35B
    /// hybrid MoE model. No effect on architectures that don't support
    /// recurrent-state rollback (`llm_arch_supports_rs_rollback` in
    /// llama.cpp) or on non-hybrid models. Kept as an experimental,
    /// off-by-default escape hatch — not a recommended path to KV reuse
    /// on hybrid/recurrent architectures today.
    pub rs_seq: u32,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            gpu_layers: -1,
            context_size: 4096,
            threads: num_cpus(),
            flash_attn: true,
            n_batch: 2048,
            // F16 is the safest default — works on all GPU architectures.
            // Quantized types (Q8_0, Q4_0) save VRAM but may cause GPU
            // fallback to CPU on some architectures.  Users can opt in
            // with --cache-type-k / --cache-type-v after verifying with nvtop.
            cache_type_k: KvCacheType::F16,
            cache_type_v: KvCacheType::F16,
            mmproj_path: None,
            cpu_moe: false,
            n_cpu_moe: 0,
            rs_seq: 0,
        }
    }
}

/// Parse a KV cache type string (e.g. "q8_0", "q4_0", "f16") into a `KvCacheType`.
pub fn parse_cache_type(s: &str) -> Result<KvCacheType, String> {
    match s.to_lowercase().as_str() {
        "f16" => Ok(KvCacheType::F16),
        "f32" => Ok(KvCacheType::F32),
        "q8_0" => Ok(KvCacheType::Q8_0),
        "q4_0" => Ok(KvCacheType::Q4_0),
        "q4_1" => Ok(KvCacheType::Q4_1),
        "q5_0" => Ok(KvCacheType::Q5_0),
        "q5_1" => Ok(KvCacheType::Q5_1),
        _ => Err(format!(
            "Unknown cache type '{s}'. Options: f16, f32, q8_0, q4_0, q4_1, q5_0, q5_1"
        )),
    }
}

/// How many threads to run inference on when the user did not say.
///
/// # Why not simply every logical CPU
///
/// That is what this used to do, via `available_parallelism()`, and it is
/// measurably wrong on two very common machines:
///
/// * **Hyperthreaded x86.** ggml's decode is SIMD- and memory-bandwidth-bound,
///   so two threads sharing one core's execution units do not do twice the
///   work — they contend. On a 6-core/12-thread i9 we were asking for 12.
/// * **Apple Silicon.** `available_parallelism()` counts efficiency cores, and
///   ggml splits a graph evenly across its threads, so every step ends up
///   waiting on the slowest one. Four performance cores plus four efficiency
///   cores run *slower* than four performance cores alone.
///
/// Measured on a 4-core machine with no SMT at all, where over-subscription is
/// the only variable: 41.9 tok/s at 4 threads, 16.9 at 8, 14.6 at 12. Asking
/// for more threads than there are cores to run them costs about 60% of
/// throughput. That is not a rounding error, and on a thermally limited laptop
/// it compounds — more AVX-heavy threads means more power, means a lower
/// sustained clock.
///
/// So this mirrors what `llama-cli` does (`common_cpu_get_num_math` →
/// `common_cpu_get_num_physical_cores`): performance physical cores, falling
/// back to all physical cores, falling back to the logical count when the
/// platform will not tell us. Matching the reference implementation is also
/// what makes a like-for-like benchmark against it meaningful.
pub fn default_thread_count() -> u32 {
    physical_core_count().unwrap_or_else(logical_core_count)
}

/// Logical CPUs — the last-resort answer, and the previous default.
fn logical_core_count() -> u32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(4)
}

/// Performance physical cores, or `None` when the platform cannot say.
#[cfg(target_os = "macos")]
fn physical_core_count() -> Option<u32> {
    // `hw.perflevel0.physicalcpu` is the performance-core count on Apple
    // Silicon and equals `hw.physicalcpu` on Intel Macs, so one query covers
    // both and the second is only a fallback for older kernels.
    sysctl_u32("hw.perflevel0.physicalcpu")
        .or_else(|| sysctl_u32("hw.physicalcpu"))
        .filter(|&n| n > 0)
}

#[cfg(target_os = "macos")]
fn sysctl_u32(name: &str) -> Option<u32> {
    use std::ffi::CString;
    let cname = CString::new(name).ok()?;
    let mut value: i32 = 0;
    let mut len = std::mem::size_of::<i32>();
    // SAFETY: `cname` is a valid NUL-terminated string for the duration of the
    // call, and `value`/`len` are a correctly sized out-parameter pair.
    let rc = unsafe {
        libc::sysctlbyname(
            cname.as_ptr(),
            (&raw mut value).cast::<libc::c_void>(),
            &raw mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc == 0 && value > 0 {
        Some(value as u32)
    } else {
        None
    }
}

#[cfg(target_os = "linux")]
fn physical_core_count() -> Option<u32> {
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    parse_physical_cores(&cpuinfo)
}

/// Count distinct `(physical id, core id)` pairs in `/proc/cpuinfo`.
///
/// Split out from the file read so the parsing is testable against real
/// `/proc/cpuinfo` shapes without needing the matching hardware.
#[cfg(target_os = "linux")]
fn parse_physical_cores(cpuinfo: &str) -> Option<u32> {
    use std::collections::HashSet;
    let mut seen: HashSet<(String, String)> = HashSet::new();
    let (mut pkg, mut core) = (None, None);
    for line in cpuinfo.lines() {
        let Some((key, value)) = line.split_once(':') else {
            // A blank line ends one processor block. Record whatever it had.
            if line.trim().is_empty()
                && let (Some(p), Some(c)) = (pkg.take(), core.take())
            {
                seen.insert((p, c));
            }
            continue;
        };
        match key.trim() {
            "physical id" => pkg = Some(value.trim().to_string()),
            "core id" => core = Some(value.trim().to_string()),
            _ => {}
        }
    }
    // The last block may not be followed by a blank line.
    if let (Some(p), Some(c)) = (pkg, core) {
        seen.insert((p, c));
    }
    // Kernels on some architectures omit these fields entirely (notably ARM),
    // in which case we learned nothing and should say so rather than answer 0.
    (!seen.is_empty()).then_some(seen.len() as u32)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn physical_core_count() -> Option<u32> {
    None
}

/// Whether the token just sampled came out of a numerically broken
/// distribution — the always-on O(1) guard.
///
/// The sequential engine needs the same protection as the scheduler: without
/// it, `--batch-size 0` still streams garbage. See
/// `scheduler::sampled_token_is_corrupt` for the full reasoning and the case
/// that motivated it.
pub(crate) fn sampled_token_is_corrupt(
    ctx: &llama_cpp_2::context::LlamaContext,
    token: llama_cpp_2::token::LlamaToken,
) -> bool {
    // These call sites all sample with the `-1` sentinel, which is not a member
    // of the initialized-logits set, so `get_logits()` is the right accessor.
    ctx.get_logits()
        .get(token.0 as usize)
        .is_some_and(|v| v.is_nan() || v.is_infinite())
}

/// The message handed to the caller when the guard fires. Identical wording in
/// both engines so an operator searching for it finds one thing.
pub(crate) fn corrupt_logits_message() -> String {
    "the model produced a numerically invalid result (NaN/Inf logits) and \
     generation was stopped — this is a compute failure, not a bad prompt. \
     Try --no-flash-attn, and --rust-debug for the full diagnostic."
        .to_string()
}

/// Whether a KV cache type is a four-bit quantization.
///
/// Split out because keys and values are not equally tolerant of it. A 4-bit
/// *value* cache is a reasonable trade at long context; a 4-bit *key* cache is
/// not, and llama.cpp's own guidance tops out at q8_0 for keys. Confirmed the
/// hard way in issue #140: `--cache-type-k q4_0 --cache-type-v q4_0` produced
/// word salad on Metal, on x86 CPU and on ARM CPU alike, while the same builds
/// on the same machines were coherent with q8_0 keys.
pub fn is_four_bit(t: KvCacheType) -> bool {
    matches!(t, KvCacheType::Q4_0 | KvCacheType::Q4_1)
}

/// A four-bit key cache combined with flash attention produces garbage output.
/// Raise the key cache to q8_0 when both are asked for, and say so.
///
/// # What was actually measured
///
/// Bisected on x86 CPU with qwen3-0.6b at temperature 0, one variable at a
/// time. `--cache-type-k q4_0` with flash attention on turns "The capital of
/// France is Paris" into word salad, occasionally in the wrong script. The
/// same binary, model and prompt with `--no-flash-attn` answers correctly. It
/// reproduces identically with the batching scheduler and with the sequential
/// engine, so it is not a scheduler problem, and `--cache-type-v q4_0` with
/// f16 keys is fine, so it is the *key* cache specifically. External testing on
/// issue #140 saw the same thing on Metal and on ARM CPU, which rules out one
/// backend's kernels.
///
/// So this is not the gradual quality loss that quantizing a cache normally
/// buys you. It is a flash-attention path that does not handle four-bit keys
/// and produces nonsense rather than refusing.
///
/// # Why correct rather than warn
///
/// A warning still leaves the process generating garbage, and the person who
/// most needs the warning is the one running headless with the output going
/// into a pipeline. Raising K to q8_0 keeps flash attention — and its
/// throughput — while still saving most of the memory the operator was after;
/// silently turning flash attention off instead would halve their speed
/// invisibly, which is a worse surprise than using slightly more VRAM. Anyone
/// who genuinely wants four-bit keys can have them by also passing
/// `--no-flash-attn`, which is the configuration that actually works.
///
/// Mirrors `correct_kv_cache_for_model`, which does the same for Gemma 4.
///
/// Returns `(effective_k, corrected)`.
pub fn correct_kv_cache_for_flash_attn(
    cache_type_k: KvCacheType,
    flash_attn: bool,
) -> (KvCacheType, bool) {
    if flash_attn && is_four_bit(cache_type_k) {
        (KvCacheType::Q8_0, true)
    } else {
        (cache_type_k, false)
    }
}

/// Print the explanation for a correction made by
/// [`correct_kv_cache_for_flash_attn`].
pub fn report_flash_attn_kv_correction(requested_k: KvCacheType) {
    eprintln!(
        "[EULLM] --cache-type-k {} requested with flash attention enabled.",
        cache_type_name(requested_k)
    );
    eprintln!(
        "[EULLM] That combination produces incoherent output (measured on CPU and Metal), \
         so the key cache is being raised to q8_0."
    );
    eprintln!(
        "[EULLM] Pass --no-flash-attn as well if you really need 4-bit keys — that path \
         works, at a throughput cost."
    );
}

/// The CLI spelling of a cache type — the inverse of [`parse_cache_type`].
///
/// Used instead of `{:?}` wherever the value appears in a message a user might
/// copy back onto a command line: `Q4_0` is not something the parser accepts
/// without lowercasing, and suggesting text that does not work as typed is a
/// small, avoidable insult.
pub fn cache_type_name(t: KvCacheType) -> &'static str {
    match t {
        KvCacheType::F16 => "f16",
        KvCacheType::F32 => "f32",
        KvCacheType::Q8_0 => "q8_0",
        KvCacheType::Q4_0 => "q4_0",
        KvCacheType::Q4_1 => "q4_1",
        KvCacheType::Q5_0 => "q5_0",
        KvCacheType::Q5_1 => "q5_1",
        other => {
            debug_assert!(false, "unhandled KV cache type {other:?}");
            "f16"
        }
    }
}

/// Gemma 4's mixed SWA architecture (25 SWA layers at head_dim=256 + 5
/// global layers at head_dim=512) is incompatible with quantized KV cache in
/// stock llama.cpp: the SWA bypass to f16 has not yet been merged upstream.
/// Auto-correct all non-f16 KV to f16/f16 for Gemma 4.
///
/// Applied to every model load path (CLI `run`, and `swap_model` for both
/// `run`'s later swaps and any `serve` swap) so the correction can't be
/// bypassed by the entry point — a request-driven swap on `serve` hits the
/// same architecture constraint as a CLI launch.
///
/// Returns the (possibly corrected) cache types and whether a correction
/// was applied, so callers can report it through their own log/print path.
pub fn correct_kv_cache_for_model(
    model_name: &str,
    cache_type_k: KvCacheType,
    cache_type_v: KvCacheType,
) -> (KvCacheType, KvCacheType, bool) {
    let model_lower = model_name.to_lowercase();
    let is_gemma4 = model_lower.contains("gemma-4") || model_lower.contains("gemma4");
    let needs_correction = cache_type_k != KvCacheType::F16 || cache_type_v != KvCacheType::F16;
    if is_gemma4 && needs_correction {
        (KvCacheType::F16, KvCacheType::F16, true)
    } else {
        (cache_type_k, cache_type_v, false)
    }
}

/// Approximate bytes per element for a KV cache type. Quantized types use the
/// GGUF block byte ratio (e.g. Q4_0 packs 32 elements into 18 bytes, so 0.5625
/// B/elem). Used by both the runtime KV-memory estimate and the `--fit` sizer.
pub fn cache_type_bytes_per_elem(ct: &KvCacheType) -> f64 {
    match ct {
        KvCacheType::F16 => 2.0,
        KvCacheType::F32 => 4.0,
        KvCacheType::Q8_0 => 34.0 / 32.0,
        KvCacheType::Q4_0 => 18.0 / 32.0,
        KvCacheType::Q4_1 => 20.0 / 32.0,
        KvCacheType::Q5_0 => 22.0 / 32.0,
        KvCacheType::Q5_1 => 24.0 / 32.0,
        _ => 2.0, // default to F16
    }
}

/// Human-readable name for a KV cache type.
pub fn cache_type_display(ct: &KvCacheType) -> String {
    match ct {
        KvCacheType::F16 => "F16".to_string(),
        KvCacheType::F32 => "F32".to_string(),
        KvCacheType::Q8_0 => "Q8_0".to_string(),
        KvCacheType::Q4_0 => "Q4_0".to_string(),
        KvCacheType::Q4_1 => "Q4_1".to_string(),
        KvCacheType::Q5_0 => "Q5_0".to_string(),
        KvCacheType::Q5_1 => "Q5_1".to_string(),
        KvCacheType::Unknown(id) => format!("Unknown({id})"),
        _ => format!("{ct:?}"),
    }
}

/// Format-template artifacts that some models hallucinate as plain text
/// (Gemma 4 12B has been observed to emit GPT-OSS-style harmony markers like
/// `<|channel>thought<channel|>` at the start of replies and stray
/// `<image|>`/`<audio|>` mid-sentence). These are NOT tokens in the model
/// vocabulary — they're generated as literal characters — so the safest fix is
/// to drop them from the streamed output without ending the generation.
///
/// Filter sequences differ from stop sequences: a stop terminates the response,
/// a filter is silently elided. Both share the same hold-back buffer in
/// `process_piece` so a marker split across token boundaries still gets caught.
///
/// Order matters: longer "combined" patterns (`<|channel>thought<channel|>`)
/// are listed BEFORE the single delimiters so they match as one unit, leaving
/// no orphan word ("thought") in the output. If the model spits a Harmony
/// variant not in the combined list (e.g. `<|channel>foo<channel|>`), the
/// single delimiters still strip the visible noise; only the word in between
/// is left dangling. Add new combined patterns when new variants surface.
pub const DEFAULT_HARMONY_FILTERS: &[&str] = &[
    // Combined patterns — match the whole Harmony channel block as one piece,
    // including the role word in the middle. Observed values in the wild:
    // `thought` (Gemma 4 12B reasoning preamble), `analysis` and `final` are
    // the canonical GPT-OSS Harmony channels. These match only EMPTY blocks
    // (no content between the role and the closing tag); blocks with content
    // pass through to the client so the UI can render them as a Reasoning
    // section — see `app.js` for the matching Gemma `<|channel>thought…` pass.
    "<|channel>thought<channel|>",
    "<|channel>analysis<channel|>",
    "<|channel>final<channel|>",
    // Same blocks with a newline after the role word. Gemma 4 12B emits
    // exactly this, observed on a real request: `<|channel>thought\n<channel|>`
    // immediately followed by the answer. Because filtering is literal
    // substring matching, the no-newline forms above do not cover it, and the
    // scaffolding reached the user. Found by running the multimodal path for
    // the first time rather than by reading the code.
    "<|channel>thought\n<channel|>",
    "<|channel>analysis\n<channel|>",
    "<|channel>final\n<channel|>",
    "<|message|>",
    // Stray non-channel markers — for mid-sentence leakage we have no UX for.
    // Note: we deliberately do NOT scrub bare `<|channel>` / `<channel|>` as
    // substrings any more. Doing so used to leave `thought\n[reasoning]` as
    // naked text whenever Gemma 4 emitted a non-empty thought block.
    "<|image>",
    "<image|>",
    "<|audio>",
    "<audio|>",
];

/// The filter list for one request, given whether the client asked for
/// thinking.
///
/// With `think: false` the prompt already carries a closed, empty think block
/// (see [`crate::chat_template::ChatTemplate::think_suppression_prefix`]), so
/// any think tag appearing in the *output* is spurious by construction and is
/// stripped here.
///
/// That case is not hypothetical: until v0.6.39 the injected prefix was one
/// newline short of what Qwen3's own template emits, which put the prompt off
/// distribution and made the model reason in the visible channel and close
/// with the tag. The prefix is fixed, so this is the second line of defence
/// rather than the fix, and it costs nothing when the model behaves.
///
/// With `think: true` the tags are legitimate output the UI renders as a
/// reasoning section, and must not be touched.
///
/// This is a guard, not a parser: the filter mechanism matches literal
/// substrings, so it removes stray tags but cannot elide a whole
/// `<think>…</think>` block with content in it. Removing the markers of
/// reasoning the client did not ask for is an improvement on showing both;
/// suppressing the reasoning itself is the prefix's job.
pub fn default_filters(think: bool) -> Vec<String> {
    let mut out: Vec<String> = DEFAULT_HARMONY_FILTERS
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    if !think {
        out.push("</think>".to_string());
        out.push("<think>".to_string());
    }
    out
}

/// Request for text generation.
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub repeat_last_n: i32,
    pub seed: Option<u32>,
    pub stop_sequences: Vec<String>,
    /// Substrings to silently drop from the streamed output without ending
    /// generation. Use for format-template artifacts hallucinated by the
    /// model (e.g. Gemma 4 emitting GPT-OSS harmony markers as plain text).
    /// `DEFAULT_HARMONY_FILTERS` is applied by default via `Default`.
    pub filter_sequences: Vec<String>,
    /// Per-request context budget (Ollama `num_ctx`).  When `Some`, the
    /// validation uses this instead of the server-level `context_size`.
    /// Must be ≤ server `context_size` (clamped at prefill time).
    pub num_ctx: Option<u32>,
    /// Optional GBNF grammar string for constrained decoding.
    /// When set, the sampler enforces that output conforms to this grammar.
    /// Used by `format: "json"` to guarantee valid JSON output.
    pub grammar: Option<String>,
    /// When true, the prompt is used as-is without adding BOS token.
    /// Required for raw ChatML prompts that already contain special tokens.
    pub raw: bool,
}

impl Default for GenerateRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            max_tokens: 512,
            temperature: 0.8,
            top_k: 40,
            top_p: 0.9,
            min_p: 0.0,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            seed: None,
            stop_sequences: Vec::new(),
            filter_sequences: DEFAULT_HARMONY_FILTERS
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            num_ctx: None,
            grammar: None,
            raw: false,
        }
    }
}

/// Standard GBNF grammar that accepts any valid JSON value.
///
/// This is the same grammar that llama.cpp and Ollama use for `format: "json"`.
pub const JSON_GBNF: &str = r#"
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
    string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
    value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^\\"\x7F\x00-\x1F] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws

ws ::= ([ \t\n] ws)?
"#;

/// Result of text generation (non-streaming).
#[derive(Debug, Clone)]
pub struct GenerateResult {
    pub text: String,
    pub tokens_generated: u32,
    pub tokens_prompt: u32,
    pub duration_ms: u64,
    /// Why generation ended — see [`StopReason`]. Carried here for the same
    /// reason it is carried on the streaming event: without it the API cannot
    /// tell a finished answer from a truncated one.
    pub stop_reason: StopReason,
}

/// Why generation stopped.
///
/// This exists because the API used to report `"stop"` unconditionally, in
/// eleven separate hardcoded places, and the scheduler never told it anything
/// else. An answer cut off because the sequence ran out of context was
/// therefore indistinguishable from one the model chose to end — which is
/// exactly how a truncated answer gets read as the model being broken. Ollama
/// reports `"length"` for this and OpenAI reports `finish_reason: "length"`;
/// we now do too.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The model emitted an end-of-generation token, or a configured stop
    /// sequence matched. The answer is complete.
    Stop,
    /// The token budget ran out: either the caller's `num_predict`/`max_tokens`
    /// or, more often, the context left for this sequence. The answer is
    /// **truncated**.
    Length,
}

impl StopReason {
    /// The string Ollama and OpenAI clients expect.
    pub fn as_api_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
        }
    }
}

/// A single token event emitted during streaming generation.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// A text fragment (one or more decoded token characters).
    Token(String),
    /// Generation is complete.
    Done {
        tokens_generated: u32,
        tokens_prompt: u32,
        duration_ms: u64,
        /// Why generation ended — see [`StopReason`].
        stop_reason: StopReason,
    },
    /// An error occurred during generation.
    Error(String),
}

/// A chat prompt rendered through a model's own GGUF-embedded Jinja template,
/// returned by [`InferenceEngine::apply_jinja_chat_template`].
///
/// llama.cpp's own rendering also reports `thinking_start_tag`/
/// `thinking_end_tag` (the template's reasoning-block delimiters, when it
/// declares any) via `llama_cpp_2::model::JinjaChatTemplateResult` — not
/// carried here yet because nothing consumes them: stripping a reasoning
/// block from the *response* on this path is separate, not-yet-done work.
pub struct DynamicChatTemplate {
    /// The rendered prompt, ready to generate from as-is (`raw: true`).
    pub prompt: String,
}

/// The loaded inference engine, holding the model and backend.
///
/// This is the **sequential** engine — one request at a time. For concurrent
/// workloads use [`BatchScheduler`] instead.
///
/// Thread-safe: generation acquires a mutex on the context.
pub struct InferenceEngine {
    backend: LlamaBackend,
    model: LlamaModel,
    config: InferenceConfig,
    /// Mutex around context because llama.cpp context is not thread-safe.
    ctx_mutex: Mutex<()>,
    /// Multimodal context (image / audio input). Present only when the
    /// `multimodal` cargo feature is enabled and a valid mmproj path was
    /// supplied in the config. Wrapped in `Option` so the same struct
    /// definition works in text-only mode; the field exists but stays None.
    #[cfg(feature = "multimodal")]
    mtmd_ctx: Option<llama_cpp_2::mtmd::MtmdContext>,
}

// SAFETY: LlamaBackend and LlamaModel are safe to share across threads.
// We guard mutable context access with a Mutex.
unsafe impl Send for InferenceEngine {}
unsafe impl Sync for InferenceEngine {}

/// Explain a failed context allocation in terms of what the user chose.
///
/// llama.cpp answers a KV cache that does not fit with a null pointer, which
/// reaches the client as "Failed to create context: null reference from
/// llama.cpp" — true, and useless. The window and the memory it costs are
/// both known here, and the flag that changes them is the one the user just
/// set. Seen with `--ctx-size 131072` on a 4B model: 17 GB of KV cache, an
/// allocation that could not be served, and nothing on screen connecting the
/// three.
fn context_alloc_error(err: &impl std::fmt::Display, ctx_size: u32, kv_mib: f64) -> String {
    format!(
        "could not allocate a context of {ctx_size} tokens: {err}. Its KV cache alone needs about \
         {kv_mib:.0} MiB, which did not fit. Lower it with --ctx-size, or halve the cache with \
         --cache-type-k q8_0 --cache-type-v q8_0."
    )
}

impl InferenceEngine {
    /// The same load-time facts the scheduler reports, for the sequential
    /// path. Without this the startup banner silently prints less depending
    /// on which loader ran: multimodal models always load sequentially, so
    /// neither the KV cache size nor the model's trained context length was
    /// ever shown for them.
    pub fn ready_info(&self) -> scheduler::ModelReadyInfo {
        scheduler::estimate_kv_memory(
            &self.model,
            u64::from(self.config.context_size),
            &self.config.cache_type_k,
            &self.config.cache_type_v,
        )
    }

    /// The context size this engine actually loaded with.
    ///
    /// May be smaller than what was requested: `load()` shrinks it when the
    /// requested size does not fit (see `probe_and_shrink_context`). Callers
    /// that report the context size to a user — the startup banner, in
    /// particular — must read it from here rather than from the flag that was
    /// passed in, or they state a KV cost that belongs to a different size
    /// than the one printed next to it.
    pub fn context_size(&self) -> u32 {
        self.config.context_size
    }

    /// Render `messages` (role, content pairs) through this model's own chat
    /// template — the Jinja template embedded in its GGUF, applied with
    /// llama.cpp's own engine, the same way llama-server does by default —
    /// instead of guessing a hardcoded format from the model's name.
    ///
    /// Returns `None`, rather than an error, whenever the caller should fall
    /// back to its own known-good hardcoded template: either the GGUF has no
    /// embedded template at all (`was_explicit` came back false — llama.cpp
    /// silently rendered a built-in ChatML fallback that has nothing to do
    /// with this model), or the FFI call itself failed (logged at `warn`).
    /// A model with a real template that fails to render is the rare case;
    /// treating it the same as "no template" keeps the caller simple and
    /// never worse off than before this existed.
    ///
    /// Text-only: message content is a single string per message. Not used
    /// for the multimodal path (`generate_multimodal` builds its own
    /// mtmd-aware prompt; matching an image marker to whatever this template
    /// happens to emit is separate, harder work — see backlog).
    pub fn apply_jinja_chat_template(
        &self,
        messages: &[(&str, &str)],
    ) -> Option<DynamicChatTemplate> {
        let messages: Vec<llama_cpp_2::model::LlamaChatMessage> = messages
            .iter()
            .map(|(role, content)| {
                llama_cpp_2::model::LlamaChatMessage::new((*role).to_string(), (*content).to_string())
            })
            .collect::<Result<_, _>>()
            .inspect_err(|e| {
                tracing::warn!("chat message contained a null byte, cannot render via Jinja: {e}");
            })
            .ok()?;

        let result = self
            .model
            .apply_jinja_chat_template(&messages, /* add_generation_prompt */ true)
            .inspect_err(|e| {
                tracing::warn!("Jinja chat template rendering failed, falling back: {e}");
            })
            .ok()?;

        if !result.was_explicit {
            return None;
        }

        Some(DynamicChatTemplate {
            prompt: result.prompt,
        })
    }

    /// Load a GGUF model and prepare the inference engine.
    pub fn load(
        mut config: InferenceConfig,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        if !config.model_path.exists() {
            return Err(format!("Model file not found: {}", config.model_path.display()).into());
        }

        // Use the RETURNED value, never `config.gpu_layers` — on a binary with
        // no GPU backend this is 0, and asking for more offloads onto a
        // backend we do not support. See `check_gpu_support`.
        let gpu_layers = check_gpu_support(config.gpu_layers);

        tracing::info!("Initializing llama.cpp backend...");
        let mut backend = LlamaBackend::init()?;
        // Suppress llama.cpp/ggml internal log messages (CUDA graph warmup etc.)
        // so they don't pollute the interactive REPL or API response streams.
        // EULLM uses tracing for its own structured logging.
        backend.void_logs();

        let model_params = if gpu_layers >= 0 {
            LlamaModelParams::default().with_n_gpu_layers(gpu_layers as u32)
        } else {
            // -1 = offload all layers
            LlamaModelParams::default().with_n_gpu_layers(1000)
        };
        let mut model_params = pin!(model_params);
        // Patterns passed to `add_cpu_buft_override` are stored as raw pointers
        // (not copied) inside `model_params`, so they must outlive the
        // `load_from_file` call below — keep them bound in this local.
        let n_cpu_moe_patterns: Vec<std::ffi::CString> = (0..config.n_cpu_moe)
            .map(|i| {
                std::ffi::CString::new(format!(r"blk\.{i}\.ffn_(up|down|gate|gate_up)_(ch|)exps"))
                    .expect("pattern has no interior NUL")
            })
            .collect();
        if config.cpu_moe {
            tracing::info!("--cpu-moe: keeping MoE expert tensors on CPU RAM");
            model_params.as_mut().add_cpu_moe_override();
        } else if config.n_cpu_moe > 0 {
            tracing::info!(
                "--n-cpu-moe {}: keeping MoE expert tensors on CPU RAM for the first {} layers",
                config.n_cpu_moe,
                config.n_cpu_moe
            );
            for pattern in &n_cpu_moe_patterns {
                model_params.as_mut().add_cpu_buft_override(pattern);
            }
        }

        tracing::info!("Loading model: {}", config.model_path.display());
        let model = LlamaModel::load_from_file(&backend, &config.model_path, &model_params)
            .map_err(|e| format!("Failed to load model: {e}"))?;

        tracing::info!("Model loaded successfully.");

        #[cfg(feature = "multimodal")]
        let mtmd_ctx = Self::init_mtmd_optional(&config, &model)?;

        // Prove the configured context actually allocates before declaring the
        // load successful, and shrink it automatically if it does not.
        //
        // Every generate call below builds its own context from scratch (see
        // `generate`/`generate_streaming`/`generate_multimodal`), which used to
        // mean an oversized `--ctx-size` loaded the weights fine and only
        // failed on the *first request* — with an error naming the KV cost,
        // which is correct but arrives after the model already announced
        // "Model loaded successfully" and, over the API, after a client has
        // been told the same thing. Run after `init_mtmd_optional` so the
        // probe sees VRAM the way it will really be spent: a projector already
        // resident, exactly as in the report this was written against (a 12B
        // Q8 model plus its vision/audio projector on a 16 GB card, where the
        // text context was the thing that did not fit).
        config.context_size = Self::probe_and_shrink_context(&backend, &model, &config)?;

        Ok(Self {
            backend,
            model,
            config,
            ctx_mutex: Mutex::new(()),
            #[cfg(feature = "multimodal")]
            mtmd_ctx,
        })
    }

    /// Find the largest context size at or below `config.context_size` that
    /// this model can actually allocate on this GPU, right now, halving until
    /// one fits or a floor is reached.
    ///
    /// Each attempt is a real `new_context` call, immediately dropped — the
    /// only reliable test. The KV cost is not one formula across every
    /// architecture (Gemma 4's mixed sliding-window layers use two different
    /// window sizes internally), so estimating instead of trying would just
    /// move the discovery to a different wrong place, which is exactly what
    /// happened testing this by hand: `--ctx-size 4096` failed, `2048` worked,
    /// found by guessing rather than being told.
    ///
    /// Returns the size actually loaded with, so every later call — which
    /// builds its own context from `config.context_size` — uses a size already
    /// proven to fit rather than repeating the same failing allocation.
    /// Errors only when even the floor does not fit: that is a real "this
    /// model cannot run here", not a size to silently accept.
    fn probe_and_shrink_context(
        backend: &LlamaBackend,
        model: &LlamaModel,
        config: &InferenceConfig,
    ) -> Result<u32, Box<dyn std::error::Error + Send + Sync>> {
        const FLOOR: u32 = 512;
        // 0 means "use the default", matching the fallback every generate
        // call already applies — not a signal to clamp upward. An explicit
        // request below FLOOR is respected as the starting point and simply
        // fails outright if it does not fit, rather than being silently
        // raised.
        let requested = if config.context_size == 0 {
            4096
        } else {
            config.context_size
        };
        let mut candidate = requested;

        loop {
            let ctx_size = NonZeroU32::new(candidate).expect(
                "candidate is never 0: it starts at `requested` (already normalized away \
                 from 0 above) and is only ever produced by `candidate / 2`, floored at \
                 FLOOR which is nonzero",
            );
            // A model loaded with an mmproj gets requests through
            // `generate_multimodal`, whose context uses `multimodal_batch_size`
            // instead of `config.n_batch`/`n_ubatch` — larger, since a whole
            // image must fit in one micro-batch. Probing with the plain text
            // batch size would prove a compute-buffer requirement smaller than
            // the one a real image request makes, which is exactly the gap a
            // 12B Q8 vision model on real hardware exposed: load-time probe
            // passed, first image sent failed with the same OOM the probe was
            // built to catch.
            let mut probe_params = build_ctx_params(config, ctx_size);
            if config.mmproj_path.is_some() {
                let mm_batch = multimodal_batch_size();
                probe_params = probe_params.with_n_batch(mm_batch).with_n_ubatch(mm_batch);
            }
            match model.new_context(backend, probe_params) {
                Ok(ctx) => {
                    drop(ctx); // probing fit, not keeping it — see the load()-site comment
                    if candidate < requested {
                        let info = scheduler::estimate_kv_memory(
                            model,
                            u64::from(requested),
                            &config.cache_type_k,
                            &config.cache_type_v,
                        );
                        let kv_mib = info.kv_k_mib + info.kv_v_mib;
                        let msg = format!(
                            "--ctx-size {requested} does not fit here (its KV cache alone \
                             needs about {kv_mib:.0} MiB); reduced automatically to \
                             {candidate}, which does. Pass --ctx-size {candidate} yourself \
                             to silence this, or free up VRAM to keep {requested}."
                        );
                        tracing::warn!("{msg}");
                        eprintln!("[EULLM] {msg}");
                    }
                    return Ok(candidate);
                }
                Err(e) if candidate <= FLOOR => {
                    let info = scheduler::estimate_kv_memory(
                        model,
                        u64::from(candidate),
                        &config.cache_type_k,
                        &config.cache_type_v,
                    );
                    return Err(
                        context_alloc_error(&e, candidate, info.kv_k_mib + info.kv_v_mib).into(),
                    );
                }
                Err(_) => candidate = (candidate / 2).max(FLOOR),
            }
        }
    }

    /// Attempt to load the multimodal projector if the config has one and the
    /// `multimodal` cargo feature is enabled. Returns:
    /// - `Ok(None)` if no mmproj was configured (text-only run is intended);
    /// - `Ok(Some(ctx))` if the projector loaded; logs capability flags;
    /// - `Err(...)` if a path was supplied but loading failed (treated as
    ///   a hard configuration error — the user clearly wants multimodal
    ///   and we should not silently degrade).
    #[cfg(feature = "multimodal")]
    fn init_mtmd_optional(
        config: &InferenceConfig,
        model: &LlamaModel,
    ) -> Result<Option<llama_cpp_2::mtmd::MtmdContext>, Box<dyn std::error::Error + Send + Sync>>
    {
        use llama_cpp_2::mtmd::{MtmdContext, MtmdContextParams};

        let Some(mmproj_path) = config.mmproj_path.as_ref() else {
            return Ok(None);
        };
        if !mmproj_path.exists() {
            return Err(format!(
                "mmproj file not found: {} — multimodal load aborted",
                mmproj_path.display()
            )
            .into());
        }

        // Mirror the text-side GPU policy: offload mmproj to GPU when the
        // text model itself is being offloaded. Threads count carries over
        // for the CPU-side image preprocessing inside mtmd.
        let mut params = MtmdContextParams {
            // Same rule as the text side: a binary with no GPU backend must
            // not ask for one, whatever the config says (`check_gpu_support`).
            use_gpu: config.gpu_layers != 0 && has_gpu_backend(),
            print_timings: false,
            n_threads: config.threads as i32,
            ..MtmdContextParams::default()
        };

        // Dynamic-resolution vision models (e.g. Gemma 4) cap image tokens
        // (Gemma4UV defaults to max 280). That low cap aggressively downscales
        // images, hurting hard cases (dark / low-contrast / small subject).
        // EULLM_IMAGE_MAX_TOKENS / EULLM_IMAGE_MIN_TOKENS let us raise the
        // budget to give the encoder more resolution. `-1` keeps the default.
        // NOTE: very high values can OOM or crash some projectors — raise in
        // moderate steps (e.g. 512, then 1024).
        let env_i32 = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<i32>().ok());
        if let Some(maxt) = env_i32("EULLM_IMAGE_MAX_TOKENS") {
            params.image_max_tokens = maxt;
            tracing::info!("Override image_max_tokens = {maxt} (EULLM_IMAGE_MAX_TOKENS)");
        }
        if let Some(mint) = env_i32("EULLM_IMAGE_MIN_TOKENS") {
            params.image_min_tokens = mint;
            tracing::info!("Override image_min_tokens = {mint} (EULLM_IMAGE_MIN_TOKENS)");
        }

        tracing::info!("Loading mmproj projector: {}", mmproj_path.display());
        let path_str = mmproj_path
            .to_str()
            .ok_or("mmproj path is not valid UTF-8")?;
        let ctx = MtmdContext::init_from_file(path_str, model, &params)
            .map_err(|e| format!("Failed to load mmproj: {e}"))?;

        // Log capability flags up front so the user (and our own logs) can
        // see at a glance what this projector enables. M-RoPE is the load-
        // time guard the plan called out: scalar position bookkeeping in
        // our scheduler/generate loop is incorrect for M-RoPE models.
        let support_vision = ctx.support_vision();
        let support_audio = ctx.support_audio();
        let use_mrope = ctx.decode_use_mrope();
        let use_non_causal = ctx.decode_use_non_causal();
        tracing::info!(
            "mmproj loaded — vision={support_vision} audio={support_audio} \
             m_rope={use_mrope} non_causal={use_non_causal}"
        );
        if use_mrope {
            tracing::warn!(
                "mmproj reports decode_use_mrope=true — this model uses \
                 multi-dimensional positions. The current MVP only supports \
                 scalar-position models (e.g. Gemma 4). Multimodal calls \
                 will refuse media input on this model."
            );
        }

        Ok(Some(ctx))
    }

    /// Generate text from a prompt (blocking, returns all at once).
    pub fn generate(
        &self,
        request: &GenerateRequest,
    ) -> Result<GenerateResult, Box<dyn std::error::Error + Send + Sync>> {
        let _lock = self.ctx_mutex.lock();
        let start = std::time::Instant::now();

        let ctx_size =
            NonZeroU32::new(self.config.context_size).unwrap_or(NonZeroU32::new(4096).unwrap());

        let has_quantized_cache = self.config.cache_type_k != KvCacheType::F16
            || self.config.cache_type_v != KvCacheType::F16;
        let ctx_params = build_ctx_params(&self.config, ctx_size);

        let mut ctx = match self.model.new_context(&self.backend, ctx_params) {
            Ok(c) => c,
            Err(e) if has_quantized_cache => {
                tracing::warn!(
                    "Context creation failed with quantized KV cache, falling back to F16"
                );
                let fallback = build_ctx_params_with_cache(
                    &self.config,
                    ctx_size,
                    KvCacheType::F16,
                    KvCacheType::F16,
                );
                self.model.new_context(&self.backend, fallback)
                    .map_err(|e2| format!("Failed to create context (F16 fallback failed too): {e2}\nOriginal error: {e}"))?
            }
            Err(e) => return Err(format!("Failed to create context: {e}").into()),
        };

        // Tokenize the prompt
        let tokens = self
            .model
            .str_to_token(
                &request.prompt,
                if request.raw {
                    AddBos::Never
                } else {
                    AddBos::Always
                },
            )
            .map_err(|e| format!("Tokenization failed: {e}"))?;

        let tokens_prompt = tokens.len() as u32;

        // Determine effective context budget: per-request num_ctx (clamped
        // to server context_size) or the server default.
        let server_ctx = ctx.n_ctx();
        let effective_ctx = request
            .num_ctx
            .map(|n| n.min(server_ctx))
            .unwrap_or(server_ctx);

        // Validate: prompt must fit in the budget and leave room for at
        // least one output token.
        if tokens_prompt >= effective_ctx {
            return Err(format!(
                "Prompt ({tokens_prompt} tokens) does not fit in context window ({effective_ctx})"
            )
            .into());
        }

        // Cap max_tokens so prompt + output stays within the budget.
        let max_output = effective_ctx - tokens_prompt;
        let max_tokens = request.max_tokens.min(max_output);
        let n_len = (tokens_prompt + max_tokens) as i32;

        if max_tokens < request.max_tokens {
            tracing::warn!(
                "num_predict capped: requested={}, effective={} (context={}, prompt_tokens={})",
                request.max_tokens,
                max_tokens,
                effective_ctx,
                tokens_prompt,
            );
        }
        tracing::info!(
            "Generate: prompt={} tokens, max_output={}, effective_ctx={}",
            tokens_prompt,
            max_tokens,
            effective_ctx,
        );

        // Prefill in chunks of n_batch tokens. llama.cpp asserts (SIGABRT)
        // if a single decode call processes more tokens than n_batch.
        {
            let chunk_size = self.config.n_batch as usize;
            let last_idx = tokens.len() - 1;

            for chunk_start in (0..tokens.len()).step_by(chunk_size) {
                let chunk_end = (chunk_start + chunk_size).min(tokens.len());
                let chunk = &tokens[chunk_start..chunk_end];
                let mut prefill_batch = LlamaBatch::new(chunk.len().max(1), 1);

                for (j, token) in chunk.iter().enumerate() {
                    let abs_pos = chunk_start + j;
                    let is_last = abs_pos == last_idx;
                    prefill_batch.add(*token, abs_pos as i32, &[0], is_last)?;
                }

                ctx.decode(&mut prefill_batch)
                    .map_err(|e| format!("Prompt decode failed: {e}"))?;
            }
        }

        // Sample tokens — use a small batch (capacity 1) for the decode loop.
        // When a grammar is requested (e.g. format:"json"), we prepend a
        // grammar sampler that constrains the output to valid syntax.
        let mut sampler = sampling::build_sampler(&self.model, request, 1234);

        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut output = String::new();
        // Trailing text that could still grow into a stop or filter sequence.
        // Dropped on EOG (it is a partial turn delimiter), flushed into
        // `output` when the token budget ends the loop (it is real text).
        let mut pending = String::new();
        let mut n_cur = tokens_prompt as i32;
        let mut tokens_generated: u32 = 0;
        let mut batch = LlamaBatch::new(1, 1);
        // Assume truncation and prove otherwise: the loop can end by exhausting
        // its token budget, and that is the case the API used to misreport as a
        // clean stop.
        let mut stop_reason = StopReason::Length;

        while n_cur <= n_len && tokens_generated < max_tokens {
            // Sample from the last output (-1). After prompt decode there is
            // exactly one output (the final prompt token with logits=true);
            // after single-token decode steps there is also exactly one output.
            let token = sampler.sample(&ctx, -1);

            // Always-on O(1) guard — see `sampled_token_is_corrupt`. A NaN on
            // the chosen token means the forward pass produced nothing usable,
            // and continuing would return garbage that reads as an answer.
            if sampled_token_is_corrupt(&ctx, token) {
                tracing::error!(
                    "sampled token {} has a NaN/Inf logit after {} tokens — aborting \
                     generation. Run with --rust-debug for the full logit scan.",
                    token.0,
                    tokens_generated,
                );
                return Err(corrupt_logits_message().into());
            }
            sampler.accept(token);

            // End of generation?
            if self.model.is_eog_token(token) {
                stop_reason = StopReason::Stop;
                break;
            }

            // Decode token to text. Stop sequences and filters go through the
            // shared hold-back logic in `output`, the same code the scheduler
            // runs — see that module for the three ways the old per-loop
            // `output.ends_with(stop)` check was wrong.
            match self.model.token_to_piece(token, &mut decoder, true, None) {
                Ok(piece) => match output::process_piece(
                    &mut pending,
                    &request.stop_sequences,
                    &request.filter_sequences,
                    &piece,
                ) {
                    output::PieceOutcome::Emit(text) => output.push_str(&text),
                    output::PieceOutcome::Stop(text) => {
                        output.push_str(&text);
                        stop_reason = StopReason::Stop;
                        break;
                    }
                },
                Err(_) => break,
            }

            // Prepare next batch — reuse the pre-allocated small batch.
            batch.clear();
            batch.add(token, n_cur, &[0], true)?;

            ctx.decode(&mut batch)
                .map_err(|e| format!("Decode failed at token {tokens_generated}: {e}"))?;

            n_cur += 1;
            tokens_generated += 1;
        }

        // The loop ran out of budget rather than hitting a marker: whatever is
        // held back is ordinary text, and dropping it would truncate the answer.
        if matches!(stop_reason, StopReason::Length) {
            output.push_str(&output::flush(&mut pending));
        }

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(GenerateResult {
            text: output,
            tokens_generated,
            tokens_prompt,
            duration_ms,
            stop_reason,
        })
    }

    /// Generate text with streaming — sends each token through the channel as it's produced.
    ///
    /// The caller receives `StreamEvent::Token(piece)` for each generated piece,
    /// then `StreamEvent::Done { ... }` when generation is complete.
    pub fn generate_streaming(&self, request: &GenerateRequest, tx: mpsc::Sender<StreamEvent>) {
        let _lock = self.ctx_mutex.lock();
        let start = std::time::Instant::now();

        let ctx_size =
            NonZeroU32::new(self.config.context_size).unwrap_or(NonZeroU32::new(4096).unwrap());

        let has_quantized_cache = self.config.cache_type_k != KvCacheType::F16
            || self.config.cache_type_v != KvCacheType::F16;
        let ctx_params = build_ctx_params(&self.config, ctx_size);

        let mut ctx = match self.model.new_context(&self.backend, ctx_params) {
            Ok(c) => c,
            Err(e) if has_quantized_cache => {
                tracing::warn!(
                    "Context creation failed with quantized KV cache, falling back to F16"
                );
                let fallback = build_ctx_params_with_cache(
                    &self.config,
                    ctx_size,
                    KvCacheType::F16,
                    KvCacheType::F16,
                );
                match self.model.new_context(&self.backend, fallback) {
                    Ok(c) => c,
                    Err(e2) => {
                        let _ = tx.blocking_send(StreamEvent::Error(format!(
                            "F16 fallback failed: {e2} (original: {e})"
                        )));
                        return;
                    }
                }
            }
            Err(e) => {
                let info = crate::inference::scheduler::estimate_kv_memory(
                    &self.model,
                    u64::from(self.config.context_size),
                    &self.config.cache_type_k,
                    &self.config.cache_type_v,
                );
                let _ = tx.blocking_send(StreamEvent::Error(context_alloc_error(
                    &e,
                    self.config.context_size,
                    info.kv_k_mib + info.kv_v_mib,
                )));
                return;
            }
        };

        let tokens = match self.model.str_to_token(
            &request.prompt,
            if request.raw {
                AddBos::Never
            } else {
                AddBos::Always
            },
        ) {
            Ok(t) => t,
            Err(e) => {
                let _ = tx.blocking_send(StreamEvent::Error(format!("Tokenization failed: {e}")));
                return;
            }
        };

        let tokens_prompt = tokens.len() as u32;

        // Determine effective context budget (same logic as generate()).
        let server_ctx = ctx.n_ctx();
        let effective_ctx = request
            .num_ctx
            .map(|n| n.min(server_ctx))
            .unwrap_or(server_ctx);

        if tokens_prompt >= effective_ctx {
            let _ = tx.blocking_send(StreamEvent::Error(format!(
                "Prompt ({tokens_prompt} tokens) does not fit in context window ({effective_ctx})"
            )));
            return;
        }

        let max_output = effective_ctx - tokens_prompt;
        let max_tokens = request.max_tokens.min(max_output);
        let n_len = (tokens_prompt + max_tokens) as i32;

        if max_tokens < request.max_tokens {
            tracing::warn!(
                "num_predict capped: requested={}, effective={} (context={}, prompt_tokens={})",
                request.max_tokens,
                max_tokens,
                effective_ctx,
                tokens_prompt,
            );
        }
        tracing::info!(
            "Stream: prompt={} tokens, max_output={}, effective_ctx={}",
            tokens_prompt,
            max_tokens,
            effective_ctx,
        );

        // Prefill in chunks of n_batch tokens (same fix as generate()).
        {
            let chunk_size = self.config.n_batch as usize;
            let last_idx = tokens.len() - 1;

            for chunk_start in (0..tokens.len()).step_by(chunk_size) {
                let chunk_end = (chunk_start + chunk_size).min(tokens.len());
                let chunk = &tokens[chunk_start..chunk_end];
                let mut prefill_batch = LlamaBatch::new(chunk.len().max(1), 1);

                for (j, token) in chunk.iter().enumerate() {
                    let abs_pos = chunk_start + j;
                    let is_last = abs_pos == last_idx;
                    if prefill_batch
                        .add(*token, abs_pos as i32, &[0], is_last)
                        .is_err()
                    {
                        let _ =
                            tx.blocking_send(StreamEvent::Error("Failed to build batch".into()));
                        return;
                    }
                }

                if ctx.decode(&mut prefill_batch).is_err() {
                    let _ = tx.blocking_send(StreamEvent::Error("Prompt decode failed".into()));
                    return;
                }
            }
        }

        let mut sampler = sampling::build_sampler(&self.model, request, 1234);

        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut full_output = String::new();
        // See the non-streaming path: held-back tail, dropped on EOG, flushed
        // to the client when the token budget is what ended the loop.
        let mut pending = String::new();
        let mut n_cur = tokens_prompt as i32;
        let mut tokens_generated: u32 = 0;
        let mut batch = LlamaBatch::new(1, 1);
        // Assume truncation and prove otherwise: the loop can end by exhausting
        // its token budget, and that is the case the API used to misreport as a
        // clean stop.
        let mut stop_reason = StopReason::Length;

        while n_cur <= n_len && tokens_generated < max_tokens {
            let token = sampler.sample(&ctx, -1);

            // Always-on O(1) guard — see `sampled_token_is_corrupt`.
            if sampled_token_is_corrupt(&ctx, token) {
                tracing::error!(
                    "sampled token {} has a NaN/Inf logit after {} tokens — aborting \
                     generation. Run with --rust-debug for the full logit scan.",
                    token.0,
                    tokens_generated,
                );
                let _ = tx.blocking_send(StreamEvent::Error(corrupt_logits_message()));
                return;
            }
            sampler.accept(token);

            if self.model.is_eog_token(token) {
                stop_reason = StopReason::Stop;
                break;
            }

            // Streaming cannot retract what it has already sent, so stop and
            // filter sequences run through the shared hold-back buffer rather
            // than being cut out of the current piece after the fact. The old
            // code sliced `&piece[..piece.len() - s.len()]`, which sent the
            // first half of a marker split across two tokens and panicked
            // outright when the byte index landed inside a character.
            match self.model.token_to_piece(token, &mut decoder, true, None) {
                Ok(piece) => {
                    full_output.push_str(&piece);
                    match output::process_piece(
                        &mut pending,
                        &request.stop_sequences,
                        &request.filter_sequences,
                        &piece,
                    ) {
                        output::PieceOutcome::Stop(text) => {
                            if !text.is_empty()
                                && tx.blocking_send(StreamEvent::Token(text)).is_err()
                            {
                                return;
                            }
                            stop_reason = StopReason::Stop;
                            break;
                        }
                        output::PieceOutcome::Emit(text) => {
                            if !text.is_empty()
                                && tx.blocking_send(StreamEvent::Token(text)).is_err()
                            {
                                return; // receiver dropped
                            }
                        }
                    }
                }
                Err(_) => break,
            }

            batch.clear();
            if batch.add(token, n_cur, &[0], true).is_err() {
                break;
            }

            if ctx.decode(&mut batch).is_err() {
                let _ = tx.blocking_send(StreamEvent::Error(format!(
                    "Decode failed at token {tokens_generated}"
                )));
                return;
            }

            n_cur += 1;
            tokens_generated += 1;
        }

        // Budget-exhausted end: the held-back tail is real text the client is
        // still owed. On EOG or a stop it is a partial marker and is dropped.
        if matches!(stop_reason, StopReason::Length) {
            let tail = output::flush(&mut pending);
            if !tail.is_empty() && tx.blocking_send(StreamEvent::Token(tail)).is_err() {
                return;
            }
        }

        let duration_ms = start.elapsed().as_millis() as u64;
        let _ = tx.blocking_send(StreamEvent::Done {
            tokens_generated,
            tokens_prompt,
            duration_ms,
            stop_reason,
        });
    }

    /// Multimodal streaming generation: accepts image / audio media alongside
    /// a text prompt, runs the mtmd-aware prefill via `eval_chunks`, then
    /// hands off to the same single-token decode loop as `generate_streaming`.
    ///
    /// The text prompt MUST contain exactly one media marker (`<__media__>`,
    /// see [`llama_cpp_2::mtmd::mtmd_default_marker`]) for each entry in
    /// `media`. `media[i]` is the raw bytes of an image (jpg/png/bmp/gif) or
    /// audio file (wav/mp3/flac) — `MtmdBitmap::from_buffer` auto-detects.
    ///
    /// Returns immediately via `StreamEvent::Error` if multimodal is not
    /// configured for this engine, the M-RoPE guard refuses the model, or
    /// the requested modality is not supported by the loaded projector.
    #[cfg(feature = "multimodal")]
    pub fn generate_multimodal(
        &self,
        request: &GenerateRequest,
        media: &[Vec<u8>],
        tx: mpsc::Sender<StreamEvent>,
    ) {
        use llama_cpp_2::mtmd::{MtmdBitmap, MtmdInputText};

        let _lock = self.ctx_mutex.lock();
        let start = std::time::Instant::now();

        // ── 1. Guards ───────────────────────────────────────────────────
        // (0.6.1 experiment reverted: rebuilding MtmdContext per request did
        // not change the per-image-type failure — dark/low-contrast/portrait
        // images still degrade identically — so the cause is upstream in the
        // Gemma 4 projector/encoder, not shared mtmd state. Back to the
        // long-lived, load-once projector.)
        let Some(mtmd_ctx) = self.mtmd_ctx.as_ref() else {
            let _ = tx.blocking_send(StreamEvent::Error(
                "Multimodal not configured: load the model with a valid mmproj_path".into(),
            ));
            return;
        };
        if mtmd_ctx.decode_use_mrope() {
            let _ = tx.blocking_send(StreamEvent::Error(
                "This model requires M-RoPE positions, which the current MVP \
                 does not yet plumb through the decode loop. Image/audio input \
                 is refused; text-only requests still work via generate_streaming."
                    .into(),
            ));
            return;
        }
        // Reject the request if the user supplied a modality the projector
        // does not support, to fail loudly rather than silently mis-decode.
        let has_audio = media.iter().any(|b| {
            // miniaudio magic bytes: RIFF (wav), ID3/MP3 sync (mp3), fLaC (flac).
            // This is a cheap heuristic; mtmd would also error at from_buffer.
            b.starts_with(b"RIFF")
                || b.starts_with(b"ID3")
                || b.starts_with(b"fLaC")
                || (b.len() >= 2 && b[0] == 0xFF && (b[1] & 0xE0) == 0xE0)
        });
        if has_audio && !mtmd_ctx.support_audio() {
            let _ = tx.blocking_send(StreamEvent::Error(
                "This mmproj does not support audio input (vision-only projector)".into(),
            ));
            return;
        }
        if !has_audio && !media.is_empty() && !mtmd_ctx.support_vision() {
            let _ = tx.blocking_send(StreamEvent::Error(
                "This mmproj does not support image input".into(),
            ));
            return;
        }

        // ── 2. Build context (same code path as generate_streaming) ─────
        let ctx_size =
            NonZeroU32::new(self.config.context_size).unwrap_or(NonZeroU32::new(4096).unwrap());
        let has_quantized_cache = self.config.cache_type_k != KvCacheType::F16
            || self.config.cache_type_v != KvCacheType::F16;

        // See `multimodal_batch_size` — must match what `probe_and_shrink_context`
        // used to prove this context size fits, or the probe proves nothing.
        let mm_batch = multimodal_batch_size();
        let mk_params = |ctk, ctv| {
            build_ctx_params_with_cache(&self.config, ctx_size, ctk, ctv)
                .with_n_batch(mm_batch)
                .with_n_ubatch(mm_batch)
        };
        let ctx_params = mk_params(self.config.cache_type_k, self.config.cache_type_v);
        let mut ctx = match self.model.new_context(&self.backend, ctx_params) {
            Ok(c) => c,
            Err(e) if has_quantized_cache => {
                let fallback = mk_params(KvCacheType::F16, KvCacheType::F16);
                match self.model.new_context(&self.backend, fallback) {
                    Ok(c) => c,
                    Err(e2) => {
                        let _ = tx.blocking_send(StreamEvent::Error(format!(
                            "F16 fallback failed: {e2} (original: {e})"
                        )));
                        return;
                    }
                }
            }
            Err(e) => {
                let info = crate::inference::scheduler::estimate_kv_memory(
                    &self.model,
                    u64::from(self.config.context_size),
                    &self.config.cache_type_k,
                    &self.config.cache_type_v,
                );
                let _ = tx.blocking_send(StreamEvent::Error(context_alloc_error(
                    &e,
                    self.config.context_size,
                    info.kv_k_mib + info.kv_v_mib,
                )));
                return;
            }
        };

        // ── 3. Decode media bytes into mtmd bitmaps ─────────────────────
        let mut bitmaps: Vec<MtmdBitmap> = Vec::with_capacity(media.len());
        for (i, bytes) in media.iter().enumerate() {
            // llama-cpp-2 0.1.151 added a `placeholder` flag to from_buffer:
            // false = decode and load the actual media (what we need for inference).
            match MtmdBitmap::from_buffer(mtmd_ctx, bytes, false) {
                Ok(b) => bitmaps.push(b),
                Err(e) => {
                    // `NullResult` on its own tells the caller nothing, and the
                    // most common cause is simply an unsupported container: a
                    // .webp fails exactly like a corrupt file does. Name the
                    // formats so the answer is in the error rather than in the
                    // source.
                    let _ = tx.blocking_send(StreamEvent::Error(format!(
                        "Media #{i} failed to decode ({e:?}). Supported images \
                         are jpg, png, bmp and gif, and audio is wav, mp3 or \
                         flac. Other containers, webp among them, are not \
                         decoded by the multimodal backend."
                    )));
                    return;
                }
            }
        }
        let bitmap_refs: Vec<&MtmdBitmap> = bitmaps.iter().collect();

        // ── 4. Tokenize text + media → MtmdInputChunks ──────────────────
        // The prompt is hand-templated by the caller (turn markers + one
        // <__media__> marker per bitmap) but does NOT include <bos>.
        // `add_special = true` makes mtmd prepend the model's BOS, matching
        // llama.cpp's reference `mtmd-cli` (`add_special = add_bos` on the
        // first turn). Omitting BOS was the bug: Gemma needs it, and without
        // it the model degrades into "wall of text / line art" confabulation
        // on all but the easiest images (the missing-BOS prompt is malformed;
        // strong landscapes survived it, weaker subjects did not).
        // `parse_special = true` so the turn tokens (Gemma <start_of_turn>)
        // are recognised rather than tokenised literally.
        let input_text = MtmdInputText {
            text: request.prompt.clone(),
            add_special: true,
            parse_special: true,
        };
        let chunks = match mtmd_ctx.tokenize(input_text, &bitmap_refs) {
            Ok(c) => c,
            Err(e) => {
                let _ =
                    tx.blocking_send(StreamEvent::Error(format!("mtmd tokenize failed: {e:?}")));
                return;
            }
        };

        let tokens_prompt = chunks.total_tokens() as u32;
        let server_ctx = ctx.n_ctx();
        let effective_ctx = request
            .num_ctx
            .map(|n| n.min(server_ctx))
            .unwrap_or(server_ctx);
        if tokens_prompt >= effective_ctx {
            let _ = tx.blocking_send(StreamEvent::Error(format!(
                "Multimodal prompt ({tokens_prompt} tokens) does not fit in context window ({effective_ctx})"
            )));
            return;
        }

        // ── 5. mtmd-aware prefill: text chunks via llama_decode, media
        //      chunks via mtmd_encode + llama_decode, all handled internally.
        let new_n_past = match chunks.eval_chunks(
            mtmd_ctx,
            &ctx,
            0,
            0,
            self.config.n_batch as i32,
            true, // we want logits on the last token to start sampling
        ) {
            Ok(p) => p,
            Err(e) => {
                let _ = tx.blocking_send(StreamEvent::Error(format!(
                    "mtmd eval_chunks failed: {e:?}"
                )));
                return;
            }
        };

        let max_output = effective_ctx.saturating_sub(tokens_prompt);
        let max_tokens = request.max_tokens.min(max_output);
        let n_len = (tokens_prompt + max_tokens) as i32;

        tracing::info!(
            "Multimodal stream: media={} ({}), prompt_tokens={}, max_output={}, ctx={}",
            media.len(),
            if has_audio { "audio" } else { "image" },
            tokens_prompt,
            max_tokens,
            effective_ctx,
        );

        // ── 6. Sampler (identical to generate_streaming) ────────────────
        let mut sampler = sampling::build_sampler(&self.model, request, 1234);

        // ── 7. Decode loop (identical pattern to generate_streaming) ────
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut full_output = String::new();
        // See the non-streaming path: held-back tail, dropped on EOG, flushed
        // to the client when the token budget is what ended the loop.
        let mut pending = String::new();
        let mut n_cur = new_n_past;
        let mut tokens_generated: u32 = 0;
        let mut batch = LlamaBatch::new(1, 1);
        // Assume truncation and prove otherwise: the loop can end by exhausting
        // its token budget, and that is the case the API used to misreport as a
        // clean stop.
        let mut stop_reason = StopReason::Length;

        while n_cur <= n_len && tokens_generated < max_tokens {
            let token = sampler.sample(&ctx, -1);

            // Always-on O(1) guard — see `sampled_token_is_corrupt`.
            if sampled_token_is_corrupt(&ctx, token) {
                tracing::error!(
                    "sampled token {} has a NaN/Inf logit after {} tokens — aborting \
                     generation. Run with --rust-debug for the full logit scan.",
                    token.0,
                    tokens_generated,
                );
                let _ = tx.blocking_send(StreamEvent::Error(corrupt_logits_message()));
                return;
            }
            sampler.accept(token);
            if self.model.is_eog_token(token) {
                stop_reason = StopReason::Stop;
                break;
            }

            match self.model.token_to_piece(token, &mut decoder, true, None) {
                Ok(piece) => {
                    full_output.push_str(&piece);
                    // Same shared path as the two text loops. This one also
                    // gains `filter_sequences`, which it never applied at all:
                    // DEFAULT_HARMONY_FILTERS exists to strip `<|channel|>`
                    // scaffolding, and on the multimodal path that scaffolding
                    // was going straight to the user.
                    match output::process_piece(
                        &mut pending,
                        &request.stop_sequences,
                        &request.filter_sequences,
                        &piece,
                    ) {
                        output::PieceOutcome::Stop(text) => {
                            if !text.is_empty()
                                && tx.blocking_send(StreamEvent::Token(text)).is_err()
                            {
                                return;
                            }
                            stop_reason = StopReason::Stop;
                            break;
                        }
                        output::PieceOutcome::Emit(text) => {
                            if !text.is_empty()
                                && tx.blocking_send(StreamEvent::Token(text)).is_err()
                            {
                                return;
                            }
                        }
                    }
                }
                Err(_) => break,
            }

            batch.clear();
            if batch.add(token, n_cur, &[0], true).is_err() {
                break;
            }
            if ctx.decode(&mut batch).is_err() {
                let _ = tx.blocking_send(StreamEvent::Error(format!(
                    "Decode failed at token {tokens_generated}"
                )));
                return;
            }
            n_cur += 1;
            tokens_generated += 1;
        }

        // Budget-exhausted end: hand over the held-back tail (see the text
        // streaming path for why EOG and stop cases drop it instead).
        if matches!(stop_reason, StopReason::Length) {
            let tail = output::flush(&mut pending);
            if !tail.is_empty() && tx.blocking_send(StreamEvent::Token(tail)).is_err() {
                return;
            }
        }

        let duration_ms = start.elapsed().as_millis() as u64;
        let _ = tx.blocking_send(StreamEvent::Done {
            tokens_generated,
            tokens_prompt,
            duration_ms,
            stop_reason,
        });
    }
}

fn num_cpus() -> u32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    // A binary with no GPU backend used to print "All inference will run on
    // CPU" and then hand llama.cpp n_gpu_layers=1000 anyway. Harmless where
    // there is no backend to offload to; on macOS ggml compiles Metal by
    // default regardless of our features, so it offloaded the whole model
    // onto Intel and AMD GPUs that produce wrong results (issue #140).
    // The warning and the request must agree.
    #[test]
    fn gpu_layers_are_forced_to_zero_when_no_backend_is_compiled() {
        if has_gpu_backend() {
            assert_eq!(check_gpu_support(-1), -1, "offload-all must survive");
            assert_eq!(check_gpu_support(35), 35, "explicit count must survive");
        } else {
            assert_eq!(check_gpu_support(-1), 0, "offload-all must be refused");
            assert_eq!(check_gpu_support(35), 0, "explicit count must be refused");
        }
        // Asking for nothing is honoured either way.
        assert_eq!(check_gpu_support(0), 0);
    }

    // The old default (a fixed seq_id, or a hardcoded 1234 on the
    // sequential-fallback path) made every unseeded request through a given
    // slot fully deterministic, unlike Ollama's real -1-seed (random)
    // behavior. Two calls in the same nanosecond are astronomically
    // unlikely, so this is a meaningful check, not a flaky one.
    #[test]
    fn random_seed_fallback_is_not_a_fixed_value() {
        let a = random_seed_fallback(1234);
        let b = random_seed_fallback(1234);
        assert_ne!(
            a, b,
            "seed fallback must not collapse to the `fallback` argument"
        );
    }

    // Real smoke test, not just "does it compile": llama_print_system_info()
    // is a pure FFI call into ggml (no backend/model init needed), so this
    // actually exercises the CStr conversion against the real linked
    // llama.cpp and prints what it returns — verifying the mechanism this
    // session's CI-verified-but-runtime-unconfirmed AVX2 build gap needed.
    #[test]
    fn cpu_features_summary_returns_real_content() {
        let summary = cpu_features_summary();
        println!("cpu_features_summary(): {summary}");
        assert!(
            !summary.is_empty() && summary != "unknown (llama_print_system_info returned null)",
            "expected real content from llama_print_system_info, got: {summary:?}"
        );
    }
}

#[cfg(test)]
mod kv_cache_naming_tests {
    use super::*;

    #[test]
    fn cache_type_name_round_trips_through_the_parser() {
        // The names are printed in messages users copy back onto a command
        // line, so every one of them has to parse.
        for t in [
            KvCacheType::F16,
            KvCacheType::F32,
            KvCacheType::Q8_0,
            KvCacheType::Q4_0,
            KvCacheType::Q4_1,
            KvCacheType::Q5_0,
            KvCacheType::Q5_1,
        ] {
            let name = cache_type_name(t);
            assert_eq!(
                parse_cache_type(name).unwrap(),
                t,
                "'{name}' must parse back to the type it names"
            );
        }
    }

    #[test]
    fn only_four_bit_key_caches_are_flagged() {
        // q8_0 keys with a q4_0 value cache is the aggressive-but-usable
        // combination; it must not be warned about, or the warning becomes
        // noise and stops being read.
        assert!(is_four_bit(KvCacheType::Q4_0));
        assert!(is_four_bit(KvCacheType::Q4_1));
        for t in [
            KvCacheType::F16,
            KvCacheType::F32,
            KvCacheType::Q8_0,
            KvCacheType::Q5_0,
            KvCacheType::Q5_1,
        ] {
            assert!(!is_four_bit(t), "{t:?} is not a 4-bit type");
        }
    }
}

#[cfg(test)]
mod flash_attn_kv_correction_tests {
    use super::*;

    #[test]
    fn four_bit_keys_are_raised_when_flash_attention_is_on() {
        for k in [KvCacheType::Q4_0, KvCacheType::Q4_1] {
            let (out, corrected) = correct_kv_cache_for_flash_attn(k, true);
            assert!(corrected, "{k:?} with flash attention must be corrected");
            assert_eq!(out, KvCacheType::Q8_0);
        }
    }

    #[test]
    fn four_bit_keys_are_left_alone_without_flash_attention() {
        // That combination is the one that actually works — measured, not
        // assumed — so overriding it would be taking away the only way to get
        // a four-bit key cache at all.
        for k in [KvCacheType::Q4_0, KvCacheType::Q4_1] {
            let (out, corrected) = correct_kv_cache_for_flash_attn(k, false);
            assert!(!corrected);
            assert_eq!(out, k);
        }
    }

    #[test]
    fn everything_else_passes_through_untouched() {
        for k in [
            KvCacheType::F16,
            KvCacheType::F32,
            KvCacheType::Q8_0,
            KvCacheType::Q5_0,
            KvCacheType::Q5_1,
        ] {
            for fa in [true, false] {
                let (out, corrected) = correct_kv_cache_for_flash_attn(k, fa);
                assert!(!corrected, "{k:?} (flash_attn={fa}) must not be corrected");
                assert_eq!(out, k);
            }
        }
    }

    #[test]
    fn the_correction_is_idempotent() {
        // Applying it to its own output must not keep changing things — the
        // startup path runs it after the Gemma correction, and a correction
        // that moves on every pass is a bug waiting for a second caller.
        let (once, _) = correct_kv_cache_for_flash_attn(KvCacheType::Q4_0, true);
        let (twice, corrected) = correct_kv_cache_for_flash_attn(once, true);
        assert!(!corrected);
        assert_eq!(once, twice);
    }
}

#[cfg(all(test, target_os = "linux"))]
mod physical_core_tests {
    use super::*;

    #[test]
    fn counts_cores_not_hyperthreads() {
        // A 2-core/4-thread part: two logical CPUs share each core id.
        let cpuinfo = "\
processor\t: 0
physical id\t: 0
core id\t\t: 0

processor\t: 1
physical id\t: 0
core id\t\t: 1

processor\t: 2
physical id\t: 0
core id\t\t: 0

processor\t: 3
physical id\t: 0
core id\t\t: 1
";
        assert_eq!(parse_physical_cores(cpuinfo), Some(2));
    }

    #[test]
    fn counts_across_sockets() {
        // Same core ids on two packages must not collapse into one.
        let cpuinfo = "\
processor\t: 0
physical id\t: 0
core id\t\t: 0

processor\t: 1
physical id\t: 1
core id\t\t: 0
";
        assert_eq!(parse_physical_cores(cpuinfo), Some(2));
    }

    #[test]
    fn the_last_block_counts_even_without_a_trailing_blank_line() {
        let cpuinfo = "processor\t: 0\nphysical id\t: 0\ncore id\t\t: 0";
        assert_eq!(parse_physical_cores(cpuinfo), Some(1));
    }

    #[test]
    fn absent_topology_yields_none_rather_than_zero() {
        // Many ARM kernels omit `core id` entirely. Answering 0 there would be
        // worse than admitting we don't know and using the logical count.
        let cpuinfo = "processor\t: 0\nBogoMIPS\t: 108.00\n\nprocessor\t: 1\n";
        assert_eq!(parse_physical_cores(cpuinfo), None);
        assert_eq!(parse_physical_cores(""), None);
    }

    #[test]
    fn the_default_is_sane_on_whatever_machine_runs_the_tests() {
        let t = default_thread_count();
        assert!(t >= 1, "must never be zero");
        assert!(
            t <= logical_core_count(),
            "physical cores ({t}) cannot exceed logical ({})",
            logical_core_count()
        );
    }
}

#[cfg(test)]
mod think_filter_tests {
    use super::*;

    // The exact bytes Gemma 4 12B produced on a real multimodal request. The
    // no-newline form was in the list, this one was not, and the scaffolding
    // reached the user because filtering is literal substring matching.
    #[test]
    fn the_gemma_channel_preamble_with_a_newline_is_filtered() {
        let filters: Vec<String> = default_filters(true);
        let mut pending = String::new();
        let piece = "<|channel>thought\n<channel|>A small metallic device.";
        match output::process_piece(&mut pending, &[], &filters, piece) {
            output::PieceOutcome::Emit(out) => {
                assert!(!out.contains("<|channel>"), "leaked opener: {out:?}");
                assert!(!out.contains("<channel|>"), "leaked closer: {out:?}");
                assert!(
                    out.contains("A small metallic device."),
                    "ate the answer: {out:?}"
                );
            }
            other => panic!("unexpected outcome: {other:?}"),
        }
    }

    // A block WITH content between the tags must still pass through: the UI
    // renders it as a reasoning section, and scrubbing the bare delimiters used
    // to leave the reasoning as naked text with no marker.
    #[test]
    fn a_channel_block_carrying_content_is_left_alone() {
        let filters: Vec<String> = default_filters(true);
        let mut pending = String::new();
        let piece = "<|channel>thought\nreasoning here<channel|>answer";
        match output::process_piece(&mut pending, &[], &filters, piece) {
            output::PieceOutcome::Emit(out) => {
                assert!(out.contains("reasoning here"), "{out:?}");
            }
            other => panic!("unexpected outcome: {other:?}"),
        }
    }

    #[test]
    fn think_true_leaves_the_tags_alone() {
        let f = default_filters(true);
        assert!(!f.iter().any(|s| s.contains("think")), "{f:?}");
        assert_eq!(f.len(), DEFAULT_HARMONY_FILTERS.len());
    }

    #[test]
    fn think_false_strips_stray_think_tags() {
        let f = default_filters(false);
        assert!(f.iter().any(|s| s == "</think>"));
        assert!(f.iter().any(|s| s == "<think>"));
    }

    // The Harmony filters exist independently of thinking and must survive
    // either way; this is the regression that a naive rewrite would cause.
    #[test]
    fn the_harmony_filters_are_present_in_both_modes() {
        for think in [true, false] {
            let f = default_filters(think);
            for expected in DEFAULT_HARMONY_FILTERS {
                assert!(
                    f.iter().any(|s| s == expected),
                    "{expected} missing with think={think}"
                );
            }
        }
    }
}

#[cfg(test)]
mod stop_reason_tests {
    use super::*;

    #[test]
    fn the_api_strings_are_what_ollama_and_openai_expect() {
        // These land verbatim in `done_reason` and `finish_reason`, and clients
        // branch on them — "length" is how a caller learns to ask for more.
        assert_eq!(StopReason::Stop.as_api_str(), "stop");
        assert_eq!(StopReason::Length.as_api_str(), "length");
    }

    #[test]
    fn a_generate_result_carries_its_reason() {
        // Guards the field being dropped in a refactor: the whole point is that
        // the non-streaming path can distinguish the two, which it could not
        // before because the API hardcoded "stop" in eleven places.
        let r = GenerateResult {
            text: "x".into(),
            tokens_generated: 1,
            tokens_prompt: 1,
            duration_ms: 1,
            stop_reason: StopReason::Length,
        };
        assert_eq!(r.stop_reason.as_api_str(), "length");
    }
}

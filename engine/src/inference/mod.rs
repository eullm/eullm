//! Inference engine powered by llama.cpp via llama-cpp-2 bindings.
//!
//! Supports two modes:
//!
//! - **Sequential** (`InferenceEngine`): one request at a time, simple mutex.
//!   Good for single-user CLI usage.
//! - **Continuous batching** (`BatchScheduler`): multiple concurrent requests
//!   decoded in parallel on a single context. Good for API server / RAG
//!   workloads with many parallel requests.

pub mod scheduler;

#[cfg(feature = "turboquant")]
pub mod turboquant;

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
use llama_cpp_2::sampling::LlamaSampler;
use parking_lot::Mutex;
use tokio::sync::mpsc;

/// Log TurboQuant module status at startup (no-op without the feature flag).
pub fn log_turboquant_status() {
    #[cfg(feature = "turboquant")]
    turboquant::log_turboquant_status();
}

/// Check if the binary was compiled with GPU support and warn if GPU was
/// requested but is not available.  Returns true when a GPU backend is present.
pub fn check_gpu_support(gpu_layers: i32) {
    let has_gpu = cfg!(feature = "cuda")
        || cfg!(feature = "rocm")
        || cfg!(feature = "vulkan")
        || cfg!(feature = "metal");

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
            "No GPU backend compiled (gpu_layers={gpu_layers}). \
             Rebuild with --features cuda/rocm/vulkan/metal for GPU acceleration."
        );
    } else if has_gpu {
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
}

/// Build context params with flash attention, n_batch, and KV cache types applied.
pub(crate) fn build_ctx_params(config: &InferenceConfig, ctx_size: NonZeroU32) -> LlamaContextParams {
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
    let mut params = LlamaContextParams::default()
        .with_n_ctx(Some(ctx_size))
        .with_n_batch(config.n_batch)
        .with_n_threads(config.threads as i32)
        .with_n_threads_batch(config.threads as i32);

    if cache_type_k != KvCacheType::F16 || cache_type_v != KvCacheType::F16 {
        params = params.with_type_k(cache_type_k).with_type_v(cache_type_v);
    }

    if config.flash_attn {
        // AUTO (-1): llama.cpp tests whether the GPU supports FA for the
        // current KV cache types.  If not → disables FA and uses regular
        // attention on GPU.  ENABLED (1) would skip this check and silently
        // route the FA op to the CPU backend on unsupported configs.
        params = params.with_flash_attention_policy(-1);
    }
    params
}

pub use scheduler::{BatchScheduler, ModelReadyInfo, SchedulerConfig, SchedulerHandle};

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
        }
    }
}

/// Parse a KV cache type string (e.g. "q8_0", "q4_0", "f16") into a `KvCacheType`.
///
/// With the `turboquant` feature enabled, also accepts TBQ/TBQP types (tbq3_0, tbq4_0, tbqp3_0, tbqp4_0)
/// and legacy aliases (tq3_0, tq4_0, turbo3, turbo4).
/// If the llama.cpp backend does not support TurboQuant natively, these
/// resolve to F16 with a warning (automatic fallback).
pub fn parse_cache_type(s: &str) -> Result<KvCacheType, String> {
    parse_cache_type_inner(s, false)
}

/// Parse a KV cache type in strict mode — errors instead of falling back.
///
/// Use for benchmarks where silent fallback would produce misleading results.
#[cfg(feature = "turboquant")]
pub fn parse_cache_type_strict(s: &str) -> Result<KvCacheType, String> {
    parse_cache_type_inner(s, true)
}

#[allow(unused_variables)]
fn parse_cache_type_inner(s: &str, strict: bool) -> Result<KvCacheType, String> {
    // Standard types — always available.
    match s.to_lowercase().as_str() {
        "f16" => return Ok(KvCacheType::F16),
        "f32" => return Ok(KvCacheType::F32),
        "q8_0" => return Ok(KvCacheType::Q8_0),
        "q4_0" => return Ok(KvCacheType::Q4_0),
        "q4_1" => return Ok(KvCacheType::Q4_1),
        "q5_0" => return Ok(KvCacheType::Q5_0),
        "q5_1" => return Ok(KvCacheType::Q5_1),
        _ => {}
    }

    // TurboQuant types — experimental, feature-gated.
    #[cfg(feature = "turboquant")]
    {
        use turboquant::config::{resolve_turboquant_cache_type, ResolvedCacheType};
        if let Some(resolved) = resolve_turboquant_cache_type(s, strict) {
            return match resolved {
                ResolvedCacheType::Native(tq) => {
                    // Map TQ type to the GGML type ID from AmesianX/TurboQuant.
                    // The KvCacheType::Unknown variant carries raw GGML type IDs
                    // through to llama.cpp — no Rust enum variant needed.
                    use turboquant::types::ggml_ids;
                    let raw_id = match tq {
                        turboquant::types::TurboquantType::TQ3_0  => ggml_ids::TBQ3_0,
                        turboquant::types::TurboquantType::TQ4_0  => ggml_ids::TBQ4_0,
                        turboquant::types::TurboquantType::TQP3_0 => ggml_ids::TBQP3_0,
                        turboquant::types::TurboquantType::TQP4_0 => ggml_ids::TBQP4_0,
                        turboquant::types::TurboquantType::TQ3_1  => ggml_ids::TBQ3_1,
                        turboquant::types::TurboquantType::TQ4_1  => ggml_ids::TBQ4_1,
                        turboquant::types::TurboquantType::TQP3_1 => ggml_ids::TBQP3_1,
                        turboquant::types::TurboquantType::TQP4_1 => ggml_ids::TBQP4_1,
                        turboquant::types::TurboquantType::TQ3_2  => ggml_ids::TBQ3_2,
                        turboquant::types::TurboquantType::TQ4_2  => ggml_ids::TBQ4_2,
                    };
                    tracing::info!("TurboQuant {tq} → GGML type {raw_id} (native backend)");
                    // Pass the raw GGML type ID via KvCacheType.
                    // The llama-cpp-2 crate maps Unknown(n) to the raw C enum value.
                    Ok(KvCacheType::Unknown(raw_id))
                }
                ResolvedCacheType::Fallback { fallback, .. } => {
                    parse_cache_type(fallback)
                }
                ResolvedCacheType::Unsupported { reason, .. } => {
                    Err(reason)
                }
            };
        }
    }

    #[cfg(feature = "turboquant")]
    let options = "f16, f32, q8_0, q4_0, q4_1, q5_0, q5_1 | \
                   head_dim=128 (Qwen3/Llama/Mistral): tbq3_1, tbq4_1, tbqp3_1, tbqp4_1 | \
                   head_dim=256: tbq3_0, tbq4_0, tbqp3_0, tbqp4_0 | \
                   head_dim=64: tbq3_2, tbq4_2";
    #[cfg(not(feature = "turboquant"))]
    let options = "f16, f32, q8_0, q4_0, q4_1, q5_0, q5_1";

    Err(format!("Unknown cache type '{s}'. Options: {options}"))
}

/// Human-readable name for a KV cache type, including TurboQuant types.
pub fn cache_type_display(ct: &KvCacheType) -> String {
    match ct {
        KvCacheType::F16 => "F16".to_string(),
        KvCacheType::F32 => "F32".to_string(),
        KvCacheType::Q8_0 => "Q8_0".to_string(),
        KvCacheType::Q4_0 => "Q4_0".to_string(),
        KvCacheType::Q4_1 => "Q4_1".to_string(),
        KvCacheType::Q5_0 => "Q5_0".to_string(),
        KvCacheType::Q5_1 => "Q5_1".to_string(),
        KvCacheType::Unknown(41) => "TBQ3_0 (TurboQuant 3-bit MSE, head_dim=256)".to_string(),
        KvCacheType::Unknown(42) => "TBQ4_0 (TurboQuant 4-bit MSE, head_dim=256)".to_string(),
        KvCacheType::Unknown(43) => "TBQP3_0 (TurboQuant 3-bit QJL, head_dim=256)".to_string(),
        KvCacheType::Unknown(44) => "TBQP4_0 (TurboQuant 4-bit QJL, head_dim=256)".to_string(),
        KvCacheType::Unknown(45) => "TBQ3_1 (TurboQuant 3-bit MSE, head_dim=128)".to_string(),
        KvCacheType::Unknown(46) => "TBQ4_1 (TurboQuant 4-bit MSE, head_dim=128)".to_string(),
        KvCacheType::Unknown(47) => "TBQP3_1 (TurboQuant 3-bit DirectSign, head_dim=128)".to_string(),
        KvCacheType::Unknown(48) => "TBQP4_1 (TurboQuant 4-bit DirectSign, head_dim=128)".to_string(),
        KvCacheType::Unknown(49) => "TBQ3_2 (TurboQuant 3-bit MSE, head_dim=64)".to_string(),
        KvCacheType::Unknown(50) => "TBQ4_2 (TurboQuant 4-bit MSE, head_dim=64)".to_string(),
        KvCacheType::Unknown(id) => format!("Unknown({id})"),
        _ => format!("{ct:?}"),
    }
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
    },
    /// An error occurred during generation.
    Error(String),
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
}

// SAFETY: LlamaBackend and LlamaModel are safe to share across threads.
// We guard mutable context access with a Mutex.
unsafe impl Send for InferenceEngine {}
unsafe impl Sync for InferenceEngine {}

impl InferenceEngine {
    /// Load a GGUF model and prepare the inference engine.
    pub fn load(config: InferenceConfig) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        if !config.model_path.exists() {
            return Err(format!(
                "Model file not found: {}",
                config.model_path.display()
            )
            .into());
        }

        check_gpu_support(config.gpu_layers);

        tracing::info!("Initializing llama.cpp backend...");
        let backend = LlamaBackend::init()?;

        let model_params = if config.gpu_layers >= 0 {
            LlamaModelParams::default().with_n_gpu_layers(config.gpu_layers as u32)
        } else {
            // -1 = offload all layers
            LlamaModelParams::default().with_n_gpu_layers(1000)
        };
        let model_params = pin!(model_params);

        tracing::info!("Loading model: {}", config.model_path.display());
        let model = LlamaModel::load_from_file(&backend, &config.model_path, &model_params)
            .map_err(|e| format!("Failed to load model: {e}"))?;

        tracing::info!("Model loaded successfully.");

        Ok(Self {
            backend,
            model,
            config,
            ctx_mutex: Mutex::new(()),
        })
    }

    /// Generate text from a prompt (blocking, returns all at once).
    pub fn generate(
        &self,
        request: &GenerateRequest,
    ) -> Result<GenerateResult, Box<dyn std::error::Error + Send + Sync>> {
        let _lock = self.ctx_mutex.lock();
        let start = std::time::Instant::now();

        let ctx_size = NonZeroU32::new(self.config.context_size)
            .unwrap_or(NonZeroU32::new(4096).unwrap());

        let has_quantized_cache = self.config.cache_type_k != KvCacheType::F16
            || self.config.cache_type_v != KvCacheType::F16;
        let ctx_params = build_ctx_params(&self.config, ctx_size);

        let mut ctx = match self.model.new_context(&self.backend, ctx_params) {
            Ok(c) => c,
            Err(e) if has_quantized_cache => {
                tracing::warn!("Context creation failed with quantized KV cache, falling back to F16");
                let fallback = build_ctx_params_with_cache(&self.config, ctx_size, KvCacheType::F16, KvCacheType::F16);
                self.model.new_context(&self.backend, fallback)
                    .map_err(|e2| format!("Failed to create context (F16 fallback failed too): {e2}\nOriginal error: {e}"))?
            }
            Err(e) => return Err(format!("Failed to create context: {e}").into()),
        };

        // Tokenize the prompt
        let tokens = self
            .model
            .str_to_token(&request.prompt, if request.raw { AddBos::Never } else { AddBos::Always })
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
        let mut sampler = {
            let seed = request.seed.unwrap_or(1234);
            let mut chain: Vec<LlamaSampler> = Vec::new();
            if let Some(ref grammar_str) = request.grammar {
                match LlamaSampler::grammar(&self.model, grammar_str, "root") {
                    Ok(gs) => chain.push(gs),
                    Err(e) => tracing::warn!("Grammar sampler init failed ({e:?}), falling back to unconstrained"),
                }
            }
            if request.repeat_penalty != 1.0 {
                chain.push(LlamaSampler::penalties(request.repeat_last_n, request.repeat_penalty, 0.0, 0.0));
            }
            if request.top_k > 0 {
                chain.push(LlamaSampler::top_k(request.top_k));
            }
            if request.top_p < 1.0 {
                chain.push(LlamaSampler::top_p(request.top_p, 1));
            }
            if request.min_p > 0.0 {
                chain.push(LlamaSampler::min_p(request.min_p, 1));
            }
            chain.push(LlamaSampler::temp(request.temperature));
            chain.push(LlamaSampler::dist(seed));
            LlamaSampler::chain_simple(chain)
        };

        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut output = String::new();
        let mut n_cur = tokens_prompt as i32;
        let mut tokens_generated: u32 = 0;
        let mut batch = LlamaBatch::new(1, 1);

        while n_cur <= n_len && tokens_generated < max_tokens {
            // Sample from the last output (-1). After prompt decode there is
            // exactly one output (the final prompt token with logits=true);
            // after single-token decode steps there is also exactly one output.
            let token = sampler.sample(&ctx, -1);
            sampler.accept(token);

            // End of generation?
            if self.model.is_eog_token(token) {
                break;
            }

            // Decode token to text
            match self.model.token_to_piece(token, &mut decoder, true, None) {
                Ok(piece) => {
                    output.push_str(&piece);

                    // Check stop sequences
                    let should_stop = request
                        .stop_sequences
                        .iter()
                        .any(|s| output.ends_with(s));
                    if should_stop {
                        // Remove the stop sequence from output
                        for s in &request.stop_sequences {
                            if output.ends_with(s) {
                                let new_len = output.len() - s.len();
                                output.truncate(new_len);
                                break;
                            }
                        }
                        break;
                    }
                }
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

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(GenerateResult {
            text: output,
            tokens_generated,
            tokens_prompt,
            duration_ms,
        })
    }

    /// Generate text with streaming — sends each token through the channel as it's produced.
    ///
    /// The caller receives `StreamEvent::Token(piece)` for each generated piece,
    /// then `StreamEvent::Done { ... }` when generation is complete.
    pub fn generate_streaming(
        &self,
        request: &GenerateRequest,
        tx: mpsc::Sender<StreamEvent>,
    ) {
        let _lock = self.ctx_mutex.lock();
        let start = std::time::Instant::now();

        let ctx_size = NonZeroU32::new(self.config.context_size)
            .unwrap_or(NonZeroU32::new(4096).unwrap());

        let has_quantized_cache = self.config.cache_type_k != KvCacheType::F16
            || self.config.cache_type_v != KvCacheType::F16;
        let ctx_params = build_ctx_params(&self.config, ctx_size);

        let mut ctx = match self.model.new_context(&self.backend, ctx_params) {
            Ok(c) => c,
            Err(e) if has_quantized_cache => {
                tracing::warn!("Context creation failed with quantized KV cache, falling back to F16");
                let fallback = build_ctx_params_with_cache(&self.config, ctx_size, KvCacheType::F16, KvCacheType::F16);
                match self.model.new_context(&self.backend, fallback) {
                    Ok(c) => c,
                    Err(e2) => {
                        let _ = tx.blocking_send(StreamEvent::Error(format!("F16 fallback failed: {e2} (original: {e})")));
                        return;
                    }
                }
            }
            Err(e) => {
                let _ = tx.blocking_send(StreamEvent::Error(format!("Failed to create context: {e}")));
                return;
            }
        };

        let tokens = match self.model.str_to_token(&request.prompt, if request.raw { AddBos::Never } else { AddBos::Always }) {
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
                    if prefill_batch.add(*token, abs_pos as i32, &[0], is_last).is_err() {
                        let _ = tx.blocking_send(StreamEvent::Error("Failed to build batch".into()));
                        return;
                    }
                }

                if ctx.decode(&mut prefill_batch).is_err() {
                    let _ = tx.blocking_send(StreamEvent::Error("Prompt decode failed".into()));
                    return;
                }
            }
        }

        let mut sampler = {
            let seed = request.seed.unwrap_or(1234);
            let mut chain: Vec<LlamaSampler> = Vec::new();
            if let Some(ref grammar_str) = request.grammar {
                match LlamaSampler::grammar(&self.model, grammar_str, "root") {
                    Ok(gs) => chain.push(gs),
                    Err(e) => tracing::warn!("Grammar sampler init failed ({e:?}), falling back to unconstrained"),
                }
            }
            if request.repeat_penalty != 1.0 {
                chain.push(LlamaSampler::penalties(request.repeat_last_n, request.repeat_penalty, 0.0, 0.0));
            }
            if request.top_k > 0 {
                chain.push(LlamaSampler::top_k(request.top_k));
            }
            if request.top_p < 1.0 {
                chain.push(LlamaSampler::top_p(request.top_p, 1));
            }
            if request.min_p > 0.0 {
                chain.push(LlamaSampler::min_p(request.min_p, 1));
            }
            chain.push(LlamaSampler::temp(request.temperature));
            chain.push(LlamaSampler::dist(seed));
            LlamaSampler::chain_simple(chain)
        };

        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut full_output = String::new();
        let mut n_cur = tokens_prompt as i32;
        let mut tokens_generated: u32 = 0;
        let mut batch = LlamaBatch::new(1, 1);

        while n_cur <= n_len && tokens_generated < max_tokens {
            let token = sampler.sample(&ctx, -1);
            sampler.accept(token);

            if self.model.is_eog_token(token) {
                break;
            }

            match self.model.token_to_piece(token, &mut decoder, true, None) {
                Ok(piece) => {
                    full_output.push_str(&piece);

                    // Check stop sequences
                    let mut stopped = false;
                    for s in &request.stop_sequences {
                        if full_output.ends_with(s) {
                            // Don't send the stop sequence token
                            let piece_without_stop = if piece.len() >= s.len() {
                                &piece[..piece.len() - s.len()]
                            } else {
                                ""
                            };
                            if !piece_without_stop.is_empty()
                                && tx.blocking_send(StreamEvent::Token(piece_without_stop.to_string())).is_err()
                            {
                                return;
                            }
                            stopped = true;
                            break;
                        }
                    }

                    if stopped {
                        break;
                    }

                    // Send the token piece
                    if tx.blocking_send(StreamEvent::Token(piece)).is_err() {
                        return; // receiver dropped
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

        let duration_ms = start.elapsed().as_millis() as u64;
        let _ = tx.blocking_send(StreamEvent::Done {
            tokens_generated,
            tokens_prompt,
            duration_ms,
        });
    }
}

fn num_cpus() -> u32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(4)
}

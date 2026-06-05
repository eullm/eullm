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
    /// Optional path to a multimodal projector (mmproj) GGUF file. When
    /// present AND the `multimodal` cargo feature is enabled, the engine
    /// loads an `MtmdContext` alongside the text model so it can accept
    /// image / audio input. When `None` or the feature is off, the engine
    /// runs text-only exactly as before.
    pub mmproj_path: Option<PathBuf>,
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
    // the canonical GPT-OSS Harmony channels.
    "<|channel>thought<channel|>",
    "<|channel>analysis<channel|>",
    "<|channel>final<channel|>",
    "<|message|>",
    // Single delimiters — fallback for Harmony variants we have not enumerated
    // and for stray `<|image|>`/`<|audio|>` markers mid-sentence. Will leave
    // the body word visible when used alone, but at least scrubs the syntax.
    "<|channel>",
    "<channel|>",
    "<|image>",
    "<image|>",
    "<|audio>",
    "<audio|>",
];

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
            filter_sequences: DEFAULT_HARMONY_FILTERS.iter().map(|s| (*s).to_string()).collect(),
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
        let mut backend = LlamaBackend::init()?;
        // Suppress llama.cpp/ggml internal log messages (CUDA graph warmup etc.)
        // so they don't pollute the interactive REPL or API response streams.
        // EULLM uses tracing for its own structured logging.
        backend.void_logs();

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

        #[cfg(feature = "multimodal")]
        let mtmd_ctx = Self::init_mtmd_optional(&config, &model)?;

        Ok(Self {
            backend,
            model,
            config,
            ctx_mutex: Mutex::new(()),
            #[cfg(feature = "multimodal")]
            mtmd_ctx,
        })
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
        let mut params = MtmdContextParams::default();
        params.use_gpu = config.gpu_layers != 0;
        params.print_timings = false;
        params.n_threads = config.threads as i32;

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
            b.starts_with(b"RIFF") || b.starts_with(b"ID3") || b.starts_with(b"fLaC")
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
        let ctx_size = NonZeroU32::new(self.config.context_size)
            .unwrap_or(NonZeroU32::new(4096).unwrap());
        let has_quantized_cache = self.config.cache_type_k != KvCacheType::F16
            || self.config.cache_type_v != KvCacheType::F16;
        let ctx_params = build_ctx_params(&self.config, ctx_size);
        let mut ctx = match self.model.new_context(&self.backend, ctx_params) {
            Ok(c) => c,
            Err(e) if has_quantized_cache => {
                let fallback = build_ctx_params_with_cache(
                    &self.config, ctx_size, KvCacheType::F16, KvCacheType::F16,
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
                let _ = tx.blocking_send(StreamEvent::Error(format!(
                    "Failed to create context: {e}"
                )));
                return;
            }
        };

        // ── 3. Decode media bytes into mtmd bitmaps ─────────────────────
        let mut bitmaps: Vec<MtmdBitmap> = Vec::with_capacity(media.len());
        for (i, bytes) in media.iter().enumerate() {
            match MtmdBitmap::from_buffer(mtmd_ctx, bytes) {
                Ok(b) => bitmaps.push(b),
                Err(e) => {
                    let _ = tx.blocking_send(StreamEvent::Error(format!(
                        "Media #{i} failed to decode: {e:?}"
                    )));
                    return;
                }
            }
        }
        let bitmap_refs: Vec<&MtmdBitmap> = bitmaps.iter().collect();

        // ── 4. Tokenize text + media → MtmdInputChunks ──────────────────
        // The prompt is already templated by the caller and contains one
        // <__media__> marker per bitmap. add_special=!raw mirrors the text
        // path; parse_special=true must be on so the chat-template special
        // tokens (e.g. Gemma <start_of_turn>) are recognised.
        let input_text = MtmdInputText {
            text: request.prompt.clone(),
            add_special: !request.raw,
            parse_special: true,
        };
        let chunks = match mtmd_ctx.tokenize(input_text, &bitmap_refs) {
            Ok(c) => c,
            Err(e) => {
                let _ = tx.blocking_send(StreamEvent::Error(format!(
                    "mtmd tokenize failed: {e:?}"
                )));
                return;
            }
        };

        let tokens_prompt = chunks.total_tokens() as u32;
        let server_ctx = ctx.n_ctx();
        let effective_ctx = request.num_ctx.map(|n| n.min(server_ctx)).unwrap_or(server_ctx);
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
        let mut sampler = {
            let seed = request.seed.unwrap_or(1234);
            let mut chain: Vec<LlamaSampler> = Vec::new();
            if let Some(ref grammar_str) = request.grammar {
                if let Ok(gs) = LlamaSampler::grammar(&self.model, grammar_str, "root") {
                    chain.push(gs);
                }
            }
            if request.repeat_penalty != 1.0 {
                chain.push(LlamaSampler::penalties(
                    request.repeat_last_n, request.repeat_penalty, 0.0, 0.0,
                ));
            }
            if request.top_k > 0 { chain.push(LlamaSampler::top_k(request.top_k)); }
            if request.top_p < 1.0 { chain.push(LlamaSampler::top_p(request.top_p, 1)); }
            if request.min_p > 0.0 { chain.push(LlamaSampler::min_p(request.min_p, 1)); }
            chain.push(LlamaSampler::temp(request.temperature));
            chain.push(LlamaSampler::dist(seed));
            LlamaSampler::chain_simple(chain)
        };

        // ── 7. Decode loop (identical pattern to generate_streaming) ────
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut full_output = String::new();
        let mut n_cur = new_n_past;
        let mut tokens_generated: u32 = 0;
        let mut batch = LlamaBatch::new(1, 1);

        while n_cur <= n_len && tokens_generated < max_tokens {
            let token = sampler.sample(&ctx, -1);
            sampler.accept(token);
            if self.model.is_eog_token(token) { break; }

            match self.model.token_to_piece(token, &mut decoder, true, None) {
                Ok(piece) => {
                    full_output.push_str(&piece);
                    let mut stopped = false;
                    for s in &request.stop_sequences {
                        if full_output.ends_with(s) {
                            let trimmed = if piece.len() >= s.len() {
                                &piece[..piece.len() - s.len()]
                            } else { "" };
                            if !trimmed.is_empty()
                                && tx.blocking_send(StreamEvent::Token(trimmed.to_string())).is_err()
                            {
                                return;
                            }
                            stopped = true;
                            break;
                        }
                    }
                    if stopped { break; }
                    if tx.blocking_send(StreamEvent::Token(piece)).is_err() {
                        return;
                    }
                }
                Err(_) => break,
            }

            batch.clear();
            if batch.add(token, n_cur, &[0], true).is_err() { break; }
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

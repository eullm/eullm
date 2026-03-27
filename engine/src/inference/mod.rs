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
///
/// KV cache type defaults to F16 (maximum GPU compatibility).  Quantized
/// types (Q8_0, Q4_0) save VRAM but can cause GPU fallback on some
/// architectures — only use them after verifying GPU utilisation with nvtop.
pub(crate) fn build_ctx_params(config: &InferenceConfig, ctx_size: NonZeroU32) -> LlamaContextParams {
    let mut params = LlamaContextParams::default()
        .with_n_ctx(Some(ctx_size))
        .with_n_batch(config.n_batch)
        .with_n_threads(config.threads as i32)
        .with_n_threads_batch(config.threads as i32);

    // Only set KV cache types when they differ from the default (F16).
    // Setting quantized types (Q8_0, Q4_0) can trigger GPU→CPU fallback
    // for batch prefill on some GPU architectures (observed on RTX 5070 Ti
    // with llama-cpp-2 0.1.140).  When F16 is requested we skip the call
    // entirely so llama.cpp uses its own default, which is known to work.
    if config.cache_type_k != KvCacheType::F16 || config.cache_type_v != KvCacheType::F16 {
        params = params.with_type_k(config.cache_type_k).with_type_v(config.cache_type_v);
        tracing::warn!(
            "Using quantized KV cache (K={:?}, V={:?}). If GPU utilisation is low, \
             try --cache-type-k f16 --cache-type-v f16",
            config.cache_type_k,
            config.cache_type_v,
        );
    }

    if config.flash_attn {
        params = params.with_flash_attention_policy(1);
    }
    params
}

pub use scheduler::{BatchScheduler, SchedulerConfig, SchedulerHandle};

/// Configuration for the inference engine.
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Path to the GGUF model file.
    pub model_path: PathBuf,
    /// Number of GPU layers to offload (-1 = all).
    pub gpu_layers: i32,
    /// Context window size (per sequence).
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
pub fn parse_cache_type(s: &str) -> Result<KvCacheType, String> {
    match s.to_lowercase().as_str() {
        "f16" => Ok(KvCacheType::F16),
        "f32" => Ok(KvCacheType::F32),
        "q8_0" => Ok(KvCacheType::Q8_0),
        "q4_0" => Ok(KvCacheType::Q4_0),
        "q4_1" => Ok(KvCacheType::Q4_1),
        "q5_0" => Ok(KvCacheType::Q5_0),
        "q5_1" => Ok(KvCacheType::Q5_1),
        _ => Err(format!("Unknown cache type '{s}'. Options: f16, f32, q8_0, q4_0, q4_1, q5_0, q5_1")),
    }
}


/// Request for text generation.
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub stop_sequences: Vec<String>,
    /// Per-request context budget (Ollama `num_ctx`).  When `Some`, the
    /// validation uses this instead of the server-level `context_size`.
    /// Must be ≤ server `context_size` (clamped at prefill time).
    pub num_ctx: Option<u32>,
    /// Optional GBNF grammar string for constrained decoding.
    /// When set, the sampler enforces that output conforms to this grammar.
    /// Used by `format: "json"` to guarantee valid JSON output.
    pub grammar: Option<String>,
}

impl Default for GenerateRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            max_tokens: 512,
            temperature: 0.7,
            stop_sequences: Vec::new(),
            num_ctx: None,
            grammar: None,
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

        let ctx_params = build_ctx_params(&self.config, ctx_size);

        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| format!("Failed to create context: {e}"))?;

        // Tokenize the prompt
        let tokens = self
            .model
            .str_to_token(&request.prompt, AddBos::Always)
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
        let mut sampler = if let Some(ref grammar_str) = request.grammar {
            match LlamaSampler::grammar(&self.model, grammar_str, "root") {
                Ok(grammar_sampler) => LlamaSampler::chain_simple([
                    grammar_sampler,
                    LlamaSampler::temp(request.temperature),
                    LlamaSampler::dist(1234),
                    LlamaSampler::greedy(),
                ]),
                Err(e) => {
                    tracing::warn!("Grammar sampler init failed ({e:?}), falling back to unconstrained");
                    LlamaSampler::chain_simple([
                        LlamaSampler::temp(request.temperature),
                        LlamaSampler::dist(1234),
                        LlamaSampler::greedy(),
                    ])
                }
            }
        } else {
            LlamaSampler::chain_simple([
                LlamaSampler::temp(request.temperature),
                LlamaSampler::dist(1234),
                LlamaSampler::greedy(),
            ])
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

        let ctx_params = build_ctx_params(&self.config, ctx_size);

        let mut ctx = match self.model.new_context(&self.backend, ctx_params) {
            Ok(c) => c,
            Err(e) => {
                let _ = tx.blocking_send(StreamEvent::Error(format!("Failed to create context: {e}")));
                return;
            }
        };

        let tokens = match self.model.str_to_token(&request.prompt, AddBos::Always) {
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

        let mut sampler = if let Some(ref grammar_str) = request.grammar {
            match LlamaSampler::grammar(&self.model, grammar_str, "root") {
                Ok(grammar_sampler) => LlamaSampler::chain_simple([
                    grammar_sampler,
                    LlamaSampler::temp(request.temperature),
                    LlamaSampler::dist(1234),
                    LlamaSampler::greedy(),
                ]),
                Err(e) => {
                    tracing::warn!("Grammar sampler init failed ({e:?}), falling back to unconstrained");
                    LlamaSampler::chain_simple([
                        LlamaSampler::temp(request.temperature),
                        LlamaSampler::dist(1234),
                        LlamaSampler::greedy(),
                    ])
                }
            }
        } else {
            LlamaSampler::chain_simple([
                LlamaSampler::temp(request.temperature),
                LlamaSampler::dist(1234),
                LlamaSampler::greedy(),
            ])
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

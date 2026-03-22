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

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use parking_lot::Mutex;
use tokio::sync::mpsc;

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
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            gpu_layers: -1,
            context_size: 4096,
            threads: num_cpus(),
        }
    }
}

/// Request for text generation.
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub stop_sequences: Vec<String>,
}

impl Default for GenerateRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            max_tokens: 512,
            temperature: 0.7,
            stop_sequences: Vec::new(),
        }
    }
}

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

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(ctx_size))
            .with_n_threads(self.config.threads as i32)
            .with_n_threads_batch(self.config.threads as i32);

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
        let n_len = (tokens.len() as u32 + request.max_tokens) as i32;

        // Check context fits
        let n_ctx = ctx.n_ctx() as i32;
        if n_len > n_ctx {
            return Err(format!(
                "Prompt ({tokens_prompt} tokens) + max_tokens ({}) exceeds context window ({n_ctx})",
                request.max_tokens
            )
            .into());
        }

        // Build initial batch with prompt tokens
        let mut batch = LlamaBatch::new(self.config.context_size as usize, 1);
        let last_idx = (tokens.len() - 1) as i32;
        for (i, token) in (0_i32..).zip(tokens.into_iter()) {
            let is_last = i == last_idx;
            batch.add(token, i, &[0], is_last)?;
        }

        // Decode prompt
        ctx.decode(&mut batch)
            .map_err(|e| format!("Prompt decode failed: {e}"))?;

        // Sample tokens
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(request.temperature),
            LlamaSampler::dist(1234),
            LlamaSampler::greedy(),
        ]);

        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut output = String::new();
        let mut n_cur = batch.n_tokens();
        let mut tokens_generated: u32 = 0;

        while n_cur <= n_len && tokens_generated < request.max_tokens {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
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

            // Prepare next batch
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

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(ctx_size))
            .with_n_threads(self.config.threads as i32)
            .with_n_threads_batch(self.config.threads as i32);

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
        let n_len = (tokens.len() as u32 + request.max_tokens) as i32;

        let n_ctx = ctx.n_ctx() as i32;
        if n_len > n_ctx {
            let _ = tx.blocking_send(StreamEvent::Error(format!(
                "Prompt ({tokens_prompt} tokens) + max_tokens ({}) exceeds context window ({n_ctx})",
                request.max_tokens
            )));
            return;
        }

        let mut batch = LlamaBatch::new(self.config.context_size as usize, 1);
        let last_idx = (tokens.len() - 1) as i32;
        for (i, token) in (0_i32..).zip(tokens.into_iter()) {
            if batch.add(token, i, &[0], i == last_idx).is_err() {
                let _ = tx.blocking_send(StreamEvent::Error("Failed to build batch".into()));
                return;
            }
        }

        if ctx.decode(&mut batch).is_err() {
            let _ = tx.blocking_send(StreamEvent::Error("Prompt decode failed".into()));
            return;
        }

        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(request.temperature),
            LlamaSampler::dist(1234),
            LlamaSampler::greedy(),
        ]);

        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut full_output = String::new();
        let mut n_cur = batch.n_tokens();
        let mut tokens_generated: u32 = 0;

        while n_cur <= n_len && tokens_generated < request.max_tokens {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
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

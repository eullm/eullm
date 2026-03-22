//! Continuous batching scheduler for multi-request inference.
//!
//! Instead of processing one request at a time behind a mutex, the scheduler
//! maintains a queue of pending requests and a set of active sequences. On each
//! iteration of its decode loop it:
//!
//! 1. Drains newly submitted requests from the queue.
//! 2. Prefills their prompts (adds tokens to `LlamaBatch` with unique `seq_id`).
//! 3. Decodes one token for every active sequence in a single `ctx.decode()` call.
//! 4. Sends each generated token to the corresponding response channel.
//! 5. Removes completed sequences (EOS, max tokens, or dropped receiver).
//!
//! This gives near-linear throughput scaling with concurrent requests (up to
//! `max_batch_size`) while keeping latency stable.

use std::num::NonZeroU32;
use std::pin::pin;
use std::sync::Arc;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use tokio::sync::mpsc;

use super::{GenerateRequest, InferenceConfig, StreamEvent};

/// Configuration for the batch scheduler.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Maximum number of sequences decoded in parallel.
    pub max_batch_size: usize,
    /// How many requests can wait in the submission queue before back-pressure.
    pub queue_capacity: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            queue_capacity: 64,
        }
    }
}

/// A request submitted to the scheduler together with its response channel.
pub(crate) struct ScheduledRequest {
    pub request: GenerateRequest,
    pub tx: mpsc::Sender<StreamEvent>,
}

/// State for a single active sequence being decoded.
struct ActiveSequence {
    seq_id: i32,
    tx: mpsc::Sender<StreamEvent>,
    sampler: LlamaSampler,
    decoder: encoding_rs::Decoder,
    full_output: String,
    tokens_prompt: u32,
    tokens_generated: u32,
    max_tokens: u32,
    n_past: i32,
    stop_sequences: Vec<String>,
    start: std::time::Instant,
    /// Set to true once the initial prompt has been prefilled.
    prefilled: bool,
    /// The last token sampled (used to feed the next decode step).
    last_token: Option<LlamaToken>,
}

/// Handle returned to callers for submitting requests.
#[derive(Clone)]
pub struct SchedulerHandle {
    tx: crossbeam_channel::Sender<ScheduledRequest>,
    /// Notify the scheduler thread that new work is available.
    notify: Arc<std::sync::Condvar>,
    notify_mutex: Arc<std::sync::Mutex<()>>,
}

impl SchedulerHandle {
    /// Submit a request for inference. Returns immediately.
    ///
    /// The caller should listen on the returned `mpsc::Receiver<StreamEvent>`
    /// for token events.
    pub fn submit(&self, request: GenerateRequest) -> mpsc::Receiver<StreamEvent> {
        let (tx, rx) = mpsc::channel::<StreamEvent>(32);

        // Best-effort send — if the queue is full the request is rejected.
        if self.tx.try_send(ScheduledRequest { request, tx: tx.clone() }).is_err() {
            let _ = tx.try_send(StreamEvent::Error(
                "Scheduler queue full — try again later".into(),
            ));
        } else {
            // Wake up the scheduler thread.
            let _lock = self.notify_mutex.lock().unwrap();
            self.notify.notify_one();
        }

        rx
    }
}

/// The batch scheduler. Owns the llama.cpp backend, model, and context.
pub struct BatchScheduler {
    config: InferenceConfig,
    sched_config: SchedulerConfig,
}

impl BatchScheduler {
    pub fn new(config: InferenceConfig, sched_config: SchedulerConfig) -> Self {
        Self {
            config,
            sched_config,
        }
    }

    /// Start the scheduler. Returns a handle for submitting requests.
    ///
    /// This spawns a dedicated OS thread that runs the decode loop. The thread
    /// lives until the returned `SchedulerHandle` (and all its clones) are
    /// dropped.
    pub fn start(self) -> Result<SchedulerHandle, Box<dyn std::error::Error + Send + Sync>> {
        // Validate model exists before spawning the thread.
        if !self.config.model_path.exists() {
            return Err(format!(
                "Model file not found: {}",
                self.config.model_path.display()
            )
            .into());
        }

        let (req_tx, req_rx) =
            crossbeam_channel::bounded::<ScheduledRequest>(self.sched_config.queue_capacity);

        let notify = Arc::new(std::sync::Condvar::new());
        let notify_mutex = Arc::new(std::sync::Mutex::new(()));
        let notify_clone = Arc::clone(&notify);
        let notify_mutex_clone = Arc::clone(&notify_mutex);

        let config = self.config.clone();
        let sched_config = self.sched_config.clone();

        std::thread::Builder::new()
            .name("eullm-scheduler".into())
            .spawn(move || {
                if let Err(e) = run_scheduler_loop(config, sched_config, req_rx, notify_clone, notify_mutex_clone) {
                    tracing::error!("Scheduler thread exited with error: {e}");
                }
            })?;

        Ok(SchedulerHandle {
            tx: req_tx,
            notify,
            notify_mutex,
        })
    }
}

// ── Scheduler loop (runs on a dedicated thread) ─────────────────────────────

fn run_scheduler_loop(
    config: InferenceConfig,
    sched_config: SchedulerConfig,
    req_rx: crossbeam_channel::Receiver<ScheduledRequest>,
    notify: Arc<std::sync::Condvar>,
    notify_mutex: Arc<std::sync::Mutex<()>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing::info!("Initializing llama.cpp backend (scheduler)...");
    let backend = LlamaBackend::init()?;

    let model_params = if config.gpu_layers >= 0 {
        LlamaModelParams::default().with_n_gpu_layers(config.gpu_layers as u32)
    } else {
        LlamaModelParams::default().with_n_gpu_layers(1000)
    };
    let model_params = pin!(model_params);

    tracing::info!("Loading model: {}", config.model_path.display());
    let model = LlamaModel::load_from_file(&backend, &config.model_path, &model_params)
        .map_err(|e| format!("Failed to load model: {e}"))?;
    tracing::info!("Model loaded (scheduler ready).");

    // Create a single shared context with enough room for all sequences.
    let total_ctx = config.context_size * sched_config.max_batch_size as u32;
    let ctx_size = NonZeroU32::new(total_ctx).unwrap_or(NonZeroU32::new(4096).unwrap());

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(ctx_size))
        .with_n_threads(config.threads as i32)
        .with_n_threads_batch(config.threads as i32);

    let mut ctx = model
        .new_context(&backend, ctx_params)
        .map_err(|e| format!("Failed to create context: {e}"))?;

    let mut active: Vec<ActiveSequence> = Vec::with_capacity(sched_config.max_batch_size);
    let mut next_seq_id: i32 = 0;

    tracing::info!(
        "Scheduler running — max_batch_size={}, queue_capacity={}, context={}",
        sched_config.max_batch_size,
        sched_config.queue_capacity,
        total_ctx,
    );

    loop {
        // ── 1. Drain new requests from the queue ────────────────────────
        while active.len() < sched_config.max_batch_size {
            match req_rx.try_recv() {
                Ok(scheduled) => {
                    let seq_id = next_seq_id;
                    next_seq_id = next_seq_id.wrapping_add(1);

                    let seq = ActiveSequence {
                        seq_id,
                        tx: scheduled.tx,
                        sampler: LlamaSampler::chain_simple([
                            LlamaSampler::temp(scheduled.request.temperature),
                            LlamaSampler::dist(seq_id as u32),
                            LlamaSampler::greedy(),
                        ]),
                        decoder: encoding_rs::UTF_8.new_decoder(),
                        full_output: String::new(),
                        tokens_prompt: 0,
                        tokens_generated: 0,
                        max_tokens: scheduled.request.max_tokens,
                        n_past: 0,
                        stop_sequences: scheduled.request.stop_sequences.clone(),
                        start: std::time::Instant::now(),
                        prefilled: false,
                        last_token: None,
                    };

                    // Tokenize and prefill immediately.
                    match prefill_sequence(&model, &mut ctx, &config, &scheduled.request, &seq) {
                        Ok((n_tokens, n_past)) => {
                            let mut seq = seq;
                            seq.tokens_prompt = n_tokens;
                            seq.n_past = n_past;
                            seq.prefilled = true;
                            active.push(seq);
                            tracing::debug!("Sequence {seq_id} prefilled ({n_tokens} prompt tokens)");
                        }
                        Err(e) => {
                            let _ = seq.tx.blocking_send(StreamEvent::Error(format!(
                                "Prefill failed: {e}"
                            )));
                        }
                    }
                }
                Err(crossbeam_channel::TryRecvError::Empty) => break,
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    tracing::info!("Request channel closed — scheduler shutting down.");
                    // Finish active sequences gracefully.
                    for seq in &active {
                        let _ = seq.tx.blocking_send(StreamEvent::Error(
                            "Server shutting down".into(),
                        ));
                    }
                    return Ok(());
                }
            }
        }

        // ── 2. If nothing active, wait for new work ─────────────────────
        if active.is_empty() {
            let lock = notify_mutex.lock().unwrap();
            // Wait with a timeout so we can check for channel disconnect.
            let _ = notify.wait_timeout(lock, std::time::Duration::from_millis(100));
            continue;
        }

        // ── 3. Build batch with one token per active sequence ───────────
        let mut batch = LlamaBatch::new(active.len().max(1), 1);

        for seq in active.iter() {
            if let Some(token) = seq.last_token {
                if batch.add(token, seq.n_past, &[seq.seq_id], true).is_err() {
                    tracing::warn!("Failed to add token to batch for seq {}", seq.seq_id);
                }
            }
        }

        // ── 4. Decode the batch ─────────────────────────────────────────
        if batch.n_tokens() > 0 {
            if let Err(e) = ctx.decode(&mut batch) {
                tracing::error!("Batch decode failed: {e}");
                // Send errors to all active sequences and clear.
                for seq in active.drain(..) {
                    let _ = seq.tx.blocking_send(StreamEvent::Error(format!(
                        "Decode failed: {e}"
                    )));
                }
                continue;
            }
        }

        // ── 5. Sample one token per sequence, send events ───────────────
        let mut to_remove: Vec<usize> = Vec::new();
        let mut logit_idx: i32 = 0;

        for (i, seq) in active.iter_mut().enumerate() {
            if seq.last_token.is_none() {
                // First decode after prefill — logit is from the prefill batch.
                // We need to sample from the last position set during prefill.
                continue;
            }

            let token = seq.sampler.sample(&ctx, logit_idx);
            seq.sampler.accept(token);
            logit_idx += 1;

            // End of generation?
            if model.is_eog_token(token) {
                send_done(seq);
                to_remove.push(i);
                let _ = ctx.clear_kv_cache_seq(Some(seq.seq_id as u32), None, None);
                continue;
            }

            seq.tokens_generated += 1;

            // Decode token to text.
            match model.token_to_piece(token, &mut seq.decoder, true, None) {
                Ok(piece) => {
                    seq.full_output.push_str(&piece);

                    // Check stop sequences.
                    let mut stopped = false;
                    for s in &seq.stop_sequences {
                        if seq.full_output.ends_with(s) {
                            let trimmed = if piece.len() >= s.len() {
                                &piece[..piece.len() - s.len()]
                            } else {
                                ""
                            };
                            if !trimmed.is_empty()
                                && seq.tx.blocking_send(StreamEvent::Token(trimmed.to_string())).is_err()
                            {
                                to_remove.push(i);
                                let _ = ctx.clear_kv_cache_seq(Some(seq.seq_id as u32), None, None);
                                stopped = true;
                                break;
                            }
                            send_done(seq);
                            to_remove.push(i);
                            let _ = ctx.clear_kv_cache_seq(Some(seq.seq_id as u32), None, None);
                            stopped = true;
                            break;
                        }
                    }

                    if stopped {
                        continue;
                    }

                    // Send the token piece.
                    if seq.tx.blocking_send(StreamEvent::Token(piece)).is_err() {
                        // Receiver dropped — client disconnected.
                        to_remove.push(i);
                        let _ = ctx.clear_kv_cache_seq(Some(seq.seq_id as u32), None, None);
                        continue;
                    }
                }
                Err(_) => {
                    send_done(seq);
                    to_remove.push(i);
                    let _ = ctx.clear_kv_cache_seq(Some(seq.seq_id as u32), None, None);
                    continue;
                }
            }

            // Check max tokens.
            if seq.tokens_generated >= seq.max_tokens {
                send_done(seq);
                to_remove.push(i);
                let _ = ctx.clear_kv_cache_seq(Some(seq.seq_id as u32), None, None);
                continue;
            }

            seq.last_token = Some(token);
            seq.n_past += 1;
        }

        // ── 6. Remove completed sequences (reverse order) ──────────────
        to_remove.sort_unstable();
        to_remove.dedup();
        for &i in to_remove.iter().rev() {
            active.swap_remove(i);
        }
    }
}

/// Prefill a sequence's prompt tokens into the context.
///
/// Returns `(prompt_token_count, n_past)` on success.
fn prefill_sequence(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    config: &InferenceConfig,
    request: &GenerateRequest,
    seq: &ActiveSequence,
) -> Result<(u32, i32), String> {
    let tokens = model
        .str_to_token(&request.prompt, AddBos::Always)
        .map_err(|e| format!("Tokenization failed: {e}"))?;

    let n_tokens = tokens.len() as u32;
    let total_needed = n_tokens + request.max_tokens;

    if total_needed > config.context_size {
        return Err(format!(
            "Prompt ({n_tokens} tokens) + max_tokens ({}) exceeds per-sequence context ({})",
            request.max_tokens, config.context_size
        ));
    }

    // Add all prompt tokens to a batch.
    let mut batch = LlamaBatch::new(tokens.len().max(1), 1);
    let last_idx = (tokens.len() - 1) as i32;

    for (i, token) in tokens.iter().enumerate() {
        let is_last = i as i32 == last_idx;
        batch
            .add(*token, i as i32, &[seq.seq_id], is_last)
            .map_err(|e| format!("Failed to add prompt token: {e}"))?;
    }

    ctx.decode(&mut batch)
        .map_err(|e| format!("Prompt decode failed: {e}"))?;

    Ok((n_tokens, tokens.len() as i32))
}

/// Send a `StreamEvent::Done` to the sequence's channel.
fn send_done(seq: &ActiveSequence) {
    let duration_ms = seq.start.elapsed().as_millis() as u64;
    let _ = seq.tx.blocking_send(StreamEvent::Done {
        tokens_generated: seq.tokens_generated,
        tokens_prompt: seq.tokens_prompt,
        duration_ms,
    });
}

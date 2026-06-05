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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

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
    /// Hold-back buffer for streaming stop-sequence detection. Any trailing
    /// text that could still grow into (a prefix of) a stop sequence is kept
    /// here and only flushed once confirmed not to be part of one. On a full
    /// stop match it is truncated at the match; on EOG it is discarded — so a
    /// model that spells the turn delimiter out as plain text (e.g. Gemma
    /// emitting `<end_of_turn` followed by an EOG token) never leaks it.
    pending: String,
    tokens_prompt: u32,
    tokens_generated: u32,
    max_tokens: u32,
    n_past: i32,
    stop_sequences: Vec<String>,
    /// Substrings silently elided from the streamed output (e.g. harmony
    /// format artifacts that some models hallucinate).
    filter_sequences: Vec<String>,
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
    /// Explicit shutdown flag — checked by the scheduler loop every iteration.
    /// Using AtomicBool because channel disconnect alone is unreliable
    /// (cloned handles in in-flight SlotSnapshots keep the channel open).
    shutdown: Arc<AtomicBool>,
    /// Join handle for the scheduler thread.
    thread: Arc<std::sync::Mutex<Option<std::thread::JoinHandle<()>>>>,
}

impl SchedulerHandle {
    /// Shut down the scheduler and wait for the thread to exit.
    ///
    /// Sets the shutdown flag, wakes the thread, and joins it to
    /// guarantee the old LlamaBackend / model / context are fully
    /// destroyed before returning.
    pub fn shutdown(self) {
        // Set the flag — the scheduler loop checks this every iteration.
        self.shutdown.store(true, Ordering::SeqCst);
        // Wake the thread so it sees the flag immediately.
        {
            let _lock = self.notify_mutex.lock().unwrap();
            self.notify.notify_one();
        }
        // Wait for the thread to finish.
        if let Some(handle) = self.thread.lock().unwrap().take() {
            tracing::info!("Waiting for scheduler thread to exit...");
            let _ = handle.join();
            tracing::info!("Scheduler thread exited");
        }
    }

    /// Submit a request for inference. Returns immediately.
    ///
    /// The caller should listen on the returned `mpsc::Receiver<StreamEvent>`
    /// for token events.
    pub fn submit(&self, request: GenerateRequest) -> mpsc::Receiver<StreamEvent> {
        let (tx, rx) = mpsc::channel::<StreamEvent>(256);

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

/// Info reported by the scheduler after model load.
pub struct ModelReadyInfo {
    /// KV cache key memory estimate in MiB.
    pub kv_k_mib: f64,
    /// KV cache value memory estimate in MiB.
    pub kv_v_mib: f64,
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
    ///
    /// Blocks until the model is fully loaded so that callers can rely on the
    /// handle being ready for inference when this method returns.
    pub fn start(self) -> Result<(SchedulerHandle, ModelReadyInfo), Box<dyn std::error::Error + Send + Sync>> {
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
        let shutdown = Arc::new(AtomicBool::new(false));
        let notify_clone = Arc::clone(&notify);
        let notify_mutex_clone = Arc::clone(&notify_mutex);
        let shutdown_clone = Arc::clone(&shutdown);

        // Channel for the scheduler thread to signal model load completion.
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<ModelReadyInfo, String>>();

        let config = self.config.clone();
        let sched_config = self.sched_config.clone();

        let join_handle = std::thread::Builder::new()
            .name("eullm-scheduler".into())
            .spawn(move || {
                // Catch panics from Rust code. This won't catch C-level abort()
                // from llama.cpp (see SIGABRT handler in main.rs for that).
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run_scheduler_loop(config, sched_config, req_rx, notify_clone, notify_mutex_clone, shutdown_clone, ready_tx)
                })) {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => {
                        tracing::error!("Scheduler thread exited with error: {e}");
                    }
                    Err(panic_info) => {
                        let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                            s.to_string()
                        } else if let Some(s) = panic_info.downcast_ref::<String>() {
                            s.clone()
                        } else {
                            "unknown panic".to_string()
                        };
                        tracing::error!("Scheduler thread panicked: {msg}");
                        eprintln!("\n[EULLM] FATAL: Scheduler thread panicked: {msg}");
                        eprintln!("[EULLM] The inference engine has stopped. Restart eullm to continue.");
                    }
                }
            })?;

        // Wait for the model to finish loading before returning.
        let model_info = match ready_rx.recv() {
            Ok(Ok(info)) => info,
            Ok(Err(e)) => return Err(e.into()),
            Err(_) => return Err("Scheduler thread exited before model was loaded".into()),
        };

        Ok((SchedulerHandle {
            tx: req_tx,
            notify,
            notify_mutex,
            shutdown,
            thread: Arc::new(std::sync::Mutex::new(Some(join_handle))),
        }, model_info))
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Build a human-readable hint for context allocation failures.
///
/// Suggests smaller `--ctx-size` values and, if the requested size is large,
/// recommends quantizing the KV cache to cut VRAM use.
fn ctx_oom_hint(requested_tokens: u32) -> String {
    let smaller = [requested_tokens / 2, requested_tokens / 4]
        .iter()
        .filter(|&&v| v >= 512)
        .map(|v| format!("--ctx-size {v}"))
        .collect::<Vec<_>>()
        .join("  or  ");

    let kv_tip = if requested_tokens >= 8192 {
        "\n  Or quantize the KV cache: --cache-type-k q4_0 --cache-type-v q4_0 (cuts KV-cache VRAM by ~4×)."
    } else {
        ""
    };

    format!(
        "  Requested context: {requested_tokens} tokens — KV cache likely exceeds available VRAM.\
        \n  Try a smaller context: {smaller}{kv_tip}"
    )
}

// ── Scheduler loop (runs on a dedicated thread) ─────────────────────────────

fn run_scheduler_loop(
    config: InferenceConfig,
    sched_config: SchedulerConfig,
    req_rx: crossbeam_channel::Receiver<ScheduledRequest>,
    notify: Arc<std::sync::Condvar>,
    notify_mutex: Arc<std::sync::Mutex<()>>,
    shutdown: Arc<AtomicBool>,
    ready_tx: std::sync::mpsc::Sender<Result<ModelReadyInfo, String>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    super::check_gpu_support(config.gpu_layers);

    tracing::info!("Initializing llama.cpp backend (scheduler)...");
    let mut backend = LlamaBackend::init()?;
    // NOTE: backend.void_logs() is intentionally deferred until AFTER the
    // context is created.  During model load and KV cache allocation, llama.cpp
    // prints useful diagnostics (VRAM offload, tensor sizes, OOM details) that
    // help users diagnose failures like out-of-VRAM context allocation.
    // Once the context is up, void_logs() is called to suppress the repetitive
    // per-decode messages (CUDA graph warmup etc.) that pollute interactive output.

    let model_params = if config.gpu_layers >= 0 {
        LlamaModelParams::default().with_n_gpu_layers(config.gpu_layers as u32)
    } else {
        LlamaModelParams::default().with_n_gpu_layers(1000)
    };
    let model_params = pin!(model_params);

    tracing::info!("Loading model: {}", config.model_path.display());
    let model = match LlamaModel::load_from_file(&backend, &config.model_path, &model_params) {
        Ok(m) => m,
        Err(e) => {
            let msg = format!("Failed to load model: {e}");
            let _ = ready_tx.send(Err(msg.clone()));
            return Err(msg.into());
        }
    };

    // Use context_size as the TOTAL KV cache budget, shared across all
    // sequences (matching Ollama / llama.cpp server behaviour).  Previous
    // code multiplied context_size × max_batch_size, which easily overflowed
    // VRAM and forced llama.cpp to fall back to CPU for KV cache ops.
    let total_ctx = config.context_size;
    let per_seq_ctx = config.context_size / sched_config.max_batch_size as u32;
    tracing::info!(
        "Allocating context: {} total tokens, {} per sequence ({} slots, flash_attn={}, n_batch={})",
        total_ctx,
        per_seq_ctx,
        sched_config.max_batch_size,
        config.flash_attn,
        config.n_batch,
    );
    let ctx_size = NonZeroU32::new(total_ctx).unwrap_or(NonZeroU32::new(4096).unwrap());

    let has_quantized_cache = config.cache_type_k != super::KvCacheType::F16
        || config.cache_type_v != super::KvCacheType::F16;

    // Mixed unknown KV types (carry-through to raw GGML IDs) — fallback logic.
    let has_mixed_tq = matches!(
        (&config.cache_type_k, &config.cache_type_v),
        (super::KvCacheType::Unknown(k), super::KvCacheType::Unknown(v)) if k != v
    );

    let ctx_params = super::build_ctx_params(&config, ctx_size)
        .with_n_seq_max(sched_config.max_batch_size as u32);

    let mut ctx = match model.new_context(&backend, ctx_params) {
        Ok(c) => c,
        Err(e) => {
            // Build a hint suggesting smaller ctx-size values to try.
            let hint = ctx_oom_hint(total_ctx);
            if has_quantized_cache {
                if has_mixed_tq {
                    // Mixed TQ fallback: try the heavier (more precise) type for both.
                    let heavier = if let (super::KvCacheType::Unknown(k), super::KvCacheType::Unknown(v))
                        = (config.cache_type_k, config.cache_type_v) {
                        if k >= v { config.cache_type_k } else { config.cache_type_v }
                    } else {
                        config.cache_type_k
                    };
                    let name = super::cache_type_display(&heavier);
                    eprintln!("[EULLM] Mixed TQ failed — trying {name}/{name}...");

                    let ctx_params = super::build_ctx_params_with_cache(
                        &config, ctx_size, heavier, heavier,
                    ).with_n_seq_max(sched_config.max_batch_size as u32);

                    match model.new_context(&backend, ctx_params) {
                        Ok(c) => c,
                        Err(_) => {
                            eprintln!("[EULLM] {name} also failed — falling back to F16/F16");
                            let ctx_params = super::build_ctx_params_with_cache(
                                &config, ctx_size,
                                super::KvCacheType::F16, super::KvCacheType::F16,
                            ).with_n_seq_max(sched_config.max_batch_size as u32);
                            match model.new_context(&backend, ctx_params) {
                                Ok(c) => c,
                                Err(e3) => {
                                    let msg = format!(
                                        "Context allocation failed (likely out of VRAM): {e3}\n{hint}"
                                    );
                                    let _ = ready_tx.send(Err(msg.clone()));
                                    return Err(msg.into());
                                }
                            }
                        }
                    }
                } else {
                    eprintln!(
                        "[EULLM] KV cache fallback: {:?}/{:?} → F16/F16",
                        config.cache_type_k, config.cache_type_v,
                    );
                    let ctx_params = super::build_ctx_params_with_cache(
                        &config, ctx_size,
                        super::KvCacheType::F16, super::KvCacheType::F16,
                    ).with_n_seq_max(sched_config.max_batch_size as u32);

                    match model.new_context(&backend, ctx_params) {
                        Ok(c) => c,
                        Err(e2) => {
                            let msg = format!(
                                "Context allocation failed (likely out of VRAM): {e2}\n{hint}"
                            );
                            let _ = ready_tx.send(Err(msg.clone()));
                            return Err(msg.into());
                        }
                    }
                }
            } else {
                let msg = format!(
                    "Context allocation failed (likely out of VRAM): {e}\n{hint}"
                );
                let _ = ready_tx.send(Err(msg.clone()));
                return Err(msg.into());
            }
        }
    };

    // Context created successfully — now suppress repetitive per-decode
    // llama.cpp messages (CUDA graph warmup, etc.) to keep output clean.
    backend.void_logs();

    let mut active: Vec<ActiveSequence> = Vec::with_capacity(sched_config.max_batch_size);
    // Pool of reusable seq_ids in range [0, max_batch_size).
    // llama.cpp requires seq_id < n_seq_max, so we recycle them.
    let mut free_seq_ids: Vec<i32> = (0..sched_config.max_batch_size as i32).rev().collect();
    // Pre-allocate the decode batch once — reused every iteration to avoid
    // repeated malloc/free in the hot decode loop.
    let mut decode_batch = LlamaBatch::new(sched_config.max_batch_size.max(1), 1);

    tracing::info!(
        "Scheduler running — max_batch_size={}, queue_capacity={}, total_ctx={}, per_seq_ctx={}",
        sched_config.max_batch_size,
        sched_config.queue_capacity,
        total_ctx,
        per_seq_ctx,
    );

    // Estimate KV cache memory from model dimensions and cache types.
    let kv_info = estimate_kv_memory(
        &model,
        total_ctx as u64,
        &config.cache_type_k,
        &config.cache_type_v,
    );

    // Signal that the model is loaded and ready.
    let _ = ready_tx.send(Ok(kv_info));

    loop {
        // ── 0. Check shutdown flag ────────────────────────────────────
        if shutdown.load(Ordering::SeqCst) {
            tracing::info!("Shutdown requested — draining {} active sequences", active.len());
            for seq in &active {
                let _ = seq.tx.try_send(StreamEvent::Error("Server shutting down".into()));
            }
            return Ok(());
        }

        // ── 1. Drain new requests from the queue ────────────────────────
        while active.len() < sched_config.max_batch_size {
            match req_rx.try_recv() {
                Ok(scheduled) => {
                    let seq_id = match free_seq_ids.pop() {
                        Some(id) => id,
                        None => break, // No free slots — should not happen due to active.len() check
                    };

                    let req = &scheduled.request;
                    let seed = req.seed.unwrap_or(seq_id as u32);
                    let sampler = {
                        let mut chain: Vec<LlamaSampler> = Vec::new();
                        // Grammar (if any) must be first in the chain
                        if let Some(ref grammar_str) = req.grammar {
                            match LlamaSampler::grammar(&model, grammar_str, "root") {
                                Ok(gs) => chain.push(gs),
                                Err(e) => tracing::warn!("Grammar sampler init failed ({e:?}), falling back to unconstrained"),
                            }
                        }
                        // Repeat penalty (Ollama default: 1.1, last 64 tokens)
                        if req.repeat_penalty != 1.0 {
                            chain.push(LlamaSampler::penalties(req.repeat_last_n, req.repeat_penalty, 0.0, 0.0));
                        }
                        // Top-K (Ollama default: 40)
                        if req.top_k > 0 {
                            chain.push(LlamaSampler::top_k(req.top_k));
                        }
                        // Top-P (Ollama default: 0.9)
                        if req.top_p < 1.0 {
                            chain.push(LlamaSampler::top_p(req.top_p, 1));
                        }
                        // Min-P (Ollama default: 0.0)
                        if req.min_p > 0.0 {
                            chain.push(LlamaSampler::min_p(req.min_p, 1));
                        }
                        // Temperature
                        chain.push(LlamaSampler::temp(req.temperature));
                        // Sampling distribution
                        chain.push(LlamaSampler::dist(seed));
                        LlamaSampler::chain_simple(chain)
                    };

                    let seq = ActiveSequence {
                        seq_id,
                        tx: scheduled.tx,
                        sampler,
                        decoder: encoding_rs::UTF_8.new_decoder(),
                        pending: String::new(),
                        tokens_prompt: 0,
                        tokens_generated: 0,
                        max_tokens: scheduled.request.max_tokens,
                        n_past: 0,
                        stop_sequences: scheduled.request.stop_sequences.clone(),
                        filter_sequences: scheduled.request.filter_sequences.clone(),
                        start: std::time::Instant::now(),
                        prefilled: false,
                        last_token: None,
                    };

                    // Tokenize and prefill immediately.
                    match prefill_sequence(&model, &mut ctx, &config, &scheduled.request, &seq, per_seq_ctx) {
                        Ok((n_tokens, n_past, effective_max)) => {
                            let mut seq = seq;
                            seq.tokens_prompt = n_tokens;
                            seq.n_past = n_past;
                            seq.max_tokens = effective_max;
                            seq.prefilled = true;

                            // Sample the first generated token directly from prefill logits.
                            // Use output index -1 (= last output). Only the final prompt
                            // token had logits enabled, so there is exactly one output entry.
                            let token = seq.sampler.sample(&ctx, -1);
                            seq.sampler.accept(token);

                            if model.is_eog_token(token) {
                                send_done(&seq);
                                let _ = ctx.clear_kv_cache_seq(Some(seq.seq_id as u32), None, None);
                                free_seq_ids.push(seq.seq_id);
                            } else {
                                seq.tokens_generated += 1;

                                match model.token_to_piece(token, &mut seq.decoder, true, None) {
                                    Ok(piece) => {
                                        match process_piece(&mut seq.pending, &seq.stop_sequences, &seq.filter_sequences, &piece) {
                                            PieceOutcome::Stop(out) => {
                                                if !out.is_empty() {
                                                    let _ = seq.tx.try_send(StreamEvent::Token(out));
                                                }
                                                send_done(&seq);
                                                let _ = ctx.clear_kv_cache_seq(Some(seq.seq_id as u32), None, None);
                                                free_seq_ids.push(seq.seq_id);
                                            }
                                            PieceOutcome::Emit(out) => {
                                                if seq.tokens_generated >= seq.max_tokens {
                                                    // Truncation: flush whatever was held back too.
                                                    let tail = std::mem::take(&mut seq.pending);
                                                    let final_out = out + &tail;
                                                    if !final_out.is_empty() {
                                                        let _ = seq.tx.try_send(StreamEvent::Token(final_out));
                                                    }
                                                    send_done(&seq);
                                                    let _ = ctx.clear_kv_cache_seq(Some(seq.seq_id as u32), None, None);
                                                    free_seq_ids.push(seq.seq_id);
                                                } else {
                                                    if !out.is_empty() {
                                                        let _ = seq.tx.try_send(StreamEvent::Token(out));
                                                    }
                                                    seq.last_token = Some(token);
                                                    active.push(seq);
                                                }
                                            }
                                        }
                                    }
                                    Err(_) => {
                                        send_done(&seq);
                                        let _ = ctx.clear_kv_cache_seq(Some(seq.seq_id as u32), None, None);
                                        free_seq_ids.push(seq.seq_id);
                                    }
                                }
                            }

                            tracing::debug!("Sequence {seq_id} prefilled ({n_tokens} prompt tokens)");
                        }
                        Err(e) => {
                            let _ = seq.tx.try_send(StreamEvent::Error(format!(
                                "Prefill failed: {e}"
                            )));
                            free_seq_ids.push(seq.seq_id);
                        }
                    }
                }
                Err(crossbeam_channel::TryRecvError::Empty) => break,
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    tracing::info!("Request channel closed — scheduler shutting down.");
                    // Finish active sequences gracefully.
                    for seq in &active {
                        let _ = seq.tx.try_send(StreamEvent::Error(
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
        decode_batch.clear();

        for seq in active.iter() {
            if let Some(token) = seq.last_token
                && decode_batch.add(token, seq.n_past, &[seq.seq_id], true).is_err()
            {
                tracing::warn!("Failed to add token to batch for seq {}", seq.seq_id);
            }
        }

        // ── 4. Decode the batch ─────────────────────────────────────────
        if decode_batch.n_tokens() > 0 {
            tracing::debug!(
                "Decoding batch: {} tokens, {} active sequences",
                decode_batch.n_tokens(),
                active.len(),
            );
            if let Err(e) = ctx.decode(&mut decode_batch) {
                tracing::error!("Batch decode failed: {e}");
                // Send errors to all active sequences and clear.
                for seq in active.drain(..) {
                    let _ = seq.tx.try_send(StreamEvent::Error(format!(
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
                    match process_piece(&mut seq.pending, &seq.stop_sequences, &seq.filter_sequences, &piece) {
                        PieceOutcome::Stop(out) => {
                            if !out.is_empty() {
                                let _ = seq.tx.try_send(StreamEvent::Token(out));
                            }
                            send_done(seq);
                            to_remove.push(i);
                            let _ = ctx.clear_kv_cache_seq(Some(seq.seq_id as u32), None, None);
                            continue;
                        }
                        PieceOutcome::Emit(out) => {
                            if !out.is_empty()
                                && seq.tx.try_send(StreamEvent::Token(out)).is_err()
                            {
                                // Receiver dropped — client disconnected.
                                to_remove.push(i);
                                let _ = ctx.clear_kv_cache_seq(Some(seq.seq_id as u32), None, None);
                                continue;
                            }
                        }
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
                // Truncation (not a stop): flush any held-back tail as real text.
                let tail = std::mem::take(&mut seq.pending);
                if !tail.is_empty() {
                    let _ = seq.tx.try_send(StreamEvent::Token(tail));
                }
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
            let removed = active.swap_remove(i);
            free_seq_ids.push(removed.seq_id);
        }
    }
}

/// Prefill a sequence's prompt tokens into the context.
///
/// Returns `(prompt_tokens, n_past, effective_max_tokens)`.
fn prefill_sequence(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    config: &InferenceConfig,
    request: &GenerateRequest,
    seq: &ActiveSequence,
    per_seq_ctx: u32,
) -> Result<(u32, i32, u32), String> {
    let bos = if request.raw { AddBos::Never } else { AddBos::Always };
    let tokens = model
        .str_to_token(&request.prompt, bos)
        .map_err(|e| format!("Tokenization failed: {e}"))?;

    let n_tokens = tokens.len() as u32;

    // Effective context: per-request num_ctx (clamped to per-sequence limit)
    // or the per-sequence default.
    let effective_ctx = request
        .num_ctx
        .map(|n| n.min(per_seq_ctx))
        .unwrap_or(per_seq_ctx);

    if n_tokens >= effective_ctx {
        return Err(format!(
            "Prompt ({n_tokens} tokens) does not fit in context window ({effective_ctx})"
        ));
    }

    // Cap max_tokens to remaining budget within effective context.
    let max_output = effective_ctx - n_tokens;
    let effective_max_tokens = request.max_tokens.min(max_output);

    if effective_max_tokens < request.max_tokens {
        tracing::warn!(
            "num_predict capped: requested={}, effective={} (context={}, prompt_tokens={})",
            request.max_tokens,
            effective_max_tokens,
            effective_ctx,
            n_tokens,
        );
    }
    tracing::info!(
        "Seq {}: prompt={} tokens, max_output={}, effective_ctx={}",
        seq.seq_id,
        n_tokens,
        effective_max_tokens,
        effective_ctx,
    );

    // Prefill in chunks of n_batch tokens. llama.cpp asserts if a single
    // decode call processes more tokens than n_batch, which causes SIGABRT.
    // Long RAG prompts easily exceed the default 2048 n_batch.
    let chunk_size = config.n_batch as usize;
    let last_idx = tokens.len() - 1;

    tracing::debug!(
        "Prefilling seq {} with {} tokens in chunks of {} (context_size={})",
        seq.seq_id,
        tokens.len(),
        chunk_size,
        config.context_size,
    );

    for chunk_start in (0..tokens.len()).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(tokens.len());
        let chunk = &tokens[chunk_start..chunk_end];
        let mut batch = LlamaBatch::new(chunk.len().max(1), 1);

        for (j, token) in chunk.iter().enumerate() {
            let abs_pos = chunk_start + j;
            let is_last = abs_pos == last_idx;
            batch
                .add(*token, abs_pos as i32, &[seq.seq_id], is_last)
                .map_err(|e| format!("Failed to add prompt token: {e}"))?;
        }

        ctx.decode(&mut batch)
            .map_err(|e| format!("Prompt decode failed at chunk {chunk_start}..{chunk_end}: {e}"))?;
    }

    Ok((n_tokens, tokens.len() as i32, effective_max_tokens))
}

/// Outcome of feeding one decoded piece through the stop-sequence filter.
enum PieceOutcome {
    /// Safe-to-stream text (may be empty); generation continues.
    Emit(String),
    /// A full stop sequence was hit. The contained text is whatever preceded
    /// the stop marker and should be streamed before finishing.
    Stop(String),
}

/// Length (in bytes) of the longest suffix of `buf` that is a *proper* prefix
/// of any stop sequence. This is the amount of trailing text that must be held
/// back: it could still grow into a full stop sequence on the next token.
///
/// A full match is handled by the caller (via `find`) before this is consulted,
/// so we never report the entire stop sequence here.
fn stop_prefix_holdback(buf: &str, stops: &[String]) -> usize {
    let mut max = 0;
    for s in stops {
        if s.is_empty() {
            continue;
        }
        // Try the longest possible overlap first; the suffix of `buf` must be a
        // prefix of `s` and strictly shorter than `s` (full matches handled elsewhere).
        let upper = buf.len().min(s.len().saturating_sub(1));
        let mut k = upper;
        while k >= 1 {
            let start = buf.len() - k;
            if buf.is_char_boundary(start) && s.as_bytes().starts_with(&buf.as_bytes()[start..]) {
                if k > max {
                    max = k;
                }
                break;
            }
            k -= 1;
        }
    }
    max
}

/// Feed one decoded `piece` into the per-sequence `pending` hold-back buffer and
/// decide what is safe to stream.
///
/// - If appending the piece completes a stop sequence, everything up to the
///   stop marker is returned as `Stop(..)` and `pending` is cleared.
/// - Otherwise the longest trailing run that could still become a stop sequence
///   is retained in `pending`, and the rest is returned as `Emit(..)`.
///
/// This makes streaming robust against models that spell a turn delimiter out
/// as ordinary text and only then emit an EOG token (e.g. Gemma emitting
/// `<end_of_turn` + EOG): the partial delimiter sits in `pending` and is
/// discarded when the caller observes EOG, instead of leaking to the client.
fn process_piece(
    pending: &mut String,
    stops: &[String],
    filters: &[String],
    piece: &str,
) -> PieceOutcome {
    pending.push_str(piece);

    // 1. Stop sequences win: terminate generation at the earliest hit.
    let mut cut: Option<usize> = None;
    for s in stops {
        if let Some(pos) = pending.find(s.as_str()) {
            cut = Some(cut.map_or(pos, |c| c.min(pos)));
        }
    }
    if let Some(pos) = cut {
        let out = pending[..pos].to_string();
        pending.clear();
        return PieceOutcome::Stop(out);
    }

    // 2. Filter sequences: silently elide every completed occurrence, then
    // continue. Unlike stops, these do not terminate the response.
    for f in filters {
        if f.is_empty() {
            continue;
        }
        while let Some(pos) = pending.find(f.as_str()) {
            pending.replace_range(pos..pos + f.len(), "");
        }
    }

    // 3. Hold back any trailing run that could still complete EITHER a stop
    // or a filter sequence. Reuses `stop_prefix_holdback` for both — it just
    // looks at suffix→prefix overlaps and is agnostic to the list's meaning.
    let holdback = stop_prefix_holdback(pending, stops)
        .max(stop_prefix_holdback(pending, filters));
    let emit_upto = pending.len() - holdback;
    let out = pending[..emit_upto].to_string();
    pending.drain(..emit_upto);
    PieceOutcome::Emit(out)
}

/// Bytes per element for a KV cache type (approximate for quantized types).
fn cache_type_bytes_per_elem(ct: &super::KvCacheType) -> f64 {
    match ct {
        super::KvCacheType::F16 => 2.0,
        super::KvCacheType::F32 => 4.0,
        super::KvCacheType::Q8_0 => 34.0 / 32.0,
        super::KvCacheType::Q4_0 => 18.0 / 32.0,
        super::KvCacheType::Q4_1 => 20.0 / 32.0,
        super::KvCacheType::Q5_0 => 22.0 / 32.0,
        super::KvCacheType::Q5_1 => 24.0 / 32.0,
        _ => 2.0, // default to F16
    }
}

/// Estimate KV cache memory from model dimensions and cache types.
fn estimate_kv_memory(
    model: &LlamaModel,
    n_ctx: u64,
    cache_type_k: &super::KvCacheType,
    cache_type_v: &super::KvCacheType,
) -> ModelReadyInfo {
    // Try to get model dimensions. If any method fails or returns 0,
    // fall back to reporting 0 (unknown).
    let n_embd = model.n_embd() as f64;     // c_int
    let n_layer = model.n_layer() as f64;   // u32
    let n_head = model.n_head() as f64;     // u32
    let n_head_kv = model.n_head_kv() as f64; // u32

    if n_embd <= 0.0 || n_layer <= 0.0 || n_head <= 0.0 || n_head_kv <= 0.0 {
        return ModelReadyInfo { kv_k_mib: 0.0, kv_v_mib: 0.0 };
    }

    let head_dim = n_embd / n_head;
    // Elements per K or V = n_layer × n_ctx × n_head_kv × head_dim
    let n_elements = n_layer * n_ctx as f64 * n_head_kv * head_dim;

    let kv_k_bytes = n_elements * cache_type_bytes_per_elem(cache_type_k);
    let kv_v_bytes = n_elements * cache_type_bytes_per_elem(cache_type_v);

    ModelReadyInfo {
        kv_k_mib: kv_k_bytes / (1024.0 * 1024.0),
        kv_v_mib: kv_v_bytes / (1024.0 * 1024.0),
    }
}

/// Send a `StreamEvent::Done` to the sequence's channel.
fn send_done(seq: &ActiveSequence) {
    let duration_ms = seq.start.elapsed().as_millis() as u64;
    let _ = seq.tx.try_send(StreamEvent::Done {
        tokens_generated: seq.tokens_generated,
        tokens_prompt: seq.tokens_prompt,
        duration_ms,
    });
}

#[cfg(test)]
mod tests {
    use super::{process_piece, stop_prefix_holdback, PieceOutcome};

    fn gemma_stops() -> Vec<String> {
        vec!["<end_of_turn>".to_string()]
    }

    fn drain(pending: &mut String, stops: &[String], pieces: &[&str]) -> (String, bool) {
        drain_with_filters(pending, stops, &[], pieces)
    }

    fn drain_with_filters(
        pending: &mut String,
        stops: &[String],
        filters: &[String],
        pieces: &[&str],
    ) -> (String, bool) {
        let mut emitted = String::new();
        let mut stopped = false;
        for p in pieces {
            match process_piece(pending, stops, filters, p) {
                PieceOutcome::Emit(s) => emitted.push_str(&s),
                PieceOutcome::Stop(s) => {
                    emitted.push_str(&s);
                    stopped = true;
                    break;
                }
            }
        }
        (emitted, stopped)
    }

    #[test]
    fn holdback_detects_partial_prefix() {
        let stops = gemma_stops();
        // A trailing partial delimiter must be held back.
        assert_eq!(stop_prefix_holdback("hello<end_of_turn", &stops), "<end_of_turn".len());
        // No overlap → nothing held back.
        assert_eq!(stop_prefix_holdback("hello world", &stops), 0);
        // Only the suffix that overlaps a prefix is held back.
        assert_eq!(stop_prefix_holdback("a<end", &stops), "<end".len());
    }

    #[test]
    fn gemma_text_delimiter_then_eog_does_not_leak() {
        // Model spells the delimiter out as plain text (the real Gemma 4 case):
        // < end _ of _ turn, then an EOG token (no further piece). The partial
        // `<end_of_turn` must stay buffered and never be emitted.
        let stops = gemma_stops();
        let mut pending = String::new();
        let (emitted, stopped) =
            drain(&mut pending, &stops, &["hello", " world", "<", "end", "_", "of", "_", "turn"]);
        assert_eq!(emitted, "hello world");
        assert!(!stopped, "no full stop seq seen yet — EOG handles termination");
        // Caller discards `pending` on EOG; confirm it holds only the partial.
        assert_eq!(pending, "<end_of_turn");
    }

    #[test]
    fn full_stop_sequence_truncates_cleanly() {
        let stops = gemma_stops();
        let mut pending = String::new();
        let (emitted, stopped) =
            drain(&mut pending, &stops, &["Rome.", "<end_of_turn>"]);
        assert!(stopped);
        assert_eq!(emitted, "Rome.");
        assert!(pending.is_empty());
    }

    #[test]
    fn false_alarm_prefix_is_released() {
        // `<end` looks like the start of `<end_of_turn>` but turns out to be
        // ordinary text — it must be released once disambiguated.
        let stops = gemma_stops();
        let mut pending = String::new();
        let (emitted, stopped) =
            drain(&mut pending, &stops, &["the ", "<end", "point", " is near"]);
        assert!(!stopped);
        assert_eq!(emitted, "the <endpoint is near");
        assert!(pending.is_empty());
    }

    fn harmony_filters() -> Vec<String> {
        crate::inference::DEFAULT_HARMONY_FILTERS
            .iter()
            .map(|s| (*s).to_string())
            .collect()
    }

    #[test]
    fn filter_strips_complete_harmony_block_with_known_role() {
        // Gemma 4 12B emits `<|channel>thought<channel|>` at the start of a
        // reply. The combined pattern in DEFAULT_HARMONY_FILTERS catches the
        // whole block (including the "thought" word) as one unit, so nothing
        // residual appears in the output.
        let stops = gemma_stops();
        let filters = harmony_filters();
        let mut pending = String::new();
        let (emitted, stopped) = drain_with_filters(
            &mut pending,
            &stops,
            &filters,
            &["<|channel>thought<channel|>Ciao!"],
        );
        assert!(!stopped, "filter must NEVER terminate generation");
        assert_eq!(emitted, "Ciao!");
    }

    #[test]
    fn filter_falls_back_to_delimiter_strip_for_unknown_role() {
        // If the model produces a Harmony channel with a role word we have
        // not enumerated in DEFAULT_HARMONY_FILTERS, the single-delimiter
        // entries still scrub the visible syntax; the word between is left
        // dangling. Documents the degradation contract.
        let stops = gemma_stops();
        let filters = harmony_filters();
        let mut pending = String::new();
        let (emitted, stopped) = drain_with_filters(
            &mut pending,
            &stops,
            &filters,
            &["<|channel>banana<channel|>Hello"],
        );
        assert!(!stopped);
        assert_eq!(emitted, "bananaHello");
    }

    #[test]
    fn filter_works_across_token_boundaries() {
        // Realistic case: marker split across two model pieces.
        let stops = gemma_stops();
        let filters = harmony_filters();
        let mut pending = String::new();
        let (emitted, stopped) = drain_with_filters(
            &mut pending,
            &stops,
            &filters,
            &["before <|chan", "nel>after"],
        );
        assert!(!stopped);
        assert_eq!(emitted, "before after");
        assert!(pending.is_empty());
    }

    #[test]
    fn filter_mid_word_artifact_is_removed() {
        // Real screenshot: the model wrote "vor<image|>resti" — must become "vorresti".
        let stops = gemma_stops();
        let filters = harmony_filters();
        let mut pending = String::new();
        let (emitted, stopped) =
            drain_with_filters(&mut pending, &stops, &filters, &["vor<image|>resti"]);
        assert!(!stopped);
        assert_eq!(emitted, "vorresti");
    }

    #[test]
    fn stop_wins_over_filter_when_both_appear() {
        // If a stop sequence and a filter both occur, the stop takes precedence
        // and terminates the response; the filter doesn't get a chance to act
        // beyond the cut.
        let stops = gemma_stops();
        let filters = harmony_filters();
        let mut pending = String::new();
        let (emitted, stopped) = drain_with_filters(
            &mut pending,
            &stops,
            &filters,
            &["answer.<end_of_turn><|channel>noise"],
        );
        assert!(stopped);
        assert_eq!(emitted, "answer.");
    }

    #[test]
    fn ordinary_text_unaffected_by_filters() {
        // Regression guard: the presence of filter rules must not change
        // streaming for content that contains none of them.
        let stops = gemma_stops();
        let filters = harmony_filters();
        let mut pending = String::new();
        let (emitted, stopped) = drain_with_filters(
            &mut pending,
            &stops,
            &filters,
            &["hello", " world", "."],
        );
        assert!(!stopped);
        assert_eq!(emitted, "hello world.");
        assert!(pending.is_empty());
    }
}

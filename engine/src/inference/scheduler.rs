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

use super::{random_seed_fallback, GenerateRequest, InferenceConfig, StreamEvent};

/// Configuration for the batch scheduler.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Maximum number of sequences decoded in parallel.
    pub max_batch_size: usize,
    /// How many requests can wait in the submission queue before back-pressure.
    pub queue_capacity: usize,
    /// Max number of full-sequence-state checkpoints kept for prompt-prefix
    /// restore, across all sequences (content-addressed, like idle slots —
    /// not tied to any one seq_id). `0` (default) disables checkpointing
    /// entirely: no snapshot is ever taken, matching pre-checkpoint
    /// behavior exactly. Mirrors llama.cpp server's `--ctx-checkpoints`
    /// (default there: 32); kept off by default here because each
    /// checkpoint costs one full sequence-state snapshot and most setups
    /// don't need it. Exists specifically for hybrid/recurrent
    /// architectures (Qwen3.5/3.6) where KV-cache prefix reuse otherwise
    /// falls back to a full re-prefill whenever the live resident slot's
    /// matched prefix is shorter than what an earlier checkpoint of the
    /// same conversation would cover — see the `--ctx-checkpoints` section
    /// of the README.
    pub ctx_checkpoints: usize,
    /// Minimum number of new tokens since the closest existing checkpoint
    /// of the same lineage before taking another one. Mirrors llama.cpp
    /// server's `--checkpoint-min-step` (default there: 8192). Only
    /// consulted when `ctx_checkpoints > 0`.
    pub checkpoint_min_step: u32,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            queue_capacity: 64,
            ctx_checkpoints: 0,
            checkpoint_min_step: 8192,
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
    /// The full prompt token sequence for this turn (post reuse-decision),
    /// i.e. exactly what `model.str_to_token()` produced. Needed to
    /// reconstruct this sequence's full resident-token history when it
    /// completes.
    prompt_tokens: Vec<LlamaToken>,
    /// Every token actually sampled and successfully decoded so far this
    /// turn, appended in lockstep with confirmed decode steps, never
    /// speculatively.
    generated_tokens: Vec<LlamaToken>,
    /// The exact request prompt text this turn was given, verbatim. Paired
    /// with `prompt_tokens` (its tokenization) so a future request whose
    /// prompt text starts with `prompt_text ++ raw_generated_pieces` can
    /// reuse `prompt_tokens ++ generated_tokens` directly instead of
    /// retokenizing — see `text_prefix_match`.
    prompt_text: String,
    /// Every text piece this turn has actually produced via `token_to_piece`,
    /// one entry per `generated_tokens` push, same order — deliberately NOT
    /// the client-visible stream (which has stop sequences erased for
    /// display). Kept as one `String` per token rather than pre-concatenated
    /// so `resident_text` can drop exactly the trailing entries for
    /// not-yet-decoded tokens (see its doc comment: `generated_tokens` is
    /// pushed for the newly-sampled token before that token's own decode
    /// call has happened, so it's briefly one ahead of `n_past`).
    raw_generated_pieces: Vec<String>,
}

/// An idle sequence slot together with the exact token history currently
/// resident in its KV cache, reused across requests via longest-common-prefix
/// matching, mirroring upstream llama.cpp server's slot model
/// (tools/server/server-context.cpp, get_available_slot /
/// server_tokens::get_common_prefix).
struct CachedSlot {
    seq_id: i32,
    /// Full token history (prompt ++ generated) currently resident in this
    /// seq_id's KV cache. Empty for a never-used or just-hard-reset slot.
    tokens: Vec<LlamaToken>,
    /// The exact text (see `ActiveSequence::prompt_text`/`raw_generated_pieces`)
    /// that `tokens` is the tokenization of. Empty for a never-used or
    /// just-hard-reset slot. Lets `text_prefix_match` detect "this new
    /// request is exactly this slot's text plus more" WITHOUT retokenizing
    /// the shared part — sidesteps BPE re-tokenization instability entirely
    /// for the common continuing-conversation case, rather than trying to
    /// detect/tolerate its consequences after the fact.
    text: String,
    last_used: std::time::Instant,
}

/// A full-sequence-state snapshot captured at a "clean completion" boundary
/// (end of a turn), keyed by the exact token prefix it corresponds to.
///
/// Exists for one specific case `CachedSlot`/`pick_slot` can't handle: an
/// idle slot's *live* resident history is a single, ever-mutating position
/// (whatever this seq_id decoded most recently) — if a later request's
/// matched prefix against it is short (e.g. the resident history moved on
/// to a different conversation, or — the case that motivated this — the
/// prompt re-tokenizes with a shorter common prefix than the true shared
/// content would suggest), `prefill_sequence`'s in-place KV-cache trim to
/// that short prefix is what gets rejected on hybrid/recurrent
/// architectures (their recurrent state can't roll back to an arbitrary
/// earlier position — see `prefill_sequence`). A checkpoint is a
/// *point-in-time copy*, not a live position: restoring one doesn't ask the
/// recurrent state to roll back anything, it just loads a complete,
/// internally-consistent snapshot into a freshly-cleared seq_id, exactly as
/// if that sequence had just finished decoding up to `tokens.len()`.
///
/// A small, bounded count of these (see `SchedulerConfig::ctx_checkpoints`)
/// mirrors llama.cpp server's `server_prompt_checkpoint`
/// (`--ctx-checkpoints`/`--checkpoint-min-step`) rather than the
/// unbounded-memory-cost `n_rs_seq` rollback window — see the README's
/// `--ctx-checkpoints` section for the full rationale.
struct PromptCheckpoint {
    /// The exact resident token vector this snapshot was captured at.
    tokens: Vec<LlamaToken>,
    /// Raw bytes from `state_seq_get_data_ext` — opaque to us, meaningful
    /// only to `state_seq_set_data_ext` on a context loaded with the same
    /// model.
    state: Vec<u8>,
    created_at: std::time::Instant,
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

/// Minimum fraction of a new prompt that must match an idle slot's cached
/// history before we route to it preferentially — mirrors llama.cpp server's
/// default `--slot-prompt-similarity` (0.1).
const SLOT_PROMPT_SIMILARITY: f32 = 0.1;

/// Length of the longest common prefix between two token sequences, compared
/// by token id (never by text — this is what makes it immune to BPE
/// tokenizer merge-boundary drift at the seam between old and new content).
/// Mirrors llama.cpp server's `server_tokens::get_common_prefix`.
fn common_prefix_len(a: &[LlamaToken], b: &[LlamaToken]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// Pick the best idle slot for a newly-tokenized prompt, removing it from
/// the idle pool. Returns `(seq_id, reuse_len)` where `reuse_len` is how many
/// leading tokens of `new_tokens` are already correctly resident in that
/// slot's KV cache and do NOT need to be re-decoded.
///
/// Selection mirrors llama.cpp server's `get_available_slot`: prefer the
/// idle slot with the highest match fraction (`common_prefix_len /
/// new_tokens.len()`) if it clears `SLOT_PROMPT_SIMILARITY` (ties broken by
/// least-recently-used); otherwise fall back to the least-recently-used idle
/// slot regardless of match fraction.
///
/// `reuse_len` is always capped at `new_tokens.len().saturating_sub(1)`: even
/// a 100% cache hit must leave the final prompt token to be freshly decoded,
/// because llama.cpp only produces logits for positions decoded with
/// `logits=true` in THIS call — a cached KV entry alone cannot yield a fresh
/// logit distribution to sample the first reply token from. This also
/// guarantees the "sample from prefill logits at output index -1" convention
/// (exactly one token in the batch has logits enabled) stays correct in
/// every case, including a full cache hit.
fn pick_slot(idle_slots: &mut Vec<CachedSlot>, new_tokens: &[LlamaToken]) -> (i32, usize) {
    debug_assert!(!idle_slots.is_empty(), "pick_slot called with no idle slots");

    let mut best_idx = 0usize;
    let mut best_prefix = 0usize;
    let mut best_sim = -1.0f32;

    for (i, slot) in idle_slots.iter().enumerate() {
        let prefix = common_prefix_len(&slot.tokens, new_tokens);
        let sim = if new_tokens.is_empty() {
            0.0
        } else {
            prefix as f32 / new_tokens.len() as f32
        };

        // Use `total_cmp` (rather than `==`/`>` on `f32` directly) to keep
        // this clippy::float_cmp-clean while still breaking exact ties by
        // least-recently-used.
        let better = match sim.total_cmp(&best_sim) {
            std::cmp::Ordering::Greater => true,
            std::cmp::Ordering::Equal => slot.last_used < idle_slots[best_idx].last_used,
            std::cmp::Ordering::Less => false,
        };
        if better {
            best_idx = i;
            best_prefix = prefix;
            best_sim = sim;
        }
    }

    if best_sim < SLOT_PROMPT_SIMILARITY {
        // No slot clears the similarity threshold — fall back to the
        // least-recently-used idle slot, regardless of match fraction.
        let (lru_idx, _) = idle_slots
            .iter()
            .enumerate()
            .min_by_key(|(_, s)| s.last_used)
            .expect("idle_slots is non-empty (see debug_assert above)");
        best_idx = lru_idx;
        best_prefix = common_prefix_len(&idle_slots[best_idx].tokens, new_tokens);
    }

    let slot = idle_slots.swap_remove(best_idx);
    let reuse_len = best_prefix.min(new_tokens.len().saturating_sub(1));
    (slot.seq_id, reuse_len)
}

/// The idle slot (if any) whose exact resident text is a non-empty prefix
/// of `prompt` — i.e. this request is exactly that slot's conversation
/// continued with more text appended, byte for byte. Picks the LONGEST such
/// match if more than one slot's text qualifies. Returns its index into
/// `idle_slots`.
///
/// Deliberately a plain string comparison, not a tokenization. Prompt
/// templates built from a growing message list (see `chat_template.rs`)
/// are purely deterministic string concatenation — no timestamps, no
/// randomness — so the shared prefix of two prompts from the same
/// conversation is guaranteed byte-identical at the TEXT level. The token
/// level has no such guarantee: retokenizing that same shared text
/// independently, twice (once as it was originally decoded, again as part
/// of a longer growing string), is not guaranteed by BPE to produce the
/// same token ids — merge decisions can be sensitive to what follows.
/// Matching on text and reusing the slot's already-known-correct tokens for
/// the matched portion sidesteps that instability entirely for the common
/// continuing-conversation case, instead of retokenizing the whole prompt
/// every turn and relying on `pick_slot`'s token-level LCP to detect (but
/// not prevent) however much of it came out different this time.
fn text_prefix_match(idle_slots: &[CachedSlot], prompt: &str) -> Option<usize> {
    idle_slots
        .iter()
        .enumerate()
        .filter(|(_, slot)| !slot.text.is_empty() && prompt.starts_with(slot.text.as_str()))
        .max_by_key(|(_, slot)| slot.text.len())
        .map(|(idx, _)| idx)
}

/// The best checkpoint (if any) whose full token vector is an exact prefix
/// of `tokens` — the longest one wins. A checkpoint is a point-in-time
/// snapshot, not diffable, so any divergence within its own span makes it
/// unusable; only an exact, complete prefix match is considered.
fn best_checkpoint<'a>(
    checkpoints: &'a [PromptCheckpoint],
    tokens: &[LlamaToken],
) -> Option<&'a PromptCheckpoint> {
    checkpoints
        .iter()
        .filter(|c| !c.tokens.is_empty() && common_prefix_len(&c.tokens, tokens) == c.tokens.len())
        .max_by_key(|c| c.tokens.len())
}

/// Capture a full-state snapshot of `seq_id`'s current context state, keyed
/// by its exact resident token vector. Checkpointing is a best-effort
/// optimization, never a correctness requirement, so any failure (zero-size
/// state, a short read) is swallowed and reported as `None` rather than
/// propagated — the caller simply skips storing a checkpoint for this turn.
fn take_checkpoint(ctx: &LlamaContext, seq_id: i32, tokens: Vec<LlamaToken>) -> Option<PromptCheckpoint> {
    let flags = llama_cpp_2::LlamaStateSeqFlags::empty();
    let size = ctx.state_seq_get_size_ext(seq_id, flags);
    if size == 0 {
        return None;
    }
    let mut buf = vec![0u8; size];
    // SAFETY: `buf` is exactly `size` bytes, the amount `state_seq_get_size_ext`
    // just reported is needed for this seq_id's current state.
    let written = unsafe { ctx.state_seq_get_data_ext(buf.as_mut_ptr(), seq_id, flags) };
    if written == 0 || written > buf.len() {
        return None;
    }
    buf.truncate(written);
    Some(PromptCheckpoint {
        tokens,
        state: buf,
        created_at: std::time::Instant::now(),
    })
}

/// Called at every "clean completion" point (EOG / stop-sequence / max-tokens
/// reached via a decode this scheduler controlled — never via an error or a
/// disconnect) where the KV/recurrent state for `seq.seq_id` is confirmed
/// correct and safe to keep. Returns the sequence to the idle-slot pool for
/// ordinary longest-common-prefix reuse and, if `--ctx-checkpoints` is
/// enabled and enough new tokens have accrued since the closest existing
/// checkpoint of the same lineage, additionally snapshots the full state so
/// a later request that diverges earlier than this slot's live resident
/// history can restore from here instead of paying for a full re-prefill
/// from position 0 (see `PromptCheckpoint`).
fn finish_sequence_clean(
    ctx: &LlamaContext,
    seq: &ActiveSequence,
    idle_slots: &mut Vec<CachedSlot>,
    checkpoints: &mut Vec<PromptCheckpoint>,
    sched_config: &SchedulerConfig,
) {
    let tokens = resident_tokens(seq);

    if sched_config.ctx_checkpoints > 0 && !tokens.is_empty() {
        let due = match best_checkpoint(checkpoints, &tokens) {
            Some(ancestor) => tokens.len() - ancestor.tokens.len() >= sched_config.checkpoint_min_step as usize,
            None => true,
        };
        if due && let Some(checkpoint) = take_checkpoint(ctx, seq.seq_id, tokens.clone()) {
            if checkpoints.len() >= sched_config.ctx_checkpoints
                && let Some((evict_idx, _)) = checkpoints.iter().enumerate().min_by_key(|(_, c)| c.created_at)
            {
                let evicted = checkpoints.swap_remove(evict_idx);
                tracing::debug!(
                    "Checkpoint pool full — evicting oldest ({} tokens) to make room",
                    evicted.tokens.len(),
                );
            }
            tracing::debug!(
                "Seq {}: checkpointed at {} tokens ({} bytes, {} of {} slots used)",
                seq.seq_id,
                checkpoint.tokens.len(),
                checkpoint.state.len(),
                checkpoints.len() + 1,
                sched_config.ctx_checkpoints,
            );
            checkpoints.push(checkpoint);
        }
    }

    idle_slots.push(CachedSlot {
        seq_id: seq.seq_id,
        tokens,
        text: resident_text(seq),
        last_used: std::time::Instant::now(),
    });
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
    // Pool of idle sequence slots in range [0, max_batch_size), each carrying
    // the token history currently resident in its KV cache so a later
    // request can reuse a matching prefix instead of a full re-prefill.
    // llama.cpp requires seq_id < n_seq_max, so we recycle seq_ids.
    let mut idle_slots: Vec<CachedSlot> = (0..sched_config.max_batch_size as i32)
        .map(|seq_id| CachedSlot {
            seq_id,
            tokens: Vec::new(),
            text: String::new(),
            last_used: std::time::Instant::now(),
        })
        .collect();
    // Bounded pool of full-sequence-state snapshots (see `PromptCheckpoint`).
    // Empty and never grows when `sched_config.ctx_checkpoints == 0`.
    let mut checkpoints: Vec<PromptCheckpoint> = Vec::new();
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
                    let prompt_text = scheduled.request.prompt.clone();

                    // Fast path: if some idle slot's cached text is an exact
                    // (byte-for-byte) prefix of this prompt, reuse its
                    // already-known-correct tokens directly for that part
                    // and tokenize ONLY the new suffix text — never
                    // retokenizing the shared history at all. This sidesteps
                    // BPE re-tokenization instability entirely for the
                    // common continuing-conversation case, instead of
                    // retokenizing the whole growing prompt every turn and
                    // hoping it lands on the same token ids as last time
                    // (see `text_prefix_match`'s doc comment).
                    let fast_path = text_prefix_match(&idle_slots, &prompt_text).and_then(|idx| {
                        let matched_len = idle_slots[idx].text.len();
                        match model.str_to_token(&prompt_text[matched_len..], AddBos::Never) {
                            Ok(suffix_tokens) => {
                                let slot = idle_slots.swap_remove(idx);
                                let base_len = slot.tokens.len();
                                let mut tokens = slot.tokens;
                                tokens.extend(suffix_tokens);
                                let reuse_len = base_len.min(tokens.len().saturating_sub(1));
                                tracing::debug!(
                                    "Seq {}: exact text-prefix match — reusing {base_len} tokens \
                                     without retokenizing, {} new",
                                    slot.seq_id,
                                    tokens.len() - base_len,
                                );
                                Some((tokens, slot.seq_id, reuse_len))
                            }
                            // Suffix tokenization failed (rare) — the slot
                            // was never removed from idle_slots, so this
                            // falls through to the full-tokenize path below
                            // exactly as if there had been no text match.
                            Err(_) => None,
                        }
                    });

                    // Tokenize BEFORE picking a slot: slot selection is
                    // content-addressed (longest-common-prefix against each
                    // idle slot's cached history), so the token ids must be
                    // known first. If tokenization fails, no slot has been
                    // touched yet, so there is nothing to reclaim.
                    let (tokens, seq_id, reuse_len) = match fast_path {
                        Some(result) => result,
                        None => {
                            let bos = if scheduled.request.raw { AddBos::Never } else { AddBos::Always };
                            let tokens = match model.str_to_token(&prompt_text, bos) {
                                Ok(t) => t,
                                Err(e) => {
                                    let _ = scheduled.tx.try_send(StreamEvent::Error(format!(
                                        "Tokenization failed: {e}"
                                    )));
                                    continue;
                                }
                            };

                            // A prompt that tokenizes to zero tokens (e.g. raw:true
                            // with an empty prompt string, AddBos::Never) has no
                            // token to ever produce logits from — reject early
                            // rather than let `tokens.len() - 1` underflow below.
                            if tokens.is_empty() {
                                let _ = scheduled.tx.try_send(StreamEvent::Error(
                                    "Prompt must contain at least one token".into(),
                                ));
                                continue;
                            }

                            if idle_slots.is_empty() {
                                break; // No free slots — should not happen due to active.len() check
                            }
                            let (seq_id, reuse_len) = pick_slot(&mut idle_slots, &tokens);
                            (tokens, seq_id, reuse_len)
                        }
                    };

                    let req = &scheduled.request;
                    let seed = req.seed.unwrap_or_else(|| random_seed_fallback(seq_id as u32));
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
                        prompt_tokens: tokens.clone(),
                        generated_tokens: Vec::new(),
                        prompt_text,
                        raw_generated_pieces: Vec::new(),
                    };

                    // Prefill the unreused suffix of the prompt into the context.
                    //
                    // If a reused (partial) prefill fails, try a bounded
                    // checkpoint restore (see `PromptCheckpoint`) before
                    // falling back to a full fresh prefill (reuse_len=0).
                    // This keeps prefix reuse strictly fallback-safe: whatever
                    // the underlying cause (llama.cpp's llama_decode returns
                    // -1 for several distinct reasons that this binding's
                    // error type collapses into one message, so the exact
                    // cause isn't always visible here), the worst case is
                    // paying for the old, proven full-reprefill behavior —
                    // never a hard failure of the user's request.
                    let mut effective_reuse_len = reuse_len;
                    let mut prefill_result = prefill_sequence(
                        &mut ctx, &config, &scheduled.request, &seq, per_seq_ctx, &tokens, effective_reuse_len,
                    );
                    if let Err(ref e) = prefill_result
                        && effective_reuse_len > 0
                    {
                        tracing::warn!("Seq {}: reused prefill failed ({e})", seq.seq_id);
                        let _ = ctx.clear_kv_cache_seq(Some(seq.seq_id as u32), None, None);
                        effective_reuse_len = 0;

                        // Before giving up to a full re-prefill from position
                        // 0, check whether a checkpoint covers a longer
                        // prefix of this request than the rejected in-place
                        // rollback did. This is exactly the scenario
                        // checkpoints exist for: the live slot's resident
                        // history diverged too early for the (hybrid/
                        // recurrent-incapable) in-place trim above, but an
                        // earlier full-state snapshot of the same lineage
                        // might still cover most of the prompt. Restoring is
                        // best-effort: if the restore call itself fails, or
                        // the retried prefill still errors, this simply falls
                        // through to the reuse_len=0 attempt below exactly as
                        // if no checkpoint had been found.
                        if let Some(checkpoint) = best_checkpoint(&checkpoints, &tokens) {
                            let restore_len = checkpoint.tokens.len();
                            // SAFETY: `checkpoint.state` was produced by this
                            // same scheduler's `take_checkpoint` on this same
                            // context/model earlier in its lifetime.
                            let restored = unsafe {
                                ctx.state_seq_set_data_ext(
                                    &checkpoint.state,
                                    seq.seq_id,
                                    llama_cpp_2::LlamaStateSeqFlags::empty(),
                                )
                            };
                            if restored {
                                tracing::info!(
                                    "Seq {}: restoring from a {restore_len}-token checkpoint instead of a full re-prefill",
                                    seq.seq_id,
                                );
                                effective_reuse_len = restore_len;
                            }
                        }

                        tracing::warn!(
                            "Seq {}: retrying prefill with reuse_len={effective_reuse_len}",
                            seq.seq_id,
                        );
                        prefill_result = prefill_sequence(
                            &mut ctx, &config, &scheduled.request, &seq, per_seq_ctx, &tokens, effective_reuse_len,
                        );
                    }

                    match prefill_result {
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
                                // Clean completion — the KV cache holds exactly
                                // the prompt (this EOG token was never decoded),
                                // so it's safe to keep and offer for reuse.
                                finish_sequence_clean(&ctx, &seq, &mut idle_slots, &mut checkpoints, &sched_config);
                            } else {
                                seq.tokens_generated += 1;
                                seq.generated_tokens.push(token);

                                match model.token_to_piece(token, &mut seq.decoder, true, None) {
                                    Ok(piece) => {
                                        // Mirrors generated_tokens: every decoded piece, unfiltered
                                        // by stop-sequence truncation (see field doc comment).
                                        seq.raw_generated_pieces.push(piece.clone());
                                        match process_piece(&mut seq.pending, &seq.stop_sequences, &seq.filter_sequences, &piece) {
                                            PieceOutcome::Stop(out) => {
                                                if !out.is_empty() {
                                                    let _ = seq.tx.try_send(StreamEvent::Token(out));
                                                }
                                                send_done(&seq);
                                                // Clean completion.
                                                finish_sequence_clean(&ctx, &seq, &mut idle_slots, &mut checkpoints, &sched_config);
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
                                                    // Clean completion.
                                                    finish_sequence_clean(&ctx, &seq, &mut idle_slots, &mut checkpoints, &sched_config);
                                                } else {
                                                    let disconnected = send_or_detect_disconnect(&seq.tx, out, seq.seq_id);
                                                    if disconnected {
                                                        // Cache state is not confirmed-safe (mid
                                                        // first-token piece): full wipe.
                                                        let _ = ctx.clear_kv_cache_seq(Some(seq.seq_id as u32), None, None);
                                                        idle_slots.push(CachedSlot {
                                                            seq_id: seq.seq_id,
                                                            tokens: Vec::new(),
                                                            text: String::new(),
                                                            last_used: std::time::Instant::now(),
                                                        });
                                                    } else {
                                                        seq.last_token = Some(token);
                                                        active.push(seq);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    Err(_) => {
                                        send_done(&seq);
                                        // Decode error — cache state suspect: full wipe.
                                        let _ = ctx.clear_kv_cache_seq(Some(seq.seq_id as u32), None, None);
                                        idle_slots.push(CachedSlot {
                                            seq_id: seq.seq_id,
                                            tokens: Vec::new(),
                                            text: String::new(),
                                            last_used: std::time::Instant::now(),
                                        });
                                    }
                                }
                            }

                            tracing::debug!("Sequence {seq_id} prefilled ({n_tokens} prompt tokens)");
                        }
                        Err(e) => {
                            let _ = seq.tx.try_send(StreamEvent::Error(format!(
                                "Prefill failed: {e}"
                            )));
                            // Prefill may have partially decoded (or trimmed)
                            // this slot's cache before failing — wipe it and
                            // return an empty-history slot, matching the
                            // other error paths' conservative default.
                            let _ = ctx.clear_kv_cache_seq(Some(seq.seq_id as u32), None, None);
                            idle_slots.push(CachedSlot {
                                seq_id: seq.seq_id,
                                tokens: Vec::new(),
                                text: String::new(),
                                last_used: std::time::Instant::now(),
                            });
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
                // Send errors to all active sequences and clear. This is a
                // hard error potentially affecting every active sequence
                // simultaneously — unsafe to assume any partial cache state,
                // so fully wipe each one and return an empty-history slot
                // (also fixes a pre-existing leak: these seq_ids used to be
                // dropped from `active` without ever being returned to the
                // pool).
                for seq in active.drain(..) {
                    let _ = seq.tx.try_send(StreamEvent::Error(format!(
                        "Decode failed: {e}"
                    )));
                    let _ = ctx.clear_kv_cache_seq(Some(seq.seq_id as u32), None, None);
                    idle_slots.push(CachedSlot {
                        seq_id: seq.seq_id,
                        tokens: Vec::new(),
                        text: String::new(),
                        last_used: std::time::Instant::now(),
                    });
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

            // `seq.last_token` was just decoded by this iteration's batch
            // `ctx.decode()` call (step 4, above) and is now physically
            // resident in the KV cache at what was `seq.n_past` — bump it
            // here, unconditionally, before any termination branch below
            // can call `resident_tokens()`. (Previously this increment only
            // happened on the non-terminal fallthrough at the bottom of the
            // loop, which under-counted residency by exactly one token on
            // every EOG/stop/max-tokens completion reached via this loop.)
            seq.n_past += 1;

            let token = seq.sampler.sample(&ctx, logit_idx);
            seq.sampler.accept(token);
            logit_idx += 1;

            // End of generation?
            if model.is_eog_token(token) {
                send_done(seq);
                to_remove.push(i);
                // Clean completion — the KV cache holds exactly seq.n_past
                // resident tokens (this EOG token was never decoded), so
                // it's safe to keep and offer for reuse.
                finish_sequence_clean(&ctx, seq, &mut idle_slots, &mut checkpoints, &sched_config);
                continue;
            }

            seq.tokens_generated += 1;
            seq.generated_tokens.push(token);

            // Decode token to text.
            match model.token_to_piece(token, &mut seq.decoder, true, None) {
                Ok(piece) => {
                    // Mirrors generated_tokens: every decoded piece, unfiltered
                    // by stop-sequence truncation (see field doc comment).
                    seq.raw_generated_pieces.push(piece.clone());
                    match process_piece(&mut seq.pending, &seq.stop_sequences, &seq.filter_sequences, &piece) {
                        PieceOutcome::Stop(out) => {
                            if !out.is_empty() {
                                let _ = seq.tx.try_send(StreamEvent::Token(out));
                            }
                            send_done(seq);
                            to_remove.push(i);
                            // Clean completion.
                            finish_sequence_clean(&ctx, seq, &mut idle_slots, &mut checkpoints, &sched_config);
                            continue;
                        }
                        PieceOutcome::Emit(out) => {
                            let disconnected = send_or_detect_disconnect(&seq.tx, out, seq.seq_id);
                            if disconnected {
                                // Receiver dropped — client disconnected.
                                // Cache state is not confirmed-safe: full wipe.
                                to_remove.push(i);
                                let _ = ctx.clear_kv_cache_seq(Some(seq.seq_id as u32), None, None);
                                idle_slots.push(CachedSlot {
                                    seq_id: seq.seq_id,
                                    tokens: Vec::new(),
                                    text: String::new(),
                                    last_used: std::time::Instant::now(),
                                });
                                continue;
                            }
                        }
                    }
                }
                Err(_) => {
                    send_done(seq);
                    to_remove.push(i);
                    // Decode error — cache state suspect: full wipe.
                    let _ = ctx.clear_kv_cache_seq(Some(seq.seq_id as u32), None, None);
                    idle_slots.push(CachedSlot {
                        seq_id: seq.seq_id,
                        tokens: Vec::new(),
                        text: String::new(),
                        last_used: std::time::Instant::now(),
                    });
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
                // Clean completion.
                finish_sequence_clean(&ctx, seq, &mut idle_slots, &mut checkpoints, &sched_config);
                continue;
            }

            // n_past was already bumped at the top of this iteration (see
            // above) once the decode of this token was confirmed.
            seq.last_token = Some(token);
        }

        // ── 6. Remove completed sequences (reverse order) ──────────────
        // idle_slots already received exactly one CachedSlot per completed
        // sequence above (clean-with-tokens or unsafe-with-empty-tokens) —
        // this step only removes them from `active`.
        to_remove.sort_unstable();
        to_remove.dedup();
        for &i in to_remove.iter().rev() {
            active.swap_remove(i);
        }
    }
}

/// Prefill the unreused suffix of a sequence's prompt tokens into the context.
///
/// `tokens` is the FULL prompt token sequence (already tokenized by the
/// caller) — used for context-budget accounting exactly as before.
/// `reuse_len` is how many of its leading tokens are already resident in
/// `seq.seq_id`'s KV cache from a previous turn and must NOT be re-decoded;
/// only `tokens[reuse_len..]` is actually sent through `ctx.decode`.
///
/// Returns `(prompt_tokens, n_past, effective_max_tokens)`.
fn prefill_sequence(
    ctx: &mut LlamaContext,
    config: &InferenceConfig,
    request: &GenerateRequest,
    seq: &ActiveSequence,
    per_seq_ctx: u32,
    tokens: &[LlamaToken],
    reuse_len: usize,
) -> Result<(u32, i32, u32), String> {
    let n_tokens = tokens.len() as u32;

    // Effective context: per-request num_ctx (clamped to per-sequence limit)
    // or the per-sequence default. This operates on the FULL prompt token
    // count regardless of how much of it is reused — context budget is
    // about total resident positions, not how much was freshly decoded.
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

    let fresh_decoded = tokens.len() - reuse_len;
    tracing::info!(
        "Seq {}: prompt={} tokens, reused {} from cache, decoded {} fresh, max_output={}, effective_ctx={}",
        seq.seq_id,
        n_tokens,
        reuse_len,
        fresh_decoded,
        effective_max_tokens,
        effective_ctx,
    );

    // Trim the chosen slot's stale KV-cache tail before decoding the fresh
    // suffix. Always call this, even when reuse_len == 0: a picked slot can
    // carry a non-empty, UNRELATED history in that case too (e.g. the LRU
    // fallback, or any 1-token prompt where reuse_len is capped to 0
    // regardless of match quality) — skipping the clear would leave a prior,
    // unrelated conversation's KV data attached to this seq_id, silently
    // blended into the new request's generation. Safe in every case:
    // clearing an empty range (a never-used slot) is a no-op.
    //
    // The return value matters for reuse_len > 0: llama.cpp's recurrent
    // memory (Mamba/SSM-style layers, e.g. Qwen3.6's hybrid attention+SSM
    // architecture) can only roll back a bounded number of tokens
    // (`n_rs_seq`, 0 by default — we never set it). A partial trim request
    // beyond that bound returns `false` and leaves the recurrent state
    // untouched, silently stale relative to the trimmed attention KV cache.
    // Treat that as a hard failure of this reuse attempt rather than
    // proceeding on divergent state — the caller retries with reuse_len=0,
    // which requests a full clear (`p0=0, p1=max`) and always succeeds.
    let cleared = ctx
        .clear_kv_cache_seq(Some(seq.seq_id as u32), Some(reuse_len as u32), None)
        .unwrap_or(false);
    if reuse_len > 0 && !cleared {
        return Err(format!(
            "Seq {}: KV-cache rollback to reuse_len={reuse_len} was rejected (likely a \
             recurrent/hybrid model architecture); reuse is unsafe here",
            seq.seq_id,
        ));
    }

    // Prefill in chunks of n_batch tokens. llama.cpp asserts if a single
    // decode call processes more tokens than n_batch, which causes SIGABRT.
    // Long RAG prompts easily exceed the default 2048 n_batch.
    let chunk_size = config.n_batch as usize;
    let last_idx = tokens.len() - 1;

    tracing::debug!(
        "Prefilling seq {} with {} tokens (reused {}) in chunks of {} (context_size={})",
        seq.seq_id,
        tokens.len(),
        reuse_len,
        chunk_size,
        config.context_size,
    );

    for chunk_start in (reuse_len..tokens.len()).step_by(chunk_size) {
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
/// Thin alias over the shared helper so existing call sites stay unchanged.
fn cache_type_bytes_per_elem(ct: &super::KvCacheType) -> f64 {
    super::cache_type_bytes_per_elem(ct)
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

/// Build the token history actually resident in `seq`'s KV cache right now,
/// for caching as a reusable `CachedSlot` on a clean completion.
///
/// `seq.prompt_tokens ++ seq.generated_tokens` can be exactly one token
/// longer than what is truly resident: the token that ends generation (EOG,
/// a matched stop sequence, or a max_tokens truncation) is always sampled
/// from the *previous* decode's logits, and only ever becomes physically
/// resident in the KV cache once it is itself fed through `ctx.decode()` on
/// a later iteration — which never happens when it is the token that ends
/// the turn. `seq.n_past` is advanced only once a token is confirmed
/// decoded, so it is always the true resident count; truncating to it drops
/// that phantom trailing token when present, and is a no-op otherwise (e.g.
/// on EOG, where `generated_tokens` was never extended for the terminating
/// token in the first place).
fn resident_tokens(seq: &ActiveSequence) -> Vec<LlamaToken> {
    let mut full = seq.prompt_tokens.clone();
    full.extend(seq.generated_tokens.iter().copied());
    full.truncate(seq.n_past as usize);
    full
}

/// Text companion to `resident_tokens`: the exact text whose tokenization
/// is `resident_tokens(seq)`, used by `text_prefix_match` to skip
/// retokenizing history that's already known.
///
/// Mirrors `resident_tokens`'s own phantom-trailing-token handling exactly,
/// at the text level: drops the same number of trailing entries from
/// `raw_generated_pieces` (each one token's decoded text, same order as
/// `generated_tokens`) as `resident_tokens` drops from `generated_tokens`,
/// so the two stay a consistent (text, tokens) pair for the same resident
/// content. Falls back to an empty string — never matched by
/// `text_prefix_match` — if the two ever get out of step (defensive; should
/// not happen since both are extended in lockstep by the same call sites).
fn resident_text(seq: &ActiveSequence) -> String {
    let full_len = seq.prompt_tokens.len() + seq.generated_tokens.len();
    let n_past = seq.n_past as usize;
    if n_past < seq.prompt_tokens.len() || n_past > full_len {
        return String::new();
    }
    let extra = full_len - n_past;
    let keep = seq.raw_generated_pieces.len().saturating_sub(extra);
    let mut text = seq.prompt_text.clone();
    for piece in &seq.raw_generated_pieces[..keep] {
        text.push_str(piece);
    }
    text
}

/// Deliver a decoded chunk to the client and report whether it has actually
/// disconnected.
///
/// Checked regardless of whether `out` is empty (the piece may have been
/// fully absorbed into the stop/filter holdback buffer in `process_piece`)
/// — a disconnect must not depend on there being bytes to send. A `Full`
/// channel (a slow-consuming but still-connected client) must NOT be treated
/// as a disconnect: the token is best-effort dropped (logged), but the
/// sequence and its KV cache stay alive. Only a `Closed` channel — the
/// receiver was actually dropped — is a real disconnect.
fn send_or_detect_disconnect(tx: &mpsc::Sender<StreamEvent>, out: String, seq_id: i32) -> bool {
    if out.is_empty() {
        return tx.is_closed();
    }
    match tx.try_send(StreamEvent::Token(out)) {
        Ok(()) => false,
        Err(mpsc::error::TrySendError::Full(_)) => {
            tracing::warn!(
                "Seq {seq_id}: event channel full, dropping a token (client consuming too slowly)",
            );
            false
        }
        Err(mpsc::error::TrySendError::Closed(_)) => true,
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
    use super::{
        best_checkpoint, common_prefix_len, pick_slot, process_piece, send_or_detect_disconnect,
        stop_prefix_holdback, text_prefix_match, CachedSlot, PieceOutcome, PromptCheckpoint,
        StreamEvent,
    };
    use llama_cpp_2::token::LlamaToken;
    use std::time::{Duration, Instant};
    use tokio::sync::mpsc;

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
    fn unknown_channel_role_passes_through_untouched() {
        // Contract (since `<|channel>` / `<channel|>` were removed from the
        // single-delimiter fallback): if the model produces a Harmony channel
        // with a role we have NOT enumerated, the whole `<|channel>ROLE<channel|>`
        // block reaches the client verbatim. The UI can then choose how to
        // surface it (e.g. as a Reasoning box for `thought`). The old fallback
        // was actively harmful for Gemma 4: it stripped `<|channel>` and
        // `<channel|>` separately and left `thought\n[reasoning]` as naked text.
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
        assert_eq!(emitted, "<|channel>banana<channel|>Hello");
    }

    #[test]
    fn filter_works_across_token_boundaries() {
        // Realistic case: a known stray marker (`<|image>`) split across two
        // model pieces still gets scrubbed.
        let stops = gemma_stops();
        let filters = harmony_filters();
        let mut pending = String::new();
        let (emitted, stopped) = drain_with_filters(
            &mut pending,
            &stops,
            &filters,
            &["before <|ima", "ge>after"],
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

    fn toks(ids: &[i32]) -> Vec<LlamaToken> {
        ids.iter().map(|&id| LlamaToken::new(id)).collect()
    }

    #[test]
    fn common_prefix_len_empty_vs_empty() {
        assert_eq!(common_prefix_len(&[], &[]), 0);
    }

    #[test]
    fn common_prefix_len_full_match() {
        let a = toks(&[1, 2, 3]);
        let b = toks(&[1, 2, 3]);
        assert_eq!(common_prefix_len(&a, &b), 3);
    }

    #[test]
    fn common_prefix_len_partial_match() {
        let a = toks(&[1, 2, 3, 4]);
        let b = toks(&[1, 2, 9, 9]);
        assert_eq!(common_prefix_len(&a, &b), 2);
    }

    #[test]
    fn common_prefix_len_totally_disjoint() {
        let a = toks(&[1, 2, 3]);
        let b = toks(&[9, 8, 7]);
        assert_eq!(common_prefix_len(&a, &b), 0);
    }

    #[test]
    fn common_prefix_len_one_shorter_than_other() {
        let a = toks(&[1, 2, 3, 4, 5]);
        let b = toks(&[1, 2, 3]);
        assert_eq!(common_prefix_len(&a, &b), 3);
        assert_eq!(common_prefix_len(&b, &a), 3);
    }

    /// Build a `CachedSlot` whose `last_used` is `age_secs_ago` seconds in
    /// the past — larger values are "more stale" (least recently used).
    fn slot(seq_id: i32, tokens: Vec<LlamaToken>, age_secs_ago: u64) -> CachedSlot {
        CachedSlot {
            seq_id,
            tokens,
            text: String::new(),
            last_used: Instant::now() - Duration::from_secs(age_secs_ago),
        }
    }

    /// Like `slot`, but with resident text set (for `text_prefix_match` tests).
    fn text_slot(seq_id: i32, tokens: Vec<LlamaToken>, text: &str, age_secs_ago: u64) -> CachedSlot {
        CachedSlot {
            seq_id,
            tokens,
            text: text.to_string(),
            last_used: Instant::now() - Duration::from_secs(age_secs_ago),
        }
    }

    #[test]
    fn pick_slot_best_match_above_threshold_wins_regardless_of_order() {
        // Slot 0 is listed first and has a long history, but shares nothing
        // with the new prompt. Slot 1 is listed second, has a shorter
        // history, but is an exact prefix match — it must win even though
        // it is neither first nor the longest.
        let mut idle_slots = vec![
            slot(0, toks(&[100, 101, 102, 103, 104, 105]), 100),
            slot(1, toks(&[1, 2, 3]), 5),
        ];
        let new_tokens = toks(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let (seq_id, reuse_len) = pick_slot(&mut idle_slots, &new_tokens);
        assert_eq!(seq_id, 1);
        assert_eq!(reuse_len, 3);
        // The winning slot must have been removed from the idle pool.
        assert_eq!(idle_slots.len(), 1);
        assert_eq!(idle_slots[0].seq_id, 0);
    }

    #[test]
    fn pick_slot_falls_back_to_lru_when_nothing_clears_threshold() {
        // None of these slots share a meaningful prefix with the new
        // prompt (similarity 0.0 for all — well below the 0.1 threshold),
        // so selection must fall back to the least-recently-used slot.
        let mut idle_slots = vec![
            slot(0, toks(&[900, 901]), 2),  // recently used
            slot(1, toks(&[800, 801]), 50), // least recently used
            slot(2, toks(&[700, 701]), 10),
        ];
        let new_tokens = toks(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let (seq_id, _reuse_len) = pick_slot(&mut idle_slots, &new_tokens);
        assert_eq!(seq_id, 1, "must fall back to the least-recently-used slot");
    }

    #[test]
    fn pick_slot_reuse_len_capped_at_len_minus_one_on_full_match() {
        let full = toks(&[1, 2, 3, 4]);
        let mut idle_slots = vec![slot(0, full.clone(), 1)];
        let (seq_id, reuse_len) = pick_slot(&mut idle_slots, &full);
        assert_eq!(seq_id, 0);
        // Even a 100% cache hit must leave the final prompt token to be
        // freshly decoded, so logits can be sampled from it.
        assert_eq!(reuse_len, full.len() - 1);
    }

    #[test]
    fn pick_slot_empty_new_tokens_does_not_panic() {
        let mut idle_slots = vec![slot(0, toks(&[1, 2, 3]), 5), slot(1, Vec::new(), 10)];
        let (seq_id, reuse_len) = pick_slot(&mut idle_slots, &[]);
        // No panic (no divide-by-zero), and nothing to reuse.
        assert_eq!(reuse_len, 0);
        // Similarity is 0.0 for every slot against an empty prompt, so this
        // falls back to LRU (slot 1, the more stale of the two).
        assert_eq!(seq_id, 1);
    }

    #[test]
    fn text_prefix_match_finds_exact_continuation() {
        let idle_slots = vec![text_slot(0, toks(&[1, 2, 3]), "hello world", 5)];
        let idx = text_prefix_match(&idle_slots, "hello world, how are you?");
        assert_eq!(idx, Some(0));
    }

    #[test]
    fn text_prefix_match_rejects_non_prefix() {
        // Shares a long common substring, but not as a PREFIX of the new
        // prompt — must not match.
        let idle_slots = vec![text_slot(0, toks(&[1, 2, 3]), "hello world", 5)];
        let idx = text_prefix_match(&idle_slots, "well, hello world");
        assert_eq!(idx, None);
    }

    #[test]
    fn text_prefix_match_picks_longest_among_multiple() {
        let idle_slots = vec![
            text_slot(0, toks(&[1]), "hello", 10),
            text_slot(1, toks(&[1, 2]), "hello world", 5),
        ];
        let idx = text_prefix_match(&idle_slots, "hello world, extended further");
        assert_eq!(idx, Some(1));
    }

    #[test]
    fn text_prefix_match_ignores_empty_slot_text() {
        let idle_slots = vec![text_slot(0, Vec::new(), "", 1)];
        assert_eq!(text_prefix_match(&idle_slots, "anything"), None);
    }

    #[test]
    fn text_prefix_match_exact_equal_text_matches() {
        // The new prompt is IDENTICAL to the slot's resident text (a
        // retried/duplicate request) — an empty suffix, still a valid match.
        let idle_slots = vec![text_slot(0, toks(&[1, 2]), "same text", 1)];
        assert_eq!(text_prefix_match(&idle_slots, "same text"), Some(0));
    }

    #[test]
    fn text_prefix_match_none_when_no_slots() {
        assert_eq!(text_prefix_match(&[], "anything"), None);
    }

    /// Build a `PromptCheckpoint` with a given token vector and age (used
    /// only for eviction-order assertions — `take_checkpoint`'s actual
    /// state-byte capture needs a real llama.cpp context and isn't
    /// exercised by these pure-logic tests).
    fn checkpoint(tokens: Vec<LlamaToken>, age_secs_ago: u64) -> PromptCheckpoint {
        PromptCheckpoint {
            tokens,
            state: vec![0u8; 1],
            created_at: Instant::now() - Duration::from_secs(age_secs_ago),
        }
    }

    #[test]
    fn best_checkpoint_picks_longest_exact_prefix_match() {
        let checkpoints = vec![
            checkpoint(toks(&[1, 2, 3]), 10),
            checkpoint(toks(&[1, 2, 3, 4, 5]), 5),
            checkpoint(toks(&[9, 9, 9]), 1), // not a prefix at all
        ];
        let tokens = toks(&[1, 2, 3, 4, 5, 6, 7]);
        let best = best_checkpoint(&checkpoints, &tokens).expect("a matching checkpoint exists");
        assert_eq!(best.tokens, toks(&[1, 2, 3, 4, 5]));
    }

    #[test]
    fn best_checkpoint_rejects_non_prefix_matches() {
        // Shares a long common prefix by token-id comparison, but the
        // checkpoint's own tail (token 99) diverges from the request before
        // the checkpoint ends — not a valid restore point.
        let checkpoints = vec![checkpoint(toks(&[1, 2, 99]), 1)];
        let tokens = toks(&[1, 2, 3, 4]);
        assert!(best_checkpoint(&checkpoints, &tokens).is_none());
    }

    #[test]
    fn best_checkpoint_none_when_no_checkpoints() {
        assert!(best_checkpoint(&[], &toks(&[1, 2, 3])).is_none());
    }

    #[test]
    fn best_checkpoint_ignores_empty_checkpoint() {
        // An empty-token checkpoint would trivially "match" every request
        // (a zero-length prefix always matches) but restoring it would be
        // equivalent to restoring nothing — must never be selected.
        let checkpoints = vec![checkpoint(Vec::new(), 1)];
        assert!(best_checkpoint(&checkpoints, &toks(&[1, 2, 3])).is_none());
    }

    #[test]
    fn full_channel_is_not_treated_as_a_disconnect() {
        let (tx, _rx) = mpsc::channel::<StreamEvent>(1);
        // Fill the one buffered slot without ever draining it — a
        // slow-consuming but still-connected client, not a dropped one.
        tx.try_send(StreamEvent::Token("first".to_string())).unwrap();
        let disconnected = send_or_detect_disconnect(&tx, "second".to_string(), 0);
        assert!(!disconnected, "a Full channel must not be treated as a disconnect");
    }

    #[test]
    fn closed_channel_is_detected_with_pending_output() {
        let (tx, rx) = mpsc::channel::<StreamEvent>(4);
        drop(rx);
        let disconnected = send_or_detect_disconnect(&tx, "hello".to_string(), 0);
        assert!(disconnected, "a Closed channel with bytes to send must be detected");
    }

    #[test]
    fn closed_channel_is_detected_even_with_empty_output() {
        let (tx, rx) = mpsc::channel::<StreamEvent>(4);
        drop(rx);
        // Empty `out` (piece fully absorbed into the stop/filter holdback
        // buffer) must not let a disconnect slip through undetected.
        let disconnected = send_or_detect_disconnect(&tx, String::new(), 0);
        assert!(disconnected, "a Closed channel must be detected even with nothing to send");
    }

    #[test]
    fn open_channel_with_empty_output_is_not_a_disconnect() {
        let (tx, _rx) = mpsc::channel::<StreamEvent>(4);
        let disconnected = send_or_detect_disconnect(&tx, String::new(), 0);
        assert!(!disconnected);
    }
}

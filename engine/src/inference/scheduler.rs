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

use std::collections::VecDeque;
use std::num::NonZeroU32;
use std::pin::pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextType;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use crate::model_tokens::{AddBos, ModelTokens};
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::speculative::{MtpDrafter, MtpSpeculativeParams};
use llama_cpp_2::token::LlamaToken;
use tokio::sync::mpsc;

use super::output::{PieceOutcome, process_piece};
use super::{AnswerStats, GenerateRequest, InferenceConfig, StopReason, StreamEvent};

/// Below this many tokens per slot, a reasoning model routinely runs out of
/// room mid-answer. Not a hard limit — just the threshold at which staying
/// quiet about the division stops being defensible.
const MIN_COMFORTABLE_SEQ_CTX: u32 = 2048;

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
    /// of the README. A conversation's next turn needs none of it: each
    /// slot's own recurrent state, kept just before its prompt's end
    /// (`StateCheckpoint`), covers that whatever this is set to.
    pub ctx_checkpoints: usize,
    /// Minimum number of new tokens since the closest existing checkpoint
    /// of the same lineage before taking another one. Mirrors llama.cpp
    /// server's `--checkpoint-min-step` (default there: 8192). Only
    /// consulted when `ctx_checkpoints > 0`.
    pub checkpoint_min_step: u32,
    /// Enable extra internal diagnostics for the Rust engine layer. Today
    /// this gates exactly one check — `warn_if_logits_corrupt`, a NaN/Inf
    /// scan of every generated token's logits before sampling — added for
    /// issue #140's investigation into unexplained garbage output. `false`
    /// (default) matches upstream llama.cpp behavior exactly: zero added
    /// per-token cost. `true` adds one linear scan over the vocab
    /// (~100-150k f32 values) per token, which is cheap relative to the
    /// forward pass but not free — opt-in via `--rust-debug`, not something
    /// every production request should pay for.
    pub debug_logit_check: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            queue_capacity: 64,
            ctx_checkpoints: 0,
            checkpoint_min_step: 8192,
            debug_logit_check: false,
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
    sampler: super::sampling::RequestSampler,
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
    /// Tokens the MTP head drafted this turn, and how many of them the
    /// model kept (see `mtp_step`); both 0 without `--mtp`.
    mtp_drafted: u32,
    mtp_accepted: u32,
    /// How long reading the prompt took, from `start`; the answer's own time
    /// is what follows it. Set when the prefill succeeds.
    prompt_time: std::time::Duration,
}

/// A prompt read a chunk at a time, when there is more than one slot: in the
/// decode steps of the sequences already answering (see `run_scheduler_loop`,
/// step 3), or between them (step 7) once a step it was part of failed. Its
/// sequence joins `active`, with the answer's first token, once the whole
/// prompt is read.
struct PendingPrefill {
    seq: ActiveSequence,
    /// The next token of `seq.prompt_tokens` to decode; starts past the
    /// prefix reused from the slot.
    cursor: usize,
    /// Whether a chunk that fails may still fall back from the reused prefix
    /// to a checkpoint or to the whole prompt: once, as for a prompt read
    /// whole (`fall_back_from_reuse`).
    may_fall_back: bool,
    /// Read on its own between the steps (step 7), no longer in them: a step
    /// this prompt was part of failed, and on its own a failure is its alone,
    /// with the fall-back above, and not every answer's in the step.
    apart: bool,
    /// From `prefill_setup`: the prompt's length and the answer's cap.
    n_tokens: u32,
    effective_max: u32,
    /// Whether its recurrent state is kept on the way (`StateCheckpoint`):
    /// on a model whose memory cannot be cut back, unless the request asked
    /// for no cache (`cache_prompt: false`).
    keep_states: bool,
}

/// A waiting prompt's part of one decode step (step 3): its tokens from
/// `pending.cursor` to `to`, and the position in the batch of the prompt's
/// last token when this part reaches it, where the answer's first token is
/// sampled from.
struct PromptPart {
    pending: PendingPrefill,
    to: usize,
    logit: Option<i32>,
}

/// Where the scheduler's time goes, summed over ten seconds and logged as one
/// `steps:` line at the end of them, while it works. Only when asked for,
/// with `RUST_LOG=eullm=info,eullm::steps=debug`: otherwise nothing is timed.
///
/// c07 measured EuLLM 11-36% below llama-server with 16 requests at once,
/// and level with it at one. The parts of a step say why: `decode` is the
/// GPU's forward pass, with the waiting prompts read in it, `sample`
/// choosing each sequence's token on the CPU, one after the other, `emit` its
/// text and the reply channel, `prefill` the prompts read outside the steps:
/// with nothing answering, or apart after a step failed. To part the GPU's
/// time from the CPU's,
/// a timed decode waits for the GPU before sampling starts, which sampling
/// did anyway.
struct StepTimes {
    on: bool,
    since: std::time::Instant,
    steps: u64,
    /// Tokens decoded by those steps: one per answering sequence each.
    tokens: u64,
    /// Passes of the model those steps took (see `decode_passes`).
    passes: u64,
    decode: std::time::Duration,
    sample: std::time::Duration,
    emit: std::time::Duration,
    prefill: std::time::Duration,
    prefill_tokens: u64,
    /// Prompt tokens read in the steps themselves, beside the answers'
    /// tokens: their time is the steps' decode time.
    step_prompt_tokens: u64,
    idle: std::time::Duration,
}

/// The target of the `steps:` lines.
const STEPS_TARGET: &str = "eullm::steps";
const STEPS_EVERY: std::time::Duration = std::time::Duration::from_secs(10);

impl StepTimes {
    fn new(on: bool) -> Self {
        Self {
            on,
            since: std::time::Instant::now(),
            steps: 0,
            tokens: 0,
            passes: 0,
            decode: Default::default(),
            sample: Default::default(),
            emit: Default::default(),
            prefill: Default::default(),
            prefill_tokens: 0,
            step_prompt_tokens: 0,
            idle: Default::default(),
        }
    }

    /// Now, when timing; nothing to take the time from otherwise.
    fn start(&self) -> Option<std::time::Instant> {
        self.on.then(std::time::Instant::now)
    }

    /// Log the window and start the next, once it is `STEPS_EVERY` long. A
    /// window in which the scheduler only waited is not logged.
    fn log_if_due(&mut self) {
        if !self.on {
            return;
        }
        let wall = self.since.elapsed();
        if wall < STEPS_EVERY {
            return;
        }
        if self.steps > 0 || self.prefill_tokens > 0 {
            tracing::debug!(target: STEPS_TARGET, "steps: {}", self.summary(wall));
        }
        *self = Self::new(true);
    }

    /// One line: the window, the steps and their tokens, and each part's
    /// share of the window with its time per step or per token.
    fn summary(&self, wall: std::time::Duration) -> String {
        let s = wall.as_secs_f64().max(1e-9);
        let pct = |d: std::time::Duration| 100.0 * d.as_secs_f64() / s;
        let ms_per = |d: std::time::Duration, n: u64| {
            if n == 0 {
                0.0
            } else {
                1e3 * d.as_secs_f64() / n as f64
            }
        };
        let parts = self.decode + self.sample + self.emit + self.prefill + self.idle;
        let other = wall.saturating_sub(parts);
        format!(
            "{:.1} s, {} steps, {:.1} seqs/step, {:.2} passes/step, {:.0} tok/s | \
             decode {:.1}% ({:.2} ms/step, {} prompt tok), \
             sample {:.1}% ({:.3} ms/token), emit {:.1}% ({:.3} ms/token), \
             prefill {:.1}% ({} tok), idle {:.1}%, other {:.1}%",
            s,
            self.steps,
            if self.steps == 0 {
                0.0
            } else {
                self.tokens as f64 / self.steps as f64
            },
            if self.steps == 0 {
                0.0
            } else {
                self.passes as f64 / self.steps as f64
            },
            self.tokens as f64 / s,
            pct(self.decode),
            ms_per(self.decode, self.steps),
            self.step_prompt_tokens,
            pct(self.sample),
            ms_per(self.sample, self.tokens),
            pct(self.emit),
            ms_per(self.emit, self.tokens),
            pct(self.prefill),
            self.prefill_tokens,
            pct(self.idle),
            pct(other),
        )
    }
}

/// How many passes of the model llama.cpp makes for a decode batch,
/// `seq_ids` holding each token's sequence in the order of the batch: one per
/// sequence for the answers' tokens, a run of the same one for a prompt's.
///
/// With one KV cache per sequence (llama.cpp's default with several), the
/// batch is cut by `llama_batch_allocr::split_equal` in sequential mode: a
/// micro-batch starts at the first token not yet taken and takes, in batch
/// order, only the sequences whose ids follow one another (`id == last + 1`),
/// then the same number of tokens from each of them, as many as the one with
/// fewest has and `n_ubatch` allows. Sequences 0, 1, 3, 4 are two passes, and
/// so are 3, 0, 1, 2; a prompt beside one token of each answer is a pass for
/// the answers and the prompt's first token, then passes for the rest of it.
/// Every pass reads all of the model's weights. With one KV cache for all
/// (`unified`) any tokens go together, `n_ubatch` at a time.
pub(crate) fn decode_passes(seq_ids: &[i32], unified: bool, n_ubatch: usize) -> u32 {
    let n_ubatch = n_ubatch.max(1);
    if unified {
        return seq_ids.len().div_ceil(n_ubatch) as u32;
    }
    // Each sequence's tokens, as positions in the batch, and how many of
    // them are taken: a sequence's tokens are taken in batch order.
    let mut of_seq: std::collections::HashMap<i32, (Vec<usize>, usize)> = Default::default();
    for (i, &id) in seq_ids.iter().enumerate() {
        of_seq.entry(id).or_default().0.push(i);
    }
    let mut used = vec![false; seq_ids.len()];
    let mut left = seq_ids.len();
    let mut passes = 0;
    while left > 0 {
        let mut seqs: Vec<i32> = Vec::new();
        for (i, &id) in seq_ids.iter().enumerate() {
            if used[i] || seqs.contains(&id) || seqs.last().is_some_and(|&l| id != l + 1) {
                continue;
            }
            seqs.push(id);
            if seqs.len() > n_ubatch {
                break;
            }
        }
        let mut rounds = 0;
        while seqs.iter().all(|id| of_seq[id].1 < of_seq[id].0.len()) {
            for id in &seqs {
                let (at, taken) = of_seq.get_mut(id).expect("a sequence of the batch");
                used[at[*taken]] = true;
                *taken += 1;
                left -= 1;
            }
            rounds += 1;
            if (rounds + 1) * seqs.len() > n_ubatch {
                break;
            }
        }
        passes += 1;
    }
    passes
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

/// The recurrent state of one slot's sequence after its first `tokens.len()`
/// prompt tokens, taken while a prompt was read, on a model whose memory
/// cannot be cut back to an earlier token (a hybrid model such as Qwen3.5 and
/// Qwen3.6, three layers in four gated delta net, or a recurrent one).
///
/// Why. Such a model keeps one state per sequence, at its last token: a slot
/// that read a conversation's previous turn can be reused only if the next
/// prompt extends what it holds exactly, and a chat template does not. Qwen3's
/// writes the empty thinking block (`<think>\n\n</think>\n\n`) after the
/// assistant's header of the prompt and leaves it out of the turn when it is
/// history: the next prompt shares all of the slot but the prompt's last 4
/// tokens, and the whole conversation was read again, every turn (2,774
/// tokens for 55 new on Qwen3.5-0.8B).
///
/// Only the recurrent part is kept (`LlamaStateSeqFlags::PARTIAL_ONLY`): the
/// attention's cache still holds the earlier tokens and is cut back to
/// `tokens.len()` as any model's is. On Qwen3.5-0.8B that is 20 MiB per
/// checkpoint, whatever the context; a whole sequence's state would be all
/// its keys and values as well. Taken where llama.cpp's server takes them
/// (`--ctx-checkpoints`, its `checkpoint_offsets`): 4 tokens before the
/// prompt's end, and a micro-batch before that (`checkpoint_cuts`).
///
/// Per slot (`StateCheckpoints`), since the attention's cache it goes with
/// is that slot's. Valid while the slot still holds `tokens`: checked
/// against the new prompt and what the slot shares with it when one is
/// restored (`restore_state_checkpoint`), never assumed.
struct StateCheckpoint {
    tokens: Vec<LlamaToken>,
    state: Vec<u8>,
}

/// Every slot's recurrent-state checkpoints, by seq_id; empty for a model
/// whose memory can be cut back (see `StateCheckpoint`).
type StateCheckpoints = Vec<Vec<StateCheckpoint>>;

/// How many checkpoints a slot keeps: the two its last prompt left.
const STATE_CHECKPOINTS_PER_SLOT: usize = 2;

/// How many tokens before a prompt's end its recurrent state is kept: 4, the
/// header of an answer a chat template may write again otherwise (Qwen3's
/// empty thinking block), and a micro-batch more, as llama.cpp's server does.
fn checkpoint_cuts(n_prompt: usize, n_ubatch: usize) -> [usize; 2] {
    [
        n_prompt.saturating_sub(4 + n_ubatch),
        n_prompt.saturating_sub(4),
    ]
}

/// The first place after `from` where a prompt of `n_prompt` tokens stops
/// for a checkpoint, if any: what a chunk of it read from `from` ends at, at
/// most.
fn next_cut(from: usize, n_prompt: usize, n_ubatch: usize) -> Option<usize> {
    checkpoint_cuts(n_prompt, n_ubatch)
        .into_iter()
        .filter(|&cut| cut > from && cut > 0)
        .min()
}

/// Keeps the recurrent state of `seq_id` after `tokens`, its prompt so far,
/// among the slot's checkpoints: the ones of another history go, and the
/// oldest past `STATE_CHECKPOINTS_PER_SLOT`. Best effort, as
/// `take_checkpoint`: a state that cannot be read is no checkpoint.
fn keep_state_checkpoint(
    ctx: &LlamaContext,
    slot: &mut Vec<StateCheckpoint>,
    seq_id: i32,
    tokens: &[LlamaToken],
) {
    let flags = llama_cpp_2::LlamaStateSeqFlags::PARTIAL_ONLY;
    let size = ctx.state_seq_get_size_ext(seq_id, flags);
    if size == 0 {
        return;
    }
    let mut state = vec![0u8; size];
    // SAFETY: `state` is exactly the `size` bytes `state_seq_get_size_ext`
    // just reported for this seq_id's state with these flags.
    let written = unsafe { ctx.state_seq_get_data_ext(state.as_mut_ptr(), seq_id, flags) };
    if written == 0 || written > state.len() {
        return;
    }
    state.truncate(written);
    slot.retain(|c| c.tokens.len() < tokens.len() && tokens.starts_with(&c.tokens));
    if slot.len() >= STATE_CHECKPOINTS_PER_SLOT {
        slot.remove(0);
    }
    tracing::debug!(
        "Seq {seq_id}: kept its recurrent state at {} tokens ({} bytes)",
        tokens.len(),
        state.len(),
    );
    slot.push(StateCheckpoint {
        tokens: tokens.to_vec(),
        state,
    });
}

/// For a slot that cannot cut its memory back to `reuse_len`, the prompt's
/// tokens the slot shares with it: restores the longest of the slot's
/// checkpoints that `tokens` and the slot both still begin with, leaving at
/// least one token to read, and returns its length, where reading the
/// prompt starts. `None` when no checkpoint fits, or its state would not
/// load: then the slot is read from the start, as before.
fn restore_state_checkpoint(
    ctx: &mut LlamaContext,
    slot: &[StateCheckpoint],
    seq_id: i32,
    tokens: &[LlamaToken],
    reuse_len: usize,
) -> Option<usize> {
    let best = slot
        .iter()
        .filter(|c| {
            !c.tokens.is_empty()
                && c.tokens.len() <= reuse_len
                && c.tokens.len() < tokens.len()
                && tokens.starts_with(&c.tokens)
        })
        .max_by_key(|c| c.tokens.len())?;
    // SAFETY: `best.state` came from `keep_state_checkpoint` on this context,
    // with the same flags.
    let restored = unsafe {
        ctx.state_seq_set_data_ext(
            &best.state,
            seq_id,
            llama_cpp_2::LlamaStateSeqFlags::PARTIAL_ONLY,
        )
    };
    if !restored {
        tracing::warn!(
            "Seq {seq_id}: the recurrent state kept at {} tokens did not load",
            best.tokens.len(),
        );
        return None;
    }
    tracing::info!(
        "Seq {seq_id}: restored the recurrent state kept at {} tokens; reading {} of {}",
        best.tokens.len(),
        tokens.len() - best.tokens.len(),
        tokens.len(),
    );
    Some(best.tokens.len())
}

/// The scheduler's loaded model, shared read-only outside the decode thread.
///
/// Exists so `SchedulerHandle::apply_jinja_chat_template` can render the
/// model's GGUF-embedded chat template from API/CLI threads while the decode
/// loop runs — the same pattern upstream llama-server uses, where HTTP
/// threads render prompts against the model concurrently with slot decoding.
/// That is safe because `llama_model` is immutable after load: decoding
/// mutates only the *context*, and template rendering
/// (`common_chat_templates_init` + `apply`) only reads model metadata and
/// vocab strings.
///
/// Lifetime: the decode thread holds the only long-lived `Arc`; everyone
/// else gets a `Weak` and upgrades per call. Model teardown therefore still
/// happens on the scheduler thread when its loop exits (shutdown or swap),
/// exactly as before this type existed — a handle held by an in-flight
/// request cannot pin the old model's VRAM across a swap.
pub struct SharedModel(LlamaModel);

// SAFETY: see the type-level comment — the wrapped llama_model is immutable
// after load and every cross-thread access through this type is read-only.
unsafe impl Send for SharedModel {}
unsafe impl Sync for SharedModel {}

impl std::ops::Deref for SharedModel {
    type Target = LlamaModel;
    fn deref(&self) -> &LlamaModel {
        &self.0
    }
}

/// Handle returned to callers for submitting requests.
#[derive(Clone)]
pub struct SchedulerHandle {
    tx: crossbeam_channel::Sender<ScheduledRequest>,
    /// Weak reference to the decode thread's model, for chat-template
    /// rendering only — see [`SharedModel`] for why weak.
    model: std::sync::Weak<SharedModel>,
    /// Notify the scheduler thread that new work is available.
    notify: Arc<std::sync::Condvar>,
    notify_mutex: Arc<std::sync::Mutex<()>>,
    /// Explicit shutdown flag — checked by the scheduler loop every iteration.
    /// Using AtomicBool because channel disconnect alone is unreliable
    /// (cloned handles in in-flight SlotSnapshots keep the channel open).
    shutdown: Arc<AtomicBool>,
    /// Join handle for the scheduler thread.
    thread: Arc<std::sync::Mutex<Option<std::thread::JoinHandle<()>>>>,
    /// Taken by the decode loop at every turn in tests only: a test holds it
    /// so that the requests it makes meanwhile reach the loop together
    /// (`SchedulerHandle::gate`).
    #[cfg_attr(not(test), allow(dead_code))]
    gate: Arc<std::sync::Mutex<()>>,
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

    /// Render `messages` through the loaded model's own GGUF-embedded Jinja
    /// chat template — the batching-path twin of
    /// [`super::InferenceEngine::apply_jinja_chat_template`], so both doors
    /// onto a loaded model build prompts the same way (found the hard way on
    /// QwQ-32B-Preview: the CLI seeded a system turn and worked while the
    /// web path, hardcoded-ChatML-only with no system turn, answered off
    /// distribution and leaked `<|im_start|>` into the visible reply).
    ///
    /// Returns `None` when the caller should fall back to its hardcoded
    /// per-family template: the GGUF embeds no template, rendering failed,
    /// or the decode thread has already dropped the model (shutdown/swap).
    pub fn apply_jinja_chat_template(
        &self,
        messages: &[(&str, &str)],
        think: bool,
    ) -> Option<super::DynamicChatTemplate> {
        let model = self.model.upgrade()?;
        super::render_jinja_chat_template(&model, messages, think)
    }

    /// Tool-aware twin of [`Self::apply_jinja_chat_template`], fed the
    /// request's own OpenAI-format JSON. See
    /// [`super::render_oai_chat_template`].
    pub fn apply_oai_chat_template(
        &self,
        messages_json: &str,
        tools_json: Option<&str>,
        tool_choice: Option<&str>,
        think: bool,
    ) -> Option<super::DynamicOaiTemplate> {
        let model = self.model.upgrade()?;
        super::render_oai_chat_template(&model, messages_json, tools_json, tool_choice, think)
    }

    /// Queue a request for inference, or say at once why it was not queued.
    /// Returns immediately either way.
    ///
    /// The caller listens on the returned `mpsc::Receiver<StreamEvent>` for
    /// token events. A refusal comes before anything is generated, which is
    /// what lets the HTTP server answer it with a status code instead of
    /// opening a stream only to carry the error (see [`NotQueued`]).
    pub fn try_submit(
        &self,
        request: GenerateRequest,
    ) -> Result<mpsc::Receiver<StreamEvent>, NotQueued> {
        let (tx, rx) = mpsc::channel::<StreamEvent>(256);
        match self.tx.try_send(ScheduledRequest { request, tx }) {
            Ok(()) => {
                // Wake up the scheduler thread.
                let _lock = self.notify_mutex.lock().unwrap();
                self.notify.notify_one();
                Ok(rx)
            }
            Err(crossbeam_channel::TrySendError::Full(_)) => Err(NotQueued::QueueFull),
            // The decode thread has exited: the model was unloaded between
            // this request finding it and reaching it. This used to say the
            // queue was full, which sent people looking for a load problem.
            Err(crossbeam_channel::TrySendError::Disconnected(_)) => Err(NotQueued::Unloaded),
        }
    }

    /// [`Self::try_submit`] for a caller with nothing to do with a refusal
    /// but show it: the refusal arrives as the stream's only event, an
    /// error, as every other failure does.
    pub fn submit(&self, request: GenerateRequest) -> mpsc::Receiver<StreamEvent> {
        self.try_submit(request).unwrap_or_else(|refused| {
            let (tx, rx) = mpsc::channel::<StreamEvent>(1);
            let _ = tx.try_send(StreamEvent::Error(refused.to_string()));
            rx
        })
    }
}

#[cfg(test)]
impl SchedulerHandle {
    /// A handle with no model and no decode thread behind it, and the far
    /// end of its queue of `capacity` places: what a server does with a
    /// full queue, or with a model unloaded under a request (drop the
    /// queue's end), tested without loading a model.
    pub(crate) fn detached(
        capacity: usize,
    ) -> (Self, crossbeam_channel::Receiver<ScheduledRequest>) {
        let (tx, rx) = crossbeam_channel::bounded(capacity);
        let handle = Self {
            tx,
            model: std::sync::Weak::new(),
            notify: Arc::new(std::sync::Condvar::new()),
            notify_mutex: Arc::new(std::sync::Mutex::new(())),
            shutdown: Arc::new(AtomicBool::new(false)),
            thread: Arc::new(std::sync::Mutex::new(None)),
            gate: Arc::new(std::sync::Mutex::new(())),
        };
        (handle, rx)
    }

    /// What the decode loop takes at every turn: while a test holds it, the
    /// loop waits, and the requests made meanwhile are taken together when
    /// it is let go.
    pub(crate) fn gate(&self) -> Arc<std::sync::Mutex<()>> {
        Arc::clone(&self.gate)
    }
}

/// Why [`SchedulerHandle::try_submit`] did not queue a request. Neither is
/// the request's fault, and the same request can succeed when sent again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotQueued {
    /// Every place in the queue (`SchedulerConfig::queue_capacity`) is
    /// taken: it frees up as the running requests finish.
    QueueFull,
    /// The model was unloaded between the request finding it and reaching
    /// its queue. Sent again, the request loads it back.
    Unloaded,
}

impl std::fmt::Display for NotQueued {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::QueueFull => "Scheduler queue full — try again later",
            Self::Unloaded => MODEL_UNLOADED,
        })
    }
}

impl std::error::Error for NotQueued {}

/// What a request gets when the model it was sent to is unloaded before it
/// starts: it was queued, or on its way to the queue.
const MODEL_UNLOADED: &str =
    "The model was unloaded before this request started — send it again to load it back";

/// Info reported by the scheduler after model load.
///
/// `Default` is all zeroes, which the banner treats as "not known": it omits
/// the KV memory line and the trained-context hint rather than printing a
/// confident 0. That is the right reading when a load path could not produce
/// the estimate, and it is why this is a `Default` rather than an `Option`
/// threaded through every field.
#[derive(Default)]
pub struct ModelReadyInfo {
    /// KV cache key memory estimate in MiB.
    pub kv_k_mib: f64,
    /// KV cache value memory estimate in MiB.
    pub kv_v_mib: f64,
    /// The context length the model was trained with. Reported so the banner
    /// can say when the configured window throws away most of the model's
    /// capacity: llama.cpp mentions it once, in a log line buried between two
    /// hundred others, and the flag that fixes it is not named there.
    pub n_ctx_train: u32,
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
    pub fn start(
        self,
        backend: Arc<LlamaBackend>,
    ) -> Result<(SchedulerHandle, ModelReadyInfo), Box<dyn std::error::Error + Send + Sync>> {
        // Validate model exists before spawning the thread.
        if !self.config.model_path.exists() {
            return Err(
                format!("Model file not found: {}", self.config.model_path.display()).into(),
            );
        }

        let (req_tx, req_rx) =
            crossbeam_channel::bounded::<ScheduledRequest>(self.sched_config.queue_capacity);

        let notify = Arc::new(std::sync::Condvar::new());
        let notify_mutex = Arc::new(std::sync::Mutex::new(()));
        let shutdown = Arc::new(AtomicBool::new(false));
        let notify_clone = Arc::clone(&notify);
        let notify_mutex_clone = Arc::clone(&notify_mutex);
        let shutdown_clone = Arc::clone(&shutdown);
        let gate = Arc::new(std::sync::Mutex::new(()));
        let gate_clone = Arc::clone(&gate);

        // Channel for the scheduler thread to signal model load completion.
        // Carries a weak reference to the loaded model alongside the load
        // info, so the handle can render chat templates (see `SharedModel`).
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<
            Result<(ModelReadyInfo, std::sync::Weak<SharedModel>), String>,
        >();

        let config = self.config.clone();
        let sched_config = self.sched_config.clone();

        let join_handle = std::thread::Builder::new()
            .name("eullm-scheduler".into())
            .spawn(move || {
                // Catch panics from Rust code. This won't catch C-level abort()
                // from llama.cpp (see SIGABRT handler in main.rs for that).
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run_scheduler_loop(
                        config,
                        sched_config,
                        req_rx,
                        notify_clone,
                        notify_mutex_clone,
                        shutdown_clone,
                        ready_tx,
                        backend,
                        gate_clone,
                    )
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
                        eprintln!(
                            "[EULLM] The inference engine has stopped. Restart eullm to continue."
                        );
                    }
                }
            })?;

        // Wait for the model to finish loading before returning.
        let (model_info, model) = match ready_rx.recv() {
            Ok(Ok(pair)) => pair,
            Ok(Err(e)) => return Err(e.into()),
            Err(_) => return Err("Scheduler thread exited before model was loaded".into()),
        };

        Ok((
            SchedulerHandle {
                tx: req_tx,
                model,
                notify,
                notify_mutex,
                shutdown,
                thread: Arc::new(std::sync::Mutex::new(Some(join_handle))),
                gate,
            },
            model_info,
        ))
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

    // Deliberately q8_0 for keys, not q4_0. A four-bit *key* cache reliably
    // destroys output — reproduced on Metal, on x86 CPU and on ARM CPU during
    // issue #140, where `--cache-type-k q4_0 --cache-type-v q4_0` turned
    // coherent answers into word salad on every machine it was tried on, while
    // the same machines were fine with q8_0 keys. This message used to suggest
    // exactly that combination.
    let kv_tip = if requested_tokens >= 8192 {
        "\n  Or quantize the KV cache: --cache-type-k q8_0 --cache-type-v q4_0 (cuts KV-cache VRAM by ~2.5×).\
         \n  Avoid q4_0 for keys: it saves little more and degrades output badly."
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
    debug_assert!(
        !idle_slots.is_empty(),
        "pick_slot called with no idle slots"
    );

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
        .filter(|c| {
            // A STRICT prefix: `c.tokens.len() < tokens.len()`, not just `==`.
            // Restoring a checkpoint sets `effective_reuse_len =
            // c.tokens.len()`, and `prefill_sequence` then decodes only
            // `tokens[reuse_len..]`. A checkpoint covering the *entire* prompt
            // leaves that range empty: no decode runs, no fresh logits row is
            // produced, and the caller's `sampler.sample(&ctx, -1)` reads
            // whatever logits the context last held — a different sequence's
            // or a previous turn's distribution — and samples the first reply
            // token from it, silently. Every other reuse path already caps at
            // `len-1` (see `pick_slot`, and the text-prefix fast path) for
            // exactly this reason; requiring a strict prefix here is the
            // checkpoint equivalent — there is always at least one token left
            // to decode, so there is always a genuine logits row to sample.
            // The full-prompt-resubmitted case simply falls through to a
            // reuse_len=0 re-prefill, which is correct.
            !c.tokens.is_empty()
                && c.tokens.len() < tokens.len()
                && common_prefix_len(&c.tokens, tokens) == c.tokens.len()
        })
        .max_by_key(|c| c.tokens.len())
}

/// Capture a full-state snapshot of `seq_id`'s current context state, keyed
/// by its exact resident token vector. Checkpointing is a best-effort
/// optimization, never a correctness requirement, so any failure (zero-size
/// state, a short read) is swallowed and reported as `None` rather than
/// propagated — the caller simply skips storing a checkpoint for this turn.
fn take_checkpoint(
    ctx: &LlamaContext,
    seq_id: i32,
    tokens: Vec<LlamaToken>,
) -> Option<PromptCheckpoint> {
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
            Some(ancestor) => {
                tokens.len() - ancestor.tokens.len() >= sched_config.checkpoint_min_step as usize
            }
            None => true,
        };
        if due && let Some(checkpoint) = take_checkpoint(ctx, seq.seq_id, tokens.clone()) {
            if checkpoints.len() >= sched_config.ctx_checkpoints
                && let Some((evict_idx, _)) = checkpoints
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, c)| c.created_at)
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

/// How many tokens the MTP head drafts per step for this load: `--mtp`,
/// when the model was loaded with MTP layers and the scheduler serves one
/// request at a time; otherwise 0, with the reason to log.
fn mtp_drafts(asked: u32, max_batch_size: usize, n_layer_nextn: u32) -> (u32, Option<String>) {
    if asked == 0 {
        return (0, None);
    }
    if n_layer_nextn == 0 {
        return (
            0,
            Some(format!(
                "--mtp {asked}: this model has no MTP layers (its GGUF was converted \
                 without them; unsloth's *-MTP-GGUF files keep them) — generating \
                 without drafts"
            )),
        );
    }
    if max_batch_size != 1 {
        return (
            0,
            Some(format!(
                "--mtp {asked}: drafting serves one request at a time, and --batch-size \
                 is {max_batch_size} — generating without drafts"
            )),
        );
    }
    (asked, None)
}

/// The MTP drafter, the draft context it runs on, and the batch a step
/// checks its drafts with.
struct MtpState<'m> {
    // First, so it is dropped before the draft context it points at. The
    // target context outlives the whole state: `run_scheduler_loop` declares
    // it first.
    drafter: MtpDrafter,
    /// The MTP head's context. It keeps a memory of its own, which llama.cpp
    /// leaves to the caller to keep in step with the target's: what the
    /// target holds, and nothing past it (see `mtp_step` and
    /// `prefill_sequence`), as llama-server does.
    draft_ctx: LlamaContext<'m>,
    /// The last token and up to `n_max` drafts, every one with logits.
    batch: LlamaBatch<'static>,
}

/// The draft context and drafter for `--mtp n_max`, set up as llama-server
/// sets up `--spec-type draft-mtp`: a context of type MTP on the same model,
/// bound to the target context (`ctx_other`), sized like it, with no
/// recurrent rollback of its own.
fn start_mtp<'m>(
    model: &'m LlamaModel,
    draft_model: Option<&'m LlamaModel>,
    backend: &LlamaBackend,
    config: &InferenceConfig,
    ctx_size: NonZeroU32,
    target: &LlamaContext<'_>,
    n_max: u32,
) -> Result<MtpState<'m>, String> {
    let params = super::build_ctx_params(config, ctx_size)
        .with_context_type(LlamaContextType::Mtp)
        .with_moe_cache_size(0)
        .with_moe_prefetch_slots(0)
        .with_n_rs_seq(0)
        .with_n_seq_max(1)
        .with_n_outputs_max(n_max + 1);
    // The head's own file when there is one, else the model itself.
    let draft_ctx = draft_model
        .unwrap_or(model)
        .new_context_with_ctx_other(backend, params, target)
        .map_err(|e| format!("could not create the MTP draft context: {e}"))?;
    // What the head's own context takes beside the target's: `--fit`
    // reserves an estimate of it (`fit::mtp_reserve_bytes`), and this is the
    // figure to check that estimate against.
    tracing::info!("MTP draft context memory:");
    draft_ctx.memory_breakdown_print();
    let draft = MtpSpeculativeParams {
        n_max: n_max as i32,
        n_min: 0,
        p_min: config.mtp_p_min,
    };
    // SAFETY: `MtpState` declares the drafter before the draft context, so
    // it is dropped first, and the target context outlives the state.
    let drafter = unsafe { MtpDrafter::new(target, &draft_ctx, draft) }
        .map_err(|e| format!("llama.cpp could not start its MTP drafter: {e}"))?;
    Ok(MtpState {
        drafter,
        draft_ctx,
        batch: LlamaBatch::new(n_max as usize + 1, 1),
    })
}

/// What became of a sequence once a sampled token was handed to it.
#[derive(Debug, PartialEq, Eq)]
enum Emitted {
    /// Still generating: the token is its next input.
    Continue,
    /// Done, and already handed back to `idle_slots`.
    Finished,
}

/// A sequence that ended cleanly: its memory is kept for reuse. `trim_kv`
/// drops what the context holds past `seq.n_past` first — drafts an MTP
/// step decoded beyond where the answer ended — so the memory kept holds
/// exactly the tokens recorded for it.
fn finish_clean(
    ctx: &mut LlamaContext,
    seq: &ActiveSequence,
    idle_slots: &mut Vec<CachedSlot>,
    checkpoints: &mut Vec<PromptCheckpoint>,
    sched_config: &SchedulerConfig,
    trim_kv: bool,
) {
    if trim_kv {
        let _ = ctx.clear_kv_cache_seq(Some(seq.seq_id as u32), Some(seq.n_past as u32), None);
    }
    finish_sequence_clean(ctx, seq, idle_slots, checkpoints, sched_config);
}

/// A sequence that ended on an error: its memory is suspect, so it is
/// wiped rather than offered for reuse.
fn finish_wiped(ctx: &mut LlamaContext, seq_id: i32, idle_slots: &mut Vec<CachedSlot>) {
    let _ = ctx.clear_kv_cache_seq(Some(seq_id as u32), None, None);
    idle_slots.push(CachedSlot {
        seq_id,
        tokens: Vec::new(),
        text: String::new(),
        last_used: std::time::Instant::now(),
    });
}

/// After a prefill from a reused prefix failed: the slot's memory is wiped,
/// and the longest checkpoint of this prompt's lineage restored if there is
/// one (see `PromptCheckpoint`). Returns where reading the prompt starts
/// again: the checkpoint's length, or 0.
fn fall_back_from_reuse(
    ctx: &mut LlamaContext,
    checkpoints: &[PromptCheckpoint],
    seq_id: i32,
    tokens: &[LlamaToken],
    error: &str,
) -> usize {
    tracing::warn!("Seq {seq_id}: reused prefill failed ({error})");
    let _ = ctx.clear_kv_cache_seq(Some(seq_id as u32), None, None);
    let mut effective_reuse_len = 0;

    // Before giving up to a full re-prefill from position 0, check whether a
    // checkpoint covers a longer prefix of this request than the rejected
    // in-place rollback did. This is exactly the scenario checkpoints exist
    // for: the live slot's resident history diverged too early for the
    // (hybrid/recurrent-incapable) in-place trim, but an earlier full-state
    // snapshot of the same lineage might still cover most of the prompt.
    // Restoring is best-effort for the *restore call itself*: if
    // `state_seq_set_data_ext` returns false the snapshot is ignored and the
    // prompt is read again from 0. (There is no further retry after that: a
    // decode error then fails the request.) `best_checkpoint` only ever
    // returns a STRICT prefix, so a successful restore always leaves at least
    // one token to decode and therefore a real logits row to sample from.
    if let Some(checkpoint) = best_checkpoint(checkpoints, tokens) {
        let restore_len = checkpoint.tokens.len();
        // SAFETY: `checkpoint.state` was produced by this same scheduler's
        // `take_checkpoint` on this same context/model earlier in its lifetime.
        let restored = unsafe {
            ctx.state_seq_set_data_ext(
                &checkpoint.state,
                seq_id,
                llama_cpp_2::LlamaStateSeqFlags::empty(),
            )
        };
        if restored {
            tracing::info!(
                "Seq {seq_id}: restoring from a {restore_len}-token checkpoint instead of a full re-prefill",
            );
            effective_reuse_len = restore_len;
        }
    }

    tracing::warn!("Seq {seq_id}: retrying prefill with reuse_len={effective_reuse_len}");
    effective_reuse_len
}

/// The answer's first token, sampled from the logits of the prompt's last
/// token once the whole prompt is read, and handed to the sequence like
/// every later one (`emit_token`). Returns the sequence if it goes on.
///
/// `logit` is the prompt's last token's position in the batch that read it:
/// -1, the last output, when the prompt was read on its own, as only that
/// token had logits; its own position when it was read in a decode step
/// beside other sequences' tokens (step 3 of `run_scheduler_loop`).
fn begin_generation(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    mut seq: ActiveSequence,
    logit: i32,
    idle_slots: &mut Vec<CachedSlot>,
    checkpoints: &mut Vec<PromptCheckpoint>,
    sched_config: &SchedulerConfig,
) -> Option<ActiveSequence> {
    if sched_config.debug_logit_check {
        warn_if_logits_corrupt(ctx, logit, seq.seq_id);
    }
    // `sample` also accepts the token into the sampler's history (penalties,
    // grammar): see `build_sampler`.
    let token = seq.sampler.sample(ctx, logit);

    // Always-on O(1) guard: a NaN here means the whole forward pass produced
    // nothing usable, and continuing would stream garbage that reads as a
    // real answer.
    if sampled_token_is_corrupt(ctx, logit, token) {
        tracing::error!(
            "Seq {}: sampled token {} has a NaN/Inf logit — \
             aborting generation instead of emitting garbage. \
             Run with --rust-debug for the full logit scan.",
            seq.seq_id,
            token.0,
        );
        let _ = seq.tx.try_send(StreamEvent::Error(corrupt_logits_error()));
        // The KV cache for this sequence is suspect — whatever produced NaN
        // is in it. Wipe rather than offer it for prefix reuse.
        finish_wiped(ctx, seq.seq_id, idle_slots);
        return None;
    }

    match emit_token(
        model,
        ctx,
        &mut seq,
        token,
        idle_slots,
        checkpoints,
        sched_config,
        false,
    ) {
        Emitted::Finished => None,
        Emitted::Continue => {
            seq.last_token = Some(token);
            Some(seq)
        }
    }
}

/// Hand a sampled token, already accepted by the sequence's sampler, to its
/// sequence: end it on an end-of-generation token, a stop sequence or
/// `max_tokens`, or stream its text. Shared by the plain decode step and the
/// MTP step, which hands over several tokens per decode (`trim_kv`: see
/// `finish_clean`).
#[allow(clippy::too_many_arguments)]
fn emit_token(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    seq: &mut ActiveSequence,
    token: LlamaToken,
    idle_slots: &mut Vec<CachedSlot>,
    checkpoints: &mut Vec<PromptCheckpoint>,
    sched_config: &SchedulerConfig,
    trim_kv: bool,
) -> Emitted {
    // End of generation?
    if model.is_eog_token(token) {
        send_done(seq, StopReason::Stop);
        // Clean completion — the KV cache holds exactly seq.n_past
        // resident tokens (this EOG token was never decoded), so
        // it's safe to keep and offer for reuse.
        finish_clean(ctx, seq, idle_slots, checkpoints, sched_config, trim_kv);
        return Emitted::Finished;
    }

    seq.tokens_generated += 1;
    seq.generated_tokens.push(token);

    // Decode token to text.
    match model.token_to_piece(token, &mut seq.decoder, true, None) {
        Ok(piece) => {
            // Mirrors generated_tokens: every decoded piece, unfiltered
            // by stop-sequence truncation (see field doc comment).
            seq.raw_generated_pieces.push(piece.clone());
            match process_piece(
                &mut seq.pending,
                &seq.stop_sequences,
                &seq.filter_sequences,
                &piece,
            ) {
                PieceOutcome::Stop(out) => {
                    if !out.is_empty() {
                        let _ = seq.tx.try_send(StreamEvent::Token(out));
                    }
                    send_done(seq, StopReason::Stop);
                    // Clean completion.
                    finish_clean(ctx, seq, idle_slots, checkpoints, sched_config, trim_kv);
                    return Emitted::Finished;
                }
                PieceOutcome::Emit(out) => match try_send_piece(&seq.tx, out, seq.seq_id) {
                    SendOutcome::Disconnected => {
                        // Receiver dropped — client disconnected.
                        // Cache state is not confirmed-safe: full wipe.
                        finish_wiped(ctx, seq.seq_id, idle_slots);
                        return Emitted::Finished;
                    }
                    // Slow client: keep the text at the front of the
                    // hold-back buffer instead of dropping it, and
                    // carry on generating.
                    SendOutcome::Backpressure(text) => {
                        seq.pending.insert_str(0, &text);
                    }
                    SendOutcome::Sent => {}
                },
            }
        }
        Err(_) => {
            let _ = seq.tx.try_send(StreamEvent::Error(
                "decode failed mid-generation — the answer is incomplete".to_string(),
            ));
            // Decode error — cache state suspect: full wipe.
            finish_wiped(ctx, seq.seq_id, idle_slots);
            return Emitted::Finished;
        }
    }

    // Check max tokens.
    if seq.tokens_generated >= seq.max_tokens {
        // Truncation (not a stop): flush any held-back tail as real text.
        let tail = std::mem::take(&mut seq.pending);
        if !tail.is_empty() {
            let _ = seq.tx.try_send(StreamEvent::Token(tail));
        }
        send_done(seq, StopReason::Length);
        // Clean completion.
        finish_clean(ctx, seq, idle_slots, checkpoints, sched_config, trim_kv);
        return Emitted::Finished;
    }
    Emitted::Continue
}

/// How many drafts a step keeps, from the token the sampler picked at each
/// position: drafts are kept from the first while the pick there matches,
/// and the pick after the last match — a correction, or one past every
/// draft — is the next step's input. `picks` holds one pick per position
/// up to and including the first mismatch, the way `mtp_step` samples.
fn drafts_kept(drafts: &[LlamaToken], picks: &[LlamaToken]) -> usize {
    drafts
        .iter()
        .zip(picks)
        .take_while(|(draft, pick)| draft == pick)
        .count()
}

/// Test-only: every MTP draft replaced by a token the model never picks, so
/// a test can show that an answer whose every draft is taken back is the
/// answer with real drafts (`api::real_model_tests`).
#[cfg(test)]
pub(crate) static FORCE_WRONG_DRAFTS: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// One step of speculative decoding for the single active sequence: the
/// MTP head drafts up to `n_max` tokens after `seq.last_token`, one decode
/// reads that token and every draft, and the sequence's own sampler picks a
/// token at each position in turn, stopping at the first that differs from
/// the draft there. Every pick is what a plain step would have picked at
/// that position, so the answer is the one decoding without drafts writes;
/// the drafts only decide how many positions one decode settles.
#[allow(clippy::too_many_arguments)]
fn mtp_step(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    mtp: &mut MtpState,
    seq: &mut ActiveSequence,
    per_seq_ctx: u32,
    idle_slots: &mut Vec<CachedSlot>,
    checkpoints: &mut Vec<PromptCheckpoint>,
    sched_config: &SchedulerConfig,
) -> Emitted {
    let Some(last) = seq.last_token else {
        return Emitted::Continue;
    };
    let pos0 = seq.n_past;

    let mut drafts = match mtp.drafter.draft(pos0, last, &resident_tokens(seq)) {
        Ok(drafts) => drafts,
        Err(e) => {
            tracing::warn!(
                "Seq {}: the MTP head could not draft ({e}) — this step decodes one token",
                seq.seq_id
            );
            Vec::new()
        }
    };
    // A non-empty draft must be settled with `accept` before the next one.
    let pending = !drafts.is_empty();
    // Drafts past the context or past `max_tokens` could never be kept.
    let room_ctx = (per_seq_ctx as i32 - pos0 - 1).max(0) as usize;
    let room_tokens = seq
        .max_tokens
        .saturating_sub(seq.tokens_generated.saturating_add(1)) as usize;
    drafts.truncate(room_ctx.min(room_tokens));
    #[cfg(test)]
    if FORCE_WRONG_DRAFTS.load(std::sync::atomic::Ordering::Relaxed) {
        // The vocabulary's last token, a reserved one no answer picks.
        let never = LlamaToken(model.n_vocab() - 1);
        drafts.iter_mut().for_each(|draft| *draft = never);
    }
    // Drafting decoded `last` and the drafts into the draft context; the
    // verify decode below hands it those positions again, read by the
    // target. llama-server drops them first, as here.
    let _ = mtp
        .draft_ctx
        .clear_kv_cache_seq(Some(seq.seq_id as u32), Some(pos0 as u32), None);

    mtp.batch.clear();
    for (j, &token) in std::iter::once(&last).chain(drafts.iter()).enumerate() {
        if let Err(e) = mtp.batch.add(token, pos0 + j as i32, &[seq.seq_id], true) {
            tracing::error!("Seq {}: MTP batch rejected a token ({e})", seq.seq_id);
            let _ = seq
                .tx
                .try_send(StreamEvent::Error(format!("Decode failed: {e}")));
            finish_wiped(ctx, seq.seq_id, idle_slots);
            return Emitted::Finished;
        }
    }
    if let Err(e) = ctx.decode(&mut mtp.batch) {
        tracing::error!("Batch decode failed: {e}");
        let _ = seq
            .tx
            .try_send(StreamEvent::Error(format!("Decode failed: {e}")));
        finish_wiped(ctx, seq.seq_id, idle_slots);
        return Emitted::Finished;
    }
    if let Err(e) = mtp.drafter.process(&mtp.batch) {
        tracing::warn!(
            "Seq {}: the MTP head could not read the step ({e})",
            seq.seq_id
        );
    }

    // The pick at each position, up to the first that differs from its draft.
    let mut picks: Vec<LlamaToken> = Vec::with_capacity(drafts.len() + 1);
    for j in 0..=drafts.len() {
        let idx = j as i32;
        if sched_config.debug_logit_check {
            warn_if_logits_corrupt(ctx, idx, seq.seq_id);
        }
        let token = seq.sampler.sample(ctx, idx);
        // Always-on O(1) guard — see `sampled_token_is_corrupt`.
        if sampled_token_is_corrupt(ctx, idx, token) {
            tracing::error!(
                "Seq {}: sampled token {} has a NaN/Inf logit after {} tokens — \
                 aborting generation instead of emitting garbage. \
                 Run with --rust-debug for the full logit scan.",
                seq.seq_id,
                token.0,
                seq.tokens_generated,
            );
            let _ = seq.tx.try_send(StreamEvent::Error(corrupt_logits_error()));
            finish_wiped(ctx, seq.seq_id, idle_slots);
            return Emitted::Finished;
        }
        picks.push(token);
        // Past an end-of-generation token nothing is read, and a grammar
        // sampler must not be fed tokens after the end it accepted.
        if drafts.get(j) != Some(&token) || model.is_eog_token(token) {
            break;
        }
    }
    let kept = drafts_kept(&drafts, &picks).min(picks.len() - 1);
    seq.mtp_drafted += drafts.len() as u32;
    seq.mtp_accepted += kept as u32;

    // Take back the drafts the model did not keep: the memory then holds
    // `last` and the kept drafts. Within `--mtp` positions of the end, which
    // the recurrent rollback window of a hybrid model was sized for. The
    // draft context read every position of the step, and drops them too.
    let resident = (pos0 + 1 + kept as i32) as u32;
    let _ = mtp
        .draft_ctx
        .clear_kv_cache_seq(Some(seq.seq_id as u32), Some(resident), None);
    if kept < drafts.len()
        && !matches!(
            ctx.clear_kv_cache_seq(Some(seq.seq_id as u32), Some(resident), None),
            Ok(true)
        )
    {
        tracing::error!(
            "Seq {}: the memory refused to take back {} rejected drafts",
            seq.seq_id,
            drafts.len() - kept
        );
        let _ = seq.tx.try_send(StreamEvent::Error(
            "the model's memory could not drop a rejected draft — the answer is incomplete"
                .to_string(),
        ));
        finish_wiped(ctx, seq.seq_id, idle_slots);
        return Emitted::Finished;
    }
    if pending && let Err(e) = mtp.drafter.accept(kept as u16) {
        tracing::warn!(
            "Seq {}: the MTP head could not settle its draft ({e})",
            seq.seq_id
        );
    }

    // Hand the picks over in order, as a plain step would one at a time.
    for (i, &token) in picks.iter().enumerate() {
        // `last` and the drafts before this pick are in memory.
        seq.n_past = pos0 + 1 + i as i32;
        if emit_token(
            model,
            ctx,
            seq,
            token,
            idle_slots,
            checkpoints,
            sched_config,
            true,
        ) == Emitted::Finished
        {
            if seq.mtp_drafted > 0 {
                tracing::info!(
                    "Seq {}: MTP drafted {} tokens, the model kept {} ({:.0}%)",
                    seq.seq_id,
                    seq.mtp_drafted,
                    seq.mtp_accepted,
                    100.0 * f64::from(seq.mtp_accepted) / f64::from(seq.mtp_drafted),
                );
            }
            return Emitted::Finished;
        }
    }
    seq.last_token = picks.last().copied();
    Emitted::Continue
}

// ── Scheduler loop (runs on a dedicated thread) ─────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_scheduler_loop(
    config: InferenceConfig,
    sched_config: SchedulerConfig,
    req_rx: crossbeam_channel::Receiver<ScheduledRequest>,
    notify: Arc<std::sync::Condvar>,
    notify_mutex: Arc<std::sync::Mutex<()>>,
    shutdown: Arc<AtomicBool>,
    ready_tx: std::sync::mpsc::Sender<Result<(ModelReadyInfo, std::sync::Weak<SharedModel>), String>>,
    // Shared across every model this process loads — a generation engine here,
    // an `EmbeddingModel`, and a second scheduler load on model swap all use
    // the SAME `LlamaBackend` instance rather than each creating their own:
    // `LlamaBackend::init()` marks a process-wide `AtomicBool` and returns
    // `BackendAlreadyInitialized` on a second call while the first is still
    // alive, so two independently-initialized backends can never coexist in
    // one process. See `main.rs`, which creates the one instance the whole
    // process shares.
    backend: Arc<LlamaBackend>,
    // Taken at every turn of the loop, in tests only (`SchedulerHandle::gate`).
    #[cfg_attr(not(test), allow(unused_variables))] gate: Arc<std::sync::Mutex<()>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Use the RETURNED value, never `config.gpu_layers` — on a binary with no
    // GPU backend this is 0. See `super::check_gpu_support` for what asking
    // for more did on Intel Macs.
    let gpu_layers = super::check_gpu_support(config.gpu_layers);

    let model_params = if gpu_layers >= 0 {
        LlamaModelParams::default().with_n_gpu_layers(gpu_layers as u32)
    } else {
        LlamaModelParams::default().with_n_gpu_layers(1000)
    };
    // MTP layers are skipped at load unless asked for (`--mtp`).
    let model_params = model_params.with_load_mtp(config.mtp > 0);
    // `--no-mmap`: experts kept in RAM go to the backend's pinned host buffer.
    // Only when asked: llama.cpp's default load mode (auto) maps the file
    // unless a device cannot use mapped memory, and is left to decide.
    let model_params = if config.no_mmap {
        model_params.with_use_mmap(false)
    } else {
        model_params
    };
    // `--split-mode`, `--tensor-split`: llama.cpp's defaults unless given.
    if !config.gpu_split.is_default() {
        tracing::info!("Split across the GPUs: {}", config.gpu_split.describe());
    }
    let model_params = match config.gpu_split.apply(model_params) {
        Ok(params) => params,
        Err(msg) => {
            let _ = ready_tx.send(Err(msg.clone()));
            return Err(msg.into());
        }
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
    let ahead = crate::readahead::start(&config.model_path, config.load_threads);
    let model = LlamaModel::load_from_file(&backend, &config.model_path, &model_params);
    if let Some(ahead) = ahead {
        ahead.finish();
    }
    let model = match model {
        Ok(m) => m,
        Err(e) => {
            let msg = format!("Failed to load model: {e}");
            let _ = ready_tx.send(Err(msg.clone()));
            return Err(msg.into());
        }
    };
    // The decode thread keeps the only strong reference; the handle gets a
    // weak one for chat-template rendering (see `SharedModel`). Declared
    // before `ctx` below so the context — which borrows the model — drops
    // first when the loop exits.
    let model_owner = Arc::new(SharedModel(model));
    let model = &*model_owner;

    // `--mtp-model`: the MTP head in a file of its own, for a model whose GGUF
    // has none. Loaded whole onto the GPU, as llama-server loads a draft model
    // (`-ngld 99`), and declared before `ctx` and `mtp_state` below so that it
    // drops after them: the draft context borrows it.
    let draft_model: Option<LlamaModel> = match (config.mtp > 0, config.mtp_model.as_ref()) {
        (true, Some(path)) => {
            tracing::info!("MTP: loading the head from {}", path.display());
            let params = LlamaModelParams::default()
                .with_n_gpu_layers(1000)
                .with_load_mtp(true);
            let params = pin!(params);
            match LlamaModel::load_from_file(&backend, path, &params) {
                Ok(m) => Some(m),
                Err(e) => {
                    tracing::warn!(
                        "MTP: could not load {}: {e} — generating without drafts",
                        path.display()
                    );
                    None
                }
            }
        }
        (false, Some(_)) => {
            tracing::warn!("--mtp-model has no effect without --mtp");
            None
        }
        _ => None,
    };
    // A head in a file of its own that failed to load leaves nothing to draft with.
    let mtp_layers = match (config.mtp_model.is_some(), draft_model.as_ref()) {
        (true, Some(m)) => m.n_layer_nextn(),
        (true, None) => 0,
        (false, _) => model.n_layer_nextn(),
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
    // `--ctx-size` is the *total* KV budget and every slot gets an equal share,
    // so raising `--batch-size` silently shrinks what each request can hold.
    // `serve` used to default to 8 slots against the 4096 default context,
    // which is 512 tokens per request — a reasoning model spends that before it
    // finishes thinking, and the answer then stops mid-sentence and is reported
    // as `done_reason="length"`. Nothing in that points back at a flag the
    // operator never set, so the default is now 1 on both commands and the
    // concurrency has to be asked for.
    //
    // This warning still matters: the number is now deliberate, but choosing 8
    // without also raising `--ctx-size` produces exactly the same truncation.
    // Say so, with the two numbers that fix it.
    if per_seq_ctx < MIN_COMFORTABLE_SEQ_CTX && sched_config.max_batch_size > 1 {
        tracing::warn!(
            "Each of the {} slots gets only {} tokens of context ({} total ÷ {} slots). \
             Answers longer than that will be cut off and reported with \
             done_reason=\"length\". Either raise --ctx-size to {} or lower --batch-size.",
            sched_config.max_batch_size,
            per_seq_ctx,
            total_ctx,
            sched_config.max_batch_size,
            MIN_COMFORTABLE_SEQ_CTX * sched_config.max_batch_size as u32,
        );
    }
    let ctx_size = NonZeroU32::new(total_ctx).unwrap_or(NonZeroU32::new(4096).unwrap());

    // `--mtp`: drafting needs the model's MTP layers and one request at a
    // time. When it runs, a step decodes the last token and every draft
    // with logits, and may take back up to `mtp_draft` positions: the
    // recurrent rollback window of a hybrid model is raised to that, as
    // llama-server does (`need_n_rs_seq`).
    let (mtp_draft, mtp_off) = mtp_drafts(
        config.mtp,
        sched_config.max_batch_size,
        mtp_layers,
    );
    if let Some(reason) = mtp_off {
        tracing::warn!("{reason}");
    }
    let mut config = config;
    if mtp_draft > 0 {
        config.rs_seq = config.rs_seq.max(mtp_draft);
        tracing::info!("MTP: the model's MTP head drafts up to {mtp_draft} tokens per step");
        if super::few_row_mmq_off_for_mtp() {
            tracing::info!(
                "MTP: CUDA's few-row kernel off, so drafts are checked with one token's numbers \
                 (GGML_CUDA_MMQ_FEW=1 keeps it)"
            );
        }
    }
    let n_outputs_max = (sched_config.max_batch_size as u32).max(mtp_draft + 1);

    let has_quantized_cache = config.cache_type_k != super::KvCacheType::F16
        || config.cache_type_v != super::KvCacheType::F16;

    // n_outputs_max: how many logit rows a single decode can ask for. Left at
    // llama.cpp's default (0 = n_batch = 2048), the load-time worst-case graph
    // reservation sizes the LM-head output as [n_vocab, min(n_ubatch,
    // n_outputs_max)] in the DEVICE compute buffer — ~300 MiB of VRAM for a
    // ~150k-vocab model at n_ubatch 512, reserved for logit rows this
    // scheduler never requests: it reads exactly one logit per sequence per
    // decode step (prefill sets `logits=true` only on each sequence's last
    // token). Found comparing memory-breakdown tables against stock
    // llama-cli on identical settings — its compute buffer was a third of
    // ours (164 vs 507 MiB on a 27B) because its in-graph backend samplers
    // cap the reserve at n_outputs_max_per_seq=1 per sequence, while our
    // host-side sampling path fell into the n_batch-sized default. The true
    // ceiling here is one output per concurrent slot.
    // `--moe-cache auto`: the VRAM the plan left over goes to the cache, measured
    // with the context this scheduler really makes.
    config.moe_cache_bytes = super::grow_moe_cache(&backend, model, &config, |c| {
        super::build_ctx_params(c, ctx_size)
            .with_n_seq_max(sched_config.max_batch_size as u32)
            .with_n_outputs_max(n_outputs_max)
    });
    let ctx_params = super::build_ctx_params(&config, ctx_size)
        .with_n_seq_max(sched_config.max_batch_size as u32)
        .with_n_outputs_max(n_outputs_max);

    let mut ctx = match model.new_context(&backend, ctx_params) {
        Ok(c) => c,
        Err(e) => {
            // Build a hint suggesting smaller ctx-size values to try.
            let hint = ctx_oom_hint(total_ctx);
            if has_quantized_cache {
                // A quantized KV cache the driver would not allocate: retry
                // once at F16/F16, which always fits where the quantized form
                // did not, and say so rather than failing with the user's flags
                // still on screen as if they had been honoured.
                {
                    eprintln!(
                        "[EULLM] KV cache fallback: {:?}/{:?} → F16/F16",
                        config.cache_type_k, config.cache_type_v,
                    );
                    let ctx_params = super::build_ctx_params_with_cache(
                        &config,
                        ctx_size,
                        super::KvCacheType::F16,
                        super::KvCacheType::F16,
                    )
                    .with_n_seq_max(sched_config.max_batch_size as u32)
                    .with_n_outputs_max(n_outputs_max);

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
                let msg = format!("Context allocation failed (likely out of VRAM): {e}\n{hint}");
                let _ = ready_tx.send(Err(msg.clone()));
                return Err(msg.into());
            }
        }
    };

    // llama.cpp's own log lines (CUDA graph warmup etc.) are routed once,
    // process-wide, when the shared backend is created, rather than per-load
    // here: see `inference::init_shared_backend` and `inference::log_filter`.

    // The actual n_ubatch/n_batch llama.cpp applied — read back from the
    // context rather than trusted from what we asked for, since this is
    // exactly the number a VRAM-discrepancy comparison against a stock
    // binary needs and the startup banner never printed it (only n_batch,
    // the logical ceiling, not n_ubatch, the physical micro-batch that
    // actually drives compute-buffer size).
    tracing::info!(
        "Context params applied: n_ctx={}, n_batch={}, n_ubatch={}",
        ctx.n_ctx(),
        ctx.n_batch(),
        ctx.n_ubatch(),
    );

    // The same per-device memory breakdown table stock llama-cli/llama-server
    // print at load time (`total | free | self = model + context + compute`),
    // straight from llama.cpp's own accounting rather than our own estimate —
    // logged here so a real allocation can be compared apples-to-apples
    // against a stock binary's, instead of guessing from two processes' free
    // VRAM numbers.
    ctx.memory_breakdown_print();

    // After `ctx`, so it is dropped first: the drafter points at both.
    let mut mtp_state = if mtp_draft > 0 {
        match start_mtp(
            model,
            draft_model.as_ref(),
            &backend,
            &config,
            ctx_size,
            &ctx,
            mtp_draft,
        ) {
            Ok(state) => Some(state),
            Err(e) => {
                tracing::warn!("MTP: {e} — generating without drafts");
                None
            }
        }
    } else {
        None
    };

    let mut active: Vec<ActiveSequence> = Vec::with_capacity(sched_config.max_batch_size);
    // Prompts waiting to be read, or being read, a chunk at a time in the
    // decode steps (step 3): with more than one slot. With one, and so with
    // the MTP head, a prompt is read whole when its request is taken, as
    // nothing else is answering meanwhile.
    let chunked = sched_config.max_batch_size > 1 && mtp_state.is_none();
    let mut prefilling: VecDeque<PendingPrefill> = VecDeque::new();
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
    // Every slot's recurrent-state checkpoints (`StateCheckpoint`), on a model
    // whose memory cannot be cut back to an earlier token.
    let keeps_states = model.is_hybrid() || model.is_recurrent();
    let mut states: StateCheckpoints = (0..sched_config.max_batch_size)
        .map(|_| Vec::new())
        .collect();
    // Pre-allocate the decode batch once — reused every iteration to avoid
    // repeated malloc/free in the hot decode loop. A step holds the answers'
    // tokens and, with several slots, waiting prompts' up to `n_batch`.
    let step_capacity = if chunked {
        (ctx.n_batch() as usize).max(sched_config.max_batch_size)
    } else {
        sched_config.max_batch_size
    };
    let mut decode_batch = LlamaBatch::new(step_capacity.max(1), 1);
    // (seq_id, position of its token in the current decode batch), rebuilt
    // each iteration in step 3 and read back in step 5. Reused rather than
    // reallocated for the same reason as the batch itself.
    let mut logit_of_seq: Vec<(i32, i32)> = Vec::with_capacity(sched_config.max_batch_size);
    // Every token's sequence in the current decode batch, in its order: what
    // `decode_passes` counts the step's passes of the model from.
    let mut batch_seqs: Vec<i32> = Vec::with_capacity(step_capacity);

    tracing::info!(
        "Scheduler running — max_batch_size={}, queue_capacity={}, total_ctx={}, per_seq_ctx={}",
        sched_config.max_batch_size,
        sched_config.queue_capacity,
        total_ctx,
        per_seq_ctx,
    );

    // Estimate KV cache memory from model dimensions and cache types.
    let kv_info = estimate_kv_memory(
        model,
        total_ctx as u64,
        &config.cache_type_k,
        &config.cache_type_v,
    );

    // Signal that the model is loaded and ready.
    let _ = ready_tx.send(Ok((kv_info, Arc::downgrade(&model_owner))));

    let timed = tracing::enabled!(target: STEPS_TARGET, tracing::Level::DEBUG);
    let mut times = StepTimes::new(timed);

    loop {
        #[cfg(test)]
        drop(gate.lock().unwrap_or_else(|e| e.into_inner()));
        times.log_if_due();

        // ── 0. Check shutdown flag ────────────────────────────────────
        if shutdown.load(Ordering::SeqCst) {
            tracing::info!(
                "Shutdown requested — draining {} active sequences",
                active.len()
            );
            for seq in active.iter().chain(prefilling.iter().map(|p| &p.seq)) {
                let _ = seq
                    .tx
                    .try_send(StreamEvent::Error("Server shutting down".into()));
            }
            // Requests still queued are told too. Left there, they were never
            // answered: the queue keeps its messages for as long as any
            // handle to it is alive, and a request waiting for its answer
            // holds one itself.
            while let Ok(queued) = req_rx.try_recv() {
                let _ = queued
                    .tx
                    .try_send(StreamEvent::Error(MODEL_UNLOADED.into()));
            }
            return Ok(());
        }

        // ── 1. Drain new requests from the queue ────────────────────────
        while active.len() + prefilling.len() < sched_config.max_batch_size {
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
                    //
                    // A request with `cache_prompt: false` reuses nothing:
                    // it skips this and starts its slot from position 0.
                    let cache_prompt = scheduled.request.cache_prompt;
                    let matched = cache_prompt
                        .then(|| text_prefix_match(&idle_slots, &prompt_text))
                        .flatten();
                    let fast_path = matched.and_then(|idx| {
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
                            let bos = if scheduled.request.raw {
                                AddBos::Never
                            } else {
                                AddBos::Always
                            };
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
                            let reuse_len = if cache_prompt { reuse_len } else { 0 };
                            (tokens, seq_id, reuse_len)
                        }
                    };

                    let req = &scheduled.request;
                    // Slot id as the seed fallback: distinct per slot, so two
                    // concurrent unseeded requests cannot collapse onto the
                    // same sequence even in the clock-unusable case.
                    let sampler = super::sampling::build_sampler(model, req, seq_id as u32);

                    let seq = ActiveSequence {
                        seq_id,
                        tx: scheduled.tx,
                        sampler,
                        decoder: encoding_rs::UTF_8.new_decoder(),
                        // The response prefix goes out with the first
                        // piece, as if the model had written it.
                        pending: scheduled.request.response_prefix.clone(),
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
                        mtp_drafted: 0,
                        mtp_accepted: 0,
                        prompt_time: std::time::Duration::ZERO,
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
                    // With several slots the prompt is read a chunk at a time
                    // in the decode steps of the sequences already answering
                    // (step 3), so a long prompt holds none of them up; here
                    // it is only checked and its slot prepared. With one
                    // slot, and so with the MTP head, it is read whole, here.
                    // On a model whose memory cannot be cut back, a slot that
                    // holds more than the prompt shares with it is first
                    // brought back to its longest checkpoint of the prompt
                    // (`StateCheckpoint`), and its recurrent state is kept on
                    // the way through this prompt.
                    let keep_states = keeps_states && cache_prompt;
                    let slot_states = &mut states[seq_id as usize];
                    slot_states.retain(|c| tokens.starts_with(&c.tokens));
                    if chunked {
                        let mut effective_reuse_len = reuse_len;
                        let mut fell_back = false;
                        let mut setup = prefill_setup(
                            &mut ctx,
                            &scheduled.request,
                            &seq,
                            per_seq_ctx,
                            &tokens,
                            effective_reuse_len,
                            None,
                        );
                        if setup.is_err()
                            && effective_reuse_len > 0
                            && keep_states
                            && let Some(at) = restore_state_checkpoint(
                                &mut ctx,
                                slot_states,
                                seq.seq_id,
                                &tokens,
                                effective_reuse_len,
                            )
                        {
                            effective_reuse_len = at;
                            setup = prefill_setup(
                                &mut ctx,
                                &scheduled.request,
                                &seq,
                                per_seq_ctx,
                                &tokens,
                                effective_reuse_len,
                                None,
                            );
                        }
                        if let Err(ref e) = setup
                            && effective_reuse_len > 0
                        {
                            effective_reuse_len = fall_back_from_reuse(
                                &mut ctx,
                                &checkpoints,
                                seq.seq_id,
                                &tokens,
                                e,
                            );
                            fell_back = true;
                            setup = prefill_setup(
                                &mut ctx,
                                &scheduled.request,
                                &seq,
                                per_seq_ctx,
                                &tokens,
                                effective_reuse_len,
                                None,
                            );
                        }
                        match setup {
                            Ok((n_tokens, effective_max)) => {
                                prefilling.push_back(PendingPrefill {
                                    seq,
                                    cursor: effective_reuse_len,
                                    may_fall_back: effective_reuse_len > 0 && !fell_back,
                                    apart: false,
                                    n_tokens,
                                    effective_max,
                                    keep_states,
                                });
                            }
                            Err(e) => {
                                let _ = seq
                                    .tx
                                    .try_send(StreamEvent::Error(format!("Prefill failed: {e}")));
                                finish_wiped(&mut ctx, seq.seq_id, &mut idle_slots);
                            }
                        }
                        continue;
                    }

                    let mut effective_reuse_len = reuse_len;
                    let mut prefill_result = prefill_sequence(
                        &mut ctx,
                        &config,
                        &scheduled.request,
                        &seq,
                        per_seq_ctx,
                        &tokens,
                        effective_reuse_len,
                        mtp_state.as_mut(),
                        keep_states.then_some(&mut *slot_states),
                    );
                    if prefill_result.is_err()
                        && effective_reuse_len > 0
                        && keep_states
                        && let Some(at) = restore_state_checkpoint(
                            &mut ctx,
                            slot_states,
                            seq.seq_id,
                            &tokens,
                            effective_reuse_len,
                        )
                    {
                        effective_reuse_len = at;
                        prefill_result = prefill_sequence(
                            &mut ctx,
                            &config,
                            &scheduled.request,
                            &seq,
                            per_seq_ctx,
                            &tokens,
                            effective_reuse_len,
                            mtp_state.as_mut(),
                            Some(&mut *slot_states),
                        );
                    }
                    if let Err(ref e) = prefill_result
                        && effective_reuse_len > 0
                    {
                        effective_reuse_len =
                            fall_back_from_reuse(&mut ctx, &checkpoints, seq.seq_id, &tokens, e);
                        prefill_result = prefill_sequence(
                            &mut ctx,
                            &config,
                            &scheduled.request,
                            &seq,
                            per_seq_ctx,
                            &tokens,
                            effective_reuse_len,
                            mtp_state.as_mut(),
                            keep_states.then_some(&mut *slot_states),
                        );
                    }

                    match prefill_result {
                        Ok((n_tokens, n_past, effective_max)) => {
                            let mut seq = seq;
                            seq.tokens_prompt = n_tokens;
                            seq.n_past = n_past;
                            seq.max_tokens = effective_max;
                            seq.prefilled = true;
                            seq.prompt_time = seq.start.elapsed();

                            if let Some(seq) = begin_generation(
                                model,
                                &mut ctx,
                                seq,
                                -1,
                                &mut idle_slots,
                                &mut checkpoints,
                                &sched_config,
                            ) {
                                active.push(seq);
                            }

                            tracing::debug!(
                                "Sequence {seq_id} prefilled ({n_tokens} prompt tokens)"
                            );
                        }
                        Err(e) => {
                            let _ = seq
                                .tx
                                .try_send(StreamEvent::Error(format!("Prefill failed: {e}")));
                            // Prefill may have partially decoded (or trimmed)
                            // this slot's cache before failing — wipe it and
                            // return an empty-history slot, matching the
                            // other error paths' conservative default.
                            finish_wiped(&mut ctx, seq.seq_id, &mut idle_slots);
                        }
                    }
                }
                Err(crossbeam_channel::TryRecvError::Empty) => break,
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    tracing::info!("Request channel closed — scheduler shutting down.");
                    // Finish active sequences gracefully.
                    for seq in active.iter().chain(prefilling.iter().map(|p| &p.seq)) {
                        let _ = seq
                            .tx
                            .try_send(StreamEvent::Error("Server shutting down".into()));
                    }
                    return Ok(());
                }
            }
        }

        // ── 2. If nothing active, wait for new work ─────────────────────
        if active.is_empty() && prefilling.is_empty() {
            let lock = notify_mutex.lock().unwrap();
            let t = times.start();
            // Wait with a timeout so we can check for channel disconnect.
            let _ = notify.wait_timeout(lock, std::time::Duration::from_millis(100));
            if let Some(t) = t {
                times.idle += t.elapsed();
            }
            continue;
        }

        // ── 3'. With the MTP head: one speculative step instead ─────────
        if let Some(mtp) = mtp_state.as_mut()
            && active.len() == 1
            && active[0].last_token.is_some()
        {
            let finished = mtp_step(
                model,
                &mut ctx,
                mtp,
                &mut active[0],
                per_seq_ctx,
                &mut idle_slots,
                &mut checkpoints,
                &sched_config,
            ) == Emitted::Finished;
            if finished {
                active.clear();
            }
            continue;
        }

        // ── 3. Build the step: one token per answering sequence, and the
        //       waiting prompts' next tokens ─────────────────────────────
        //
        // `logit_of_seq` records, per seq_id, where in THIS batch that
        // sequence's logits are: the position of its token, which is what
        // llama.cpp's `llama_get_logits_ith` (and so `sample`) takes.
        //
        // Recording it as the token is added, rather than re-counting in
        // step 5, is a correctness requirement, not tidiness. A failed `add`
        // used to only log a warning: the sequence was absent from the batch,
        // yet step 5 still handed it the next index and every sequence after
        // it shifted by one, so each would sample from another conversation's
        // distribution and emit plausible-looking text belonging to someone
        // else. Silent, and impossible to distinguish from a model quality
        // problem after the fact.
        //
        // In slot order: with a KV cache per sequence llama.cpp makes one
        // pass of the model per run of consecutive slots in the order the
        // batch holds them (`decode_passes`), and `active` is in no order
        // once answers end (`swap_remove` in step 6) and slots are taken
        // again: 3, 0, 1, 2 was two passes where 0, 1, 2, 3 is one.
        active.sort_unstable_by_key(|seq| seq.seq_id);

        // The waiting prompts are read in the step itself, oldest first, as
        // much of them as `prompt_budget` gives, in slot order among the
        // answers' tokens. Read between the steps instead (until 0.7.50),
        // sixteen requests arriving together started over sixteen steps,
        // each a pass for the answers and one for a prompt; in the step, the
        // first one reads all of them that fit. With one KV cache for all,
        // the answers' tokens and a micro-batch of prompt are one pass; with
        // a KV cache per sequence, a prompt's slot fills the gap it left in
        // the run of answering slots, and the rest of it takes its passes.
        let mut parts: Vec<PromptPart> = Vec::new();
        if chunked && !prefilling.is_empty() {
            let answering = active.iter().filter(|seq| seq.last_token.is_some()).count();
            let oldest = prefilling
                .iter()
                .find(|p| !p.apart)
                .map_or(0, |p| p.seq.prompt_tokens.len() - p.cursor);
            let mut budget = prompt_budget(
                answering,
                oldest,
                ctx.n_batch() as usize,
                ctx.n_ubatch() as usize,
            );
            let mut waiting = VecDeque::with_capacity(prefilling.len());
            while let Some(pending) = prefilling.pop_front() {
                if pending.seq.tx.is_closed() {
                    // The client went away while its prompt was waiting or
                    // being read: what the slot holds of it is of no use to
                    // anyone.
                    finish_wiped(&mut ctx, pending.seq.seq_id, &mut idle_slots);
                    continue;
                }
                if pending.apart || budget == 0 {
                    waiting.push_back(pending);
                    continue;
                }
                let n_prompt = pending.seq.prompt_tokens.len();
                let mut to = (pending.cursor + budget).min(n_prompt);
                if pending.keep_states
                    && let Some(cut) = next_cut(pending.cursor, n_prompt, ctx.n_ubatch() as usize)
                {
                    to = to.min(cut);
                }
                budget -= to - pending.cursor;
                parts.push(PromptPart {
                    pending,
                    to,
                    logit: None,
                });
            }
            prefilling = waiting;
            parts.sort_unstable_by_key(|part| part.pending.seq.seq_id);
        }

        // ── 4. Decode the step ──────────────────────────────────────────
        let t = times.start();
        let mut decoded = fill_step_batch(
            &mut decode_batch,
            &active,
            &mut parts,
            &mut logit_of_seq,
            &mut batch_seqs,
        )
        .and_then(|()| decode_step(&mut ctx, &mut decode_batch));
        if let Err(e) = &decoded
            && !parts.is_empty()
        {
            // A prompt in the step may be what failed it: a prefix reused
            // from its slot that llama.cpp would not continue, or no room
            // left in a KV cache for all. Read between the steps, such a
            // failure was the prompt's alone, with the fall-back from a
            // reused prefix; so it is again. The step's prompts are read
            // apart from now on (step 7), and the step is decoded again
            // without them. A batch llama.cpp refuses (positions that do not
            // follow what the slot holds, no room in the cache) it refuses
            // before computing any of it: such a step left nothing behind.
            tracing::warn!(
                "A decode step with {} prompts in it failed ({e}): reading them on their own \
                 from now on, and the step again without them",
                parts.len(),
            );
            for part in parts.drain(..) {
                let mut pending = part.pending;
                pending.apart = true;
                prefilling.push_back(pending);
            }
            prefilling.make_contiguous().sort_by_key(|p| p.seq.start);
            decoded = fill_step_batch(
                &mut decode_batch,
                &active,
                &mut parts,
                &mut logit_of_seq,
                &mut batch_seqs,
            )
            .and_then(|()| decode_step(&mut ctx, &mut decode_batch));
        }
        if let Some(t) = t
            && decode_batch.n_tokens() > 0
        {
            // The decode only queues the work on a GPU: wait for it here,
            // as the first sample would, so that its time is the GPU's.
            ctx.synchronize();
            let read: u64 = parts.iter().map(|p| (p.to - p.pending.cursor) as u64).sum();
            if logit_of_seq.is_empty() {
                // Prompts only, with nothing answering: their reading time.
                times.prefill += t.elapsed();
                times.prefill_tokens += read;
            } else {
                times.decode += t.elapsed();
                times.steps += 1;
                times.tokens += logit_of_seq.len() as u64;
                times.step_prompt_tokens += read;
                times.passes +=
                    decode_passes(&batch_seqs, config.kv_unified, ctx.n_ubatch() as usize) as u64;
            }
        }
        if let Err(e) = decoded {
            tracing::error!("Batch decode failed: {e}");
            // Send errors to all active sequences and clear. This is a
            // hard error potentially affecting every active sequence
            // simultaneously — unsafe to assume any partial cache state,
            // so fully wipe each one and return an empty-history slot
            // (also fixes a pre-existing leak: these seq_ids used to be
            // dropped from `active` without ever being returned to the
            // pool). The prompts of the step are not among them: they
            // were set apart above.
            for seq in active.drain(..) {
                let _ = seq
                    .tx
                    .try_send(StreamEvent::Error(format!("Decode failed: {e}")));
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

        // ── 5. Sample one token per sequence, send events ───────────────
        let mut to_remove: Vec<usize> = Vec::new();

        for (i, seq) in active.iter_mut().enumerate() {
            if seq.last_token.is_none() {
                // First decode after prefill — logit is from the prefill batch.
                // We need to sample from the last position set during prefill.
                continue;
            }

            // Read this sequence's own logits position from step 3 rather
            // than assuming it matches iteration order (see the comment there).
            let Some(logit_idx) = logit_of_seq
                .iter()
                .find(|(seq_id, _)| *seq_id == seq.seq_id)
                .map(|(_, idx)| *idx)
            else {
                let _ = seq.tx.try_send(StreamEvent::Error(
                    "Internal scheduler error: sequence was not decoded in this batch".into(),
                ));
                to_remove.push(i);
                let _ = ctx.clear_kv_cache_seq(Some(seq.seq_id as u32), None, None);
                idle_slots.push(CachedSlot {
                    seq_id: seq.seq_id,
                    tokens: Vec::new(),
                    text: String::new(),
                    last_used: std::time::Instant::now(),
                });
                continue;
            };
            // `seq.last_token` was just decoded by this iteration's batch
            // `ctx.decode()` call (step 4, above) and is now physically
            // resident in the KV cache at what was `seq.n_past` — bump it
            // here, unconditionally, before any termination branch below
            // can call `resident_tokens()`. (Previously this increment only
            // happened on the non-terminal fallthrough at the bottom of the
            // loop, which under-counted residency by exactly one token on
            // every EOG/stop/max-tokens completion reached via this loop.)
            seq.n_past += 1;

            if sched_config.debug_logit_check {
                warn_if_logits_corrupt(&ctx, logit_idx, seq.seq_id);
            }
            let t = times.start();
            let token = seq.sampler.sample(&ctx, logit_idx);
            if let Some(t) = t {
                times.sample += t.elapsed();
            }

            // Always-on O(1) guard — see `sampled_token_is_corrupt`.
            if sampled_token_is_corrupt(&ctx, logit_idx, token) {
                tracing::error!(
                    "Seq {}: sampled token {} has a NaN/Inf logit after {} tokens — \
                     aborting generation instead of emitting garbage. \
                     Run with --rust-debug for the full logit scan.",
                    seq.seq_id,
                    token.0,
                    seq.tokens_generated,
                );
                let _ = seq.tx.try_send(StreamEvent::Error(corrupt_logits_error()));
                to_remove.push(i);
                let _ = ctx.clear_kv_cache_seq(Some(seq.seq_id as u32), None, None);
                idle_slots.push(CachedSlot {
                    seq_id: seq.seq_id,
                    text: String::new(),
                    tokens: Vec::new(),
                    last_used: std::time::Instant::now(),
                });
                continue;
            }

            let t = times.start();
            let emitted = emit_token(
                model,
                &mut ctx,
                seq,
                token,
                &mut idle_slots,
                &mut checkpoints,
                &sched_config,
                false,
            );
            if let Some(t) = t {
                times.emit += t.elapsed();
            }
            if emitted == Emitted::Finished {
                to_remove.push(i);
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

        // ── 6'. The prompts read in the step ────────────────────────────
        //
        // Before any other decode: the answer's first token is sampled from
        // this step's logits. A prompt read whole starts its answer, which
        // joins the steps from the next one; the others wait for their next
        // part.
        for part in parts.drain(..) {
            let PromptPart {
                mut pending,
                to,
                logit,
            } = part;
            let seq_id = pending.seq.seq_id;
            let n_prompt = pending.seq.prompt_tokens.len();
            if to < n_prompt {
                if pending.keep_states
                    && checkpoint_cuts(n_prompt, ctx.n_ubatch() as usize).contains(&to)
                {
                    keep_state_checkpoint(
                        &ctx,
                        &mut states[seq_id as usize],
                        seq_id,
                        &pending.seq.prompt_tokens[..to],
                    );
                }
                pending.cursor = to;
                prefilling.push_back(pending);
                continue;
            }
            let Some(logit) = logit else {
                // Unreachable: the part that reaches a prompt's end gave its
                // last token logits (`fill_step_batch`).
                let _ = pending.seq.tx.try_send(StreamEvent::Error(
                    "Internal scheduler error: the prompt's last token had no logits".into(),
                ));
                finish_wiped(&mut ctx, seq_id, &mut idle_slots);
                continue;
            };
            let mut seq = pending.seq;
            seq.tokens_prompt = pending.n_tokens;
            seq.n_past = to as i32;
            seq.max_tokens = pending.effective_max;
            seq.prefilled = true;
            seq.prompt_time = seq.start.elapsed();
            if let Some(seq) = begin_generation(
                model,
                &mut ctx,
                seq,
                logit,
                &mut idle_slots,
                &mut checkpoints,
                &sched_config,
            ) {
                active.push(seq);
            }
            tracing::debug!(
                "Sequence {seq_id} prefilled ({} prompt tokens)",
                pending.n_tokens
            );
        }
        // Oldest first, as step 3 reads them.
        prefilling.make_contiguous().sort_by_key(|p| p.seq.start);

        // ── 7. Read the next chunk of a prompt set apart ────────────────
        //
        // Only a prompt that was in a step that failed (step 4) is read here,
        // on its own: after the step, as all prompts were read until 0.7.50.
        // While others answer the chunk is one micro-batch, so each of their
        // tokens waits for one micro-batch of prompt at most instead of the
        // whole prompt; alone, it is the whole batch, as a prompt read whole.
        // Either way llama.cpp computes it a micro-batch at a time, from the
        // same start, so the prompt is read the same way as
        // `prefill_sequence` reads it.
        if let Some(at) = prefilling.iter().position(|p| p.apart) {
            let mut pending = prefilling.remove(at).expect("a position in the queue");
            let seq_id = pending.seq.seq_id;
            let n_prompt = pending.seq.prompt_tokens.len();
            if pending.seq.tx.is_closed() {
                // The client went away while its prompt was waiting or being
                // read: what the slot holds of it is of no use to anyone.
                finish_wiped(&mut ctx, seq_id, &mut idle_slots);
                continue;
            }
            let chunk_size = if active.is_empty() {
                config.n_batch
            } else {
                ctx.n_ubatch()
            } as usize;
            let mut chunk_end = (pending.cursor + chunk_size).min(n_prompt);
            if pending.keep_states
                && let Some(cut) = next_cut(pending.cursor, n_prompt, ctx.n_ubatch() as usize)
            {
                chunk_end = chunk_end.min(cut);
            }
            let t = times.start();
            let read = prefill_chunk(
                &mut ctx,
                seq_id,
                &pending.seq.prompt_tokens,
                pending.cursor,
                chunk_end,
                None,
            );
            if let Some(t) = t {
                ctx.synchronize();
                times.prefill += t.elapsed();
                times.prefill_tokens += (chunk_end - pending.cursor) as u64;
            }
            match read {
                Ok(()) if chunk_end < n_prompt => {
                    if pending.keep_states
                        && checkpoint_cuts(n_prompt, ctx.n_ubatch() as usize).contains(&chunk_end)
                    {
                        keep_state_checkpoint(
                            &ctx,
                            &mut states[seq_id as usize],
                            seq_id,
                            &pending.seq.prompt_tokens[..chunk_end],
                        );
                    }
                    pending.cursor = chunk_end;
                    prefilling.insert(at, pending);
                }
                Ok(()) => {
                    let mut seq = pending.seq;
                    seq.tokens_prompt = pending.n_tokens;
                    seq.n_past = n_prompt as i32;
                    seq.max_tokens = pending.effective_max;
                    seq.prefilled = true;
                    seq.prompt_time = seq.start.elapsed();
                    if let Some(seq) = begin_generation(
                        model,
                        &mut ctx,
                        seq,
                        -1,
                        &mut idle_slots,
                        &mut checkpoints,
                        &sched_config,
                    ) {
                        active.push(seq);
                    }
                    tracing::debug!(
                        "Sequence {seq_id} prefilled ({} prompt tokens)",
                        pending.n_tokens
                    );
                }
                Err(e) if pending.may_fall_back => {
                    pending.cursor = fall_back_from_reuse(
                        &mut ctx,
                        &checkpoints,
                        seq_id,
                        &pending.seq.prompt_tokens,
                        &e,
                    );
                    pending.may_fall_back = false;
                    prefilling.insert(at, pending);
                }
                Err(e) => {
                    let _ = pending
                        .seq
                        .tx
                        .try_send(StreamEvent::Error(format!("Prefill failed: {e}")));
                    finish_wiped(&mut ctx, seq_id, &mut idle_slots);
                }
            }
        }
    }
}

/// How many prompt tokens a decode step reads beside the tokens of
/// `answering` answers, `oldest` being what is left of the first prompt it
/// reads.
///
/// With some answering, a micro-batch less their tokens: with one KV cache
/// for all the step is then one pass of the model, and each answer's token
/// waits for a micro-batch of prompt at most, as when prompts were read
/// between the steps. Never under half a micro-batch, so that prompts still
/// move with small micro-batches, nor past the batch.
///
/// With none answering, up to a whole batch, as a prompt read on its own,
/// but no further than the end of the micro-batch the first prompt ends in:
/// the prompts after it fill that micro-batch and wait for the next step,
/// rather than make the first wait for them. Eight 209-token prompts
/// arriving together on four CPU cores, read in one batch, all started
/// answering after the eight were read.
fn prompt_budget(answering: usize, oldest: usize, n_batch: usize, n_ubatch: usize) -> usize {
    let n_ubatch = n_ubatch.max(1);
    if answering == 0 {
        return oldest.max(1).next_multiple_of(n_ubatch).min(n_batch);
    }
    n_ubatch
        .saturating_sub(answering)
        .max(n_ubatch / 2)
        .min(n_batch.saturating_sub(answering))
}

/// Fill `batch` with one decode step, in slot order (see `decode_passes`):
/// the last token of every answering sequence of `active` (sorted by slot),
/// with logits, and the tokens of each prompt part (sorted by slot), with
/// logits on a prompt's last token only. Where each answer's logits are goes
/// in `logit_of_seq`, a finishing prompt's in its part, and every token's
/// sequence, in batch order, in `seqs`.
///
/// An answer's token that does not fit is left out and logged: step 5 then
/// fails that sequence rather than hand it another's logits. A prompt part
/// that does not fit fails the step, as a failed decode does. Neither
/// happens: the batch holds `n_batch` tokens, and `prompt_budget` keeps the
/// step within it.
fn fill_step_batch(
    batch: &mut LlamaBatch,
    active: &[ActiveSequence],
    parts: &mut [PromptPart],
    logit_of_seq: &mut Vec<(i32, i32)>,
    seqs: &mut Vec<i32>,
) -> Result<(), String> {
    fn add_answer(
        batch: &mut LlamaBatch,
        seq: &ActiveSequence,
        logit_of_seq: &mut Vec<(i32, i32)>,
        seqs: &mut Vec<i32>,
    ) {
        let Some(token) = seq.last_token else {
            return;
        };
        let at = batch.n_tokens();
        match batch.add(token, seq.n_past, &[seq.seq_id], true) {
            Ok(()) => {
                logit_of_seq.push((seq.seq_id, at));
                seqs.push(seq.seq_id);
            }
            Err(e) => tracing::error!(
                "Seq {}: could not be added to the decode batch ({e}) — failing it \
                 rather than sampling from another sequence's logits",
                seq.seq_id,
            ),
        }
    }

    fn add_part(
        batch: &mut LlamaBatch,
        part: &mut PromptPart,
        seqs: &mut Vec<i32>,
    ) -> Result<(), String> {
        let seq_id = part.pending.seq.seq_id;
        let tokens = &part.pending.seq.prompt_tokens;
        let last = tokens.len() - 1;
        for (pos, token) in tokens
            .iter()
            .enumerate()
            .take(part.to)
            .skip(part.pending.cursor)
        {
            let at = batch.n_tokens();
            batch
                .add(*token, pos as i32, &[seq_id], pos == last)
                .map_err(|e| format!("Failed to add prompt token: {e}"))?;
            seqs.push(seq_id);
            if pos == last {
                part.logit = Some(at);
            }
        }
        Ok(())
    }

    batch.clear();
    logit_of_seq.clear();
    seqs.clear();
    let mut next = 0;
    for seq in active {
        while next < parts.len() && parts[next].pending.seq.seq_id < seq.seq_id {
            add_part(batch, &mut parts[next], seqs)?;
            next += 1;
        }
        add_answer(batch, seq, logit_of_seq, seqs);
    }
    for part in &mut parts[next..] {
        add_part(batch, part, seqs)?;
    }
    Ok(())
}

/// Decode `batch`, when it holds anything.
fn decode_step(ctx: &mut LlamaContext, batch: &mut LlamaBatch) -> Result<(), String> {
    if batch.n_tokens() == 0 {
        return Ok(());
    }
    ctx.decode(batch).map_err(|e| e.to_string())
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
#[allow(clippy::too_many_arguments)]
fn prefill_sequence(
    ctx: &mut LlamaContext,
    config: &InferenceConfig,
    request: &GenerateRequest,
    seq: &ActiveSequence,
    per_seq_ctx: u32,
    tokens: &[LlamaToken],
    reuse_len: usize,
    mut mtp: Option<&mut MtpState<'_>>,
    mut states: Option<&mut Vec<StateCheckpoint>>,
) -> Result<(u32, i32, u32), String> {
    let (n_tokens, effective_max_tokens) = prefill_setup(
        ctx,
        request,
        seq,
        per_seq_ctx,
        tokens,
        reuse_len,
        mtp.as_deref_mut(),
    )?;

    // Prefill in chunks of n_batch tokens. llama.cpp asserts if a single
    // decode call processes more tokens than n_batch, which causes SIGABRT.
    // Long RAG prompts easily exceed the default 2048 n_batch.
    let chunk_size = config.n_batch as usize;

    tracing::debug!(
        "Prefilling seq {} with {} tokens (reused {}) in chunks of {} (context_size={})",
        seq.seq_id,
        tokens.len(),
        reuse_len,
        chunk_size,
        config.context_size,
    );

    // With `states`, a chunk also ends where the recurrent state is kept
    // (`checkpoint_cuts`), and the state is kept there.
    let n_ubatch = ctx.n_ubatch() as usize;
    let mut chunk_start = reuse_len;
    while chunk_start < tokens.len() {
        let cut = states
            .is_some()
            .then(|| next_cut(chunk_start, tokens.len(), n_ubatch))
            .flatten();
        let chunk_end = (chunk_start + chunk_size)
            .min(tokens.len())
            .min(cut.unwrap_or(usize::MAX));
        prefill_chunk(
            ctx,
            seq.seq_id,
            tokens,
            chunk_start,
            chunk_end,
            mtp.as_deref_mut(),
        )?;
        if cut == Some(chunk_end)
            && let Some(states) = states.as_deref_mut()
        {
            keep_state_checkpoint(ctx, states, seq.seq_id, &tokens[..chunk_end]);
        }
        chunk_start = chunk_end;
    }

    // Told once the head has read the prompt: `begin` checks that it did, and
    // run before the chunks it warned of a head that had read nothing on every
    // request.
    if let Some(mtp) = mtp
        && let Err(e) = mtp.drafter.begin(tokens)
    {
        tracing::warn!("Seq {}: the MTP head could not start ({e})", seq.seq_id);
    }

    Ok((n_tokens, tokens.len() as i32, effective_max_tokens))
}

/// What `prefill_sequence` does before decoding the prompt, and the chunked
/// prefill of `run_scheduler_loop` (step 7) does when the request is taken:
/// check the prompt against the context, cap the answer to what is left of
/// it, and drop what the slot holds past the reused prefix.
///
/// Returns `(prompt_tokens, effective_max_tokens)`.
fn prefill_setup(
    ctx: &mut LlamaContext,
    request: &GenerateRequest,
    seq: &ActiveSequence,
    per_seq_ctx: u32,
    tokens: &[LlamaToken],
    reuse_len: usize,
    mtp: Option<&mut MtpState<'_>>,
) -> Result<(u32, u32), String> {
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

    // The MTP head reads every decode of the target: this prompt's chunks
    // here, each speculative step in `mtp_step`. Its own memory keeps what
    // the target keeps of this sequence, up to `reuse_len`.
    if let Some(mtp) = mtp {
        let _ =
            mtp.draft_ctx
                .clear_kv_cache_seq(Some(seq.seq_id as u32), Some(reuse_len as u32), None);
    }

    Ok((n_tokens, effective_max_tokens))
}

/// Decodes `tokens[chunk_start..chunk_end]` of a prompt into sequence
/// `seq_id`, with logits only for the prompt's last token, which is where
/// the answer's first token is sampled from.
fn prefill_chunk(
    ctx: &mut LlamaContext,
    seq_id: i32,
    tokens: &[LlamaToken],
    chunk_start: usize,
    chunk_end: usize,
    mtp: Option<&mut MtpState<'_>>,
) -> Result<(), String> {
    let last_idx = tokens.len() - 1;
    let chunk = &tokens[chunk_start..chunk_end];
    let mut batch = LlamaBatch::new(chunk.len().max(1), 1);

    for (j, token) in chunk.iter().enumerate() {
        let abs_pos = chunk_start + j;
        let is_last = abs_pos == last_idx;
        batch
            .add(*token, abs_pos as i32, &[seq_id], is_last)
            .map_err(|e| format!("Failed to add prompt token: {e}"))?;
    }

    ctx.decode(&mut batch)
        .map_err(|e| format!("Prompt decode failed at chunk {chunk_start}..{chunk_end}: {e}"))?;
    if let Some(mtp) = mtp
        && let Err(e) = mtp.drafter.process(&batch)
    {
        tracing::warn!("Seq {seq_id}: the MTP head could not read the prompt ({e})");
    }
    Ok(())
}

/// Bytes per element for a KV cache type (approximate for quantized types).
/// Thin alias over the shared helper so existing call sites stay unchanged.
fn cache_type_bytes_per_elem(ct: &super::KvCacheType) -> f64 {
    super::cache_type_bytes_per_elem(ct)
}

/// Estimate KV cache memory from model dimensions and cache types.
/// Also called from the sequential engine, which loads without a scheduler
/// and so has no other way to report the same two numbers.
pub(crate) fn estimate_kv_memory(
    model: &LlamaModel,
    n_ctx: u64,
    cache_type_k: &super::KvCacheType,
    cache_type_v: &super::KvCacheType,
) -> ModelReadyInfo {
    // Try to get model dimensions. If any method fails or returns 0,
    // fall back to reporting 0 (unknown).
    let n_embd = model.n_embd() as f64; // c_int
    let n_layer = model.n_layer() as f64; // u32
    let n_head = model.n_head() as f64; // u32
    let n_head_kv = model.n_head_kv() as f64; // u32

    if n_embd <= 0.0 || n_layer <= 0.0 || n_head <= 0.0 || n_head_kv <= 0.0 {
        return ModelReadyInfo {
            kv_k_mib: 0.0,
            kv_v_mib: 0.0,
            n_ctx_train: model.n_ctx_train(),
        };
    }

    // `n_embd / n_head` is the *default* head dimension, not a definition: a
    // model may declare `<arch>.attention.key_length` and `value_length`
    // independently, and several do. Qwen3-0.6B carries n_embd 1024 over 16
    // heads but a key length of 128, so the division gives 64 and the estimate
    // came out at exactly half of what llama.cpp then allocated (112 MiB
    // reported against 224 MiB actual, spotted in a user's startup log).
    // Understating is the harmful direction: it invites a context size that
    // does not fit. Read the declared lengths and fall back to the division,
    // which is what the format says the default is.
    let default_head_dim = n_embd / n_head;
    let declared = |suffix: &str| -> Option<f64> {
        let arch = model.meta_val_str("general.architecture").ok()?;
        let v = model
            .meta_val_str(&format!("{arch}.attention.{suffix}"))
            .ok()?;
        v.trim().parse::<f64>().ok().filter(|d| *d > 0.0)
    };
    let head_dim_k = declared("key_length").unwrap_or(default_head_dim);
    let head_dim_v = declared("value_length").unwrap_or(default_head_dim);

    // Elements per K or V = n_layer × n_ctx × n_head_kv × head_dim
    let elements = |head_dim: f64| n_layer * n_ctx as f64 * n_head_kv * head_dim;

    let kv_k_bytes = elements(head_dim_k) * cache_type_bytes_per_elem(cache_type_k);
    let kv_v_bytes = elements(head_dim_v) * cache_type_bytes_per_elem(cache_type_v);

    ModelReadyInfo {
        kv_k_mib: kv_k_bytes / (1024.0 * 1024.0),
        kv_v_mib: kv_v_bytes / (1024.0 * 1024.0),
        n_ctx_train: model.n_ctx_train(),
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

/// What happened when a decoded chunk was handed to the client's channel.
enum SendOutcome {
    /// Delivered (or there was nothing to deliver and the client is still there).
    Sent,
    /// The channel is full — a connected but slow-consuming client. The text
    /// was NOT delivered and is returned so the caller can hold onto it.
    Backpressure(String),
    /// The receiver was dropped: the client is gone.
    Disconnected,
}

/// Deliver a decoded chunk to the client's channel and report what happened.
///
/// Checked regardless of whether `out` is empty (the piece may have been fully
/// absorbed into the stop/filter holdback buffer in `process_piece`) — a
/// disconnect must not depend on there being bytes to send.
///
/// A `Full` channel is not a disconnect and must not end the sequence, but it
/// also must not lose the text: this used to log a warning and **drop the
/// token**, so a client that read slowly received a reply with pieces silently
/// missing from the middle and a final token count that didn't match what it
/// had been given. The text now comes back to the caller, which returns it to
/// the hold-back buffer to be flushed on a later iteration.
fn try_send_piece(tx: &mpsc::Sender<StreamEvent>, out: String, seq_id: i32) -> SendOutcome {
    if out.is_empty() {
        return if tx.is_closed() {
            SendOutcome::Disconnected
        } else {
            SendOutcome::Sent
        };
    }
    match tx.try_send(StreamEvent::Token(out)) {
        Ok(()) => SendOutcome::Sent,
        Err(mpsc::error::TrySendError::Full(StreamEvent::Token(text))) => {
            tracing::debug!(
                "Seq {seq_id}: event channel full, holding {} bytes for the next iteration \
                 (client consuming slowly)",
                text.len(),
            );
            SendOutcome::Backpressure(text)
        }
        // `try_send` gives back exactly the event that was passed in, so the
        // other variants are unreachable; treat anything unexpected as a
        // delivery we can't account for rather than silently losing it.
        Err(mpsc::error::TrySendError::Full(_)) => SendOutcome::Sent,
        Err(mpsc::error::TrySendError::Closed(_)) => SendOutcome::Disconnected,
    }
}

/// Send a `StreamEvent::Done` to the sequence's channel.
/// Tell the client generation ended, and why.
///
/// The reason is passed in rather than inferred here: the call sites know
/// whether they got an end-of-generation token, matched a stop sequence, or
/// simply ran out of budget, and inferring it after the fact from
/// `tokens_generated >= max_tokens` would misreport an EOS that happens to
/// land exactly on the last allowed token.
fn send_done(seq: &ActiveSequence, stop_reason: StopReason) {
    let elapsed = seq.start.elapsed();
    let _ = seq.tx.try_send(StreamEvent::Done {
        tokens_generated: seq.tokens_generated,
        tokens_prompt: seq.tokens_prompt,
        duration_ms: elapsed.as_millis() as u64,
        stop_reason,
        stats: AnswerStats {
            prompt_time: seq.prompt_time,
            eval_time: elapsed.saturating_sub(seq.prompt_time),
            draft_n: seq.mtp_drafted,
            draft_n_accepted: seq.mtp_accepted,
        },
    });
}

/// Scan a token's logits for NaN/Inf right before sampling from them, and
/// log loudly if found. This is the only way today to positively confirm
/// whether "garbage" output (e.g. one token repeated hundreds of times) is
/// caused by NaN/Inf collapsing the sampler onto a fixed token, versus some
/// other cause — no NaN/Inf check existed anywhere in the engine before
/// this. Added after issue #140's investigation into unexplained garbage
/// output on specific x86_64 hardware hit exactly this wall: no way to
/// confirm or rule out numerical corruption from the outside, only from
/// user-visible symptoms downstream of it.
///
/// Cheap relative to the forward pass that produced these logits (a linear
/// scan over the vocab, typically ~100-150k f32 values, versus the
/// billions of FLOPs in the matmuls that computed them) — but not free,
/// and it is pure overhead pure llama.cpp does not pay. Gated behind
/// `SchedulerConfig::debug_logit_check` (`--rust-debug`), off by default:
/// callers must opt in rather than every production request paying an
/// extra per-token vocab scan for a diagnostic almost nobody needs.
/// Whether the token just sampled came out of a broken distribution.
///
/// This is the always-on counterpart to [`warn_if_logits_corrupt`], which scans
/// all ~150k logits and is therefore behind `--rust-debug`. Here we look at a
/// single value: the logit of the token the sampler actually chose. That is
/// O(1), costs nothing per token, and is exactly the right question — if the
/// chosen token's own logit is NaN then the comparison that chose it was
/// meaningless, whatever the rest of the tensor looks like.
///
/// # Why this needs to exist
///
/// On a 2018 Intel Mac in issue #140, every one of the 151,936 logits comes
/// back NaN, from the first token of the first request. The sampler then picks
/// the same token every time — the comparisons all degenerate the same way —
/// and the engine cheerfully streamed a thousand tokens of `@` to the client
/// and reported `done_reason: "stop"`. Three weeks of reports read that as the
/// model misbehaving.
///
/// This does not repair the NaN; nothing here can. It stops us handing the
/// caller something that looks like an answer when we already know the
/// computation failed. The same reasoning as reporting `length` instead of
/// `stop`: the value is in not lying, and on a machine where the corruption is
/// intermittent rather than total, it is the only thing standing between a
/// corrupt answer and a RAG index.
fn sampled_token_is_corrupt(ctx: &LlamaContext, idx: i32, token: LlamaToken) -> bool {
    // `get_logits_ith` panics when `idx` is not in the context's
    // initialized-logits set, and unlike `warn_if_logits_corrupt` this runs on
    // every token of every request rather than behind `--rust-debug`. Two
    // reasons that is safe here, both worth stating because a panic on this
    // path would kill the scheduler thread for every concurrent request:
    //
    // 1. `idx` is the output index recorded when this token's slot was added
    //    with `decode_batch.add(..., true)` — the very call that registers the
    //    index in `LlamaBatch::initialized_logits`, which `decode()` then
    //    transfers to the context. It is in the set by construction.
    // 2. `LlamaSampler::sample(&ctx, idx)`, one line above, has already read
    //    that same row through the raw C API. If the index were wrong we would
    //    be sampling from the wrong sequence's logits regardless, so the
    //    assertion could only ever surface a bug that had already bitten.
    //
    // The `-1` sentinel used on the prefill path is *not* a member of that set
    // — see `warn_if_logits_corrupt` — so it goes through `get_logits()`.
    let logits = if idx == -1 {
        ctx.get_logits()
    } else {
        ctx.get_logits_ith(idx)
    };
    logits
        .get(token.0 as usize)
        .is_some_and(|v| v.is_nan() || v.is_infinite())
}

/// The message handed to the client when [`sampled_token_is_corrupt`] fires.
fn corrupt_logits_error() -> String {
    "the model produced a numerically invalid result (NaN/Inf logits) and \
     generation was stopped — this is a compute failure, not a bad prompt. \
     Try --no-flash-attn, and --rust-debug for the full diagnostic."
        .to_string()
}

fn warn_if_logits_corrupt(ctx: &LlamaContext, idx: i32, seq_id: i32) {
    // `get_logits_ith` asserts `idx` is a literal member of the context's
    // internally-tracked `initialized_logits` set — which only ever holds
    // real, positive offsets (populated from `LlamaBatch::initialized_logits`),
    // never the literal sentinel `-1` meaning "last available logits" that
    // `LlamaSampler::sample` accepts directly via the raw C API. Calling
    // `get_logits_ith(-1)` therefore panics unconditionally, on every
    // request, at the prefill-first-token call site below — `get_logits()`
    // (no index, same raw-FFI path `sample(..., -1)` itself uses) is the
    // correct equivalent for that case.
    let logits = if idx == -1 {
        ctx.get_logits()
    } else {
        ctx.get_logits_ith(idx)
    };
    let mut nan_count = 0usize;
    let mut inf_count = 0usize;
    let mut first_bad: Option<(usize, f32)> = None;
    for (i, &v) in logits.iter().enumerate() {
        if v.is_nan() {
            nan_count += 1;
            first_bad.get_or_insert((i, v));
        } else if v.is_infinite() {
            inf_count += 1;
            first_bad.get_or_insert((i, v));
        }
    }
    if nan_count > 0 || inf_count > 0 {
        tracing::error!(
            "Seq {seq_id}: corrupt logits before sampling — {nan_count} NaN, \
             {inf_count} Inf out of {} values (first bad: index {:?}). \
             This points at a numerical bug in the compute path (CPU SIMD \
             kernel, quantization, or similar), not a config issue.",
            logits.len(),
            first_bad,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::super::output::stop_prefix_holdback;
    use super::{
        CachedSlot, GenerateRequest, NotQueued, PieceOutcome, PromptCheckpoint, SchedulerHandle,
        SendOutcome, StepTimes, StreamEvent, best_checkpoint, checkpoint_cuts, common_prefix_len,
        decode_passes, drafts_kept, mtp_drafts, next_cut, pick_slot, process_piece,
        prompt_budget, text_prefix_match, try_send_piece,
    };
    use llama_cpp_2::token::LlamaToken;
    use std::time::{Duration, Instant};
    use tokio::sync::mpsc;

    /// A `steps:` line splits its window into the parts of a step, each with
    /// its time per step (decode) or per token (sample, emit), and says what
    /// none of them account for.
    #[test]
    fn a_steps_line_splits_the_window_into_its_parts() {
        let mut times = StepTimes::new(true);
        times.steps = 400;
        times.tokens = 6400;
        times.passes = 600;
        times.decode = Duration::from_millis(6000);
        times.sample = Duration::from_millis(3200);
        times.emit = Duration::from_millis(64);
        times.prefill = Duration::from_millis(500);
        times.prefill_tokens = 2048;
        let line = times.summary(Duration::from_secs(10));
        assert_eq!(
            line,
            "10.0 s, 400 steps, 16.0 seqs/step, 1.50 passes/step, 640 tok/s | \
             decode 60.0% (15.00 ms/step, 0 prompt tok), \
             sample 32.0% (0.500 ms/token), emit 0.6% (0.010 ms/token), \
             prefill 5.0% (2048 tok), idle 0.0%, other 2.4%"
        );
        times.step_prompt_tokens = 4960;
        assert!(
            times
                .summary(Duration::from_secs(10))
                .contains("decode 60.0% (15.00 ms/step, 4960 prompt tok)")
        );
    }

    /// A decode batch costs one pass of the model per run of consecutive
    /// slot numbers in the order it holds them, as llama.cpp cuts it with a
    /// KV cache per sequence; one pass per micro-batch with one for all.
    #[test]
    fn a_decode_batch_takes_one_pass_per_run_of_consecutive_slots() {
        assert_eq!(decode_passes(&[0, 1, 2, 3], false, 512), 1);
        assert_eq!(decode_passes(&[4, 5, 6], false, 512), 1);
        // A slot left out (its prompt waiting, or idle) cuts the run.
        assert_eq!(decode_passes(&[0, 1, 3, 4], false, 512), 2);
        assert_eq!(decode_passes(&[0, 2, 4, 6], false, 512), 4);
        // Order counts: the first token not yet taken starts each pass.
        assert_eq!(decode_passes(&[3, 0, 1, 2], false, 512), 2);
        assert_eq!(decode_passes(&[0, 2, 1, 3], false, 512), 2);
        assert_eq!(decode_passes(&[3, 2, 1, 0], false, 512), 4);
        // No more than n_ubatch sequences in a pass.
        assert_eq!(decode_passes(&[0, 1, 2, 3], false, 2), 2);
        // One cache for all: any slots, any order.
        assert_eq!(decode_passes(&[3, 0, 6, 2], true, 512), 1);
        assert_eq!(decode_passes(&[], false, 512), 0);
    }

    /// A step that reads a prompt beside the answers' tokens: with a KV cache
    /// per sequence, one pass takes a token of every sequence in the run,
    /// the prompt's first among them, and the rest of the prompt is passes of
    /// its own, `n_ubatch` tokens at most, shared equally between prompts.
    #[test]
    fn a_prompt_in_a_step_fills_its_slot_in_the_answers_run() {
        let step = |answers: &[i32], prompts: &[(i32, usize)]| -> Vec<i32> {
            let mut parts: Vec<(i32, usize)> = answers.iter().map(|&id| (id, 1)).collect();
            parts.extend_from_slice(prompts);
            parts.sort_unstable();
            parts
                .iter()
                .flat_map(|&(id, n)| std::iter::repeat_n(id, n))
                .collect()
        };
        // Answers 0, 1, 3, 4 and slot 2's prompt between them: one pass for
        // all five slots, then the 99 prompt tokens left. Read after the
        // step instead, the answers were two passes and the prompt one.
        let batch = step(&[0, 1, 3, 4], &[(2, 100)]);
        assert_eq!(decode_passes(&batch, false, 512), 2);
        assert_eq!(decode_passes(&[0, 1, 3, 4], false, 512), 2);
        // A prompt longer than a micro-batch: its passes are the micro-batches.
        assert_eq!(decode_passes(&step(&[0, 1], &[(2, 1200)]), false, 512), 4);
        // Two prompts share micro-batches: equal parts of each, as long as
        // the shorter lasts, then the longer alone.
        assert_eq!(
            decode_passes(&step(&[], &[(0, 100), (1, 300)]), false, 512),
            2
        );
        // Sixteen prompts arriving together, 100 tokens each: 32 of each per
        // pass, in four passes; read one per step, they were sixteen.
        let sixteen: Vec<(i32, usize)> = (0..16).map(|id| (id, 100)).collect();
        assert_eq!(decode_passes(&step(&[], &sixteen), false, 512), 4);
        // One cache for all: n_ubatch tokens a pass, whatever their slots.
        assert_eq!(
            decode_passes(&step(&[0, 1, 3, 4], &[(2, 100)]), true, 512),
            1
        );
        assert_eq!(decode_passes(&step(&[], &sixteen), true, 512), 4);
    }

    /// Beside answers a step reads a micro-batch of prompt less their
    /// tokens; with none, up to a batch, to the end of the micro-batch the
    /// first prompt ends in; never past the batch.
    #[test]
    fn a_step_reads_a_micro_batch_of_prompt_beside_the_answers() {
        assert_eq!(prompt_budget(16, 3000, 2048, 512), 496);
        assert_eq!(prompt_budget(1, 3000, 256, 16), 15);
        // Small micro-batches: half of one at least.
        assert_eq!(prompt_budget(12, 3000, 2048, 16), 8);
        // A batch already full of answers' tokens.
        assert_eq!(prompt_budget(16, 3000, 16, 512), 0);
        assert_eq!(prompt_budget(16, 3000, 300, 512), 284);
        // Nothing answering: a long prompt is read a batch at a time, as
        // read on its own; a short one's micro-batch is filled with the
        // next prompts, which wait for the next step for the rest.
        assert_eq!(prompt_budget(0, 3000, 2048, 512), 2048);
        assert_eq!(prompt_budget(0, 952, 2048, 512), 1024);
        assert_eq!(prompt_budget(0, 209, 2048, 512), 512);
        assert_eq!(prompt_budget(0, 512, 2048, 512), 512);
        assert_eq!(prompt_budget(0, 1, 2048, 512), 512);
        assert_eq!(prompt_budget(0, 600, 512, 512), 512);
    }

    /// Off, nothing is timed: there is no clock to take a part's time from,
    /// and a window is never logged nor reset.
    /// A prompt's recurrent state is kept 4 tokens before its end and a
    /// micro-batch before that, where llama.cpp's server keeps it; a chunk
    /// read from anywhere stops at the next of them, and none is past the
    /// end, at the start, or behind where reading begins.
    #[test]
    fn a_prompt_stops_for_its_checkpoints_where_llama_server_takes_them() {
        assert_eq!(checkpoint_cuts(2723, 512), [2207, 2719]);
        assert_eq!(next_cut(0, 2723, 512), Some(2207));
        assert_eq!(next_cut(2207, 2723, 512), Some(2719));
        assert_eq!(next_cut(2719, 2723, 512), None);
        // A turn read from a restored state keeps only the nearer one.
        assert_eq!(next_cut(2300, 2769, 512), Some(2765));
        // Short prompts: a cut at the start is none.
        assert_eq!(next_cut(0, 100, 512), Some(96));
        assert_eq!(next_cut(0, 4, 512), None);
        assert_eq!(next_cut(0, 1, 512), None);
    }

    #[test]
    fn steps_are_not_timed_unless_asked_for() {
        let mut times = StepTimes::new(false);
        assert!(times.start().is_none());
        let since = times.since;
        times.log_if_due();
        assert_eq!(times.since, since);
        assert!(StepTimes::new(true).start().is_some());
    }

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
        assert_eq!(
            stop_prefix_holdback("hello<end_of_turn", &stops),
            "<end_of_turn".len()
        );
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
        let (emitted, stopped) = drain(
            &mut pending,
            &stops,
            &["hello", " world", "<", "end", "_", "of", "_", "turn"],
        );
        assert_eq!(emitted, "hello world");
        assert!(
            !stopped,
            "no full stop seq seen yet — EOG handles termination"
        );
        // Caller discards `pending` on EOG; confirm it holds only the partial.
        assert_eq!(pending, "<end_of_turn");
    }

    #[test]
    fn full_stop_sequence_truncates_cleanly() {
        let stops = gemma_stops();
        let mut pending = String::new();
        let (emitted, stopped) = drain(&mut pending, &stops, &["Rome.", "<end_of_turn>"]);
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
        let (emitted, stopped) =
            drain_with_filters(&mut pending, &stops, &filters, &["hello", " world", "."]);
        assert!(!stopped);
        assert_eq!(emitted, "hello world.");
        assert!(pending.is_empty());
    }

    fn toks(ids: &[i32]) -> Vec<LlamaToken> {
        ids.iter().map(|&id| LlamaToken::new(id)).collect()
    }

    /// `--mtp` drafts only with MTP layers loaded and one request at a time,
    /// and says why not otherwise; 0 asks for nothing and says nothing.
    #[test]
    fn mtp_drafts_only_with_mtp_layers_and_one_slot() {
        assert_eq!(mtp_drafts(0, 1, 1), (0, None));
        assert_eq!(mtp_drafts(3, 1, 1), (3, None));
        let (n, why) = mtp_drafts(3, 1, 0);
        assert_eq!(n, 0);
        assert!(why.unwrap().contains("no MTP layers"));
        let (n, why) = mtp_drafts(3, 4, 1);
        assert_eq!(n, 0);
        assert!(why.unwrap().contains("--batch-size is 4"));
    }

    /// Drafts are kept from the first while the pick at their position
    /// matches; the pick after the last match is the next input, never a
    /// kept draft.
    #[test]
    fn drafts_are_kept_up_to_the_first_pick_that_differs() {
        let drafts = toks(&[10, 11, 12]);
        // Every draft matched, and a fourth pick follows the last one.
        assert_eq!(drafts_kept(&drafts, &toks(&[10, 11, 12, 13])), 3);
        // The second pick differs: one draft kept, the pick replaces the rest.
        assert_eq!(drafts_kept(&drafts, &toks(&[10, 99])), 1);
        // The first pick differs: nothing kept, as a plain step.
        assert_eq!(drafts_kept(&drafts, &toks(&[99])), 0);
        // No drafts: the one pick is a plain step's.
        assert_eq!(drafts_kept(&[], &toks(&[42])), 0);
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
    fn text_slot(
        seq_id: i32,
        tokens: Vec<LlamaToken>,
        text: &str,
        age_secs_ago: u64,
    ) -> CachedSlot {
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
    fn best_checkpoint_rejects_a_full_length_match() {
        // A checkpoint whose tokens EQUAL the request leaves nothing to
        // decode: restoring it would produce no fresh logits row, and the
        // first reply token would be sampled from stale/foreign logits. Must
        // fall through to a full re-prefill instead. A strictly-shorter
        // checkpoint in the same set is still usable and must win.
        let checkpoints = vec![
            checkpoint(toks(&[1, 2, 3, 4]), 5), // exactly the request — rejected
            checkpoint(toks(&[1, 2, 3]), 3),    // strict prefix — usable
        ];
        let tokens = toks(&[1, 2, 3, 4]);
        let best = best_checkpoint(&checkpoints, &tokens).expect("the strict prefix is usable");
        assert_eq!(best.tokens, toks(&[1, 2, 3]));

        // With only the full-length checkpoint, nothing is selected.
        let only_full = vec![checkpoint(toks(&[1, 2, 3, 4]), 5)];
        assert!(best_checkpoint(&only_full, &tokens).is_none());
    }

    #[test]
    fn full_channel_is_not_a_disconnect_and_gives_the_text_back() {
        let (tx, _rx) = mpsc::channel::<StreamEvent>(1);
        // Fill the one buffered slot without ever draining it — a
        // slow-consuming but still-connected client, not a dropped one.
        tx.try_send(StreamEvent::Token("first".to_string()))
            .unwrap();
        match try_send_piece(&tx, "second".to_string(), 0) {
            // The text must come back so the caller can re-buffer it. Dropping
            // it here is what silently removed pieces from the middle of a slow
            // client's reply while the token count kept counting them.
            SendOutcome::Backpressure(text) => assert_eq!(text, "second"),
            SendOutcome::Sent => panic!("a full channel must not report Sent"),
            SendOutcome::Disconnected => panic!("a full channel is not a disconnect"),
        }
    }

    #[test]
    fn closed_channel_is_detected_with_pending_output() {
        let (tx, rx) = mpsc::channel::<StreamEvent>(4);
        drop(rx);
        assert!(matches!(
            try_send_piece(&tx, "hello".to_string(), 0),
            SendOutcome::Disconnected
        ));
    }

    #[test]
    fn closed_channel_is_detected_even_with_empty_output() {
        let (tx, rx) = mpsc::channel::<StreamEvent>(4);
        drop(rx);
        // Empty `out` (piece fully absorbed into the stop/filter holdback
        // buffer) must not let a disconnect slip through undetected.
        assert!(matches!(
            try_send_piece(&tx, String::new(), 0),
            SendOutcome::Disconnected
        ));
    }

    #[test]
    fn open_channel_with_empty_output_is_a_no_op_send() {
        let (tx, _rx) = mpsc::channel::<StreamEvent>(4);
        assert!(matches!(
            try_send_piece(&tx, String::new(), 0),
            SendOutcome::Sent
        ));
    }

    #[test]
    fn a_slow_client_eventually_receives_every_byte_in_order() {
        // End-to-end on the two pieces that matter together: `process_piece`
        // decides what is safe to emit, and backpressure returns it when the
        // channel is full. Re-buffering at the FRONT of `pending` is what keeps
        // the order intact once the client drains.
        let (tx, mut rx) = mpsc::channel::<StreamEvent>(1);
        let stops: Vec<String> = Vec::new();
        let filters: Vec<String> = Vec::new();
        let mut pending = String::new();
        let mut delivered = String::new();

        for piece in ["alpha", "beta", "gamma", "delta"] {
            let out = match process_piece(&mut pending, &stops, &filters, piece) {
                PieceOutcome::Emit(out) => out,
                PieceOutcome::Stop(out) => out,
            };
            if let SendOutcome::Backpressure(text) = try_send_piece(&tx, out, 0) {
                pending.insert_str(0, &text);
            }
            // Drain whatever the client managed to read this round.
            while let Ok(StreamEvent::Token(t)) = rx.try_recv() {
                delivered.push_str(&t);
            }
        }
        // Flush the tail the same way the max-tokens path does.
        let tail = std::mem::take(&mut pending);
        if !tail.is_empty() {
            tx.try_send(StreamEvent::Token(tail)).unwrap();
        }
        while let Ok(StreamEvent::Token(t)) = rx.try_recv() {
            delivered.push_str(&t);
        }

        assert_eq!(
            delivered, "alphabetagammadelta",
            "no byte may be lost or reordered when the channel back-pressures"
        );
    }

    /// A full queue refuses the next request at once, before anything is
    /// generated, so the server can answer with a status code (503) rather
    /// than a stream that carries only the error.
    #[test]
    fn a_full_queue_refuses_the_next_request_before_it_starts() {
        let (handle, queue) = SchedulerHandle::detached(1);
        let queued = handle.try_submit(GenerateRequest::default());
        assert!(queued.is_ok(), "the queue's one place was free");
        assert_eq!(
            handle.try_submit(GenerateRequest::default()).err(),
            Some(NotQueued::QueueFull)
        );

        // Its place freed — the decode thread took the first request — the
        // same request goes in.
        drop(queue.try_recv().expect("the first request is queued"));
        assert!(handle.try_submit(GenerateRequest::default()).is_ok());
    }

    #[test]
    fn a_queue_no_decode_thread_reads_any_more_is_an_unloaded_model() {
        let (handle, queue) = SchedulerHandle::detached(4);
        drop(queue);
        assert_eq!(
            handle.try_submit(GenerateRequest::default()).err(),
            Some(NotQueued::Unloaded)
        );
    }

    /// `submit`, which the terminal chat uses, still reports a refusal as
    /// the stream's one event, and the stream then ends.
    #[test]
    fn submit_reports_a_refusal_as_the_streams_only_event() {
        let (handle, _queue) = SchedulerHandle::detached(0);
        let mut rx = handle.submit(GenerateRequest::default());
        match rx.try_recv() {
            Ok(StreamEvent::Error(e)) => assert_eq!(e, NotQueued::QueueFull.to_string()),
            other => panic!("expected the refusal, got {other:?}"),
        }
        assert!(matches!(
            rx.try_recv(),
            Err(mpsc::error::TryRecvError::Disconnected)
        ));
    }
}

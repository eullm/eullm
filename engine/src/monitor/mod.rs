//! What the chat UI's Monitor page shows: the requests the server is
//! answering and has answered, and the machine it runs on.
//!
//! Kept in memory only, and only on this machine: the last
//! [`HISTORY`] requests and two minutes of system samples, read by
//! `GET /api/monitor` behind the same perimeter as the rest of `/api`.
//! Nothing is written to disk (the audit trail is the durable record) and
//! nothing is sent anywhere.
//!
//! A request is watched by [`watch`], which every streamed generation puts
//! between the engine's event channel and its own reader: it forwards each
//! event unchanged and notes when the request started, when its first piece
//! of answer came and what its last event said. An answer read whole
//! (`stream: false` without the scheduler) is recorded by a [`Tracker`].

pub mod system;

use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use tokio::sync::mpsc;

use crate::inference::{AnswerStats, GenerateResult, StopReason, StreamEvent};

/// How many finished requests the page lists.
pub const HISTORY: usize = 50;

/// Where a request is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Phase {
    /// Queued or reading its prompt: nothing of the answer yet.
    Reading,
    /// Writing its answer.
    Generating,
    /// Answered.
    Done,
    /// Ended with an error, or the client left before the end.
    Failed,
}

/// One request, running or finished.
#[derive(Debug, Clone, Serialize)]
pub struct Request {
    pub id: u64,
    pub model: String,
    /// The endpoint it came through: `generate`, `chat`, `chat.completions`.
    pub endpoint: &'static str,
    /// When it started, in milliseconds since 1970 (the UI shows the time).
    pub started_ms: u64,
    pub phase: Phase,
    /// Prompt tokens read, as the engine counted them (when it has said).
    pub prompt_tokens: Option<u32>,
    /// Answer tokens: pieces received while it runs, the engine's count at the end.
    pub output_tokens: u32,
    /// Tokens per second reading the prompt, and writing the answer.
    pub prefill_tps: Option<f64>,
    pub decode_tps: Option<f64>,
    /// Time to the first piece of answer, and in all.
    pub first_token_ms: Option<u64>,
    pub duration_ms: Option<u64>,
    /// The MTP drafts proposed and kept, when the model drafted.
    pub drafts: Option<(u32, u32)>,
    /// `stop`, `length`, or the error.
    pub ended: Option<String>,
}

struct Live {
    request: Request,
    started: Instant,
}

#[derive(Default)]
struct Book {
    active: Vec<Live>,
    history: VecDeque<Request>,
}

static BOOK: Mutex<Option<Book>> = Mutex::new(None);
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn with_book<T>(f: impl FnOnce(&mut Book) -> T) -> T {
    let mut guard = BOOK.lock().unwrap_or_else(|e| e.into_inner());
    f(guard.get_or_insert_with(Book::default))
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// A request being recorded: [`begin`] puts it among the running ones, and
/// it moves to the history with [`Tracker::done`], or as failed when the
/// tracker is dropped first (an error, or the client leaving).
pub struct Tracker {
    id: u64,
    finished: bool,
}

/// Start recording a request on `model` that came through `endpoint`.
pub fn begin(model: &str, endpoint: &'static str) -> Tracker {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let request = Request::new(id, model, endpoint);
    with_book(|b| {
        b.active.push(Live {
            started: Instant::now(),
            request,
        })
    });
    Tracker {
        id,
        finished: false,
    }
}

impl Request {
    fn new(id: u64, model: &str, endpoint: &'static str) -> Self {
        Request {
            id,
            model: model.to_string(),
            endpoint,
            started_ms: now_ms(),
            phase: Phase::Reading,
            prompt_tokens: None,
            output_tokens: 0,
            prefill_tps: None,
            decode_tps: None,
            first_token_ms: None,
            duration_ms: None,
            drafts: None,
            ended: None,
        }
    }
}

impl Tracker {
    /// A piece of the answer came.
    pub fn piece(&self) {
        with_book(|b| {
            if let Some(live) = b.active.iter_mut().find(|l| l.request.id == self.id) {
                let r = &mut live.request;
                if r.first_token_ms.is_none() {
                    r.first_token_ms = Some(live.started.elapsed().as_millis() as u64);
                    r.phase = Phase::Generating;
                }
                r.output_tokens += 1;
            }
        });
    }

    /// The request was answered, with what the engine counted.
    pub fn done(
        mut self,
        generated: u32,
        prompt: u32,
        duration_ms: u64,
        stop: StopReason,
        stats: &AnswerStats,
    ) {
        self.finished = true;
        let stats = *stats;
        finish(
            self.id,
            Phase::Done,
            Some(stop.as_api_str().to_string()),
            move |r| {
                r.prompt_tokens = Some(prompt);
                r.output_tokens = generated;
                r.duration_ms = Some(duration_ms);
                r.prefill_tps = rate(prompt, stats.prompt_time);
                r.decode_tps = rate(generated, stats.eval_time);
                if stats.draft_n > 0 {
                    r.drafts = Some((stats.draft_n, stats.draft_n_accepted));
                }
            },
        );
    }

    /// The request was answered whole, without streaming.
    pub fn answered(self, result: &GenerateResult) {
        self.done(
            result.tokens_generated,
            result.tokens_prompt,
            result.duration_ms,
            result.stop_reason,
            &result.stats,
        );
    }

    /// The request ended with an error.
    pub fn failed(mut self, error: &str) {
        self.finished = true;
        finish(self.id, Phase::Failed, Some(error.to_string()), |_| {});
    }
}

impl Drop for Tracker {
    fn drop(&mut self) {
        if !self.finished {
            finish(
                self.id,
                Phase::Failed,
                Some("ended before the answer".to_string()),
                |_| {},
            );
        }
    }
}

/// Watch a request on `model` that came through `endpoint`: the returned
/// channel carries the same events as `rx`, in the same order.
///
/// A forwarding task does the watching, so this needs a Tokio runtime; with
/// none (a test, or a caller outside one) `rx` comes back as it is. When the
/// reader of the returned channel goes away, the task drops `rx` at once, so
/// the engine sees the request abandoned exactly as it did before.
pub fn watch(
    rx: mpsc::Receiver<StreamEvent>,
    model: &str,
    endpoint: &'static str,
) -> mpsc::Receiver<StreamEvent> {
    let Ok(runtime) = tokio::runtime::Handle::try_current() else {
        return rx;
    };
    // As deep as the scheduler's own channel, so a fast decode loop never
    // waits on this hop.
    let (tx, out) = mpsc::channel(256);
    runtime.spawn(forward(rx, tx, begin(model, endpoint)));
    out
}

async fn forward(
    mut rx: mpsc::Receiver<StreamEvent>,
    tx: mpsc::Sender<StreamEvent>,
    tracker: Tracker,
) {
    let mut tracker = Some(tracker);
    while let Some(event) = rx.recv().await {
        match &event {
            StreamEvent::Token(_) => {
                if let Some(t) = tracker.as_ref() {
                    t.piece();
                }
            }
            StreamEvent::Done {
                tokens_generated,
                tokens_prompt,
                duration_ms,
                stop_reason,
                stats,
            } => {
                if let Some(t) = tracker.take() {
                    t.done(
                        *tokens_generated,
                        *tokens_prompt,
                        *duration_ms,
                        *stop_reason,
                        stats,
                    );
                }
            }
            StreamEvent::Error(e) => {
                if let Some(t) = tracker.take() {
                    t.failed(e);
                }
            }
        }
        if tx.send(event).await.is_err() {
            break;
        }
    }
    // A tracker still here is a request the client left, or whose engine
    // side ended without a last event: dropping it records it as failed.
}

fn rate(tokens: u32, time: std::time::Duration) -> Option<f64> {
    let s = time.as_secs_f64();
    (tokens > 0 && s > 0.0).then(|| f64::from(tokens) / s)
}

/// Move request `id` from the running ones to the history.
fn finish(id: u64, phase: Phase, ended: Option<String>, update: impl FnOnce(&mut Request)) {
    with_book(|b| {
        let Some(i) = b.active.iter().position(|l| l.request.id == id) else {
            return;
        };
        let live = b.active.remove(i);
        let mut r = live.request;
        r.phase = phase;
        r.ended = ended;
        if r.duration_ms.is_none() {
            r.duration_ms = Some(live.started.elapsed().as_millis() as u64);
        }
        update(&mut r);
        b.record(r);
    });
}

impl Book {
    fn record(&mut self, r: Request) {
        self.history.push_front(r);
        self.history.truncate(HISTORY);
    }
}

/// The running requests, oldest first, and the finished ones, newest first.
pub fn requests() -> (Vec<Request>, Vec<Request>) {
    with_book(|b| {
        let active = b
            .active
            .iter()
            .map(|l| {
                let mut r = l.request.clone();
                // A running request's speed so far, from the pieces received.
                if let Some(first) = r.first_token_ms {
                    let writing = l.started.elapsed().as_secs_f64() - first as f64 / 1e3;
                    if writing > 0.5 {
                        r.decode_tps = Some(f64::from(r.output_tokens) / writing);
                    }
                }
                r
            })
            .collect();
        (active, b.history.iter().cloned().collect())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn a_watched_request_passes_its_events_through_and_is_recorded() {
        let (tx, rx) = mpsc::channel(8);
        let mut out = watch(rx, "test-model-watched", "chat");
        tx.send(StreamEvent::Token("Hel".into())).await.unwrap();
        tx.send(StreamEvent::Token("lo".into())).await.unwrap();
        tx.send(StreamEvent::Done {
            tokens_generated: 2,
            tokens_prompt: 10,
            duration_ms: 1500,
            stop_reason: StopReason::Stop,
            stats: AnswerStats {
                prompt_time: Duration::from_millis(500),
                eval_time: Duration::from_secs(1),
                draft_n: 0,
                draft_n_accepted: 0,
            },
        })
        .await
        .unwrap();
        drop(tx);
        let mut pieces = Vec::new();
        while let Some(e) = out.recv().await {
            if let StreamEvent::Token(t) = e {
                pieces.push(t);
            }
        }
        assert_eq!(pieces, vec!["Hel", "lo"]);
        let (_, history) = requests();
        let r = history
            .iter()
            .find(|r| r.model == "test-model-watched")
            .expect("recorded");
        assert_eq!(r.phase, Phase::Done);
        assert_eq!((r.prompt_tokens, r.output_tokens), (Some(10), 2));
        assert_eq!(r.prefill_tps, Some(20.0));
        assert_eq!(r.decode_tps, Some(2.0));
        assert_eq!(r.duration_ms, Some(1500));
        assert_eq!(r.ended.as_deref(), Some("stop"));
    }

    #[tokio::test]
    async fn a_client_that_leaves_drops_the_engine_s_channel() {
        let (tx, rx) = mpsc::channel(8);
        let out = watch(rx, "test-model-left", "generate");
        drop(out);
        tx.send(StreamEvent::Token("x".into())).await.unwrap();
        // The forwarding task sees the reader gone at its first send and lets
        // go of the engine's channel, which is how the engine learns of it.
        tokio::time::timeout(Duration::from_secs(2), tx.closed())
            .await
            .expect("closed");
        let (active, history) = requests();
        assert!(!active.iter().any(|r| r.model == "test-model-left"));
        let r = history
            .iter()
            .find(|r| r.model == "test-model-left")
            .expect("recorded");
        assert_eq!(r.phase, Phase::Failed);
    }

    #[test]
    fn a_tracker_dropped_before_the_answer_records_a_failure() {
        let tracker = begin("test-model-dropped", "generate");
        tracker.piece();
        let (active, _) = requests();
        let r = active
            .iter()
            .find(|r| r.model == "test-model-dropped")
            .expect("running");
        assert_eq!((r.phase, r.output_tokens), (Phase::Generating, 1));
        drop(tracker);
        let (active, history) = requests();
        assert!(!active.iter().any(|r| r.model == "test-model-dropped"));
        let r = history
            .iter()
            .find(|r| r.model == "test-model-dropped")
            .expect("recorded");
        assert_eq!(r.phase, Phase::Failed);
    }

    #[test]
    fn the_history_keeps_the_newest_requests() {
        // A book of its own: the shared one is filled by the other tests at once.
        let mut book = Book::default();
        for i in 0..HISTORY as u64 + 5 {
            book.record(Request::new(i, "test-model-history", "chat"));
        }
        assert_eq!(book.history.len(), HISTORY);
        assert_eq!(book.history.front().map(|r| r.id), Some(HISTORY as u64 + 4));
        assert_eq!(book.history.back().map(|r| r.id), Some(5));
    }

    #[test]
    fn without_a_runtime_the_channel_comes_back_as_it_is() {
        let (_tx, rx) = mpsc::channel::<StreamEvent>(1);
        let _ = watch(rx, "no-runtime", "chat");
        let (active, _) = requests();
        assert!(!active.iter().any(|r| r.model == "no-runtime"));
    }
}

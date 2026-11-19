//! Generation models on a real GGUF, over HTTP: what a request does to the
//! model that answers it. CPU, one thread, a tiny model copied under several
//! names into a temporary store so that every copy is a distinct model to the
//! server — a hard link would be one file, and one model.
//!
//! ```text
//! EULLM_GENERATION_TEST_MODEL=/path/to/stories260K.gguf \
//!     cargo test -p eullm-engine -- --ignored real_model_ --test-threads=1
//! ```
//!
//! The MTP test needs a model that keeps its MTP layers instead, such as
//! unsloth's Qwen3.5-0.8B-MTP (about 50 s on four CPU cores):
//!
//! ```text
//! EULLM_MTP_TEST_MODEL=/path/to/Qwen3.5-0.8B-MTP-Q4_K_M.gguf \
//!     cargo test -p eullm-engine -- --ignored real_model_mtp --test-threads=1
//! ```
//!
//! The hybrid test needs a model with a recurrent state per sequence, such as
//! Qwen3.5-0.8B (about 30 s on four CPU cores):
//!
//! ```text
//! EULLM_HYBRID_TEST_MODEL=/path/to/Qwen3.5-0.8B-Q4_K_M.gguf \
//!     cargo test -p eullm-engine -- --ignored real_model_a_hybrid --test-threads=1
//! ```
//!
//! The sliding-window test needs a model with local attention layers, such
//! as Gemma 3 270M (a few seconds on four CPU cores):
//!
//! ```text
//! EULLM_SWA_TEST_MODEL=/path/to/gemma-3-270m-it-Q8_0.gguf \
//!     cargo test -p eullm-engine -- --ignored real_model_a_sliding --test-threads=1
//! ```
//!
//! The reasoning test needs a model whose chat template opens the reasoning
//! block at the end of the prompt, such as Spark-X2.5-1.7B (about a minute
//! on four CPU cores):
//!
//! ```text
//! EULLM_PREOPENED_THINKING_TEST_MODEL=/path/to/Spark-X2.5-1.7B-Q4_K_M.gguf \
//!     cargo test -p eullm-engine -- --ignored real_model_a_reasoning_block --test-threads=1
//! ```

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use serde_json::{Value, json};
use tokio::net::TcpListener;

use super::{AppState, auth};
use crate::models::ModelStore;

/// The GGUF every test here runs on.
fn test_model() -> PathBuf {
    std::env::var("EULLM_GENERATION_TEST_MODEL")
        .expect("set EULLM_GENERATION_TEST_MODEL to a GGUF file")
        .into()
}

/// The GGUF the MTP test runs on: one that keeps its MTP layers, such as
/// unsloth's Qwen3.5-0.8B-MTP.
fn mtp_test_model() -> PathBuf {
    std::env::var("EULLM_MTP_TEST_MODEL")
        .expect("set EULLM_MTP_TEST_MODEL to a GGUF with MTP layers")
        .into()
}

/// The GGUF the hybrid test runs on: one with a recurrent state per
/// sequence, such as Qwen3.5-0.8B.
fn hybrid_test_model() -> PathBuf {
    std::env::var("EULLM_HYBRID_TEST_MODEL")
        .expect("set EULLM_HYBRID_TEST_MODEL to a hybrid GGUF, such as Qwen3.5-0.8B")
        .into()
}

/// The GGUF the sliding-window test runs on: one with local attention
/// layers, such as Gemma 3 270M (a window of 512 tokens).
fn swa_test_model() -> PathBuf {
    std::env::var("EULLM_SWA_TEST_MODEL")
        .expect("set EULLM_SWA_TEST_MODEL to a GGUF with a sliding window, such as Gemma 3 270M")
        .into()
}

/// The GGUF the reasoning test runs on: one whose chat template opens the
/// reasoning block at the end of the prompt, such as Spark-X2.5-1.7B.
fn preopened_thinking_test_model() -> PathBuf {
    std::env::var("EULLM_PREOPENED_THINKING_TEST_MODEL")
        .expect("set EULLM_PREOPENED_THINKING_TEST_MODEL to a GGUF whose template opens <think>")
        .into()
}

/// A store holding a copy of `source` under each of `names`, laid out the
/// way a pull leaves one.
fn store_of_copies(source: &Path, dir: &Path, names: &[&str]) -> ModelStore {
    for name in names {
        let model_dir = dir.join(name);
        std::fs::create_dir_all(&model_dir).expect("model dir");
        std::fs::copy(source, model_dir.join("model.gguf")).expect("copy the test model");
        let manifest = json!({
            "id": name, "name": name, "description": "test copy", "languages": ["en"],
            "base": "test", "vram_gb": 1, "size_bytes": 0, "license": "MIT",
            "digest": "sha256:0", "pulled_at": "2026-10-01T00:00:00Z", "status": "ready",
            "gguf_file": "model.gguf",
        });
        std::fs::write(model_dir.join("manifest.json"), manifest.to_string()).expect("manifest");
    }
    ModelStore::at(dir.to_path_buf())
}

/// A server on 127.0.0.1:0, with its idle-unload loop running, over a store
/// of copies of the test model. The store goes when this does.
struct TestServer {
    base: String,
    state: Arc<AppState>,
    dir: PathBuf,
}

impl Drop for TestServer {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Small and on one thread: a CPU shared with other work, and a model of a
/// few hundred thousand parameters, need no more.
async fn start(names: &[&str], configure: impl FnOnce(&mut AppState)) -> TestServer {
    start_on(&test_model(), names, configure).await
}

/// [`start`], over copies of `source` rather than the test model.
async fn start_on(
    source: &Path,
    names: &[&str],
    configure: impl FnOnce(&mut AppState),
) -> TestServer {
    let dir = std::env::temp_dir().join(format!("eullm-resident-{}", uuid::Uuid::new_v4()));
    let store = store_of_copies(source, &dir, names);
    let absent = Path::new("/nonexistent/eullm-test/.env");
    let mut state = AppState::for_tests(store, auth::ApiKeys::load(absent).expect("no keys"));
    state.ctx_size = 2048;
    configure(&mut state);
    let state = Arc::new(state);
    tokio::spawn(Arc::clone(&state).run_idle_unload_loop());

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local addr");
    let app = super::api_router(Arc::clone(&state));
    tokio::spawn(async move {
        let _ = axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await;
    });
    TestServer {
        base: format!("http://{addr}"),
        state,
        dir,
    }
}

impl TestServer {
    /// The generation models loaded right now, by name, sorted.
    async fn loaded(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .state
            .models
            .read()
            .await
            .by_recent_use()
            .iter()
            .map(|m| m.name.clone())
            .collect();
        names.sort();
        names
    }

    /// Which load of `name` is resident, to tell a model kept from one
    /// unloaded and loaded again.
    async fn load_id(&self, name: &str) -> Option<u64> {
        self.state.models.read().await.find(name).map(|m| m.id)
    }

    /// Wait up to `limit` for exactly `expected` to be loaded.
    async fn wait_for_loaded(&self, expected: &[&str], limit: Duration) {
        let start = Instant::now();
        loop {
            let loaded = self.loaded().await;
            if loaded == expected {
                return;
            }
            assert!(
                start.elapsed() < limit,
                "expected {expected:?} loaded within {limit:?}, still {loaded:?}"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    /// POST `body` to `/api/generate`; the status and every JSON line of the
    /// answer — one for a plain response, one per event for a stream.
    async fn generate(&self, body: Value) -> (reqwest::StatusCode, Vec<Value>) {
        let response = reqwest::Client::new()
            .post(format!("{}/api/generate", self.base))
            .json(&body)
            .send()
            .await
            .expect("request");
        let status = response.status();
        let text = response.text().await.expect("body");
        let lines = text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("{e}: {l}")))
            .collect();
        (status, lines)
    }

    /// POST `body` to `path`; the status and every JSON object of the answer
    /// — one for a plain response, one per NDJSON line or SSE event for a
    /// stream (`data: [DONE]` left out).
    async fn post(&self, path: &str, body: Value) -> (reqwest::StatusCode, Vec<Value>) {
        let response = reqwest::Client::new()
            .post(format!("{}{path}", self.base))
            .json(&body)
            .send()
            .await
            .expect("request");
        let status = response.status();
        let text = response.text().await.expect("body");
        let objects = text
            .lines()
            .map(|l| l.strip_prefix("data:").unwrap_or(l).trim())
            .filter(|l| !l.is_empty() && *l != "[DONE]")
            .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("{e}: {l}")))
            .collect();
        (status, objects)
    }
}

/// The text of a generation's answer, from its lines.
fn answer(lines: &[Value]) -> String {
    lines
        .iter()
        .filter_map(|l| l["response"].as_str())
        .collect()
}

/// A raw prompt of exactly `n` tokens on the test model: "the" is one token,
/// and so is each " the" after it (a leading space would be one more).
fn the_times(n: usize) -> String {
    format!("the{}", " the".repeat(n.saturating_sub(1)))
}

/// The last line of a generation that ended as it should: done, for a reason
/// a generation ends with, and no error anywhere.
fn assert_finished(lines: &[Value]) {
    let errors: Vec<&Value> = lines.iter().filter(|l| l.get("error").is_some()).collect();
    assert!(
        errors.is_empty(),
        "an error after {} lines: {errors:?}",
        lines.len()
    );
    let last = lines.last().expect("at least one line");
    assert_eq!(last["done"], true, "{last}");
    assert!(
        matches!(last["done_reason"].as_str(), Some("stop" | "length")),
        "{last}"
    );
}

/// `keep_alive: 0` with a prompt used to unload the model before the
/// request reached it: the scheduler's thread was gone, and the request
/// failed with "Scheduler queue full". It is answered now, and the model
/// unloaded once the answer is over — streamed or not.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a GGUF model in EULLM_GENERATION_TEST_MODEL"]
async fn real_model_keep_alive_zero_with_a_prompt_answers_then_unloads() {
    let server = start(&["tiny-a"], |_| {}).await;
    for stream in [false, true] {
        let (status, lines) = server
            .generate(json!({
                "model": "tiny-a", "prompt": "Once upon a time", "keep_alive": 0,
                "stream": stream, "options": { "num_predict": 24 },
            }))
            .await;
        assert_eq!(status, 200, "{lines:?}");
        assert_finished(&lines);
        assert!(!answer(&lines).is_empty(), "{lines:?}");
        server.wait_for_loaded(&[], Duration::from_secs(5)).await;
    }
}

/// `format: "json"` constrains the answer with a grammar. Every token used to
/// reach the sampler chain twice (`sample` accepts the token it draws, and
/// the scheduler accepted it again), so the grammar advanced twice per token
/// and llama.cpp aborted the whole server on the first one:
/// `GGML_ASSERT(!stacks.empty())`. The answer is JSON now, finished or cut
/// short, and the server answers the next request.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a GGUF model in EULLM_GENERATION_TEST_MODEL"]
async fn real_model_a_json_answer_keeps_the_server_up() {
    let server = start(&["tiny-a"], |state| state.batch_size = 2).await;
    for stream in [false, true] {
        let (status, lines) = server
            .generate(json!({
                "model": "tiny-a", "prompt": "Once upon a time", "format": "json",
                "stream": stream, "options": { "num_predict": 32, "temperature": 0 },
            }))
            .await;
        assert_eq!(status, 200, "{lines:?}");
        assert_finished(&lines);
        let text = answer(&lines);
        assert!(text.trim_start().starts_with('{'), "not JSON: {text:?}");
    }
}

/// keep_alive counts from the end of the answer. It used to count from its
/// start, so an answer longer than keep_alive lost its model halfway: the
/// idle-unload loop shut the scheduler down under it. A short request that
/// finishes beside a long one must not do it either: its deadline passes
/// while the long one is still answering.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a GGUF model in EULLM_GENERATION_TEST_MODEL"]
async fn real_model_a_long_generation_outlives_a_short_keep_alive() {
    let tick = Duration::from_millis(25);
    let keep_alive = Duration::from_millis(25);
    let server = start(&["tiny-a"], |state| {
        state.idle_tick = tick;
        state.ctx_size = 8192;
        state.batch_size = 2;
    })
    .await;

    let started = Instant::now();
    let response = reqwest::Client::new()
        .post(format!("{}/api/generate", server.base))
        .json(&json!({
            "model": "tiny-a", "prompt": "Once upon a time",
            "keep_alive": keep_alive.as_secs_f64(), "stream": true,
            "options": { "num_predict": 3000 },
        }))
        .send()
        .await
        .expect("request");
    assert_eq!(response.status(), 200);

    // Past keep_alive and a few ticks of the loop, for as long as the answer
    // is still coming, the model must still be there.
    let midway = keep_alive + 4 * tick;
    let mut short_one_done = false;
    let mut checks_past_midway = 0;
    let mut pending = String::new();
    let mut lines = Vec::new();
    let mut body = response.bytes_stream();
    while let Some(chunk) = body.next().await {
        pending.push_str(&String::from_utf8_lossy(&chunk.expect("chunk")));
        while let Some(end) = pending.find('\n') {
            let line: String = pending.drain(..=end).collect();
            lines.push(serde_json::from_str::<Value>(line.trim()).expect("a JSON line"));
        }
        if !short_one_done {
            let (status, short) = server
                .generate(json!({
                    "model": "tiny-a", "prompt": "Once", "stream": false,
                    "keep_alive": keep_alive.as_secs_f64(), "options": { "num_predict": 4 },
                }))
                .await;
            assert_eq!(status, 200, "{short:?}");
            assert_finished(&short);
            short_one_done = true;
        }
        let done = lines.last().is_some_and(|l| l["done"] == true);
        if started.elapsed() > midway && !done {
            assert_eq!(
                server.loaded().await,
                ["tiny-a"],
                "unloaded {:?} into an answer that was still coming",
                started.elapsed()
            );
            checks_past_midway += 1;
        }
    }
    assert_finished(&lines);
    assert!(
        checks_past_midway > 10,
        "the answer took {:?}, not long enough to outlast keep_alive; raise num_predict",
        started.elapsed()
    );

    // And once it is over, keep_alive runs out and the model goes.
    server.wait_for_loaded(&[], keep_alive + 10 * tick).await;
}

/// One model at a time, as before: each request for another model replaces
/// the resident one, and coming back loads it again. A request for the
/// resident model under another spelling of its name, or by the path of its
/// file, is answered by it without a reload.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a GGUF model in EULLM_GENERATION_TEST_MODEL"]
async fn real_model_one_model_swaps_exactly_as_before() {
    let server = start(&["tiny-a", "tiny-b"], |state| {
        state.allow_model_paths = true
    })
    .await;
    let ask = |model: &str| {
        json!({
            "model": model, "prompt": "Once upon a time", "stream": false,
            "options": { "num_predict": 8 },
        })
    };

    let mut loads = Vec::new();
    for model in ["tiny-a", "tiny-b", "tiny-a"] {
        let (status, lines) = server.generate(ask(model)).await;
        assert_eq!(status, 200, "{lines:?}");
        assert_finished(&lines);
        assert_eq!(lines[0]["model"], model);
        assert_eq!(server.loaded().await, [model]);
        loads.push(server.load_id(model).await.expect("loaded"));
    }
    assert!(
        loads[0] != loads[2],
        "coming back to tiny-a loads it again: {loads:?}"
    );

    let kept = loads[2];
    let path = server.dir.join("tiny-a").join("model.gguf");
    for spelling in ["tiny:a", "TINY-A", path.to_str().expect("utf-8 path")] {
        let (status, lines) = server.generate(ask(spelling)).await;
        assert_eq!(status, 200, "{spelling}: {lines:?}");
        assert_finished(&lines);
        assert_eq!(lines[0]["model"], "tiny-a", "{spelling}");
        assert_eq!(
            server.load_id("tiny-a").await,
            Some(kept),
            "{spelling} reloaded it"
        );
    }
}

/// A sequential engine — every multimodal model, and `--batch-size 0` — is
/// freed when the last request running on it lets go. A swap used to take it
/// out of the slot and load the next model at once, sized against VRAM that
/// still held it. It waits now: the answer streaming from the old model is
/// finished in full, and only then is the new model loaded.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a GGUF model in EULLM_GENERATION_TEST_MODEL"]
async fn real_model_a_swap_waits_for_a_sequential_engine_to_be_released() {
    let server = start(&["tiny-a", "tiny-b"], |state| {
        state.batch_size = 0;
        state.ctx_size = 4096;
    })
    .await;

    let response = reqwest::Client::new()
        .post(format!("{}/api/generate", server.base))
        .json(&json!({
            "model": "tiny-a", "prompt": "Once upon a time", "stream": true,
            "options": { "num_predict": 2000 },
        }))
        .send()
        .await
        .expect("request");
    assert_eq!(response.status(), 200);
    let mut body = response.bytes_stream();
    let first = body.next().await.expect("a first chunk").expect("chunk");
    assert!(!first.is_empty());

    let swap = {
        let base = server.base.clone();
        tokio::spawn(async move {
            let response = reqwest::Client::new()
                .post(format!("{base}/api/generate"))
                .json(&json!({
                    "model": "tiny-b", "prompt": "Once", "stream": false,
                    "options": { "num_predict": 4 },
                }))
                .send()
                .await
                .expect("request");
            let status = response.status();
            let line: Value = response.json().await.expect("json");
            (status, line, Instant::now())
        })
    };

    let mut text = String::from_utf8_lossy(&first).into_owned();
    while let Some(chunk) = body.next().await {
        text.push_str(&String::from_utf8_lossy(&chunk.expect("chunk")));
    }
    let a_ended = Instant::now();
    let lines: Vec<Value> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("a JSON line"))
        .collect();
    assert_finished(&lines);

    let (status, line, b_answered) = swap.await.expect("the swap");
    assert_eq!(status, 200, "{line}");
    assert_finished(&[line]);
    assert!(
        b_answered >= a_ended,
        "tiny-b answered {:?} before tiny-a's answer was over",
        a_ended - b_answered
    );
    assert_eq!(server.loaded().await, ["tiny-b"]);
}

impl TestServer {
    /// `/api/version`, where the residency counters are.
    async fn version(&self) -> Value {
        reqwest::get(format!("{}/api/version", self.base))
            .await
            .expect("request")
            .json()
            .await
            .expect("json")
    }

    /// A streamed generation from `model` that has started: its first chunk
    /// is in, so the model is answering.
    async fn start_streaming(&self, model: &str, num_predict: u32) -> reqwest::Response {
        let response = reqwest::Client::new()
            .post(format!("{}/api/generate", self.base))
            .json(&json!({
                "model": model, "prompt": "Once upon a time", "stream": true,
                "options": { "num_predict": num_predict },
            }))
            .send()
            .await
            .expect("request");
        assert_eq!(response.status(), 200, "{model}");
        response
    }
}

/// The rest of a streamed answer, as JSON lines, and when it ended.
async fn read_to_end(response: reqwest::Response) -> (Vec<Value>, Instant) {
    let text = response.text().await.expect("body");
    let lines = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("{e}: {l}")))
        .collect();
    (lines, Instant::now())
}

fn short(model: &str) -> Value {
    json!({
        "model": model, "prompt": "Once upon a time", "stream": false,
        "options": { "num_predict": 8 },
    })
}

/// Two models resident at once, each answering — at the same time too —
/// without either being loaded again.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a GGUF model in EULLM_GENERATION_TEST_MODEL"]
async fn real_model_two_models_answer_side_by_side() {
    let server = start(&["tiny-a", "tiny-b"], |state| state.max_loaded_models = 2).await;
    for model in ["tiny-a", "tiny-b"] {
        let (status, lines) = server.generate(short(model)).await;
        assert_eq!(status, 200, "{lines:?}");
        assert_finished(&lines);
    }
    assert_eq!(server.loaded().await, ["tiny-a", "tiny-b"]);
    let loads = (
        server.load_id("tiny-a").await,
        server.load_id("tiny-b").await,
    );

    let streamed = |model: &'static str| {
        json!({
            "model": model, "prompt": "Once upon a time", "stream": true,
            "options": { "num_predict": 200 },
        })
    };
    let ((status_a, a), (status_b, b)) = tokio::join!(
        server.generate(streamed("tiny-a")),
        server.generate(streamed("tiny-b"))
    );
    assert_eq!(status_a, 200);
    assert_eq!(status_b, 200);
    assert_finished(&a);
    assert_finished(&b);
    assert_eq!(a[0]["model"], "tiny-a");
    assert_eq!(b[0]["model"], "tiny-b");
    assert_eq!(
        (
            server.load_id("tiny-a").await,
            server.load_id("tiny-b").await
        ),
        loads,
        "neither was loaded again"
    );

    let version = server.version().await;
    assert_eq!(version["max_loaded_models"], 2);
    assert_eq!(version["loaded_models"], 2);
    assert_eq!(version["generation_evictions"], 0);
}

/// Past `--max-loaded-models`, the least recently used idle model makes
/// room, and only it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a GGUF model in EULLM_GENERATION_TEST_MODEL"]
async fn real_model_a_third_model_evicts_the_least_recently_used() {
    let server = start(&["tiny-a", "tiny-b", "tiny-c"], |state| {
        state.max_loaded_models = 2
    })
    .await;
    // tiny-b is used before tiny-a is used again: tiny-b is the least
    // recently used when tiny-c comes.
    for model in ["tiny-a", "tiny-b", "tiny-a"] {
        let (status, lines) = server.generate(short(model)).await;
        assert_eq!(status, 200, "{lines:?}");
    }
    let kept = server.load_id("tiny-a").await;

    let (status, lines) = server.generate(short("tiny-c")).await;
    assert_eq!(status, 200, "{lines:?}");
    assert_finished(&lines);
    assert_eq!(server.loaded().await, ["tiny-a", "tiny-c"]);
    assert_eq!(server.load_id("tiny-a").await, kept);
    let version = server.version().await;
    assert_eq!(version["generation_evictions"], 1);
    assert_eq!(version["loaded_models"], 2);
}

/// With every resident answering a request, a load that needs one of them
/// waits for one to finish; it does not cut another client's answer off, as
/// a swap of the one resident model does.
#[tokio::test(flavor = "multi_thread", worker_threads = 3)]
#[ignore = "needs a GGUF model in EULLM_GENERATION_TEST_MODEL"]
async fn real_model_eviction_waits_for_a_busy_model_instead_of_aborting_it() {
    let server = start(&["tiny-a", "tiny-b", "tiny-c"], |state| {
        state.max_loaded_models = 2;
        state.ctx_size = 8192;
    })
    .await;
    let started = Instant::now();
    let a = server.start_streaming("tiny-a", 3000).await;
    let b = server.start_streaming("tiny-b", 3000).await;
    let third = {
        let base = server.base.clone();
        tokio::spawn(async move {
            let response = reqwest::Client::new()
                .post(format!("{base}/api/generate"))
                .json(&short("tiny-c"))
                .send()
                .await
                .expect("request");
            let status = response.status();
            let line: Value = response.json().await.expect("json");
            (status, line, Instant::now())
        })
    };

    let ((a, a_ended), (b, b_ended)) = tokio::join!(read_to_end(a), read_to_end(b));
    assert_finished(&a);
    assert_finished(&b);
    let (status, line, c_answered) = third.await.expect("the third request");
    assert_eq!(status, 200, "{line}");
    assert_finished(&[line]);
    // Compared on the client's clocks, across connections: the server frees a
    // model a moment before its client has read the end of the body, so a
    // few milliseconds either way say nothing. Not waiting would have
    // answered tiny-c a second or more before either answer was over.
    let first_free = a_ended.min(b_ended);
    let tolerance = Duration::from_millis(250);
    assert!(
        first_free.duration_since(started) > 4 * tolerance,
        "the answers took {:?}, too short to tell waiting from not; raise num_predict",
        first_free.duration_since(started)
    );
    assert!(
        c_answered + tolerance >= first_free,
        "tiny-c answered {:?} before a model it needed was free",
        first_free - c_answered
    );
    let loaded = server.loaded().await;
    assert!(
        loaded.len() == 2 && loaded.contains(&"tiny-c".to_string()),
        "{loaded:?}"
    );
    assert_eq!(server.version().await["generation_evictions"], 1);
}

/// When no resident finishes within the wait, the load gives up with a 503
/// a client can retry — and the answers that kept it waiting are not
/// touched.
#[tokio::test(flavor = "multi_thread", worker_threads = 3)]
#[ignore = "needs a GGUF model in EULLM_GENERATION_TEST_MODEL"]
async fn real_model_a_load_that_cannot_make_room_in_time_is_a_503() {
    let server = start(&["tiny-a", "tiny-b", "tiny-c"], |state| {
        state.max_loaded_models = 2;
        state.ctx_size = 8192;
        state.busy_wait = Duration::from_millis(200);
    })
    .await;
    let a = server.start_streaming("tiny-a", 3000).await;
    let b = server.start_streaming("tiny-b", 3000).await;

    let response = reqwest::Client::new()
        .post(format!("{}/api/generate", server.base))
        .json(&short("tiny-c"))
        .send()
        .await
        .expect("request");
    assert_eq!(response.status(), 503);
    assert_eq!(
        response
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok()),
        Some("5")
    );
    let body: Value = response.json().await.expect("json");
    assert!(
        body["error"].as_str().is_some_and(|e| e.contains("tiny-c")),
        "{body}"
    );

    let ((a, _), (b, _)) = tokio::join!(read_to_end(a), read_to_end(b));
    assert_finished(&a);
    assert_finished(&b);
    assert_eq!(server.loaded().await, ["tiny-a", "tiny-b"]);

    // Retried once they are done, it goes through.
    let (status, lines) = server.generate(short("tiny-c")).await;
    assert_eq!(status, 200, "{lines:?}");
}

/// A request to a resident model takes no lock a load holds: it is
/// answered while another model is still loading.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a GGUF model in EULLM_GENERATION_TEST_MODEL"]
async fn real_model_a_request_to_a_resident_model_is_not_held_by_a_load() {
    let gate = Arc::new(super::LoadGate {
        arrived: tokio::sync::Notify::new(),
        proceed: tokio::sync::Notify::new(),
    });
    let server = start(&["tiny-a", "tiny-b"], |state| {
        state.max_loaded_models = 2;
        state.load_gate = Some(Arc::clone(&gate));
    })
    .await;
    let spawn_request = |model: &'static str| {
        let base = server.base.clone();
        tokio::spawn(async move {
            reqwest::Client::new()
                .post(format!("{base}/api/generate"))
                .json(&short(model))
                .send()
                .await
                .expect("request")
                .status()
        })
    };

    let first = spawn_request("tiny-a");
    gate.arrived.notified().await;
    gate.proceed.notify_one();
    assert_eq!(first.await.expect("tiny-a"), 200);

    // tiny-b's load stops at the gate, holding the load lock...
    let loading = spawn_request("tiny-b");
    gate.arrived.notified().await;
    // ...and tiny-a answers all the same.
    let (status, lines) =
        tokio::time::timeout(Duration::from_secs(20), server.generate(short("tiny-a")))
            .await
            .expect("tiny-a answered while tiny-b was loading");
    assert_eq!(status, 200, "{lines:?}");
    assert_finished(&lines);
    assert!(!loading.is_finished(), "tiny-b was still loading");

    gate.proceed.notify_one();
    assert_eq!(loading.await.expect("tiny-b"), 200);
    assert_eq!(server.loaded().await, ["tiny-a", "tiny-b"]);
}

/// Each resident keeps its own keep_alive: one expiring leaves the others
/// alone, and one kept for good stays.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a GGUF model in EULLM_GENERATION_TEST_MODEL"]
async fn real_model_each_model_expires_on_its_own() {
    let tick = Duration::from_millis(25);
    let server = start(&["tiny-a", "tiny-b"], |state| {
        state.max_loaded_models = 2;
        state.idle_tick = tick;
    })
    .await;
    let ask = |model: &str, keep_alive: Value| {
        json!({
            "model": model, "prompt": "Once", "stream": false, "keep_alive": keep_alive,
            "options": { "num_predict": 4 },
        })
    };
    let (status, _) = server.generate(ask("tiny-b", json!(-1))).await;
    assert_eq!(status, 200);
    let kept = server.load_id("tiny-b").await;
    let (status, _) = server.generate(ask("tiny-a", json!(0.3))).await;
    assert_eq!(status, 200);
    assert_eq!(server.loaded().await, ["tiny-a", "tiny-b"]);

    server
        .wait_for_loaded(&["tiny-b"], Duration::from_secs(5))
        .await;
    // Kept for good: still there well past tiny-a's keep_alive.
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert_eq!(server.loaded().await, ["tiny-b"]);
    assert_eq!(server.load_id("tiny-b").await, kept);

    // Its own keep_alive changes with its own next request.
    let (status, _) = server.generate(ask("tiny-b", json!(0.1))).await;
    assert_eq!(status, 200);
    server.wait_for_loaded(&[], Duration::from_secs(5)).await;
}

/// An empty request with `keep_alive: 0` unloads the model it names and only
/// that one, without loading it first; a model still answering finishes its
/// answer and goes after.
#[tokio::test(flavor = "multi_thread", worker_threads = 3)]
#[ignore = "needs a GGUF model in EULLM_GENERATION_TEST_MODEL"]
async fn real_model_an_empty_request_with_keep_alive_zero_unloads_only_that_model() {
    let server = start(&["tiny-a", "tiny-b", "tiny-c"], |state| {
        state.max_loaded_models = 2;
        state.ctx_size = 8192;
    })
    .await;
    let unload = |model: Option<&str>| {
        let mut body = json!({ "prompt": "", "keep_alive": 0 });
        if let Some(model) = model {
            body["model"] = json!(model);
        }
        body
    };
    for model in ["tiny-a", "tiny-b"] {
        let (status, _) = server.generate(short(model)).await;
        assert_eq!(status, 200);
    }

    let (status, lines) = server.generate(unload(Some("tiny-a"))).await;
    assert_eq!(status, 200, "{lines:?}");
    assert_eq!(lines[0]["done_reason"], "unload");
    assert_eq!(
        server.loaded().await,
        ["tiny-b"],
        "unloaded before the answer"
    );

    // A model that is not loaded is not loaded to be unloaded.
    let (status, lines) = server.generate(unload(Some("tiny-c"))).await;
    assert_eq!(status, 200, "{lines:?}");
    assert_eq!(lines[0]["done_reason"], "unload");
    assert_eq!(server.loaded().await, ["tiny-b"]);

    // Busy: its answer is finished, and the model goes after it.
    let answering = server.start_streaming("tiny-b", 3000).await;
    let (status, lines) = server.generate(unload(Some("tiny-b"))).await;
    assert_eq!(status, 200, "{lines:?}");
    assert_eq!(server.loaded().await, ["tiny-b"], "still answering");
    let (lines, _) = read_to_end(answering).await;
    assert_finished(&lines);
    server.wait_for_loaded(&[], Duration::from_secs(5)).await;

    // With no model named, the most recently used one goes.
    for model in ["tiny-a", "tiny-c"] {
        let (status, _) = server.generate(short(model)).await;
        assert_eq!(status, 200);
    }
    let (status, lines) = server.generate(unload(None)).await;
    assert_eq!(status, 200, "{lines:?}");
    assert_eq!(lines[0]["model"], "tiny-c");
    assert_eq!(server.loaded().await, ["tiny-a"]);
}

/// `/api/unload` with a model unloads that one and leaves the others; without
/// one, every generation model goes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a GGUF model in EULLM_GENERATION_TEST_MODEL"]
async fn real_model_unload_names_one_model_or_takes_them_all() {
    let server = start(&["tiny-a", "tiny-b"], |state| state.max_loaded_models = 2).await;
    for model in ["tiny-a", "tiny-b"] {
        let (status, _) = server.generate(short(model)).await;
        assert_eq!(status, 200);
    }
    let unload = |body: Option<Value>| {
        let url = format!("{}/api/unload", server.base);
        async move {
            let request = reqwest::Client::new().post(url);
            let request = match body {
                Some(body) => request.json(&body),
                None => request,
            };
            let response = request.send().await.expect("request");
            assert_eq!(response.status(), 200);
            response.json::<Value>().await.expect("json")
        }
    };

    let answer = unload(Some(json!({ "model": "tiny:a" }))).await;
    assert_eq!(answer["unloaded"], "tiny-a");
    assert_eq!(answer["unloaded_all"], json!(["tiny-a"]));
    assert_eq!(server.loaded().await, ["tiny-b"]);

    let (status, _) = server.generate(short("tiny-a")).await;
    assert_eq!(status, 200);
    let answer = unload(None).await;
    let mut all: Vec<String> = answer["unloaded_all"]
        .as_array()
        .expect("a list")
        .iter()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect();
    all.sort();
    assert_eq!(all, ["tiny-a", "tiny-b"]);
    assert!(answer["unloaded"].is_string());
    assert!(server.loaded().await.is_empty());
}

/// `/api/ps` lists every resident with its own expiry, and `/api/tags` and
/// `/v1/models` list every resident as loaded; a request that loaded its
/// model says how long that took.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a GGUF model in EULLM_GENERATION_TEST_MODEL"]
async fn real_model_every_resident_is_listed_with_its_own_expiry() {
    let server = start(&["tiny-a", "tiny-b"], |state| state.max_loaded_models = 2).await;
    let ask = |model: &str, keep_alive: Value| {
        json!({
            "model": model, "prompt": "Once", "stream": false, "keep_alive": keep_alive,
            "options": { "num_predict": 4 },
        })
    };
    let (status, loaded_a) = server.generate(ask("tiny-a", json!("10m"))).await;
    assert_eq!(status, 200);
    let (status, _) = server.generate(ask("tiny-b", json!(-1))).await;
    assert_eq!(status, 200);
    assert!(
        loaded_a[0]["load_duration"]
            .as_u64()
            .is_some_and(|ns| ns > 0),
        "{loaded_a:?}"
    );
    let (_, again) = server.generate(ask("tiny-a", json!("10m"))).await;
    assert_eq!(again[0]["load_duration"], 0, "already loaded: {again:?}");

    let ps: Value = reqwest::get(format!("{}/api/ps", server.base))
        .await
        .expect("request")
        .json()
        .await
        .expect("json");
    let models = ps["models"].as_array().expect("models");
    let names: Vec<&str> = models.iter().filter_map(|m| m["name"].as_str()).collect();
    assert_eq!(names, ["tiny-a", "tiny-b"], "most recently used first");
    let expires = |i: usize| {
        chrono::DateTime::parse_from_rfc3339(models[i]["expires_at"].as_str().expect("a date"))
            .expect("RFC 3339")
            .to_utc()
    };
    let in_ten_minutes = (expires(0) - chrono::Utc::now()).num_seconds();
    assert!((590..=600).contains(&in_ten_minutes), "{in_ten_minutes}");
    assert!(
        expires(1)
            .format("%Y")
            .to_string()
            .parse::<i32>()
            .expect("a year")
            > 2200,
        "kept for good"
    );
    for m in models {
        assert!(m["size"].as_u64().is_some_and(|b| b > 0), "{m}");
        assert!(m["context_length"].as_u64().is_some_and(|c| c > 0), "{m}");
        assert_eq!(m["eullm"]["slot"], "generation");
    }

    let tags: Value = reqwest::get(format!("{}/api/tags", server.base))
        .await
        .expect("request")
        .json()
        .await
        .expect("json");
    let mut loaded: Vec<&str> = tags["models"]
        .as_array()
        .expect("models")
        .iter()
        .filter(|m| m["loaded"] == true)
        .filter_map(|m| m["name"].as_str())
        .collect();
    loaded.sort();
    assert_eq!(loaded, ["tiny-a", "tiny-b"]);
    let openai: Value = reqwest::get(format!("{}/v1/models", server.base))
        .await
        .expect("request")
        .json()
        .await
        .expect("json");
    let ids: Vec<&str> = openai["data"]
        .as_array()
        .expect("data")
        .iter()
        .filter_map(|m| m["id"].as_str())
        .collect();
    assert!(
        ids.contains(&"tiny-a") && ids.contains(&"tiny-b"),
        "{ids:?}"
    );
}

/// With `--default-model`, a request that names no model is answered by that
/// model, loaded for it beside the others, and not by the one used last.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a GGUF model in EULLM_GENERATION_TEST_MODEL"]
async fn real_model_a_request_naming_no_model_goes_to_the_default_one() {
    let server = start(&["tiny-a", "tiny-b"], |state| {
        state.max_loaded_models = 2;
        let path = state
            .store
            .gguf_path("tiny-b")
            .expect("tiny-b in the store");
        state.default_model = Some(super::NamedModel {
            name: "tiny-b".into(),
            path,
        });
    })
    .await;
    let (status, _) = server.generate(short("tiny-a")).await;
    assert_eq!(status, 200);
    let unnamed = json!({
        "prompt": "Once upon a time", "stream": false, "options": { "num_predict": 8 },
    });
    let (status, lines) = server.generate(unnamed.clone()).await;
    assert_eq!(status, 200, "{lines:?}");
    assert_finished(&lines);
    assert_eq!(lines[0]["model"], "tiny-b", "not tiny-a, the one used last");
    assert_eq!(server.loaded().await, ["tiny-a", "tiny-b"]);
    let kept = server.load_id("tiny-b").await;

    // Loaded now: the same load answers.
    let (status, lines) = server.generate(unnamed).await;
    assert_eq!(status, 200, "{lines:?}");
    assert_eq!(lines[0]["model"], "tiny-b");
    assert_eq!(server.load_id("tiny-b").await, kept);
}

// ── Routing: `"model": "auto"` ───────────────────────────────────────────
//
// These also need a decision model: Qwen3-0.6B reads a choice as any
// instruction-tuned model does.
//
//     EULLM_DECISION_TEST_MODEL=/path/to/Qwen3-0.6B-Q8_0.gguf

/// The decision model in `EULLM_DECISION_TEST_MODEL`, loaded on one thread.
async fn decision_slot() -> super::DecisionSlot {
    let path = std::env::var("EULLM_DECISION_TEST_MODEL")
        .expect("set EULLM_DECISION_TEST_MODEL to a GGUF file");
    let model = tokio::task::spawn_blocking(move || {
        crate::inference::decision::DecisionModel::load(
            Path::new(&path),
            1,
            crate::inference::decision::DEFAULT_DECISION_CTX,
            true,
            crate::inference::test_backend(),
        )
    })
    .await
    .expect("the load task")
    .expect("load the decision model");
    super::DecisionSlot {
        model_name: "test-decision".into(),
        model: Arc::new(model),
        is_reserved_companion: true,
        reserve_bytes: 0,
    }
}

/// `"model": "auto"` between `tiny-a` and `tiny-b`, the latter the fallback.
fn route_table(state: &AppState, timeout: Duration) -> super::route::RouteTable {
    let candidate = |name: &str, description: &str| super::route::RouteCandidate {
        name: name.to_string(),
        path: state.store.gguf_path(name).expect("in the store"),
        description: description.to_string(),
        source: super::route::DescriptionSource::Flag,
        has_projector: false,
    };
    super::route::RouteTable {
        candidates: vec![
            candidate("tiny-a", "Short everyday requests and simple facts"),
            candidate(
                "tiny-b",
                "Multi-step reasoning, maths, code and long answers",
            ),
        ],
        fallback: 1,
        timeout,
    }
}

impl TestServer {
    /// POST `body` to `/api/route`.
    async fn route(&self, body: Value) -> (reqwest::StatusCode, Value) {
        let response = reqwest::Client::new()
            .post(format!("{}/api/route", self.base))
            .json(&body)
            .send()
            .await
            .expect("request");
        let status = response.status();
        (status, response.json().await.expect("json"))
    }
}

/// The decision model chooses between the two, with probabilities that sum
/// to one, and nothing is loaded for it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs GGUF models in EULLM_GENERATION_TEST_MODEL and EULLM_DECISION_TEST_MODEL"]
async fn real_model_a_route_is_decided_with_probabilities_that_sum_to_one() {
    let decision = decision_slot().await;
    let server = start(&["tiny-a", "tiny-b"], |state| {
        state.router = Some(route_table(state, Duration::from_secs(300)));
        state.decision = tokio::sync::RwLock::new(Some(decision));
    })
    .await;
    let (status, route) = server
        .route(json!({ "model": "auto", "messages": [
            { "role": "user", "content": "What is the capital of France?" }
        ] }))
        .await;
    assert_eq!(status, 200, "{route}");
    assert_eq!(route["reason"], "decided", "{route}");
    assert_eq!(route["decision_model"], "test-decision");
    let candidates = route["candidates"].as_array().expect("candidates");
    let total: f64 = candidates
        .iter()
        .map(|c| c["probability"].as_f64().expect("a probability"))
        .sum();
    assert!((total - 1.0).abs() < 1e-6, "{route}");
    let chosen = route["model"].as_str().expect("a model");
    assert!(["tiny-a", "tiny-b"].contains(&chosen), "{route}");
    let confidence = route["confidence"].as_f64().expect("a confidence");
    assert!((0.0..=1.0).contains(&confidence));
    assert!(route["decision_ms"].as_f64().is_some_and(|ms| ms > 0.0));
    assert!(server.loaded().await.is_empty(), "a dry run loads nothing");
    if let Ok(dir) = std::env::var("EULLM_AUDIT_DIR") {
        let audit = std::fs::read_to_string(Path::new(&dir).join("audit.jsonl")).expect("audit");
        let id = route["route_id"].as_str().expect("an id");
        let line: Value = audit
            .lines()
            .filter_map(|l| serde_json::from_str::<Value>(l).ok())
            .find(|l| l["id"] == id)
            .expect("the route is audited under its id");
        assert_eq!(line["request_type"], "route");
        assert_eq!(line["routing"]["model"], chosen);
        assert_eq!(line["routing"]["dry_run"], true);
        assert_eq!(line["decision"]["answers"][0]["answer"], chosen);
    }
}

/// A decision that takes longer than the timeout is abandoned, and the
/// fallback answers.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs GGUF models in EULLM_GENERATION_TEST_MODEL and EULLM_DECISION_TEST_MODEL"]
async fn real_model_a_route_past_its_timeout_falls_back() {
    let decision = decision_slot().await;
    let server = start(&["tiny-a", "tiny-b"], |state| {
        state.router = Some(route_table(state, Duration::from_millis(10)));
        state.decision = tokio::sync::RwLock::new(Some(decision));
    })
    .await;
    let long = "Explain, step by step, how a lighthouse lens focuses light. ".repeat(20);
    let (status, route) = server
        .route(json!({ "model": "auto", "messages": [{ "role": "user", "content": long }] }))
        .await;
    assert_eq!(status, 200, "{route}");
    assert_eq!(route["reason"], "timeout", "{route}");
    assert_eq!(route["model"], "tiny-b");
    assert!(route["confidence"].is_null());
    assert!(
        route["decision_ms"].as_f64().is_some_and(|ms| ms < 5000.0),
        "waited for the decision: {route}"
    );
}

/// `"model": "auto"` on every endpoint, streamed and not: the decision model
/// chooses, the chosen model answers — every line and chunk says which —
/// the headers say it before the body, and the last line or chunk, or the
/// response, carries the route. With `EULLM_AUDIT_DIR` set, the
/// generation's audit line names the route's.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs GGUF models in EULLM_GENERATION_TEST_MODEL and EULLM_DECISION_TEST_MODEL"]
async fn real_model_auto_answers_on_every_endpoint_and_says_how() {
    let decision = decision_slot().await;
    let server = start(&["tiny-a", "tiny-b"], |state| {
        state.max_loaded_models = 2;
        state.router = Some(route_table(state, Duration::from_secs(300)));
        state.decision = tokio::sync::RwLock::new(Some(decision));
    })
    .await;
    let messages = json!([{ "role": "user", "content": "What is the capital of France?" }]);
    for path in ["/api/generate", "/api/chat", "/v1/chat/completions"] {
        for stream in [false, true] {
            let mut body = json!({
                "model": "auto", "stream": stream, "think": false,
                "options": { "num_predict": 8 }, "max_tokens": 8,
            });
            if path == "/api/generate" {
                body["prompt"] = json!("The capital of France is");
            } else {
                body["messages"] = messages.clone();
            }
            let response = reqwest::Client::new()
                .post(format!("{}{path}", server.base))
                .json(&body)
                .send()
                .await
                .expect("request");
            assert_eq!(response.status(), 200, "{path} {body}");
            let header = |name: &str| {
                response
                    .headers()
                    .get(name)
                    .map(|v| v.to_str().expect("text").to_string())
                    .unwrap_or_else(|| panic!("{path}: no {name}"))
            };
            let (model, reason, id) = (
                header("x-eullm-model"),
                header("x-eullm-route"),
                header("x-eullm-route-id"),
            );
            assert!(["tiny-a", "tiny-b"].contains(&model.as_str()), "{model}");
            assert_eq!(reason, "decided", "{path}");
            let text = response.text().await.expect("body");
            let objects: Vec<Value> = text
                .lines()
                .map(|l| l.strip_prefix("data:").unwrap_or(l).trim())
                .filter(|l| !l.is_empty() && *l != "[DONE]")
                .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("{e}: {l}")))
                .collect();
            let (last, before) = objects.split_last().expect("an answer");
            for object in &objects {
                assert_eq!(object["model"], model.as_str(), "{path}: {object}");
            }
            assert!(
                before.iter().all(|o| o.get("eullm").is_none()),
                "{path}: only the last carries the route"
            );
            let route = &last["eullm"]["route"];
            assert_eq!(route["requested"], "auto", "{path}: {last}");
            assert_eq!(route["model"], model.as_str());
            assert_eq!(route["reason"], "decided");
            assert_eq!(route["id"], id.as_str());
            assert_eq!(route["decision_model"], "test-decision");
            let p = route["probabilities"].as_object().expect("probabilities");
            assert_eq!(p.len(), 2, "{route}");
            assert!((p.values().filter_map(Value::as_f64).sum::<f64>() - 1.0).abs() < 1e-6);

            if let Ok(dir) = std::env::var("EULLM_AUDIT_DIR") {
                let audit =
                    std::fs::read_to_string(Path::new(&dir).join("audit.jsonl")).expect("audit");
                let lines: Vec<Value> = audit
                    .lines()
                    .filter_map(|l| serde_json::from_str(l).ok())
                    .collect();
                let routed = lines
                    .iter()
                    .find(|l| l["id"] == id.as_str())
                    .expect("the route's line");
                assert_eq!(routed["routing"]["model"], model.as_str());
                let generation = lines
                    .iter()
                    .find(|l| l["route"]["id"] == id.as_str())
                    .expect("the generation's line names its route");
                assert_eq!(generation["model"], model.as_str());
                assert!(generation["route"].get("fallback").is_none());
            }
        }
    }
}

/// `--mtp` must not change what the model writes: each draft is checked by
/// the model, kept only when it is the token the model picks, and the
/// positions it rejects are taken back. Greedy, the answer with the head's
/// real drafts is therefore the answer with drafts that are all wrong —
/// every decode taken back — character for character; and the real drafts
/// are kept while the wrong ones never are. (Against the answer written
/// without drafts it can differ: a token decoded among several is computed
/// in another order than one decoded alone, and a hybrid model's recurrent
/// layers even switch algorithm. See docs/engine-guide.md.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a GGUF with MTP layers in EULLM_MTP_TEST_MODEL"]
async fn real_model_mtp_drafts_do_not_change_the_answer() {
    use crate::inference::scheduler::FORCE_WRONG_DRAFTS;
    use std::sync::atomic::Ordering;

    let server = start_on(&mtp_test_model(), &["mtp-model"], |state| {
        state.mtp = 2;
    })
    .await;
    let body = json!({
        "model": "mtp-model",
        "prompt": "List the first ten prime numbers, separated by commas.",
        "stream": false,
        "cache_prompt": false,
        "options": { "temperature": 0, "seed": 1, "num_predict": 96 },
    });

    let (status, real) = server.generate(body.clone()).await;
    assert!(status.is_success(), "{real:?}");
    assert_finished(&real);
    FORCE_WRONG_DRAFTS.store(true, Ordering::Relaxed);
    let (status, wrong) = server.generate(body).await;
    FORCE_WRONG_DRAFTS.store(false, Ordering::Relaxed);
    assert!(status.is_success(), "{wrong:?}");
    assert_finished(&wrong);

    assert_eq!(answer(&real), answer(&wrong));
    let drafts = |lines: &[Value]| {
        let last = lines.last().expect("an answer");
        (
            last["draft_n"].as_u64().unwrap_or(0),
            last["draft_n_accepted"].as_u64().unwrap_or(0),
        )
    };
    let (proposed, kept) = drafts(&real);
    assert!(
        proposed > 0 && kept > 0,
        "real drafts: {kept} of {proposed} kept"
    );
    let (proposed, kept) = drafts(&wrong);
    assert!(
        proposed > 0 && kept == 0,
        "wrong drafts: {kept} of {proposed} kept"
    );
}

/// With more than one slot a prompt is read a chunk at a time, beside the
/// tokens of the answers already going (roadmap 0.7-D). It used to be read
/// whole as soon as its request was taken, and every answer stopped until it
/// was: one long prompt froze every other conversation on the server.
///
/// An answer streams while a 3,500-token prompt arrives, read 15 tokens at a
/// time in its steps (a micro-batch of 16 less the answer's token): the
/// answer must keep coming while the prompt is read, and the prompt's own
/// answer must be the one a server with a single slot, reading it whole,
/// gives. The prompt's request asks for one token, so the time it takes is
/// the time its prompt takes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a GGUF model in EULLM_GENERATION_TEST_MODEL"]
async fn real_model_a_long_prompt_is_read_between_the_tokens_of_an_answer() {
    let chunked = start(&["tiny-a"], |state| {
        state.ctx_size = 8192;
        state.batch_size = 2;
        state.n_batch = 256;
        state.n_ubatch = Some(16);
    })
    .await;
    let long_prompt = the_times(3500);
    let long_body = json!({
        "model": "tiny-a", "prompt": long_prompt, "raw": true, "stream": false,
        "cache_prompt": false, "options": { "temperature": 0, "seed": 1, "num_predict": 1 },
    });

    // The answer already going, timed token by token.
    let response = reqwest::Client::new()
        .post(format!("{}/api/generate", chunked.base))
        .json(&json!({
            "model": "tiny-a", "prompt": "Once upon a time", "stream": true,
            "options": { "num_predict": 3000 },
        }))
        .send()
        .await
        .expect("request");
    assert_eq!(response.status(), 200);
    let token_times = Arc::new(std::sync::Mutex::new(Vec::<Instant>::new()));
    let times = Arc::clone(&token_times);
    let streaming = tokio::spawn(async move {
        let mut body = response.bytes_stream();
        let mut pending = String::new();
        let mut lines = Vec::new();
        while let Some(chunk) = body.next().await {
            pending.push_str(&String::from_utf8_lossy(&chunk.expect("chunk")));
            while let Some(end) = pending.find('\n') {
                let line: String = pending.drain(..=end).collect();
                let line: Value = serde_json::from_str(line.trim()).expect("a JSON line");
                if line["done"] != true {
                    times.lock().unwrap().push(Instant::now());
                }
                lines.push(line);
            }
        }
        lines
    });
    while token_times.lock().unwrap().len() < 8 {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }

    let sent = Instant::now();
    let (status, read_in_chunks) = chunked.generate(long_body.clone()).await;
    let answered = Instant::now();
    assert_eq!(status, 200, "{read_in_chunks:?}");
    assert_finished(&read_in_chunks);
    assert_eq!(read_in_chunks.last().unwrap()["prompt_eval_count"], 3500);

    let beside = token_times
        .lock()
        .unwrap()
        .iter()
        .filter(|t| **t > sent && **t < answered)
        .count();
    let lines = streaming.await.expect("the answer streamed");
    assert_finished(&lines);
    let answer_ended = *token_times.lock().unwrap().last().unwrap();
    assert!(
        answer_ended > answered,
        "the answer ended before the prompt was read: make it longer"
    );
    // 3,500 tokens 15 at a time are 234 steps, each with a token of the
    // answer. Read whole, the prompt let out only what the answer wrote while
    // the request was on its way: a handful.
    assert!(
        beside >= 100,
        "{beside} tokens of the answer while the prompt was read"
    );

    let whole = start(&["tiny-a"], |state| {
        state.ctx_size = 8192;
        state.n_batch = 256;
        state.n_ubatch = Some(16);
    })
    .await;
    let (status, read_whole) = whole.generate(long_body).await;
    assert_eq!(status, 200, "{read_whole:?}");
    assert_eq!(answer(&read_in_chunks), answer(&read_whole));
}

/// A prompt read in chunks answers as the same prompt read whole, at the
/// lengths where the chunks' edges fall (1 token, the batch, the batch and
/// one more), from nothing and from a prefix the slot kept from the turn
/// before. Greedy, on the CPU, where the two are the same arithmetic.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a GGUF model in EULLM_GENERATION_TEST_MODEL"]
async fn real_model_a_prompt_read_in_chunks_answers_as_one_read_whole() {
    let configure = |slots: usize| {
        move |state: &mut AppState| {
            state.ctx_size = 4096;
            state.batch_size = slots;
            state.n_batch = 64;
            state.n_ubatch = Some(16);
        }
    };
    let chunked = start(&["tiny-a"], configure(2)).await;
    let whole = start(&["tiny-a"], configure(1)).await;
    let ask = |prompt: String, cache_prompt: bool| {
        json!({
            "model": "tiny-a", "prompt": prompt, "raw": true, "stream": false,
            "cache_prompt": cache_prompt,
            "options": { "temperature": 0, "seed": 1, "num_predict": 16 },
        })
    };

    for words in [1, 64, 65, 200] {
        let prompt = the_times(words);
        let (status, a) = chunked.generate(ask(prompt.clone(), false)).await;
        assert_eq!(status, 200, "{a:?}");
        assert_finished(&a);
        assert_eq!(a.last().unwrap()["prompt_eval_count"], words, "{a:?}");
        let (status, b) = whole.generate(ask(prompt, false)).await;
        assert_eq!(status, 200, "{b:?}");
        assert_eq!(answer(&a), answer(&b), "{words} tokens");
    }

    // A second turn that extends the first: read from the prefix its slot
    // kept, in chunks on one server, whole on the other.
    let first = " Once upon a time".repeat(10);
    let second = format!("{first}{}", " there was a dog".repeat(30));
    for server in [&chunked, &whole] {
        let (status, lines) = server.generate(ask(first.clone(), true)).await;
        assert_eq!(status, 200, "{lines:?}");
    }
    let (_, a) = chunked.generate(ask(second.clone(), true)).await;
    let (_, b) = whole.generate(ask(second, true)).await;
    assert_finished(&a);
    assert_finished(&b);
    assert_eq!(answer(&a), answer(&b));
}

/// Prompts that arrive together are read together, in the same decode
/// steps, beside the token of an answer already going: a prompt's last token
/// is then one among other sequences' tokens, and its answer's first token
/// is sampled from that token's own logits. Each answer must be the one a
/// server with a single slot gives the same prompt, read whole: an answer
/// started from another position's logits, or a prompt read into another's
/// slot, would not be. Lengths from one token to several steps' worth, so
/// that prompts end in different steps and in different parts of a step.
/// Greedy, on the CPU.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs a GGUF model in EULLM_GENERATION_TEST_MODEL"]
async fn real_model_prompts_read_together_answer_as_each_read_alone() {
    let configure = |slots: usize| {
        move |state: &mut AppState| {
            state.ctx_size = 1024 * slots as u32;
            state.batch_size = slots;
            state.n_batch = 256;
            state.n_ubatch = Some(64);
        }
    };
    let together = start(&["tiny-a"], configure(6)).await;
    let alone = start(&["tiny-a"], configure(1)).await;
    let ask = |prompt: &str, num_predict: u32| {
        json!({
            "model": "tiny-a", "prompt": prompt, "raw": true, "stream": false,
            "cache_prompt": false,
            "options": { "temperature": 0, "seed": 1, "num_predict": num_predict },
        })
    };
    let prompts: Vec<String> = vec![
        "Once".into(),
        "Lily and Tom went to the park".into(),
        " Once upon a time there was a little dog".repeat(3),
        "The sun was hot and the bird sang".repeat(6),
        " One day a girl named Sue found a big red ball".repeat(12),
    ];

    // The answer already going, so that every step has a token beside the
    // prompts: streamed, to know it is.
    let response = reqwest::Client::new()
        .post(format!("{}/api/generate", together.base))
        .json(&json!({
            "model": "tiny-a", "prompt": "Once upon a time", "stream": true,
            "options": { "temperature": 0, "num_predict": 900 },
        }))
        .send()
        .await
        .expect("request");
    assert_eq!(response.status(), 200);
    let mut going = response.bytes_stream();
    going
        .next()
        .await
        .expect("the answer's first token")
        .expect("chunk");

    // The decode loop held while the requests are made, so that it takes
    // them together and reads their prompts in the same steps.
    let gate = together
        .state
        .models
        .read()
        .await
        .find("tiny-a")
        .and_then(|m| m.scheduler.as_ref().map(|s| s.gate()))
        .expect("tiny-a runs on the scheduler");
    let held = gate.lock().expect("the gate");
    let asked: Vec<_> = prompts
        .iter()
        .map(|prompt| {
            let base = together.base.clone();
            let body = ask(prompt, 24);
            tokio::spawn(async move {
                let response = reqwest::Client::new()
                    .post(format!("{base}/api/generate"))
                    .json(&body)
                    .send()
                    .await
                    .expect("request");
                let status = response.status();
                let text = response.text().await.expect("body");
                let lines: Vec<Value> = text
                    .lines()
                    .filter(|l| !l.trim().is_empty())
                    .map(|l| serde_json::from_str(l).expect("a JSON line"))
                    .collect();
                (status, lines)
            })
        })
        .collect();
    // A blocking wait, not an `.await`: the requests run on the runtime's
    // workers meanwhile, and the gate is not held across an await point.
    std::thread::sleep(Duration::from_millis(300));
    drop(held);
    let answers = futures_util::future::join_all(asked).await;
    for (prompt, asked) in prompts.iter().zip(&answers) {
        let (status, read_together) = asked.as_ref().expect("the request's task");
        assert_eq!(status.as_u16(), 200, "{read_together:?}");
        assert_finished(read_together);
        let (status, read_alone) = alone.generate(ask(prompt, 24)).await;
        assert_eq!(status, 200, "{read_alone:?}");
        assert_eq!(
            answer(read_together),
            answer(&read_alone),
            "{} prompt tokens",
            read_alone.last().unwrap()["prompt_eval_count"]
        );
    }
    let mut rest = String::new();
    while let Some(chunk) = going.next().await {
        rest.push_str(&String::from_utf8_lossy(&chunk.expect("chunk")));
    }
    assert!(rest.contains("\"done\":true"), "{rest}");
}

/// A chat template that opens the reasoning block at the end of the prompt
/// (Spark-X2.5's, Qwen3.6's, DeepSeek-R1's) leaves the model writing from
/// inside it: the opening tag is in the prompt, never in what the model
/// writes. The answer is sent starting with it, so that the client gets the
/// block whole — on both backends, streamed or not, on `/api/chat` and on
/// `/v1/chat/completions`. Asked with `think`, the reasoning then goes to
/// `thinking`, and none of it, nor a dangling `</think>`, to `content`.
///
/// Up to 0.7.40 the tag was cut from the prompt instead, for the model to
/// write it, and Spark-X2.5 does not.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a GGUF whose template opens <think> in EULLM_PREOPENED_THINKING_TEST_MODEL"]
async fn real_model_a_reasoning_block_the_template_opens_reaches_the_client_whole() {
    let messages = json!([{ "role": "user", "content": "What is 2 + 3?" }]);
    // 1 slot: the scheduler; 0: the sequential engine.
    for slots in [1, 0] {
        let server = start_on(&preopened_thinking_test_model(), &["reasoner"], |state| {
            state.batch_size = slots;
            state.threads = 4;
        })
        .await;
        for stream in [false, true] {
            let case = format!("{slots} slot(s), stream {stream}");

            // Without `think`, the reasoning stays in the answer, tags and all.
            for path in ["/api/chat", "/v1/chat/completions"] {
                let body = json!({
                    "model": "reasoner", "messages": messages, "stream": stream,
                    "options": { "temperature": 0, "seed": 1, "num_predict": 24 },
                    "temperature": 0, "seed": 1, "max_tokens": 24,
                });
                let (status, objects) = server.post(path, body).await;
                assert_eq!(status, 200, "{case}, {path}: {objects:?}");
                let content: String = objects
                    .iter()
                    .filter_map(|o| {
                        o["message"]["content"]
                            .as_str()
                            .or(o["choices"][0]["message"]["content"].as_str())
                            .or(o["choices"][0]["delta"]["content"].as_str())
                    })
                    .collect();
                assert!(
                    content.starts_with("<think>"),
                    "{case}, {path}: {content:?}"
                );
            }

            // With it, the reasoning goes apart.
            let body = json!({
                "model": "reasoner", "messages": messages, "stream": stream, "think": true,
                "options": { "temperature": 0, "seed": 1, "num_predict": 24 },
            });
            let (status, objects) = server.post("/api/chat", body).await;
            assert_eq!(status, 200, "{case}: {objects:?}");
            let part = |field: &str| -> String {
                objects
                    .iter()
                    .filter_map(|o| o["message"][field].as_str())
                    .collect()
            };
            let (thinking, content) = (part("thinking"), part("content"));
            assert!(!thinking.trim().is_empty(), "{case}: no thinking");
            assert!(
                !thinking.contains("think>") && !content.contains("think>"),
                "{case}: a tag left in {thinking:?} / {content:?}"
            );
        }
    }
}

/// A conversation's next turn on a hybrid model (Qwen3.5 and Qwen3.6: a
/// recurrent state per sequence, which cannot be cut back to an earlier
/// token). The next prompt shares all of the slot's history but the first
/// prompt's last 4 tokens, as the chat template that writes an empty
/// thinking block after the assistant's header makes it; the state kept 4
/// tokens before the first prompt's end (`StateCheckpoint`) is restored and
/// only the new tokens are read. The answer must be the one the whole prompt
/// read from the start gets, with one slot (the prompt read whole) and with
/// two (read in the decode steps, a micro-batch at a time); and reading it
/// must take a fraction of reading it whole, or nothing was restored.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a hybrid GGUF in EULLM_HYBRID_TEST_MODEL"]
async fn real_model_a_hybrid_models_next_turn_restores_its_kept_state() {
    let configure = |slots: usize| {
        move |state: &mut AppState| {
            state.ctx_size = 2048 * slots as u32;
            state.batch_size = slots;
            state.n_batch = 128;
            state.n_ubatch = Some(32);
        }
    };
    let ask = |prompt: &str, cache_prompt: bool| {
        json!({
            "model": "hybrid", "prompt": prompt, "raw": true, "stream": false,
            "cache_prompt": cache_prompt,
            "options": { "temperature": 0, "seed": 1, "num_predict": 16 },
        })
    };
    let base = "The harbour office kept a ledger of every ship that entered the bay, \
                with the hour, the cargo and the weather at the mouth of the river."
        .repeat(8);
    let first = format!("{base} alpha beta gamma delta");
    let second = format!("{base} one two three four five six seven");

    let fresh = start_on(&hybrid_test_model(), &["hybrid"], configure(1)).await;
    let (status, whole) = fresh.generate(ask(&second, false)).await;
    assert_eq!(status, 200, "{whole:?}");
    assert_finished(&whole);
    let whole_ns = whole.last().unwrap()["prompt_eval_duration"].as_u64().unwrap();

    for slots in [1, 2] {
        let server = start_on(&hybrid_test_model(), &["hybrid"], configure(slots)).await;
        let (status, lines) = server.generate(ask(&first, true)).await;
        assert_eq!(status, 200, "{lines:?}");
        assert_finished(&lines);
        let (status, next) = server.generate(ask(&second, true)).await;
        assert_eq!(status, 200, "{next:?}");
        assert_finished(&next);
        assert_eq!(answer(&next), answer(&whole), "{slots} slots");
        let next_ns = next.last().unwrap()["prompt_eval_duration"].as_u64().unwrap();
        assert!(
            next_ns * 4 < whole_ns,
            "{slots} slots: the next turn read its prompt in {next_ns} ns, the whole one in \
             {whole_ns} ns: the kept state was not restored"
        );
    }
}

/// `--no-mmproj`: a model with a projector beside its weights loads on the
/// scheduler with every slot asked for, answers text, and refuses a request
/// with an image, saying why. Without it, the projector would have sent the
/// model to the sequential engine, one request at a time: on Leonardo, Gemma 4
/// 31B with `--batch-size 8` answered 20 of 200 text requests in 25 minutes.
/// The projector here is not a projector at all, which is the point: it is
/// never opened.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a GGUF model in EULLM_GENERATION_TEST_MODEL"]
async fn real_model_no_mmproj_serves_a_model_with_a_projector_in_batches() {
    let server = start(&["tiny-mm"], |state| {
        state.no_mmproj = true;
        state.batch_size = 4;
    })
    .await;
    std::fs::write(server.dir.join("tiny-mm").join("mmproj-F16.gguf"), b"GGUF").unwrap();

    let (status, lines) = server
        .generate(json!({
            "model": "tiny-mm", "prompt": "Once upon a time", "stream": false,
            "options": { "num_predict": 8 },
        }))
        .await;
    assert_eq!(status, 200, "{lines:?}");
    assert_finished(&lines);
    {
        let models = server.state.models.read().await;
        let model = models.find("tiny-mm").expect("loaded");
        assert!(model.scheduler.is_some(), "on the scheduler");
        assert!(model.engine.is_none(), "not on the sequential engine");
        assert_eq!(model.facts.batch_size, 4);
    }

    let (status, objects) = server
        .post(
            "/api/chat",
            json!({
                "model": "tiny-mm", "stream": false,
                "messages": [{ "role": "user", "content": "what is this?",
                               "images": ["/9j/4AAQSkZJRg=="] }],
            }),
        )
        .await;
    assert_eq!(status, 503, "{objects:?}");
    let error = objects[0]["error"].as_str().expect("an error");
    assert!(error.contains("--no-mmproj"), "{error}");
}

/// A conversation's next turn on a model with sliding-window layers (Gemma),
/// whose window keeps only its last n_swa + n_ubatch tokens per slot. The
/// first turn's answer runs long enough that the window no longer reaches
/// back to where the next prompt parts from it, 4 tokens before the first
/// prompt's end: read on from there, the next prompt's first token would miss
/// what it must see, and answer as if it had never been said. The window kept
/// at that point (`StateCheckpoint`) is restored instead, and the answer must
/// be the one the whole prompt read from the start gets, with one slot and
/// with two; and reading it must take a fraction of reading it whole, or
/// nothing was restored.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a GGUF with a sliding window in EULLM_SWA_TEST_MODEL"]
async fn real_model_a_sliding_window_models_next_turn_restores_its_window() {
    let configure = |slots: usize| {
        move |state: &mut AppState| {
            state.ctx_size = 4096 * slots as u32;
            state.batch_size = slots;
            state.n_batch = 128;
            state.n_ubatch = Some(32);
        }
    };
    let ask = |prompt: &str, cache_prompt: bool, num_predict: u32| {
        json!({
            "model": "swa", "prompt": prompt, "raw": true, "stream": false,
            "cache_prompt": cache_prompt,
            "options": { "temperature": 0, "seed": 1, "num_predict": num_predict },
        })
    };
    let base = "The harbour office kept a ledger of every ship that entered the bay, \
                with the hour, the cargo and the weather at the mouth of the river."
        .repeat(30);
    let first = format!("{base} alpha beta gamma delta");
    let second = format!("{base} one two three four five six seven");

    let fresh = start_on(&swa_test_model(), &["swa"], configure(1)).await;
    let (status, whole) = fresh.generate(ask(&second, false, 16)).await;
    assert_eq!(status, 200, "{whole:?}");
    assert_finished(&whole);
    let whole_ns = whole.last().unwrap()["prompt_eval_duration"].as_u64().unwrap();

    for slots in [1, 2] {
        let server = start_on(&swa_test_model(), &["swa"], configure(slots)).await;
        let (status, lines) = server.generate(ask(&first, true, 600)).await;
        assert_eq!(status, 200, "{lines:?}");
        assert_finished(&lines);
        let generated = lines.last().unwrap()["eval_count"].as_u64().unwrap();
        assert!(generated > 600 / 2, "the first answer ran {generated} tokens: too short to slide the window past");
        let (status, next) = server.generate(ask(&second, true, 16)).await;
        assert_eq!(status, 200, "{next:?}");
        assert_finished(&next);
        assert_eq!(answer(&next), answer(&whole), "{slots} slots");
        let next_ns = next.last().unwrap()["prompt_eval_duration"].as_u64().unwrap();
        assert!(
            next_ns * 4 < whole_ns,
            "{slots} slots: the next turn read its prompt in {next_ns} ns, the whole one in \
             {whole_ns} ns: the kept window was not restored"
        );
    }
}

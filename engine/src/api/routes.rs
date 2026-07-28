//! Route definitions for the EULLM and OpenAI-compatible APIs.
//!
//! Supports both non-streaming (JSON) and streaming (SSE) responses
//! on all generation endpoints. Set `"stream": true` in the request body
//! to get token-by-token Server-Sent Events.
//!
//! When a `SchedulerHandle` is present, requests are dispatched through the
//! continuous batching scheduler. Otherwise the sequential `InferenceEngine`
//! is used as fallback.
//!
//! **Dynamic model swap:** if a request specifies a `model` that differs from
//! the currently loaded one, the server automatically swaps to the new model.

use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::IntoResponse;
use axum::response::sse::{Event, Sse};
use axum::{Json, Router, routing::get, routing::post};
use futures_util::stream::Stream;
use serde_json::{Value, json};
use tokio::sync::mpsc;

use super::AppState;
use crate::audit::{AuditEntry, AuditLogger};
use crate::inference::{
    GenerateRequest, InferenceEngine, JSON_GBNF, SchedulerHandle, StopReason, StreamEvent,
};
use crate::models::EU_CATALOG;
use crate::tools;

type S = Arc<AppState>;

/// Upper bound accepted for a request's `batch_size` override.
///
/// The override reaches `SchedulerConfig::max_batch_size` (one KV-cache slot
/// each) and `queue_capacity` (= `batch_size * 8`), so an unbounded value both
/// asks for an absurd allocation and — because slot ids are `i32` — can wrap to
/// zero usable slots while still reporting the model as loaded. 64 is far above
/// any real concurrency for a single-GPU appliance.
const MAX_BATCH_SIZE_OVERRIDE: usize = 64;

/// Bounds accepted for a request's `ctx_size` override. The lower bound leaves
/// room for a prompt plus at least one output token; the upper bound is the
/// largest context any current architecture declares, and keeps the value from
/// becoming a KV-cache allocation that fails *after* `swap_model` has already
/// unloaded the previous model.
const MIN_CTX_SIZE_OVERRIDE: u32 = 512;
const MAX_CTX_SIZE_OVERRIDE: u32 = 1_048_576;

/// Error shape returned by the JSON handlers: an HTTP status plus a JSON body.
type ApiError = (StatusCode, Json<Value>);

/// Validated `(batch_size, ctx_size)` slot overrides read from a request body.
/// `None` in either position means "keep the launch-time value".
type SlotOverrides = (Option<usize>, Option<u32>);

/// Read the `batch_size` / `ctx_size` slot overrides from a request body,
/// rejecting anything outside a serviceable range with HTTP 400.
///
/// Both fields are forwarded to `AppState::swap_model`, which rebuilds the
/// scheduler and reallocates the KV cache. A value that is merely *parsed*
/// rather than *validated* therefore turns a single request into a
/// configuration change that can leave the server unable to serve anything
/// until it is restarted — so an out-of-range value has to fail loudly here
/// instead of being clamped silently (a client asking for 4096 slots has a bug
/// worth surfacing, not an intent worth guessing).
///
/// An absent field yields `None` (keep the launch-time value). A field that is
/// present but not a non-negative integer is an error, not an absence: silently
/// ignoring `{"batch_size": -1}` would hide a client bug behind unchanged
/// behaviour.
fn parse_slot_overrides(body: &Value) -> Result<SlotOverrides, ApiError> {
    let bad = |field: &str, detail: String| {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": format!("invalid `{field}`: {detail}") })),
        )
    };

    let batch_size = match body.get("batch_size") {
        None | Some(Value::Null) => None,
        Some(v) => {
            let n = v
                .as_u64()
                .ok_or_else(|| bad("batch_size", "expected a non-negative integer".into()))?;
            if n == 0 || n > MAX_BATCH_SIZE_OVERRIDE as u64 {
                return Err(bad(
                    "batch_size",
                    format!("must be between 1 and {MAX_BATCH_SIZE_OVERRIDE}, got {n}"),
                ));
            }
            Some(n as usize)
        }
    };

    let ctx_size = match body.get("ctx_size") {
        None | Some(Value::Null) => None,
        Some(v) => {
            let n = v
                .as_u64()
                .ok_or_else(|| bad("ctx_size", "expected a non-negative integer".into()))?;
            if n < MIN_CTX_SIZE_OVERRIDE as u64 || n > MAX_CTX_SIZE_OVERRIDE as u64 {
                return Err(bad(
                    "ctx_size",
                    format!(
                        "must be between {MIN_CTX_SIZE_OVERRIDE} and {MAX_CTX_SIZE_OVERRIDE}, got {n}"
                    ),
                ));
            }
            Some(n as u32)
        }
    };

    Ok((batch_size, ctx_size))
}

/// EULLM native API routes (`/api/*`).
pub fn api_routes() -> Router<S> {
    Router::new()
        .route("/tags", get(list_models))
        .route("/generate", post(generate))
        .route("/chat", post(chat))
        .route("/show", post(show_model))
        .route("/pull", post(pull_model))
        .route("/version", get(version))
        .route("/unload", post(unload_model))
}

/// OpenAI-compatible routes (`/v1/*`).
pub fn openai_routes() -> Router<S> {
    Router::new()
        .route("/models", get(list_models_openai))
        .route("/chat/completions", post(chat_completions))
}

// ── Model slot and dynamic swap ──────────────────────────────────────────────

/// A snapshot of the model slot — cloned handles that don't hold the RwLock.
/// The `Arc` / `SchedulerHandle` clones are cheap (refcount bump + channel clone)
/// and keep the old model alive until in-flight requests finish.
struct SlotSnapshot {
    model_name: String,
    engine: Option<Arc<InferenceEngine>>,
    scheduler: Option<SchedulerHandle>,
}

/// Ensure the requested model is loaded and return a snapshot of the slot.
///
/// If `requested` differs from the currently loaded model, triggers a
/// dynamic model swap (unloads old, loads new).  In-flight requests on
/// cloned handles of the old model complete normally.
///
/// If no model is specified in the request, uses whatever is loaded.
async fn ensure_model(
    state: &AppState,
    requested: Option<&str>,
    override_batch_size: Option<usize>,
    override_ctx_size: Option<u32>,
) -> Result<SlotSnapshot, (StatusCode, Json<Value>)> {
    // Check if a swap is needed.
    if let Some(name) = requested {
        let normalized = name.replace(':', "-");
        let needs_swap = {
            let slot = state.slot.read().await;
            match slot.model_name.as_deref() {
                Some(loaded) => !model_names_match(loaded, &normalized),
                None => true,
            }
        };
        if needs_swap {
            state
                .swap_model(name, override_batch_size, override_ctx_size)
                .await
                .map_err(|e| match e {
                    // A model that does not exist is a client mistake, and a
                    // 5xx here is actively harmful: clients with automatic
                    // retry treat it as transient and hammer a request that
                    // can never succeed. Ollama answers 404 for this.
                    crate::api::ModelError::NotFound(msg) => {
                        (StatusCode::NOT_FOUND, Json(json!({ "error": msg })))
                    }
                    // The model exists but would not load: out of VRAM, a
                    // corrupt GGUF, a context that will not allocate. That is
                    // ours, and 500 is correct.
                    crate::api::ModelError::LoadFailed(msg) => (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(json!({ "error": format!("Failed to load model '{name}': {msg}") })),
                    ),
                })?;
        }
    }

    // Take a read-lock snapshot (cheap clones: Arc bump + channel clone).
    let slot = state.slot.read().await;
    if slot.engine.is_none() && slot.scheduler.is_none() {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(
                json!({ "error": "No model loaded. Send a request with a \"model\" field, or use `eullm run <model>`." }),
            ),
        ));
    }

    Ok(SlotSnapshot {
        model_name: slot.model_name.clone().unwrap_or_else(|| "unknown".into()),
        engine: slot.engine.clone(),
        scheduler: slot.scheduler.clone(),
    })
}

/// Parsed sampling parameters from the API request.
struct SamplingParams {
    max_tokens: u32,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    min_p: f32,
    repeat_penalty: f32,
    repeat_last_n: i32,
    seed: Option<u32>,
    num_ctx: Option<u32>,
}

fn parse_generate_params(body: &Value) -> SamplingParams {
    // Check top-level first (OpenAI format), then Ollama's options object.
    let options = body.get("options");

    let get = |key: &str| -> Option<&Value> {
        body.get(key).or_else(|| options.and_then(|o| o.get(key)))
    };

    // Ollama's real default is unbounded (-1: generate until context is
    // full or a stop condition), not a small fixed cap — confirmed against
    // Ollama's own docs/source, see ollama/ollama#7691. Leaving this
    // unbounded here is safe because prefill_sequence() always clamps it to
    // the remaining context budget regardless of what's requested.
    let max_tokens = get("max_tokens")
        .or_else(|| get("num_predict"))
        .and_then(|v| v.as_u64())
        .unwrap_or(u32::MAX as u64) as u32;
    // Defaults match Ollama: temperature=0.8, top_k=40, top_p=0.9, repeat_penalty=1.1
    let temperature = get("temperature").and_then(|v| v.as_f64()).unwrap_or(0.8) as f32;
    let top_k = get("top_k").and_then(|v| v.as_i64()).unwrap_or(40) as i32;
    let top_p = get("top_p").and_then(|v| v.as_f64()).unwrap_or(0.9) as f32;
    let min_p = get("min_p").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
    let repeat_penalty = get("repeat_penalty")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.1) as f32;
    let repeat_last_n = get("repeat_last_n").and_then(|v| v.as_i64()).unwrap_or(64) as i32;
    let seed = get("seed").and_then(|v| v.as_u64()).map(|v| v as u32);
    let num_ctx = get("num_ctx").and_then(|v| v.as_u64()).map(|v| v as u32);

    tracing::info!(
        "Request params: max_tokens={max_tokens}, temp={temperature:.2}, top_k={top_k}, top_p={top_p:.2}, repeat_penalty={repeat_penalty:.2}, num_ctx={num_ctx:?}"
    );
    SamplingParams {
        max_tokens,
        temperature,
        top_k,
        top_p,
        min_p,
        repeat_penalty,
        repeat_last_n,
        seed,
        num_ctx,
    }
}

fn is_streaming(body: &Value) -> bool {
    body.get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

/// Parse the `format` field from an Ollama-style request.
///
/// When `format` is `"json"`, returns the built-in JSON GBNF grammar so that
/// constrained decoding forces the model to produce valid JSON — matching
/// Ollama's behavior.
fn parse_format_grammar(body: &Value) -> Option<String> {
    let fmt = body.get("format").and_then(|v| v.as_str())?;
    if fmt == "json" {
        Some(JSON_GBNF.to_string())
    } else {
        None
    }
}

// ── Multimodal MVP helpers ─────────────────────────────────────────────
// Scope: a single user turn whose last message carries `images: [<base64>]`
// (Ollama convention). The history is ignored for now — vision turns are
// treated as one-shot probes. Gemma 4 is the only vision family in our
// catalog, so the chat template here is hard-coded; switch on
// `template.family()` when more land.

/// Pull base64-encoded images from the LAST user message. Returns
/// `(text, media)` with `media` non-empty only when the client attached
/// images. Accepts both raw base64 and the `data:...;base64,...` prefix.
#[cfg(feature = "multimodal")]
fn extract_multimodal_payload(messages: &[Value]) -> Option<(String, Vec<Vec<u8>>)> {
    use base64::{Engine as _, engine::general_purpose::STANDARD};
    let last_user = messages
        .iter()
        .rev()
        .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))?;
    let images_arr = last_user.get("images")?.as_array()?;
    if images_arr.is_empty() {
        return None;
    }
    let mut media = Vec::with_capacity(images_arr.len());
    for v in images_arr {
        let s = v.as_str()?;
        // `data:image/jpeg;base64,XXX` → keep only the payload after the comma.
        let payload = s.rsplit(',').next().unwrap_or(s);
        media.push(STANDARD.decode(payload).ok()?);
    }
    let text = last_user
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();
    Some((text, media))
}

/// Wrap a user prompt in the Gemma chat template with an mtmd media marker
/// placed inside the user turn (matches `run_multimodal_oneshot` in main.rs).
#[cfg(feature = "multimodal")]
fn gemma_multimodal_prompt(user_text: &str) -> String {
    let marker = llama_cpp_2::mtmd::mtmd_default_marker();
    format!("<start_of_turn>user\n{marker}\n{user_text}<end_of_turn>\n<start_of_turn>model\n")
}

/// Background mtmd-aware generation, mirroring `sequential_to_channel`.
#[cfg(feature = "multimodal")]
fn multimodal_to_channel(
    engine: Arc<InferenceEngine>,
    request: GenerateRequest,
    media: Vec<Vec<u8>>,
) -> mpsc::Receiver<StreamEvent> {
    let (tx, rx) = mpsc::channel::<StreamEvent>(64);
    tokio::task::spawn_blocking(move || {
        engine.generate_multimodal(&request, &media, tx);
    });
    rx
}

/// Format chat messages into a prompt string using the model-appropriate template.
/// When `think` is false, suppresses Qwen3 thinking mode (ChatML only).
fn format_chat_prompt(messages: &[Value], think: bool, model_name: &str) -> String {
    let template = crate::chat_template::ChatTemplate::detect(model_name);
    let pairs: Vec<(&str, &str)> = messages
        .iter()
        .map(|msg| {
            let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");
            let content = msg.get("content").and_then(|v| v.as_str()).unwrap_or("");
            (role, content)
        })
        .collect();
    template.build_prompt(&pairs, think)
}

/// Collect all tokens from a receiver into a final result.
async fn collect_stream(mut rx: mpsc::Receiver<StreamEvent>) -> Result<Collected, String> {
    let mut text = String::new();
    let mut tokens_generated = 0u32;
    let mut tokens_prompt = 0u32;
    let mut duration_ms = 0u64;
    // If the stream ends without a Done event the answer is incomplete, so
    // Length is the honest default rather than Stop.
    let mut stop_reason = StopReason::Length;

    while let Some(event) = rx.recv().await {
        match event {
            StreamEvent::Token(piece) => text.push_str(&piece),
            StreamEvent::Done {
                tokens_generated: tg,
                tokens_prompt: tp,
                duration_ms: d,
                stop_reason: sr,
            } => {
                tokens_generated = tg;
                tokens_prompt = tp;
                duration_ms = d;
                stop_reason = sr;
                break;
            }
            StreamEvent::Error(e) => return Err(e),
        }
    }

    Ok(Collected {
        text,
        tokens_generated,
        tokens_prompt,
        duration_ms,
        stop_reason,
    })
}

/// What a non-streaming request produced, including *why* it stopped.
struct Collected {
    text: String,
    tokens_generated: u32,
    tokens_prompt: u32,
    duration_ms: u64,
    stop_reason: StopReason,
}

// ── Web content injection ────────────────────────────────────────────────────

/// If web browsing is enabled and the last user message contains URLs,
/// fetch each URL and inject the content as a system-style message
/// prepended to the conversation (before the last user turn).
///
/// Returns a new messages Vec with injected content, or the original unchanged.
async fn inject_web_content(
    mut messages: Vec<Value>,
    web_enabled: bool,
    ctx_size: u32,
    batch_size: usize,
    web_policy: &crate::tools::guard::WebPolicy,
) -> Vec<Value> {
    if !web_enabled {
        return messages;
    }
    // With continuous batching each slot gets ctx_size/batch_size tokens.
    // Use per-slot context as the budget so injected content actually fits.
    let effective_ctx = if batch_size > 1 {
        ctx_size / batch_size as u32
    } else {
        ctx_size
    };

    // Find last user message
    let last_user_idx = messages
        .iter()
        .rposition(|m| m.get("role").and_then(|v| v.as_str()) == Some("user"));
    let idx = match last_user_idx {
        Some(i) => i,
        None => return messages,
    };

    let user_text = messages[idx]
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let urls = tools::extract_urls(&user_text);
    if urls.is_empty() {
        return messages;
    }

    // Estimate prompt chars already used (rough: all messages joined)
    let existing_chars: usize = messages
        .iter()
        .map(|m| {
            m.get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .len()
        })
        .sum();

    let mut injections: Vec<String> = Vec::new();
    for url in &urls {
        match tools::fetch_for_context(url, effective_ctx, existing_chars, &user_text, web_policy)
            .await
        {
            Ok((content, truncated)) => {
                let note = if truncated {
                    " [content truncated to fit context]"
                } else {
                    ""
                };
                injections.push(format!("[Web content from {url}{note}]\n\n{content}"));
                tracing::info!(
                    "Web: fetched {} ({} chars{})",
                    crate::audit::sanitize_for_log(url),
                    content.len(),
                    if truncated { ", truncated" } else { "" }
                );
            }
            Err(e) => {
                injections.push(format!("[Failed to fetch {url}: {e}]"));
                tracing::warn!(
                    "Web: fetch failed for {}: {e}",
                    crate::audit::sanitize_for_log(url)
                );
            }
        }
    }

    if injections.is_empty() {
        return messages;
    }

    // Instruction-tuned models (Gemma 4 12B observed) say "I can't browse the
    // web" while at the same time summarising the page just handed to them.
    // Tell them up-front that the fetch already happened on their behalf, so
    // the response is consistent with what they actually have in context.
    let preamble = "\
You have a web browsing capability provided by the inference engine. When the \
user shares a URL, the engine automatically fetches the page and gives you its \
plain-text content below. Treat the content below as pages YOU navigated and \
read. Do NOT claim you cannot browse the web — you just did. You may cite the \
source URL when answering.\n\n---\n\n";

    let web_message = json!({
        "role": "system",
        "content": format!("{preamble}{}", injections.join("\n\n---\n\n"))
    });
    messages.insert(idx, web_message);
    messages
}

// ── EULLM API handlers ──────────────────────────────────────────────────────

async fn version(State(state): State<S>) -> Json<Value> {
    // `api_port` is an EULLM extension: the chat UI (served on its own port)
    // reads it to display the canonical API endpoint. Ollama clients only
    // look at `version` and ignore unknown fields.
    Json(json!({
        "version": env!("CARGO_PKG_VERSION"),
        "api_port": state.api_port,
    }))
}

/// Unload the currently loaded model, freeing its VRAM, without loading a
/// replacement. EULLM extension (not part of the Ollama API) — the primary
/// use case is handing GPU memory to a co-resident process (e.g. an
/// embedding server used during RAG document ingestion) without restarting
/// eullm. Send a request with a `model` field afterwards (or run `eullm run
/// <model>` again) to load a model back in.
async fn unload_model(State(state): State<S>) -> (StatusCode, Json<Value>) {
    match state.unload().await {
        Ok(Some(name)) => (StatusCode::OK, Json(json!({ "unloaded": name }))),
        Ok(None) => (
            StatusCode::OK,
            Json(json!({ "unloaded": null, "message": "no model was loaded" })),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("Failed to unload model: {e}") })),
        ),
    }
}

/// List models — returns the currently loaded model (like Ollama) plus catalog entries.
///
/// Ollama's `/api/tags` returns all locally available models.  We return the
/// currently loaded model first (so health-check dashboards see it), followed
/// by catalog entries for discoverability.
async fn list_models(State(state): State<S>) -> Json<Value> {
    let mut models: Vec<Value> = Vec::new();

    let loaded_name = {
        let slot = state.slot.read().await;
        slot.model_name.clone()
    };

    // If a model is loaded, include it first (this is what dashboards check)
    if let Some(ref name) = loaded_name {
        models.push(json!({
            "name": name,
            "size": 0,
            "digest": "",
            "loaded": true,
            "details": {
                "format": "gguf",
                "family": "",
                "parameter_size": "",
                "quantization_level": "Q4_K_M",
            }
        }));
    }

    // Add catalog entries (skip duplicates if the loaded model is in the catalog).
    // The Ollama-compatible `name` field MUST be the addressable id — clients
    // pass it back as the `model` field in chat/generate requests, and that's
    // the only string the model store / picker knows how to resolve. The human
    // catalog name is exposed alongside as `details.display_name` for UIs that
    // want to show it.
    for m in EU_CATALOG.iter() {
        if loaded_name.as_deref() == Some(m.name.as_str())
            || loaded_name.as_deref() == Some(m.id.as_str())
        {
            // Replace the placeholder entry above with full catalog metadata.
            // `loaded` must survive that replacement: it is the only thing in
            // the response that says which model is in the slot. Clients used
            // to infer it from the empty digest of the placeholder, and this
            // line overwrites that with the catalog's real digest — so a
            // loaded *catalog* model looked exactly like one that had never
            // been downloaded, while a loaded raw `.gguf` path (which never
            // reaches this branch) looked correct. That is why the chat UI
            // reported "No model loaded" only after picking from the picker.
            if let Some(first) = models.first_mut() {
                *first = json!({
                    "name": m.id,
                    "size": m.size_bytes,
                    "digest": m.digest,
                    "loaded": true,
                    "downloaded": true,
                    "details": {
                        "format": "gguf",
                        "family": m.base(),
                        "parameter_size": format!("{:.1}B", m.params_b),
                        "quantization_level": m.quantization,
                        "domain": m.domain,
                        "source_model": m.source_model(),
                        "display_name": m.name,
                    }
                });
            }
            continue;
        }
        // Whether the weights are on disk. Ollama has no equivalent field
        // because its `/api/tags` lists only what has been pulled; ours lists
        // the whole catalog, and without this a client cannot tell a model it
        // can run right now from one that would first download several
        // gigabytes. The chat UI disabled every catalog entry for exactly that
        // reason, which made a downloaded-but-not-loaded model unselectable
        // even though the server swaps to it on request.
        models.push(json!({
            "name": m.id,
            "size": m.size_bytes,
            "digest": m.digest,
            "downloaded": state.store.is_present(&m.id),
            "details": {
                "format": "gguf",
                "family": m.base(),
                "parameter_size": format!("{:.1}B", m.params_b),
                "quantization_level": m.quantization,
                "domain": m.domain,
                "source_model": m.source_model(),
                "display_name": m.name,
            }
        }));
    }

    // Models on disk that the catalog does not know about — anything pulled
    // from a URL or a HuggingFace repo. Same omission as `/v1/models` had: the
    // list was assembled from the catalog and the loaded slot, so a model the
    // user had deliberately downloaded was missing from it unless it happened
    // to be loaded at that moment.
    let already: std::collections::HashSet<String> = models
        .iter()
        .filter_map(|m| m.get("name").and_then(|v| v.as_str()).map(String::from))
        .collect();
    for m in state.store.list().unwrap_or_default() {
        if m.id.is_empty() || already.contains(&m.id) {
            continue;
        }
        models.push(json!({
            "name": m.id,
            "size": m.size_bytes,
            "digest": m.digest,
            "downloaded": state.store.is_present(&m.id),
            "details": {
                "format": "gguf",
                "family": m.base,
                "parameter_size": "",
                "quantization_level": "",
                "display_name": m.name,
            }
        }));
    }

    Json(json!({ "models": models }))
}

async fn generate(
    State(state): State<S>,
    // Present on every request: the auth middleware inserts an anonymous
    // identity when no keys are configured, so this never fails to extract.
    axum::Extension(identity): axum::Extension<super::Identity>,
    Json(body): Json<Value>,
) -> Result<axum::response::Response, (StatusCode, Json<Value>)> {
    let user_id = identity.key_id().map(str::to_string);
    let requested = body.get("model").and_then(|v| v.as_str());
    let (override_batch_size, override_ctx_size) = parse_slot_overrides(&body)?;
    let snap = ensure_model(&state, requested, override_batch_size, override_ctx_size).await?;
    let model = snap.model_name.clone();

    let prompt = body
        .get("prompt")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let sp = parse_generate_params(&body);
    let raw = body.get("raw").and_then(|v| v.as_bool()).unwrap_or(false);
    let grammar = if raw {
        // GBNF grammar sampling is incompatible with raw mode — the grammar
        // sampler crashes (GGML_ASSERT) when the prompt contains pre-tokenized
        // special tokens (ChatML).  In raw mode, the caller handles formatting.
        None
    } else {
        parse_format_grammar(&body)
    };

    let request = GenerateRequest {
        prompt,
        max_tokens: sp.max_tokens,
        temperature: sp.temperature,
        top_k: sp.top_k,
        top_p: sp.top_p,
        min_p: sp.min_p,
        repeat_penalty: sp.repeat_penalty,
        repeat_last_n: sp.repeat_last_n,
        seed: sp.seed,
        num_ctx: sp.num_ctx,
        grammar,
        raw,
        ..Default::default()
    };

    if let Some(ref sched) = snap.scheduler {
        // ── Continuous batching path ────────────────────────────────
        let rx = sched.submit(request);

        if is_streaming(&body) {
            Ok(ndjson_stream_response(
                rx,
                model,
                StreamFormat::OllamaGenerate,
                user_id,
            ))
        } else {
            let Collected {
                text,
                tokens_generated,
                tokens_prompt,
                duration_ms,
                stop_reason,
            } = collect_stream(rx).await.map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": e })),
                )
            })?;

            let mut audit = AuditEntry::new(model.clone(), "generate".to_string());
            audit.input_tokens = tokens_prompt;
            audit.output_tokens = tokens_generated;
            audit.duration_ms = duration_ms;
            audit.user_id = user_id.clone();
            AuditLogger::new().log(&audit);

            Ok(Json(json!({
                "model": model,
                "created_at": chrono::Utc::now().to_rfc3339(),
                "response": text,
                "done": true,
                "done_reason": stop_reason.as_api_str(),
                "total_duration": duration_ms * 1_000_000,
                "load_duration": 0,
                "prompt_eval_count": tokens_prompt,
                "prompt_eval_duration": 0,
                "eval_count": tokens_generated,
                "eval_duration": duration_ms * 1_000_000
            }))
            .into_response())
        }
    } else {
        // ── Sequential fallback ────────────────────────────────────
        let engine = Arc::clone(snap.engine.as_ref().unwrap());

        if is_streaming(&body) {
            let rx = sequential_to_channel(engine, request);
            Ok(ndjson_stream_response(
                rx,
                model,
                StreamFormat::OllamaGenerate,
                user_id,
            ))
        } else {
            let result = tokio::task::spawn_blocking({
                let engine = engine;
                move || engine.generate(&request)
            })
            .await
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": format!("Task error: {e}") })),
                )
            })?
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": format!("Inference error: {e}") })),
                )
            })?;

            let mut audit = AuditEntry::new(model.clone(), "generate".to_string());
            audit.input_tokens = result.tokens_prompt;
            audit.output_tokens = result.tokens_generated;
            audit.duration_ms = result.duration_ms;
            audit.user_id = user_id.clone();
            AuditLogger::new().log(&audit);

            Ok(Json(json!({
                "model": model,
                "created_at": chrono::Utc::now().to_rfc3339(),
                "response": result.text,
                "done": true,
                "done_reason": result.stop_reason.as_api_str(),
                "total_duration": result.duration_ms * 1_000_000,
                "load_duration": 0,
                "prompt_eval_count": result.tokens_prompt,
                "prompt_eval_duration": 0,
                "eval_count": result.tokens_generated,
                "eval_duration": result.duration_ms * 1_000_000
            }))
            .into_response())
        }
    }
}

async fn chat(
    State(state): State<S>,
    // Present on every request: the auth middleware inserts an anonymous
    // identity when no keys are configured, so this never fails to extract.
    axum::Extension(identity): axum::Extension<super::Identity>,
    Json(body): Json<Value>,
) -> Result<axum::response::Response, (StatusCode, Json<Value>)> {
    let user_id = identity.key_id().map(str::to_string);
    let requested = body.get("model").and_then(|v| v.as_str());
    let (override_batch_size, override_ctx_size) = parse_slot_overrides(&body)?;
    let snap = ensure_model(&state, requested, override_batch_size, override_ctx_size).await?;
    let model = snap.model_name.clone();

    let messages = body
        .get("messages")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let messages = inject_web_content(
        messages,
        state.web_enabled,
        state.ctx_size,
        state.batch_size,
        &state.web_policy,
    )
    .await;

    // A build without the `multimodal` feature has no branch below at all, so
    // an `images` array used to be dropped on the floor and the request went
    // through as plain text. The model, asked about a picture it never
    // received, answers that it cannot see one — which reads as a limitation
    // of the model rather than of the binary. Reported from a source build
    // made with `--features vulkan` (issue #286), where the omission is easy:
    // the published binaries all carry multimodal, a hand-built one need not.
    #[cfg(not(feature = "multimodal"))]
    if messages
        .iter()
        .rev()
        .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
        .and_then(|m| m.get("images"))
        .and_then(|v| v.as_array())
        .is_some_and(|a| !a.is_empty())
    {
        return Err((
            StatusCode::NOT_IMPLEMENTED,
            Json(json!({
                "error": "This build has no multimodal support, so the attached media cannot be                           read. The published binaries include it; a build from source needs                           `--features multimodal` (combine it with a backend, e.g.                           `--features \"vulkan,multimodal\"`)."
            })),
        ));
    }

    // ── Multimodal MVP branch ──────────────────────────────────────────
    // If the last user message carries `images`, route through the
    // sequential mtmd path. `swap_model` forces sequential mode when the
    // loaded model has an mmproj, so `snap.engine` is expected to be Some.
    // If it isn't, the operator loaded a text-only model and the client is
    // trying to send pictures anyway → 503 with an explicit message.
    #[cfg(feature = "multimodal")]
    if let Some((user_text, media)) = extract_multimodal_payload(&messages) {
        let engine = match snap.engine.as_ref() {
            Some(e) => Arc::clone(e),
            None => {
                return Err((
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(json!({
                        "error": "Multimodal request but engine is in batched (text-only) mode \
                                 — the loaded model has no mmproj projector."
                    })),
                ));
            }
        };
        let sp = parse_generate_params(&body);
        let mm_request = GenerateRequest {
            prompt: gemma_multimodal_prompt(&user_text),
            max_tokens: sp.max_tokens,
            temperature: sp.temperature,
            top_k: sp.top_k,
            top_p: sp.top_p,
            min_p: sp.min_p,
            repeat_penalty: sp.repeat_penalty,
            repeat_last_n: sp.repeat_last_n,
            seed: sp.seed,
            num_ctx: sp.num_ctx,
            // Hand-built Gemma template with the mtmd marker — do NOT let
            // generate() add its own BOS/template on top.
            raw: true,
            // The same list the text path uses. It was spelled out here, so a
            // stop sequence added for Gemma applied everywhere except the one
            // path that only runs with a Gemma model.
            stop_sequences: crate::chat_template::ChatTemplate::Gemma.stop_sequences(),
            // Gemma has no think toggle, so nothing extra to strip here.
            filter_sequences: crate::inference::default_filters(true),
            grammar: None,
        };
        if is_streaming(&body) {
            let rx = multimodal_to_channel(engine, mm_request, media);
            return Ok(ndjson_stream_response(
                rx,
                model,
                StreamFormat::OllamaChat,
                user_id,
            ));
        }
        let rx = multimodal_to_channel(engine, mm_request, media);
        let Collected {
            text,
            tokens_generated,
            tokens_prompt,
            duration_ms,
            stop_reason,
        } = collect_stream(rx).await.map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e })),
            )
        })?;
        let mut audit = AuditEntry::new(model.clone(), "chat".to_string());
        audit.input_tokens = tokens_prompt;
        audit.output_tokens = tokens_generated;
        audit.duration_ms = duration_ms;
        audit.user_id = user_id.clone();
        AuditLogger::new().log(&audit);
        return Ok(Json(json!({
            "model": model,
            "created_at": chrono::Utc::now().to_rfc3339(),
            "message": { "role": "assistant", "content": text },
            "done": true,
            "done_reason": stop_reason.as_api_str(),
            "total_duration": duration_ms * 1_000_000,
            "load_duration": 0,
            "prompt_eval_count": tokens_prompt,
            "prompt_eval_duration": 0,
            "eval_count": tokens_generated,
            "eval_duration": duration_ms * 1_000_000
        }))
        .into_response());
    }

    let think = body.get("think").and_then(|v| v.as_bool()).unwrap_or(true);
    let model_name_ref: &str = &model;
    let template = crate::chat_template::ChatTemplate::detect(model_name_ref);
    let prompt = format_chat_prompt(&messages, think, model_name_ref);
    let sp = parse_generate_params(&body);
    let grammar = parse_format_grammar(&body);

    let mut stop_sequences = template.stop_sequences();
    stop_sequences.push("<|end|>".to_string());

    let request = GenerateRequest {
        prompt,
        max_tokens: sp.max_tokens,
        temperature: sp.temperature,
        top_k: sp.top_k,
        top_p: sp.top_p,
        min_p: sp.min_p,
        repeat_penalty: sp.repeat_penalty,
        repeat_last_n: sp.repeat_last_n,
        seed: sp.seed,
        num_ctx: sp.num_ctx,
        stop_sequences,
        // think-aware: with think:false a think tag in the OUTPUT is
        // spurious, since the prompt already carries a closed empty block.
        filter_sequences: crate::inference::default_filters(think),
        grammar,
        raw: false,
    };

    if let Some(ref sched) = snap.scheduler {
        let rx = sched.submit(request);

        if is_streaming(&body) {
            Ok(ndjson_stream_response(
                rx,
                model,
                StreamFormat::OllamaChat,
                user_id,
            ))
        } else {
            let Collected {
                text,
                tokens_generated,
                tokens_prompt,
                duration_ms,
                stop_reason,
            } = collect_stream(rx).await.map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": e })),
                )
            })?;

            let mut audit = AuditEntry::new(model.clone(), "chat".to_string());
            audit.input_tokens = tokens_prompt;
            audit.output_tokens = tokens_generated;
            audit.duration_ms = duration_ms;
            audit.user_id = user_id.clone();
            AuditLogger::new().log(&audit);

            Ok(Json(json!({
                "model": model,
                "created_at": chrono::Utc::now().to_rfc3339(),
                "message": {
                    "role": "assistant",
                    "content": text
                },
                "done": true,
                "done_reason": stop_reason.as_api_str(),
                "total_duration": duration_ms * 1_000_000,
                "load_duration": 0,
                "prompt_eval_count": tokens_prompt,
                "prompt_eval_duration": 0,
                "eval_count": tokens_generated,
                "eval_duration": duration_ms * 1_000_000
            }))
            .into_response())
        }
    } else {
        let engine = Arc::clone(snap.engine.as_ref().unwrap());

        if is_streaming(&body) {
            let rx = sequential_to_channel(engine, request);
            Ok(ndjson_stream_response(
                rx,
                model,
                StreamFormat::OllamaChat,
                user_id,
            ))
        } else {
            let result = tokio::task::spawn_blocking({
                let engine = engine;
                move || engine.generate(&request)
            })
            .await
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": format!("Task error: {e}") })),
                )
            })?
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": format!("Inference error: {e}") })),
                )
            })?;

            let mut audit = AuditEntry::new(model.clone(), "chat".to_string());
            audit.input_tokens = result.tokens_prompt;
            audit.output_tokens = result.tokens_generated;
            audit.duration_ms = result.duration_ms;
            audit.user_id = user_id.clone();
            AuditLogger::new().log(&audit);

            Ok(Json(json!({
                "model": model,
                "created_at": chrono::Utc::now().to_rfc3339(),
                "message": {
                    "role": "assistant",
                    "content": result.text
                },
                "done": true,
                "done_reason": result.stop_reason.as_api_str(),
                "total_duration": result.duration_ms * 1_000_000,
                "load_duration": 0,
                "prompt_eval_count": result.tokens_prompt,
                "prompt_eval_duration": 0,
                "eval_count": result.tokens_generated,
                "eval_duration": result.duration_ms * 1_000_000
            }))
            .into_response())
        }
    }
}

async fn show_model(Json(body): Json<Value>) -> Json<Value> {
    let name = body
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    if let Some(entry) = crate::models::catalog::find_model(name) {
        Json(json!({
            "modelfile": format!(
                "# EULLM model: {}\n# Domain: {}\n# Base: {}\n# Source: {}\n# License: {}",
                entry.name, entry.domain, entry.base(), entry.source_model(), entry.license
            ),
            "parameters": "num_ctx 4096\ntemperature 0.7",
            "template": "{{ .Prompt }}",
            "details": {
                "format": "gguf",
                "family": entry.base(),
                "parameter_size": format!("{:.1}B", entry.params_b),
                "quantization_level": entry.quantization,
                "domain": entry.domain,
                "source_model": entry.source_model()
            }
        }))
    } else {
        Json(json!({ "error": format!("model '{name}' not found") }))
    }
}

async fn pull_model(Json(body): Json<Value>) -> Json<Value> {
    let name = body
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    Json(json!({
        "status": format!("pulling {name} from EU registry (not yet implemented)")
    }))
}

// ── OpenAI-compatible handlers ───────────────────────────────────────────────

/// Ids of the models actually present in the local store, in listing order.
///
/// Both model-list endpoints answered from the catalog alone, so a model pulled
/// from a URL or a HuggingFace repo — anything not in the 22 curated entries —
/// was invisible to them even while it was loaded and answering. On `/v1/models`
/// that is the difference between usable and not: a coding editor picks the
/// model from that list, so a model it does not name cannot be selected at all
/// (issue #294).
fn local_model_ids(state: &AppState) -> Vec<String> {
    state
        .store
        .list()
        .unwrap_or_default()
        .into_iter()
        .filter(|m| !m.id.is_empty())
        .map(|m| m.id)
        .collect()
}

async fn list_models_openai(State(state): State<S>) -> Json<Value> {
    let mut data: Vec<Value> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    // What is on this machine comes first: it can be used right now, and it
    // includes models the catalog has never heard of.
    for id in local_model_ids(&state) {
        if seen.insert(id.clone()) {
            data.push(json!({
                "id": id,
                "object": "model",
                "created": 1700000000_u64,
                "owned_by": "eullm"
            }));
        }
    }

    // The model in the slot, when it was launched from a path rather than
    // pulled, has no manifest and so is not in the list above.
    if let Some(name) = state.slot.read().await.model_name.clone()
        && seen.insert(name.clone())
    {
        data.push(json!({
            "id": name,
            "object": "model",
            "created": 1700000000_u64,
            "owned_by": "eullm"
        }));
    }

    // The OpenAI `id` field is what clients echo back as the `model` parameter
    // in chat requests, so it has to be the addressable catalog id, not the
    // human display name.
    data.extend(
        EU_CATALOG
            .iter()
            .filter(|m| !seen.contains(&m.id))
            .map(|m| {
                json!({
                    "id": m.id,
                    "object": "model",
                    "created": 1700000000_u64,
                    "owned_by": "eullm"
                })
            }),
    );

    Json(json!({ "object": "list", "data": data }))
}

async fn chat_completions(
    State(state): State<S>,
    // Present on every request: the auth middleware inserts an anonymous
    // identity when no keys are configured, so this never fails to extract.
    axum::Extension(identity): axum::Extension<super::Identity>,
    Json(body): Json<Value>,
) -> Result<axum::response::Response, (StatusCode, Json<Value>)> {
    let user_id = identity.key_id().map(str::to_string);
    let requested = body.get("model").and_then(|v| v.as_str());
    let (override_batch_size, override_ctx_size) = parse_slot_overrides(&body)?;
    let snap = ensure_model(&state, requested, override_batch_size, override_ctx_size).await?;
    let model = snap.model_name.clone();

    let messages = body
        .get("messages")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let messages = inject_web_content(
        messages,
        state.web_enabled,
        state.ctx_size,
        state.batch_size,
        &state.web_policy,
    )
    .await;

    let think = body.get("think").and_then(|v| v.as_bool()).unwrap_or(true);
    let model_name_ref: &str = &model;
    let template = crate::chat_template::ChatTemplate::detect(model_name_ref);
    let prompt = format_chat_prompt(&messages, think, model_name_ref);
    let sp = parse_generate_params(&body);
    let grammar = parse_format_grammar(&body);

    let mut stop_sequences = template.stop_sequences();
    stop_sequences.push("<|end|>".to_string());

    let request = GenerateRequest {
        prompt,
        max_tokens: sp.max_tokens,
        temperature: sp.temperature,
        top_k: sp.top_k,
        top_p: sp.top_p,
        min_p: sp.min_p,
        repeat_penalty: sp.repeat_penalty,
        repeat_last_n: sp.repeat_last_n,
        seed: sp.seed,
        num_ctx: sp.num_ctx,
        stop_sequences,
        // think-aware: with think:false a think tag in the OUTPUT is
        // spurious, since the prompt already carries a closed empty block.
        filter_sequences: crate::inference::default_filters(think),
        grammar,
        raw: false,
    };

    if let Some(ref sched) = snap.scheduler {
        let rx = sched.submit(request);

        if is_streaming(&body) {
            let stream = stream_from_channel_sse(rx, model, StreamFormat::OpenAI, user_id);
            Ok(Sse::new(stream).into_response())
        } else {
            let Collected {
                text,
                tokens_generated,
                tokens_prompt,
                duration_ms,
                stop_reason,
            } = collect_stream(rx).await.map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": e })),
                )
            })?;

            let mut audit = AuditEntry::new(model.clone(), "chat.completions".to_string());
            audit.input_tokens = tokens_prompt;
            audit.output_tokens = tokens_generated;
            audit.duration_ms = duration_ms;
            audit.user_id = user_id.clone();
            AuditLogger::new().log(&audit);

            Ok(Json(json!({
                "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                "object": "chat.completion",
                "created": chrono::Utc::now().timestamp(),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text
                    },
                    "finish_reason": stop_reason.as_api_str()
                }],
                "usage": {
                    "prompt_tokens": tokens_prompt,
                    "completion_tokens": tokens_generated,
                    "total_tokens": tokens_prompt + tokens_generated
                }
            }))
            .into_response())
        }
    } else {
        let engine = Arc::clone(snap.engine.as_ref().unwrap());

        if is_streaming(&body) {
            let rx = sequential_to_channel(engine, request);
            let stream = stream_from_channel_sse(rx, model, StreamFormat::OpenAI, user_id);
            Ok(Sse::new(stream).into_response())
        } else {
            let result = tokio::task::spawn_blocking({
                let engine = engine;
                move || engine.generate(&request)
            })
            .await
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": format!("Task error: {e}") })),
                )
            })?
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": format!("Inference error: {e}") })),
                )
            })?;

            let mut audit = AuditEntry::new(model.clone(), "chat.completions".to_string());
            audit.input_tokens = result.tokens_prompt;
            audit.output_tokens = result.tokens_generated;
            audit.duration_ms = result.duration_ms;
            audit.user_id = user_id.clone();
            AuditLogger::new().log(&audit);

            Ok(Json(json!({
                "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                "object": "chat.completion",
                "created": chrono::Utc::now().timestamp(),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result.text
                    },
                    "finish_reason": result.stop_reason.as_api_str()
                }],
                "usage": {
                    "prompt_tokens": result.tokens_prompt,
                    "completion_tokens": result.tokens_generated,
                    "total_tokens": result.tokens_prompt + result.tokens_generated
                }
            }))
            .into_response())
        }
    }
}

// ── Streaming helpers ────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum StreamFormat {
    OllamaGenerate,
    OllamaChat,
    OpenAI,
}

/// Convert an mpsc channel of StreamEvents into an SSE event stream.
///
/// Used for OpenAI-compatible `/v1/chat/completions` streaming only.
fn stream_from_channel_sse(
    mut rx: mpsc::Receiver<StreamEvent>,
    model: String,
    format: StreamFormat,
    user_id: Option<String>,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
    async_stream::stream! {
        let completion_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());

        while let Some(event) = rx.recv().await {
            match event {
                StreamEvent::Token(piece) => {
                    let data = format_token_event(&piece, &model, &completion_id, format);
                    yield Ok(Event::default().data(data.to_string()));
                }
                StreamEvent::Done { tokens_generated, tokens_prompt, duration_ms, stop_reason } => {
                    // Audit log
                    let mut audit = AuditEntry::new(model.clone(), match format {
                        StreamFormat::OllamaGenerate => "generate",
                        StreamFormat::OllamaChat => "chat",
                        StreamFormat::OpenAI => "chat.completions",
                    }.to_string());
                    audit.input_tokens = tokens_prompt;
                    audit.output_tokens = tokens_generated;
                    audit.duration_ms = duration_ms;
                    audit.user_id = user_id.clone();
                    AuditLogger::new().log(&audit);

                    let data = format_done_event(
                        &model, &completion_id, format,
                        tokens_generated, tokens_prompt, duration_ms, stop_reason,
                    );
                    yield Ok(Event::default().data(data.to_string()));

                    if matches!(format, StreamFormat::OpenAI) {
                        yield Ok(Event::default().data("[DONE]"));
                    }
                    break;
                }
                StreamEvent::Error(msg) => {
                    let data = json!({ "error": msg });
                    yield Ok(Event::default().data(data.to_string()));
                    break;
                }
            }
        }
    }
}

/// Convert an mpsc channel of StreamEvents into an NDJSON stream.
///
/// Ollama uses NDJSON (newline-delimited JSON) for `/api/generate` and
/// `/api/chat` streaming — NOT SSE. Each line is a complete JSON object
/// followed by a newline. No `data:` prefix, no double newlines.
///
/// This is what Ollama clients (RAG Enterprise, Open WebUI, etc.) expect.
fn ndjson_stream_response(
    mut rx: mpsc::Receiver<StreamEvent>,
    model: String,
    format: StreamFormat,
    user_id: Option<String>,
) -> axum::response::Response {
    let stream = async_stream::stream! {
        let completion_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());

        while let Some(event) = rx.recv().await {
            match event {
                StreamEvent::Token(piece) => {
                    let data = format_token_event(&piece, &model, &completion_id, format);
                    let mut line = data.to_string();
                    line.push('\n');
                    yield Ok::<_, std::convert::Infallible>(line);
                }
                StreamEvent::Done { tokens_generated, tokens_prompt, duration_ms, stop_reason } => {
                    let mut audit = AuditEntry::new(model.clone(), match format {
                        StreamFormat::OllamaGenerate => "generate",
                        StreamFormat::OllamaChat => "chat",
                        StreamFormat::OpenAI => "chat.completions",
                    }.to_string());
                    audit.input_tokens = tokens_prompt;
                    audit.output_tokens = tokens_generated;
                    audit.duration_ms = duration_ms;
                    audit.user_id = user_id.clone();
                    AuditLogger::new().log(&audit);

                    let data = format_done_event(
                        &model, &completion_id, format,
                        tokens_generated, tokens_prompt, duration_ms, stop_reason,
                    );
                    let mut line = data.to_string();
                    line.push('\n');
                    yield Ok(line);
                    break;
                }
                StreamEvent::Error(msg) => {
                    let data = json!({ "error": msg });
                    let mut line = data.to_string();
                    line.push('\n');
                    yield Ok(line);
                    break;
                }
            }
        }
    };

    axum::response::Response::builder()
        .header(header::CONTENT_TYPE, "application/x-ndjson")
        .header(header::TRANSFER_ENCODING, "chunked")
        .body(Body::from_stream(stream))
        .unwrap()
}

/// Start sequential inference in a background thread, returning an mpsc receiver.
fn sequential_to_channel(
    engine: Arc<crate::inference::InferenceEngine>,
    request: GenerateRequest,
) -> mpsc::Receiver<StreamEvent> {
    let (tx, rx) = mpsc::channel::<StreamEvent>(32);
    tokio::task::spawn_blocking(move || {
        engine.generate_streaming(&request, tx);
    });
    rx
}

fn format_token_event(
    piece: &str,
    model: &str,
    completion_id: &str,
    format: StreamFormat,
) -> Value {
    match format {
        StreamFormat::OllamaGenerate => json!({
            "model": model,
            "created_at": chrono::Utc::now().to_rfc3339(),
            "response": piece,
            "done": false,
        }),
        StreamFormat::OllamaChat => json!({
            "model": model,
            "created_at": chrono::Utc::now().to_rfc3339(),
            "message": {
                "role": "assistant",
                "content": piece,
            },
            "done": false,
        }),
        StreamFormat::OpenAI => json!({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": chrono::Utc::now().timestamp(),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {
                    "content": piece,
                },
                "finish_reason": Value::Null,
            }],
        }),
    }
}

fn format_done_event(
    model: &str,
    completion_id: &str,
    format: StreamFormat,
    tokens_generated: u32,
    tokens_prompt: u32,
    duration_ms: u64,
    stop_reason: StopReason,
) -> Value {
    let reason = stop_reason.as_api_str();
    match format {
        StreamFormat::OllamaGenerate => json!({
            "model": model,
            "created_at": chrono::Utc::now().to_rfc3339(),
            "response": "",
            "done": true,
            "done_reason": reason,
            "total_duration": duration_ms * 1_000_000,
            "load_duration": 0,
            "prompt_eval_count": tokens_prompt,
            "prompt_eval_duration": 0,
            "eval_count": tokens_generated,
            "eval_duration": duration_ms * 1_000_000,
        }),
        StreamFormat::OllamaChat => json!({
            "model": model,
            "created_at": chrono::Utc::now().to_rfc3339(),
            "message": {
                "role": "assistant",
                "content": "",
            },
            "done": true,
            "done_reason": reason,
            "total_duration": duration_ms * 1_000_000,
            "load_duration": 0,
            "prompt_eval_count": tokens_prompt,
            "prompt_eval_duration": 0,
            "eval_count": tokens_generated,
            "eval_duration": duration_ms * 1_000_000,
        }),
        StreamFormat::OpenAI => json!({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": chrono::Utc::now().timestamp(),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": reason,
            }],
            "usage": {
                "prompt_tokens": tokens_prompt,
                "completion_tokens": tokens_generated,
                "total_tokens": tokens_prompt + tokens_generated,
            },
        }),
    }
}

/// Check if a loaded model name matches a requested name.
///
/// Handles the common case where the loaded model is a full path
/// (e.g. `/models/qwen3-8b.gguf`) but the request uses a short name
/// (e.g. `qwen3-8b` or `qwen3:8b`).
fn model_names_match(loaded: &str, normalized_request: &str) -> bool {
    // Exact match.
    if loaded == normalized_request {
        return true;
    }
    // Compare file stems: "/models/qwen3-8b.gguf" → "qwen3-8b".
    let loaded_stem = std::path::Path::new(loaded)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(loaded);
    let request_stem = std::path::Path::new(normalized_request)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(normalized_request);
    loaded_stem == request_stem
}

#[cfg(test)]
mod tests {
    use super::*;

    // A reasoning model doing free-text tool-calling can burn through
    // hundreds of tokens of <think> before producing anything else — a
    // small fixed cap here (the old default was 512) truncates mid-block
    // and leaves a text-based tool-calling client (e.g. Cline) holding an
    // unclosed tag. Confirmed on real hardware: see the v0.6.27 changelog.
    #[test]
    fn max_tokens_defaults_to_unbounded_not_a_small_fixed_cap() {
        let sp = parse_generate_params(&json!({}));
        assert_eq!(sp.max_tokens, u32::MAX);
    }

    #[test]
    fn max_tokens_still_honors_an_explicit_client_value() {
        let sp = parse_generate_params(&json!({ "max_tokens": 128 }));
        assert_eq!(sp.max_tokens, 128);

        let sp = parse_generate_params(&json!({ "options": { "num_predict": 256 } }));
        assert_eq!(sp.max_tokens, 256);
    }

    // ── Slot override validation ────────────────────────────────────────
    //
    // These two fields are the only request-body values that reach a
    // model-load decision (scheduler slot count, KV-cache size). Before
    // validation existed, `{"batch_size": 4294967296}` truncated to zero
    // usable slots — the model reported as loaded while no request could
    // ever be served again — and an absurd `ctx_size` failed the context
    // allocation *after* swap_model had already unloaded the previous
    // model, leaving the slot empty.

    fn overrides(body: Value) -> Result<SlotOverrides, StatusCode> {
        parse_slot_overrides(&body).map_err(|(status, _)| status)
    }

    #[test]
    fn absent_overrides_keep_the_launch_time_values() {
        assert_eq!(overrides(json!({})), Ok((None, None)));
        assert_eq!(
            overrides(json!({ "batch_size": null, "ctx_size": null })),
            Ok((None, None))
        );
    }

    #[test]
    fn in_range_overrides_are_accepted() {
        assert_eq!(
            overrides(json!({ "batch_size": 8, "ctx_size": 32768 })),
            Ok((Some(8), Some(32768)))
        );
        // Exact bounds are inclusive.
        assert_eq!(
            overrides(json!({ "batch_size": MAX_BATCH_SIZE_OVERRIDE })),
            Ok((Some(MAX_BATCH_SIZE_OVERRIDE), None))
        );
        assert_eq!(
            overrides(json!({ "ctx_size": MIN_CTX_SIZE_OVERRIDE })),
            Ok((None, Some(MIN_CTX_SIZE_OVERRIDE)))
        );
    }

    #[test]
    fn batch_size_beyond_i32_no_longer_wedges_the_scheduler() {
        // The historical case: 2^32 truncated to 0 slots via `as i32`.
        assert_eq!(
            overrides(json!({ "batch_size": 4_294_967_296u64 })),
            Err(StatusCode::BAD_REQUEST)
        );
    }

    #[test]
    fn out_of_range_overrides_are_rejected() {
        assert_eq!(
            overrides(json!({ "batch_size": 0 })),
            Err(StatusCode::BAD_REQUEST)
        );
        assert_eq!(
            overrides(json!({ "batch_size": MAX_BATCH_SIZE_OVERRIDE + 1 })),
            Err(StatusCode::BAD_REQUEST)
        );
        assert_eq!(
            overrides(json!({ "ctx_size": MIN_CTX_SIZE_OVERRIDE - 1 })),
            Err(StatusCode::BAD_REQUEST)
        );
        assert_eq!(
            overrides(json!({ "ctx_size": u32::MAX })),
            Err(StatusCode::BAD_REQUEST)
        );
    }

    #[test]
    fn a_present_but_unparseable_override_is_an_error_not_an_absence() {
        // Silently treating these as "not supplied" would hide a client bug
        // behind unchanged server behaviour.
        for body in [
            json!({ "batch_size": -1 }),
            json!({ "batch_size": "8" }),
            json!({ "batch_size": 1.5 }),
            json!({ "ctx_size": -4096 }),
            json!({ "ctx_size": "32768" }),
        ] {
            assert_eq!(
                overrides(body.clone()),
                Err(StatusCode::BAD_REQUEST),
                "expected 400 for {body}"
            );
        }
    }
}

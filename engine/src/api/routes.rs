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
use axum::http::{header, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::IntoResponse;
use axum::{routing::get, routing::post, Json, Router};
use futures_util::stream::Stream;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use super::AppState;
use crate::audit::{AuditEntry, AuditLogger};
use crate::inference::{GenerateRequest, InferenceEngine, SchedulerHandle, StreamEvent, JSON_GBNF};
use crate::models::EU_CATALOG;
use crate::tools;

type S = Arc<AppState>;

/// EULLM native API routes (`/api/*`).
pub fn api_routes() -> Router<S> {
    Router::new()
        .route("/tags", get(list_models))
        .route("/generate", post(generate))
        .route("/chat", post(chat))
        .route("/show", post(show_model))
        .route("/pull", post(pull_model))
        .route("/version", get(version))
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
            state.swap_model(name, override_batch_size, override_ctx_size).await.map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": format!("Failed to load model '{name}': {e}") })),
                )
            })?;
        }
    }

    // Take a read-lock snapshot (cheap clones: Arc bump + channel clone).
    let slot = state.slot.read().await;
    if slot.engine.is_none() && slot.scheduler.is_none() {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({ "error": "No model loaded. Send a request with a \"model\" field, or use `eullm run <model>`." })),
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

    let max_tokens = get("max_tokens")
        .or_else(|| get("num_predict"))
        .and_then(|v| v.as_u64())
        .unwrap_or(512) as u32;
    // Defaults match Ollama: temperature=0.8, top_k=40, top_p=0.9, repeat_penalty=1.1
    let temperature = get("temperature").and_then(|v| v.as_f64()).unwrap_or(0.8) as f32;
    let top_k = get("top_k").and_then(|v| v.as_i64()).unwrap_or(40) as i32;
    let top_p = get("top_p").and_then(|v| v.as_f64()).unwrap_or(0.9) as f32;
    let min_p = get("min_p").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
    let repeat_penalty = get("repeat_penalty").and_then(|v| v.as_f64()).unwrap_or(1.1) as f32;
    let repeat_last_n = get("repeat_last_n").and_then(|v| v.as_i64()).unwrap_or(64) as i32;
    let seed = get("seed").and_then(|v| v.as_u64()).map(|v| v as u32);
    let num_ctx = get("num_ctx").and_then(|v| v.as_u64()).map(|v| v as u32);

    tracing::info!(
        "Request params: max_tokens={max_tokens}, temp={temperature:.2}, top_k={top_k}, top_p={top_p:.2}, repeat_penalty={repeat_penalty:.2}, num_ctx={num_ctx:?}"
    );
    SamplingParams { max_tokens, temperature, top_k, top_p, min_p, repeat_penalty, repeat_last_n, seed, num_ctx }
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
async fn collect_stream(
    mut rx: mpsc::Receiver<StreamEvent>,
) -> Result<(String, u32, u32, u64), String> {
    let mut text = String::new();
    let mut tokens_generated = 0u32;
    let mut tokens_prompt = 0u32;
    let mut duration_ms = 0u64;

    while let Some(event) = rx.recv().await {
        match event {
            StreamEvent::Token(piece) => text.push_str(&piece),
            StreamEvent::Done {
                tokens_generated: tg,
                tokens_prompt: tp,
                duration_ms: d,
            } => {
                tokens_generated = tg;
                tokens_prompt = tp;
                duration_ms = d;
                break;
            }
            StreamEvent::Error(e) => return Err(e),
        }
    }

    Ok((text, tokens_generated, tokens_prompt, duration_ms))
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
) -> Vec<Value> {
    if !web_enabled {
        return messages;
    }

    // Find last user message
    let last_user_idx = messages.iter().rposition(|m| {
        m.get("role").and_then(|v| v.as_str()) == Some("user")
    });
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
    let existing_chars: usize = messages.iter()
        .map(|m| m.get("content").and_then(|v| v.as_str()).unwrap_or("").len())
        .sum();

    let mut injections: Vec<String> = Vec::new();
    for url in &urls {
        match tools::fetch_for_context(url, ctx_size, existing_chars, &user_text).await {
            Ok((content, truncated)) => {
                let note = if truncated {
                    format!(" [content truncated to fit context]")
                } else {
                    String::new()
                };
                injections.push(format!(
                    "[Web content from {url}{note}]\n\n{content}"
                ));
                tracing::info!("Web: fetched {} ({} chars{})", url, content.len(), if truncated { ", truncated" } else { "" });
            }
            Err(e) => {
                injections.push(format!("[Failed to fetch {url}: {e}]"));
                tracing::warn!("Web: fetch failed for {url}: {e}");
            }
        }
    }

    if injections.is_empty() {
        return messages;
    }

    // Insert a synthetic "tool" message before the last user turn
    let web_message = json!({
        "role": "system",
        "content": injections.join("\n\n---\n\n")
    });
    messages.insert(idx, web_message);
    messages
}

// ── EULLM API handlers ──────────────────────────────────────────────────────

async fn version() -> Json<Value> {
    Json(json!({ "version": env!("CARGO_PKG_VERSION") }))
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
            "details": {
                "format": "gguf",
                "family": "",
                "parameter_size": "",
                "quantization_level": "Q4_K_M",
            }
        }));
    }

    // Add catalog entries (skip duplicates if the loaded model is in the catalog)
    for m in EU_CATALOG.iter() {
        if loaded_name.as_deref() == Some(m.name.as_str()) {
            // Replace the placeholder entry above with full catalog metadata
            if let Some(first) = models.first_mut() {
                *first = json!({
                    "name": m.name,
                    "size": m.size_bytes,
                    "digest": m.digest,
                    "details": {
                        "format": "gguf",
                        "family": m.base,
                        "parameter_size": format!("{}B", m.vram_gb * 2),
                        "quantization_level": "Q4_K_M",
                        "domain": m.domain,
                        "source_model": m.source_model
                    }
                });
            }
            continue;
        }
        models.push(json!({
            "name": m.name,
            "size": m.size_bytes,
            "digest": m.digest,
            "details": {
                "format": "gguf",
                "family": m.base,
                "parameter_size": format!("{}B", m.vram_gb * 2),
                "quantization_level": "Q4_K_M",
                "domain": m.domain,
                "source_model": m.source_model
            }
        }));
    }

    Json(json!({ "models": models }))
}

async fn generate(
    State(state): State<S>,
    Json(body): Json<Value>,
) -> Result<axum::response::Response, (StatusCode, Json<Value>)> {
    let requested = body.get("model").and_then(|v| v.as_str());
    let override_batch_size = body.get("batch_size").and_then(|v| v.as_u64()).map(|v| v as usize);
    let override_ctx_size = body.get("ctx_size").and_then(|v| v.as_u64()).map(|v| v as u32);
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
            Ok(ndjson_stream_response(rx, model, StreamFormat::OllamaGenerate))
        } else {
            let (text, tokens_generated, tokens_prompt, duration_ms) =
                collect_stream(rx).await.map_err(|e| {
                    (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e })))
                })?;

            let mut audit = AuditEntry::new(model.clone(), "generate".to_string());
            audit.input_tokens = tokens_prompt;
            audit.output_tokens = tokens_generated;
            audit.duration_ms = duration_ms;
            AuditLogger::new().log(&audit);

            Ok(Json(json!({
                "model": model,
                "created_at": chrono::Utc::now().to_rfc3339(),
                "response": text,
                "done": true,
                "done_reason": "stop",
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
            Ok(ndjson_stream_response(rx, model, StreamFormat::OllamaGenerate))
        } else {
            let result = tokio::task::spawn_blocking({
                let engine = engine;
                move || engine.generate(&request)
            })
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": format!("Task error: {e}") }))))?
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": format!("Inference error: {e}") }))))?;

            let mut audit = AuditEntry::new(model.clone(), "generate".to_string());
            audit.input_tokens = result.tokens_prompt;
            audit.output_tokens = result.tokens_generated;
            audit.duration_ms = result.duration_ms;
            AuditLogger::new().log(&audit);

            Ok(Json(json!({
                "model": model,
                "created_at": chrono::Utc::now().to_rfc3339(),
                "response": result.text,
                "done": true,
                "done_reason": "stop",
                "total_duration": result.duration_ms * 1_000_000,
                "load_duration": 0,
                "prompt_eval_count": result.tokens_prompt,
                "prompt_eval_duration": 0,
                "eval_count": result.tokens_generated,
                "eval_duration": result.duration_ms * 1_000_000
            })).into_response())
        }
    }
}

async fn chat(
    State(state): State<S>,
    Json(body): Json<Value>,
) -> Result<axum::response::Response, (StatusCode, Json<Value>)> {
    let requested = body.get("model").and_then(|v| v.as_str());
    let override_batch_size = body.get("batch_size").and_then(|v| v.as_u64()).map(|v| v as usize);
    let override_ctx_size = body.get("ctx_size").and_then(|v| v.as_u64()).map(|v| v as u32);
    let snap = ensure_model(&state, requested, override_batch_size, override_ctx_size).await?;
    let model = snap.model_name.clone();

    let messages = body
        .get("messages")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let messages = inject_web_content(messages, state.web_enabled, state.ctx_size).await;

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
        grammar,
        raw: false,
    };

    if let Some(ref sched) = snap.scheduler {
        let rx = sched.submit(request);

        if is_streaming(&body) {
            Ok(ndjson_stream_response(rx, model, StreamFormat::OllamaChat))
        } else {
            let (text, tokens_generated, tokens_prompt, duration_ms) =
                collect_stream(rx).await.map_err(|e| {
                    (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e })))
                })?;

            let mut audit = AuditEntry::new(model.clone(), "chat".to_string());
            audit.input_tokens = tokens_prompt;
            audit.output_tokens = tokens_generated;
            audit.duration_ms = duration_ms;
            AuditLogger::new().log(&audit);

            Ok(Json(json!({
                "model": model,
                "created_at": chrono::Utc::now().to_rfc3339(),
                "message": {
                    "role": "assistant",
                    "content": text
                },
                "done": true,
                "done_reason": "stop",
                "total_duration": duration_ms * 1_000_000,
                "load_duration": 0,
                "prompt_eval_count": tokens_prompt,
                "prompt_eval_duration": 0,
                "eval_count": tokens_generated,
                "eval_duration": duration_ms * 1_000_000
            })).into_response())
        }
    } else {
        let engine = Arc::clone(snap.engine.as_ref().unwrap());

        if is_streaming(&body) {
            let rx = sequential_to_channel(engine, request);
            Ok(ndjson_stream_response(rx, model, StreamFormat::OllamaChat))
        } else {
            let result = tokio::task::spawn_blocking({
                let engine = engine;
                move || engine.generate(&request)
            })
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": format!("Task error: {e}") }))))?
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": format!("Inference error: {e}") }))))?;

            let mut audit = AuditEntry::new(model.clone(), "chat".to_string());
            audit.input_tokens = result.tokens_prompt;
            audit.output_tokens = result.tokens_generated;
            audit.duration_ms = result.duration_ms;
            AuditLogger::new().log(&audit);

            Ok(Json(json!({
                "model": model,
                "created_at": chrono::Utc::now().to_rfc3339(),
                "message": {
                    "role": "assistant",
                    "content": result.text
                },
                "done": true,
                "done_reason": "stop",
                "total_duration": result.duration_ms * 1_000_000,
                "load_duration": 0,
                "prompt_eval_count": result.tokens_prompt,
                "prompt_eval_duration": 0,
                "eval_count": result.tokens_generated,
                "eval_duration": result.duration_ms * 1_000_000
            })).into_response())
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
                entry.name, entry.domain, entry.base, entry.source_model, entry.license
            ),
            "parameters": "num_ctx 4096\ntemperature 0.7",
            "template": "{{ .Prompt }}",
            "details": {
                "format": "gguf",
                "family": entry.base,
                "parameter_size": format!("{}B", entry.vram_gb * 2),
                "quantization_level": "Q4_K_M",
                "domain": entry.domain,
                "source_model": entry.source_model
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

async fn list_models_openai() -> Json<Value> {
    let data: Vec<Value> = EU_CATALOG
        .iter()
        .map(|m| {
            json!({
                "id": m.name,
                "object": "model",
                "created": 1700000000_u64,
                "owned_by": "eullm"
            })
        })
        .collect();

    Json(json!({ "object": "list", "data": data }))
}

async fn chat_completions(
    State(state): State<S>,
    Json(body): Json<Value>,
) -> Result<axum::response::Response, (StatusCode, Json<Value>)> {
    let requested = body.get("model").and_then(|v| v.as_str());
    let override_batch_size = body.get("batch_size").and_then(|v| v.as_u64()).map(|v| v as usize);
    let override_ctx_size = body.get("ctx_size").and_then(|v| v.as_u64()).map(|v| v as u32);
    let snap = ensure_model(&state, requested, override_batch_size, override_ctx_size).await?;
    let model = snap.model_name.clone();

    let messages = body
        .get("messages")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let messages = inject_web_content(messages, state.web_enabled, state.ctx_size).await;

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
        grammar,
        raw: false,
    };

    if let Some(ref sched) = snap.scheduler {
        let rx = sched.submit(request);

        if is_streaming(&body) {
            let stream = stream_from_channel_sse(rx, model, StreamFormat::OpenAI);
            Ok(Sse::new(stream).into_response())
        } else {
            let (text, tokens_generated, tokens_prompt, duration_ms) =
                collect_stream(rx).await.map_err(|e| {
                    (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e })))
                })?;

            let mut audit = AuditEntry::new(model.clone(), "chat.completions".to_string());
            audit.input_tokens = tokens_prompt;
            audit.output_tokens = tokens_generated;
            audit.duration_ms = duration_ms;
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
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": tokens_prompt,
                    "completion_tokens": tokens_generated,
                    "total_tokens": tokens_prompt + tokens_generated
                }
            })).into_response())
        }
    } else {
        let engine = Arc::clone(snap.engine.as_ref().unwrap());

        if is_streaming(&body) {
            let rx = sequential_to_channel(engine, request);
            let stream = stream_from_channel_sse(rx, model, StreamFormat::OpenAI);
            Ok(Sse::new(stream).into_response())
        } else {
            let result = tokio::task::spawn_blocking({
                let engine = engine;
                move || engine.generate(&request)
            })
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": format!("Task error: {e}") }))))?
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": format!("Inference error: {e}") }))))?;

            let mut audit = AuditEntry::new(model.clone(), "chat.completions".to_string());
            audit.input_tokens = result.tokens_prompt;
            audit.output_tokens = result.tokens_generated;
            audit.duration_ms = result.duration_ms;
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
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": result.tokens_prompt,
                    "completion_tokens": result.tokens_generated,
                    "total_tokens": result.tokens_prompt + result.tokens_generated
                }
            })).into_response())
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
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
    async_stream::stream! {
        let completion_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());

        while let Some(event) = rx.recv().await {
            match event {
                StreamEvent::Token(piece) => {
                    let data = format_token_event(&piece, &model, &completion_id, format);
                    yield Ok(Event::default().data(data.to_string()));
                }
                StreamEvent::Done { tokens_generated, tokens_prompt, duration_ms } => {
                    // Audit log
                    let mut audit = AuditEntry::new(model.clone(), match format {
                        StreamFormat::OllamaGenerate => "generate",
                        StreamFormat::OllamaChat => "chat",
                        StreamFormat::OpenAI => "chat.completions",
                    }.to_string());
                    audit.input_tokens = tokens_prompt;
                    audit.output_tokens = tokens_generated;
                    audit.duration_ms = duration_ms;
                    AuditLogger::new().log(&audit);

                    let data = format_done_event(
                        &model, &completion_id, format,
                        tokens_generated, tokens_prompt, duration_ms,
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
                StreamEvent::Done { tokens_generated, tokens_prompt, duration_ms } => {
                    let mut audit = AuditEntry::new(model.clone(), match format {
                        StreamFormat::OllamaGenerate => "generate",
                        StreamFormat::OllamaChat => "chat",
                        StreamFormat::OpenAI => "chat.completions",
                    }.to_string());
                    audit.input_tokens = tokens_prompt;
                    audit.output_tokens = tokens_generated;
                    audit.duration_ms = duration_ms;
                    AuditLogger::new().log(&audit);

                    let data = format_done_event(
                        &model, &completion_id, format,
                        tokens_generated, tokens_prompt, duration_ms,
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

fn format_token_event(piece: &str, model: &str, completion_id: &str, format: StreamFormat) -> Value {
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
) -> Value {
    match format {
        StreamFormat::OllamaGenerate => json!({
            "model": model,
            "created_at": chrono::Utc::now().to_rfc3339(),
            "response": "",
            "done": true,
            "done_reason": "stop",
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
            "done_reason": "stop",
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
                "finish_reason": "stop",
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

//! Route definitions for the EULLM and OpenAI-compatible APIs.
//!
//! Supports both non-streaming (JSON) and streaming (SSE) responses
//! on all generation endpoints. Set `"stream": true` in the request body
//! to get token-by-token Server-Sent Events.
//!
//! When a `SchedulerHandle` is present in `AppState`, requests are dispatched
//! through the continuous batching scheduler. Otherwise the sequential
//! `InferenceEngine` is used as fallback.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::IntoResponse;
use axum::{routing::get, routing::post, Json, Router};
use futures_util::stream::Stream;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use super::AppState;
use crate::audit::{AuditEntry, AuditLogger};
use crate::inference::{GenerateRequest, StreamEvent};
use crate::models::EU_CATALOG;

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

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Returns true if the scheduler is available.
fn has_scheduler(state: &AppState) -> bool {
    state.scheduler.is_some()
}

fn has_engine(state: &AppState) -> bool {
    state.engine.is_some() || state.scheduler.is_some()
}

fn require_engine(state: &AppState) -> Result<(), (StatusCode, Json<Value>)> {
    if !has_engine(state) {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({ "error": "No model loaded. Use `eullm run <model>` to load a model." })),
        ));
    }
    Ok(())
}

fn parse_generate_params(body: &Value) -> (u32, f32) {
    let max_tokens = body
        .get("max_tokens")
        .or_else(|| body.get("num_predict"))
        .and_then(|v| v.as_u64())
        .unwrap_or(512) as u32;
    let temperature = body
        .get("temperature")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.7) as f32;
    (max_tokens, temperature)
}

fn is_streaming(body: &Value) -> bool {
    body.get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

/// Format chat messages into a ChatML-style prompt string.
fn format_chat_prompt(messages: &[Value]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");
        let content = msg
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        prompt.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

/// Submit a request through the scheduler and return the receiver.
fn scheduler_submit(state: &AppState, request: GenerateRequest) -> mpsc::Receiver<StreamEvent> {
    state.scheduler.as_ref().unwrap().submit(request)
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

// ── EULLM API handlers ──────────────────────────────────────────────────────

async fn version() -> Json<Value> {
    Json(json!({ "version": env!("CARGO_PKG_VERSION") }))
}

async fn list_models(State(_state): State<S>) -> Json<Value> {
    let models: Vec<Value> = EU_CATALOG
        .iter()
        .map(|m| {
            json!({
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
            })
        })
        .collect();

    Json(json!({ "models": models }))
}

async fn generate(
    State(state): State<S>,
    Json(body): Json<Value>,
) -> Result<axum::response::Response, (StatusCode, Json<Value>)> {
    require_engine(&state)?;

    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .or(state.model_name.as_deref())
        .unwrap_or("unknown")
        .to_string();
    let prompt = body
        .get("prompt")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let (max_tokens, temperature) = parse_generate_params(&body);

    let request = GenerateRequest {
        prompt,
        max_tokens,
        temperature,
        ..Default::default()
    };

    if has_scheduler(&state) {
        // ── Continuous batching path ────────────────────────────────
        let rx = scheduler_submit(&state, request);

        if is_streaming(&body) {
            let stream = stream_from_channel(rx, model, StreamFormat::OllamaGenerate);
            Ok(Sse::new(stream).into_response())
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
                "total_duration": duration_ms * 1_000_000,
                "eval_count": tokens_generated,
                "eval_duration": duration_ms * 1_000_000,
                "prompt_eval_count": tokens_prompt
            }))
            .into_response())
        }
    } else {
        // ── Sequential fallback ────────────────────────────────────
        let engine = Arc::clone(state.engine.as_ref().unwrap());

        if is_streaming(&body) {
            let stream = stream_generate_sequential(engine, request, model);
            Ok(Sse::new(stream).into_response())
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
                "total_duration": result.duration_ms * 1_000_000,
                "eval_count": result.tokens_generated,
                "eval_duration": result.duration_ms * 1_000_000,
                "prompt_eval_count": result.tokens_prompt
            })).into_response())
        }
    }
}

async fn chat(
    State(state): State<S>,
    Json(body): Json<Value>,
) -> Result<axum::response::Response, (StatusCode, Json<Value>)> {
    require_engine(&state)?;

    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .or(state.model_name.as_deref())
        .unwrap_or("unknown")
        .to_string();

    let messages = body
        .get("messages")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let prompt = format_chat_prompt(&messages);
    let (max_tokens, temperature) = parse_generate_params(&body);

    let request = GenerateRequest {
        prompt,
        max_tokens,
        temperature,
        stop_sequences: vec![
            "<|im_end|>".to_string(),
            "<|end|>".to_string(),
        ],
    };

    if has_scheduler(&state) {
        let rx = scheduler_submit(&state, request);

        if is_streaming(&body) {
            let stream = stream_from_channel(rx, model, StreamFormat::OllamaChat);
            Ok(Sse::new(stream).into_response())
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
                "total_duration": duration_ms * 1_000_000,
                "eval_count": tokens_generated,
                "eval_duration": duration_ms * 1_000_000,
                "prompt_eval_count": tokens_prompt
            })).into_response())
        }
    } else {
        let engine = Arc::clone(state.engine.as_ref().unwrap());

        if is_streaming(&body) {
            let stream = stream_generate_sequential(engine, request, model.clone());
            let stream = stream_remap(stream, model, StreamFormat::OllamaChat);
            Ok(Sse::new(stream).into_response())
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
                "total_duration": result.duration_ms * 1_000_000,
                "eval_count": result.tokens_generated,
                "eval_duration": result.duration_ms * 1_000_000,
                "prompt_eval_count": result.tokens_prompt
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
    require_engine(&state)?;

    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .or(state.model_name.as_deref())
        .unwrap_or("eullm/general-eu-7b")
        .to_string();

    let messages = body
        .get("messages")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let prompt = format_chat_prompt(&messages);
    let (max_tokens, temperature) = parse_generate_params(&body);

    let request = GenerateRequest {
        prompt,
        max_tokens,
        temperature,
        stop_sequences: vec![
            "<|im_end|>".to_string(),
            "<|end|>".to_string(),
        ],
    };

    if has_scheduler(&state) {
        let rx = scheduler_submit(&state, request);

        if is_streaming(&body) {
            let stream = stream_from_channel(rx, model, StreamFormat::OpenAI);
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
        let engine = Arc::clone(state.engine.as_ref().unwrap());

        if is_streaming(&body) {
            let stream = stream_generate_sequential(engine, request, model.clone());
            let stream = stream_remap(stream, model, StreamFormat::OpenAI);
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
/// Used for both scheduler and sequential backends.
fn stream_from_channel(
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

/// Sequential streaming: spawn_blocking + mpsc (legacy path).
fn stream_generate_sequential(
    engine: Arc<crate::inference::InferenceEngine>,
    request: GenerateRequest,
    model: String,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
    let (tx, rx) = mpsc::channel::<StreamEvent>(32);

    tokio::task::spawn_blocking(move || {
        engine.generate_streaming(&request, tx);
    });

    stream_from_channel(rx, model, StreamFormat::OllamaGenerate)
}

/// Re-map a stream from OllamaGenerate format to another format.
/// Used by sequential chat/completions paths that always produce OllamaGenerate events.
fn stream_remap(
    inner: impl Stream<Item = Result<Event, std::convert::Infallible>>,
    _model: String,
    _format: StreamFormat,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
    // The inner stream already uses the correct format via stream_from_channel.
    inner
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
            "total_duration": duration_ms * 1_000_000,
            "eval_count": tokens_generated,
            "eval_duration": duration_ms * 1_000_000,
            "prompt_eval_count": tokens_prompt,
        }),
        StreamFormat::OllamaChat => json!({
            "model": model,
            "created_at": chrono::Utc::now().to_rfc3339(),
            "message": {
                "role": "assistant",
                "content": "",
            },
            "done": true,
            "total_duration": duration_ms * 1_000_000,
            "eval_count": tokens_generated,
            "eval_duration": duration_ms * 1_000_000,
            "prompt_eval_count": tokens_prompt,
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

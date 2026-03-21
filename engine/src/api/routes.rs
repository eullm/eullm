//! Route definitions for the EULLM and OpenAI-compatible APIs.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::{routing::get, routing::post, Json, Router};
use serde_json::{json, Value};

use super::AppState;
use crate::audit::{AuditEntry, AuditLogger};
use crate::inference::GenerateRequest;
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

// -- EULLM API handlers --

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
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .or(state.model_name.as_deref())
        .unwrap_or("unknown");
    let prompt = body
        .get("prompt")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let engine = state.engine.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({ "error": "No model loaded. Use `eullm run <model>` to load a model." })),
        )
    })?;

    let max_tokens = body
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(512) as u32;

    let temperature = body
        .get("temperature")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.7) as f32;

    let request = GenerateRequest {
        prompt: prompt.to_string(),
        max_tokens,
        temperature,
        ..Default::default()
    };

    let result = tokio::task::spawn_blocking({
        let engine = Arc::clone(engine);
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

    // Audit log
    let mut audit = AuditEntry::new(model.to_string(), "generate".to_string());
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
    })))
}

async fn chat(
    State(state): State<S>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .or(state.model_name.as_deref())
        .unwrap_or("unknown");

    let messages = body
        .get("messages")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let engine = state.engine.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({ "error": "No model loaded. Use `eullm run <model>` to load a model." })),
        )
    })?;

    // Build prompt from messages (simple ChatML-style)
    let prompt = format_chat_prompt(&messages);

    let max_tokens = body
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(512) as u32;

    let temperature = body
        .get("temperature")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.7) as f32;

    let request = GenerateRequest {
        prompt,
        max_tokens,
        temperature,
        stop_sequences: vec![
            "<|im_end|>".to_string(),
            "<|end|>".to_string(),
        ],
    };

    let result = tokio::task::spawn_blocking({
        let engine = Arc::clone(engine);
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

    // Audit log
    let mut audit = AuditEntry::new(model.to_string(), "chat".to_string());
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
    })))
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

// -- OpenAI-compatible handlers --

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
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .or(state.model_name.as_deref())
        .unwrap_or("eullm/general-eu-7b");

    let messages = body
        .get("messages")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let engine = state.engine.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({ "error": "No model loaded. Use `eullm run <model>` to load a model." })),
        )
    })?;

    let prompt = format_chat_prompt(&messages);

    let max_tokens = body
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(512) as u32;

    let temperature = body
        .get("temperature")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.7) as f32;

    let request = GenerateRequest {
        prompt,
        max_tokens,
        temperature,
        stop_sequences: vec![
            "<|im_end|>".to_string(),
            "<|end|>".to_string(),
        ],
    };

    let result = tokio::task::spawn_blocking({
        let engine = Arc::clone(engine);
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

    // Audit log
    let mut audit = AuditEntry::new(model.to_string(), "chat.completions".to_string());
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
    })))
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

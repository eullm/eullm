//! Route definitions for the EULLM and OpenAI-compatible APIs.

use std::sync::Arc;

use axum::extract::State;
use axum::{routing::get, routing::post, Json, Router};
use serde_json::{json, Value};

use super::AppState;
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
                    "quantization_level": "Q4_K_M"
                }
            })
        })
        .collect();

    Json(json!({ "models": models }))
}

async fn generate(State(state): State<S>, Json(body): Json<Value>) -> Json<Value> {
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .or(state.model.as_deref())
        .unwrap_or("unknown");
    let prompt = body
        .get("prompt")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    // Mock response
    let response = format!(
        "Hello! I'm {model}, running on eullm. \
         This is a mock response. You asked: \"{prompt}\""
    );

    Json(json!({
        "model": model,
        "created_at": chrono::Utc::now().to_rfc3339(),
        "response": response,
        "done": true,
        "total_duration": 150_000_000,
        "eval_count": 25,
        "eval_duration": 100_000_000
    }))
}

async fn chat(State(state): State<S>, Json(body): Json<Value>) -> Json<Value> {
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .or(state.model.as_deref())
        .unwrap_or("unknown");

    let last_message = body
        .get("messages")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.last())
        .and_then(|m| m.get("content"))
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let response = format!(
        "Hello! I'm {model}, running on eullm. \
         This is a mock response to: \"{last_message}\""
    );

    Json(json!({
        "model": model,
        "created_at": chrono::Utc::now().to_rfc3339(),
        "message": {
            "role": "assistant",
            "content": response
        },
        "done": true,
        "total_duration": 150_000_000,
        "eval_count": 25,
        "eval_duration": 100_000_000
    }))
}

async fn show_model(Json(body): Json<Value>) -> Json<Value> {
    let name = body
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    if let Some(entry) = crate::models::catalog::find_model(name) {
        Json(json!({
            "modelfile": format!("# EULLM model: {}\n# Base: {}\n# License: {}", entry.name, entry.base, entry.license),
            "parameters": format!("num_ctx 4096\ntemperature 0.7"),
            "template": "{{ .Prompt }}",
            "details": {
                "format": "gguf",
                "family": entry.base,
                "parameter_size": format!("{}B", entry.vram_gb * 2),
                "quantization_level": "Q4_K_M"
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
        "status": format!("pulling {name} from EU registry (mock)")
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

async fn chat_completions(State(state): State<S>, Json(body): Json<Value>) -> Json<Value> {
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .or(state.model.as_deref())
        .unwrap_or("eullm/general-eu-14b");

    let last_message = body
        .get("messages")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.last())
        .and_then(|m| m.get("content"))
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let response = format!(
        "Hello! I'm {model}, running on eullm. \
         This is a mock response to: \"{last_message}\""
    );

    Json(json!({
        "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        "object": "chat.completion",
        "created": chrono::Utc::now().timestamp(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 25,
            "total_tokens": 35
        }
    }))
}

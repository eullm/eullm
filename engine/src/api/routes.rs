//! Route definitions for the Ollama-compatible and OpenAI-compatible APIs.

use axum::{routing::get, routing::post, Json, Router};
use serde_json::{json, Value};

/// Ollama-compatible API routes (`/api/*`).
pub fn api_routes() -> Router {
    Router::new()
        .route("/tags", get(list_models))
        .route("/generate", post(generate))
        .route("/chat", post(chat))
        .route("/show", post(show_model))
        .route("/pull", post(pull_model))
}

/// OpenAI-compatible routes (`/v1/*`).
pub fn openai_routes() -> Router {
    Router::new()
        .route("/models", get(list_models_openai))
        .route("/chat/completions", post(chat_completions))
}

// -- Ollama API handlers (stubs) --

async fn list_models() -> Json<Value> {
    Json(json!({ "models": [] }))
}

async fn generate(Json(_body): Json<Value>) -> Json<Value> {
    Json(json!({ "error": "not implemented yet" }))
}

async fn chat(Json(_body): Json<Value>) -> Json<Value> {
    Json(json!({ "error": "not implemented yet" }))
}

async fn show_model(Json(_body): Json<Value>) -> Json<Value> {
    Json(json!({ "error": "not implemented yet" }))
}

async fn pull_model(Json(_body): Json<Value>) -> Json<Value> {
    Json(json!({ "error": "not implemented yet" }))
}

// -- OpenAI-compatible handlers (stubs) --

async fn list_models_openai() -> Json<Value> {
    Json(json!({ "object": "list", "data": [] }))
}

async fn chat_completions(Json(_body): Json<Value>) -> Json<Value> {
    Json(json!({ "error": "not implemented yet" }))
}

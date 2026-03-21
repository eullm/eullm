//! EULLM Hub — EU-hosted model registry API.
//!
//! Serves model metadata and download URLs from S3-compatible
//! storage on European infrastructure.

use axum::{routing::get, Json, Router};
use serde_json::{json, Value};
use tokio::net::TcpListener;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "eullm_hub=info".into()),
        )
        .init();

    let app = Router::new()
        .route("/v1/models", get(list_models))
        .route("/health", get(health));

    let addr = "0.0.0.0:8080";
    tracing::info!("EULLM Hub listening on {addr}");

    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn list_models() -> Json<Value> {
    Json(json!({
        "models": [
            {
                "name": "eullm/general-eu-14b",
                "description": "General purpose multilingual model",
                "languages": ["en", "it", "de", "fr", "es", "pt", "nl"],
                "base": "qwen3",
                "vram_gb": 10,
                "license": "Apache-2.0",
                "status": "coming_soon"
            }
        ]
    }))
}

async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

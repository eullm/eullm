//! eullm REST API.
//!
//! Exposes a standard LLM API (both `/api` and `/v1` OpenAI-compatible)
//! so that existing tools (Open WebUI, LangChain, n8n) work out of the box.
//!
//! Supports two inference backends:
//! - **Sequential** (`InferenceEngine`): one request at a time.
//! - **Continuous batching** (`SchedulerHandle`): multiple concurrent requests.

mod routes;

use axum::Router;
use std::sync::Arc;
use tokio::net::TcpListener;
use tower_http::cors::{Any, CorsLayer};

use crate::inference::{InferenceEngine, SchedulerHandle};

/// Shared state for API handlers.
#[derive(Clone)]
pub struct AppState {
    /// Currently loaded model name (if any).
    pub model_name: Option<String>,
    /// Sequential inference engine (fallback, one request at a time).
    pub engine: Option<Arc<InferenceEngine>>,
    /// Continuous batching scheduler (preferred when available).
    pub scheduler: Option<SchedulerHandle>,
}

/// Start the API server on the given port.
pub async fn serve(
    port: u16,
    model_name: Option<String>,
    engine: Option<Arc<InferenceEngine>>,
    scheduler: Option<SchedulerHandle>,
) -> Result<(), Box<dyn std::error::Error>> {
    let state = Arc::new(AppState {
        model_name,
        engine,
        scheduler,
    });
    let app = router(state);
    let addr = format!("0.0.0.0:{port}");
    tracing::info!("eullm listening on {addr}");

    let listener = TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// Build the EULLM API router with CORS enabled for Open WebUI and other frontends.
fn router(state: Arc<AppState>) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .nest("/api", routes::api_routes())
        .nest("/v1", routes::openai_routes())
        .layer(cors)
        .with_state(state)
}

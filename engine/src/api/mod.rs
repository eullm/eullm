//! eullm REST API.
//!
//! Exposes a standard LLM API (both `/api` and `/v1` OpenAI-compatible)
//! so that existing tools (Open WebUI, LangChain, n8n) work out of the box.
//!
//! Supports two inference backends:
//! - **Sequential** (`InferenceEngine`): one request at a time.
//! - **Continuous batching** (`SchedulerHandle`): multiple concurrent requests.
//!
//! Supports **dynamic model swapping**: when a request specifies a different
//! model name, the server automatically unloads the current model and loads
//! the new one.  In-flight requests on the old model complete normally.

mod routes;

use axum::Router;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::TcpListener;
use tower_http::cors::{Any, CorsLayer};

use crate::inference::{
    BatchScheduler, InferenceConfig, InferenceEngine, SchedulerConfig, SchedulerHandle,
};
use crate::models::ModelStore;

/// The currently loaded model — swapped atomically when a different model
/// is requested via the API.
pub struct ModelSlot {
    /// Currently loaded model name (if any).
    pub model_name: Option<String>,
    /// Sequential inference engine (fallback, one request at a time).
    pub engine: Option<Arc<InferenceEngine>>,
    /// Continuous batching scheduler (preferred when available).
    pub scheduler: Option<SchedulerHandle>,
}

/// Shared state for API handlers.
pub struct AppState {
    /// Mutable model slot — protected by RwLock for concurrent reads,
    /// exclusive writes during model swap.
    pub slot: tokio::sync::RwLock<ModelSlot>,

    // ── Immutable inference settings (from CLI flags) ────────────────
    pub gpu_layers: i32,
    pub ctx_size: u32,
    pub threads: u32,
    pub flash_attn: bool,
    pub n_batch: u32,
    /// 0 = sequential, >0 = continuous batching with this many slots.
    pub batch_size: usize,

    /// Model store for resolving names → GGUF paths.
    pub store: ModelStore,
}

impl AppState {
    /// Swap the currently loaded model.  Drops the old engine/scheduler
    /// (in-flight requests on cloned handles still complete) and loads
    /// the new model with the same inference settings.
    ///
    /// This is the **write** path — only one swap can run at a time.
    pub async fn swap_model(&self, name: &str) -> Result<(), String> {
        // Resolve model name → GGUF path.
        let gguf_path = self.resolve_model(name)?;

        tracing::info!("Swapping model → {name} ({})", gguf_path.display());

        let config = InferenceConfig {
            model_path: gguf_path,
            gpu_layers: self.gpu_layers,
            context_size: self.ctx_size,
            threads: self.threads,
            flash_attn: self.flash_attn,
            n_batch: self.n_batch,
        };

        // Load on a blocking thread (model loading is CPU-heavy).
        let batch_size = self.batch_size;
        let model_name = name.to_string();

        let (new_engine, new_scheduler) = tokio::task::spawn_blocking(move || {
            if batch_size > 0 {
                let sched_config = SchedulerConfig {
                    max_batch_size: batch_size,
                    queue_capacity: batch_size * 8,
                };
                let sched = BatchScheduler::new(config, sched_config);
                match sched.start() {
                    Ok(handle) => Ok((None, Some(handle))),
                    Err(e) => Err(format!("Failed to start scheduler: {e}")),
                }
            } else {
                match InferenceEngine::load(config) {
                    Ok(eng) => Ok((Some(Arc::new(eng)), None)),
                    Err(e) => Err(format!("Failed to load model: {e}")),
                }
            }
        })
        .await
        .map_err(|e| format!("Task join error: {e}"))??;

        // Take the write lock and swap.  The old engine/scheduler are
        // dropped here — the scheduler thread exits once all cloned
        // handles (from in-flight requests) are also dropped.
        {
            let mut slot = self.slot.write().await;
            slot.model_name = Some(model_name.clone());
            slot.engine = new_engine;
            slot.scheduler = new_scheduler;
        }

        tracing::info!("Model swap complete → {model_name}");
        Ok(())
    }

    /// Resolve a model name to a GGUF file path.
    fn resolve_model(&self, name: &str) -> Result<PathBuf, String> {
        let path = PathBuf::from(name);

        // Direct GGUF file path?
        if path.exists() && path.extension().is_some_and(|e| e == "gguf") {
            return Ok(path);
        }

        // Check model store (imported / pulled models).
        if let Some(p) = self.store.gguf_path(name) {
            return Ok(p);
        }

        Err(format!(
            "Model '{name}' not found.  Import it first: eullm import-ollama {name}"
        ))
    }
}

/// Configuration for starting the API server.
pub struct ServeConfig {
    pub port: u16,
    pub model_name: Option<String>,
    pub engine: Option<Arc<InferenceEngine>>,
    pub scheduler: Option<SchedulerHandle>,
    pub gpu_layers: i32,
    pub ctx_size: u32,
    pub threads: u32,
    pub flash_attn: bool,
    pub n_batch: u32,
    pub batch_size: usize,
    pub store: ModelStore,
}

/// Start the API server on the given port with graceful shutdown support.
///
/// The server shuts down cleanly on SIGTERM or SIGINT (Ctrl+C), finishing
/// in-flight requests before exiting. This is critical for Docker containers
/// (which send SIGTERM on `docker stop`) and systemd services.
pub async fn serve(cfg: ServeConfig) -> Result<(), Box<dyn std::error::Error>> {
    let state = Arc::new(AppState {
        slot: tokio::sync::RwLock::new(ModelSlot {
            model_name: cfg.model_name,
            engine: cfg.engine,
            scheduler: cfg.scheduler,
        }),
        gpu_layers: cfg.gpu_layers,
        ctx_size: cfg.ctx_size,
        threads: cfg.threads,
        flash_attn: cfg.flash_attn,
        n_batch: cfg.n_batch,
        batch_size: cfg.batch_size,
        store: cfg.store,
    });
    let port = cfg.port;
    let app = router(state);
    let addr = format!("0.0.0.0:{port}");
    tracing::info!("eullm listening on {addr}");

    let listener = TcpListener::bind(&addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    tracing::info!("Server shut down gracefully.");
    Ok(())
}

/// Wait for a shutdown signal (SIGTERM, SIGINT, or Ctrl+C).
async fn shutdown_signal() {
    let ctrl_c = tokio::signal::ctrl_c();

    #[cfg(unix)]
    {
        let mut sigterm =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("failed to register SIGTERM handler");
        tokio::select! {
            _ = ctrl_c => { tracing::info!("Received SIGINT, shutting down..."); }
            _ = sigterm.recv() => { tracing::info!("Received SIGTERM, shutting down..."); }
        }
    }

    #[cfg(not(unix))]
    {
        ctrl_c.await.ok();
        tracing::info!("Received Ctrl+C, shutting down...");
    }
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

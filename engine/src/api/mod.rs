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

use axum::extract::DefaultBodyLimit;
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
    /// Serializes model swaps — prevents multiple concurrent requests
    /// from triggering parallel swaps (which would OOM the GPU).
    swap_lock: tokio::sync::Mutex<()>,

    // ── Immutable inference settings (from CLI flags) ────────────────
    pub gpu_layers: i32,
    pub ctx_size: u32,
    pub threads: u32,
    pub flash_attn: bool,
    pub n_batch: u32,
    /// KV cache quantization type for keys (e.g. Q8_0 — reduces VRAM).
    pub cache_type_k: crate::inference::KvCacheType,
    /// KV cache quantization type for values (e.g. Q4_0 — reduces VRAM).
    pub cache_type_v: crate::inference::KvCacheType,
    /// 0 = sequential, >0 = continuous batching with this many slots.
    pub batch_size: usize,
    /// Keep MoE expert tensors on CPU RAM (see `InferenceConfig::cpu_moe`).
    /// Applied to every model this server loads or swaps to.
    pub cpu_moe: bool,
    /// Keep MoE expert tensors on CPU RAM for only the first N layers (see
    /// `InferenceConfig::n_cpu_moe`). Applied to every model this server
    /// loads or swaps to.
    pub n_cpu_moe: u32,
    /// Recurrent-state rollback window for hybrid/recurrent architectures
    /// (see `InferenceConfig::rs_seq`). Applied to every model this server
    /// loads or swaps to.
    pub rs_seq: u32,
    /// Max full-sequence-state checkpoints kept for prompt-prefix restore
    /// (see `SchedulerConfig::ctx_checkpoints`). 0 disables checkpointing.
    /// Applied to every model this server loads or swaps to.
    pub ctx_checkpoints: usize,
    /// Min new tokens since the closest checkpoint before taking another
    /// one (see `SchedulerConfig::checkpoint_min_step`).
    pub checkpoint_min_step: u32,

    /// Enable transparent web fetching: URLs in user messages are fetched
    /// and their content is injected into the prompt before inference.
    pub web_enabled: bool,

    /// Port the canonical API listener runs on (Ollama-compatible, default
    /// 11434). Exposed via `/api/version` so the chat UI (served on its own
    /// port) can display the endpoint external clients should point at.
    pub api_port: u16,

    /// Model store for resolving names → GGUF paths.
    pub store: ModelStore,
}

impl AppState {
    /// Swap the currently loaded model.  Drops the old engine/scheduler
    /// (in-flight requests on cloned handles still complete) and loads
    /// the new model with the same inference settings.
    ///
    /// `override_batch_size` allows the caller to change the number of
    /// concurrent batch slots for the new model (e.g. more slots for a
    /// smaller model that uses less VRAM).  Pass `None` to keep the
    /// batch size from the CLI launch.
    ///
    /// This is the **write** path — only one swap can run at a time.
    pub async fn swap_model(&self, name: &str, override_batch_size: Option<usize>, override_ctx_size: Option<u32>) -> Result<(), String> {
        // Serialize swaps — if another request is already swapping,
        // wait for it to finish instead of starting a parallel swap.
        let _swap_guard = self.swap_lock.lock().await;

        // Normalize Ollama-style names: "qwen3:14b" → "qwen3-14b"
        let normalized = normalize_model_name(name);

        // Re-check after acquiring the lock — another thread may have
        // already completed the swap while we were waiting.
        {
            let slot = self.slot.read().await;
            if let Some(ref loaded) = slot.model_name {
                let loaded_stem = std::path::Path::new(loaded.as_str())
                    .file_stem().and_then(|s| s.to_str()).unwrap_or(loaded);
                let req_stem = std::path::Path::new(normalized.as_str())
                    .file_stem().and_then(|s| s.to_str()).unwrap_or(&normalized);
                if loaded_stem == req_stem {
                    tracing::info!("Model {} already loaded (swapped by another request)", crate::audit::sanitize_for_log(&normalized));
                    return Ok(());
                }
            }
        }

        let gguf_path = self.resolve_model(&normalized)?;
        // Resolve an mmproj sibling (vision projector) if the model store
        // declares one. Presence of a projector is the signal that this is
        // a multimodal model — we then force sequential loading (next step)
        // because the continuous-batching scheduler is text-only.
        let mmproj_path = self.store.mmproj_path(&normalized);
        if let Some(ref p) = mmproj_path {
            tracing::info!("Multimodal model detected — mmproj: {}", p.display());
        }
        tracing::info!("Swapping model → {} ({})", crate::audit::sanitize_for_log(&normalized), gguf_path.display());

        // ── 1. Unload the current model and WAIT for the scheduler
        //       thread to fully exit before loading the new model.
        //
        // Without this, both old and new LlamaBackend instances would
        // coexist, and both models would be in VRAM simultaneously —
        // causing OOM or a C-level crash in llama.cpp.
        self.unload_current().await?;
        tracing::info!("Previous model fully unloaded");

        // ── 2. Load the new model ───────────────────────────────────
        let config = InferenceConfig {
            model_path: gguf_path,
            gpu_layers: self.gpu_layers,
            context_size: override_ctx_size.unwrap_or(self.ctx_size),
            threads: self.threads,
            flash_attn: self.flash_attn,
            n_batch: self.n_batch,
            cache_type_k: self.cache_type_k,
            cache_type_v: self.cache_type_v,
            // Multimodal: when the model store declares an mmproj sibling we
            // load it here so HTTP requests with `images` can route through
            // `engine.generate_multimodal()`. Models without an mmproj keep
            // the text-only fast path (None → no extra VRAM, no init cost).
            mmproj_path: mmproj_path.clone(),
            cpu_moe: self.cpu_moe,
            n_cpu_moe: self.n_cpu_moe,
            rs_seq: self.rs_seq,
        };

        // The continuous-batching scheduler is text-only — it does not route
        // mtmd chunks. For multimodal models we therefore force the sequential
        // `InferenceEngine` (batch_size=0). Vision is interactive single-user
        // anyway, so losing batching here is not a practical regression.
        let batch_size = if mmproj_path.is_some() {
            0
        } else {
            override_batch_size.unwrap_or(self.batch_size)
        };
        let model_name = normalized.clone();
        let ctx_checkpoints_for_swap = self.ctx_checkpoints;
        let checkpoint_min_step_for_swap = self.checkpoint_min_step;

        let (new_engine, new_scheduler) = tokio::task::spawn_blocking(move || {
            if batch_size > 0 {
                let sched_config = SchedulerConfig {
                    max_batch_size: batch_size,
                    queue_capacity: batch_size * 8,
                    ctx_checkpoints: ctx_checkpoints_for_swap,
                    checkpoint_min_step: checkpoint_min_step_for_swap,
                };
                let sched = BatchScheduler::new(config, sched_config);
                match sched.start() {
                    Ok((handle, _model_info)) => Ok((None, Some(handle))),
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

        // ── 3. Install the new model in the slot ─────────────────────
        {
            let mut slot = self.slot.write().await;
            slot.model_name = Some(model_name.clone());
            slot.engine = new_engine;
            slot.scheduler = new_scheduler;
        }

        tracing::info!("Model swap complete → {} (batch_size={batch_size})", crate::audit::sanitize_for_log(&model_name));
        Ok(())
    }

    /// Unload the currently loaded model, freeing its VRAM, and leave the
    /// slot empty. Unlike `swap_model`, this does not load a replacement —
    /// a later request with a `model` field (or another `eullm run`) loads
    /// a model again.
    ///
    /// The primary use case is freeing VRAM for a co-resident process (e.g.
    /// an embedding model used during RAG document ingestion) without
    /// restarting the eullm server. Serialized against `swap_model` via the
    /// same lock, so an unload can't race a concurrent swap.
    ///
    /// Returns the name of the model that was unloaded, or `None` if the
    /// slot was already empty (a no-op, not an error).
    pub async fn unload(&self) -> Result<Option<String>, String> {
        let _swap_guard = self.swap_lock.lock().await;

        let previous = {
            let slot = self.slot.read().await;
            slot.model_name.clone()
        };
        if previous.is_none() {
            return Ok(None);
        }

        self.unload_current().await?;
        tracing::info!("Model unloaded — slot empty");
        Ok(previous)
    }

    /// Shared unload step used by both `swap_model` and `unload`: take the
    /// scheduler/engine out of the slot and wait for the scheduler's
    /// dedicated OS thread to fully exit before returning, so the old
    /// model's VRAM is guaranteed freed by the time this resolves — critical
    /// both for swap (avoids two models coexisting in VRAM → OOM) and for a
    /// standalone unload (the caller needs the VRAM actually free before
    /// handing it to another process).
    async fn unload_current(&self) -> Result<(), String> {
        let old_scheduler = {
            let mut slot = self.slot.write().await;
            let sched = slot.scheduler.take();
            slot.engine = None;
            slot.model_name = None;
            sched
        };
        if let Some(handle) = old_scheduler {
            tokio::task::spawn_blocking(move || handle.shutdown())
                .await
                .map_err(|e| format!("Failed to join scheduler thread: {e}"))?;
        }
        Ok(())
    }

    /// Resolve a model name to a GGUF file path.
    ///
    /// Search order:
    /// 1. Direct GGUF file path (absolute or relative, e.g. `/models/qwen3-14b.gguf`)
    /// 2. Directory containing a single .gguf file (e.g. `/models/qwen3-14b/`)
    /// 3. Path without extension — try appending `.gguf`
    /// 4. Exact name in model store (`~/.eullm/models/{name}/*.gguf`)
    /// 5. Normalized name (Ollama tags: `qwen3:14b` → `qwen3-14b`)
    fn resolve_model(&self, name: &str) -> Result<PathBuf, String> {
        let path = PathBuf::from(name);

        // 1. Direct GGUF file path?
        if path.is_file() {
            return Ok(path);
        }

        // 2. Directory containing .gguf files? Pick the first one.
        if path.is_dir() && let Some(gguf) = find_gguf_in_dir(&path) {
            return Ok(gguf);
        }

        // 3. Try appending .gguf extension.
        let with_ext = path.with_extension("gguf");
        if with_ext.is_file() {
            return Ok(with_ext);
        }

        // 4. Try common model directories (Docker volumes, etc.).
        for dir in &["/models", "/data/models"] {
            let candidate = PathBuf::from(dir).join(format!("{name}.gguf"));
            if candidate.is_file() {
                return Ok(candidate);
            }
        }

        // 5. Exact name in model store.
        if let Some(p) = self.store.gguf_path(name) {
            return Ok(p);
        }

        // 5. Try normalized name (Ollama tag format).
        let normalized = normalize_model_name(name);
        if normalized != name && let Some(p) = self.store.gguf_path(&normalized) {
            return Ok(p);
        }

        Err(format!(
            "Model '{name}' not found. Accepted formats:\n  \
             - GGUF file path: /models/model.gguf\n  \
             - Directory with GGUF: /models/mymodel/\n  \
             - Registered name: eullm import-ollama {name}"
        ))
    }
}

/// Find the first `.gguf` file in a directory.
fn find_gguf_in_dir(dir: &std::path::Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_file() && p.extension().is_some_and(|e| e == "gguf") {
            return Some(p);
        }
    }
    None
}

/// Normalize an Ollama-style model name for EULLM's store.
///
/// Ollama uses `name:tag` (e.g. `qwen3:14b`), but EULLM stores models
/// with dashes (e.g. `qwen3-14b`).  This converts `:` → `-` so that
/// API requests using Ollama naming conventions find the right model.
fn normalize_model_name(name: &str) -> String {
    name.replace(':', "-")
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
    pub cache_type_k: crate::inference::KvCacheType,
    pub cache_type_v: crate::inference::KvCacheType,
    pub batch_size: usize,
    pub cpu_moe: bool,
    pub n_cpu_moe: u32,
    pub rs_seq: u32,
    pub ctx_checkpoints: usize,
    pub checkpoint_min_step: u32,
    pub web_enabled: bool,
    pub store: ModelStore,
    /// Optional embedded chat UI. When `Some(port)`, a second listener is
    /// spawned on that port serving the chat at `/` (plus the API on the
    /// same port for same-origin fetches). When `None`, only the API
    /// listener on `cfg.port` is started — pure API surface, nothing on `/`.
    pub ui_port: Option<u16>,
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
        swap_lock: tokio::sync::Mutex::new(()),
        gpu_layers: cfg.gpu_layers,
        ctx_size: cfg.ctx_size,
        threads: cfg.threads,
        flash_attn: cfg.flash_attn,
        n_batch: cfg.n_batch,
        cache_type_k: cfg.cache_type_k,
        cache_type_v: cfg.cache_type_v,
        batch_size: cfg.batch_size,
        cpu_moe: cfg.cpu_moe,
        n_cpu_moe: cfg.n_cpu_moe,
        rs_seq: cfg.rs_seq,
        ctx_checkpoints: cfg.ctx_checkpoints,
        checkpoint_min_step: cfg.checkpoint_min_step,
        web_enabled: cfg.web_enabled,
        api_port: cfg.port,
        store: cfg.store,
    });
    let api_port = cfg.port;
    let ui_port_opt = cfg.ui_port;

    let api_app = api_router(state.clone());
    let api_addr = format!("0.0.0.0:{api_port}");
    let api_listener = TcpListener::bind(&api_addr).await?;
    tracing::info!("eullm API listening on {api_addr}");

    // Spawn the optional chat-UI listener on a separate port. It exposes the
    // same API surface (so the embedded JS can call same-origin) plus the
    // HTML/CSS/JS for the chat at `/`. Disabled by default for `eullm serve`
    // (headless) and enabled by default for `eullm run` (interactive).
    let ui_handle = if let Some(ui_port) = ui_port_opt {
        if ui_port == api_port {
            tracing::warn!(
                "ui_port == api_port ({ui_port}); refusing to bind UI to avoid collision. \
                 Pick a different --ui-port or pass --no-ui."
            );
            None
        } else {
            let ui_app = ui_router(state.clone());
            let ui_addr = format!("0.0.0.0:{ui_port}");
            match TcpListener::bind(&ui_addr).await {
                Ok(ui_listener) => {
                    tracing::info!(
                        "eullm chat UI listening on {ui_addr}  (open http://localhost:{ui_port}/)"
                    );
                    Some(tokio::spawn(async move {
                        if let Err(e) = axum::serve(ui_listener, ui_app)
                            .with_graceful_shutdown(shutdown_signal())
                            .await
                        {
                            tracing::error!("UI listener failed: {e}");
                        }
                    }))
                }
                Err(e) => {
                    tracing::warn!(
                        "Could not bind chat UI on {ui_addr}: {e}. \
                         API still served on {api_addr}; pass --ui-port to override."
                    );
                    None
                }
            }
        }
    } else {
        None
    };

    axum::serve(api_listener, api_app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    if let Some(h) = ui_handle {
        // The UI server listens for the same shutdown signal, but the signal
        // is observed by whichever task wakes first. Abort the leftover task
        // on the way out to avoid lingering listeners during repeated runs
        // (notably in tests).
        h.abort();
    }

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

/// Maximum request body size. Axum defaults to 2 MB, which is fine for text
/// but far too small for multimodal `/api/chat` requests: a base64-encoded
/// image or audio clip easily exceeds it (base64 inflates bytes by ~33%), so
/// a stock photo returns `413 length limit exceeded`. 64 MB comfortably fits
/// images and reasonable audio clips while still bounding abuse.
const MAX_BODY_BYTES: usize = 64 * 1024 * 1024;

/// Build the EULLM API router (Ollama + OpenAI compat) with CORS enabled
/// for Open WebUI and other frontends.
///
/// This router never serves the chat UI — clients hitting the API port get
/// only `/api/*` and `/v1/*`, so RAG systems and OpenAI-compatible tooling
/// see a pure API surface with no HTML on `/`.
fn api_router(state: Arc<AppState>) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .nest("/api", routes::api_routes())
        .nest("/v1", routes::openai_routes())
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .layer(cors)
        .with_state(state)
}

/// Build the chat-UI router. Includes the same API routes (so the embedded
/// JS can fetch same-origin), plus `/` and `/eullm-ui/*` for HTML/CSS/JS.
///
/// Always served on a separate port from the API so the two surfaces are
/// independently togglable and never collide.
fn ui_router(state: Arc<AppState>) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .nest("/api", routes::api_routes())
        .nest("/v1", routes::openai_routes())
        .merge(crate::ui::router())
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .layer(cors)
        .with_state(state)
}

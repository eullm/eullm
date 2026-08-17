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

mod auth;
mod ip_allowlist;
mod origin;
// `routes` is not part of the public API, but the terminal REPL in `main.rs`
// reuses `routes::sequential_to_channel` so that a model without a scheduler
// (multimodal forces `batch_size = 0`) streams through exactly the same code
// path as an HTTP request instead of a second, divergent one.
pub(crate) mod routes;

pub use auth::Identity;

use axum::Router;
use axum::extract::{ConnectInfo, DefaultBodyLimit, State};
use axum::response::IntoResponse;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::TcpListener;
use tower_http::cors::{AllowOrigin, Any, CorsLayer};

use crate::inference::embedding::EmbeddingModel;
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

/// The embedding slot — independent of `ModelSlot` on purpose. See
/// `AppState::ensure_embedding_model` for why this coexists with the
/// generation model rather than sharing its slot.
pub struct EmbeddingSlot {
    pub model_name: String,
    pub model: Arc<EmbeddingModel>,
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
    /// `--fit`: size the GPU offload against measured free VRAM before
    /// every load this server performs — the initial lazy load and every
    /// API-triggered swap. Never prompts (a daemon cannot ask questions):
    /// the MoE path always resolves to a loadable configuration, and the
    /// dense path proceeds with the computed split, or refuses the load
    /// when `fit_strict` is set. Without this, a swap reused the *launch*
    /// model's layer split for whatever model came next — observed live:
    /// `run --fit` sized a dense 27B at 43/64 layers, then a web-UI switch
    /// to a 22 GB MoE loaded with those same settings and OOM'd.
    pub fit: bool,
    /// `--fit-strict`: with `fit`, a model that does not fully fit is not
    /// loaded; the API caller gets the error instead of a partial split.
    pub fit_strict: bool,
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
    /// Enable extra internal diagnostics for the Rust engine layer (see
    /// `ServeConfig::rust_debug`). Applied to every model this server
    /// loads or swaps to.
    pub rust_debug: bool,

    /// Enable transparent web fetching: URLs in user messages are fetched
    /// and their content is injected into the prompt before inference.
    pub web_enabled: bool,

    /// Port the canonical API listener runs on (Ollama-compatible, default
    /// 11434). Exposed via `/api/version` so the chat UI (served on its own
    /// port) can display the endpoint external clients should point at.
    pub api_port: u16,

    /// Model store for resolving names → GGUF paths.
    pub store: ModelStore,

    /// Which source IPs may reach the API/UI — see `ip_allowlist`. Loaded
    /// once at startup; not affected by later edits to `.env` without a
    /// restart.
    pub ip_allowlist: ip_allowlist::IpAllowlist,

    /// Optional bearer-token authentication with per-key quotas — see `auth`.
    /// When enabled it runs *outside* the IP allowlist and a valid key admits
    /// the request regardless of source address, which is the only ordering
    /// that works behind Docker's address translation.
    pub api_keys: Arc<auth::ApiKeys>,

    /// Which browser origins may call the API — see `origin`.
    pub allowed_origins: origin::AllowedOrigins,

    /// What the web tool is allowed to fetch — see `tools::guard`. Resolved
    /// once at startup so a request cannot pay for re-reading the environment,
    /// and so the posture is logged before the first fetch rather than after.
    pub web_policy: crate::tools::guard::WebPolicy,

    /// Projector to fall back on when the model being loaded declares none
    /// of its own. Only ever an explicit `--mmproj`, never a projector that
    /// was discovered for some other model: pairing a projector with weights
    /// it was not trained on fails the load outright (`mismatch between text
    /// model and mmproj`), which is what happened when `run`'s auto-detected
    /// projector was passed here and then applied to every later swap.
    /// Normally `None`.
    pub fallback_mmproj: Option<PathBuf>,

    /// Whether a request's `model` field may name an arbitrary filesystem
    /// path. Off by default — see `resolve_model`.
    pub allow_model_paths: bool,

    /// The `(name, path)` this process was launched with, if any. Always
    /// resolvable even when `allow_model_paths` is off: `/api/tags` advertises
    /// this name, clients echo back what they were told, and refusing our own
    /// answer would break `eullm run ./model.gguf` on the first model swap
    /// back to it.
    pub launch_model: Option<(String, PathBuf)>,

    /// Second, independent model slot for text embeddings — see
    /// `ensure_embedding_model`. `None` until the first `/v1/embeddings` or
    /// `/api/embed` request names a model.
    pub embedding: tokio::sync::RwLock<Option<EmbeddingSlot>>,

    /// How many times a model was evicted to make VRAM room for the *other*
    /// slot (generation displacing the embedder, or the embedder displacing
    /// generation), in either direction. Not itself a problem — an
    /// ingestion run that evicts the LLM once and restores it once is
    /// exactly the intended use — but a steady-state rate of one eviction
    /// per request means a caller on a card too small for both is
    /// alternating instead of batching, paying a full model load on every
    /// call. Surfaced in `/api/version` (`model_swaps`) so that pattern is
    /// visible over time rather than only as an unexplained slowdown.
    pub cross_slot_evictions: std::sync::atomic::AtomicU64,

    /// Idle-unload deadline for the main slot — `None` means no timer is
    /// running (slot empty, or the model that loaded it asked to be kept
    /// forever). Reset on every request that touches the slot; checked by
    /// the background task spawned in `serve()`. See `KeepAlive`.
    main_deadline: tokio::sync::Mutex<Option<tokio::time::Instant>>,
    /// Same as `main_deadline`, for the embedding slot.
    embedding_deadline: tokio::sync::Mutex<Option<tokio::time::Instant>>,
    /// Applied when a request does not set its own `keep_alive` field —
    /// see `RuntimeOpts`/`ServeConfig::keep_alive`. `None` disables the
    /// idle-unload timer by default (a request can still opt in with an
    /// explicit `keep_alive`).
    pub default_keep_alive: Option<std::time::Duration>,
}

/// Why a model could not be made ready.
///
/// The distinction exists because it is the difference between a 4xx and a
/// 5xx, and getting it wrong is not cosmetic: a client with automatic retry
/// treats a 500 as "try again" and will hammer a request that can never
/// succeed. Asking for a model that does not exist is a client mistake;
/// failing to load one that does is ours.
#[derive(Debug)]
pub enum ModelError {
    /// No such model, by any accepted spelling. The caller should get a 404.
    NotFound(String),
    /// The model exists but could not be loaded: out of VRAM, corrupt GGUF,
    /// a context that will not allocate. The caller should get a 500.
    LoadFailed(String),
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(m) | Self::LoadFailed(m) => f.write_str(m),
        }
    }
}

impl From<String> for ModelError {
    /// Everything that is not explicitly a lookup miss is a load failure.
    fn from(m: String) -> Self {
        Self::LoadFailed(m)
    }
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
    pub async fn swap_model(
        &self,
        name: &str,
        override_batch_size: Option<usize>,
        override_ctx_size: Option<u32>,
    ) -> Result<(), ModelError> {
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
                let loaded_stem = model_identity_key(loaded);
                let req_stem = model_identity_key(&normalized);
                if loaded_stem == req_stem {
                    tracing::info!(
                        "Model {} already loaded (swapped by another request)",
                        crate::audit::sanitize_for_log(&normalized)
                    );
                    return Ok(());
                }
            }
        }

        let gguf_path = self.resolve_model(&normalized)?;
        // Resolve an mmproj sibling (vision projector) if the model store
        // declares one. Presence of a projector is the signal that this is
        // a multimodal model — we then force sequential loading (next step)
        // because the continuous-batching scheduler is text-only.
        // A projector beside the weights counts too: that is how every
        // HuggingFace vision repo is laid out, and a model resolved from a
        // path has no store entry to declare one. `--mmproj` is the last
        // resort, and it is logged loudly because pairing a projector with
        // weights it was not trained on produces confident nonsense rather
        // than an error.
        let mmproj_path = self
            .store
            .mmproj_path(&normalized)
            .or_else(|| crate::models::store::mmproj_beside(&gguf_path))
            .or_else(|| {
                self.fallback_mmproj.clone().inspect(|p| {
                    tracing::warn!(
                        "no projector of its own for {}; using --mmproj {}",
                        crate::audit::sanitize_for_log(&normalized),
                        p.display()
                    );
                })
            });
        if let Some(ref p) = mmproj_path {
            tracing::info!("Multimodal model detected — mmproj: {}", p.display());
        }
        tracing::info!(
            "Swapping model → {} ({})",
            crate::audit::sanitize_for_log(&normalized),
            gguf_path.display()
        );

        // ── 1. Unload the current model and WAIT for the scheduler
        //       thread to fully exit before loading the new model.
        //
        // Without this, both old and new LlamaBackend instances would
        // coexist, and both models would be in VRAM simultaneously —
        // causing OOM or a C-level crash in llama.cpp.
        self.unload_current().await?;
        tracing::info!("Previous model fully unloaded");

        // An embedder left resident from an earlier ingestion run would
        // otherwise shrink the free VRAM `--fit` measures below, sizing this
        // load as if the card were smaller than it actually is once the
        // embedder itself is later evicted by `ensure_embedding_model`. See
        // `evict_embedding_if_present_for_generation_load`.
        self.evict_embedding_if_present_for_generation_load().await;

        // ── 2. Load the new model ───────────────────────────────────
        // Gemma 4 requires f16 KV cache regardless of the server's configured
        // baseline (mixed SWA architecture) — see
        // `inference::correct_kv_cache_for_model` for the rationale. This
        // must run here, not just in the CLI `run` startup path, so a swap
        // triggered by a request (on `run` after startup, or on any `serve`)
        // hits the same correction instead of silently loading incompatible
        // KV cache types.
        let (cache_type_k, cache_type_v, kv_corrected) =
            crate::inference::correct_kv_cache_for_model(
                &normalized,
                self.cache_type_k,
                self.cache_type_v,
            );
        if kv_corrected {
            tracing::warn!(
                "Gemma 4 detected ({}) with non-f16 KV cache — auto-correcting to f16/f16 (mixed SWA architecture requires it)",
                crate::audit::sanitize_for_log(&normalized)
            );
        }
        // ── 2a. Size the offload for THIS model (--fit) ─────────────
        // The launch flags describe the launch model; whatever is being
        // swapped in has its own size, layer count, and (possibly) expert
        // layout. Runs after the unload above so the measured free VRAM is
        // real. Never prompts — same decision order as the `run` startup
        // flow: MoE auto-sizing first (always resolves), then the dense
        // split, headless.
        let effective_ctx = override_ctx_size.unwrap_or(self.ctx_size);
        let mut gpu_layers = self.gpu_layers;
        let mut cpu_moe = self.cpu_moe;
        let mut n_cpu_moe = self.n_cpu_moe;
        if self.fit {
            let kv_bpe_k = crate::inference::cache_type_bytes_per_elem(&cache_type_k);
            let kv_bpe_v = crate::inference::cache_type_bytes_per_elem(&cache_type_v);
            let moe_decision = if !cpu_moe && n_cpu_moe == 0 {
                crate::fit::run_moe_fit(&gguf_path, effective_ctx, kv_bpe_k, kv_bpe_v)
            } else {
                crate::fit::MoeFitDecision::NotMoe
            };
            match moe_decision {
                crate::fit::MoeFitDecision::Proceed { n_cpu_moe: computed } if computed > 0 => {
                    tracing::info!(
                        "--fit: MoE model — keeping expert tensors on CPU RAM for the \
                         first {computed} layers so the rest fits in VRAM"
                    );
                    n_cpu_moe = computed;
                    gpu_layers = -1;
                }
                crate::fit::MoeFitDecision::ProceedCpuMoeAndPartial { gpu_layers: gl } => {
                    tracing::info!(
                        "--fit: MoE model — even with every expert tensor on CPU RAM the \
                         rest doesn't fit fully; offloading a reduced layer split ({gl})"
                    );
                    cpu_moe = true;
                    gpu_layers = gl;
                }
                _ => match crate::fit::run_fit_headless(
                    &gguf_path,
                    self.gpu_layers,
                    effective_ctx,
                    self.fit_strict,
                    kv_bpe_k,
                    kv_bpe_v,
                ) {
                    crate::fit::FitOutcome::Proceed(n) => gpu_layers = n,
                    crate::fit::FitOutcome::Abort => {
                        return Err(ModelError::LoadFailed(format!(
                            "--fit-strict: model '{normalized}' does not fully fit in the \
                             currently free VRAM; not loading. Retry without --fit-strict \
                             to allow a partial CPU/GPU split."
                        )));
                    }
                },
            }

            // A `--gpu-layers` given at startup is an upper bound for every
            // model this server loads, not a count to apply blindly to a
            // model it was never chosen for.
            let capped = crate::fit::apply_gpu_layers_ceiling(gpu_layers, self.gpu_layers);
            if capped != gpu_layers {
                tracing::info!(
                    "--gpu-layers {}: offloading {capped} layers for {}",
                    self.gpu_layers,
                    crate::audit::sanitize_for_log(&normalized)
                );
                gpu_layers = capped;
            }
        }

        let config = InferenceConfig {
            model_path: gguf_path,
            gpu_layers,
            context_size: effective_ctx,
            threads: self.threads,
            flash_attn: self.flash_attn,
            n_batch: self.n_batch,
            cache_type_k,
            cache_type_v,
            // Multimodal: when the model store declares an mmproj sibling we
            // load it here so HTTP requests with `images` can route through
            // `engine.generate_multimodal()`. Models without an mmproj keep
            // the text-only fast path (None → no extra VRAM, no init cost).
            mmproj_path: mmproj_path.clone(),
            cpu_moe,
            n_cpu_moe,
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
        let rust_debug_for_swap = self.rust_debug;
        // What the banner shows unless the sequential engine has to shrink it.
        // The scheduler never shrinks: if its context does not fit, the whole
        // swap fails, so requested and actual are always the same there.
        let requested_ctx_size = override_ctx_size.unwrap_or(self.ctx_size);

        let (new_engine, new_scheduler, ready_info, effective_ctx_size) =
            tokio::task::spawn_blocking(move || {
                if batch_size > 0 {
                    let sched_config = SchedulerConfig {
                        max_batch_size: batch_size,
                        queue_capacity: batch_size * 8,
                        ctx_checkpoints: ctx_checkpoints_for_swap,
                        checkpoint_min_step: checkpoint_min_step_for_swap,
                        debug_logit_check: rust_debug_for_swap,
                    };
                    let sched = BatchScheduler::new(config, sched_config);
                    match sched.start() {
                        Ok((handle, model_info)) => {
                            Ok((None, Some(handle), Some(model_info), requested_ctx_size))
                        }
                        Err(e) => Err(format!("Failed to start scheduler: {e}")),
                    }
                } else {
                    match InferenceEngine::load(config) {
                        Ok(eng) => {
                            // Read it here, on the blocking thread that already
                            // owns the model, rather than after the move into the
                            // slot: the estimate needs the model's own metadata.
                            let info = eng.ready_info();
                            // May be smaller than `requested_ctx_size`: `load()`
                            // shrinks it automatically when the requested size
                            // does not fit, and the banner has to say what
                            // actually loaded — `info`'s KV estimate already
                            // reflects the shrunk size, so showing the
                            // requested one here would state a KV cost that
                            // belongs to a different context than the one
                            // printed next to it.
                            let actual_ctx_size = eng.context_size();
                            Ok((Some(Arc::new(eng)), None, Some(info), actual_ctx_size))
                        }
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

        tracing::info!(
            "Model swap complete → {} (batch_size={batch_size})",
            crate::audit::sanitize_for_log(&model_name)
        );

        // The diagnostic banner `run` prints at startup. `serve` starts with no
        // model, so this is the only place it can be emitted — and until it was
        // here, anyone driving the engine as a daemon never saw which backend
        // actually initialised, how many layers were offloaded, or what the KV
        // cache costs. That is the audience least able to guess and most likely
        // to be filing a report. See `crate::banner`.
        let info = ready_info.unwrap_or_default();
        crate::banner::ModelBanner {
            model_name: model_name
                .strip_prefix("eullm/")
                .unwrap_or(&model_name)
                .to_string(),
            gpu_layers: self.gpu_layers,
            cpu_moe: self.cpu_moe,
            n_cpu_moe: self.n_cpu_moe,
            rs_seq: self.rs_seq,
            ctx_checkpoints: self.ctx_checkpoints,
            checkpoint_min_step: self.checkpoint_min_step,
            batch_size,
            ctx_size: effective_ctx_size,
            n_ctx_train: info.n_ctx_train,
            flash_attn: self.flash_attn,
            cache_type_k,
            cache_type_v,
            kv_k_mib: info.kv_k_mib,
            kv_v_mib: info.kv_v_mib,
            web: self.web_enabled,
            threads: self.threads,
            n_batch: self.n_batch,
            rust_debug: self.rust_debug,
        }
        .print();

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

    /// Ensure the named embedding model is loaded, loading or swapping it in
    /// if needed, and return a handle to it.
    ///
    /// The residency decision — coexist with the generation model, or evict
    /// it — is made here rather than left to the caller, because the
    /// caller (a RAG pipeline) does not know how much VRAM the card has:
    /// the same request works unchanged on a 12 GB card (where the two
    /// cannot fit together) and a 16 GB one (where they can), and only this
    /// process can tell which situation it is in right now.
    ///
    /// 1. Already loaded under this name → return it, no eviction, no load.
    /// 2. Not loaded, and it fits in free VRAM alongside whatever is in the
    ///    main slot → load it into the embedding slot; the main slot is
    ///    untouched.
    /// 3. Not loaded, and it does not fit → evict the main slot first (a
    ///    generation request will reload it later; `resolve_model` and the
    ///    embedded chat UI both work unchanged against an empty main slot),
    ///    then load the embedder, which now has the whole card.
    ///
    /// On a non-CUDA build `fit::vram_bytes()` cannot answer "does it fit",
    /// so this always takes the coexist path (case 2) and lets a real
    /// allocation failure surface as a normal load error — the same
    /// posture `--fit` itself takes on those builds.
    pub async fn ensure_embedding_model(
        &self,
        name: &str,
        n_ctx: u32,
    ) -> Result<Arc<EmbeddingModel>, ModelError> {
        let _swap_guard = self.swap_lock.lock().await;

        let normalized = normalize_model_name(name);
        {
            let slot = self.embedding.read().await;
            if let Some(ref loaded) = *slot
                && model_identity_key(&loaded.model_name) == model_identity_key(&normalized)
            {
                return Ok(loaded.model.clone());
            }
        }

        let gguf_path = self.resolve_model(&normalized)?;
        let weights_bytes = std::fs::metadata(&gguf_path).map(|m| m.len()).unwrap_or(0);

        let main_loaded = self.slot.read().await.model_name.is_some();
        let fits_alongside = fits_in_free_vram(weights_bytes).unwrap_or(true);
        if main_loaded && !fits_alongside {
            tracing::info!(
                "Embedding model {} does not fit alongside the loaded generation model — \
                 evicting it to make room (will reload on the next generation request)",
                crate::audit::sanitize_for_log(&normalized)
            );
            self.unload_current().await?;
            self.cross_slot_evictions
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        tracing::info!(
            "Loading embedding model {} ({})",
            crate::audit::sanitize_for_log(&normalized),
            gguf_path.display()
        );
        let threads = self.threads;
        let model = tokio::task::spawn_blocking(move || EmbeddingModel::load(&gguf_path, threads, n_ctx))
            .await
            .map_err(|e| ModelError::LoadFailed(format!("Task join error: {e}")))?
            .map_err(|e| ModelError::LoadFailed(format!("Failed to load embedding model: {e}")))?;
        let model = Arc::new(model);

        {
            let mut slot = self.embedding.write().await;
            *slot = Some(EmbeddingSlot {
                model_name: normalized.clone(),
                model: model.clone(),
            });
        }
        tracing::info!(
            "Embedding model ready → {}",
            crate::audit::sanitize_for_log(&normalized)
        );
        Ok(model)
    }

    /// The mirror of the eviction inside `ensure_embedding_model`: called
    /// from `swap_model` before sizing a generation load, so an embedder
    /// left resident from a prior ingestion run does not silently shrink
    /// the VRAM budget `--fit` sizes against. Cheap when nothing is loaded
    /// (`RwLock::read` + an `Option` check) and a no-op unless `--fit` is
    /// on, since only `--fit` reads free VRAM to make a sizing decision in
    /// the first place — without it, evicting the embedder would trade a
    /// real, working configuration for a guess.
    async fn evict_embedding_if_present_for_generation_load(&self) {
        if !self.fit {
            return;
        }
        let was_loaded = self.embedding.read().await.is_some();
        if !was_loaded {
            return;
        }
        tracing::info!(
            "Generation request — evicting the resident embedding model to free VRAM for sizing \
             (reload it with a later /v1/embeddings or /api/embed request)"
        );
        *self.embedding.write().await = None;
        self.cross_slot_evictions
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Reset the main slot's idle-unload deadline. Called on every request
    /// that uses the main slot (after `ensure_model` in `routes.rs`), so an
    /// active conversation is never unloaded out from under it. `Immediate`
    /// unloads right away instead of scheduling a deadline — see
    /// `KeepAlive`.
    pub async fn touch_main_slot(&self, keep_alive: KeepAlive) {
        touch_deadline(&self.main_deadline, keep_alive, self.default_keep_alive);
        if keep_alive == KeepAlive::Immediate {
            let _ = self.unload().await;
        }
    }

    /// Same as `touch_main_slot`, for the embedding slot.
    pub async fn touch_embedding_slot(&self, keep_alive: KeepAlive) {
        touch_deadline(
            &self.embedding_deadline,
            keep_alive,
            self.default_keep_alive,
        );
        if keep_alive == KeepAlive::Immediate {
            *self.embedding.write().await = None;
        }
    }

    /// Background loop spawned once from `serve()`: every 30 seconds, unload
    /// either slot whose idle deadline has passed. 30s is coarse on purpose
    /// — this is a power-saving idle timer, not a latency-sensitive path,
    /// and checking every request would mean taking both slot locks on
    /// every single generation/embedding call for a comparison that is
    /// false almost all the time.
    async fn run_idle_unload_loop(self: Arc<Self>) {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));
        loop {
            interval.tick().await;
            let now = tokio::time::Instant::now();

            let main_expired = {
                let mut deadline = self.main_deadline.lock().await;
                let expired = deadline.is_some_and(|d| now >= d);
                if expired {
                    *deadline = None;
                }
                expired
            };
            if main_expired {
                tracing::info!("keep_alive expired — unloading idle generation model");
                let _ = self.unload().await;
            }

            let embedding_expired = {
                let mut deadline = self.embedding_deadline.lock().await;
                let expired = deadline.is_some_and(|d| now >= d);
                if expired {
                    *deadline = None;
                }
                expired
            };
            if embedding_expired {
                tracing::info!("keep_alive expired — unloading idle embedding model");
                *self.embedding.write().await = None;
            }
        }
    }

    /// Resolve a model name to a GGUF file path.
    ///
    /// Search order:
    /// 1. Direct GGUF file path — **only** when `allow_model_paths` is set, or
    ///    when it is the path this process was launched with
    /// 2. Directory containing a single .gguf file — same condition
    /// 3. Path without extension — try appending `.gguf`, same condition
    /// 4. Well-known container mount points (`/models`, `/data/models`)
    /// 5. Exact name in model store (`~/.eullm/models/{name}/*.gguf`)
    /// 6. Normalized name (Ollama tags: `qwen3:14b` → `qwen3-14b`)
    ///
    /// # Why steps 1–3 are gated
    ///
    /// The `model` field of an API request used to be handed straight to
    /// `PathBuf::from` and accepted if `is_file()`. Combined with an
    /// unauthenticated API that made every readable file on the host a valid
    /// model name. A caller could not read the file's contents back, but the
    /// error messages distinguished "not found" from "found but failed to
    /// load", which is a working oracle for probing the filesystem — and
    /// pointing the loader at, say, a 40 GB file is a denial of service on its
    /// own. Names that resolve inside the model store or a deliberate mount
    /// point cover every documented workflow; arbitrary paths are opt-in via
    /// `EULLM_ALLOW_MODEL_PATHS=1`.
    ///
    /// The launch path is always accepted regardless: `/api/tags` reports it as
    /// the loaded model's name, clients echo back what they were told, and
    /// refusing our own answer would break `eullm run ./model.gguf`.
    fn resolve_model(&self, name: &str) -> Result<PathBuf, ModelError> {
        let path = PathBuf::from(name);

        // 0. The model this process was launched with, by the name the API
        //    advertises for it or by its literal path. Exact match on either —
        //    never a stem or prefix comparison, which would turn this
        //    allowance into a way to reach any similarly named file.
        if let Some((launch_name, launch_path)) = &self.launch_model
            && (name == launch_name.as_str() || &path == launch_path)
            && launch_path.is_file()
        {
            return Ok(launch_path.clone());
        }

        if self.allow_model_paths {
            // 1. Direct GGUF file path?
            if path.is_file() {
                return Ok(path);
            }

            // 2. Directory containing .gguf files? Pick the first one.
            if path.is_dir()
                && let Some(gguf) = find_gguf_in_dir(&path)
            {
                return Ok(gguf);
            }

            // 3. Try appending .gguf extension.
            let with_ext = path.with_extension("gguf");
            if with_ext.is_file() {
                return Ok(with_ext);
            }
        }

        // 4. Try common model directories (Docker volumes, etc.). These are
        //    deliberate mount points, not arbitrary paths, so they stay
        //    available without the opt-in — but only a plain file name may be
        //    joined onto them, or `../` would walk straight back out.
        if crate::models::store::is_safe_filename(&format!("{name}.gguf")) {
            for dir in &["/models", "/data/models"] {
                let candidate = PathBuf::from(dir).join(format!("{name}.gguf"));
                if candidate.is_file() {
                    return Ok(candidate);
                }
            }
        }

        // 5. Exact name in model store.
        if let Some(p) = self.store.gguf_path(name) {
            return Ok(p);
        }

        // 5. Try normalized name (Ollama tag format).
        let normalized = normalize_model_name(name);
        if normalized != name
            && let Some(p) = self.store.gguf_path(&normalized)
        {
            return Ok(p);
        }

        Err(ModelError::NotFound(format!(
            "Model '{name}' not found. Accepted formats:\n  \
             - GGUF file path: /models/model.gguf\n  \
             - Directory with GGUF: /models/mymodel/\n  \
             - Registered name: eullm import-ollama {name}"
        )))
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

/// Whether `additional_bytes` fits in currently free VRAM, applying the same
/// floor `fit.rs` reserves for a normal model load
/// (`fit::MIN_FREE_TOTAL_RATIO`) plus a flat reserve for the embedder's own
/// small compute buffer — 256 MiB rather than `fit.rs`'s 640 MiB, since an
/// embedding model's context and micro-batch are both a fraction of an LLM's.
///
/// Deliberately not the layer-by-layer machinery in `fit.rs`: an embedding
/// model loads fully onto the GPU or not at all (see `EmbeddingModel::load`),
/// so this only ever needs a yes/no answer, never a partial split.
///
/// `None` when VRAM cannot be probed at all (non-CUDA build) — the caller
/// decides what "unknown" means for it; `ensure_embedding_model` treats it as
/// "assume yes" so a build that cannot measure VRAM behaves as it always has,
/// letting a real allocation failure surface as a normal load error.
fn fits_in_free_vram(additional_bytes: u64) -> Option<bool> {
    const EMBEDDING_COMPUTE_RESERVE_BYTES: u64 = 256 * 1024 * 1024;
    let (free, total) = crate::fit::vram_bytes()?;
    let floor = (total as f64 * crate::fit::MIN_FREE_TOTAL_RATIO) as u64;
    let usable = free
        .saturating_sub(floor)
        .saturating_sub(EMBEDDING_COMPUTE_RESERVE_BYTES);
    Some(additional_bytes <= usable)
}

/// How long a loaded model should be kept resident after a request, decoded
/// from a request's `keep_alive` field (Ollama's field of the same name and
/// meaning) — a pure function over the parsed JSON value, testable without a
/// running server, per the perimeter-config convention in
/// `engine/CLAUDE.md` (even though this is not a perimeter setting, the same
/// reason applies: precedence logic belongs in a function that can be
/// tested directly).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeepAlive {
    /// No `keep_alive` in the request — use the server's `--keep-alive`
    /// default (`AppState::default_keep_alive`), which may itself be "never
    /// idle-unload" if the server was started without the flag.
    Default,
    /// `keep_alive: -1` (or any negative number/duration) — never
    /// idle-unload this load; the deadline is cleared rather than set.
    Forever,
    /// `keep_alive: 0` — unload right after this request completes, instead
    /// of waiting for the idle timer.
    Immediate,
    /// A positive duration to keep the model resident, counted from the end
    /// of this request.
    For(std::time::Duration),
}

/// Parse a request body's `keep_alive` field. Accepts what Ollama accepts:
/// a bare number of seconds (`300`), a duration string with a unit
/// (`"5m"`, `"30s"`, `"2h"`), or a plain numeric string (`"300"`). An
/// absent field, `null`, or anything unparseable falls back to `Default`
/// rather than erroring — a malformed `keep_alive` should not fail the
/// request it rides along with.
pub fn parse_keep_alive(value: Option<&serde_json::Value>) -> KeepAlive {
    let Some(value) = value else {
        return KeepAlive::Default;
    };
    let seconds = if let Some(n) = value.as_f64() {
        Some(n)
    } else if let Some(s) = value.as_str() {
        parse_duration_string(s)
    } else {
        None
    };
    match seconds {
        None => KeepAlive::Default,
        Some(s) if s < 0.0 => KeepAlive::Forever,
        Some(0.0) => KeepAlive::Immediate,
        Some(s) => KeepAlive::For(std::time::Duration::from_secs_f64(s)),
    }
}

/// Parse `--keep-alive`'s CLI value into a duration, for `main.rs` to build
/// `ServeConfig::keep_alive` — the same duration grammar `parse_keep_alive`
/// accepts on a request's `keep_alive` field ("5m", "30s", "300"), minus the
/// 0/negative special cases: a *default* keep-alive of "unload immediately"
/// or "never" is nonsensical (the former means every load evicts itself, the
/// latter is just the flag being absent), so those are rejected here rather
/// than silently accepted and misread as `Duration::ZERO`.
pub fn parse_keep_alive_flag(s: &str) -> Result<std::time::Duration, String> {
    match parse_duration_string(s) {
        Some(secs) if secs > 0.0 => Ok(std::time::Duration::from_secs_f64(secs)),
        Some(_) => Err(format!(
            "--keep-alive must be a positive duration, got '{s}' \
             (0 or negative only make sense as a per-request keep_alive override)"
        )),
        None => Err(format!(
            "--keep-alive: cannot parse '{s}' as a duration — expected e.g. '5m', '30s', '2h', or a bare number of seconds"
        )),
    }
}

/// `"300"` → 300.0, `"5m"` → 300.0, `"1.5h"` → 5400.0. Returns `None` for
/// anything that is not a bare number optionally followed by one of
/// `s`/`m`/`h`.
fn parse_duration_string(s: &str) -> Option<f64> {
    let s = s.trim();
    if let Ok(n) = s.parse::<f64>() {
        return Some(n);
    }
    let (number, unit) = s.split_at(s.len() - 1);
    let n: f64 = number.parse().ok()?;
    match unit {
        "s" => Some(n),
        "m" => Some(n * 60.0),
        "h" => Some(n * 3600.0),
        _ => None,
    }
}

/// Shared implementation behind `touch_main_slot`/`touch_embedding_slot`:
/// resolve `keep_alive` (falling back to `default` when it is `Default`)
/// into a new deadline, or clear it for `Forever`/`Immediate` (the caller
/// handles the actual unload for `Immediate`).
fn touch_deadline(
    deadline: &tokio::sync::Mutex<Option<tokio::time::Instant>>,
    keep_alive: KeepAlive,
    default: Option<std::time::Duration>,
) {
    let resolved = match keep_alive {
        KeepAlive::Default => default.map(KeepAlive::For).unwrap_or(KeepAlive::Forever),
        other => other,
    };
    let new_deadline = match resolved {
        KeepAlive::For(d) => Some(tokio::time::Instant::now() + d),
        KeepAlive::Forever | KeepAlive::Immediate | KeepAlive::Default => None,
    };
    // `try_lock`: this runs on the hot request path (once per request, to
    // reset the idle timer) and must never block a response on the idle
    // loop's own lock acquisition, which happens at most once every 30s and
    // holds the lock only briefly. Losing a single deadline reset to a rare
    // collision is harmless — the next request resets it again, and the
    // idle loop only unloads a slot that has had no reset for the entire
    // keep_alive window.
    if let Ok(mut guard) = deadline.try_lock() {
        *guard = new_deadline;
    }
}

#[cfg(test)]
mod keep_alive_tests {
    use super::*;
    use std::time::Duration;

    fn v(s: &str) -> serde_json::Value {
        serde_json::from_str(s).expect("valid json literal")
    }

    #[test]
    fn an_absent_field_is_the_server_default() {
        assert_eq!(parse_keep_alive(None), KeepAlive::Default);
    }

    #[test]
    fn a_bare_number_is_seconds() {
        assert_eq!(
            parse_keep_alive(Some(&v("300"))),
            KeepAlive::For(Duration::from_secs(300))
        );
    }

    #[test]
    fn duration_strings_match_ollamas_grammar() {
        assert_eq!(
            parse_keep_alive(Some(&v("\"30s\""))),
            KeepAlive::For(Duration::from_secs(30))
        );
        assert_eq!(
            parse_keep_alive(Some(&v("\"5m\""))),
            KeepAlive::For(Duration::from_secs(300))
        );
        assert_eq!(
            parse_keep_alive(Some(&v("\"2h\""))),
            KeepAlive::For(Duration::from_secs(7200))
        );
        // A numeric string with no unit is still seconds.
        assert_eq!(
            parse_keep_alive(Some(&v("\"300\""))),
            KeepAlive::For(Duration::from_secs(300))
        );
    }

    /// `0` means "unload right after this request" — distinct from an
    /// absent field, which means "use the server default" and may well
    /// keep the model resident.
    #[test]
    fn zero_means_immediate() {
        assert_eq!(parse_keep_alive(Some(&v("0"))), KeepAlive::Immediate);
        assert_eq!(parse_keep_alive(Some(&v("\"0\""))), KeepAlive::Immediate);
    }

    /// Negative, in either form, means "never idle-unload this load" —
    /// Ollama's `-1`.
    #[test]
    fn negative_means_forever() {
        assert_eq!(parse_keep_alive(Some(&v("-1"))), KeepAlive::Forever);
        assert_eq!(parse_keep_alive(Some(&v("\"-1\""))), KeepAlive::Forever);
    }

    /// A malformed value must not fail the request it rides along with —
    /// it falls back to the server default rather than erroring.
    #[test]
    fn garbage_falls_back_to_default_rather_than_erroring() {
        assert_eq!(parse_keep_alive(Some(&v("\"banana\""))), KeepAlive::Default);
        assert_eq!(parse_keep_alive(Some(&v("null"))), KeepAlive::Default);
        assert_eq!(parse_keep_alive(Some(&v("true"))), KeepAlive::Default);
    }

    #[test]
    fn the_cli_flag_parser_accepts_only_a_positive_duration() {
        assert_eq!(
            parse_keep_alive_flag("5m").unwrap(),
            Duration::from_secs(300)
        );
        assert!(
            parse_keep_alive_flag("0").is_err(),
            "0 as a *default* keep-alive is nonsensical — every load would evict itself"
        );
        assert!(parse_keep_alive_flag("-1").is_err());
        assert!(parse_keep_alive_flag("banana").is_err());
    }

    #[test]
    fn touch_deadline_resolves_default_against_the_servers_own_default() {
        let deadline = tokio::sync::Mutex::new(None);
        // No server default configured (`None`) → Default resolves to
        // Forever, i.e. no timer at all — matches every release before
        // --keep-alive existed, where nothing unloaded a model on its own.
        touch_deadline(&deadline, KeepAlive::Default, None);
        assert!(
            deadline.try_lock().unwrap().is_none(),
            "no --keep-alive configured means Default must not start a timer"
        );

        // A server default of 5 minutes turns Default into an actual deadline.
        touch_deadline(&deadline, KeepAlive::Default, Some(Duration::from_secs(300)));
        assert!(deadline.try_lock().unwrap().is_some());
    }

    #[test]
    fn touch_deadline_forever_and_immediate_both_clear_any_running_timer() {
        let deadline = tokio::sync::Mutex::new(Some(tokio::time::Instant::now()));
        touch_deadline(&deadline, KeepAlive::Forever, Some(Duration::from_secs(300)));
        assert!(deadline.try_lock().unwrap().is_none());

        let deadline = tokio::sync::Mutex::new(Some(tokio::time::Instant::now()));
        touch_deadline(&deadline, KeepAlive::Immediate, Some(Duration::from_secs(300)));
        assert!(deadline.try_lock().unwrap().is_none());
    }
}

/// Normalize an Ollama-style model name for EULLM's store.
///
/// Ollama uses `name:tag` (e.g. `qwen3:14b`), but EULLM stores models
/// with dashes (e.g. `qwen3-14b`).  This converts `:` → `-` so that
/// API requests using Ollama naming conventions find the right model.
fn normalize_model_name(name: &str) -> String {
    name.replace(':', "-")
}

/// Canonical comparison key for "is this model already loaded?" checks: the
/// last path component (a loaded model may be a full `.gguf` path), with a
/// `.gguf` extension stripped, compared case-insensitively.
///
/// Deliberately NOT `Path::file_stem`. Model names legitimately contain dots
/// (`qwen3.6-27b`, `ornith-1.0-35b-gguf-ud-q5_k_xl`), and `file_stem` cuts
/// at the LAST dot, which collapsed every `ornith-1.*` quant into the same
/// `ornith-1` identity — reported as #345: switching between two quants of
/// the same repo was a silent no-op because the swap believed the requested
/// model was already loaded.
pub(crate) fn model_identity_key(name: &str) -> String {
    let last = name.rsplit(['/', '\\']).next().unwrap_or(name);
    let base = match last.char_indices().rev().nth(4) {
        Some((i, _)) if last[i..].eq_ignore_ascii_case(".gguf") => &last[..i],
        _ => last,
    };
    base.to_ascii_lowercase()
}

/// Configuration for starting the API server.
pub struct ServeConfig {
    pub port: u16,
    /// See `AppState::fallback_mmproj`.
    pub mmproj: Option<PathBuf>,
    pub model_name: Option<String>,
    pub engine: Option<Arc<InferenceEngine>>,
    pub scheduler: Option<SchedulerHandle>,
    pub gpu_layers: i32,
    /// Auto-size the GPU offload before every model load (see
    /// `AppState::fit`). Pass the user's `--fit` flag, never a value a
    /// previous fit computed.
    pub fit: bool,
    /// With `fit`, refuse a load that does not fully fit instead of
    /// offloading a partial split (see `AppState::fit_strict`).
    pub fit_strict: bool,
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
    /// Enable extra internal diagnostics for the Rust engine layer (NaN/Inf
    /// logit scan per token — see `SchedulerConfig::debug_logit_check`).
    /// Applied to every model this server loads or swaps to. Off by
    /// default: zero added per-token cost, matches upstream llama.cpp.
    pub rust_debug: bool,
    pub web_enabled: bool,
    pub store: ModelStore,
    /// Optional embedded chat UI. When `Some(port)`, a second listener is
    /// spawned on that port serving the chat at `/` (plus the API on the
    /// same port for same-origin fetches). When `None`, only the API
    /// listener on `cfg.port` is started — pure API surface, nothing on `/`.
    pub ui_port: Option<u16>,
    /// The `(advertised name, GGUF path)` this process was launched with, when
    /// it was launched with a model (`eullm run`). `None` for headless `serve`,
    /// which starts with an empty slot. See `AppState::launch_model`.
    pub launch_model: Option<(String, PathBuf)>,
    /// Default idle-unload duration for a load whose request did not set its
    /// own `keep_alive` — see `AppState::default_keep_alive`. `None` (the
    /// default) means never idle-unload automatically, matching every
    /// release before this flag existed.
    pub keep_alive: Option<std::time::Duration>,
}

/// Start the API server on the given port with graceful shutdown support.
///
/// The server shuts down cleanly on SIGTERM or SIGINT (Ctrl+C), finishing
/// in-flight requests before exiting. This is critical for Docker containers
/// (which send SIGTERM on `docker stop`) and systemd services.
pub async fn serve(cfg: ServeConfig) -> Result<(), Box<dyn std::error::Error>> {
    let env_file = std::path::Path::new(".env");
    let ip_allowlist = ip_allowlist::IpAllowlist::load(env_file);

    // Authentication first: configuration that is present but unusable is fatal
    // here. An operator who set EULLM_API_KEYS asked for authentication, and
    // starting an open API because of a typo in it is the one outcome that must
    // not be possible.
    let api_keys = Arc::new(auth::ApiKeys::load(env_file).map_err(|e| {
        format!(
            "{e}\n  Expected id:secret[:rpm=N] entries, comma-separated. \
             Refusing to start: serving without the authentication you configured \
             would be worse than not starting."
        )
    })?);
    let allowed_origins = origin::AllowedOrigins::load(env_file);
    let web_policy = crate::tools::guard::WebPolicy::from_env();
    let allow_model_paths = matches!(
        std::env::var("EULLM_ALLOW_MODEL_PATHS")
            .ok()
            .map(|v| v.trim().to_ascii_lowercase())
            .as_deref(),
        Some("1" | "true" | "yes" | "on")
    );

    if api_keys.is_enabled() {
        tracing::info!(
            "API authentication: enabled — keys: {}  [source: {}]",
            api_keys.describe(),
            api_keys.source(),
        );
        // State this explicitly. It is the one place where enabling a control
        // relaxes another, and an operator discovering it from behaviour rather
        // than from a log line is how a deployment ends up unintentionally open.
        tracing::info!(
            "A valid API key admits a request from any source address; the IP allowlist \
             below then applies only to requests without a key (which are refused with 401)."
        );
    } else {
        tracing::info!(
            "API authentication: disabled ({}). Set EULLM_API_KEYS=id:secret to require \
             a bearer token — necessary behind Docker's published ports, where every \
             external client arrives as the bridge gateway address.",
            api_keys.source()
        );
    }
    tracing::info!(
        "Allowed source IPs/subnets: {}  [source: {}]",
        ip_allowlist.describe(),
        ip_allowlist.source(),
    );
    tracing::info!(
        "Allowed browser origins: {}  [source: {}]",
        allowed_origins.describe(),
        allowed_origins.source(),
    );
    if cfg.web_enabled {
        tracing::info!("Web tool: enabled — fetchable: {}", web_policy.describe());
    }
    if allow_model_paths {
        tracing::warn!(
            "EULLM_ALLOW_MODEL_PATHS is set: a request's `model` field may name any \
             GGUF path readable by this process. Intended for local use; do not \
             combine it with an API reachable by untrusted callers."
        );
    }

    // Fail loudly at startup if the audit destination is unusable, rather than
    // warning once per request after the fact. The trail exists to produce a
    // defensible record; degrading silently to "no record" is the one failure
    // mode it must not have.
    let audit = crate::audit::AuditLogger::new();
    {
        // Which store the server resolves model names against. Omitting this
        // is how `eullm list` and the API came to disagree about whether a
        // model existed, with no way to tell that they were reading different
        // directories.
        let (root, source) = cfg.store.root_with_source();
        tracing::info!("Model store: {}  [source: {source}]", root.display());
    }
    match audit.check_writable() {
        Ok(()) => tracing::info!("Audit trail: {}", audit.log_path().display()),
        // Explicitly configured destination that doesn't work → refuse to
        // start. Someone who set EULLM_AUDIT_DIR (or mounted a volume at it)
        // asked for the trail; serving without one silently is the failure
        // this check exists to prevent.
        Err(e) if crate::audit::AuditLogger::is_explicitly_configured() => {
            return Err(format!(
                "EULLM_AUDIT_DIR is set but the audit trail is not writable: {e}\n  \
                 Point it at a writable, persistent path (in Docker, one backed by a \
                 mounted volume), or unset it to fall back to ~/.eullm/audit."
            )
            .into());
        }
        // Nobody asked for a specific destination — warn loudly and serve.
        // Refusing to start an inference server over a log file the operator
        // never configured would turn a read-only home directory into an outage.
        Err(e) => tracing::warn!(
            "Audit trail disabled: {} is not writable ({e}). Inference will work, but \
             no audit records will be kept. Set EULLM_AUDIT_DIR to a writable path to \
             enable it.",
            audit.log_path().display(),
        ),
    }

    let state = Arc::new(AppState {
        fallback_mmproj: cfg.mmproj.clone(),
        slot: tokio::sync::RwLock::new(ModelSlot {
            model_name: cfg.model_name,
            engine: cfg.engine,
            scheduler: cfg.scheduler,
        }),
        swap_lock: tokio::sync::Mutex::new(()),
        gpu_layers: cfg.gpu_layers,
        fit: cfg.fit,
        fit_strict: cfg.fit_strict,
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
        rust_debug: cfg.rust_debug,
        web_enabled: cfg.web_enabled,
        api_port: cfg.port,
        store: cfg.store,
        ip_allowlist,
        api_keys,
        allowed_origins,
        web_policy,
        allow_model_paths,
        launch_model: cfg.launch_model,
        embedding: tokio::sync::RwLock::new(None),
        cross_slot_evictions: std::sync::atomic::AtomicU64::new(0),
        main_deadline: tokio::sync::Mutex::new(None),
        embedding_deadline: tokio::sync::Mutex::new(None),
        default_keep_alive: cfg.keep_alive,
    });
    let idle_unload_state = state.clone();
    tokio::spawn(async move {
        idle_unload_state.run_idle_unload_loop().await;
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
                        if let Err(e) = axum::serve(
                            ui_listener,
                            ui_app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
                        )
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

    axum::serve(
        api_listener,
        api_app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
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
        let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
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

/// Reject any request whose source IP isn't in `state.ip_allowlist`, before
/// it reaches CORS, body parsing, or any handler. See `ip_allowlist` for why
/// the socket always binds `0.0.0.0` regardless and this check is the real
/// boundary.
///
/// Runs *inside* [`enforce_auth`], so by the time it executes an [`Identity`]
/// is always present. A request that presented a valid key is admitted whatever
/// its source address: behind Docker's published ports every external client
/// arrives as the bridge gateway, so refusing an authenticated caller on
/// address grounds would leave the operator with no working configuration at
/// all. See the `auth` module docs for the full reasoning.
async fn enforce_ip_allowlist(
    State(state): State<Arc<AppState>>,
    ConnectInfo(addr): ConnectInfo<std::net::SocketAddr>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let authenticated = req
        .extensions()
        .get::<Identity>()
        .is_some_and(Identity::is_authenticated);
    if authenticated || state.ip_allowlist.is_allowed(addr.ip()) {
        next.run(req).await
    } else {
        tracing::warn!("Rejected request from disallowed IP {}", addr.ip());
        (
            axum::http::StatusCode::FORBIDDEN,
            "source IP not in the configured allowlist",
        )
            .into_response()
    }
}

/// Verify the bearer token, attach an [`Identity`] to the request, and charge
/// the key's quota. Outermost layer on both routers.
///
/// When no keys are configured this attaches an anonymous identity and gets out
/// of the way — the IP allowlist is then the only control, which is the right
/// default for the single-user local case the engine is most often used for.
///
/// `allow_query_token` is true only on the UI listener. The embedded chat runs
/// in a browser, which cannot be handed a header before its first navigation,
/// so `?api_key=…` bootstraps it (the page then stores the key and sends it as
/// a header on every subsequent fetch). It is deliberately **not** accepted on
/// the API listener: a token in a URL ends up in proxy logs, browser history
/// and `Referer` headers, and no programmatic client needs it.
async fn enforce_auth(
    State((state, allow_query_token)): State<(Arc<AppState>, bool)>,
    mut req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let presented = extract_token(&req, allow_query_token);
    match state.api_keys.authenticate(presented.as_deref()) {
        Ok(identity) => {
            req.extensions_mut().insert(identity);
            next.run(req).await
        }
        Err(auth::AuthError::Missing) => unauthorized(
            "missing API key — send it as `Authorization: Bearer <key>` or `X-Api-Key: <key>`",
        ),
        Err(auth::AuthError::Invalid) => {
            // No key id to name: logging the presented token would write a
            // credential into the log file, and it may be a valid key for a
            // *different* deployment.
            tracing::warn!("Rejected request with an invalid API key");
            unauthorized("invalid API key")
        }
        Err(auth::AuthError::RateLimited {
            key_id,
            retry_after_s,
        }) => {
            tracing::warn!("Key '{key_id}' is over its per-minute quota");
            (
                axum::http::StatusCode::TOO_MANY_REQUESTS,
                [(axum::http::header::RETRY_AFTER, retry_after_s.to_string())],
                axum::Json(serde_json::json!({
                    "error": format!(
                        "rate limit exceeded for key '{key_id}' — retry in {retry_after_s}s"
                    )
                })),
            )
                .into_response()
        }
    }
}

/// 401 with the `WWW-Authenticate` challenge, so a client library can tell an
/// authentication failure from a generic refusal.
fn unauthorized(message: &str) -> axum::response::Response {
    (
        axum::http::StatusCode::UNAUTHORIZED,
        [(
            axum::http::header::WWW_AUTHENTICATE,
            "Bearer realm=\"eullm\"",
        )],
        axum::Json(serde_json::json!({ "error": message })),
    )
        .into_response()
}

/// Pull the token out of `Authorization: Bearer`, `X-Api-Key`, or — on the UI
/// listener only — an `api_key` query parameter.
fn extract_token(req: &axum::extract::Request, allow_query_token: bool) -> Option<String> {
    let headers = req.headers();
    if let Some(v) = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
    {
        // Case-insensitive scheme, per RFC 7235.
        let v = v.trim();
        if let Some(rest) = v
            .split_once(' ')
            .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("bearer"))
            .map(|(_, rest)| rest)
        {
            return Some(rest.trim().to_string());
        }
    }
    if let Some(v) = headers.get("x-api-key").and_then(|v| v.to_str().ok()) {
        return Some(v.trim().to_string());
    }
    if allow_query_token {
        return req.uri().query().and_then(|q| {
            q.split('&')
                .filter_map(|kv| kv.split_once('='))
                .find(|(k, _)| *k == "api_key")
                .map(|(_, v)| v.to_string())
        });
    }
    None
}

/// Refuse a cross-origin request that has side effects, before it reaches a
/// handler.
///
/// CORS is not this check. CORS decides whether a browser hands the *response*
/// back to the calling page; the request itself is still executed. For `GET` on
/// a read-only endpoint that distinction is academic, but `POST /api/unload` or
/// a model swap take effect regardless of whether the attacker can read the
/// reply — and a simple `POST` with `Content-Type: text/plain` needs no
/// preflight, so the CORS layer never gets a chance to object.
///
/// Requests with no `Origin` header are left alone: that is every non-browser
/// client, and an origin policy has never applied to them.
async fn enforce_origin(
    State(state): State<Arc<AppState>>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let method = req.method().clone();
    let is_safe = matches!(
        method,
        axum::http::Method::GET | axum::http::Method::HEAD | axum::http::Method::OPTIONS
    );
    if !is_safe
        && let Some(origin) = req
            .headers()
            .get(axum::http::header::ORIGIN)
            .and_then(|v| v.to_str().ok())
        && !state.allowed_origins.is_allowed(origin)
    {
        tracing::warn!(
            "Rejected {} from disallowed origin {}",
            method,
            crate::audit::sanitize_for_log(origin)
        );
        return (
            axum::http::StatusCode::FORBIDDEN,
            axum::Json(serde_json::json!({
                "error": "request origin is not allowed — set EULLM_ALLOWED_ORIGINS \
                          if this frontend should be permitted"
            })),
        )
            .into_response();
    }
    next.run(req).await
}

/// CORS layer honouring `state.allowed_origins`.
///
/// `allow_headers(Any)` and `allow_methods(Any)` stay permissive on purpose:
/// once the *origin* is constrained, restricting which headers that trusted
/// origin may send buys nothing and breaks frontends that send their own
/// (Open WebUI sends several).
fn cors_layer(state: &Arc<AppState>) -> CorsLayer {
    let origins = state.allowed_origins.clone();
    CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(move |origin, _req| {
            origin
                .to_str()
                .map(|o| origins.is_allowed(o))
                .unwrap_or(false)
        }))
        .allow_methods(Any)
        .allow_headers(Any)
}

/// Build the EULLM API router (Ollama + OpenAI compat) with CORS enabled
/// for Open WebUI and other frontends.
///
/// This router never serves the chat UI — clients hitting the API port get
/// only `/api/*` and `/v1/*`, so RAG systems and OpenAI-compatible tooling
/// see a pure API surface with no HTML on `/`.
fn api_router(state: Arc<AppState>) -> Router {
    let cors = cors_layer(&state);

    // Layers run outermost-last: `enforce_auth` is added last, so it sees the
    // request first. The order is load-bearing — see `enforce_ip_allowlist` for
    // why authentication must precede the address check rather than follow it.
    Router::new()
        .nest("/api", routes::api_routes())
        .nest("/v1", routes::openai_routes())
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .layer(cors)
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            enforce_origin,
        ))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            enforce_ip_allowlist,
        ))
        .layer(axum::middleware::from_fn_with_state(
            (state.clone(), false),
            enforce_auth,
        ))
        .with_state(state)
}

/// Build the chat-UI router. Includes the same API routes (so the embedded
/// JS can fetch same-origin), plus `/` and `/eullm-ui/*` for HTML/CSS/JS.
///
/// Always served on a separate port from the API so the two surfaces are
/// independently togglable and never collide.
fn ui_router(state: Arc<AppState>) -> Router {
    let cors = cors_layer(&state);

    // The UI listener nests the same API routes, so it must enforce the same
    // controls — exempting it would simply move the open door to another port.
    // It differs in one respect: a token may arrive as `?api_key=…`, because a
    // browser cannot set a header on its first navigation. See `enforce_auth`.
    Router::new()
        .nest("/api", routes::api_routes())
        .nest("/v1", routes::openai_routes())
        .merge(crate::ui::router())
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .layer(cors)
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            enforce_origin,
        ))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            enforce_ip_allowlist,
        ))
        .layer(axum::middleware::from_fn_with_state(
            (state.clone(), true),
            enforce_auth,
        ))
        .with_state(state)
}

#[cfg(test)]
mod http_tests {
    //! End-to-end tests over a real listener.
    //!
    //! Every unit test in this crate exercises a function; none of them ever
    //! sent an HTTP request. That gap is not theoretical. Three defects found
    //! by hand in two days lived entirely on the `serve` path: the model lists
    //! ignored the store so a pulled model could not be selected from an
    //! editor, the diagnostic banner was never printed, and a whole family of
    //! models answered nothing at all. A suite of 217 green tests had nothing
    //! to say about any of them.
    //!
    //! These bind an ephemeral port and speak real HTTP through the real
    //! middleware stack, because that is where the behaviour lives: the
    //! allowlist reads a peer address, and a handler tested in isolation never
    //! has one.
    //!
    //! Deliberately no model is loaded. Inference needs a GGUF that CI cannot
    //! download on every push, and the endpoints that answer without one are
    //! exactly the ones that broke.

    use super::*;
    use std::net::SocketAddr;

    /// A store directory with one model in it, laid out the way a pull leaves
    /// it: a directory named after the id, a manifest, and the weights.
    fn store_with_one_model(dir: &std::path::Path, id: &str) -> ModelStore {
        let model_dir = dir.join(id);
        std::fs::create_dir_all(&model_dir).expect("model dir");
        std::fs::write(model_dir.join("model.gguf"), b"not a real gguf").expect("weights");
        let manifest = serde_json::json!({
            "id": id,
            "name": id,
            "description": "test fixture",
            "languages": ["en"],
            "base": "test",
            "vram_gb": 1,
            "size_bytes": 15,
            "license": "Apache-2.0",
            "digest": "sha256:0",
            "pulled_at": "2026-07-30T00:00:00Z",
            "status": "ready",
            "gguf_file": "model.gguf",
        });
        std::fs::write(
            model_dir.join("manifest.json"),
            serde_json::to_string(&manifest).expect("manifest json"),
        )
        .expect("manifest");
        ModelStore::at(dir.to_path_buf())
    }

    /// Start the API on 127.0.0.1:0 and return its base URL.
    ///
    /// Port 0 rather than a fixed one: these run in parallel with every other
    /// test in the binary, and a hardcoded port makes the suite fail depending
    /// on what else is listening on the machine.
    async fn spawn(store: ModelStore) -> String {
        // A path that does not exist, so the perimeter types fall back to
        // their defaults instead of reading a developer's real `.env`.
        let absent = std::path::Path::new("/nonexistent/eullm-test/.env");
        let state = Arc::new(AppState {
            fallback_mmproj: None,
            slot: tokio::sync::RwLock::new(ModelSlot {
                model_name: None,
                engine: None,
                scheduler: None,
            }),
            swap_lock: tokio::sync::Mutex::new(()),
            gpu_layers: 0,
            fit: false,
            fit_strict: false,
            ctx_size: 4096,
            threads: 1,
            flash_attn: false,
            n_batch: 512,
            cache_type_k: crate::inference::KvCacheType::F16,
            cache_type_v: crate::inference::KvCacheType::F16,
            batch_size: 1,
            cpu_moe: false,
            n_cpu_moe: 0,
            rs_seq: 0,
            ctx_checkpoints: 0,
            checkpoint_min_step: 8192,
            rust_debug: false,
            web_enabled: false,
            api_port: 0,
            store,
            ip_allowlist: ip_allowlist::IpAllowlist::load(absent),
            api_keys: Arc::new(auth::ApiKeys::load(absent).expect("no keys configured")),
            allowed_origins: origin::AllowedOrigins::load(absent),
            web_policy: crate::tools::guard::WebPolicy::from_env(),
            allow_model_paths: false,
            launch_model: None,
            embedding: tokio::sync::RwLock::new(None),
            cross_slot_evictions: std::sync::atomic::AtomicU64::new(0),
            main_deadline: tokio::sync::Mutex::new(None),
            embedding_deadline: tokio::sync::Mutex::new(None),
            default_keep_alive: None,
        });

        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("local addr");
        let app = api_router(state);
        tokio::spawn(async move {
            let _ = axum::serve(
                listener,
                app.into_make_service_with_connect_info::<SocketAddr>(),
            )
            .await;
        });
        format!("http://{addr}")
    }

    async fn get_json(url: &str) -> (reqwest::StatusCode, serde_json::Value) {
        let r = reqwest::get(url).await.expect("request");
        let status = r.status();
        let body = r.json().await.unwrap_or(serde_json::Value::Null);
        (status, body)
    }

    async fn post_json(url: &str, body: serde_json::Value) -> (reqwest::StatusCode, String) {
        let r = reqwest::Client::new()
            .post(url)
            .json(&body)
            .send()
            .await
            .expect("request");
        let status = r.status();
        (status, r.text().await.unwrap_or_default())
    }

    #[tokio::test]
    async fn api_tags_lists_a_model_that_is_on_disk_but_not_loaded() {
        // The shape of the bug reported in #294: both model lists were built
        // from the built-in catalog plus whatever happened to be loaded, so a
        // model pulled from a URL or a HuggingFace repo was invisible to them
        // even though it was sitting in the store, ready to run.
        let tmp = std::env::temp_dir().join(format!("eullm-tags-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let store = store_with_one_model(&tmp, "a-pulled-model");
        let base = spawn(store).await;

        let (status, body) = get_json(&format!("{base}/api/tags")).await;
        assert_eq!(status, 200);
        let names: Vec<String> = body["models"]
            .as_array()
            .expect("models array")
            .iter()
            .filter_map(|m| m["name"].as_str().map(str::to_string))
            .collect();
        assert!(
            names.iter().any(|n| n.contains("a-pulled-model")),
            "a model in the store must appear in /api/tags, got {names:?}"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn openai_models_lists_a_model_that_is_on_disk_but_not_loaded() {
        // This endpoint is the one that decides whether a model can be picked
        // at all: a coding editor offers what `/v1/models` names, so a model
        // it never names cannot be selected, however well it runs elsewhere.
        let tmp = std::env::temp_dir().join(format!("eullm-v1models-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let store = store_with_one_model(&tmp, "a-pulled-model");
        let base = spawn(store).await;

        let (status, body) = get_json(&format!("{base}/v1/models")).await;
        assert_eq!(status, 200);
        let ids: Vec<String> = body["data"]
            .as_array()
            .expect("data array")
            .iter()
            .filter_map(|m| m["id"].as_str().map(str::to_string))
            .collect();
        assert!(
            ids.iter().any(|i| i.contains("a-pulled-model")),
            "a model in the store must appear in /v1/models, got {ids:?}"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn an_unknown_model_is_refused_by_name_and_not_with_a_500() {
        // A wrong model name is a client mistake and must read like one. The
        // failure mode worth pinning is a panic or a bare 500, which tells the
        // caller nothing and looks like the server is broken.
        let tmp = std::env::temp_dir().join(format!("eullm-unknown-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let store = store_with_one_model(&tmp, "a-pulled-model");
        let base = spawn(store).await;

        let (status, body) = post_json(
            &format!("{base}/api/chat"),
            serde_json::json!({
                "model": "this-model-does-not-exist",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": false,
            }),
        )
        .await;
        assert!(
            status.is_client_error() || status == 503,
            "expected a client error or 503, got {status}"
        );
        assert!(
            body.contains("this-model-does-not-exist"),
            "the refusal must name the model the caller asked for, got: {body}"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn the_version_endpoint_answers_without_a_model() {
        // `serve` starts with an empty slot, and a client probing whether the
        // server is up must get an answer before any model exists.
        let tmp = std::env::temp_dir().join(format!("eullm-version-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let store = store_with_one_model(&tmp, "a-pulled-model");
        let base = spawn(store).await;

        let (status, body) = get_json(&format!("{base}/api/version")).await;
        assert_eq!(status, 200);
        assert_eq!(
            body["version"].as_str(),
            Some(env!("CARGO_PKG_VERSION")),
            "/api/version must report the crate version"
        );
        assert_eq!(
            body["model_swaps"].as_u64(),
            Some(0),
            "a fresh server has evicted nothing yet"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn embedding_endpoints_refuse_an_unknown_model_by_name() {
        // Same shape as `an_unknown_model_is_refused_by_name_and_not_with_a_500`
        // for generation: a model name that resolves to nothing is a client
        // mistake and must read like one on both embedding endpoints, not as
        // a generic 500.
        let tmp = std::env::temp_dir().join(format!("eullm-embed-404-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let store = store_with_one_model(&tmp, "a-pulled-model");
        let base = spawn(store).await;

        for path in ["/api/embed", "/v1/embeddings"] {
            let (status, body) = post_json(
                &format!("{base}{path}"),
                serde_json::json!({ "model": "this-model-does-not-exist", "input": "hi" }),
            )
            .await;
            assert!(
                status.is_client_error(),
                "{path}: expected a client error, got {status}"
            );
            assert!(
                body.contains("this-model-does-not-exist"),
                "{path}: the refusal must name the model the caller asked for, got: {body}"
            );
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn embedding_endpoints_require_input() {
        let tmp = std::env::temp_dir().join(format!("eullm-embed-noinput-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let store = store_with_one_model(&tmp, "a-pulled-model");
        let base = spawn(store).await;

        // Refused before any model resolution is attempted — a malformed
        // request should not trigger a load/swap on its way to being
        // rejected.
        let (status, _) = post_json(
            &format!("{base}/api/embed"),
            serde_json::json!({ "model": "a-pulled-model" }),
        )
        .await;
        assert_eq!(status, 400);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn generate_with_no_model_field_and_nothing_loaded_still_503s_before_the_warm_load_check() {
        // The empty-prompt warm-load short-circuit sits after `ensure_model`
        // on purpose (see `generate`'s comment): it must not bypass "no
        // model available" and answer a fabricated "loaded" response for a
        // server with nothing loaded and no model named.
        let tmp = std::env::temp_dir().join(format!("eullm-warmload-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let store = ModelStore::at(tmp.clone());
        let base = spawn(store).await;

        let (status, body) = post_json(
            &format!("{base}/api/generate"),
            serde_json::json!({ "prompt": "" }),
        )
        .await;
        assert_eq!(status, 503);
        assert!(body.contains("No model loaded"));
        let _ = std::fs::remove_dir_all(&tmp);
    }
}

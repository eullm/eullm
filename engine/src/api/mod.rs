//! eullm REST API.
//!
//! Exposes a standard LLM API (both `/api` and `/v1` OpenAI-compatible)
//! so that existing tools (Open WebUI, LangChain, n8n) work out of the box.
//!
//! Supports two inference backends:
//! - **Sequential** (`InferenceEngine`): one request at a time.
//! - **Continuous batching** (`SchedulerHandle`): multiple concurrent requests.
//!
//! Supports **models loaded on request**: when a request names a model that
//! is not loaded, the server loads it, and makes room first. With one model
//! at a time (`--max-loaded-models 1`, the default) that is a swap: the loaded
//! model goes, and requests it was still answering are cut off with an error,
//! as they always were. With several, the model goes only when the new one
//! needs its place or its memory, the least recently used first, and a busy
//! one is waited for rather than cut off. The residents are kept in
//! `resident::ResidentModels`.

mod auth;
mod decision_policy;
mod decision_traces;
pub(crate) mod ip_allowlist;
mod origin;
#[cfg(test)]
mod real_model_tests;
mod resident;
mod route;
// `routes` is not part of the public API, but the terminal REPL in `main.rs`
// reuses `routes::sequential_to_channel` so that a model without a scheduler
// (multimodal forces `batch_size = 0`) streams through exactly the same code
// path as an HTTP request instead of a second, divergent one.
pub(crate) mod routes;
mod systemone;
mod thinking;

pub use auth::Identity;
pub use route::{CandidateFacts, CatalogFacts, DEFAULT_AUTO_TIMEOUT_MS};

use axum::Router;
use axum::extract::{ConnectInfo, DefaultBodyLimit, State};
use axum::response::IntoResponse;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::TcpListener;
use tower_http::cors::{AllowOrigin, Any, CorsLayer};

use llama_cpp_2::llama_backend::LlamaBackend;

use crate::inference::decision::DecisionModel;
use crate::inference::embedding::EmbeddingModel;
use crate::inference::{
    BatchScheduler, InferenceConfig, InferenceEngine, SchedulerConfig, SchedulerHandle,
};
use crate::models::ModelStore;

/// The embedding slot — independent of the generation models on purpose. See
/// `AppState::ensure_embedding_model` for why this coexists with the
/// generation model rather than sharing its slot.
pub struct EmbeddingSlot {
    pub model_name: String,
    pub model: Arc<EmbeddingModel>,
    /// Whether this slot was populated by `--embedding-model` at launch
    /// (`true`) or loaded ad hoc by a runtime `/api/embed`/`/v1/embeddings`
    /// request (`false`). The distinction is what `--embedding-model` is
    /// *for*: a reserved companion is never evicted just because the
    /// generation model is being swapped (`load_generation_model`'s own `--fit`
    /// reserves its footprint instead, so it keeps its place), where an ad
    /// hoc embedder has no such guarantee and is evicted on any generation
    /// swap that needs the room back (`evict_embedding_if_present_for_generation_load`).
    pub is_reserved_companion: bool,
}

/// The decision slot behind `/v1/systemone` — a third slot, handled like the
/// embedding one and for the same reason: a small decision model can stay
/// resident next to the chat model instead of taking turns with it. See
/// `AppState::ensure_decision_model`.
pub struct DecisionSlot {
    pub model_name: String,
    pub model: Arc<DecisionModel>,
    /// Same meaning as `EmbeddingSlot::is_reserved_companion`, for
    /// `--decision-model`.
    pub is_reserved_companion: bool,
    /// VRAM a request can take on top of the weights: its KV cache at the
    /// per-request ceiling plus a compute buffer
    /// (`fit::decision_reserve_bytes`). The weights show up as used VRAM
    /// once loaded; the context the model keeps between requests is
    /// released before a generation model is sized, so it does not.
    pub reserve_bytes: u64,
}

/// Shared state for API handlers.
pub struct AppState {
    /// The one `LlamaBackend` this process created (see
    /// `inference::init_shared_backend`), shared by every model load —
    /// the launch model, every later `load_generation_model`, and every embedding
    /// model. `LlamaBackend::init()` is a process-wide one-time marker;
    /// a second independent instance fails with `BackendAlreadyInitialized`
    /// while the first is still alive, so this must be the same `Arc` the
    /// launch model itself loaded with, not a fresh one.
    pub backend: Arc<LlamaBackend>,
    /// The generation models in memory. A request takes the read guard just
    /// long enough to find its model and take a lease on it; the write guard
    /// is taken only to install a model or take one out, never across a load
    /// or a scheduler's shutdown.
    pub(crate) models: tokio::sync::RwLock<resident::ResidentModels>,
    /// Serializes loads and unloads of every slot — generation, embedding
    /// and decision — so that two loads never size themselves against the
    /// same free VRAM. Taken before any other lock here, never after one:
    /// `swap_lock` → `models` → `embedding`/`decision`. A load waiting for a
    /// busy model to finish lets go of it meanwhile.
    swap_lock: tokio::sync::Mutex<()>,
    /// `--max-loaded-models`: how many generation models may be resident at
    /// once. At 1, the default, a request for another model replaces the
    /// resident one, as every release before the flag did.
    pub max_loaded_models: usize,
    /// How long a load waits for a busy resident to finish before it gives
    /// up with a 503: `BUSY_EVICTION_WAIT`, shorter in tests.
    busy_wait: std::time::Duration,
    /// How many generation models were unloaded to make room for another
    /// one (`/api/version`'s `generation_evictions`). With one model at a
    /// time that is every swap; with several, a steady rate of one per
    /// request means the models asked for do not fit together.
    pub generation_evictions: std::sync::atomic::AtomicU64,
    /// Holds every generation load just before it starts, until the test
    /// lets it go — to prove what a request does while a load is running.
    #[cfg(test)]
    load_gate: Option<Arc<LoadGate>>,

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
    /// `--n-ubatch` as given: the physical micro-batch of every model this
    /// server loads, and the compute buffer `--fit` reserves for it. `None`
    /// is llama.cpp's 512, or what an expert cache chooses for its load
    /// (`fit::MOE_CACHE_N_UBATCH`).
    pub n_ubatch: Option<u32>,
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
    /// `--mtp`: MTP drafts per step for every model this server loads (see
    /// `InferenceConfig::mtp`).
    pub mtp: u32,
    /// `--mtp-p-min` (see `InferenceConfig::mtp_p_min`).
    pub mtp_p_min: f32,
    /// `--mtp-model` (see `InferenceConfig::mtp_model`).
    pub mtp_model: Option<std::path::PathBuf>,
    /// `--moe-cache`, as the user gave it: every load sizes its own cache
    /// from it (see `fit::plan_moe_cache`).
    pub moe_cache: Option<crate::fit::MoeCache>,
    /// `--vram-reserve` in bytes: what every load sizes itself to leave free
    /// and its context probe requires (see `fit::DEFAULT_VRAM_RESERVE_MIB`).
    pub vram_reserve: u64,
    /// `--no-mmap`: every model this server loads is read into memory
    /// rather than mapped (see `InferenceConfig::no_mmap`).
    pub no_mmap: bool,
    /// `--mmap`: a load with an expert cache keeps the file mapped rather
    /// than reading it in to pin the experts (`fit::plan_read_into_memory`).
    pub mmap: bool,
    /// `--moe-prefetch`, as the user gave it: every load with experts kept
    /// in RAM and pinned gets the slots (see `fit::prefetch_slots`), and an
    /// expert cache keeps their VRAM out of its own where it can.
    pub moe_prefetch: u32,
    /// `--load-threads`: readers of the model file ahead of every load this
    /// server makes (see `crate::readahead`).
    pub load_threads: crate::readahead::LoadThreads,
    /// `--kv-unified`: one KV cache for every sequence of a model this
    /// server loads.
    pub kv_unified: bool,
    /// `--split-mode`, `--tensor-split`: how every model this
    /// server loads goes across the GPUs (see `crate::gpu_split`).
    pub gpu_split: crate::gpu_split::GpuSplit,
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

    /// `--mmproj-offload` / `--no-mmproj-offload` as the user gave them, or
    /// `None` to let sizing place each model's projector. The flag and not a
    /// decision, for the same reason `gpu_layers` is the flag: a placement
    /// worked out for the launch model says nothing about the next one.
    pub mmproj_offload: Option<bool>,

    /// Whether a request's `model` field may name an arbitrary filesystem
    /// path. Off by default — see `resolve_model`.
    pub allow_model_paths: bool,

    /// The `(name, path)` this process was launched with, if any. Always
    /// resolvable even when `allow_model_paths` is off: `/api/tags` advertises
    /// this name, clients echo back what they were told, and refusing our own
    /// answer would break `eullm run ./model.gguf` on the first model swap
    /// back to it.
    pub launch_model: Option<(String, PathBuf)>,

    /// `--default-model`: the model a request that names none — no `model`
    /// field, or an empty one — is answered by, loaded for it when it is not
    /// loaded. `None`: the most recently used generation model answers, and
    /// with none loaded the request is refused. Resolvable by its name or
    /// path whatever `allow_model_paths` says, as the launch model is.
    pub default_model: Option<NamedModel>,

    /// The models `"model": "auto"` chooses between, and how long it may
    /// take to (`--auto-model`, `--auto-timeout-ms`); `None` without them,
    /// and `auto` is then a model name like any other. See `route`.
    pub(crate) router: Option<route::RouteTable>,
    /// `--default-model auto`: a request that names no model is routed, as
    /// one naming `auto` is.
    pub(crate) default_auto: bool,

    /// Second, independent model slot for text embeddings — see
    /// `ensure_embedding_model`. `None` until the first `/v1/embeddings` or
    /// `/api/embed` request names a model.
    pub embedding: tokio::sync::RwLock<Option<EmbeddingSlot>>,

    /// Third model slot, for `/v1/systemone` — see `ensure_decision_model`.
    /// `None` until `--decision-model` or the first request naming a model.
    pub decision: tokio::sync::RwLock<Option<DecisionSlot>>,
    /// Most tokens of context one decision request may use
    /// (`--decision-ctx`). Every model loaded into the decision slot gets it.
    pub decision_ctx: u32,
    /// The operator's rules for every `/v1/systemone` request
    /// (`EULLM_DECISION_POLICY`) — see `decision_policy`. Read once at
    /// startup.
    pub decision_policy: decision_policy::DecisionPolicy,
    /// Where every decision's redacted trace goes (`EULLM_DECISION_TRACES`)
    /// — see `decision_traces`. `None`, the default: traces are off.
    pub decision_traces: Option<Arc<decision_traces::DecisionTraces>>,

    /// How many times a model was evicted to make VRAM room for another
    /// slot (generation displacing the embedder or the decision model, or
    /// either of those displacing generation). Not itself a problem — an
    /// ingestion run that evicts the LLM once and restores it once is
    /// exactly the intended use — but a steady-state rate of one eviction
    /// per request means a caller on a card too small for both is
    /// alternating instead of batching, paying a full model load on every
    /// call. Surfaced in `/api/version` (`model_swaps`) so that pattern is
    /// visible over time rather than only as an unexplained slowdown.
    pub cross_slot_evictions: std::sync::atomic::AtomicU64,

    /// Woken when the last request on a generation model finishes, so the
    /// idle-unload loop acts on `keep_alive: 0` as soon as the request is
    /// over instead of at its next tick. Each generation model's own
    /// deadline lives in its `resident::Usage`, set when its requests end.
    idle: Arc<tokio::sync::Notify>,
    /// How often the idle-unload loop looks for an expired keep_alive: 30 s,
    /// shorter in tests.
    idle_tick: std::time::Duration,
    /// Idle-unload deadline for the embedding slot — `None` means no timer
    /// is running (slot empty, or the model that loaded it asked to be kept
    /// forever). Reset on every request that touches the slot; checked by
    /// the background task spawned in `serve()`. See `KeepAlive`.
    embedding_deadline: tokio::sync::Mutex<Option<tokio::time::Instant>>,
    /// Same as `embedding_deadline`, for the decision slot.
    decision_deadline: tokio::sync::Mutex<Option<tokio::time::Instant>>,
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
    /// Loading the model needs room only a resident model still answering
    /// requests can give, and none finished in time. The caller should get a
    /// 503 with `Retry-After`: the same request can succeed in a moment.
    Busy(String),
    /// Loading the model would unload another, and the load was asked not
    /// to (`EvictPolicy::Never`): a warm-up, which has nobody waiting for
    /// the model and must not take one away from someone who is.
    NoRoom(String),
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(m) | Self::LoadFailed(m) | Self::Busy(m) | Self::NoRoom(m) => {
                f.write_str(m)
            }
        }
    }
}

/// Whether a load may unload resident generation models to make room.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EvictPolicy {
    /// As a request needs: the count, and what fits beside what, decide.
    Allowed,
    /// Never: the load is refused with `ModelError::NoRoom` instead.
    Never,
}

impl From<String> for ModelError {
    /// Everything that is not explicitly a lookup miss is a load failure.
    fn from(m: String) -> Self {
        Self::LoadFailed(m)
    }
}

impl AppState {
    /// Load a generation model that is not resident, making room for it
    /// first (see `make_room`). The new model loads with the same inference
    /// settings as every other.
    ///
    /// `override_batch_size` allows the caller to change the number of
    /// concurrent batch slots for the new model (e.g. more slots for a
    /// smaller model that uses less VRAM).  Pass `None` to keep the
    /// batch size from the CLI launch.
    ///
    /// Returns the model with a lease on it for the request that asked,
    /// taken under the same guard that found or installed it, so the request
    /// is answered by the model it named even when another load follows
    /// straight after. `keep_alive` is that request's.
    ///
    /// This is the **write** path — only one load runs at a time, and a load
    /// that waits for a busy model to finish lets go of the lock meanwhile.
    /// `evict` says whether it may unload other models at all.
    pub(crate) async fn load_generation_model(
        &self,
        name: &str,
        override_batch_size: Option<usize>,
        override_ctx_size: Option<u32>,
        keep_alive: KeepAlive,
        evict: EvictPolicy,
    ) -> Result<resident::SlotSnapshot, ModelError> {
        // Serialize loads — if another request is already loading, wait for
        // it to finish instead of starting a parallel load.
        let mut swap_guard = Some(self.swap_lock.lock().await);

        // Normalize Ollama-style names: "qwen3:14b" → "qwen3-14b"
        let normalized = normalize_model_name(name);

        // Re-check after acquiring the lock — another request may have
        // loaded it while this one waited.
        if let Some(snapshot) = self.lease_loaded(&normalized, None, keep_alive).await {
            return Ok(snapshot);
        }

        // Resolved before anything is unloaded, so that a name that does not
        // exist is a 404 that costs nobody their model.
        let gguf_path = self.resolve_model(&normalized)?;
        // The same file under another of its names — a store name for a model
        // launched by its path, or another name `eullm pull` linked to the
        // same weights: loading it again would hold it twice.
        if let Some(snapshot) = self
            .lease_loaded(&normalized, Some(&gguf_path), keep_alive)
            .await
        {
            return Ok(snapshot);
        }
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
        let found = self
            .store
            .resolve_mmproj(&normalized)
            .or_else(|| crate::models::store::mmproj_beside(&gguf_path));
        let named = found.is_none() && self.fallback_mmproj.is_some();
        let mmproj_path = found.or_else(|| {
            self.fallback_mmproj.clone().inspect(|p| {
                tracing::warn!(
                    "no projector of its own for {}; using --mmproj {}",
                    crate::audit::sanitize_for_log(&normalized),
                    p.display()
                );
            })
        });
        // `--mtp` drafts on the scheduler's one slot only, which a projector
        // rules out: see `projector_with_mtp`.
        let (mmproj_path, mtp_note) = crate::models::store::projector_with_mtp(
            mmproj_path,
            named,
            self.mtp > 0 && override_batch_size.unwrap_or(self.batch_size) == 1,
            &crate::audit::sanitize_for_log(&normalized),
        );
        if let Some(note) = mtp_note {
            if mmproj_path.is_some() {
                tracing::warn!("{note}");
            } else {
                tracing::info!("{note}");
            }
        }
        if let Some(ref p) = mmproj_path {
            tracing::info!("Multimodal model detected — mmproj: {}", p.display());
        }
        tracing::info!(
            "Loading model → {} ({})",
            crate::audit::sanitize_for_log(&normalized),
            gguf_path.display()
        );

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

        // What sizing reads about the model, read once for every attempt.
        // The launch flags describe the launch model; whatever is being
        // loaded has its own size, layer count, and (possibly) expert
        // layout.
        let effective_ctx = override_ctx_size.unwrap_or(self.ctx_size);
        // The expert cache, where this machine can run one.
        let moe_cache = crate::fit::effective_moe_cache(
            self.moe_cache,
            self.fit,
            self.cpu_moe,
            self.n_cpu_moe,
            self.mtp,
        )
        .and_then(|request| match crate::fit::moe_cache_support() {
            Ok(()) => Some(request),
            Err(why) => {
                tracing::info!("--moe-cache: {why}; loading without the cache");
                None
            }
        });
        // The prefetch runs where the expert cache does, on one CUDA or ROCm GPU; it
        // is on by default, so elsewhere it goes without a word.
        let moe_prefetch = if crate::fit::moe_cache_support().is_ok() {
            self.moe_prefetch
        } else {
            0
        };
        let ram_total = crate::fit::system_ram_bytes();
        let info = crate::fit::read_gguf_info(&gguf_path);
        let file_size = crate::fit::model_file_bytes(&gguf_path);
        let layout = match (&info, file_size) {
            (Some(i), size) if self.fit && size > 0 => {
                crate::fit::read_model_moe_layout(&gguf_path, i.n_layers)
            }
            _ => None,
        };
        let kv_bpe_k = crate::inference::cache_type_bytes_per_elem(&cache_type_k);
        let kv_bpe_v = crate::inference::cache_type_bytes_per_elem(&cache_type_v);
        // The MTP head drafts only on the scheduler with one slot, which a
        // model with a projector never gets (see `batch_size` below); its
        // context comes after the load, so sizing must leave it room.
        let drafts = self.mtp > 0
            && mmproj_path.is_none()
            && override_batch_size.unwrap_or(self.batch_size) == 1;
        let mtp_reserve = if drafts {
            crate::fit::mtp_reserve_bytes(
                info.as_ref(),
                effective_ctx,
                kv_bpe_k,
                kv_bpe_v,
                self.n_ubatch.unwrap_or(crate::inference::DEFAULT_N_UBATCH),
            )
        } else {
            0
        };
        let sizing = Sizing {
            info: info.as_ref(),
            layout: layout.as_ref(),
            file_size,
            ctx_size: effective_ctx,
            kv_bpe_k,
            kv_bpe_v,
            // The projector is loaded with the model, always, so sizing has
            // to count it — see `fit::place_mmproj` for where it goes and why.
            mmproj_bytes: crate::fit::mmproj_footprint_bytes(mmproj_path.as_deref()),
            flags: crate::fit::OffloadFlags {
                gpu_layers: self.gpu_layers,
                cpu_moe: self.cpu_moe,
                n_cpu_moe: self.n_cpu_moe,
                mmproj_offload: self.mmproj_offload,
                moe_cache,
                vram_reserve: self.vram_reserve,
                auto_n_ubatch: self.n_ubatch.is_none(),
                moe_prefetch: crate::fit::MoePrefetch {
                    slots: moe_prefetch,
                    no_mmap: self.no_mmap,
                    keep_mapped: self.mmap,
                    ram_total,
                },
            },
            mtp_reserve,
        };

        let mut make_more_room = false;
        loop {
            // ── 1. Make room, and size the load for THIS model (--fit) ──
            // Never prompts — same decision order as the `run` startup
            // flow: projector placement, MoE auto-sizing (always resolves),
            // then the dense split, headless — all in one `fit::plan_offload`.
            let fits_now = make_more_room.then_some(false);
            let plan = match self
                .make_room(
                    &mut swap_guard,
                    &normalized,
                    &gguf_path,
                    &sizing,
                    fits_now,
                    keep_alive,
                    evict,
                )
                .await?
            {
                Room::Loaded(snapshot) => return Ok(snapshot),
                Room::Ready(plan) => plan,
            };
            if let Some(plan) = &plan {
                let hint = match (&plan.basis, plan.free_vram, sizing.info) {
                    (
                        crate::fit::OffloadBasis::Dense(crate::fit::FitDecision::Partial {
                            layers,
                            n_layers,
                        }),
                        Some(free),
                        Some(info),
                    ) => crate::fit::partial_split_hint(
                        free,
                        info,
                        file_size,
                        effective_ctx,
                        kv_bpe_k,
                        kv_bpe_v,
                        self.vram_reserve,
                        *layers,
                        *n_layers,
                    ),
                    _ => None,
                };
                plan.print_decision(file_size, self.fit_strict, hint.as_deref());
                if self.fit_strict && plan.refused_by_strict() {
                    return Err(ModelError::LoadFailed(format!(
                        "--fit-strict: model '{normalized}' does not fully fit in the \
                         currently free VRAM; not loading. Retry without --fit-strict \
                         to allow a partial CPU/GPU split."
                    )));
                }
                // A `--gpu-layers` given at startup is an upper bound for
                // every model this server loads, not a count to apply blindly
                // to a model it was never chosen for.
                if plan.capped_from.is_some() {
                    tracing::info!(
                        "--gpu-layers {}: offloading {} layers for {}",
                        self.gpu_layers,
                        plan.gpu_layers,
                        crate::audit::sanitize_for_log(&normalized)
                    );
                }
            }
            let moe_cache_grow = plan.as_ref().and_then(|plan| plan.moe_cache_grow);
            let (gpu_layers, cpu_moe, n_cpu_moe, mmproj_placement, moe_cache_bytes) = match &plan {
                Some(plan) => (
                    plan.gpu_layers,
                    plan.cpu_moe,
                    plan.n_cpu_moe,
                    plan.mmproj,
                    plan.moe_cache_bytes,
                ),
                // Without sizing a size in MiB is used as given; `auto` has
                // nothing to size against.
                None => (
                    self.gpu_layers,
                    self.cpu_moe,
                    self.n_cpu_moe,
                    crate::fit::MmprojPlacement::from_flag(self.mmproj_offload),
                    match moe_cache {
                        Some(crate::fit::MoeCache::Mib(mib)) => u64::from(mib) << 20,
                        _ => 0,
                    },
                ),
            };

            // The micro-batch and the file mapping this load takes: the
            // flags', unless the expert cache chose (`fit::plan_moe_cache`,
            // `fit::plan_read_into_memory`).
            let cache_n_ubatch = plan.as_ref().and_then(|plan| plan.n_ubatch);
            let load_n_ubatch = cache_n_ubatch
                .or(self.n_ubatch)
                .unwrap_or(crate::inference::DEFAULT_N_UBATCH);
            let load_n_batch = crate::inference::batch_for_ubatch(self.n_batch, load_n_ubatch);
            if cache_n_ubatch.is_some() {
                tracing::info!(
                    "experts in RAM: reading prompts {load_n_ubatch} tokens at a time (--n-ubatch), \
                     so that the experts in RAM are copied to the GPU once per {load_n_ubatch} \
                     prompt tokens instead of {}",
                    crate::inference::DEFAULT_N_UBATCH
                );
            }
            let (mut load_no_mmap, why) = crate::fit::plan_read_into_memory(
                self.no_mmap,
                self.mmap,
                plan.as_ref().map_or(0, |plan| plan.moe_cache_host_bytes),
                ram_total,
            );
            if let Some(why) = why {
                tracing::info!("{why}");
            }
            // A model that goes to the GPUs whole and is more than half the
            // memory, or on a network file system, is read in: mapped, it
            // loads slowly or not at all (`fit::read_in_whole_on_gpu`).
            if !load_no_mmap {
                let whole_on_gpu = crate::fit::vram_bytes().is_some_and(|(_, total)| total > 0)
                    && crate::fit::every_layer_on_gpu(
                        gpu_layers,
                        crate::fit::read_gguf_info(&gguf_path).map(|info| info.n_layers),
                        cpu_moe,
                        n_cpu_moe,
                        moe_cache_bytes,
                    );
                if let Some(why) = crate::fit::read_in_whole_on_gpu(
                    file_size,
                    whole_on_gpu,
                    crate::readahead::memory_for_cache(),
                    crate::readahead::network_file_system(&gguf_path),
                    self.mmap,
                ) {
                    tracing::info!("{why}");
                    load_no_mmap = true;
                }
            }
            let moe_prefetch_slots =
                crate::fit::prefetch_slots(moe_prefetch, load_no_mmap, cpu_moe, n_cpu_moe);
            if moe_prefetch_slots > 0 {
                tracing::info!(
                    "--moe-prefetch: the experts in RAM of a long prompt are copied to the GPU \
                     ahead of their layer, into {moe_prefetch_slots} slots of VRAM \
                     (--moe-prefetch 0 turns it off)"
                );
            }

            // ── 2. Load the new model ───────────────────────────────
            let config = InferenceConfig {
                model_path: gguf_path.clone(),
                gpu_layers,
                context_size: effective_ctx,
                threads: self.threads,
                flash_attn: self.flash_attn,
                n_batch: load_n_batch,
                n_ubatch: load_n_ubatch,
                cache_type_k,
                cache_type_v,
                // Multimodal: when the model store declares an mmproj sibling
                // we load it here so HTTP requests with `images` can route
                // through `engine.generate_multimodal()`. Models without an
                // mmproj keep the text-only fast path (None → no extra VRAM,
                // no init cost).
                mmproj_path: mmproj_path.clone(),
                mmproj_on_gpu: mmproj_placement.on_gpu(),
                cpu_moe,
                n_cpu_moe,
                rs_seq: self.rs_seq,
                mtp: self.mtp,
                mtp_p_min: self.mtp_p_min,
                mtp_model: self.mtp_model.clone(),
                moe_cache_bytes,
                moe_cache_grow,
                vram_reserve: self.vram_reserve,
                no_mmap: load_no_mmap,
                moe_prefetch_slots,
                load_threads: self.load_threads,
                kv_unified: self.kv_unified,
                gpu_split: self.gpu_split.clone(),
            };
            if mmproj_path.is_some() {
                tracing::info!("{}", mmproj_placement.describe());
            }

            // The continuous-batching scheduler is text-only — it does not
            // route mtmd chunks. For multimodal models we therefore force the
            // sequential `InferenceEngine` (batch_size=0). Vision is
            // interactive single-user anyway, so losing batching here is not
            // a practical regression.
            let batch_size = if mmproj_path.is_some() {
                0
            } else {
                override_batch_size.unwrap_or(self.batch_size)
            };
            let ctx_checkpoints_for_swap = self.ctx_checkpoints;
            let checkpoint_min_step_for_swap = self.checkpoint_min_step;
            let rust_debug_for_swap = self.rust_debug;
            // What the banner shows unless the sequential engine has to
            // shrink it. The scheduler never shrinks: if its context does not
            // fit, the whole load fails, so requested and actual are always
            // the same there.
            let requested_ctx_size = effective_ctx;
            let backend_for_swap = self.backend.clone();

            #[cfg(test)]
            if let Some(gate) = &self.load_gate {
                gate.arrived.notify_one();
                gate.proceed.notified().await;
            }
            // Loads are serialized, so what free VRAM loses across this one
            // is what this model holds: `/api/ps`'s `size_vram`.
            let free_before = crate::fit::vram_bytes().map(|(free, _)| free);

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
                        match sched.start(backend_for_swap) {
                            Ok((handle, model_info)) => {
                                Ok((None, Some(handle), Some(model_info), requested_ctx_size))
                            }
                            Err(e) => Err(format!("Failed to start scheduler: {e}")),
                        }
                    } else {
                        match InferenceEngine::load(config, backend_for_swap) {
                            Ok(eng) => {
                                // Read it here, on the blocking thread that
                                // already owns the model, rather than after
                                // the move into the slot: the estimate needs
                                // the model's own metadata.
                                let info = eng.ready_info();
                                // May be smaller than `requested_ctx_size`:
                                // `load()` shrinks it automatically when the
                                // requested size does not fit, and the banner
                                // has to say what actually loaded — `info`'s
                                // KV estimate already reflects the shrunk
                                // size, so showing the requested one here
                                // would state a KV cost that belongs to a
                                // different context than the one printed next
                                // to it.
                                let actual_ctx_size = eng.context_size();
                                Ok((Some(Arc::new(eng)), None, Some(info), actual_ctx_size))
                            }
                            Err(e) => Err(format!("Failed to load model: {e}")),
                        }
                    }
                })
                .await
                .map_err(|e| format!("Task join error: {e}"))??;

            // A sequential engine shrinks its context, silently, when the one
            // asked for does not fit. Beside other resident models that is
            // the card being divided after all: give it back, make more room,
            // and load once more. Alone, or the second time, the shrunk
            // context is what there is, as it always was.
            if self.fit
                && !make_more_room
                && effective_ctx_size < requested_ctx_size
                && !self.models.read().await.is_empty()
            {
                tracing::warn!(
                    "{}: its context had to shrink from {requested_ctx_size} to \
                     {effective_ctx_size} tokens beside the other resident models — \
                     making more room and loading it again",
                    crate::audit::sanitize_for_log(&normalized)
                );
                drop(new_engine);
                make_more_room = true;
                continue;
            }

            let size_vram = free_before
                .zip(crate::fit::vram_bytes().map(|(free, _)| free))
                .map(|(before, after)| before.saturating_sub(after));
            let kv_bytes = ready_info.as_ref().map_or(0, |info| {
                ((info.kv_k_mib + info.kv_v_mib) * 1024.0 * 1024.0) as u64
            });
            let projector_bytes = mmproj_path
                .as_deref()
                .and_then(|p| std::fs::metadata(p).ok())
                .map_or(0, |m| m.len());
            let facts = resident::LoadFacts {
                ctx_size: effective_ctx_size,
                batch_size,
                gpu_layers,
                n_layers: sizing.info.map(|info| info.n_layers),
                size_bytes: file_size
                    .saturating_add(projector_bytes)
                    .saturating_add(kv_bytes),
                size_vram,
                family: sizing.info.and_then(|info| info.architecture.clone()),
            };

            // ── 3. Install the new model among the residents ─────────
            // A sequential engine creates its context per request, so the
            // memory that context takes is free while it is idle; it is held
            // back from everything sized next to it instead (F5).
            let unallocated_reserve = match &new_engine {
                Some(engine) if gpu_layers != 0 => crate::fit::context_reserve_bytes(
                    sizing.info,
                    engine.context_size(),
                    sizing.kv_bpe_k,
                    sizing.kv_bpe_v,
                    load_n_ubatch,
                ),
                _ => 0,
            };
            let snapshot = {
                let mut models = self.models.write().await;
                let mut model = resident::LoadedModel::new(
                    normalized.clone(),
                    gguf_path,
                    new_engine,
                    new_scheduler,
                );
                model.unallocated_reserve = unallocated_reserve;
                model.facts = facts;
                let model = models.insert(model);
                self.lease(model, keep_alive)
            };
            drop(swap_guard);

            tracing::info!(
                "Model swap complete → {} (batch_size={batch_size})",
                crate::audit::sanitize_for_log(&normalized)
            );

            // The diagnostic banner `run` prints at startup. `serve` starts
            // with no model, so this is the only place it can be emitted — and
            // until it was here, anyone driving the engine as a daemon never
            // saw which backend actually initialised, how many layers were
            // offloaded, or what the KV cache costs. That is the audience
            // least able to guess and most likely to be filing a report. See
            // `crate::banner`.
            let info = ready_info.unwrap_or_default();
            crate::banner::ModelBanner {
                model_name: normalized
                    .strip_prefix("eullm/")
                    .unwrap_or(&normalized)
                    .to_string(),
                gpu_layers: self.gpu_layers,
                cpu_moe: self.cpu_moe,
                n_cpu_moe: self.n_cpu_moe,
                rs_seq: self.rs_seq,
                // Drafts on the scheduler's one slot only.
                mtp: if batch_size == 1 { self.mtp } else { 0 },
                mtp_p_min: self.mtp_p_min,
                mtp_model: self.mtp_model.clone(),
                moe_cache_bytes,
                no_mmap: load_no_mmap,
                moe_prefetch_slots,
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
                n_batch: load_n_batch,
                n_ubatch: load_n_ubatch,
                rust_debug: self.rust_debug,
            }
            .print();

            return Ok(snapshot);
        }
    }

    /// A resident's handles, with a lease on it for one request. Only with a
    /// guard on the residents held — see `resident::Usage::lease`.
    pub(crate) fn lease(
        &self,
        model: &resident::LoadedModel,
        keep_alive: KeepAlive,
    ) -> resident::SlotSnapshot {
        resident::SlotSnapshot {
            model_name: model.name.clone(),
            engine: model.engine.clone(),
            scheduler: model.scheduler.clone(),
            lease: model
                .usage
                .lease(keep_alive, self.default_keep_alive, &self.idle),
            load_duration: std::time::Duration::ZERO,
        }
    }

    /// The resident `requested` names — or, given `path`, the one loaded
    /// from that file under any name — with a lease on it, if it is loaded.
    async fn lease_loaded(
        &self,
        requested: &str,
        path: Option<&std::path::Path>,
        keep_alive: KeepAlive,
    ) -> Option<resident::SlotSnapshot> {
        let models = self.models.read().await;
        let model = match path {
            None => models.find(requested)?,
            Some(path) => models.find_file(path)?,
        };
        tracing::info!(
            "Model {} already loaded (as {})",
            crate::audit::sanitize_for_log(requested),
            crate::audit::sanitize_for_log(&model.name)
        );
        Some(self.lease(model, keep_alive))
    }

    /// Make room for `incoming` among the residents, then size its load: the
    /// steps `resident::next_step` gives, until it says load.
    ///
    /// Room is a place in the count (`--max-loaded-models`) and, under
    /// `--fit`, VRAM: a model that would not fit whole beside the residents
    /// makes one more of them go, until it does or is alone — when it is
    /// sized as any model loaded by itself, partial split included. A busy
    /// resident is unloaded mid-request only when one model is kept at a
    /// time, as a swap always did; with several, the load waits for one to go
    /// idle, up to `busy_wait`, with `swap_lock` released meanwhile, and
    /// gives up with `ModelError::Busy`. Once the count is satisfied, ad-hoc
    /// companions are evicted under `--fit` before sizing, as before.
    ///
    /// `fits_now` starts the planning: `Some(false)` makes one more resident
    /// go before anything is sized. Under `EvictPolicy::Never`, the first
    /// step that would unload a model, or wait for one to go idle so that it
    /// can, refuses the load instead.
    #[allow(clippy::too_many_arguments)]
    async fn make_room<'a>(
        &'a self,
        swap_guard: &mut Option<tokio::sync::MutexGuard<'a, ()>>,
        incoming: &str,
        path: &std::path::Path,
        sizing: &Sizing<'_>,
        mut fits_now: Option<bool>,
        keep_alive: KeepAlive,
        evict: EvictPolicy,
    ) -> Result<Room, ModelError> {
        let busy = if self.max_loaded_models <= 1 {
            resident::BusyPolicy::Abort
        } else {
            resident::BusyPolicy::Wait
        };
        let give_up_at = std::time::Instant::now() + self.busy_wait;
        let mut companions_evicted = false;
        loop {
            // Registered before the residents are read: a request that ends
            // in between still wakes the wait below.
            let mut went_idle = std::pin::pin!(self.idle.notified());
            went_idle.as_mut().enable();
            let views = self.models.read().await.views();
            let now = std::time::Instant::now();
            let step = resident::next_step(&views, self.max_loaded_models, busy, fits_now, now);
            if evict == EvictPolicy::Never && step != resident::Step::Load {
                let why = if fits_now == Some(false) {
                    "it would not fit whole beside the resident ones"
                } else {
                    "there is no room for it in --max-loaded-models"
                };
                return Err(ModelError::NoRoom(format!(
                    "Loading '{incoming}' would unload another model: {why}"
                )));
            }
            match step {
                resident::Step::Evict(i) => {
                    let view = views[i];
                    let removal = match busy {
                        resident::BusyPolicy::Abort => Removal::Always,
                        resident::BusyPolicy::Wait => Removal::IfIdle,
                    };
                    if let Some(name) = self.remove_generation(view.id, removal).await {
                        self.generation_evictions
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let why = if resident::due(&view.usage, now) {
                            "its keep_alive was over"
                        } else if fits_now == Some(false) {
                            "the new model would not fit whole beside it"
                        } else if self.max_loaded_models <= 1 {
                            "one generation model is kept at a time"
                        } else {
                            "the least recently used, with --max-loaded-models reached"
                        };
                        let state = if view.usage.in_flight > 0 {
                            format!("answering {} request(s)", view.usage.in_flight)
                        } else {
                            format!(
                                "idle for {} s",
                                now.duration_since(view.usage.last_used).as_secs()
                            )
                        };
                        tracing::info!(
                            "Unloaded {} ({state}; {why}) to make room for {}",
                            crate::audit::sanitize_for_log(&name),
                            crate::audit::sanitize_for_log(incoming)
                        );
                    }
                    fits_now = None;
                }
                resident::Step::WaitFor => {
                    let remaining = give_up_at.saturating_duration_since(now);
                    if remaining.is_zero() {
                        return Err(ModelError::Busy(format!(
                            "Loading '{incoming}' needs room that only a model still \
                             answering requests can give, and none finished within {} s. \
                             Retry shortly.",
                            self.busy_wait.as_secs()
                        )));
                    }
                    tracing::info!(
                        "{}: every model that could make room is answering a request — \
                         waiting up to {} s for one to finish",
                        crate::audit::sanitize_for_log(incoming),
                        remaining.as_secs()
                    );
                    // Embedding and decision loads are not held up meanwhile.
                    swap_guard.take();
                    let _ = tokio::time::timeout(remaining, went_idle).await;
                    *swap_guard = Some(self.swap_lock.lock().await);
                    // Loaded by another request in the meantime?
                    if let Some(snapshot) =
                        self.lease_loaded(incoming, Some(path), keep_alive).await
                    {
                        return Ok(Room::Loaded(snapshot));
                    }
                    fits_now = None;
                }
                resident::Step::Load => {
                    // An embedder left resident from an earlier ingestion run
                    // would otherwise shrink the free VRAM `--fit` measures
                    // below, sizing this load as if the card were smaller
                    // than it actually is once the embedder itself is later
                    // evicted by `ensure_embedding_model`. See
                    // `evict_embedding_if_present_for_generation_load`.
                    if !companions_evicted {
                        self.evict_embedding_if_present_for_generation_load().await;
                        self.evict_decision_if_present_for_generation_load().await;
                        companions_evicted = true;
                    }
                    if !self.fit {
                        return Ok(Room::Ready(None));
                    }
                    let plan = self.size_load(sizing).await;
                    // A second model gets what the first one left: whole, or
                    // not beside it — never a split it did not ask for.
                    if plan.sized() && !plan.full && !self.models.read().await.is_empty() {
                        tracing::info!(
                            "{} would not fit whole beside the resident models — \
                             making room",
                            crate::audit::sanitize_for_log(incoming)
                        );
                        fits_now = Some(false);
                        continue;
                    }
                    return Ok(Room::Ready(Some(plan)));
                }
            }
        }
    }

    /// Size a load against the VRAM free right now (see `fit::plan_offload`),
    /// less what the free figure does not show yet: the reserved companions'
    /// requests, and the context every sequential resident creates per
    /// request.
    async fn size_load(&self, sizing: &Sizing<'_>) -> crate::fit::OffloadPlan {
        // Counted below as reserved; the context the decision model keeps
        // between requests must not show up as used as well.
        self.release_decision_context().await;
        let reserve_bytes = self
            .reserved_embedding_bytes()
            .await
            .saturating_add(self.reserved_decision_bytes().await)
            .saturating_add(self.models.read().await.unallocated_reserve())
            // A micro-batch above the default needs a larger compute buffer
            // than the flat reserve the fit charges (`--n-ubatch`).
            .saturating_add(crate::fit::ubatch_reserve_bytes(
                self.n_ubatch.unwrap_or(crate::inference::DEFAULT_N_UBATCH),
            ))
            .saturating_add(sizing.mtp_reserve);
        crate::fit::plan_offload(
            crate::fit::vram_bytes(),
            sizing.info,
            sizing.layout,
            sizing.file_size,
            sizing.ctx_size,
            sizing.kv_bpe_k,
            sizing.kv_bpe_v,
            reserve_bytes,
            sizing.mmproj_bytes,
            sizing.flags,
        )
    }

    /// Take the generation model `id` out of the residents — if `when` still
    /// holds for it under their write guard, where no lease can be taken, so
    /// that a request which took one in the meantime keeps its model — and
    /// free its memory. Returns its name. Call with `swap_lock` held.
    async fn remove_generation(&self, id: u64, when: Removal) -> Option<String> {
        let model = {
            let mut models = self.models.write().await;
            let usage = models.get(id)?.usage.view();
            let allowed = match when {
                Removal::Always => true,
                Removal::IfIdle => usage.in_flight == 0,
                Removal::IfDue => resident::due(&usage, std::time::Instant::now()),
            };
            if !allowed {
                return None;
            }
            models.remove(id)?
        };
        let name = model.name.clone();
        retire(model).await;
        Some(name)
    }

    /// Unload every generation model, freeing its VRAM, without loading a
    /// replacement — a later request with a `model` field (or another
    /// `eullm run`) loads one again. Requests still running on one are cut
    /// off: an explicit unload means now.
    ///
    /// The primary use case is freeing VRAM for a co-resident process (e.g.
    /// an embedding model used during RAG document ingestion) without
    /// restarting the eullm server. Serialized against loads via the same
    /// lock, so an unload can't race a concurrent load.
    ///
    /// Returns the names of the models unloaded, none if nothing was loaded
    /// (a no-op, not an error).
    pub(crate) async fn unload_all(&self) -> Vec<String> {
        let _swap_guard = self.swap_lock.lock().await;
        let unloaded = self.unload_generation_models().await;
        if !unloaded.is_empty() {
            tracing::info!("Generation models unloaded — none resident");
        }
        unloaded
    }

    /// Unload the generation model `requested` names, now — requests still
    /// running on it are cut off, as with `unload_all` — and leave every
    /// other resident alone. `None` when it was not loaded, under that name or
    /// as the same file under another.
    pub(crate) async fn unload_named(&self, requested: &str) -> Option<String> {
        let _swap_guard = self.swap_lock.lock().await;
        let id = self.resident_id(requested).await.ok().flatten()?;
        let name = self.remove_generation(id, Removal::Always).await?;
        tracing::info!(
            "Generation model {} unloaded",
            crate::audit::sanitize_for_log(&name)
        );
        Some(name)
    }

    /// `keep_alive: 0` on a request that asks for nothing else — Ollama's way
    /// to unload one model: the model the request names, or the most recently
    /// used one when it names none, goes without being loaded first. Idle, it
    /// is unloaded before this returns; answering other requests, it goes
    /// when they are over, as `keep_alive: 0` on any request does. Every other
    /// resident stays.
    ///
    /// Returns its name, `None` when it was not loaded; a name that is no
    /// model at all is `ModelError::NotFound`.
    pub(crate) async fn expire_model(
        &self,
        requested: Option<&str>,
    ) -> Result<Option<String>, ModelError> {
        let id = match requested {
            Some(name) => self.resident_id(name).await?,
            None => self.models.read().await.most_recently_used().map(|m| m.id),
        };
        let Some(id) = id else {
            return Ok(None);
        };
        let name = {
            let models = self.models.read().await;
            let Some(model) = models.get(id) else {
                return Ok(None);
            };
            // As a request that asked for keep_alive 0 and is now over.
            drop(
                model
                    .usage
                    .lease(KeepAlive::Immediate, self.default_keep_alive, &self.idle),
            );
            model.name.clone()
        };
        let _swap_guard = self.swap_lock.lock().await;
        if self.remove_generation(id, Removal::IfDue).await.is_some() {
            tracing::info!(
                "keep_alive 0 — unloaded {}",
                crate::audit::sanitize_for_log(&name)
            );
        }
        Ok(Some(name))
    }

    /// Per routing candidate, in the table's order, the context one request
    /// to it gets and whether it is loaded: a resident's, from how it was
    /// loaded; another's, from how it would load — with the request's own
    /// `batch_size` and `ctx_size` when it gives them, and on the sequential
    /// engine, with the whole context, when it has a projector.
    pub(crate) async fn route_candidates(
        &self,
        table: &route::RouteTable,
        (override_batch_size, override_ctx_size): (Option<usize>, Option<u32>),
    ) -> (Vec<u32>, Vec<bool>) {
        let models = self.models.read().await;
        table
            .candidates
            .iter()
            .map(|candidate| {
                let resident = models
                    .find(&candidate.name)
                    .or_else(|| models.find_file(&candidate.path));
                match resident {
                    Some(model) => (
                        model.facts.ctx_size / model.facts.batch_size.max(1) as u32,
                        true,
                    ),
                    None => {
                        let ctx = override_ctx_size.unwrap_or(self.ctx_size);
                        let slots = if candidate.has_projector {
                            1
                        } else {
                            override_batch_size.unwrap_or(self.batch_size).max(1)
                        };
                        (ctx / slots as u32, false)
                    }
                }
            })
            .unzip()
    }

    /// Load the routing candidates that fit without unloading anything, so
    /// that the first routed requests do not pay for a load: the fallback
    /// first — it answers whenever the decision model does not decide — then
    /// the others in the table's order, until one would need another model
    /// to go (`EvictPolicy::Never`). A candidate that fails to load is
    /// logged and passed over. `keep_alive` applies to each from when it is
    /// loaded. Returns the candidates resident at the end, in that order.
    pub(crate) async fn warm_route_candidates(
        &self,
        table: &route::RouteTable,
        keep_alive: KeepAlive,
    ) -> Vec<String> {
        let order = std::iter::once(table.fallback)
            .chain((0..table.candidates.len()).filter(|&i| i != table.fallback));
        let mut warm = Vec::new();
        for i in order {
            let name = &table.candidates[i].name;
            let loaded = self
                .load_generation_model(name, None, None, keep_alive, EvictPolicy::Never)
                .await;
            match loaded {
                // The lease ends here, which starts its keep_alive.
                Ok(_) => warm.push(name.clone()),
                Err(ModelError::NoRoom(why)) => {
                    tracing::info!("Auto routing warm-up stops: {why}");
                    break;
                }
                Err(e) => tracing::warn!(
                    "Auto routing warm-up: {} did not load: {e}",
                    crate::audit::sanitize_for_log(name)
                ),
            }
        }
        warm
    }

    /// The resident `requested` names, by name or else as its file, or `None`
    /// when it is not loaded. A name that is no model at all is
    /// `ModelError::NotFound`; one the residents answer to is never looked up
    /// on disk.
    async fn resident_id(&self, requested: &str) -> Result<Option<u64>, ModelError> {
        if let Some(model) = self.models.read().await.find(requested) {
            return Ok(Some(model.id));
        }
        let path = self.resolve_model(&normalize_model_name(requested))?;
        Ok(self.models.read().await.find_file(&path).map(|m| m.id))
    }

    /// `unload_all`'s work: take every generation model out and wait for each
    /// one's scheduler thread to fully exit, so their VRAM is guaranteed
    /// freed by the time this resolves — the caller needs the VRAM actually
    /// free before handing it to another process. Call with `swap_lock` held.
    async fn unload_generation_models(&self) -> Vec<String> {
        let ids: Vec<u64> = self
            .models
            .read()
            .await
            .views()
            .iter()
            .map(|r| r.id)
            .collect();
        let mut unloaded = Vec::new();
        for id in ids {
            unloaded.extend(self.remove_generation(id, Removal::Always).await);
        }
        unloaded
    }

    /// Unload generation models until a companion `companion` — the
    /// `kind` (embedding or decision) model about to load, needing `need`
    /// bytes of VRAM and `compute_reserve` beside them — fits in what is free,
    /// or none is left: the least recently used and idle ones first, and no
    /// more than it takes (`resident::companion_evictions`). Each one unloaded
    /// counts in `cross_slot_evictions`. Without a free-VRAM figure nothing is
    /// unloaded, and the companion loads beside them as it always did. Call
    /// with `swap_lock` held.
    ///
    /// What each resident gives back is what its load measured, or an
    /// estimate; the free figure is read again after each round, and another
    /// round follows while the companion still does not fit.
    async fn make_room_for_companion(
        &self,
        companion: &str,
        kind: &str,
        need: u64,
        compute_reserve: u64,
    ) {
        loop {
            let (views, unallocated) = {
                let models = self.models.read().await;
                (models.views(), models.unallocated_reserve())
            };
            let usable = crate::fit::vram_bytes()
                .map(|card| usable_vram(card, self.vram_reserve, compute_reserve, unallocated));
            let now = std::time::Instant::now();
            let plan = resident::companion_evictions(&views, usable, need, now);
            let mut unloaded = 0;
            for id in plan {
                let Some(view) = views.iter().find(|v| v.id == id) else {
                    continue;
                };
                let Some(name) = self.remove_generation(id, Removal::Always).await else {
                    continue;
                };
                unloaded += 1;
                self.cross_slot_evictions
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let state = if view.usage.in_flight > 0 {
                    format!("answering {} request(s)", view.usage.in_flight)
                } else {
                    format!(
                        "idle for {} s",
                        now.duration_since(view.usage.last_used).as_secs()
                    )
                };
                tracing::info!(
                    "Unloaded {} ({state}) to make room for the {kind} model {}, which does \
                     not fit beside it; it reloads on its next request",
                    crate::audit::sanitize_for_log(&name),
                    crate::audit::sanitize_for_log(companion)
                );
            }
            if unloaded == 0 {
                return;
            }
        }
    }

    /// Unload every generation model that is due (`resident::due`): no
    /// request is using it, and the keep_alive that applies — the last
    /// request's to arrive — is 0, or ran out. Each is checked
    /// again under the residents' write guard, where no lease can be taken,
    /// so a request that arrived in the meantime keeps its model.
    async fn unload_due_generation_models(&self) {
        let now = std::time::Instant::now();
        let due: Vec<resident::ResidentView> = self
            .models
            .read()
            .await
            .views()
            .into_iter()
            .filter(|r| resident::due(&r.usage, now))
            .collect();
        for model in due {
            let _swap_guard = self.swap_lock.lock().await;
            let Some(name) = self.remove_generation(model.id, Removal::IfDue).await else {
                continue;
            };
            let name = crate::audit::sanitize_for_log(&name);
            if model.usage.unload_when_idle {
                tracing::info!("keep_alive 0 — unloading {name} now that its request is over");
            } else {
                tracing::info!("keep_alive expired — unloading idle generation model {name}");
            }
        }
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
    /// 2. Not loaded, and it fits in free VRAM beside the generation models
    ///    → load it into the embedding slot; they are untouched.
    /// 3. Not loaded, and it does not fit → unload generation models first,
    ///    the least recently used and idle ones before the others, until it
    ///    fits or none is left (`make_room_for_companion`; a generation
    ///    request reloads one later, and `resolve_model` and the embedded
    ///    chat UI work unchanged without it), then load the embedder.
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
        let wanted = model_identity_key(&normalized);
        {
            // By the name it was loaded under, or by its file's own: the
            // name an `--embedding-model` given as a store name was loaded
            // under until it kept the one it was given.
            let slot = self.embedding.read().await;
            if let Some(ref loaded) = *slot
                && (model_identity_key(&loaded.model_name) == wanted
                    || model_identity_key(&loaded.model.path().to_string_lossy()) == wanted)
            {
                return Ok(loaded.model.clone());
            }
        }

        let gguf_path = self.resolve_model(&normalized)?;
        // The loaded model asked for under another of its names — a store
        // name for one launched by its path, or another name `eullm pull`
        // linked to the same file: loading it again would hold it twice
        // while both are in use, and put a model no longer reserved in
        // place of a reserved companion.
        {
            let slot = self.embedding.read().await;
            if let Some(ref loaded) = *slot
                && same_file(loaded.model.path(), &gguf_path)
            {
                return Ok(loaded.model.clone());
            }
        }
        let weights_bytes = std::fs::metadata(&gguf_path).map(|m| m.len()).unwrap_or(0);
        self.make_room_for_companion(
            &normalized,
            "embedding",
            weights_bytes,
            crate::fit::EMBEDDING_COMPUTE_RESERVE_BYTES,
        )
        .await;

        tracing::info!(
            "Loading embedding model {} ({})",
            crate::audit::sanitize_for_log(&normalized),
            gguf_path.display()
        );
        let threads = self.threads;
        let backend_for_load = self.backend.clone();
        let model = tokio::task::spawn_blocking(move || {
            EmbeddingModel::load(&gguf_path, threads, n_ctx, backend_for_load)
        })
        .await
            .map_err(|e| ModelError::LoadFailed(format!("Task join error: {e}")))?
            .map_err(|e| ModelError::LoadFailed(format!("Failed to load embedding model: {e}")))?;
        let model = Arc::new(model);

        {
            let mut slot = self.embedding.write().await;
            *slot = Some(EmbeddingSlot {
                model_name: normalized.clone(),
                model: model.clone(),
                // Never reserved: a model loaded ad hoc by a runtime request
                // has no launch-time guarantee behind it. Only
                // `--embedding-model` produces a reserved companion — see
                // `EmbeddingSlot::is_reserved_companion`.
                is_reserved_companion: false,
            });
        }
        tracing::info!(
            "Embedding model ready → {}",
            crate::audit::sanitize_for_log(&normalized)
        );
        Ok(model)
    }

    /// The mirror of the eviction inside `ensure_embedding_model`: called
    /// from `load_generation_model` before sizing a generation load, so an embedder
    /// left resident from a prior ingestion run does not silently shrink
    /// the VRAM budget `--fit` sizes against. Cheap when nothing is loaded
    /// (`RwLock::read` + an `Option` check) and a no-op unless `--fit` is
    /// on, since only `--fit` reads free VRAM to make a sizing decision in
    /// the first place — without it, evicting the embedder would trade a
    /// real, working configuration for a guess.
    ///
    /// Skips a **reserved companion** (`EmbeddingSlot::is_reserved_companion`
    /// — populated via `--embedding-model` at launch): that is exactly the
    /// case `--embedding-model` exists to guarantee against — the whole
    /// point of reserving its footprint up front is that it survives a
    /// later chat-model swap instead of being kicked out for one. Its
    /// footprint is subtracted from free VRAM by the caller
    /// (`load_generation_model`'s `reserve_bytes`) instead of it being evicted.
    async fn evict_embedding_if_present_for_generation_load(&self) {
        if !self.fit {
            return;
        }
        let is_reserved = self
            .embedding
            .read()
            .await
            .as_ref()
            .is_some_and(|s| s.is_reserved_companion);
        if is_reserved {
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

    /// The VRAM footprint to protect for a **reserved companion** (see
    /// `EmbeddingSlot::is_reserved_companion`) before sizing a generation
    /// load — `0` when no reserved companion is resident, which is every
    /// server that never used `--embedding-model` and every ad hoc
    /// `/api/embed`-loaded embedder (those are evicted outright instead,
    /// see `evict_embedding_if_present_for_generation_load`).
    ///
    /// Deliberately **not** `weights_bytes + EMBEDDING_COMPUTE_RESERVE_BYTES`:
    /// a reserved companion is always already loaded by the time this runs
    /// (eagerly, at launch — see `main.rs`), so its weights already show up
    /// as used memory in the free-VRAM figure `--fit` reads. Adding
    /// `weights_bytes` again would subtract its footprint twice and
    /// under-offload the generation model for no reason. Its context too:
    /// built at launch for its longest input and kept for every request
    /// (`EmbeddingModel::reserve_context`), it is used memory already. Only
    /// the margin is genuinely not yet reflected — what a decode allocates
    /// beside the context, in the GPU backend's scratch pool — so that is
    /// the only part worth protecting ahead of time.
    async fn reserved_embedding_bytes(&self) -> u64 {
        self.embedding
            .read()
            .await
            .as_ref()
            .filter(|s| s.is_reserved_companion)
            .map(|_| crate::fit::EMBEDDING_COMPUTE_RESERVE_BYTES)
            .unwrap_or(0)
    }

    /// Ensure the named decision model is loaded into the decision slot and
    /// return it. The same residency rules as `ensure_embedding_model` —
    /// already loaded: reuse it; fits next to the generation models: load it
    /// alongside; does not: unload them, least recently used first, until it
    /// does — with the VRAM a request's context needs
    /// (`fit::decision_reserve_bytes`) counted in, not only the weights.
    ///
    /// A different decision model already in the slot is dropped before the
    /// new one loads, so the free-VRAM check sees the room it leaves. A
    /// request still running on it keeps it alive until it finishes.
    pub async fn ensure_decision_model(
        &self,
        name: &str,
    ) -> Result<Arc<DecisionModel>, ModelError> {
        let _swap_guard = self.swap_lock.lock().await;

        let normalized = normalize_model_name(name);
        {
            let slot = self.decision.read().await;
            if let Some(ref loaded) = *slot
                && model_identity_key(&loaded.model_name) == model_identity_key(&normalized)
            {
                return Ok(loaded.model.clone());
            }
        }

        let gguf_path = self.resolve_model(&normalized)?;
        *self.decision.write().await = None;
        let weights_bytes = std::fs::metadata(&gguf_path).map(|m| m.len()).unwrap_or(0);
        let reserve_bytes = crate::fit::decision_reserve_bytes(&gguf_path, self.decision_ctx);
        self.make_room_for_companion(
            &normalized,
            "decision",
            weights_bytes.saturating_add(reserve_bytes),
            0,
        )
        .await;

        tracing::info!(
            "Loading decision model {} ({})",
            crate::audit::sanitize_for_log(&normalized),
            gguf_path.display()
        );
        let threads = self.threads;
        let max_ctx = self.decision_ctx;
        let flash_attn = self.flash_attn;
        let backend_for_load = self.backend.clone();
        let model = tokio::task::spawn_blocking(move || {
            DecisionModel::load(&gguf_path, threads, max_ctx, flash_attn, backend_for_load)
        })
        .await
        .map_err(|e| ModelError::LoadFailed(format!("Task join error: {e}")))?
        .map_err(|e| ModelError::LoadFailed(format!("Failed to load decision model: {e}")))?;
        let model = Arc::new(model);

        *self.decision.write().await = Some(DecisionSlot {
            model_name: normalized.clone(),
            model: model.clone(),
            // Loaded by a request, so no launch-time guarantee behind it —
            // see `EmbeddingSlot::is_reserved_companion`.
            is_reserved_companion: false,
            reserve_bytes,
        });
        tracing::info!(
            "Decision model ready → {}",
            crate::audit::sanitize_for_log(&normalized)
        );
        Ok(model)
    }

    /// The decision slot's counterpart of
    /// `evict_embedding_if_present_for_generation_load`, with the same
    /// conditions: only under `--fit`, never a reserved companion.
    async fn evict_decision_if_present_for_generation_load(&self) {
        if !self.fit {
            return;
        }
        let evictable = self
            .decision
            .read()
            .await
            .as_ref()
            .is_some_and(|s| !s.is_reserved_companion);
        if !evictable {
            return;
        }
        tracing::info!(
            "Generation request — evicting the resident decision model to free VRAM for sizing \
             (reload it with a later /v1/systemone request)"
        );
        *self.decision.write().await = None;
        self.cross_slot_evictions
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Free the context the loaded decision model keeps between requests
    /// (`DecisionModel::release_context`), so a free-VRAM figure measured
    /// next shows its weights as used and its context as free — the context
    /// being what `reserved_decision_bytes` counts. The next decision
    /// request creates it again. Waits for a request in progress to finish.
    async fn release_decision_context(&self) {
        let model = self
            .decision
            .read()
            .await
            .as_ref()
            .map(|slot| Arc::clone(&slot.model));
        if let Some(model) = model {
            let _ = tokio::task::spawn_blocking(move || model.release_context()).await;
        }
    }

    /// The decision slot's counterpart of `reserved_embedding_bytes`: a
    /// reserved companion's per-request context, which the free-VRAM figure
    /// does not show once `release_decision_context` has run. Its weights
    /// do, so they are not counted again.
    async fn reserved_decision_bytes(&self) -> u64 {
        self.decision
            .read()
            .await
            .as_ref()
            .filter(|s| s.is_reserved_companion)
            .map(|s| s.reserve_bytes)
            .unwrap_or(0)
    }

    /// Reset the embedding slot's idle-unload deadline. Called on every
    /// request that uses the slot, so an active ingestion run is never
    /// unloaded out from under it. `Immediate` empties the slot right away:
    /// the request holds its own `Arc` to the model and finishes on it. See
    /// `KeepAlive`.
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

    /// Same as `touch_embedding_slot`, for the decision slot.
    pub async fn touch_decision_slot(&self, keep_alive: KeepAlive) {
        touch_deadline(&self.decision_deadline, keep_alive, self.default_keep_alive);
        if keep_alive == KeepAlive::Immediate {
            *self.decision.write().await = None;
        }
    }

    /// Background loop spawned once from `serve()`: unload whatever is due.
    /// It looks every `idle_tick` (30 s), which is coarse on purpose — this
    /// is a power-saving idle timer, not a latency-sensitive path, and
    /// checking on every request would mean taking the slot locks on every
    /// single call for a comparison that is false almost all the time — and
    /// also whenever the generation model's last request finishes, so that
    /// `keep_alive: 0` takes effect when the request is over rather than up
    /// to 30 s later.
    async fn run_idle_unload_loop(self: Arc<Self>) {
        loop {
            // Registered before anything is checked: a request that ends
            // while this pass runs wakes the next pass instead of being
            // missed until the tick after.
            let mut went_idle = std::pin::pin!(self.idle.notified());
            went_idle.as_mut().enable();

            self.unload_due_generation_models().await;

            let now = tokio::time::Instant::now();
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

            let decision_expired = {
                let mut deadline = self.decision_deadline.lock().await;
                let expired = deadline.is_some_and(|d| now >= d);
                if expired {
                    *deadline = None;
                }
                expired
            };
            if decision_expired {
                tracing::info!("keep_alive expired — unloading idle decision model");
                *self.decision.write().await = None;
            }

            tokio::select! {
                () = went_idle.as_mut() => {}
                () = tokio::time::sleep(self.idle_tick) => {}
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
    /// refusing our own answer would break `eullm run ./model.gguf`. So is
    /// `--default-model`'s and every `--auto-model`'s: they were named on
    /// the command line, not in a request.
    fn resolve_model(&self, name: &str) -> Result<PathBuf, ModelError> {
        let path = PathBuf::from(name);

        // 0. The model this process was launched with, or one that
        //    `--default-model` or `--auto-model` names, by the name the API
        //    uses for it or by
        //    its literal path. Exact match on either — never a stem or prefix
        //    comparison, which would turn this allowance into a way to reach
        //    any similarly named file.
        let named = self
            .launch_model
            .iter()
            .map(|(name, path)| (name.as_str(), path))
            .chain(
                self.default_model
                    .iter()
                    .map(|m| (m.name.as_str(), &m.path)),
            )
            .chain(
                self.router
                    .iter()
                    .flat_map(|table| &table.candidates)
                    .map(|c| (c.name.as_str(), &c.path)),
            );
        for (named, named_path) in named {
            if (name == named || &path == named_path) && named_path.is_file() {
                return Ok(named_path.clone());
            }
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
        if let Some(p) = self.store.resolve_gguf(name) {
            return Ok(p);
        }

        // 5. Try normalized name (Ollama tag format).
        let normalized = normalize_model_name(name);
        if normalized != name
            && let Some(p) = self.store.resolve_gguf(&normalized)
        {
            return Ok(p);
        }

        Err(ModelError::NotFound(format!(
            "Model '{name}' not found. A request can name:\n  \
             - a model on this machine, by the name `eullm list` gives it (a catalog model is \
             downloaded first: `eullm pull NAME`, or the chat UI's Model hub)\n  \
             - NAME for /models/NAME.gguf or /data/models/NAME.gguf\n  \
             - a GGUF file, or a directory holding one, by its path: only with \
             EULLM_ALLOW_MODEL_PATHS=1 set where the server starts"
        )))
    }
}

/// When `AppState::remove_generation` may take a model out.
#[derive(Debug, Clone, Copy)]
enum Removal {
    /// Whatever it is doing: requests still running on it are cut off.
    Always,
    /// Only if no request is running on it.
    IfIdle,
    /// Only if it is still due — idle, and its keep_alive over.
    IfDue,
}

/// What `AppState::make_room` ends with.
enum Room {
    /// Another request loaded the model while this one waited for room.
    Loaded(resident::SlotSnapshot),
    /// There is room, and this is the load's plan (`None` without `--fit`).
    Ready(Option<crate::fit::OffloadPlan>),
}

/// What sizing a load reads about the model: once per load, however many
/// times it is sized while room is made for it.
struct Sizing<'a> {
    info: Option<&'a crate::fit::GgufInfo>,
    layout: Option<&'a crate::fit::MoeLayout>,
    file_size: u64,
    ctx_size: u32,
    kv_bpe_k: f64,
    kv_bpe_v: f64,
    mmproj_bytes: u64,
    flags: crate::fit::OffloadFlags,
    /// What `--mtp`'s draft context will take once the model has loaded
    /// (`fit::mtp_reserve_bytes`); `0` when the load will draft nothing.
    mtp_reserve: u64,
}

/// Free a generation model taken out of the residents, so that its memory
/// is free when this returns — the next load is sized against what is free.
///
/// A scheduler is stopped, and its thread joined on a blocking thread: the
/// join waits for the decode loop to notice. A sequential engine has no
/// thread to join; it is freed when the last `Arc` to it goes, and every
/// request running on it holds one. Those are waited for (F4): taking it out
/// of the residents used to be all, so the next load measured free VRAM with
/// the old weights still in it.
async fn retire(model: resident::LoadedModel) {
    if let Some(handle) = model.scheduler
        && let Err(e) = tokio::task::spawn_blocking(move || handle.shutdown()).await
    {
        tracing::warn!("Failed to join scheduler thread: {e}");
    }
    if let Some(engine) = model.engine {
        let released = resident::wait_for_release(
            &engine,
            &model.usage,
            ENGINE_RELEASE_LIMIT,
            ENGINE_RELEASE_GRACE,
        )
        .await;
        if !released {
            tracing::warn!(
                "{} is still held elsewhere — the terminal chat, or a request \
                 whose client left before it finished; its memory is freed when \
                 that ends",
                crate::audit::sanitize_for_log(&model.name)
            );
        }
    }
}

/// The longest a sequential engine's in-flight requests are waited for
/// before its memory is given up on (see `retire`): a load waits behind it.
const ENGINE_RELEASE_LIMIT: std::time::Duration = std::time::Duration::from_secs(30);

/// How long a sequential engine still held after its last request ended is
/// waited for: the moment the blocking thread that ran the request takes to
/// let go. A holder that outlasts it is not a request — `eullm run`'s
/// terminal chat holds the launch model for as long as it runs.
const ENGINE_RELEASE_GRACE: std::time::Duration = std::time::Duration::from_secs(2);

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

/// Whether two paths name the same file: through the symlinks a model store
/// is often reached by (`~/.eullm/models` linked to a data disk), and, where
/// the filesystem says so, through the hard links `eullm pull` gives a model
/// pulled under a second name.
fn same_file(a: &std::path::Path, b: &std::path::Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if let (Ok(a), Ok(b)) = (std::fs::metadata(a), std::fs::metadata(b)) {
            return a.dev() == b.dev() && a.ino() == b.ino();
        }
    }
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}

#[cfg(test)]
mod residency_config_tests {
    use super::*;

    #[test]
    fn ollamas_variable_gets_a_hint_and_is_never_read() {
        assert_eq!(ollama_env_hint(None, true), None);
        assert_eq!(ollama_env_hint(Some("  "), true), None);
        let hint = ollama_env_hint(Some("3"), true).expect("a hint");
        assert!(
            hint.contains("OLLAMA_MAX_LOADED_MODELS=3") && hint.contains("--max-loaded-models")
        );
        let ignored = ollama_env_hint(Some("3"), false).expect("a hint");
        assert!(ignored.contains("ignored"), "{ignored}");
    }

    fn flags(max_loaded_models: usize, default_model: Option<&str>) -> ResidencyFlags {
        ResidencyFlags {
            max_loaded_models,
            default_model: default_model.map(str::to_string),
            auto_models: Vec::new(),
            auto_timeout_ms: route::DEFAULT_AUTO_TIMEOUT_MS,
        }
    }

    /// A store of `qwen3-4b` and `qwen3-8b`, and nothing else.
    fn lookup(name: &str) -> Option<CandidateFacts> {
        matches!(name, "qwen3-4b" | "qwen3-8b").then(|| CandidateFacts {
            model: NamedModel {
                name: name.to_string(),
                path: PathBuf::from(format!("/store/{name}/model.gguf")),
            },
            store_description: None,
            catalog: None,
            has_projector: false,
        })
    }

    #[test]
    fn the_residency_config_holds_at_least_one_model() {
        let resolve = |n| ResidencyConfig::resolve(&flags(n, None), |_| None).expect("no default");
        assert_eq!(resolve(0).max_loaded_models, 1);
        assert_eq!(resolve(4).max_loaded_models, 4);
        assert_eq!(resolve(4).default_model, None);
        assert_eq!(resolve(4).router, None);
    }

    /// `--default-model` is resolved at startup: a model the server cannot
    /// find stops it there, rather than failing every request naming none.
    #[test]
    fn the_default_model_is_resolved_at_startup() {
        let config =
            ResidencyConfig::resolve(&flags(2, Some(" qwen3-8b ")), lookup).expect("found");
        assert_eq!(
            config.default_model.map(|m| m.name).as_deref(),
            Some("qwen3-8b")
        );
        let refused = ResidencyConfig::resolve(&flags(2, Some("qwen3-80b")), lookup).unwrap_err();
        assert!(
            refused.contains("--default-model 'qwen3-80b'") && refused.contains("eullm pull"),
            "{refused}"
        );
    }

    /// `--auto-model` is resolved with the rest, its fallback the default
    /// model when that is one of its candidates.
    #[test]
    fn the_route_table_is_resolved_with_the_default_model() {
        let mut flags = flags(2, Some("qwen3-4b"));
        flags.auto_models = vec!["qwen3-4b=Small".into(), "qwen3-8b=Large".into()];
        flags.auto_timeout_ms = 250;
        let router = ResidencyConfig::resolve(&flags, lookup)
            .expect("resolved")
            .router
            .expect("a table");
        assert_eq!(router.fallback().name, "qwen3-4b");
        assert_eq!(router.timeout, std::time::Duration::from_millis(250));
        flags.auto_models.push("qwen3-80b".into());
        let refused = ResidencyConfig::resolve(&flags, lookup).unwrap_err();
        assert!(refused.contains("'qwen3-80b' is not a model"), "{refused}");
    }

    /// `--default-model auto` routes the requests that name no model, which
    /// takes models to route between; the fallback is then the last one.
    #[test]
    fn default_model_auto_routes_and_needs_auto_models() {
        let mut flags = flags(2, Some("Auto"));
        let refused = ResidencyConfig::resolve(&flags, lookup).unwrap_err();
        assert!(refused.contains("--auto-model"), "{refused}");
        flags.auto_models = vec!["qwen3-4b=Small".into(), "qwen3-8b=Large".into()];
        let config = ResidencyConfig::resolve(&flags, lookup).expect("resolved");
        assert!(config.default_auto);
        assert_eq!(config.default_model, None);
        assert_eq!(config.router.expect("a table").fallback().name, "qwen3-8b");
    }
}

#[cfg(test)]
mod fits_tests {
    use super::usable_vram;

    const GIB: u64 = 1024 * 1024 * 1024;
    const MIB: u64 = 1024 * 1024;

    /// A sequential generation model creates its context per request, so
    /// while it is idle that memory looks free. An embedder sized into it
    /// left the next image request without room for its context (F5).
    #[test]
    fn a_sequential_resident_reserves_its_context_for_a_companion() {
        // 16 GiB card, 8 GiB free; the 1 GiB reserve keeps that much back
        // (12% of the card, 1.9 GiB, before 0.7.80).
        let card = (8 * GIB, 16 * GIB);
        let reserve = GIB;
        let embedder = 2 * GIB;
        assert!(embedder <= usable_vram(card, reserve, 256 * MIB, 0));
        // 4 GiB before 0.7.80; the larger usable memory needs a larger
        // context to leave the embedder without room.
        let vision_model_context = 6 * GIB;
        assert!(embedder > usable_vram(card, reserve, 256 * MIB, vision_model_context));
        // Saturating, not wrapping, when the reservations exceed what is free.
        assert_eq!(usable_vram(card, reserve, 256 * MIB, 64 * GIB), 0);
    }
}

#[cfg(all(test, unix))]
mod same_file_tests {
    use super::same_file;

    #[test]
    fn a_file_is_the_same_through_symlinks_and_hard_links() {
        let root = std::env::temp_dir().join(format!("eullm-same-file-{}", uuid::Uuid::new_v4()));
        let disk = root.join("disk");
        std::fs::create_dir_all(disk.join("emb")).unwrap();
        std::fs::write(disk.join("emb/Model-Q8_0.gguf"), b"GGUF").unwrap();
        std::fs::write(disk.join("emb/Other-Q8_0.gguf"), b"GGUF").unwrap();
        std::os::unix::fs::symlink(&disk, root.join("models")).unwrap();

        let direct = disk.join("emb/Model-Q8_0.gguf");
        std::fs::create_dir_all(disk.join("alias")).unwrap();
        std::fs::hard_link(&direct, disk.join("alias/Model-Q8_0.gguf")).unwrap();
        assert!(same_file(&direct, &root.join("models/emb/Model-Q8_0.gguf")));
        assert!(same_file(&direct, &disk.join("emb/../emb/Model-Q8_0.gguf")));
        assert!(same_file(&direct, &root.join("models/alias/Model-Q8_0.gguf")));
        assert!(!same_file(&direct, &disk.join("emb/Other-Q8_0.gguf")));
        assert!(!same_file(&direct, &root.join("missing.gguf")));
        std::fs::remove_dir_all(&root).unwrap();
    }
}

/// What a card with `(free, total)` VRAM leaves for a companion model — an
/// embedder or a decision model — beside what is loaded: free VRAM, less the
/// reserve `fit.rs` keeps for every load (`vram_reserve`, `--vram-reserve`), less
/// `compute_reserve_bytes` for the companion's own compute buffer —
/// `fit::EMBEDDING_COMPUTE_RESERVE_BYTES` for an embedder, 256 MiB rather
/// than `fit.rs`'s 640 MiB, since an embedding model's context and
/// micro-batch are both a fraction of an LLM's; a decision model passes 0 and
/// counts its whole per-request context in what it needs instead
/// (`fit::decision_reserve_bytes`) — and less `unallocated_bytes`, memory the
/// free figure shows but is already spoken for: the contexts sequential
/// residents create per request (F5).
///
/// Deliberately not the layer-by-layer machinery in `fit.rs`: an embedding
/// model loads fully onto the GPU or not at all (see `EmbeddingModel::load`),
/// so a companion only ever needs a yes/no answer, never a partial split.
fn usable_vram(
    (free, _total): (u64, u64),
    vram_reserve: u64,
    compute_reserve_bytes: u64,
    unallocated_bytes: u64,
) -> u64 {
    free.saturating_sub(vram_reserve)
        .saturating_sub(compute_reserve_bytes)
        .saturating_sub(unallocated_bytes)
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

impl KeepAlive {
    /// `Default` replaced by what it stands for: the server's `--keep-alive`
    /// (`default`), or `Forever` when none was given. Every other value is
    /// the request's own and is returned as it is.
    pub(crate) fn resolve(self, default: Option<std::time::Duration>) -> KeepAlive {
        match self {
            KeepAlive::Default => default.map_or(KeepAlive::Forever, KeepAlive::For),
            other => other,
        }
    }
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
        // `from_secs_f64` panics on NaN, infinity, and magnitudes past what
        // a Duration holds — all reachable from a request body so go
        // through the standard library's checked constructor and take the
        // malformed-value fallback for whatever it refuses.
        Some(s) => match std::time::Duration::try_from_secs_f64(s) {
            Ok(d) => KeepAlive::For(d),
            Err(_) => KeepAlive::Default,
        },
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
        Some(secs) if secs > 0.0 => match std::time::Duration::try_from_secs_f64(secs) {
            Ok(d) => Ok(d),
            Err(_) => Err(format!(
                "--keep-alive must be a positive duration, got '{s}' \
                 (the value is too large to represent as a duration)"
            )),
        },
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
    if s.is_empty() {
        // An empty (or whitespace-only) string has no number to split off:
        // without this, the split below underflows.
        return None;
    }
    if let Ok(n) = s.parse::<f64>() {
        return Some(n);
    }
    // Split off the last CHARACTER, not the last byte. `s.len() - 1` lands
    // inside a multi-byte one and panics the handler task: `"¡"` is the
    // smallest input that does it, and a `keep_alive` of `"5à"` or `"30s€"`
    // arrives from a request body like any other string. The empty case is
    // already out above, so there is always a last character to measure.
    let split = s.len() - s.chars().next_back().map_or(0, char::len_utf8);
    let (number, unit) = s.split_at(split);
    let n: f64 = number.parse().ok()?;
    match unit {
        "s" => Some(n),
        "m" => Some(n * 60.0),
        "h" => Some(n * 3600.0),
        _ => None,
    }
}

/// Shared implementation behind `touch_embedding_slot`/`touch_decision_slot`:
/// resolve `keep_alive` (falling back to `default` when it is `Default`)
/// into a new deadline, or clear it for `Forever`/`Immediate` (the caller
/// handles the actual unload for `Immediate`).
fn touch_deadline(
    deadline: &tokio::sync::Mutex<Option<tokio::time::Instant>>,
    keep_alive: KeepAlive,
    default: Option<std::time::Duration>,
) {
    let new_deadline = match keep_alive.resolve(default) {
        // `checked_add`: a duration a request may carry (up to ~584 billion
        // years) overflows an `Instant`, and `+` panics on that. A deadline
        // that far away is no deadline.
        KeepAlive::For(d) => tokio::time::Instant::now().checked_add(d),
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

    /// An empty (or whitespace-only) string is malformed like any other
    /// garbage — the documented fallback above — except it used to never get
    /// there: `split_at(s.len() - 1)` underflows on a zero-length string and
    /// panics the request task instead.
    #[test]
    fn an_empty_string_falls_back_to_default_rather_than_panicking() {
        assert_eq!(parse_keep_alive(Some(&v("\"\""))), KeepAlive::Default);
        assert_eq!(parse_keep_alive(Some(&v("\"   \""))), KeepAlive::Default);
        assert!(parse_keep_alive_flag("").is_err());
        assert!(parse_keep_alive_flag("   ").is_err());
    }

    /// Absurd magnitudes must behave like any other malformed value rather
    /// than panicking inside `from_secs_f64`: NaN, infinity, and anything
    /// past what a `Duration` holds are all reachable from a request body
    /// (`"nan"`/`"inf"` parse as floats; JSON numbers have no range check).
    #[test]
    fn absurd_durations_fall_back_to_default_rather_than_panicking() {
        for raw in ["1e20", "1e30", "\"1e30\"", "\"nan\"", "\"inf\""] {
            assert_eq!(parse_keep_alive(Some(&v(raw))), KeepAlive::Default);
        }
        assert_eq!(
            parse_keep_alive(Some(&v("300"))),
            KeepAlive::For(Duration::from_secs(300))
        );
        for s in ["1e30", "inf", "nan"] {
            assert!(parse_keep_alive_flag(s).is_err());
        }
        assert_eq!(
            parse_keep_alive_flag("5m").unwrap(),
            Duration::from_secs(300)
        );
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

    /// 1e19 seconds is a `Duration`, so it parses, and is not an `Instant`
    /// away from now: adding it panicked the embedding and decision handlers.
    #[test]
    fn a_keep_alive_past_the_end_of_time_is_no_deadline_not_a_panic() {
        let huge = parse_keep_alive(Some(&v("1e19")));
        assert!(matches!(huge, KeepAlive::For(_)), "{huge:?}");
        let deadline = tokio::sync::Mutex::new(None);
        touch_deadline(&deadline, huge, None);
        assert!(deadline.try_lock().unwrap().is_none());
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

    /// A duration whose last character is multi-byte used to panic the
    /// handler task: the unit was split off by byte index, and `s.len() - 1`
    /// lands inside such a character. The empty-string guard above covered
    /// only the length-zero case. Found by the property below, which shrank
    /// it to a single `"¡"`; kept here by name because a failing list reads
    /// better than a seed hash.
    #[test]
    fn a_multibyte_tail_is_malformed_not_a_panic() {
        for raw in ["¡", "5à", "30s€", "1h☃", "¡¡¡"] {
            assert_eq!(
                parse_keep_alive(Some(&serde_json::json!(raw))),
                KeepAlive::Default
            );
            assert!(parse_keep_alive_flag(raw).is_err());
        }
        // The ASCII grammar is untouched.
        assert_eq!(
            parse_keep_alive_flag("5m").unwrap(),
            Duration::from_secs(300)
        );
        assert_eq!(
            parse_keep_alive_flag("30s").unwrap(),
            Duration::from_secs(30)
        );
        assert_eq!(
            parse_keep_alive_flag("2h").unwrap(),
            Duration::from_secs(7200)
        );
    }

    // ── Properties ───────────────────────────────────────────────────────
    //
    // The tests above name values somebody thought of. These name the rules
    // instead, and let proptest hunt for the value that breaks one. Both
    // panics fixed in this function during September were a value nobody had
    // thought to write down: an empty string (#454) and `1e20` (#483). A
    // property would have produced each of them on the first run.
    //
    // `keep_alive` arrives inside a request body, so the strategy generates
    // what a body can actually carry: any JSON scalar, and strings both
    // arbitrary and duration-shaped. Note that NaN and infinity cannot be
    // JSON *numbers* — `serde_json` has no representation for them — which is
    // exactly why `"nan"` and `"inf"` have to be reachable as strings.
    use proptest::prelude::*;

    /// Any JSON scalar a `keep_alive` field can hold.
    fn any_keep_alive_value() -> impl Strategy<Value = serde_json::Value> {
        prop_oneof![
            proptest::num::f64::ANY.prop_map(|f| serde_json::json!(f)),
            any::<i64>().prop_map(|i| serde_json::json!(i)),
            ".*".prop_map(|s: String| serde_json::json!(s)),
            // Duration-shaped strings, the grammar the parser documents.
            (
                any::<f64>(),
                prop_oneof![Just(""), Just("s"), Just("m"), Just("h")]
            )
                .prop_map(|(n, u)| serde_json::json!(format!("{n}{u}"))),
            any::<bool>().prop_map(|b| serde_json::json!(b)),
            Just(serde_json::Value::Null),
        ]
    }

    proptest! {
        /// Whatever a request body carries, this returns — it never takes the
        /// handler task down with it. The documented contract is that a
        /// malformed value falls back to the server default; a panic is not a
        /// fallback.
        #[test]
        fn parse_keep_alive_never_panics(v in any_keep_alive_value()) {
            let _ = parse_keep_alive(Some(&v));
        }

        /// Sign decides the variant, and nothing else does. Negative means
        /// "never unload", zero means "unload now" — for every negative and
        /// every zero, not just the ones in the examples above.
        #[test]
        fn sign_alone_decides_forever_and_immediate(f in proptest::num::f64::NEGATIVE) {
            prop_assert_eq!(parse_keep_alive(Some(&serde_json::json!(f))), KeepAlive::Forever);
            prop_assert_eq!(parse_keep_alive(Some(&serde_json::json!(0.0))), KeepAlive::Immediate);
        }

        /// The CLI flag answers or refuses, never panics.
        #[test]
        fn parse_keep_alive_flag_never_panics(s in ".*") {
            let _ = parse_keep_alive_flag(&s);
        }

        /// And when it answers, the answer is a positive duration. Accepting
        /// zero here is the bug the flag parser exists to prevent: it would be
        /// read as `Duration::ZERO` and evict every model the moment it loads.
        #[test]
        fn the_flag_never_accepts_a_non_positive_duration(s in ".*") {
            if let Ok(d) = parse_keep_alive_flag(&s) {
                prop_assert!(d > Duration::ZERO, "accepted {s:?} as {d:?}");
            }
        }
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

/// Check if a loaded model name matches a requested name.
///
/// Handles the common case where the loaded model is a full path
/// (e.g. `/models/qwen3-8b.gguf`) but the request uses a short name
/// (e.g. `qwen3-8b` or `qwen3:8b`).
pub(crate) fn model_names_match(loaded: &str, normalized_request: &str) -> bool {
    // Exact match.
    if loaded == normalized_request {
        return true;
    }
    // Otherwise compare identity keys: last path component, `.gguf` stripped,
    // case-insensitive. See `model_identity_key` for why this is not
    // `file_stem` — model names contain dots, and cutting at the last one
    // made every quant of a repo look like the same model (#345).
    model_identity_key(loaded) == model_identity_key(normalized_request)
}

/// How generation models are kept resident, which one answers a request
/// that names none, and which ones `"model": "auto"` chooses between: the
/// user's flags, resolved once in `main.rs` and handed to `serve` whole.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidencyConfig {
    /// `--max-loaded-models` (see `RuntimeOpts::max_loaded_models`).
    pub max_loaded_models: usize,
    /// `--default-model`, resolved: the model a request that names none is
    /// answered by (see `RuntimeOpts::default_model`).
    pub default_model: Option<NamedModel>,
    /// `--auto-model` and `--auto-timeout-ms`, resolved; `None` without
    /// `--auto-model` (see `route`).
    pub router: Option<route::RouteTable>,
    /// `--default-model auto`: a request that names no model is routed.
    /// Only ever set with a `router`.
    pub default_auto: bool,
}

/// The flags `ResidencyConfig` is resolved from, as the command line gave
/// them.
#[derive(Debug, Clone, Default)]
pub struct ResidencyFlags {
    /// `--max-loaded-models`.
    pub max_loaded_models: usize,
    /// `--default-model`.
    pub default_model: Option<String>,
    /// Every `--auto-model`, in order.
    pub auto_models: Vec<String>,
    /// `--auto-timeout-ms`.
    pub auto_timeout_ms: u64,
}

/// A model named on the command line, resolved when the server started:
/// the name requests use for it, and its GGUF. The server loads it by that
/// name, or that path, even when a request's `model` may not name a path
/// (`EULLM_ALLOW_MODEL_PATHS`): whoever started the server named it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamedModel {
    pub name: String,
    pub path: PathBuf,
}

impl ResidencyConfig {
    /// From the flags as the command line gave them. `lookup` resolves a
    /// model named on the command line the way `--decision-model` is
    /// resolved — a store name or a GGUF path — to the name requests will
    /// use and its file, with what the store and the catalog say about it.
    /// A `--default-model` or an `--auto-model` it cannot resolve is an
    /// error: the server would otherwise start, and fail every request that
    /// needs it (see `route::RouteTable::resolve` for what else
    /// `--auto-model` refuses). `--default-model auto` routes a request that
    /// names no model, and needs `--auto-model` to route with.
    pub fn resolve(
        flags: &ResidencyFlags,
        lookup: impl Fn(&str) -> Option<CandidateFacts>,
    ) -> Result<Self, String> {
        let default_flag = flags.default_model.as_deref().map(str::trim);
        let default_auto = default_flag.is_some_and(|name| name.eq_ignore_ascii_case(route::AUTO));
        if default_auto && flags.auto_models.is_empty() {
            return Err(
                "--default-model auto routes requests that name no model, and needs the models \
                 to route between: give two or more --auto-model"
                    .to_string(),
            );
        }
        let default_model = match default_flag.filter(|_| !default_auto) {
            None => None,
            Some(name) => Some(lookup(name).map(|facts| facts.model).ok_or_else(|| {
                format!(
                    "--default-model '{name}' is not a model: give a GGUF path or a name \
                     `eullm list` shows (a catalog model has to be pulled first: eullm pull \
                     {name})"
                )
            })?),
        };
        let router = route::RouteTable::resolve(
            &flags.auto_models,
            default_model.as_ref(),
            std::time::Duration::from_millis(flags.auto_timeout_ms),
            &lookup,
        )?;
        Ok(Self {
            max_loaded_models: flags.max_loaded_models.max(1),
            default_model,
            router,
            default_auto,
        })
    }
}

/// What `"model": "auto"` chooses between, as the startup log says it: each
/// candidate with the description the decision model reads and where it came
/// from — a warning when it is the catalog's product text or the bare name —
/// and what will keep routing from working as configured.
fn log_route_table(
    table: &route::RouteTable,
    max_loaded_models: usize,
    decision_model: bool,
    keep_alive: bool,
) {
    tracing::info!(
        "\"model\": \"{}\" chooses between {} models, the fallback {}, deciding within {} ms \
         (--auto-model, --auto-timeout-ms)",
        route::AUTO,
        table.candidates.len(),
        table.fallback().name,
        table.timeout.as_millis()
    );
    for candidate in &table.candidates {
        let source = candidate.source.describe();
        if candidate.source.warns() {
            tracing::warn!(
                "  {}: \"{}\" — from {source}",
                candidate.name,
                candidate.description
            );
        } else {
            tracing::info!(
                "  {}: \"{}\" — from {source}",
                candidate.name,
                candidate.description
            );
        }
    }
    if !decision_model {
        tracing::warn!(
            "--auto-model without --decision-model: until a request loads a decision model, \
             every routed request is answered by the fallback, {}",
            table.fallback().name
        );
    }
    if max_loaded_models < table.candidates.len() {
        tracing::warn!(
            "--max-loaded-models {max_loaded_models} keeps fewer models loaded than the {} \
             --auto-model candidates: a request routed to one that is not loaded unloads \
             another, and alternating between them reloads a model each time",
            table.candidates.len()
        );
    }
    if keep_alive {
        tracing::warn!(
            "--keep-alive with --auto-model: an idle decision model is unloaded like any \
             other, and routing then falls back until a request loads one again"
        );
    }
}

/// The startup line for `OLLAMA_MAX_LOADED_MODELS`, when it is set: EuLLM
/// does not read it. Ollama counts every model, embedders included, and takes
/// it from the environment; here the number is a flag, model configuration
/// rather than perimeter, and counts generation models only. `flag_default`:
/// `--max-loaded-models` was left at its default.
fn ollama_env_hint(env: Option<&str>, flag_default: bool) -> Option<String> {
    let value = env.map(str::trim).filter(|v| !v.is_empty())?;
    Some(if flag_default {
        format!(
            "OLLAMA_MAX_LOADED_MODELS={value} is set, but EuLLM does not read it: pass \
             --max-loaded-models to keep several generation models resident (embedding \
             and decision models are not counted)"
        )
    } else {
        format!("OLLAMA_MAX_LOADED_MODELS={value} is set and ignored: --max-loaded-models decides")
    })
}

/// Configuration for starting the API server.
pub struct ServeConfig {
    /// The one `LlamaBackend` the process created at startup — see
    /// `AppState::backend`. Every model this server ever loads, launch
    /// model included, must share this same instance.
    pub backend: Arc<LlamaBackend>,
    pub port: u16,
    /// See `AppState::fallback_mmproj`.
    pub mmproj: Option<PathBuf>,
    /// See `AppState::mmproj_offload`.
    pub mmproj_offload: Option<bool>,
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
    /// `--n-ubatch`, for every model this server loads (see `AppState::n_ubatch`).
    pub n_ubatch: Option<u32>,
    pub cache_type_k: crate::inference::KvCacheType,
    pub cache_type_v: crate::inference::KvCacheType,
    pub batch_size: usize,
    pub cpu_moe: bool,
    pub n_cpu_moe: u32,
    pub rs_seq: u32,
    /// `--mtp`: MTP drafts per step for every model this server loads (see
    /// `InferenceConfig::mtp`).
    pub mtp: u32,
    /// `--mtp-p-min` (see `InferenceConfig::mtp_p_min`).
    pub mtp_p_min: f32,
    /// `--mtp-model` (see `InferenceConfig::mtp_model`).
    pub mtp_model: Option<std::path::PathBuf>,
    /// `--moe-cache` (see `AppState::moe_cache`).
    pub moe_cache: Option<crate::fit::MoeCache>,
    /// `--vram-reserve` in bytes (see `AppState::vram_reserve`).
    pub vram_reserve: u64,
    /// `--no-mmap` (see `AppState::no_mmap`).
    pub no_mmap: bool,
    /// `--mmap` (see `AppState::mmap`).
    pub mmap: bool,
    /// `--moe-prefetch` (see `AppState::moe_prefetch`).
    pub moe_prefetch: u32,
    /// `--load-threads` (see `AppState::load_threads`).
    pub load_threads: crate::readahead::LoadThreads,
    /// `--kv-unified` (see `AppState::kv_unified`).
    pub kv_unified: bool,
    /// `--split-mode`, `--tensor-split` (see `AppState::gpu_split`).
    pub gpu_split: crate::gpu_split::GpuSplit,
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
    /// An embedding model already loaded via `--embedding-model`, ready to
    /// seed the embedding slot at startup instead of waiting for the first
    /// `/api/embed`/`/v1/embeddings` request. Its
    /// `EmbeddingSlot::is_reserved_companion` must be `true` — the caller
    /// (`cmd_run`/`cmd_serve`) already reserved its footprint against the
    /// generation model's own `--fit` sizing before loading it, and that
    /// guarantee only means anything if it is then treated as reserved here
    /// too. `None` when `--embedding-model` was not given — the ordinary
    /// case, unaffected by any of this.
    pub launch_embedding: Option<EmbeddingSlot>,
    /// The `--decision-model` counterpart of `launch_embedding`.
    pub launch_decision: Option<DecisionSlot>,
    /// `--decision-ctx`: see `AppState::decision_ctx`.
    pub decision_ctx: u32,
    /// `--max-loaded-models` and `--default-model`, as the user gave them.
    pub residency: ResidencyConfig,
    /// How many layers the launch model actually put on the GPU, after its
    /// own sizing: reported by `/api/ps`, and nothing else. Never a setting
    /// for the next load — that is `gpu_layers`, the user's flag.
    pub launch_gpu_layers: Option<i32>,
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
    // Fatal when configured but unusable, for the reason the API keys are: an
    // operator who wrote a policy and gets every option through because of a
    // typo in it is worse off than one whose server refused to start.
    let decision_policy = decision_policy::DecisionPolicy::load(env_file).map_err(|e| {
        format!(
            "{e}\n  Expected {{\"version\": {}, \"deny_options\": [\"pattern\", …]}}. Refusing to \
             start: serving decisions without the policy you configured would be worse than \
             not starting.",
            decision_policy::POLICY_VERSION
        )
    })?;
    // Fatal too when set but unusable, as for an explicitly set
    // EULLM_AUDIT_DIR: whoever set it asked for the traces, and a server that
    // ran without them would leave a hole found only when the training data
    // is.
    let decision_traces = decision_traces::DecisionTraces::load(env_file)
        .map_err(|e| {
            format!(
                "EULLM_DECISION_TRACES is set but the decision traces cannot be written: {e}\n  \
                 Point it at a writable directory, or unset it to keep no traces."
            )
        })?
        .map(Arc::new);
    let web_policy = crate::tools::guard::WebPolicy::load();
    // From the environment or `.env`, as `.env.example` shows it: it used to
    // be read from the environment only.
    let allow_model_paths = crate::settings::flag("EULLM_ALLOW_MODEL_PATHS");

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
    tracing::info!(
        "Decision policy: {}  [source: {}]",
        decision_policy.describe(),
        decision_policy.source(),
    );
    if cfg.web_enabled {
        tracing::info!(
            "Web tool: enabled — fetchable: {}  [source: {}]",
            web_policy.describe(),
            web_policy.source()
        );
    }
    if let Some(source) = allow_model_paths {
        tracing::warn!(
            "EULLM_ALLOW_MODEL_PATHS is set ({}): a request's `model` field may name any \
             GGUF path readable by this process. Intended for local use; do not \
             combine it with an API reachable by untrusted callers.",
            source.label()
        );
    }
    let allow_model_paths = allow_model_paths.is_some();

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
        let (libraries, source) = cfg.store.libraries_with_source();
        for l in libraries {
            tracing::info!("Model library (read only): {}  [source: {source}]", l.display());
        }
    }
    match audit.check_writable() {
        Ok(()) => tracing::info!(
            "Audit trail: {}  [source: {}]",
            audit.log_path().display(),
            crate::audit::AuditLogger::source()
        ),
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
    match &decision_traces {
        Some(traces) => tracing::info!(
            "Decision traces: on — every decision's state, questions and answers, personal \
             data redacted, go to {}  [source: {}]",
            traces.decisions_path().display(),
            traces.source(),
        ),
        None => tracing::info!(
            "Decision traces: off (EULLM_DECISION_TRACES not set) — decisions are audited \
             with their state as a SHA-256 only"
        ),
    }

    let max_loaded_models = cfg.residency.max_loaded_models;
    tracing::info!(
        "Generation models kept resident: up to {max_loaded_models} (--max-loaded-models)"
    );
    match &cfg.residency.default_model {
        Some(model) => tracing::info!(
            "A request that names no model is answered by {} ({}; --default-model)",
            model.name,
            model.path.display()
        ),
        None if cfg.residency.default_auto => tracing::info!(
            "A request that names no model is routed, as \"model\": \"auto\" is \
             (--default-model auto)"
        ),
        None => tracing::info!(
            "A request that names no model is answered by the most recently used generation \
             model (no --default-model)"
        ),
    }
    if let Some(table) = &cfg.residency.router {
        log_route_table(
            table,
            max_loaded_models,
            cfg.launch_decision.is_some(),
            cfg.keep_alive.is_some(),
        );
    }
    if max_loaded_models > 1 && (!cfg.fit || crate::fit::vram_bytes().is_none()) {
        tracing::warn!(
            "--max-loaded-models {max_loaded_models}: generation models are kept up to the \
             count only; whether they fit together is not checked ({})",
            if cfg.fit {
                "free VRAM cannot be read on this build"
            } else {
                "--no-fit"
            }
        );
    }
    if let Some(hint) = ollama_env_hint(
        std::env::var("OLLAMA_MAX_LOADED_MODELS").ok().as_deref(),
        max_loaded_models == 1,
    ) {
        tracing::warn!("{hint}");
    }

    // `eullm run`'s model, loaded before the server started, is the first
    // resident; `serve` starts with none.
    let mut models = resident::ResidentModels::default();
    if let Some(name) = cfg.model_name
        && (cfg.engine.is_some() || cfg.scheduler.is_some())
    {
        let path = cfg
            .launch_model
            .as_ref()
            .map_or_else(|| PathBuf::from(&name), |(_, path)| path.clone());
        // A sequential launch model reserves its per-request context like
        // any other (see `LoadedModel::unallocated_reserve`).
        let unallocated_reserve = match &cfg.engine {
            Some(engine) if cfg.gpu_layers != 0 => crate::fit::context_reserve_bytes(
                crate::fit::read_gguf_info(&path).as_ref(),
                engine.context_size(),
                crate::inference::cache_type_bytes_per_elem(&cfg.cache_type_k),
                crate::inference::cache_type_bytes_per_elem(&cfg.cache_type_v),
                engine.n_ubatch(),
            ),
            _ => 0,
        };
        let info = crate::fit::read_gguf_info(&path);
        let file_size = std::fs::metadata(&path).map_or(0, |m| m.len());
        let facts = resident::LoadFacts {
            ctx_size: cfg
                .engine
                .as_ref()
                .map_or(cfg.ctx_size, |engine| engine.context_size()),
            batch_size: if cfg.scheduler.is_some() {
                cfg.batch_size
            } else {
                0
            },
            gpu_layers: cfg.launch_gpu_layers.unwrap_or(cfg.gpu_layers),
            n_layers: info.as_ref().map(|info| info.n_layers),
            size_bytes: file_size,
            size_vram: None,
            family: info.and_then(|info| info.architecture),
        };
        let mut launch = resident::LoadedModel::new(name, path, cfg.engine, cfg.scheduler);
        launch.launch = true;
        launch.unallocated_reserve = unallocated_reserve;
        launch.facts = facts;
        models.insert(launch);
    }

    let state = Arc::new(AppState {
        backend: cfg.backend,
        fallback_mmproj: cfg.mmproj.clone(),
        mmproj_offload: cfg.mmproj_offload,
        models: tokio::sync::RwLock::new(models),
        swap_lock: tokio::sync::Mutex::new(()),
        max_loaded_models,
        busy_wait: BUSY_EVICTION_WAIT,
        generation_evictions: std::sync::atomic::AtomicU64::new(0),
        #[cfg(test)]
        load_gate: None,
        gpu_layers: cfg.gpu_layers,
        fit: cfg.fit,
        fit_strict: cfg.fit_strict,
        ctx_size: cfg.ctx_size,
        threads: cfg.threads,
        flash_attn: cfg.flash_attn,
        n_batch: cfg.n_batch,
        n_ubatch: cfg.n_ubatch,
        cache_type_k: cfg.cache_type_k,
        cache_type_v: cfg.cache_type_v,
        batch_size: cfg.batch_size,
        cpu_moe: cfg.cpu_moe,
        n_cpu_moe: cfg.n_cpu_moe,
        rs_seq: cfg.rs_seq,
        mtp: cfg.mtp,
        mtp_p_min: cfg.mtp_p_min,
        mtp_model: cfg.mtp_model.clone(),
        moe_cache: cfg.moe_cache,
        vram_reserve: cfg.vram_reserve,
        no_mmap: cfg.no_mmap,
        mmap: cfg.mmap,
        moe_prefetch: cfg.moe_prefetch,
        load_threads: cfg.load_threads,
        kv_unified: cfg.kv_unified,
        gpu_split: cfg.gpu_split.clone(),
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
        default_model: cfg.residency.default_model,
        router: cfg.residency.router,
        default_auto: cfg.residency.default_auto,
        embedding: tokio::sync::RwLock::new(cfg.launch_embedding),
        decision: tokio::sync::RwLock::new(cfg.launch_decision),
        decision_ctx: cfg.decision_ctx,
        decision_policy,
        decision_traces,
        cross_slot_evictions: std::sync::atomic::AtomicU64::new(0),
        idle: Arc::new(tokio::sync::Notify::new()),
        idle_tick: IDLE_TICK,
        embedding_deadline: tokio::sync::Mutex::new(None),
        decision_deadline: tokio::sync::Mutex::new(None),
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

    // Started once the port is bound, so that requests are taken while the
    // candidates load; a request that wants one first loads it itself.
    if let Some(table) = state.router.clone() {
        let warming = state.clone();
        tokio::spawn(async move {
            let started = std::time::Instant::now();
            let warm = warming
                .warm_route_candidates(&table, KeepAlive::Default)
                .await;
            tracing::info!(
                "Auto routing warm-up: {} of {} candidates resident after {:.1} s ({})",
                warm.len(),
                table.candidates.len(),
                started.elapsed().as_secs_f64(),
                crate::audit::sanitize_for_log(&warm.join(", "))
            );
        });
    }

    // Spawn the optional chat-UI listener on a separate port. It exposes the
    // same API surface (so the embedded JS can call same-origin) plus the
    // HTML/CSS/JS for the chat at `/`. On by default for both `eullm run` and
    // `eullm serve`; `--no-ui` turns it off.
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

    // Free every model before returning, on the thread that owns it
    // (`unload_all` joins each scheduler thread, an embedding worker joins on
    // drop), while the GPU runtime is whole. Returning with a scheduler
    // thread still holding its model and context left the C runtime's exit
    // handlers, ggml's and ROCm's static destructors among them, to take the
    // device down under it: on LUMI-G nearly every server aborted on SIGTERM
    // ("double free or corruption", "corrupted size vs. prev_size", a ggml
    // abort on the way out of `main`), on every build from next10 to next16.
    // It is also the path a model swap takes, so a fault here is one a swap
    // would hit.
    let unloaded = state.unload_all().await;
    *state.embedding.write().await = None;
    *state.decision.write().await = None;
    tracing::info!(
        "Server shut down gracefully ({} generation model(s) unloaded).",
        unloaded.len()
    );
    Ok(())
}

/// How often the idle-unload loop looks for an expired keep_alive — see
/// `AppState::run_idle_unload_loop`.
const IDLE_TICK: std::time::Duration = std::time::Duration::from_secs(30);

/// How long a load waits for a busy resident model to finish its requests
/// when it needs that model's room, before it answers 503: long enough for
/// a long answer to end, short enough that a client is not left hanging.
const BUSY_EVICTION_WAIT: std::time::Duration = std::time::Duration::from_secs(120);

/// Holds a generation load just before it starts (`AppState::load_gate`):
/// `arrived` is notified when one gets there, and it goes on when `proceed`
/// is.
#[cfg(test)]
pub(crate) struct LoadGate {
    pub(crate) arrived: tokio::sync::Notify,
    pub(crate) proceed: tokio::sync::Notify,
}

#[cfg(test)]
impl AppState {
    /// A server with nothing loaded, and every other field at a test's
    /// starting value — named once, here, so that adding a field does not
    /// break every test that builds a state. Perimeter settings take their
    /// defaults: `.env` is read from a path that does not exist.
    pub(crate) fn for_tests(store: ModelStore, api_keys: auth::ApiKeys) -> Self {
        let absent = std::path::Path::new("/nonexistent/eullm-test/.env");
        Self {
            backend: crate::inference::test_backend(),
            fallback_mmproj: None,
            mmproj_offload: None,
            models: tokio::sync::RwLock::new(resident::ResidentModels::default()),
            swap_lock: tokio::sync::Mutex::new(()),
            max_loaded_models: 1,
            busy_wait: BUSY_EVICTION_WAIT,
            generation_evictions: std::sync::atomic::AtomicU64::new(0),
            load_gate: None,
            gpu_layers: 0,
            fit: false,
            fit_strict: false,
            ctx_size: 4096,
            threads: 1,
            flash_attn: false,
            n_batch: 512,
            n_ubatch: None,
            cache_type_k: crate::inference::KvCacheType::F16,
            cache_type_v: crate::inference::KvCacheType::F16,
            batch_size: 1,
            cpu_moe: false,
            n_cpu_moe: 0,
            rs_seq: 0,
            mtp: 0,
            mtp_p_min: 0.0,
            mtp_model: None,
            moe_cache: None,
            vram_reserve: crate::fit::DEFAULT_VRAM_RESERVE_BYTES,
            no_mmap: false,
            mmap: false,
            moe_prefetch: crate::fit::MOE_PREFETCH_SLOTS,
            load_threads: crate::readahead::LoadThreads::default(),
            kv_unified: false,
            gpu_split: Default::default(),
            ctx_checkpoints: 0,
            checkpoint_min_step: 8192,
            rust_debug: false,
            web_enabled: false,
            api_port: 0,
            store,
            ip_allowlist: ip_allowlist::IpAllowlist::load(absent),
            api_keys: Arc::new(api_keys),
            allowed_origins: origin::AllowedOrigins::load(absent),
            web_policy: crate::tools::guard::WebPolicy::default(),
            allow_model_paths: false,
            launch_model: None,
            default_model: None,
            router: None,
            default_auto: false,
            embedding: tokio::sync::RwLock::new(None),
            decision: tokio::sync::RwLock::new(None),
            decision_ctx: crate::inference::decision::DEFAULT_DECISION_CTX,
            decision_policy: decision_policy::DecisionPolicy::none(),
            decision_traces: None,
            cross_slot_evictions: std::sync::atomic::AtomicU64::new(0),
            idle: Arc::new(tokio::sync::Notify::new()),
            idle_tick: IDLE_TICK,
            embedding_deadline: tokio::sync::Mutex::new(None),
            decision_deadline: tokio::sync::Mutex::new(None),
            default_keep_alive: None,
        }
    }
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
        let message = "source IP not in the configured allowlist";
        if systemone::has_structured_errors(req.uri().path()) {
            return systemone::ApiError::new(
                axum::http::StatusCode::FORBIDDEN,
                "forbidden",
                message,
            )
            .into_response();
        }
        (axum::http::StatusCode::FORBIDDEN, message).into_response()
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
    // `/v1/systemone` refusals, and its feedback's, carry the body its
    // clients parse (see `systemone::ApiError`); every other endpoint keeps
    // the one it had.
    let structured = systemone::has_structured_errors(req.uri().path());
    match state.api_keys.authenticate(presented.as_deref()) {
        Ok(identity) => {
            req.extensions_mut().insert(identity);
            next.run(req).await
        }
        Err(auth::AuthError::Missing) => unauthorized(
            "missing API key — send it as `Authorization: Bearer <key>` or `X-Api-Key: <key>`",
            structured,
        ),
        Err(auth::AuthError::Invalid) => {
            // No key id to name: logging the presented token would write a
            // credential into the log file, and it may be a valid key for a
            // *different* deployment.
            tracing::warn!("Rejected request with an invalid API key");
            unauthorized("invalid API key", structured)
        }
        Err(auth::AuthError::RateLimited {
            key_id,
            retry_after_s,
        }) => {
            tracing::warn!("Key '{key_id}' is over its per-minute quota");
            let message =
                format!("rate limit exceeded for key '{key_id}' — retry in {retry_after_s}s");
            let retry_after = [(axum::http::header::RETRY_AFTER, retry_after_s.to_string())];
            if structured {
                let status = axum::http::StatusCode::TOO_MANY_REQUESTS;
                let error = systemone::ApiError::new(status, "too_many_requests", message);
                return (retry_after, error).into_response();
            }
            (
                axum::http::StatusCode::TOO_MANY_REQUESTS,
                retry_after,
                axum::Json(serde_json::json!({ "error": message })),
            )
                .into_response()
        }
    }
}

/// 401 with the `WWW-Authenticate` challenge, so a client library can tell an
/// authentication failure from a generic refusal. `structured`: the body of
/// a `/v1/systemone` error.
fn unauthorized(message: &str, structured: bool) -> axum::response::Response {
    let challenge = [(
        axum::http::header::WWW_AUTHENTICATE,
        "Bearer realm=\"eullm\"",
    )];
    if structured {
        let status = axum::http::StatusCode::UNAUTHORIZED;
        return (
            challenge,
            systemone::ApiError::new(status, "unauthorized", message),
        )
            .into_response();
    }
    (
        axum::http::StatusCode::UNAUTHORIZED,
        challenge,
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
        let message = "request origin is not allowed — set EULLM_ALLOWED_ORIGINS \
                       if this frontend should be permitted";
        if systemone::has_structured_errors(req.uri().path()) {
            return systemone::ApiError::new(
                axum::http::StatusCode::FORBIDDEN,
                "forbidden",
                message,
            )
            .into_response();
        }
        return (
            axum::http::StatusCode::FORBIDDEN,
            axum::Json(serde_json::json!({ "error": message })),
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
        // Which model a routed request went to, and why: a browser client
        // can read only the response headers named here.
        .expose_headers([
            axum::http::HeaderName::from_static(route::MODEL_HEADER),
            axum::http::HeaderName::from_static(route::ROUTE_HEADER),
            axum::http::HeaderName::from_static(route::ROUTE_ID_HEADER),
        ])
}

/// Read a request's body as JSON whatever its `Content-Type` says, as Ollama
/// does. Ollama's own examples are `curl … -d '{…}'`, which labels the body
/// `application/x-www-form-urlencoded`; a script's `fetch` with a string body
/// sends `text/plain`, and some clients send no type at all. axum's `Json`
/// refused all three with 415, so an Ollama example copied as it stands
/// failed here.
///
/// Only those three are relabelled: a body that says it is something else,
/// `multipart/form-data` or `application/octet-stream`, is still refused. Nor
/// is this an opening for a web page: those three are the types a page may
/// send without a CORS preflight, which is why `enforce_origin` refuses every
/// unsafe request from an origin not allowed, before any handler reads a
/// body. The content type was never that control.
async fn read_body_as_json(
    mut req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    if body_read_as_json(req.method(), req.headers()) {
        req.headers_mut().insert(
            axum::http::header::CONTENT_TYPE,
            axum::http::HeaderValue::from_static("application/json"),
        );
    }
    next.run(req).await
}

/// Whether [`read_body_as_json`] relabels this request: one that can carry a
/// body, whose type is none, `application/x-www-form-urlencoded` or
/// `text/plain`, parameters such as `charset` aside.
fn body_read_as_json(method: &axum::http::Method, headers: &axum::http::HeaderMap) -> bool {
    use axum::http::Method;
    if matches!(*method, Method::GET | Method::HEAD | Method::OPTIONS) {
        return false;
    }
    let Some(value) = headers.get(axum::http::header::CONTENT_TYPE) else {
        return true;
    };
    let Ok(value) = value.to_str() else {
        return false;
    };
    let media_type = value.split(';').next().unwrap_or_default().trim();
    media_type.is_empty()
        || media_type.eq_ignore_ascii_case("application/x-www-form-urlencoded")
        || media_type.eq_ignore_ascii_case("text/plain")
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
        .layer(axum::middleware::from_fn(read_body_as_json))
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
        .layer(axum::middleware::from_fn(read_body_as_json))
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
        spawn_with(store, Setup::default()).await
    }

    /// `spawn`, with API keys configured.
    async fn spawn_with_keys(store: ModelStore, api_keys: auth::ApiKeys) -> String {
        let setup = Setup {
            api_keys: Some(api_keys),
            ..Setup::default()
        };
        spawn_with(store, setup).await
    }

    /// What a test server starts with besides its store; by default, what
    /// `serve` starts with when nothing is configured.
    #[derive(Default)]
    struct Setup {
        api_keys: Option<auth::ApiKeys>,
        decision_policy: Option<decision_policy::DecisionPolicy>,
        decision_traces: Option<decision_traces::DecisionTraces>,
        decision: Option<DecisionSlot>,
    }

    /// Start the API with this setup, on `AppState::for_tests`.
    async fn spawn_with(store: ModelStore, setup: Setup) -> String {
        // A path that does not exist, so the perimeter types fall back to
        // their defaults instead of reading a developer's real `.env`.
        let absent = std::path::Path::new("/nonexistent/eullm-test/.env");
        let api_keys = setup
            .api_keys
            .unwrap_or_else(|| auth::ApiKeys::load(absent).expect("no keys configured"));
        let mut state = AppState::for_tests(store, api_keys);
        if let Some(policy) = setup.decision_policy {
            state.decision_policy = policy;
        }
        state.decision_traces = setup.decision_traces.map(Arc::new);
        state.decision = tokio::sync::RwLock::new(setup.decision);
        spawn_state(state).await
    }

    /// `spawn`, for a state the test has prepared.
    async fn spawn_state(state: AppState) -> String {
        let state = Arc::new(state);
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

    /// `DELETE /api/delete` removes a model from the models folder, says 404
    /// for one that is not there, and refuses one in a library, whose files
    /// are the user's; `/api/tags` says which is which.
    #[tokio::test]
    async fn a_model_is_deleted_from_the_folder_and_never_from_a_library() {
        let tmp = std::env::temp_dir().join(format!("eullm-delete-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let library = tmp.join("library");
        std::fs::create_dir_all(&library).expect("library dir");
        std::fs::write(library.join("shelf-model.gguf"), b"not a real gguf").expect("library file");
        let store = store_with_one_model(&tmp.join("store"), "a-pulled-model").with_libraries(vec![library.clone()]);
        let base = spawn(store).await;

        let (_, tags) = get_json(&format!("{base}/api/tags")).await;
        let source = |name: &str| {
            tags["models"]
                .as_array()
                .expect("models")
                .iter()
                .find(|m| m["name"] == name)
                .map(|m| m["details"]["source"].clone())
        };
        assert_eq!(source("a-pulled-model"), Some(serde_json::json!("store")));
        assert_eq!(source("shelf-model"), Some(serde_json::json!("library")));

        let delete = |model: &str| {
            let url = format!("{base}/api/delete");
            let body = serde_json::json!({ "model": model });
            async move {
                reqwest::Client::new()
                    .delete(url)
                    .json(&body)
                    .send()
                    .await
                    .expect("request")
                    .status()
            }
        };
        assert_eq!(delete("shelf-model").await, 403);
        assert!(library.join("shelf-model.gguf").exists(), "a library file is never deleted");
        assert_eq!(delete("a-pulled-model").await, 200);
        assert!(!tmp.join("store/a-pulled-model").exists());
        assert_eq!(delete("a-pulled-model").await, 404);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Every resident is listed as loaded, the most recently used first —
    /// the one the chat UI preselects — and a catalog model keeps its
    /// catalog metadata while marked so.
    #[tokio::test]
    async fn every_resident_is_marked_loaded_and_keeps_catalog_metadata() {
        let tmp = std::env::temp_dir().join(format!("eullm-tags-loaded-{}", uuid::Uuid::new_v4()));
        let store = store_with_one_model(&tmp, "a-pulled-model");
        let absent = std::path::Path::new("/nonexistent/eullm-test/.env");
        let state = AppState::for_tests(store, auth::ApiKeys::load(absent).expect("no keys"));
        let catalog = &crate::models::EU_CATALOG[0];
        {
            let mut models = state.models.write().await;
            for name in [catalog.id.as_str(), "/elsewhere/Custom-Model-Q4_K_M.gguf"] {
                models.insert(resident::LoadedModel::new(
                    name.into(),
                    name.into(),
                    None,
                    None,
                ));
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        }
        let base = spawn_state(state).await;

        let (status, body) = get_json(&format!("{base}/api/tags")).await;
        assert_eq!(status, 200);
        let models = body["models"].as_array().expect("models array");
        let loaded: Vec<&serde_json::Value> =
            models.iter().filter(|m| m["loaded"] == true).collect();
        let names: Vec<&str> = loaded.iter().filter_map(|m| m["name"].as_str()).collect();
        assert_eq!(
            names,
            ["/elsewhere/Custom-Model-Q4_K_M.gguf", catalog.id.as_str()],
            "every resident, the most recently used first"
        );
        assert_eq!(loaded[1]["digest"], catalog.digest.as_str());
        assert_eq!(loaded[1]["details"]["family"], catalog.base());
        assert_eq!(
            models
                .iter()
                .filter(|m| m["name"] == catalog.id.as_str())
                .count(),
            1,
            "listed once, as loaded"
        );

        let (_, body) = get_json(&format!("{base}/v1/models")).await;
        let ids: Vec<&str> = body["data"]
            .as_array()
            .expect("data array")
            .iter()
            .filter_map(|m| m["id"].as_str())
            .collect();
        for name in names {
            assert_eq!(
                ids.iter().filter(|id| **id == name).count(),
                1,
                "{name} in {ids:?}"
            );
        }
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
        // Still an OpenAI list; `models`, the System One SDKs' list of
        // decision models, is there and empty with none loaded.
        assert_eq!(body["object"], "list");
        assert_eq!(body["models"], serde_json::json!([]));
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

    // The catalog endpoints answer before any model is loaded, and both
    // refuse malformed input without reaching the network -- so these run in
    // CI, where there is none.
    #[tokio::test]
    async fn the_catalog_search_answers_an_empty_query_without_calling_out() {
        let tmp = std::env::temp_dir().join(format!("eullm-hfsearch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let store = store_with_one_model(&tmp, "a-pulled-model");
        let base = spawn(store).await;

        let (status, body) = get_json(&format!("{base}/api/hf/search?q=")).await;
        assert_eq!(status, 200);
        assert_eq!(
            body["models"].as_array().map(Vec::len),
            Some(0),
            "an empty query is an empty result, not a search for everything"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    // This id is interpolated into a huggingface.co URL, so it is validated
    // before the request is built rather than after it comes back.
    #[tokio::test]
    async fn the_catalog_refuses_a_repo_id_that_is_not_owner_slash_repo() {
        let tmp = std::env::temp_dir().join(format!("eullm-hfrepo-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let store = store_with_one_model(&tmp, "a-pulled-model");
        let base = spawn(store).await;

        for bad in ["", "owner", "a/b/c", "..%2Fetc"] {
            let (status, _) = get_json(&format!("{base}/api/hf/repo?id={bad}")).await;
            assert_eq!(status, 400, "`{bad}` must be refused, not requested");
        }
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
        assert_eq!(
            body["max_loaded_models"], 1,
            "one model at a time by default"
        );
        assert_eq!(body["loaded_models"], 0);
        assert_eq!(body["generation_evictions"], 0);
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

    /// Ollama reads a body as JSON whatever its type says, and its examples
    /// rely on it: `curl … -d '{…}'` labels the body
    /// `application/x-www-form-urlencoded`. Each type a client sends JSON
    /// under without saying so reaches the handler, which names the model it
    /// does not have; a body that says it is something else is still refused,
    /// and so is a page on another origin, whatever its type.
    #[tokio::test]
    async fn a_json_body_is_read_whatever_its_content_type_says() {
        let tmp = std::env::temp_dir().join(format!("eullm-content-type-{}", uuid::Uuid::new_v4()));
        let store = store_with_one_model(&tmp, "a-pulled-model");
        let base = spawn(store).await;
        let client = reqwest::Client::new();
        let body = r#"{"model": "this-model-does-not-exist", "input": "hi"}"#;

        for path in ["/api/embed", "/v1/embeddings"] {
            for content_type in [
                None,
                Some("application/x-www-form-urlencoded"),
                Some("text/plain;charset=UTF-8"),
            ] {
                let mut request = client.post(format!("{base}{path}")).body(body);
                if let Some(content_type) = content_type {
                    request = request.header("content-type", content_type);
                }
                let r = request.send().await.expect("request");
                let status = r.status();
                let text = r.text().await.unwrap_or_default();
                assert!(
                    status.is_client_error() && status != 415,
                    "{path}, {content_type:?}: {status} {text}"
                );
                assert!(
                    text.contains("this-model-does-not-exist"),
                    "{path}, {content_type:?}: the handler must have read the body: {text}"
                );
            }
            let r = client
                .post(format!("{base}{path}"))
                .header("content-type", "multipart/form-data; boundary=x")
                .body(body)
                .send()
                .await
                .expect("request");
            assert_eq!(r.status(), 415, "{path}");
        }

        let r = client
            .post(format!("{base}/api/embed"))
            .header("origin", "https://elsewhere.example")
            .header("content-type", "text/plain")
            .body(body)
            .send()
            .await
            .expect("request");
        assert_eq!(r.status(), 403);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn only_a_body_of_no_type_a_form_or_text_is_read_as_json() {
        use axum::http::{HeaderMap, HeaderValue, Method, header::CONTENT_TYPE};
        let typed = |value: &'static str| {
            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static(value));
            headers
        };
        assert!(body_read_as_json(&Method::POST, &HeaderMap::new()));
        assert!(body_read_as_json(&Method::DELETE, &HeaderMap::new()));
        assert!(body_read_as_json(
            &Method::POST,
            &typed("Application/X-WWW-Form-Urlencoded")
        ));
        assert!(body_read_as_json(
            &Method::POST,
            &typed("text/plain; charset=utf-8")
        ));
        // Already JSON, or something else: left as it is.
        assert!(!body_read_as_json(
            &Method::POST,
            &typed("application/json")
        ));
        assert!(!body_read_as_json(
            &Method::POST,
            &typed("application/octet-stream")
        ));
        assert!(!body_read_as_json(&Method::POST, &typed("text/html")));
        // No body to read.
        assert!(!body_read_as_json(&Method::GET, &HeaderMap::new()));
    }

    /// The `error` object of a `/v1/systemone` error body: the shape its
    /// clients parse, `{"error": {"code", "message", "question"?}}`.
    fn systemone_error(body: &str) -> serde_json::Value {
        let body: serde_json::Value = serde_json::from_str(body).expect("a JSON body");
        let error = body["error"].clone();
        assert!(error["code"].is_string(), "{body}");
        assert!(error["message"].is_string(), "{body}");
        error
    }

    /// `/v1/systemone` on a server with no decision model: the ways a
    /// request can fail before any model runs must each read as the
    /// client's mistake it is, with the body System One clients parse.
    #[tokio::test]
    async fn systemone_refuses_what_it_cannot_answer_as_client_errors() {
        let tmp = std::env::temp_dir().join(format!("eullm-systemone-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let store = store_with_one_model(&tmp, "a-pulled-model");
        let base = spawn(store).await;
        let url = format!("{base}/v1/systemone");
        let question = serde_json::json!({ "q": { "type": "noul", "instructions": "Urgent?" } });

        // `jev-latest` means "the loaded decision model", and there is none.
        let (status, body) = post_json(
            &url,
            serde_json::json!({ "model": "jev-latest", "state": "x", "questions": question }),
        )
        .await;
        assert_eq!(status, 400, "{body}");
        let error = systemone_error(&body);
        assert_eq!(error["code"], "model_not_loaded");
        assert!(body.contains("--decision-model"), "{body}");

        let (status, body) = post_json(
            &url,
            serde_json::json!({
                "model": "this-model-does-not-exist", "state": "x", "questions": question
            }),
        )
        .await;
        assert_eq!(status, 404, "{body}");
        assert_eq!(systemone_error(&body)["code"], "not_found");
        assert!(body.contains("this-model-does-not-exist"), "{body}");

        // Malformed: refused before any model resolution, not with a 500.
        let (status, body) = post_json(
            &url,
            serde_json::json!({ "model": "a-pulled-model", "state": "x", "questions": {} }),
        )
        .await;
        assert_eq!(status, 422, "{body}");
        assert_eq!(systemone_error(&body)["code"], "invalid_request");
        assert!(body.contains("at least one question"), "{body}");

        // One question wrong: named in `question`.
        let (status, body) = post_json(
            &url,
            serde_json::json!({ "state": "x", "questions": {
                "fine": { "type": "noul", "instructions": "Urgent?" },
                "team": { "type": "choice", "instructions": "Which?", "criteria": { "a": null } }
            } }),
        )
        .await;
        assert_eq!(status, 422, "{body}");
        let error = systemone_error(&body);
        assert_eq!(error["code"], "invalid_question");
        assert_eq!(error["question"], "team");

        let client = reqwest::Client::new();
        let r = client
            .post(&url)
            .header("content-type", "application/json")
            .body("{\"state\": ")
            .send()
            .await
            .expect("request");
        assert_eq!(r.status(), 422);
        assert_eq!(
            systemone_error(&r.text().await.unwrap())["code"],
            "invalid_json"
        );

        let r = client.get(&url).send().await.expect("request");
        assert_eq!(r.status(), 405);
        assert_eq!(
            systemone_error(&r.text().await.unwrap())["code"],
            "method_not_allowed"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// The middleware's refusals: in the structured shape on
    /// `/v1/systemone`, unchanged everywhere else, where Ollama and OpenAI
    /// clients read `{"error": "<message>"}`.
    #[tokio::test]
    async fn systemone_refusals_from_the_middleware_are_structured_there_only() {
        let tmp = std::env::temp_dir().join(format!("eullm-systemone-auth-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let keys = auth::ApiKeys::from_spec("ci:0123456789abcdef01").expect("keys");
        let base = spawn_with_keys(ModelStore::at(tmp.clone()), keys).await;

        let (status, body) = post_json(
            &format!("{base}/v1/systemone"),
            serde_json::json!({ "state": "x", "questions": {} }),
        )
        .await;
        assert_eq!(status, 401, "{body}");
        let error = systemone_error(&body);
        assert_eq!(error["code"], "unauthorized");
        assert!(
            error["message"]
                .as_str()
                .unwrap()
                .contains("missing API key")
        );

        let (status, body) = get_json(&format!("{base}/api/tags")).await;
        assert_eq!(status, 401);
        assert!(body["error"].is_string(), "{body}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Ollama clients call `/api/ps` to see what is loaded; it answered 404.
    #[tokio::test]
    async fn api_ps_answers_with_no_model() {
        let tmp = std::env::temp_dir().join(format!("eullm-ps-{}", uuid::Uuid::new_v4()));
        let base = spawn(ModelStore::at(tmp.clone())).await;
        let (status, body) = get_json(&format!("{base}/api/ps")).await;
        assert_eq!(status, 200);
        assert_eq!(body, serde_json::json!({ "models": [] }));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Unloading a model that is not loaded is not an error: it is already
    /// what was asked for. `unloaded` stays a string or null, which `eullm
    /// unload` from earlier releases reads.
    #[tokio::test]
    async fn api_unload_of_a_model_not_loaded_is_a_200_null() {
        let tmp = std::env::temp_dir().join(format!("eullm-unload-{}", uuid::Uuid::new_v4()));
        let store = store_with_one_model(&tmp, "a-pulled-model");
        let base = spawn(store).await;
        let url = format!("{base}/api/unload");
        for body in [
            serde_json::json!({ "model": "a-pulled-model" }),
            serde_json::json!({ "model": "this-model-does-not-exist" }),
            serde_json::json!({}),
        ] {
            let (status, text) = post_json(&url, body.clone()).await;
            assert_eq!(status, 200, "{body}: {text}");
            let answer: serde_json::Value = serde_json::from_str(&text).expect("json");
            assert!(answer["unloaded"].is_null(), "{answer}");
            assert_eq!(answer["unloaded_all"], serde_json::json!([]));
        }
        // No body at all, as `eullm unload` sends.
        let r = reqwest::Client::new()
            .post(&url)
            .send()
            .await
            .expect("request");
        assert_eq!(r.status(), 200);
        let answer: serde_json::Value = r.json().await.expect("json");
        assert!(answer["unloaded"].is_null(), "{answer}");

        let (status, _) = post_json(&url, serde_json::json!({ "model": 7 })).await;
        assert_eq!(status, 400);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// `keep_alive: 0` with an empty prompt or empty messages unloads the
    /// model without loading it first. The fixture's weights are not a GGUF,
    /// so a load would answer 500: this one answers `done_reason: "unload"`.
    #[tokio::test]
    async fn an_empty_request_with_keep_alive_zero_unloads_without_loading() {
        let tmp = std::env::temp_dir().join(format!("eullm-expire-{}", uuid::Uuid::new_v4()));
        let store = store_with_one_model(&tmp, "a-pulled-model");
        let base = spawn(store).await;
        for (path, body) in [
            (
                "/api/generate",
                serde_json::json!({ "model": "a-pulled-model", "prompt": "", "keep_alive": 0 }),
            ),
            (
                "/api/chat",
                serde_json::json!({ "model": "a-pulled-model", "messages": [], "keep_alive": 0 }),
            ),
        ] {
            let (status, text) = post_json(&format!("{base}{path}"), body).await;
            assert_eq!(status, 200, "{path}: {text}");
            let answer: serde_json::Value = serde_json::from_str(&text).expect("json");
            assert_eq!(answer["done_reason"], "unload", "{path}: {answer}");
            assert_eq!(answer["model"], "a-pulled-model");
            assert_eq!(answer["done"], true);
        }
        let (status, text) = post_json(
            &format!("{base}/api/generate"),
            serde_json::json!({ "model": "this-model-does-not-exist", "prompt": "", "keep_alive": 0 }),
        )
        .await;
        assert_eq!(status, 404, "{text}");
        let (status, text) = post_json(
            &format!("{base}/api/generate"),
            serde_json::json!({ "prompt": "", "keep_alive": 0 }),
        )
        .await;
        assert_eq!(status, 503, "{text}");
        assert!(text.contains("No model loaded"), "{text}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// A question the decision policy leaves without a choice is refused
    /// before any model is resolved, naming the question; one it leaves two
    /// options goes on to the model — here, to the 400 of a server without
    /// one.
    #[tokio::test]
    async fn the_decision_policy_refuses_a_question_it_leaves_without_a_choice() {
        let tmp = std::env::temp_dir().join(format!("eullm-policy-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let policy = decision_policy::DecisionPolicy::parse(
            r#"{"version": 1, "deny_options": ["delete_*", "Transfer_Funds"]}"#,
            "test".to_string(),
        )
        .expect("policy");
        let setup = Setup {
            decision_policy: Some(policy),
            ..Setup::default()
        };
        let base = spawn_with(ModelStore::at(tmp.clone()), setup).await;
        let url = format!("{base}/v1/systemone");

        let (status, body) = post_json(
            &url,
            serde_json::json!({ "state": "Close my account and send the balance to Bob.",
                "questions": {
                    "urgent": { "type": "noul", "instructions": "Urgent?" },
                    "action": { "type": "choice", "instructions": "What should the agent do?",
                                "criteria": { "delete_account": "Delete it",
                                              "transfer_funds": "Send the money",
                                              "ask_human": "Ask a person" } } } }),
        )
        .await;
        assert_eq!(status, 422, "{body}");
        let error = systemone_error(&body);
        assert_eq!(error["code"], "policy_denied");
        assert_eq!(error["question"], "action");
        let message = error["message"].as_str().unwrap();
        assert!(
            message.contains("\"delete_account\" and \"transfer_funds\""),
            "{message}"
        );
        assert!(
            message.contains("1 of this question's 3 options is left"),
            "{message}"
        );

        // Two options left: past the policy, to model resolution.
        let (status, body) = post_json(
            &url,
            serde_json::json!({ "state": "x", "questions": {
                "action": { "type": "choice", "instructions": "What should the agent do?",
                            "criteria": { "delete_account": null, "reply": null, "ask_human": null } } } }),
        )
        .await;
        assert_eq!(status, 400, "{body}");
        assert_eq!(systemone_error(&body)["code"], "model_not_loaded");

        // A client's own mistake is still reported as that, not as the
        // policy's: a one-option choice is invalid whatever it holds.
        let (status, body) = post_json(
            &url,
            serde_json::json!({ "state": "x", "questions": {
                "action": { "type": "choice", "instructions": "?", "criteria": { "delete_all": null } } } }),
        )
        .await;
        assert_eq!(status, 422, "{body}");
        assert_eq!(systemone_error(&body)["code"], "invalid_question");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// A feedback as a client sends it, about one decision.
    fn a_feedback() -> serde_json::Value {
        serde_json::json!({
            "id": "b1149e83-332b-48c0-bab7-53725922c4de",
            "answers": { "action": "refund", "is_urgent": true, "severity": 2 },
            "outcome": "Refunded; the customer confirmed from mario.rossi@example.com",
            "source": "user"
        })
    }

    /// With traces off there is nowhere to store feedback, and the endpoint
    /// says so, in the body System One clients parse.
    #[tokio::test]
    async fn feedback_is_refused_while_traces_are_off() {
        let tmp = std::env::temp_dir().join(format!("eullm-feedback-off-{}", std::process::id()));
        let base = spawn(ModelStore::at(tmp)).await;
        let url = format!("{base}/v1/systemone/feedback");

        let (status, body) = post_json(&url, a_feedback()).await;
        assert_eq!(status, 409, "{body}");
        let error = systemone_error(&body);
        assert_eq!(error["code"], "traces_disabled");
        assert!(body.contains("EULLM_DECISION_TRACES"), "{body}");

        let r = reqwest::get(&url).await.expect("request");
        assert_eq!(r.status(), 405);
        let error = systemone_error(&r.text().await.unwrap());
        assert_eq!(error["code"], "method_not_allowed");
        assert!(
            error["message"]
                .as_str()
                .unwrap()
                .starts_with("/v1/systemone/feedback accepts POST only"),
            "{error}"
        );
    }

    /// With traces on, a feedback is one line of `feedback.jsonl`, its text
    /// redacted; one that does not check out is refused and writes nothing.
    #[tokio::test]
    async fn feedback_is_appended_next_to_the_traces() {
        let tmp = std::env::temp_dir().join(format!("eullm-feedback-{}", uuid::Uuid::new_v4()));
        let traces = decision_traces::DecisionTraces::at(tmp.join("traces"), "test".into());
        let setup = Setup {
            decision_traces: Some(traces),
            ..Setup::default()
        };
        let base = spawn_with(ModelStore::at(tmp.join("store")), setup).await;
        let url = format!("{base}/v1/systemone/feedback");
        let file = tmp.join("traces").join("feedback.jsonl");

        let (status, body) = post_json(&url, a_feedback()).await;
        assert_eq!(status, 200, "{body}");
        let answer: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(
            answer,
            serde_json::json!({ "id": "b1149e83-332b-48c0-bab7-53725922c4de", "recorded": true })
        );
        let lines = std::fs::read_to_string(&file).expect("feedback.jsonl");
        assert_eq!(lines.lines().count(), 1);
        let line: serde_json::Value = serde_json::from_str(lines.trim_end()).unwrap();
        assert_eq!(line["schema"], 1);
        assert_eq!(line["kind"], "feedback");
        assert_eq!(line["id"], "b1149e83-332b-48c0-bab7-53725922c4de");
        assert_eq!(
            line["answers"],
            serde_json::json!({ "action": "refund", "is_urgent": true, "severity": 2 })
        );
        assert_eq!(
            line["outcome"],
            "Refunded; the customer confirmed from [EMAIL]"
        );
        assert_eq!(line["source"], "user");
        assert!(line["timestamp"].is_string());

        // Refused, and nothing written: a wrong answer names its question.
        let mut wrong = a_feedback();
        wrong["answers"]["severity"] = serde_json::json!(2.5);
        let (status, body) = post_json(&url, wrong).await;
        assert_eq!(status, 422, "{body}");
        let error = systemone_error(&body);
        assert_eq!(error["code"], "invalid_question");
        assert_eq!(error["question"], "severity");
        let mut no_id = a_feedback();
        no_id.as_object_mut().unwrap().remove("id");
        let (status, body) = post_json(&url, no_id).await;
        assert_eq!(status, 422, "{body}");
        assert_eq!(systemone_error(&body)["code"], "invalid_request");

        // A body far past what a feedback holds stops at the route's limit.
        let mut huge = a_feedback();
        huge["outcome"] = serde_json::json!("x".repeat(300 * 1024));
        let (status, body) = post_json(&url, huge).await;
        assert_eq!(status, 413, "{body}");
        assert_eq!(systemone_error(&body)["code"], "payload_too_large");

        // A body labelled as something JSON is not: refused, in the body
        // System One clients parse. (No label at all is read as JSON, as
        // everywhere else.)
        let r = reqwest::Client::new()
            .post(&url)
            .header("content-type", "multipart/form-data; boundary=x")
            .body(a_feedback().to_string())
            .send()
            .await
            .expect("request");
        assert_eq!(r.status(), 415);
        assert_eq!(
            systemone_error(&r.text().await.unwrap())["code"],
            "unsupported_media_type"
        );

        // A second feedback is a second line.
        let (status, _) = post_json(&url, a_feedback()).await;
        assert_eq!(status, 200);
        let lines = std::fs::read_to_string(&file).unwrap();
        assert_eq!(lines.lines().count(), 2);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Feedback sits behind the same checks as the decisions it is about,
    /// refused in the same body.
    #[tokio::test]
    async fn feedback_needs_the_key_a_decision_needs() {
        let tmp =
            std::env::temp_dir().join(format!("eullm-feedback-auth-{}", uuid::Uuid::new_v4()));
        let setup = Setup {
            api_keys: Some(auth::ApiKeys::from_spec("ci:0123456789abcdef01").expect("keys")),
            decision_traces: Some(decision_traces::DecisionTraces::at(
                tmp.clone(),
                "test".into(),
            )),
            ..Setup::default()
        };
        let base = spawn_with(ModelStore::at(tmp.join("store")), setup).await;
        let url = format!("{base}/v1/systemone/feedback");

        let (status, body) = post_json(&url, a_feedback()).await;
        assert_eq!(status, 401, "{body}");
        assert_eq!(systemone_error(&body)["code"], "unauthorized");
        assert!(!tmp.join("feedback.jsonl").exists());

        let r = reqwest::Client::new()
            .post(&url)
            .bearer_auth("0123456789abcdef01")
            .json(&a_feedback())
            .send()
            .await
            .expect("request");
        assert_eq!(r.status(), 200);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// A real decision through the whole stack, on the GGUF in
    /// `EULLM_DECISION_TEST_MODEL`: the response names its audit record,
    /// the trace has the same id, the state redacted and the option the
    /// policy removed, and feedback on the decision is stored under it. The
    /// audit record goes where `EULLM_AUDIT_DIR` says, so point it at a
    /// scratch directory:
    ///
    /// ```text
    /// EULLM_AUDIT_DIR=/tmp/eullm-test-audit \
    /// EULLM_DECISION_TEST_MODEL=/path/to/Jev-Style-0.8B-Decision-v3-Q4_K_M.gguf \
    ///     cargo test -p eullm-engine http_tests::real_model -- --ignored
    /// ```
    #[tokio::test]
    #[ignore = "needs a GGUF model in EULLM_DECISION_TEST_MODEL"]
    async fn real_model_a_decision_is_traced_and_takes_feedback() {
        let path = std::env::var("EULLM_DECISION_TEST_MODEL")
            .expect("set EULLM_DECISION_TEST_MODEL to a GGUF file");
        let backend = crate::inference::test_backend();
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get() as u32);
        let model = tokio::task::spawn_blocking(move || {
            DecisionModel::load(
                std::path::Path::new(&path),
                threads,
                crate::inference::decision::DEFAULT_DECISION_CTX,
                true,
                backend,
            )
        })
        .await
        .unwrap()
        .expect("load the model");
        let tmp = std::env::temp_dir().join(format!("eullm-real-traces-{}", uuid::Uuid::new_v4()));
        let policy = decision_policy::DecisionPolicy::parse(
            r#"{"version": 1, "deny_options": ["delete_*"]}"#,
            "test".to_string(),
        )
        .expect("policy");
        let setup = Setup {
            decision_policy: Some(policy),
            decision_traces: Some(decision_traces::DecisionTraces::at(
                tmp.join("traces"),
                "test".into(),
            )),
            decision: Some(DecisionSlot {
                model_name: "test-decision-model".into(),
                model: Arc::new(model),
                is_reserved_companion: true,
                reserve_bytes: 0,
            }),
            ..Setup::default()
        };
        let base = spawn_with(ModelStore::at(tmp.join("store")), setup).await;

        let (status, body) = post_json(
            &format!("{base}/v1/systemone"),
            serde_json::json!({
                "state": "Mario Rossi (mario.rossi@example.com, +39 333 1234567) was charged twice for March.",
                "questions": {
                    "billing": { "type": "noul", "instructions": "Is this about billing?" },
                    "action": { "type": "choice", "instructions": "What should the agent do?",
                                "criteria": { "refund": "Refund the duplicate charge",
                                              "delete_account": "Delete the account",
                                              "ask_human": "Hand it to a person" } } } }),
        )
        .await;
        assert_eq!(status, 200, "{body}");
        let response: serde_json::Value = serde_json::from_str(&body).unwrap();
        let audit_id = response["eullm"]["audit_id"]
            .as_str()
            .expect("eullm.audit_id");
        assert!(uuid::Uuid::parse_str(audit_id).is_ok(), "{audit_id}");
        let mut keys: Vec<&str> = response
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        assert_eq!(keys, ["answers", "eullm", "model", "timing", "usage"]);
        assert_eq!(
            response["eullm"]["policy_removed"],
            serde_json::json!({ "action": ["delete_account"] })
        );
        let probabilities = &response["answers"]["action"]["probabilities"];
        assert!(
            probabilities.get("delete_account").is_none(),
            "{probabilities}"
        );

        let traces = std::fs::read_to_string(tmp.join("traces/decisions.jsonl")).unwrap();
        assert_eq!(traces.lines().count(), 1);
        let line: serde_json::Value = serde_json::from_str(traces.trim_end()).unwrap();
        assert_eq!(line["id"], audit_id);
        assert_eq!(
            line["state"],
            "Mario Rossi ([EMAIL], [PHONE]) was charged twice for March."
        );
        assert_eq!(line["answers"]["action"], {
            let mut answer = response["answers"]["action"].clone();
            answer.as_object_mut().unwrap().remove("eullm");
            answer
        });
        assert_eq!(
            line["policy_removed"],
            serde_json::json!({ "action": ["delete_account"] })
        );
        if let Ok(dir) = std::env::var("EULLM_AUDIT_DIR") {
            let audit = std::fs::read_to_string(std::path::Path::new(&dir).join("audit.jsonl"))
                .expect("the audit trail");
            assert!(
                audit.contains(audit_id),
                "the decision is audited under its id"
            );
        }

        let (status, body) = post_json(
            &format!("{base}/v1/systemone/feedback"),
            serde_json::json!({ "id": audit_id, "answers": { "billing": true, "action": "refund" },
                                "outcome": "Refunded; Mario wrote back from mario.rossi@example.com",
                                "source": "user" }),
        )
        .await;
        assert_eq!(status, 200, "{body}");
        let feedback = std::fs::read_to_string(tmp.join("traces/feedback.jsonl")).unwrap();
        let line: serde_json::Value = serde_json::from_str(feedback.trim_end()).unwrap();
        assert_eq!(line["id"], audit_id);
        assert_eq!(line["outcome"], "Refunded; Mario wrote back from [EMAIL]");
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

    /// A route table over two fixtures of `store`, the second the fallback.
    fn two_candidates(store: &ModelStore) -> route::RouteTable {
        let candidate = |name: &str, description: &str| route::RouteCandidate {
            name: name.to_string(),
            path: store.gguf_path(name).expect("a fixture"),
            description: description.to_string(),
            source: route::DescriptionSource::Flag,
            has_projector: false,
        };
        route::RouteTable {
            candidates: vec![
                candidate("small-m", "Short everyday requests"),
                candidate("large-m", "Reasoning, maths and code"),
            ],
            fallback: 1,
            timeout: std::time::Duration::from_secs(1),
        }
    }

    /// Without `--auto-model` there is nothing to route with, and
    /// `/api/route` says so.
    #[tokio::test]
    async fn api_route_without_auto_model_is_a_404() {
        let tmp = std::env::temp_dir().join(format!("eullm-route-off-{}", uuid::Uuid::new_v4()));
        let base = spawn(store_with_one_model(&tmp, "a-pulled-model")).await;
        let (status, text) = post_json(
            &format!("{base}/api/route"),
            serde_json::json!({ "model": "auto", "prompt": "hi" }),
        )
        .await;
        assert_eq!(status, 404, "{text}");
        assert!(text.contains("auto routing is not configured"), "{text}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Routing configured but no decision model: the fallback answers, the
    /// reason says why, and what the decision model would have read — the
    /// state and the question, its options in order — is in the answer.
    /// Nothing is loaded.
    #[tokio::test]
    async fn api_route_without_a_decision_model_answers_with_the_fallback() {
        let tmp = std::env::temp_dir().join(format!("eullm-route-{}", uuid::Uuid::new_v4()));
        store_with_one_model(&tmp, "small-m");
        let store = store_with_one_model(&tmp, "large-m");
        let absent = std::path::Path::new("/nonexistent/eullm-test/.env");
        let mut state = AppState::for_tests(store, auth::ApiKeys::load(absent).expect("no keys"));
        state.router = Some(two_candidates(&state.store));
        let base = spawn_state(state).await;
        let url = format!("{base}/api/route");

        let (status, text) = post_json(
            &url,
            serde_json::json!({ "model": "auto", "messages": [
                { "role": "user", "content": "What is 2 + 2?" }
            ] }),
        )
        .await;
        assert_eq!(status, 200, "{text}");
        let route: serde_json::Value = serde_json::from_str(&text).expect("json");
        assert_eq!(route["model"], "large-m");
        assert_eq!(route["fallback"], "large-m");
        assert_eq!(route["reason"], "no_decision_model");
        assert!(route["decision_model"].is_null() && route["confidence"].is_null());
        let candidates = route["candidates"].as_array().expect("candidates");
        assert_eq!(candidates.len(), 2);
        for candidate in candidates {
            assert!(candidate["probability"].is_null(), "{candidate}");
            assert_eq!(candidate["resident"], false);
        }
        let state_text = route["state"].as_str().expect("the state");
        assert!(
            state_text.starts_with("Request to answer, with its context.")
                && state_text.ends_with("Latest message:\nWhat is 2 + 2?"),
            "{state_text}"
        );
        assert_eq!(route["question"]["type"], "choice");
        assert_eq!(route["question"]["instructions"], route::ROUTE_QUESTION);
        assert!(
            text.contains(
                r#""criteria":{"small-m":"Short everyday requests","large-m":"Reasoning, maths and code"}"#
            ),
            "the options in their order: {text}"
        );
        assert!(uuid::Uuid::parse_str(route["route_id"].as_str().unwrap()).is_ok());

        // A prompt routes too; a body with neither is refused.
        let (status, text) = post_json(&url, serde_json::json!({ "prompt": "Why?" })).await;
        assert_eq!(status, 200, "{text}");
        assert!(text.contains(r#"Prompt:\nWhy?"#), "{text}");
        let (status, _) = post_json(&url, serde_json::json!({ "model": "auto" })).await;
        assert_eq!(status, 400);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// With `--default-model`, a request that names no model — or names it
    /// empty — goes to that model, on every endpoint: here a fixture whose
    /// weights are not a GGUF, so the load fails with a 500 that names it,
    /// where without the flag the answer is the 503 above. Unloading with
    /// an empty request and `keep_alive: 0` names it too.
    #[tokio::test]
    async fn a_request_that_names_no_model_goes_to_the_default_model() {
        let tmp = std::env::temp_dir().join(format!("eullm-default-{}", uuid::Uuid::new_v4()));
        let store = store_with_one_model(&tmp, "a-pulled-model");
        let path = store.gguf_path("a-pulled-model").expect("the fixture");
        let absent = std::path::Path::new("/nonexistent/eullm-test/.env");
        let mut state = AppState::for_tests(store, auth::ApiKeys::load(absent).expect("no keys"));
        state.default_model = Some(NamedModel {
            name: "a-pulled-model".into(),
            path,
        });
        let base = spawn_state(state).await;
        let hi = serde_json::json!([{ "role": "user", "content": "hi" }]);
        for (endpoint, body) in [
            ("/api/generate", serde_json::json!({ "prompt": "hi" })),
            (
                "/api/generate",
                serde_json::json!({ "model": "", "prompt": "hi" }),
            ),
            ("/api/chat", serde_json::json!({ "messages": hi })),
            (
                "/v1/chat/completions",
                serde_json::json!({ "model": " ", "messages": hi }),
            ),
        ] {
            let (status, text) = post_json(&format!("{base}{endpoint}"), body.clone()).await;
            assert_eq!(status, 500, "{endpoint} {body}: {text}");
            assert!(text.contains("a-pulled-model"), "{endpoint}: {text}");
        }
        let (status, text) = post_json(
            &format!("{base}/api/generate"),
            serde_json::json!({ "prompt": "", "keep_alive": 0 }),
        )
        .await;
        assert_eq!(status, 200, "{text}");
        let answer: serde_json::Value = serde_json::from_str(&text).expect("json");
        assert_eq!(answer["done_reason"], "unload");
        assert_eq!(answer["model"], "a-pulled-model");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Without `--auto-model`, `auto` is a name like any other, and no model
    /// has it: a 404 that names it, as before routing existed. Nor is it
    /// listed.
    #[tokio::test]
    async fn auto_without_auto_model_is_a_model_that_does_not_exist() {
        let tmp = std::env::temp_dir().join(format!("eullm-auto-off-{}", uuid::Uuid::new_v4()));
        let base = spawn(store_with_one_model(&tmp, "a-pulled-model")).await;
        let hi = serde_json::json!([{ "role": "user", "content": "hi" }]);
        for (endpoint, body) in [
            (
                "/api/generate",
                serde_json::json!({ "model": "auto", "prompt": "hi" }),
            ),
            (
                "/api/chat",
                serde_json::json!({ "model": "auto", "messages": hi }),
            ),
            (
                "/v1/chat/completions",
                serde_json::json!({ "model": "auto", "messages": hi }),
            ),
        ] {
            let (status, text) = post_json(&format!("{base}{endpoint}"), body).await;
            assert_eq!(status, 404, "{endpoint}: {text}");
            assert!(text.contains("auto"), "{endpoint}: {text}");
        }
        let (_, tags) = get_json(&format!("{base}/api/tags")).await;
        assert!(!tags.to_string().contains("eullm-router"), "{tags}");
        let (_, models) = get_json(&format!("{base}/v1/models")).await;
        assert!(!models.to_string().contains(r#""id":"auto""#), "{models}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Routing configured, and neither candidate loads (their weights are
    /// not GGUFs): with no decision model the fallback is chosen, and the
    /// request fails as one naming it would, on every endpoint — also when
    /// the model chosen was the other one, which the fallback then stands
    /// in for, and when the request names no model under `--default-model
    /// auto`. A raw prompt is refused before anything is chosen; the
    /// requests with nothing to generate answer as `auto`; and `auto` is
    /// listed.
    #[tokio::test]
    async fn a_routed_request_is_answered_by_the_model_chosen_or_the_fallback() {
        let tmp = std::env::temp_dir().join(format!("eullm-auto-{}", uuid::Uuid::new_v4()));
        store_with_one_model(&tmp, "small-m");
        let store = store_with_one_model(&tmp, "large-m");
        let absent = std::path::Path::new("/nonexistent/eullm-test/.env");
        let mut state = AppState::for_tests(store, auth::ApiKeys::load(absent).expect("no keys"));
        let mut table = two_candidates(&state.store);
        // The only one that can read an image: a request with one has
        // nothing to decide, and goes to it.
        table.candidates[0].has_projector = true;
        state.router = Some(table);
        state.default_auto = true;
        let base = spawn_state(state).await;

        let hi = serde_json::json!([{ "role": "user", "content": "hi" }]);
        let photo = serde_json::json!([{ "role": "user", "content": "what is this?",
            "images": ["aGVsbG8="] }]);
        for (endpoint, body) in [
            (
                "/api/generate",
                serde_json::json!({ "model": "auto", "prompt": "hi" }),
            ),
            (
                "/api/generate",
                serde_json::json!({ "model": "AUTO", "prompt": "hi", "stream": true }),
            ),
            ("/api/generate", serde_json::json!({ "prompt": "hi" })),
            (
                "/api/chat",
                serde_json::json!({ "model": "auto", "messages": hi }),
            ),
            (
                "/api/chat",
                serde_json::json!({ "model": "auto", "messages": photo }),
            ),
            (
                "/v1/chat/completions",
                serde_json::json!({ "model": "auto", "messages": hi, "stream": true }),
            ),
            (
                "/v1/chat/completions",
                serde_json::json!({ "messages": hi }),
            ),
        ] {
            let (status, text) = post_json(&format!("{base}{endpoint}"), body.clone()).await;
            assert_eq!(status, 500, "{endpoint} {body}: {text}");
            assert!(
                text.contains("Failed to load model 'large-m'"),
                "{endpoint} {body}: {text}"
            );
        }

        let (status, text) = post_json(
            &format!("{base}/api/generate"),
            serde_json::json!({ "model": "auto", "prompt": "<|im_start|>", "raw": true }),
        )
        .await;
        assert_eq!(status, 400, "{text}");
        assert!(text.contains("raw prompt"), "{text}");

        for (endpoint, body, done_reason) in [
            (
                "/api/generate",
                serde_json::json!({ "model": "auto", "prompt": "" }),
                "load",
            ),
            (
                "/api/chat",
                serde_json::json!({ "model": "auto", "messages": [], "keep_alive": 0 }),
                "unload",
            ),
        ] {
            let (status, text) = post_json(&format!("{base}{endpoint}"), body).await;
            assert_eq!(status, 200, "{endpoint}: {text}");
            let answer: serde_json::Value = serde_json::from_str(&text).expect("json");
            assert_eq!(answer["model"], "auto", "{answer}");
            assert_eq!(answer["done_reason"], done_reason, "{answer}");
        }

        let (_, tags) = get_json(&format!("{base}/api/tags")).await;
        let auto = tags["models"]
            .as_array()
            .and_then(|models| models.iter().find(|m| m["name"] == "auto"))
            .expect("auto in /api/tags");
        assert_eq!(auto["details"]["family"], "eullm-router");
        assert_eq!(
            auto["details"]["candidates"],
            serde_json::json!(["small-m", "large-m"])
        );
        let (_, models) = get_json(&format!("{base}/v1/models")).await;
        assert!(
            models["data"]
                .as_array()
                .is_some_and(|data| data.iter().any(|m| m["id"] == "auto")),
            "{models}"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// A server whose one resident, `busy-m`, is answered by `scheduler`, a
    /// handle with no decode thread behind it (`SchedulerHandle::detached`).
    async fn spawn_with_scheduler(
        tmp: &std::path::Path,
        scheduler: crate::inference::SchedulerHandle,
    ) -> String {
        let store = store_with_one_model(tmp, "busy-m");
        let absent = std::path::Path::new("/nonexistent/eullm-test/.env");
        let state = AppState::for_tests(store, auth::ApiKeys::load(absent).expect("no keys"));
        state
            .models
            .write()
            .await
            .insert(resident::LoadedModel::new(
                "busy-m".into(),
                tmp.join("busy-m").join("model.gguf"),
                None,
                Some(scheduler),
            ));
        spawn_state(state).await
    }

    /// Every endpoint that queues a generation, streamed and not, and the
    /// status and `Retry-After` each answers with.
    async fn refusals(base: &str) -> Vec<(String, u16, Option<String>, serde_json::Value)> {
        let hi = serde_json::json!([{ "role": "user", "content": "hi" }]);
        let mut answers = Vec::new();
        for stream in [false, true] {
            for (endpoint, body) in [
                (
                    "/api/generate",
                    serde_json::json!({ "model": "busy-m", "prompt": "hi", "stream": stream }),
                ),
                (
                    "/api/chat",
                    serde_json::json!({ "model": "busy-m", "messages": hi, "stream": stream }),
                ),
                (
                    "/v1/chat/completions",
                    serde_json::json!({ "model": "busy-m", "messages": hi, "stream": stream }),
                ),
            ] {
                let response = reqwest::Client::new()
                    .post(format!("{base}{endpoint}"))
                    .json(&body)
                    .send()
                    .await
                    .expect("request");
                let status = response.status().as_u16();
                let retry_after = response
                    .headers()
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .map(str::to_string);
                let body = response.json().await.unwrap_or(serde_json::Value::Null);
                answers.push((
                    format!("{endpoint} stream={stream}"),
                    status,
                    retry_after,
                    body,
                ));
            }
        }
        answers
    }

    /// A full queue is a 503 with `Retry-After` on every endpoint, streamed
    /// or not, answered before any stream opens — as Ollama answers its own
    /// full queue. It was a 500 without streaming, which says the fault is
    /// the server's, and with streaming a 200 whose only line was the error,
    /// which a client cannot tell from an answer that failed halfway.
    #[tokio::test]
    async fn a_full_queue_is_a_503_with_retry_after_on_every_endpoint() {
        let tmp = std::env::temp_dir().join(format!("eullm-queue-full-{}", uuid::Uuid::new_v4()));
        let (scheduler, _queue) = crate::inference::SchedulerHandle::detached(1);
        let _first = scheduler
            .try_submit(crate::inference::GenerateRequest::default())
            .expect("the queue's one place was free");
        let base = spawn_with_scheduler(&tmp, scheduler).await;

        for (request, status, retry_after, body) in refusals(&base).await {
            assert_eq!(status, 503, "{request}: {body}");
            assert_eq!(retry_after.as_deref(), Some("5"), "{request}");
            assert_eq!(
                body["error"], "Scheduler queue full — try again later",
                "{request}: a JSON error, not a stream"
            );
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// A model unloaded between a request finding it and reaching its queue
    /// is a 503 too, which the request can be sent again after at once: it
    /// loads the model back.
    #[tokio::test]
    async fn a_model_unloaded_before_the_request_starts_is_a_503() {
        let tmp = std::env::temp_dir().join(format!("eullm-unloaded-{}", uuid::Uuid::new_v4()));
        let (scheduler, queue) = crate::inference::SchedulerHandle::detached(1);
        drop(queue);
        let base = spawn_with_scheduler(&tmp, scheduler).await;

        for (request, status, retry_after, body) in refusals(&base).await {
            assert_eq!(status, 503, "{request}: {body}");
            assert_eq!(retry_after.as_deref(), Some("1"), "{request}");
            assert!(
                body["error"]
                    .as_str()
                    .is_some_and(|e| e.contains("send it again")),
                "{request}: {body}"
            );
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }
}

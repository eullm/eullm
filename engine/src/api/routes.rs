//! Route definitions for the EULLM and OpenAI-compatible APIs.
//!
//! Supports both non-streaming (JSON) and streaming (SSE) responses
//! on all generation endpoints. Set `"stream": true` in the request body
//! to get token-by-token Server-Sent Events.
//!
//! When a `SchedulerHandle` is present, requests are dispatched through the
//! continuous batching scheduler. Otherwise the sequential `InferenceEngine`
//! is used as fallback.
//!
//! **Dynamic model swap:** if a request specifies a `model` that differs from
//! the currently loaded one, the server automatically swaps to the new model.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use axum::body::Body;
use axum::extract::{Query, State};
use axum::http::{StatusCode, header};
use axum::response::IntoResponse;
use axum::response::sse::{Event, Sse};
use axum::{Json, Router, routing::get, routing::post};
use futures_util::stream::Stream;
use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::inference::embedding::{DEFAULT_EMBEDDING_CTX, Embedded, EmbeddingModel};

use super::AppState;
use super::resident::{Lease, SlotSnapshot};
use super::thinking::{Part, ThinkingSplitter};
use crate::audit::{AuditEntry, AuditLogger};
use crate::inference::scheduler::NotQueued;
use crate::inference::{
    AnswerStats, GenerateRequest, InferenceEngine, JSON_GBNF, StopReason, StreamEvent,
};
use crate::models::EU_CATALOG;
use crate::tools;

type S = Arc<AppState>;

/// Upper bound accepted for a request's `batch_size` override.
///
/// The override reaches `SchedulerConfig::max_batch_size` (one KV-cache slot
/// each) and `queue_capacity` (= `batch_size * 8`), so an unbounded value both
/// asks for an absurd allocation and — because slot ids are `i32` — can wrap to
/// zero usable slots while still reporting the model as loaded. 64 is far above
/// any real concurrency for a single-GPU appliance.
const MAX_BATCH_SIZE_OVERRIDE: usize = 64;

/// Bounds accepted for a request's `ctx_size` override. The lower bound leaves
/// room for a prompt plus at least one output token; the upper bound is the
/// largest context any current architecture declares, and keeps the value from
/// becoming a KV-cache allocation that fails *after* `load_generation_model` has already
/// unloaded the previous model.
const MIN_CTX_SIZE_OVERRIDE: u32 = 512;
const MAX_CTX_SIZE_OVERRIDE: u32 = 1_048_576;

/// Error shape returned by the JSON handlers: an HTTP status plus a JSON body.
type ApiError = (StatusCode, Json<Value>);

/// What a generation handler answers when it cannot answer: an [`ApiError`],
/// and for a 503 a client may simply retry, the `Retry-After` that says when.
struct Refusal {
    status: StatusCode,
    body: Json<Value>,
    retry_after_secs: Option<u64>,
}

impl From<ApiError> for Refusal {
    fn from((status, body): ApiError) -> Self {
        Self {
            status,
            body,
            retry_after_secs: None,
        }
    }
}

impl IntoResponse for Refusal {
    fn into_response(self) -> axum::response::Response {
        let mut response = (self.status, self.body).into_response();
        if let Some(secs) = self.retry_after_secs {
            response
                .headers_mut()
                .insert(header::RETRY_AFTER, axum::http::HeaderValue::from(secs));
        }
        response
    }
}

/// When a client refused with `ModelError::Busy`, or by a full queue, may
/// try again.
const BUSY_RETRY_AFTER_SECS: u64 = 5;

/// When a client may resend a request whose model was unloaded before it
/// started: at once, since the request itself loads the model back.
const UNLOADED_RETRY_AFTER_SECS: u64 = 1;

impl From<NotQueued> for Refusal {
    /// A request the scheduler did not queue is answered 503, as Ollama
    /// answers its own full queue, with the `Retry-After` that says when to
    /// send it again — and before any stream is opened: a streaming request
    /// used to get a 200 whose only line was the error, which a client
    /// cannot tell from an answer that failed halfway, and a non-streaming
    /// one a 500, which says the fault is the server's.
    fn from(not_queued: NotQueued) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            body: Json(json!({ "error": not_queued.to_string() })),
            retry_after_secs: Some(match not_queued {
                NotQueued::QueueFull => BUSY_RETRY_AFTER_SECS,
                NotQueued::Unloaded => UNLOADED_RETRY_AFTER_SECS,
            }),
        }
    }
}

/// Validated `(batch_size, ctx_size)` slot overrides read from a request body.
/// `None` in either position means "keep the launch-time value".
type SlotOverrides = (Option<usize>, Option<u32>);

/// Read the `batch_size` / `ctx_size` slot overrides from a request body,
/// rejecting anything outside a serviceable range with HTTP 400.
///
/// Both fields are forwarded to `AppState::load_generation_model`, which rebuilds the
/// scheduler and reallocates the KV cache. A value that is merely *parsed*
/// rather than *validated* therefore turns a single request into a
/// configuration change that can leave the server unable to serve anything
/// until it is restarted — so an out-of-range value has to fail loudly here
/// instead of being clamped silently (a client asking for 4096 slots has a bug
/// worth surfacing, not an intent worth guessing).
///
/// An absent field yields `None` (keep the launch-time value). A field that is
/// present but not a non-negative integer is an error, not an absence: silently
/// ignoring `{"batch_size": -1}` would hide a client bug behind unchanged
/// behaviour.
fn parse_slot_overrides(body: &Value) -> Result<SlotOverrides, ApiError> {
    let bad = |field: &str, detail: String| {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": format!("invalid `{field}`: {detail}") })),
        )
    };

    let batch_size = match body.get("batch_size") {
        None | Some(Value::Null) => None,
        Some(v) => {
            let n = v
                .as_u64()
                .ok_or_else(|| bad("batch_size", "expected a non-negative integer".into()))?;
            if n == 0 || n > MAX_BATCH_SIZE_OVERRIDE as u64 {
                return Err(bad(
                    "batch_size",
                    format!("must be between 1 and {MAX_BATCH_SIZE_OVERRIDE}, got {n}"),
                ));
            }
            Some(n as usize)
        }
    };

    let ctx_size = match body.get("ctx_size") {
        None | Some(Value::Null) => None,
        Some(v) => {
            let n = v
                .as_u64()
                .ok_or_else(|| bad("ctx_size", "expected a non-negative integer".into()))?;
            if n < MIN_CTX_SIZE_OVERRIDE as u64 || n > MAX_CTX_SIZE_OVERRIDE as u64 {
                return Err(bad(
                    "ctx_size",
                    format!(
                        "must be between {MIN_CTX_SIZE_OVERRIDE} and {MAX_CTX_SIZE_OVERRIDE}, got {n}"
                    ),
                ));
            }
            Some(n as u32)
        }
    };

    Ok((batch_size, ctx_size))
}

/// EULLM native API routes (`/api/*`).
pub fn api_routes() -> Router<S> {
    Router::new()
        .route("/tags", get(list_models))
        .route("/ps", get(list_running))
        .route("/monitor", get(monitor))
        .route("/about", get(about))
        .route("/generate", post(generate))
        .route("/chat", post(chat))
        .route("/show", post(show_model))
        .route("/pull", post(pull_model))
        .route("/version", get(version))
        .route("/unload", post(unload_model))
        .route("/delete", axum::routing::delete(delete_model))
        .route("/route", post(route_request))
        .route("/embed", post(embed))
        .route("/hf/search", get(hf_search))
        .route("/hf/repo", get(hf_repo))
}

/// `GET /api/hf/search?q=<text>&limit=<n>&sort=<order>` — search the
/// HuggingFace Hub for repos containing GGUF files, most downloaded first or
/// in `sort` order (`downloads`, `likes`, `trending`, `newest`). With a
/// `sort` and no `q`, every GGUF repo in that order.
///
/// The engine makes the call, not the browser. That keeps the viewer's
/// address away from the Hub, leaves `EULLM_WEB_ALLOWED_DOMAINS` as the one
/// place the perimeter is decided, and means the catalog still works from a
/// machine whose browser has no route out but whose engine does — an HPC
/// login node, most of all, which is where downloads have to be started when
/// compute nodes are offline.
async fn hf_search(Query(params): Query<HashMap<String, String>>) -> impl IntoResponse {
    let q = params.get("q").map(String::as_str).unwrap_or("").trim();
    // Without a query, the GGUF repos in this order: `downloads` (the
    // default), `likes`, `trending` or `newest`.
    let sort = crate::registry::HfSort::parse(params.get("sort").map(String::as_str).unwrap_or(""));
    if q.is_empty() && !params.contains_key("sort") {
        return (StatusCode::OK, Json(json!({ "models": [] })));
    }
    let limit = params
        .get("limit")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(25);
    match crate::registry::search_hf_models(q, limit, sort).await {
        Ok(models) => (StatusCode::OK, Json(json!({ "models": models }))),
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            Json(json!({ "error": format!("HuggingFace search failed: {e}") })),
        ),
    }
}

/// `GET /api/hf/repo?id=<owner>/<repo>` — the quantizations one repo offers,
/// each with its total download size and whether it is expected to run here.
///
/// The verdict is an estimate from the file size, not a fit: the exact answer
/// needs the GGUF header, which needs the file. It is here to answer "is this
/// download worth starting", which is a question that has to be answered
/// before the download, or not at all.
async fn hf_repo(
    State(state): State<S>,
    Query(params): Query<HashMap<String, String>>,
) -> impl IntoResponse {
    let Some(id) = params.get("id").map(|s| s.trim()) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "missing `id` (owner/repo)" })),
        );
    };
    // Validated before it reaches a URL: this string is interpolated into the
    // Hub path, and the caller is a browser.
    if !crate::registry::is_plausible_repo_id(id) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "`id` must be owner/repo" })),
        );
    }
    // Two Hub documents, fetched together: the file tree, and the model card
    // the architecture was parsed out of. One user action should cost one wait.
    let (contents, facts) = tokio::join!(
        crate::registry::list_hf_repo_contents(id),
        crate::registry::hf_model_facts(id),
    );
    let contents = match contents {
        Ok(c) => c,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(json!({ "error": format!("could not read {id}: {e}") })),
            );
        }
    };
    // Not being able to read the architecture is not being unable to read the
    // repo: the quantizations and their sizes are what this call is for, and
    // they are already in hand. A failure here becomes "not declared", which
    // the UI renders as a question rather than as a verdict.
    let facts = facts.unwrap_or_default();
    let arch_supported = facts
        .architecture
        .as_deref()
        .map(crate::llama_archs::is_supported);

    let vram = crate::fit::vram_bytes();
    let ram = crate::fit::system_ram_bytes();
    let quants: Vec<Value> = contents
        .quants
        .iter()
        .map(|q| {
            // The projector rides along with whatever quantization is picked,
            // so it counts towards the download and towards what is resident.
            let total = q.total_bytes + contents.mmproj_bytes;
            let verdict = crate::fit::classify_download(total, vram, ram, state.vram_reserve);
            json!({
                "label": q.label,
                "files": q.files,
                "bytes": q.total_bytes,
                "shards": q.files.len(),
                "verdict": match verdict {
                    crate::fit::RunVerdict::Fits => "fits",
                    crate::fit::RunVerdict::Tight => "tight",
                    crate::fit::RunVerdict::TooLarge => "too_large",
                    crate::fit::RunVerdict::Unknown => "unknown",
                },
                // What `eullm pull` takes, so the UI never builds this itself.
                "pull": format!("hf.co/{id}:{}", q.quant),
            })
        })
        .collect();

    (
        StatusCode::OK,
        Json(json!({
            "id": id,
            "quants": quants,
            "mmproj": contents.mmproj,
            "mmproj_bytes": contents.mmproj_bytes,
            // Echoed so the UI can say what the traffic light was judged
            // against, rather than presenting a colour with no basis.
            "vram_free_bytes": vram.map(|(f, _)| f),
            "vram_total_bytes": vram.map(|(_, t)| t),
            "ram_total_bytes": ram,
            // Three states, and the third is not the second. `null` means the
            // Hub did not say which architecture this is — not that this build
            // cannot load it. Only `architecture_supported: false` means that,
            // and only then may the UI tell a user the download is wasted.
            "architecture": facts.architecture,
            "architecture_supported": arch_supported,
            // Reported, not judged. `apache-2.0` is a name the reader knows or
            // can look up; `other` is the Hub itself saying the weights carry
            // terms of their own, with `license_name` saying which. What those
            // terms permit is between the licence and whoever runs the model —
            // our part is to put it in front of them before the download, not
            // after, and to link the text rather than summarise it.
            "license": facts.license,
            "license_name": facts.license_name,
            "license_url": facts.license_url,
        })),
    )
}

/// OpenAI-compatible routes (`/v1/*`).
pub fn openai_routes() -> Router<S> {
    Router::new()
        .route("/models", get(list_models_openai))
        .route("/chat/completions", post(chat_completions))
        .route("/embeddings", post(embeddings_openai))
        // System One-compatible decisions (TypeSafe's Jev API shape): typed
        // answers read from a decision model's logits. Under /v1 because
        // that is where System One clients look for it.
        .route(
            "/systemone",
            post(super::systemone::systemone).fallback(super::systemone::method_not_allowed),
        )
        // Feedback on those decisions, stored next to their traces. Its own
        // body limit: a valid feedback is a few kilobytes, not the 64 MB an
        // image for /api/chat may need.
        .route(
            "/systemone/feedback",
            post(super::decision_traces::feedback)
                .fallback(super::systemone::method_not_allowed)
                .layer(axum::extract::DefaultBodyLimit::max(
                    super::decision_traces::FEEDBACK_MAX_BODY_BYTES,
                )),
        )
}

// ── Model slot and dynamic swap ──────────────────────────────────────────────

/// Ensure the requested model is loaded and return a snapshot of it, with a
/// lease on the model for this request (see `resident::Lease`).
///
/// A model that is resident answers at once: finding it takes the residents'
/// read guard and nothing else. One that is not is loaded, which makes room
/// for it first (see `AppState::load_generation_model`).
///
/// A request that names no model (see [`requested_model`]) is answered by
/// `--default-model` when the server has one, as if it had named it, and
/// otherwise by the most recently used resident.
///
/// `keep_alive` is the request's own: it applies to the model that answers
/// it, from the moment the response is over.
async fn ensure_model(
    state: &AppState,
    requested: Option<&str>,
    override_batch_size: Option<usize>,
    override_ctx_size: Option<u32>,
    keep_alive: super::KeepAlive,
) -> Result<SlotSnapshot, Refusal> {
    let requested = requested.or(state.default_model.as_ref().map(|m| m.name.as_str()));
    {
        // The lease is taken under this guard: see `resident::Usage::lease`.
        let models = state.models.read().await;
        let found = match requested {
            Some(name) => models.find(name),
            None => models.most_recently_used(),
        };
        if let Some(model) = found {
            return Ok(state.lease(model, keep_alive));
        }
    }

    let Some(name) = requested else {
        return Err(no_model_loaded().into());
    };
    let loading = std::time::Instant::now();
    let mut snapshot = state
        .load_generation_model(
            name,
            override_batch_size,
            override_ctx_size,
            keep_alive,
            crate::api::EvictPolicy::Allowed,
        )
        .await
        .map_err(|e| match e {
            // A model that does not exist is a client mistake, and a
            // 5xx here is actively harmful: clients with automatic
            // retry treat it as transient and hammer a request that
            // can never succeed. Ollama answers 404 for this.
            crate::api::ModelError::NotFound(msg) => {
                (StatusCode::NOT_FOUND, Json(json!({ "error": msg }))).into()
            }
            // The model exists but would not load: out of VRAM, a
            // corrupt GGUF, a context that will not allocate. That is
            // ours, and 500 is correct.
            crate::api::ModelError::LoadFailed(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("Failed to load model '{name}': {msg}") })),
            )
                .into(),
            // Every model that could make room was busy for as long as the
            // load would wait. Nothing is wrong with the request: the same
            // one succeeds once a model finishes, which a 503 and its
            // `Retry-After` tell a client with automatic retry.
            crate::api::ModelError::Busy(msg) | crate::api::ModelError::NoRoom(msg) => Refusal {
                status: StatusCode::SERVICE_UNAVAILABLE,
                body: Json(json!({ "error": msg })),
                retry_after_secs: Some(BUSY_RETRY_AFTER_SECS),
            },
        })?;
    snapshot.load_duration = loading.elapsed();
    Ok(snapshot)
}

/// The model a request names in its `model` field; `None` when the field is
/// absent or empty, which leaves the choice to the server (`ensure_model`).
fn requested_model(body: &Value) -> Option<&str> {
    body.get("model")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty())
}

/// A duration in nanoseconds, the unit Ollama's response timings are in.
fn nanos(d: std::time::Duration) -> u64 {
    u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
}

/// `keep_alive: 0` on an empty request: unload the model it names — when it
/// names none, `--default-model`, or else the most recently used one —
/// without loading it first (`AppState::expire_model`), and say which. A
/// model that exists but is not loaded is answered as unloaded, as Ollama
/// does; with no model named and none loaded there is nothing to answer for,
/// as on the load path.
async fn unload_for_keep_alive_zero(
    state: &AppState,
    requested: Option<&str>,
) -> Result<String, Refusal> {
    let requested = requested.or(state.default_model.as_ref().map(|m| m.name.as_str()));
    match state.expire_model(requested).await {
        Ok(Some(name)) => Ok(requested.map_or(name, str::to_string)),
        Ok(None) => match requested {
            Some(name) => Ok(name.to_string()),
            None => Err(no_model_loaded().into()),
        },
        Err(crate::api::ModelError::NotFound(msg)) => {
            Err((StatusCode::NOT_FOUND, Json(json!({ "error": msg }))).into())
        }
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into()),
    }
}

/// The 503 for a request that names no model, with none loaded.
fn no_model_loaded() -> ApiError {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(
            json!({ "error": "No model loaded. Send a request with a \"model\" field, or use `eullm run <model>`." }),
        ),
    )
}

// ── "model": "auto" ──────────────────────────────────────────────────────────

/// What `"model": "auto"` made of a request, before it is answered.
enum Routing {
    /// Not routed: it names a model, or routing is not configured.
    Not,
    /// Routed, and answered already: a request with nothing to generate,
    /// which loads or unloads the candidates.
    Answered(axum::response::Response),
    /// Routed to the model the snapshot holds, with a lease on it.
    To(SlotSnapshot, super::route::RouteInfo),
}

/// Route a request `"model": "auto"` names — or that names no model, under
/// `--default-model auto` — and load the model chosen. When that one does
/// not load, the fallback answers in its place, once; when the fallback
/// does not load either, the request fails as one naming it would.
///
/// Routing reads the request as the client sent it, before any web content
/// is added. A request with nothing to generate is Ollama's way to load or,
/// with `keep_alive: 0`, unload a model: for `auto` that is every candidate
/// — loaded as far as they fit without unloading anything — answered as
/// `auto`. A raw prompt on `/api/generate` is written for one model's
/// template, and is refused.
async fn route_auto(
    state: &AppState,
    body: &Value,
    format: StreamFormat,
    user_id: Option<String>,
) -> Result<Routing, Refusal> {
    use super::route::{AUTO, RouteInfo, RouteInput, decide_route};

    let Some(table) = state.router.as_ref() else {
        return Ok(Routing::Not);
    };
    let routed = match requested_model(body) {
        Some(name) => name.eq_ignore_ascii_case(AUTO),
        None => state.default_auto,
    };
    if !routed {
        return Ok(Routing::Not);
    }

    let keep_alive = super::parse_keep_alive(body.get("keep_alive"));
    let input = match format {
        StreamFormat::OllamaGenerate => RouteInput::Generate {
            prompt: body.get("prompt").and_then(Value::as_str).unwrap_or(""),
        },
        _ => match RouteInput::of(body) {
            Some(chat @ RouteInput::Chat { .. }) => chat,
            _ => RouteInput::Chat {
                messages: &[],
                tools: Vec::new(),
            },
        },
    };
    let nothing_to_generate = format != StreamFormat::OpenAI
        && match &input {
            RouteInput::Generate { prompt } => prompt.is_empty(),
            RouteInput::Chat { messages, .. } => messages.is_empty(),
        };
    if nothing_to_generate {
        let reason = if keep_alive == super::KeepAlive::Immediate {
            for candidate in &table.candidates {
                if let Err(e) = state.expire_model(Some(&candidate.name)).await {
                    tracing::warn!(
                        "Unloading {}: {e}",
                        crate::audit::sanitize_for_log(&candidate.name)
                    );
                }
            }
            "unload"
        } else {
            state.warm_route_candidates(table, keep_alive).await;
            "load"
        };
        return Ok(Routing::Answered(
            Json(nothing_generated(format, AUTO, reason)).into_response(),
        ));
    }
    if format == StreamFormat::OllamaGenerate
        && body.get("raw").and_then(Value::as_bool) == Some(true)
    {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": "a raw prompt is written for one model's template and cannot be routed: \
                          name the model it is for instead of \"auto\""
            })),
        )
            .into());
    }

    let overrides = parse_slot_overrides(body)?;
    let (contexts, resident) = state.route_candidates(table, overrides).await;
    let route = decide_route(state, table, &input, &contexts, resident, user_id, false).await;
    let mut info = RouteInfo::new(&route, table);
    let (batch_size, ctx_size) = overrides;
    match ensure_model(state, Some(&route.model), batch_size, ctx_size, keep_alive).await {
        Ok(snapshot) => Ok(Routing::To(snapshot, info)),
        Err(refusal) if route.model != route.fallback => {
            let why = refusal.body.0["error"]
                .as_str()
                .unwrap_or("it did not load")
                .to_string();
            tracing::warn!(
                "Auto routing chose {}, which did not load ({why}); the fallback {} answers",
                crate::audit::sanitize_for_log(&route.model),
                crate::audit::sanitize_for_log(&route.fallback)
            );
            info.fell_back(&route.fallback, why);
            let snapshot = ensure_model(
                state,
                Some(&route.fallback),
                batch_size,
                ctx_size,
                keep_alive,
            )
            .await?;
            Ok(Routing::To(snapshot, info))
        }
        Err(refusal) => Err(refusal),
    }
}

/// Answer a generation request with `answer`, routed first when it asks for
/// `auto` (see [`route_auto`]): `answer` gets the model chosen, with a lease
/// on it, and an audit context carrying the route, and its response gets
/// the route's headers — before its body, so a streaming client has them
/// first. A request that is not routed reaches `answer` with no model, to
/// find its own.
async fn answer_routed<F, Fut>(
    state: S,
    identity: super::Identity,
    body: Value,
    format: StreamFormat,
    answer: F,
) -> Result<axum::response::Response, Refusal>
where
    F: FnOnce(S, AuditCtx, Option<SlotSnapshot>, Value) -> Fut,
    Fut: std::future::Future<Output = Result<axum::response::Response, Refusal>>,
{
    let mut audit = AuditCtx::of(&identity);
    let snapshot = match route_auto(&state, &body, format, audit.user_id.clone()).await? {
        Routing::Not => None,
        Routing::Answered(response) => return Ok(response),
        Routing::To(snapshot, route) => {
            audit.route = Some(route);
            Some(snapshot)
        }
    };
    let headers = audit.route.as_ref().map(super::route::RouteInfo::headers);
    let mut response = answer(state, audit, snapshot, body).await?;
    for (name, value) in headers.into_iter().flatten() {
        response.headers_mut().insert(name, value);
    }
    Ok(response)
}

/// The answer to an Ollama request with nothing to generate, which only
/// loaded or unloaded `model`: `done_reason` says which.
fn nothing_generated(format: StreamFormat, model: &str, done_reason: &str) -> Value {
    let mut answer = json!({
        "model": model,
        "created_at": chrono::Utc::now().to_rfc3339(),
        "done": true,
        "done_reason": done_reason,
    });
    if format == StreamFormat::OllamaGenerate {
        answer["response"] = json!("");
    } else {
        answer["message"] = json!({ "role": "assistant", "content": "" });
    }
    answer
}

/// Parsed sampling parameters from the API request.
struct SamplingParams {
    max_tokens: u32,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    min_p: f32,
    repeat_penalty: f32,
    repeat_last_n: i32,
    seed: Option<u32>,
    num_ctx: Option<u32>,
    /// `cache_prompt` (see `GenerateRequest::cache_prompt`), true unless a
    /// request sets it to false.
    cache_prompt: bool,
}

/// Sampling parameters of a request to `/api/generate` or `/api/chat`.
fn parse_generate_params(body: &Value) -> SamplingParams {
    parse_sampling_params(body, false)
}

/// Sampling parameters of a `/v1/chat/completions` request: those of
/// [`parse_generate_params`], with the output limit also read from
/// `max_completion_tokens`, the name OpenAI's Chat Completions API gives it
/// now.
fn parse_chat_completions_params(body: &Value) -> SamplingParams {
    parse_sampling_params(body, true)
}

/// Read a request's sampling parameters. `with_max_completion_tokens` says
/// whether that field names the output limit too: it does on
/// `/v1/chat/completions` only, Ollama's API having no such field.
fn parse_sampling_params(body: &Value, with_max_completion_tokens: bool) -> SamplingParams {
    // Check top-level first (OpenAI format), then Ollama's options object.
    let options = body.get("options");

    let get = |key: &str| -> Option<&Value> {
        body.get(key).or_else(|| options.and_then(|o| o.get(key)))
    };

    // OpenAI's Chat Completions API names the output limit
    // `max_completion_tokens` and keeps `max_tokens` only as a deprecated
    // alias, which older clients still send, so a request carrying both is
    // limited by the current name. A `null` is how JSON leaves a field
    // unset, and must not discard a `max_tokens` the same request did set.
    // Whichever name is used, its value is then read in the same way.
    let limit = with_max_completion_tokens
        .then(|| get("max_completion_tokens"))
        .flatten()
        .filter(|v| !v.is_null())
        .or_else(|| get("max_tokens"))
        .or_else(|| get("num_predict"));
    // Ollama's real default is unbounded (-1: generate until context is
    // full or a stop condition), not a small fixed cap — confirmed against
    // Ollama's own docs/source, see ollama/ollama#7691. Leaving this
    // unbounded here is safe because prefill_sequence() always clamps it to
    // the remaining context budget regardless of what's requested.
    let max_tokens = limit
        .and_then(|v| v.as_u64())
        .unwrap_or(u32::MAX as u64) as u32;
    // Defaults match Ollama: temperature=0.8, top_k=40, top_p=0.9, repeat_penalty=1.1
    let temperature = get("temperature").and_then(|v| v.as_f64()).unwrap_or(0.8) as f32;
    let top_k = get("top_k").and_then(|v| v.as_i64()).unwrap_or(40) as i32;
    let top_p = get("top_p").and_then(|v| v.as_f64()).unwrap_or(0.9) as f32;
    let min_p = get("min_p").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
    let repeat_penalty = get("repeat_penalty")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.1) as f32;
    let repeat_last_n = get("repeat_last_n").and_then(|v| v.as_i64()).unwrap_or(64) as i32;
    let seed = get("seed").and_then(|v| v.as_u64()).map(|v| v as u32);
    let num_ctx = get("num_ctx").and_then(|v| v.as_u64()).map(|v| v as u32);
    // llama.cpp's server reads it at the top level; an Ollama client passes
    // what it does not know in `options`. Anything but `false` keeps reuse.
    let cache_prompt = get("cache_prompt").and_then(|v| v.as_bool()) != Some(false);

    tracing::info!(
        "Request params: max_tokens={max_tokens}, temp={temperature:.2}, top_k={top_k}, top_p={top_p:.2}, repeat_penalty={repeat_penalty:.2}, num_ctx={num_ctx:?}{}",
        if cache_prompt { "" } else { ", cache_prompt=false" }
    );
    SamplingParams {
        max_tokens,
        temperature,
        top_k,
        top_p,
        min_p,
        repeat_penalty,
        repeat_last_n,
        seed,
        num_ctx,
        cache_prompt,
    }
}

fn is_streaming(body: &Value) -> bool {
    body.get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

/// Parse the `format` field from an Ollama-style request.
///
/// When `format` is `"json"`, returns the built-in JSON GBNF grammar so that
/// constrained decoding forces the model to produce valid JSON — matching
/// Ollama's behavior.
fn parse_format_grammar(body: &Value) -> Option<String> {
    let fmt = body.get("format").and_then(|v| v.as_str())?;
    if fmt == "json" {
        Some(JSON_GBNF.to_string())
    } else {
        None
    }
}

// ── Multimodal helpers ─────────────────────────────────────────────────
// Attachments arrive the Ollama way, as `images: [<base64>]` on a message
// (audio rides the same field; the engine tells the two apart by their
// bytes). Every message's attachments count, not only the latest turn's: a
// follow-up question about a photo sent two turns ago must still see the
// photo. It used to be the latest turn's only, and the model — asked "and is
// that the tongue?" one turn later — reasoned about a picture it no longer
// had, from its own earlier description, without saying so. Ollama reads
// images from every message it keeps too (`chatPrompt` in its
// `server/prompt.go`), so a client that re-sends them is following the
// contract, not relying on an extension. Where the two differ is what gives
// way when the context runs short: Ollama drops whole messages from the
// start, while here the oldest attachments go first and the text stays (see
// `InferenceEngine::generate_multimodal`).

/// Every attachment in a chat conversation, in the order the prompt carries
/// them.
#[derive(Debug, Default, PartialEq)]
struct ChatMedia {
    /// The decoded bytes of each attachment, oldest first.
    items: Vec<Vec<u8>>,
    /// How many of `items` each message carries, indexed like the messages.
    per_message: Vec<usize>,
    /// How many of the newest `items` belong to the latest user message —
    /// the attachments the current question is about.
    current_turn: usize,
}

/// Decode the `images` of every message. Accepts raw base64 and the
/// `data:...;base64,...` form, with line breaks or other whitespace inside
/// it ignored, as Ollama's decoder does.
///
/// An attachment that does not decode is an error that names it. It used to
/// make the request go through as text instead, and the model answered that
/// it could see no image — true, and no help in finding the broken upload.
fn collect_chat_media(messages: &[Value]) -> Result<ChatMedia, String> {
    use base64::{Engine as _, engine::general_purpose::STANDARD};
    let last_user = messages
        .iter()
        .rposition(|m| m.get("role").and_then(Value::as_str) == Some("user"));
    let mut media = ChatMedia::default();
    for (i, message) in messages.iter().enumerate() {
        let images = message
            .get("images")
            .and_then(Value::as_array)
            .map_or(&[][..], Vec::as_slice);
        for (j, image) in images.iter().enumerate() {
            let text = image
                .as_str()
                .ok_or_else(|| format!("messages[{i}].images[{j}] is not a string"))?;
            // `data:image/jpeg;base64,XXX` → keep only the payload after the comma.
            let payload: Vec<u8> = text
                .rsplit(',')
                .next()
                .unwrap_or(text)
                .bytes()
                .filter(|b| !b.is_ascii_whitespace())
                .collect();
            let bytes = STANDARD
                .decode(&payload)
                .map_err(|e| format!("messages[{i}].images[{j}] is not valid base64 ({e})"))?;
            if bytes.is_empty() {
                return Err(format!("messages[{i}].images[{j}] is empty"));
            }
            media.items.push(bytes);
        }
        media.per_message.push(images.len());
        if Some(i) == last_user {
            media.current_turn = images.len();
        }
    }
    Ok(media)
}

/// Replace every attachment with the note the engine writes for one it had
/// to drop, and take the `images` off the messages.
///
/// For a conversation whose attachments are all in earlier turns, now running
/// on a model that cannot read them — the user switched to a text model
/// halfway through. Refusing would fail every turn from there on; dropping
/// the attachments silently would leave "what is in this photo?" in the
/// history with nothing next to it.
fn media_as_notes(mut messages: Vec<Value>, media: &ChatMedia) -> Vec<Value> {
    let mut items = media.items.iter();
    for (message, &count) in messages.iter_mut().zip(&media.per_message) {
        if count == 0 {
            continue;
        }
        let mut content: String = items
            .by_ref()
            .take(count)
            .map(|bytes| {
                let note = crate::inference::dropped_media_note(
                    crate::inference::media_looks_like_audio(bytes),
                );
                format!("{note}\n")
            })
            .collect();
        content.push_str(message.get("content").and_then(Value::as_str).unwrap_or(""));
        if let Some(fields) = message.as_object_mut() {
            fields.remove("images");
            fields.insert("content".to_string(), Value::String(content));
        }
    }
    messages
}

/// The error for a turn whose own attachments the loaded model cannot read.
fn cannot_read_media(model_name: &str) -> (StatusCode, Json<Value>) {
    if cfg!(feature = "multimodal") {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "error": format!(
                    "`{model_name}` has no multimodal projector, so it cannot read images or audio. \
                     Either use a model that ships one — `eullm pull gemma-4-e4b` fetches \
                     the weights and the projector together — or point at a projector you \
                     already have with `--mmproj <path>`. A projector found next to the \
                     weights as `mmproj*.gguf` is picked up on its own."
                )
            })),
        )
    } else {
        // A build without the `multimodal` feature used to drop an `images`
        // array on the floor and answer as text. The model, asked about a
        // picture it never received, says it cannot see one — which reads as
        // a limitation of the model rather than of the binary. Reported from
        // a source build made with `--features vulkan` (issue #286), where the
        // omission is easy: the published binaries all carry multimodal, a
        // hand-built one need not.
        (
            StatusCode::NOT_IMPLEMENTED,
            Json(json!({
                "error": "This build has no multimodal support, so the attached media cannot be read. \
                          The published binaries include it; a build from source needs \
                          `--features multimodal` (combine it with a backend, e.g. \
                          `--features \"vulkan,multimodal\"`)."
            })),
        )
    }
}

/// The (role, content) pairs a template renders for a multimodal request:
/// every message, with one `marker` per attachment at the head of the
/// message that carried it.
///
/// A marker already in a message's text is removed first. The engine pairs
/// markers with attachments by position, so one typed or pasted into a
/// message would shift every attachment after it onto the wrong turn, or
/// fail the request on a count mismatch.
#[cfg(feature = "multimodal")]
fn with_media_markers(
    messages: &[Value],
    per_message: &[usize],
    marker: &str,
) -> Vec<(String, String)> {
    messages
        .iter()
        .zip(per_message)
        .map(|(message, &count)| {
            let role = message
                .get("role")
                .and_then(Value::as_str)
                .unwrap_or("user");
            let text = message
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or("")
                .replace(marker, "");
            let content = format!("{}{text}", format!("{marker}\n").repeat(count));
            (role.to_string(), content)
        })
        .collect()
}

/// Build the prompt for a multimodal conversation, preferring the model's own
/// GGUF-embedded Jinja template — the same choice [`build_chat_prompt`] makes
/// for text.
///
/// The mtmd media markers go inside the messages' **content** (see
/// [`with_media_markers`]), before the template renders, so the model
/// receives the turn markers it was actually trained on: `<|im_start|>user`
/// for a Qwen VL, `<start_of_turn>user` for Gemma. This is also how
/// llama-server does it.
///
/// It was Gemma's markers in every case until now, and the reason is worth
/// recording because it is the divergence `engine/CLAUDE.md` warns about. The
/// multimodal branch landed two days before `build_chat_prompt` learned to
/// render the embedded template, and it returns early — above that call — so
/// it never picked the change up. Gemma 4 was the only multimodal model in the
/// catalog, so nothing looked wrong until a Qwen VL arrived and was handed
/// `<start_of_turn>` tokens it has never seen. A wrong template does not
/// error; it degrades the answer quietly, which is the worst way to be wrong.
///
/// Returns the prompt together with the stop sequences that belong to it —
/// none when the embedded template rendered (its EOG token ends generation,
/// the contract `build_chat_prompt` already uses), Gemma's when we fell back
/// — and the response prefix, as `build_chat_prompt` does.
///
/// The whole conversation is rendered. Only the latest user turn used to be,
/// so every turn after the one carrying a picture reached the model without
/// the picture and without anything said before it.
#[cfg(feature = "multimodal")]
fn multimodal_chat_prompt(
    engine: &InferenceEngine,
    messages: &[(&str, &str)],
    think: bool,
) -> (String, Vec<String>, String) {
    if let Some(dynamic) = engine.apply_jinja_chat_template(messages, think) {
        return (dynamic.prompt, Vec::new(), dynamic.preopened);
    }
    let gemma = crate::chat_template::ChatTemplate::Gemma;
    (
        gemma.build_prompt(messages, think),
        gemma.stop_sequences(),
        String::new(),
    )
}

/// Background mtmd-aware generation, mirroring `sequential_to_channel`.
/// `pinned` is how many of the newest `media` belong to the current turn —
/// see [`InferenceEngine::generate_multimodal`].
#[cfg(feature = "multimodal")]
fn multimodal_to_channel(
    engine: Arc<InferenceEngine>,
    request: GenerateRequest,
    media: Vec<Vec<u8>>,
    pinned: usize,
) -> mpsc::Receiver<StreamEvent> {
    let (tx, rx) = mpsc::channel::<StreamEvent>(64);
    tokio::task::spawn_blocking(move || {
        engine.generate_multimodal(&request, &media, pinned, tx);
    });
    rx
}

#[cfg(test)]
mod chat_media_tests {
    use super::*;
    use crate::inference::{DROPPED_AUDIO_NOTE, DROPPED_IMAGE_NOTE};
    use base64::{Engine as _, engine::general_purpose::STANDARD};

    const JPEG: &[u8] = &[0xFF, 0xD8, 0xFF, 0xE0, 1, 2, 3];
    const WAV: &[u8] = b"RIFF\x24\x00\x00\x00WAVE";

    fn b64(bytes: &[u8]) -> String {
        STANDARD.encode(bytes)
    }

    #[test]
    fn attachments_from_every_turn_are_collected_in_order() {
        let messages = vec![
            json!({ "role": "system", "content": "be brief" }),
            json!({ "role": "user", "content": "what is this?", "images": [b64(JPEG)] }),
            json!({ "role": "assistant", "content": "a dog" }),
            json!({
                "role": "user",
                "content": "and this?",
                "images": [b64(WAV), format!("data:image/jpeg;base64,{}", b64(JPEG))],
            }),
        ];
        let media = collect_chat_media(&messages).unwrap();
        assert_eq!(
            media.items,
            vec![JPEG.to_vec(), WAV.to_vec(), JPEG.to_vec()]
        );
        assert_eq!(media.per_message, vec![0, 1, 0, 2]);
        assert_eq!(media.current_turn, 2);
    }

    // The turn this was written for: "and is that thread the tongue?" has no
    // attachment of its own, and the photo it asks about came a turn earlier.
    #[test]
    fn a_follow_up_keeps_the_earlier_photo_and_pins_nothing_of_its_own() {
        let messages = vec![
            json!({ "role": "user", "content": "what is this?", "images": [b64(JPEG)] }),
            json!({ "role": "assistant", "content": "a dog" }),
            json!({ "role": "user", "content": "is that thread the tongue?" }),
        ];
        let media = collect_chat_media(&messages).unwrap();
        assert_eq!(media.items, vec![JPEG.to_vec()]);
        assert_eq!(media.current_turn, 0);
    }

    // A client continuing a reply it prefilled ends on an assistant message;
    // the attachments the question is about are still the user's.
    #[test]
    fn the_current_turn_is_the_last_user_message_not_the_last_message() {
        let messages = vec![
            json!({ "role": "user", "content": "what is this?", "images": [b64(JPEG)] }),
            json!({ "role": "assistant", "content": "It is" }),
        ];
        let media = collect_chat_media(&messages).unwrap();
        assert_eq!(media.current_turn, 1);
    }

    #[test]
    fn a_conversation_without_attachments_has_none() {
        let messages = vec![
            json!({ "role": "user", "content": "hi" }),
            json!({ "role": "user", "content": "x", "images": [] }),
        ];
        let media = collect_chat_media(&messages).unwrap();
        assert!(media.items.is_empty());
        assert_eq!(media.per_message, vec![0, 0]);
        assert_eq!(media.current_turn, 0);
    }

    // Line-wrapped base64 is what some clients send, and Ollama's decoder
    // skips the line breaks.
    #[test]
    fn whitespace_inside_the_base64_is_ignored() {
        let encoded = b64(&[7u8; 100]);
        let wrapped = format!("{}\n{}\r\n", &encoded[..60], &encoded[60..]);
        let messages = [json!({ "role": "user", "content": "", "images": [wrapped] })];
        let media = collect_chat_media(&messages).unwrap();
        assert_eq!(media.items, vec![vec![7u8; 100]]);
    }

    // It used to send the request on as text, and the model said it could
    // see no image.
    #[test]
    fn an_attachment_that_does_not_decode_is_an_error_naming_it() {
        let messages = vec![
            json!({ "role": "user", "content": "a", "images": [b64(JPEG)] }),
            json!({ "role": "user", "content": "b", "images": [b64(JPEG), "not base64!"] }),
        ];
        let err = collect_chat_media(&messages).unwrap_err();
        assert!(
            err.starts_with("messages[1].images[1] is not valid base64"),
            "{err}"
        );
        let err = collect_chat_media(&[json!({ "role": "user", "images": [""] })]).unwrap_err();
        assert_eq!(err, "messages[0].images[0] is empty");
        let err = collect_chat_media(&[json!({ "role": "user", "images": [42] })]).unwrap_err();
        assert_eq!(err, "messages[0].images[0] is not a string");
    }

    #[test]
    fn a_model_that_cannot_read_them_gets_notes_in_their_place() {
        let messages = vec![
            json!({ "role": "user", "content": "what is this?", "images": [b64(JPEG), b64(WAV)] }),
            json!({ "role": "assistant", "content": "a dog barking" }),
            json!({ "role": "user", "content": "thanks" }),
        ];
        let media = collect_chat_media(&messages).unwrap();
        let notes = media_as_notes(messages, &media);
        assert_eq!(
            notes,
            vec![
                json!({
                    "role": "user",
                    "content": format!("{DROPPED_IMAGE_NOTE}\n{DROPPED_AUDIO_NOTE}\nwhat is this?"),
                }),
                json!({ "role": "assistant", "content": "a dog barking" }),
                json!({ "role": "user", "content": "thanks" }),
            ]
        );
    }

    #[cfg(feature = "multimodal")]
    #[test]
    fn every_message_is_rendered_with_one_marker_per_attachment() {
        let messages = vec![
            json!({ "role": "system", "content": "be brief" }),
            json!({ "role": "user", "content": "what is this?", "images": [b64(JPEG)] }),
            json!({ "role": "assistant", "content": "a dog" }),
            json!({ "role": "user", "content": "compare", "images": [b64(JPEG), b64(JPEG)] }),
        ];
        let media = collect_chat_media(&messages).unwrap();
        let marked = with_media_markers(&messages, &media.per_message, "<M>");
        let expected: Vec<(String, String)> = [
            ("system", "be brief"),
            ("user", "<M>\nwhat is this?"),
            ("assistant", "a dog"),
            ("user", "<M>\n<M>\ncompare"),
        ]
        .iter()
        .map(|(role, content)| (role.to_string(), content.to_string()))
        .collect();
        assert_eq!(marked, expected);
    }

    // The engine pairs markers with attachments by position: one arriving in
    // a message's own text would hand the next attachment to the wrong turn.
    #[cfg(feature = "multimodal")]
    #[test]
    fn a_marker_typed_into_a_message_cannot_claim_an_attachment() {
        let messages = vec![
            json!({ "role": "user", "content": "what does <M> mean?" }),
            json!({ "role": "user", "content": "and this <M>", "images": [b64(JPEG)] }),
        ];
        let media = collect_chat_media(&messages).unwrap();
        let marked = with_media_markers(&messages, &media.per_message, "<M>");
        let markers: usize = marked.iter().map(|(_, c)| c.matches("<M>").count()).sum();
        assert_eq!(markers, media.items.len());
        assert_eq!(marked[0].1, "what does  mean?");
        assert_eq!(marked[1].1, "<M>\nand this ");
    }

    // The fallback was a `format!` of exactly this until it had to render
    // whole conversations; for one turn it must not change by a byte.
    #[test]
    fn the_gemma_fallback_for_a_single_turn_is_unchanged() {
        let marked = "<__media__>\nwhat is this?";
        assert_eq!(
            crate::chat_template::ChatTemplate::Gemma.build_prompt(&[("user", marked)], true),
            format!("<start_of_turn>user\n{marked}<end_of_turn>\n<start_of_turn>model\n")
        );
    }
}

/// Build the prompt and stop sequences for a chat request.
///
/// Tries the model's own GGUF-embedded Jinja template first, via
/// [`InferenceEngine::apply_jinja_chat_template`] — the same mechanism
/// llama-server uses by default, and the only correct choice for models
/// whose real template doesn't match any hardcoded family we guess from the
/// name (found on `gemma-4-12b-q8`, which turns out to use a GPT-OSS-style
/// reasoning-channel template, not Gemma's `<start_of_turn>` format at all).
///
/// Works on both loading paths: the sequential engine exposes it directly,
/// and the continuous-batching scheduler exposes it through
/// [`SchedulerHandle::apply_jinja_chat_template`] (a weak reference to the
/// decode thread's model — see `SharedModel` in the scheduler). The
/// batching path used to be excluded and silently fell back to the
/// hardcoded per-family template, which is how QwQ-32B-Preview — ChatML by
/// name-detection, but trained with its own default system turn — answered
/// off distribution on the web path while `--cli` (sequential at the time)
/// was fine.
///
/// Falls back to the hardcoded template whenever the dynamic one isn't
/// available: the GGUF has no embedded template at all, rendering it
/// failed, or the model is mid-swap. No hardcoded stop sequence is added on
/// the dynamic path — the model's own end-of-generation token ends the turn
/// regardless of which template built the prompt, and a marker like
/// `<|end|>` belongs to a different template, not this one.
///
/// The third element is what the answer is sent starting with
/// ([`GenerateRequest::response_prefix`]): the reasoning block the dynamic
/// template opened at the end of the prompt, if it opened one. The
/// hardcoded templates never do.
fn build_chat_prompt(
    snap: &SlotSnapshot,
    messages: &[Value],
    think: bool,
    model_name: &str,
) -> (String, Vec<String>, String) {
    {
        let pairs: Vec<(&str, &str)> = messages
            .iter()
            .map(|msg| {
                let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");
                let content = msg.get("content").and_then(|v| v.as_str()).unwrap_or("");
                (role, content)
            })
            .collect();
        let dynamic = if let Some(engine) = snap.engine.as_ref() {
            engine.apply_jinja_chat_template(&pairs, think)
        } else if let Some(scheduler) = snap.scheduler.as_ref() {
            scheduler.apply_jinja_chat_template(&pairs, think)
        } else {
            None
        };
        if let Some(dynamic) = dynamic {
            return (dynamic.prompt, Vec::new(), dynamic.preopened);
        }
    }

    let template = crate::chat_template::ChatTemplate::detect(model_name);
    let prompt = format_chat_prompt(messages, think, model_name);
    let mut stop_sequences = template.stop_sequences();
    stop_sequences.push("<|end|>".to_string());
    (prompt, stop_sequences, String::new())
}

/// Format chat messages into a prompt string using the model-appropriate template.
/// When `think` is false, suppresses Qwen3 thinking mode (ChatML only).
fn format_chat_prompt(messages: &[Value], think: bool, model_name: &str) -> String {
    let template = crate::chat_template::ChatTemplate::detect(model_name);
    let pairs: Vec<(&str, &str)> = messages
        .iter()
        .map(|msg| {
            let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");
            let content = msg.get("content").and_then(|v| v.as_str()).unwrap_or("");
            (role, content)
        })
        .collect();
    template.build_prompt(&pairs, think)
}

/// Collect all tokens from a receiver into a final result.
async fn collect_stream(mut rx: mpsc::Receiver<StreamEvent>) -> Result<Collected, String> {
    let mut text = String::new();
    let mut tokens_generated = 0u32;
    let mut tokens_prompt = 0u32;
    let mut duration_ms = 0u64;
    // If the stream ends without a Done event the answer is incomplete, so
    // Length is the honest default rather than Stop.
    let mut stop_reason = StopReason::Length;
    let mut stats = AnswerStats::default();

    while let Some(event) = rx.recv().await {
        match event {
            StreamEvent::Token(piece) => text.push_str(&piece),
            StreamEvent::Done {
                tokens_generated: tg,
                tokens_prompt: tp,
                duration_ms: d,
                stop_reason: sr,
                stats: st,
            } => {
                tokens_generated = tg;
                tokens_prompt = tp;
                duration_ms = d;
                stop_reason = sr;
                stats = st;
                break;
            }
            StreamEvent::Error(e) => return Err(e),
        }
    }

    Ok(Collected {
        text,
        tokens_generated,
        tokens_prompt,
        duration_ms,
        stop_reason,
        stats,
    })
}

/// What a non-streaming request produced, including *why* it stopped.
struct Collected {
    text: String,
    tokens_generated: u32,
    tokens_prompt: u32,
    duration_ms: u64,
    stop_reason: StopReason,
    stats: AnswerStats,
}

// ── Web content injection ────────────────────────────────────────────────────

/// If web browsing is enabled and the last user message contains URLs,
/// fetch each URL and inject the content as a system-style message
/// prepended to the conversation (before the last user turn).
///
/// Returns a new messages Vec with injected content, or the original unchanged.
async fn inject_web_content(
    mut messages: Vec<Value>,
    web_enabled: bool,
    ctx_size: u32,
    batch_size: usize,
    web_policy: &crate::tools::guard::WebPolicy,
) -> Vec<Value> {
    if !web_enabled {
        return messages;
    }
    // With continuous batching each slot gets ctx_size/batch_size tokens.
    // Use per-slot context as the budget so injected content actually fits.
    let effective_ctx = if batch_size > 1 {
        ctx_size / batch_size as u32
    } else {
        ctx_size
    };

    // Find last user message
    let last_user_idx = messages
        .iter()
        .rposition(|m| m.get("role").and_then(|v| v.as_str()) == Some("user"));
    let idx = match last_user_idx {
        Some(i) => i,
        None => return messages,
    };

    let user_text = messages[idx]
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let urls = tools::extract_urls(&user_text);
    if urls.is_empty() {
        return messages;
    }

    // Estimate prompt chars already used (rough: all messages joined)
    let existing_chars: usize = messages
        .iter()
        .map(|m| {
            m.get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .len()
        })
        .sum();

    let mut injections: Vec<String> = Vec::new();
    for url in &urls {
        match tools::fetch_for_context(url, effective_ctx, existing_chars, &user_text, web_policy)
            .await
        {
            Ok((content, truncated)) => {
                let note = if truncated {
                    " [content truncated to fit context]"
                } else {
                    ""
                };
                injections.push(format!("[Web content from {url}{note}]\n\n{content}"));
                tracing::info!(
                    "Web: fetched {} ({} chars{})",
                    crate::audit::sanitize_for_log(url),
                    content.len(),
                    if truncated { ", truncated" } else { "" }
                );
            }
            Err(e) => {
                injections.push(format!("[Failed to fetch {url}: {e}]"));
                tracing::warn!(
                    "Web: fetch failed for {}: {e}",
                    crate::audit::sanitize_for_log(url)
                );
            }
        }
    }

    if injections.is_empty() {
        return messages;
    }

    // Instruction-tuned models (Gemma 4 12B observed) say "I can't browse the
    // web" while at the same time summarising the page just handed to them.
    // Tell them up-front that the fetch already happened on their behalf, so
    // the response is consistent with what they actually have in context.
    let preamble = "\
You have a web browsing capability provided by the inference engine. When the \
user shares a URL, the engine automatically fetches the page and gives you its \
plain-text content below. Treat the content below as pages YOU navigated and \
read. Do NOT claim you cannot browse the web — you just did. You may cite the \
source URL when answering.\n\n---\n\n";

    let web_message = json!({
        "role": "system",
        "content": format!("{preamble}{}", injections.join("\n\n---\n\n"))
    });
    messages.insert(idx, web_message);
    messages
}

// ── EULLM API handlers ──────────────────────────────────────────────────────

async fn version(State(state): State<S>) -> Json<Value> {
    // `api_port` is an EULLM extension: the chat UI (served on its own port)
    // reads it to display the canonical API endpoint. Ollama clients only
    // look at `version` and ignore unknown fields.
    Json(json!({
        "version": env!("CARGO_PKG_VERSION"),
        "api_port": state.api_port,
        // EULLM extension: how many times a model was evicted to make VRAM
        // room for another slot (generation <-> embedding or decision). See
        // `AppState::cross_slot_evictions`.
        "model_swaps": state
            .cross_slot_evictions
            .load(std::sync::atomic::Ordering::Relaxed),
        // EULLM extension: `--max-loaded-models`, how many generation models
        // are resident now, and how many were unloaded to make room for
        // another (`AppState::generation_evictions`).
        "max_loaded_models": state.max_loaded_models,
        "loaded_models": state.models.read().await.len(),
        "generation_evictions": state
            .generation_evictions
            .load(std::sync::atomic::Ordering::Relaxed),
    }))
}

/// `DELETE /api/delete` — Ollama's endpoint of the name: remove a model's
/// files from the models folder, as `eullm rm` does. The body names it in
/// `model` (Ollama's current field) or `name` (its older one).
///
/// 404 when no model of that name is in the folder; 403 for a model in a
/// library (`EULLM_MODEL_LIBRARIES`), whose files EuLLM only reads; 409 for
/// a model in memory, which is unloaded first (`/api/unload`) so that no
/// request it is answering loses its weights.
async fn delete_model(State(state): State<S>, Json(body): Json<Value>) -> Result<Json<Value>, ApiError> {
    let refuse = |status: StatusCode, msg: String| (status, Json(json!({ "error": msg })));
    let name = body
        .get("model")
        .or_else(|| body.get("name"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|n| !n.is_empty())
        .ok_or_else(|| refuse(StatusCode::BAD_REQUEST, "name the model to delete in `model`".to_string()))?
        .to_string();
    let in_store = state.store.get(&name).ok().flatten();
    if in_store.is_none() {
        if state.store.library_model(&name).is_some() {
            return Err(refuse(
                StatusCode::FORBIDDEN,
                format!(
                    "'{name}' is a file EuLLM uses where it is (in a model library, or put in the models \
                     folder by hand) and never deletes: delete the file yourself"
                ),
            ));
        }
        return Err(refuse(StatusCode::NOT_FOUND, format!("model '{name}' not found")));
    }
    let loaded = state
        .models
        .read()
        .await
        .by_recent_use()
        .iter()
        .any(|m| crate::api::model_names_match(&m.name, &name));
    if loaded {
        return Err(refuse(
            StatusCode::CONFLICT,
            format!("'{name}' is in memory: unload it first (POST /api/unload)"),
        ));
    }
    let removed = state
        .store
        .delete(&name)
        .map_err(|e| refuse(StatusCode::INTERNAL_SERVER_ERROR, format!("could not delete '{name}': {e}")))?;
    tracing::info!("Deleted model {} from the models folder", crate::audit::sanitize_for_log(&name));
    Ok(Json(json!({ "deleted": name, "freed_bytes": removed.map(|r| r.freed) })))
}

/// Unload generation models, freeing their VRAM, without loading a
/// replacement. EULLM extension (not part of the Ollama API) — the primary
/// use case is handing GPU memory to a co-resident process (e.g. an
/// embedding server used during RAG document ingestion) without restarting
/// eullm. Send a request with a `model` field afterwards (or run `eullm run
/// <model>` again) to load a model back in.
///
/// An optional `{"model": "..."}` body unloads that model only, leaving
/// every other resident; without one, every generation model goes. Either
/// way now: requests still running on an unloaded model are cut off.
/// `unloaded` names the first model unloaded, or is null — a string, which
/// `eullm unload` from older releases reads — and `unloaded_all` lists them
/// all.
async fn unload_model(
    State(state): State<S>,
    body: axum::body::Bytes,
) -> Result<Json<Value>, ApiError> {
    let requested = if body.iter().all(u8::is_ascii_whitespace) {
        None
    } else {
        let body: Value = serde_json::from_slice(&body).map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": format!("invalid JSON body: {e}") })),
            )
        })?;
        match body.get("model") {
            None | Some(Value::Null) => None,
            Some(Value::String(name)) => Some(name.clone()),
            Some(_) => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(json!({ "error": "\"model\" must be a string" })),
                ));
            }
        }
    };
    let unloaded: Vec<String> = match &requested {
        Some(name) => state.unload_named(name).await.into_iter().collect(),
        None => state.unload_all().await,
    };
    Ok(Json(match unloaded.first() {
        Some(first) => json!({ "unloaded": first, "unloaded_all": unloaded }),
        None => json!({
            "unloaded": null,
            "unloaded_all": [],
            "message": match &requested {
                Some(name) => format!("'{name}' was not loaded"),
                None => "no model was loaded".to_string(),
            },
        }),
    }))
}

/// `POST /api/route` — EuLLM extension: the model `"model": "auto"` would
/// give this request, and why, without generating anything or loading any
/// generation model. The body is the one `/api/chat`, `/api/generate` or
/// `/v1/chat/completions` takes. The answer adds what the decision model
/// read, the state and the question in `/v1/systemone`'s shape, so that the
/// same question, worded another way, can be asked about the very state the
/// server built. Audited as any routing, marked as a dry run.
async fn route_request(
    State(state): State<S>,
    axum::Extension(identity): axum::Extension<super::Identity>,
    Json(body): Json<Value>,
) -> Result<Json<super::route::RouteResponse>, ApiError> {
    let Some(table) = state.router.as_ref() else {
        return Err((
            StatusCode::NOT_FOUND,
            Json(json!({
                "error": "auto routing is not configured: start the server with two or more \
                          --auto-model to choose between"
            })),
        ));
    };
    let input = super::route::RouteInput::of(&body).ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": "the body needs \"messages\" or \"prompt\", as for /api/chat, \
                          /v1/chat/completions or /api/generate"
            })),
        )
    })?;
    let overrides = parse_slot_overrides(&body)?;
    let (contexts, resident) = state.route_candidates(table, overrides).await;
    let route = super::route::decide_route(
        &state,
        table,
        &input,
        &contexts,
        resident,
        AuditCtx::of(&identity).user_id,
        true,
    )
    .await;
    Ok(Json(super::route::RouteResponse::new(route, table)))
}

/// `input`, in either shape the two embedding endpoints accept: one string,
/// or an array of strings. Ollama and OpenAI both accept both forms.
fn parse_embedding_input(body: &Value) -> Result<Vec<String>, (StatusCode, Json<Value>)> {
    let bad_input = || {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "\"input\" must be a string or an array of strings" })),
        )
    };
    match body.get("input") {
        Some(Value::String(s)) => Ok(vec![s.clone()]),
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| v.as_str().map(str::to_string).ok_or_else(bad_input))
            .collect(),
        _ => Err(bad_input()),
    }
}

/// `POST /api/embed` — Ollama-compatible.
///
/// Request: `{"model": "...", "input": "text" | ["text", ...], "keep_alive": ...}`.
/// Response: `{"model": "...", "embeddings": [[...], ...], "total_duration": ns,
/// "load_duration": ns, "prompt_eval_count": tokens}` — see `ollama_embed_response`.
async fn embed(
    State(state): State<S>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let started = std::time::Instant::now();
    let requested = body.get("model").and_then(|v| v.as_str()).ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "\"model\" is required" })),
        )
    })?;
    let inputs = parse_embedding_input(&body)?;
    let n_ctx = body
        .get("options")
        .and_then(|o| o.get("num_ctx"))
        .and_then(|v| v.as_u64())
        .map(|v| v as u32)
        .unwrap_or(DEFAULT_EMBEDDING_CTX);

    let keep_alive = super::parse_keep_alive(body.get("keep_alive"));
    let loading = std::time::Instant::now();
    let model = state
        .ensure_embedding_model(requested, n_ctx)
        .await
        .map_err(embedding_model_error)?;
    let load_duration = loading.elapsed();
    state.touch_embedding_slot(keep_alive).await;

    let embedded = embed_on_a_blocking_thread(model, inputs).await?;

    Ok(Json(ollama_embed_response(
        requested,
        embedded,
        load_duration,
        started.elapsed(),
    )))
}

/// The `/api/embed` response, in Ollama's shape: the vectors, the tokens
/// read as `prompt_eval_count`, and the durations in nanoseconds —
/// `load_duration` the time spent getting the model into its slot (close to
/// nothing when it was already there), `total_duration` the whole request.
fn ollama_embed_response(
    model: &str,
    embedded: Embedded,
    load_duration: std::time::Duration,
    total_duration: std::time::Duration,
) -> Value {
    let nanos = |d: std::time::Duration| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX);
    json!({
        "model": model,
        "embeddings": embedded.vectors,
        "total_duration": nanos(total_duration),
        "load_duration": nanos(load_duration),
        "prompt_eval_count": embedded.prompt_tokens,
    })
}

/// `POST /v1/embeddings` — OpenAI-compatible.
///
/// Request: `{"model": "...", "input": "text" | ["text", ...]}`.
/// Response: `{"object": "list", "data": [{"object":"embedding","embedding":[...],"index":N}, ...], "model": "...", "usage": {...}}`.
async fn embeddings_openai(
    State(state): State<S>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let requested = body.get("model").and_then(|v| v.as_str()).ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "\"model\" is required" })),
        )
    })?;
    let inputs = parse_embedding_input(&body)?;
    let n_ctx = body
        .get("options")
        .and_then(|o| o.get("num_ctx"))
        .and_then(|v| v.as_u64())
        .map(|v| v as u32)
        .unwrap_or(DEFAULT_EMBEDDING_CTX);

    let keep_alive = super::parse_keep_alive(body.get("keep_alive"));
    let model = state
        .ensure_embedding_model(requested, n_ctx)
        .await
        .map_err(embedding_model_error)?;
    state.touch_embedding_slot(keep_alive).await;

    let embedded = embed_on_a_blocking_thread(model, inputs).await?;

    Ok(Json(openai_embeddings_response(requested, embedded)))
}

/// `EmbeddingModel::embed` on tokio's blocking pool, where the generation
/// and decision handlers already run their model calls. `embed` is a forward
/// pass per input — 5.7 s for a 631-token passage on a 4-core CPU — and
/// called straight from the handler it held one of the runtime's worker
/// threads, of which there is one per core, for all of it: whatever else
/// that worker had queued waited, and as many long requests as there are
/// cores left no worker for anything at all. Four 1,162-token requests at
/// once on a 4-core CPU kept `/api/version` from answering for up to 25.7 s;
/// on the blocking pool it answers within 35 ms while they run.
///
/// At most [`EMBEDDING_SLOTS`] run at once; the others wait for a slot
/// without holding a thread.
async fn embed_on_a_blocking_thread(
    model: Arc<EmbeddingModel>,
    inputs: Vec<String>,
) -> Result<Embedded, (StatusCode, Json<Value>)> {
    let failed = |e: String| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("Embedding failed: {e}") })),
        )
    };
    let slot = EMBEDDING_SLOTS
        .acquire()
        .await
        .map_err(|e| failed(e.to_string()))?;
    tokio::task::spawn_blocking(move || {
        // Held until the embedding is done, not until the handler is: a
        // client that disconnects does not free a slot its request is still
        // running in.
        let _slot = slot;
        model.embed(&inputs)
    })
    .await
    .map_err(|e| failed(e.to_string()))?
    .map_err(failed)
}

/// How many embedding requests may take a blocking thread at once: one per
/// core, the number that could when each one held a runtime worker thread;
/// the blocking pool alone would allow hundreds. They wait there for the
/// embedder, which embeds one input at a time in the one context it keeps
/// (`inference::embedding`): running more would not be faster, as each
/// input already uses every core, or the one GPU.
static EMBEDDING_SLOTS: LazyLock<tokio::sync::Semaphore> = LazyLock::new(|| {
    tokio::sync::Semaphore::new(std::thread::available_parallelism().map_or(4, |n| n.get()))
});

/// The `/v1/embeddings` response, in OpenAI's shape.
///
/// `usage` counts the tokens the model read, after truncation to the
/// embedder's context. It used to be a fixed 0 because `embed` did not
/// return a count, which is exactly what a client billing or rate-limiting
/// by `usage` cannot tell apart from an empty request. An embedding request
/// generates nothing, so the total is the prompt.
fn openai_embeddings_response(model: &str, embedded: Embedded) -> Value {
    let data: Vec<Value> = embedded
        .vectors
        .into_iter()
        .enumerate()
        .map(|(i, v)| json!({ "object": "embedding", "embedding": v, "index": i }))
        .collect();
    json!({
        "object": "list",
        "data": data,
        "model": model,
        "usage": {
            "prompt_tokens": embedded.prompt_tokens,
            "total_tokens": embedded.prompt_tokens,
        },
    })
}

/// `ModelError` → the same 404-vs-500 split `ensure_model` uses (see its
/// comment): a model that does not exist is a client mistake, one that
/// exists but would not load is ours.
fn embedding_model_error(e: crate::api::ModelError) -> (StatusCode, Json<Value>) {
    match e {
        crate::api::ModelError::NotFound(msg) => {
            (StatusCode::NOT_FOUND, Json(json!({ "error": msg })))
        }
        crate::api::ModelError::LoadFailed(msg) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": msg })),
        ),
        crate::api::ModelError::Busy(msg) | crate::api::ModelError::NoRoom(msg) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({ "error": msg })),
        ),
    }
}

#[cfg(test)]
mod embedding_input_tests {
    use super::*;

    #[test]
    fn a_single_string_becomes_one_input() {
        let body = json!({ "input": "hello" });
        assert_eq!(parse_embedding_input(&body).unwrap(), vec!["hello".to_string()]);
    }

    #[test]
    fn an_array_of_strings_is_preserved_in_order() {
        let body = json!({ "input": ["a", "b", "c"] });
        assert_eq!(
            parse_embedding_input(&body).unwrap(),
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn a_missing_input_field_is_rejected() {
        let body = json!({ "model": "bge" });
        assert!(parse_embedding_input(&body).is_err());
    }

    #[test]
    fn a_non_string_array_element_is_rejected() {
        let body = json!({ "input": ["a", 5, "c"] });
        assert!(parse_embedding_input(&body).is_err());
    }

    #[test]
    fn a_wrong_shaped_input_is_rejected() {
        let body = json!({ "input": 42 });
        assert!(parse_embedding_input(&body).is_err());
    }

    fn two_vectors() -> Embedded {
        Embedded {
            vectors: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            prompt_tokens: 7,
        }
    }

    // `usage` was a fixed 0 whatever the request held.
    #[test]
    fn the_openai_usage_counts_the_tokens_read() {
        let r = openai_embeddings_response("m", two_vectors());
        assert_eq!(r["usage"]["prompt_tokens"], 7);
        assert_eq!(r["usage"]["total_tokens"], 7);
        assert_eq!(r["data"][1]["index"], 1);
        assert_eq!(r["data"][1]["embedding"], json!([0.0, 1.0]));
    }

    // Ollama's own client reads these three next to the vectors.
    #[test]
    fn api_embed_reports_the_same_count_and_its_durations() {
        let r = ollama_embed_response(
            "m",
            two_vectors(),
            std::time::Duration::from_millis(2),
            std::time::Duration::from_millis(30),
        );
        assert_eq!(r["prompt_eval_count"], 7);
        assert_eq!(r["load_duration"], 2_000_000);
        assert_eq!(r["total_duration"], 30_000_000);
        assert_eq!(r["embeddings"], json!([[1.0, 0.0], [0.0, 1.0]]));
    }
}

/// List models — returns the loaded models (like Ollama) plus catalog entries.
///
/// Ollama's `/api/tags` returns all locally available models.  We return the
/// loaded models first, the most recently used first (so health-check
/// dashboards see them, and the chat UI preselects the one in use), followed
/// by catalog entries for discoverability.
async fn list_models(State(state): State<S>) -> Json<Value> {
    let loaded: Vec<String> = state
        .models
        .read()
        .await
        .by_recent_use()
        .iter()
        .map(|m| m.name.clone())
        .collect();

    // The loaded models first (this is what dashboards check).
    // What the store or a library knows of it; a model loaded from a path
    // has neither. Its quantization is not guessed: this said Q4_K_M for
    // every such model, whatever it was.
    let mut models: Vec<Value> = loaded
        .iter()
        .map(|name| {
            let known = state.store.get_any(name).ok().flatten();
            json!({
                "name": name,
                "size": known.as_ref().map_or(0, |m| m.size_bytes),
                "digest": known.as_ref().map_or_else(String::new, |m| m.digest.clone()),
                "loaded": true,
                "downloaded": true,
                "details": {
                    "format": "gguf",
                    "family": known.as_ref().map_or_else(String::new, |m| m.base.clone()),
                    "parameter_size": "",
                    "quantization_level": "",
                    "source": known.as_ref().map(|m| if m.status == "library" { "library" } else { "store" }),
                }
            })
        })
        .collect();

    // Add catalog entries (skip duplicates if the loaded model is in the catalog).
    // The Ollama-compatible `name` field MUST be the addressable id — clients
    // pass it back as the `model` field in chat/generate requests, and that's
    // the only string the model store / picker knows how to resolve. The human
    // catalog name is exposed alongside as `details.display_name` for UIs that
    // want to show it.
    for m in EU_CATALOG.iter() {
        if let Some(i) = loaded.iter().position(|n| *n == m.name || *n == m.id) {
            // Replace the placeholder entry above with full catalog metadata.
            // `loaded` must survive that replacement: it is the only thing in
            // the response that says which model is in the slot. Clients used
            // to infer it from the empty digest of the placeholder, and this
            // line overwrites that with the catalog's real digest — so a
            // loaded *catalog* model looked exactly like one that had never
            // been downloaded, while a loaded raw `.gguf` path (which never
            // reaches this branch) looked correct. That is why the chat UI
            // reported "No model loaded" only after picking from the picker.
            models[i] = json!({
                "name": m.id,
                "size": m.size_bytes,
                "digest": m.digest,
                "loaded": true,
                "downloaded": true,
                "details": {
                    "format": "gguf",
                    "family": m.base(),
                    "parameter_size": format!("{:.1}B", m.params_b),
                    "quantization_level": m.quantization,
                    "domain": m.domain,
                    "source_model": m.source_model(),
                    "display_name": m.name,
                }
            });
            continue;
        }
        // Whether the weights are on disk. Ollama has no equivalent field
        // because its `/api/tags` lists only what has been pulled; ours lists
        // the whole catalog, and without this a client cannot tell a model it
        // can run right now from one that would first download several
        // gigabytes. The chat UI disabled every catalog entry for exactly that
        // reason, which made a downloaded-but-not-loaded model unselectable
        // even though the server swaps to it on request.
        models.push(json!({
            "name": m.id,
            "size": m.size_bytes,
            "digest": m.digest,
            "downloaded": state.store.is_present(&m.id),
            "details": {
                "format": "gguf",
                "family": m.base(),
                "parameter_size": format!("{:.1}B", m.params_b),
                "quantization_level": m.quantization,
                "domain": m.domain,
                "source_model": m.source_model(),
                "display_name": m.name,
            }
        }));
    }

    // Models on disk that the catalog does not know about — anything pulled
    // from a URL or a HuggingFace repo. Same omission as `/v1/models` had: the
    // list was assembled from the catalog and the loaded slot, so a model the
    // user had deliberately downloaded was missing from it unless it happened
    // to be loaded at that moment.
    let already: std::collections::HashSet<String> = models
        .iter()
        .filter_map(|m| m.get("name").and_then(|v| v.as_str()).map(String::from))
        .collect();
    for m in state.store.list_all().unwrap_or_default() {
        if m.id.is_empty() || already.contains(&m.id) {
            continue;
        }
        // EuLLM extension: where the files are. A library's are read only
        // (`EULLM_MODEL_LIBRARIES`), so a client offers no delete for them.
        let source = if m.status == "library" { "library" } else { "store" };
        models.push(json!({
            "name": m.id,
            "size": m.size_bytes,
            "digest": m.digest,
            "downloaded": state.store.is_present(&m.id),
            "details": {
                "format": "gguf",
                "family": m.base,
                "parameter_size": "",
                "quantization_level": "",
                "display_name": m.name,
                "source": source,
            }
        }));
    }

    // `auto`, when routing is configured: a name a client picks like any
    // model's, answered by one of the candidates. Last, so that a client
    // which takes the first model it is offered keeps the one it took.
    if let Some(table) = &state.router {
        let candidates: Vec<&str> = table.candidates.iter().map(|c| c.name.as_str()).collect();
        models.push(json!({
            "name": super::route::AUTO,
            "size": 0,
            "digest": "",
            "downloaded": true,
            "details": {
                "format": "",
                "family": "eullm-router",
                "parameter_size": "",
                "quantization_level": "",
                "display_name": format!("auto ({})", candidates.join(", ")),
                "candidates": candidates,
            }
        }));
    }

    Json(json!({ "models": models }))
}

/// One model as `GET /api/ps` lists it — what `running_model_entry` needs,
/// gathered from whichever slot holds it.
struct RunningModel {
    name: String,
    /// Its weights (and, for a generation model, its projector and KV cache).
    size: u64,
    /// The VRAM it holds, measured at load or estimated.
    size_vram: u64,
    size_vram_measured: bool,
    /// Tokens of context a request to it gets.
    context_length: u32,
    /// Until it is unloaded for being idle; `None`: never.
    expires_in: Option<std::time::Duration>,
    /// Since it was last used.
    idle_for: std::time::Duration,
    digest: String,
    family: String,
    parameter_size: String,
    quantization_level: String,
    /// `generation`, `embedding` or `decision`.
    slot: &'static str,
    in_flight: usize,
    batch_size: Option<usize>,
    gpu_layers: Option<i32>,
    reserved_companion: bool,
    launch: bool,
}

/// A model in Ollama's `/api/ps` shape, as of `now`, with what EuLLM adds
/// under `eullm`. A model nothing will unload expires so far ahead it never
/// comes — `null` is not what Ollama sends, and its clients parse a date.
fn running_model_entry(m: &RunningModel, now: chrono::DateTime<chrono::Utc>) -> Value {
    let at = |from_now: Option<std::time::Duration>| {
        from_now
            .and_then(|d| chrono::TimeDelta::from_std(d).ok())
            .and_then(|d| now.checked_add_signed(d))
            .unwrap_or_else(|| now + chrono::TimeDelta::nanoseconds(i64::MAX))
            .to_rfc3339()
    };
    let since = chrono::TimeDelta::from_std(m.idle_for).unwrap_or(chrono::TimeDelta::zero());
    json!({
        "name": m.name,
        "model": m.name,
        "size": m.size,
        "digest": m.digest,
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": m.family,
            "families": if m.family.is_empty() { Vec::new() } else { vec![m.family.clone()] },
            "parameter_size": m.parameter_size,
            "quantization_level": m.quantization_level,
        },
        "expires_at": at(m.expires_in),
        "size_vram": m.size_vram,
        "context_length": m.context_length,
        "eullm": {
            "slot": m.slot,
            "in_flight": m.in_flight,
            "last_used": (now - since).to_rfc3339(),
            "batch_size": m.batch_size,
            "gpu_layers": m.gpu_layers,
            "reserved_companion": m.reserved_companion,
            "launch": m.launch,
            "size_vram_measured": m.size_vram_measured,
        },
    })
}

/// `GET /api/monitor` — what the chat UI's Monitor page shows: the requests
/// running (oldest first) and the last ones answered (newest first), and the
/// machine's samples of the last two minutes (oldest first). The models in
/// memory are `/api/ps`. In memory only: see [`crate::monitor`].
async fn monitor() -> Json<Value> {
    let (active, history) = crate::monitor::requests();
    let system = tokio::task::spawn_blocking(crate::monitor::system::snapshot)
        .await
        .unwrap_or_default();
    Json(json!({ "active": active, "history": history, "system": system }))
}

/// `GET /api/about` — what the chat UI's About page shows: this build, the
/// machine and the devices llama.cpp can use, where models are kept, and
/// the server's protections: whether API keys are required (never the keys)
/// and which networks are admitted without one. Behind the same perimeter
/// as the rest of `/api`, so it tells nothing to someone not let in.
async fn about(State(state): State<S>) -> Json<Value> {
    let machine = tokio::task::spawn_blocking(crate::monitor::system::machine)
        .await
        .ok();
    let (models_dir, models_source) = state.store.root_with_source();
    let (libraries, _) = state.store.libraries_with_source();
    Json(json!({
        "version": env!("CARGO_PKG_VERSION"),
        "commit": env!("EULLM_GIT_HASH"),
        "backend": crate::inference::backend_label(),
        "llama_cpp": env!("EULLM_LLAMA_CPP"),
        "release_asset": Some(env!("EULLM_RELEASE_ASSET")).filter(|a| !a.is_empty()),
        "cpu_features": crate::inference::cpu_features_summary().trim(),
        "machine": machine,
        "models_dir": models_dir,
        "models_dir_source": models_source,
        "model_libraries": libraries,
        "audit_log": crate::audit::AuditLogger::new().log_path(),
        "api_port": state.api_port,
        "protections": {
            "api_keys": state.api_keys.is_enabled(),
            "allowed_ips": state.ip_allowlist.describe(),
            "allowed_ips_source": state.ip_allowlist.source(),
        },
    }))
}

/// `GET /api/ps` — the models in memory, Ollama's endpoint of the name: every
/// generation model, the most recently used first, then the embedding and
/// decision models, which Ollama lists too.
async fn list_running(State(state): State<S>) -> Json<Value> {
    let now = std::time::Instant::now();
    // A build with no GPU backend puts nothing on a GPU, whatever was asked.
    let on_gpu = crate::inference::has_gpu_backend();
    let mut running: Vec<RunningModel> = state
        .models
        .read()
        .await
        .by_recent_use()
        .into_iter()
        .map(|m| {
            let usage = m.usage.view();
            let facts = &m.facts;
            RunningModel {
                name: m.name.clone(),
                size: facts.size_bytes,
                size_vram: if on_gpu { facts.vram_bytes() } else { 0 },
                size_vram_measured: facts.size_vram.is_some(),
                // What one request gets: the context is split between a
                // scheduler's slots.
                context_length: facts.ctx_size / facts.batch_size.max(1) as u32,
                expires_in: usage.expires_in(now),
                idle_for: now.saturating_duration_since(usage.last_used),
                digest: String::new(),
                family: facts.family.clone().unwrap_or_default(),
                parameter_size: String::new(),
                quantization_level: String::new(),
                slot: "generation",
                in_flight: usage.in_flight,
                batch_size: Some(facts.batch_size),
                gpu_layers: Some(if on_gpu { facts.gpu_layers } else { 0 }),
                reserved_companion: false,
                launch: m.launch,
            }
        })
        .collect();

    let tokio_now = tokio::time::Instant::now();
    let left = |deadline: Option<tokio::time::Instant>| {
        deadline.map(|d| d.saturating_duration_since(tokio_now))
    };
    let embedding = state.embedding.read().await.as_ref().map(|slot| {
        (
            slot.model_name.clone(),
            slot.model.weights_bytes(),
            slot.model.largest_context(),
            slot.is_reserved_companion,
        )
    });
    if let Some((name, weights, context, reserved)) = embedding {
        let expires_in = left(*state.embedding_deadline.lock().await);
        running.push(companion(
            "embedding",
            name,
            weights,
            context,
            reserved,
            expires_in,
            on_gpu,
        ));
    }
    let decision = state.decision.read().await.as_ref().map(|slot| {
        (
            slot.model_name.clone(),
            slot.model.weights_bytes(),
            slot.model.info().context_tokens as u32,
            slot.is_reserved_companion,
        )
    });
    if let Some((name, weights, context, reserved)) = decision {
        let expires_in = left(*state.decision_deadline.lock().await);
        running.push(companion(
            "decision", name, weights, context, reserved, expires_in, on_gpu,
        ));
    }

    // What the store and the catalog know about each, read with no lock held.
    for m in &mut running {
        if let Ok(Some(manifest)) = state.store.get(&m.name) {
            m.digest = manifest.digest;
            if m.family.is_empty() {
                m.family = manifest.base;
            }
        }
        if let Some(entry) = crate::models::catalog::find_model(&m.name) {
            if m.digest.is_empty() {
                m.digest = entry.digest.clone();
            }
            if m.family.is_empty() {
                m.family = entry.base();
            }
            m.parameter_size = format!("{:.1}B", entry.params_b);
            m.quantization_level = entry.quantization.clone();
        }
    }
    let wall = chrono::Utc::now();
    Json(json!({
        "models": running.iter().map(|m| running_model_entry(m, wall)).collect::<Vec<_>>(),
    }))
}

/// An embedding or decision model as `/api/ps` lists it: loaded whole onto
/// the GPU when this build has one, in RAM otherwise (see
/// `EmbeddingModel::load`), so its weights are its VRAM or none of it.
fn companion(
    slot: &'static str,
    name: String,
    weights: u64,
    context_length: u32,
    reserved_companion: bool,
    expires_in: Option<std::time::Duration>,
    on_gpu: bool,
) -> RunningModel {
    RunningModel {
        name,
        size: weights,
        size_vram: if on_gpu { weights } else { 0 },
        size_vram_measured: false,
        context_length,
        expires_in,
        idle_for: std::time::Duration::ZERO,
        digest: String::new(),
        family: String::new(),
        parameter_size: String::new(),
        quantization_level: String::new(),
        slot,
        in_flight: 0,
        batch_size: None,
        gpu_layers: None,
        reserved_companion,
        launch: false,
    }
}

async fn generate(
    State(state): State<S>,
    // Present on every request: the auth middleware inserts an anonymous
    // identity when no keys are configured, so this never fails to extract.
    axum::Extension(identity): axum::Extension<super::Identity>,
    Json(body): Json<Value>,
) -> Result<axum::response::Response, Refusal> {
    answer_routed(
        state,
        identity,
        body,
        StreamFormat::OllamaGenerate,
        generate_with,
    )
    .await
}

/// `/api/generate`, answered by `routed` when the request was routed to
/// it, and otherwise by the model it names.
async fn generate_with(
    state: S,
    audit: AuditCtx,
    routed: Option<SlotSnapshot>,
    body: Value,
) -> Result<axum::response::Response, Refusal> {
    let requested = requested_model(&body);
    let (override_batch_size, override_ctx_size) = parse_slot_overrides(&body)?;
    let keep_alive = super::parse_keep_alive(body.get("keep_alive"));
    let prompt = body
        .get("prompt")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    // An empty prompt with `keep_alive: 0` is Ollama's way to unload one
    // model: that model goes, and only it, without being loaded first.
    if prompt.is_empty() && keep_alive == super::KeepAlive::Immediate {
        let model = unload_for_keep_alive_zero(&state, requested).await?;
        let answer = nothing_generated(StreamFormat::OllamaGenerate, &model, "unload");
        return Ok(Json(answer).into_response());
    }

    let snap = match routed {
        Some(snap) => snap,
        None => {
            ensure_model(
                &state,
                requested,
                override_batch_size,
                override_ctx_size,
                keep_alive,
            )
            .await?
        }
    };
    let model = snap.model_name.clone();

    // An empty prompt is Ollama's documented way to load a model (or apply a
    // `keep_alive` to one already loaded) without generating anything —
    // `ensure_model` above already did the loading/swap, so there is nothing
    // left to do. Returning here, before `GenerateRequest` is built, is what
    // makes this genuinely free: it never reaches the scheduler or engine.
    // The lease ends with it, which is what applies the `keep_alive`.
    if prompt.is_empty() {
        let answer = nothing_generated(StreamFormat::OllamaGenerate, &model, "load");
        return Ok(Json(answer).into_response());
    }

    let sp = parse_generate_params(&body);
    let raw = body.get("raw").and_then(|v| v.as_bool()).unwrap_or(false);
    let split_thinking = super::thinking::wants_thinking(&body);
    let grammar = if raw {
        // GBNF grammar sampling is incompatible with raw mode — the grammar
        // sampler crashes (GGML_ASSERT) when the prompt contains pre-tokenized
        // special tokens (ChatML).  In raw mode, the caller handles formatting.
        None
    } else {
        parse_format_grammar(&body)
    };

    let request = GenerateRequest {
        prompt,
        max_tokens: sp.max_tokens,
        temperature: sp.temperature,
        top_k: sp.top_k,
        top_p: sp.top_p,
        min_p: sp.min_p,
        repeat_penalty: sp.repeat_penalty,
        repeat_last_n: sp.repeat_last_n,
        seed: sp.seed,
        num_ctx: sp.num_ctx,
        grammar,
        raw,
        cache_prompt: sp.cache_prompt,
        ..Default::default()
    };

    if let Some(ref sched) = snap.scheduler {
        // ── Continuous batching path ────────────────────────────────
        let rx = sched.try_submit(request)?;
        let rx = crate::monitor::watch(rx, &snap.model_name, "generate");

        if is_streaming(&body) {
            Ok(ndjson_stream_response(
                rx,
                model,
                StreamFormat::OllamaGenerate,
                audit,
                snap.lease,
                snap.load_duration,
                split_thinking,
            ))
        } else {
            let Collected {
                text,
                tokens_generated,
                tokens_prompt,
                duration_ms,
                stop_reason,
                stats,
            } = collect_stream(rx).await.map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": e })),
                )
            })?;

            audit.log(
                &model,
                StreamFormat::OllamaGenerate,
                tokens_prompt,
                tokens_generated,
                duration_ms,
                &stats,
            );

            let (response, thinking) = generate_answer(text, split_thinking);
            let mut answer = json!({
                "model": model,
                "created_at": chrono::Utc::now().to_rfc3339(),
                "response": response,
                "done": true,
                "done_reason": stop_reason.as_api_str(),
                "total_duration": duration_ms * 1_000_000 + nanos(snap.load_duration),
                "load_duration": nanos(snap.load_duration),
                "prompt_eval_count": tokens_prompt,
                "prompt_eval_duration": nanos(stats.prompt_time),
                "eval_count": tokens_generated,
                "eval_duration": nanos(stats.eval_time)
            });
            if let Some(thinking) = thinking {
                answer["thinking"] = Value::String(thinking);
            }
            Ok(Json(audit.extended(with_drafts(answer, &stats))).into_response())
        }
    } else {
        // ── Sequential fallback ────────────────────────────────────
        let engine = Arc::clone(snap.engine.as_ref().unwrap());

        if is_streaming(&body) {
            let rx = sequential_to_channel(engine, request);
            let rx = crate::monitor::watch(rx, &snap.model_name, "generate");
            Ok(ndjson_stream_response(
                rx,
                model,
                StreamFormat::OllamaGenerate,
                audit,
                snap.lease,
                snap.load_duration,
                split_thinking,
            ))
        } else {
            let tracker = crate::monitor::begin(&snap.model_name, "generate");
            let result = tokio::task::spawn_blocking({
                let engine = engine;
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

            tracker.answered(&result);
            audit.log(
                &model,
                StreamFormat::OllamaGenerate,
                result.tokens_prompt,
                result.tokens_generated,
                result.duration_ms,
                &result.stats,
            );

            let (response, thinking) = generate_answer(result.text, split_thinking);
            let mut answer = json!({
                "model": model,
                "created_at": chrono::Utc::now().to_rfc3339(),
                "response": response,
                "done": true,
                "done_reason": result.stop_reason.as_api_str(),
                "total_duration": result.duration_ms * 1_000_000 + nanos(snap.load_duration),
                "load_duration": nanos(snap.load_duration),
                "prompt_eval_count": result.tokens_prompt,
                "prompt_eval_duration": nanos(result.stats.prompt_time),
                "eval_count": result.tokens_generated,
                "eval_duration": nanos(result.stats.eval_time)
            });
            if let Some(thinking) = thinking {
                answer["thinking"] = Value::String(thinking);
            }
            Ok(Json(audit.extended(with_drafts(answer, &result.stats))).into_response())
        }
    }
}

async fn chat(
    State(state): State<S>,
    // Present on every request: the auth middleware inserts an anonymous
    // identity when no keys are configured, so this never fails to extract.
    axum::Extension(identity): axum::Extension<super::Identity>,
    Json(body): Json<Value>,
) -> Result<axum::response::Response, Refusal> {
    answer_routed(state, identity, body, StreamFormat::OllamaChat, chat_with).await
}

/// `/api/chat`, answered by `routed` when the request was routed to it, and
/// otherwise by the model it names.
async fn chat_with(
    state: S,
    audit: AuditCtx,
    routed: Option<SlotSnapshot>,
    body: Value,
) -> Result<axum::response::Response, Refusal> {
    let requested = requested_model(&body);
    let (override_batch_size, override_ctx_size) = parse_slot_overrides(&body)?;
    let keep_alive = super::parse_keep_alive(body.get("keep_alive"));
    let messages = body
        .get("messages")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let split_thinking = super::thinking::wants_thinking(&body);

    // The unload counterpart, as on `/api/generate`.
    if messages.is_empty() && keep_alive == super::KeepAlive::Immediate {
        let model = unload_for_keep_alive_zero(&state, requested).await?;
        let answer = nothing_generated(StreamFormat::OllamaChat, &model, "unload");
        return Ok(Json(answer).into_response());
    }

    let snap = match routed {
        Some(snap) => snap,
        None => {
            ensure_model(
                &state,
                requested,
                override_batch_size,
                override_ctx_size,
                keep_alive,
            )
            .await?
        }
    };
    let model = snap.model_name.clone();

    // Same warm-load shape as `/api/generate` with an empty prompt: an empty
    // `messages` array asks only for the load (and, with it, a `keep_alive`)
    // to happen. Ollama-specific — `/v1/chat/completions` (OpenAI) has no
    // such convention and requires non-empty messages, so this stays out of
    // `chat_completions`.
    if messages.is_empty() {
        let answer = nothing_generated(StreamFormat::OllamaChat, &model, "load");
        return Ok(Json(answer).into_response());
    }

    let messages = inject_web_content(
        messages,
        state.web_enabled,
        state.ctx_size,
        state.batch_size,
        &state.web_policy,
    )
    .await;

    // ── Attachments ────────────────────────────────────────────────────
    // A conversation carrying any goes through the sequential mtmd path,
    // whole. `load_generation_model` forces sequential mode when the loaded model has an
    // mmproj, so a model that can read them has `snap.engine` with a
    // projector in it. One that cannot is refused when the current turn has
    // its own attachments — the question is about them — and otherwise gets
    // the conversation with each attachment replaced by a note.
    let media = collect_chat_media(&messages)
        .map_err(|e| (StatusCode::BAD_REQUEST, Json(json!({ "error": e }))))?;
    let projector = snap
        .engine
        .as_ref()
        .filter(|engine| !media.items.is_empty() && engine.has_projector())
        .map(Arc::clone);
    let messages = if media.items.is_empty() || projector.is_some() {
        messages
    } else if media.current_turn > 0 {
        return Err(cannot_read_media(&snap.model_name).into());
    } else {
        tracing::info!(
            "`{}` cannot read attachments: {} from earlier turns become notes",
            crate::audit::sanitize_for_log(&snap.model_name),
            media.items.len()
        );
        media_as_notes(messages, &media)
    };

    #[cfg(feature = "multimodal")]
    if let Some(engine) = projector {
        let sp = parse_generate_params(&body);
        // Parsed here rather than reused from the text path below: the
        // multimodal branch returns before that code runs.
        let mm_think = body.get("think").and_then(|v| v.as_bool()).unwrap_or(true);
        let marked = with_media_markers(
            &messages,
            &media.per_message,
            llama_cpp_2::mtmd::mtmd_default_marker(),
        );
        let pairs: Vec<(&str, &str)> = marked
            .iter()
            .map(|(role, content)| (role.as_str(), content.as_str()))
            .collect();
        let (mm_prompt, mm_stops, mm_prefix) = multimodal_chat_prompt(&engine, &pairs, mm_think);
        let mm_request = GenerateRequest {
            prompt: mm_prompt,
            max_tokens: sp.max_tokens,
            temperature: sp.temperature,
            top_k: sp.top_k,
            top_p: sp.top_p,
            min_p: sp.min_p,
            repeat_penalty: sp.repeat_penalty,
            repeat_last_n: sp.repeat_last_n,
            seed: sp.seed,
            num_ctx: sp.num_ctx,
            // The prompt is already templated, with the mtmd markers in place —
            // do NOT let generate() add its own BOS/template on top.
            raw: true,
            // Empty when the model's own template rendered: its EOG token ends
            // generation. See `multimodal_chat_prompt`.
            stop_sequences: mm_stops,
            // think-aware, like the text path: a model that can be asked not to
            // think can also emit a think tag that has to be stripped.
            filter_sequences: crate::inference::default_filters(mm_think),
            grammar: None,
            cache_prompt: sp.cache_prompt,
            response_prefix: mm_prefix,
        };
        if is_streaming(&body) {
            let rx = multimodal_to_channel(engine, mm_request, media.items, media.current_turn);
            let rx = crate::monitor::watch(rx, &snap.model_name, "chat");
            return Ok(ndjson_stream_response(
                rx,
                model,
                StreamFormat::OllamaChat,
                audit,
                snap.lease,
                snap.load_duration,
                split_thinking,
            ));
        }
        let rx = multimodal_to_channel(engine, mm_request, media.items, media.current_turn);
        let rx = crate::monitor::watch(rx, &snap.model_name, "chat");
        let Collected {
            text,
            tokens_generated,
            tokens_prompt,
            duration_ms,
            stop_reason,
            stats,
        } = collect_stream(rx).await.map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e })),
            )
        })?;
        audit.log(
            &model,
            StreamFormat::OllamaChat,
            tokens_prompt,
            tokens_generated,
            duration_ms,
            &stats,
        );
        return Ok(Json(audit.extended(with_drafts(
            json!({
                "model": model,
                "created_at": chrono::Utc::now().to_rfc3339(),
                "message": chat_message(text, split_thinking),
                "done": true,
                "done_reason": stop_reason.as_api_str(),
                "total_duration": duration_ms * 1_000_000 + nanos(snap.load_duration),
                "load_duration": nanos(snap.load_duration),
                "prompt_eval_count": tokens_prompt,
                "prompt_eval_duration": nanos(stats.prompt_time),
                "eval_count": tokens_generated,
                "eval_duration": nanos(stats.eval_time)
            }),
            &stats,
        )))
        .into_response());
    }

    let think = body.get("think").and_then(|v| v.as_bool()).unwrap_or(true);
    let model_name_ref: &str = &model;
    let (prompt, stop_sequences, response_prefix) =
        build_chat_prompt(&snap, &messages, think, model_name_ref);
    let sp = parse_generate_params(&body);
    let grammar = parse_format_grammar(&body);

    let request = GenerateRequest {
        prompt,
        max_tokens: sp.max_tokens,
        temperature: sp.temperature,
        top_k: sp.top_k,
        top_p: sp.top_p,
        min_p: sp.min_p,
        repeat_penalty: sp.repeat_penalty,
        repeat_last_n: sp.repeat_last_n,
        seed: sp.seed,
        num_ctx: sp.num_ctx,
        stop_sequences,
        // think-aware: with think:false a think tag in the OUTPUT is
        // spurious, since the prompt already carries a closed empty block.
        filter_sequences: crate::inference::default_filters(think),
        grammar,
        raw: false,
        cache_prompt: sp.cache_prompt,
        response_prefix,
    };

    if let Some(ref sched) = snap.scheduler {
        let rx = sched.try_submit(request)?;
        let rx = crate::monitor::watch(rx, &snap.model_name, "chat");

        if is_streaming(&body) {
            Ok(ndjson_stream_response(
                rx,
                model,
                StreamFormat::OllamaChat,
                audit,
                snap.lease,
                snap.load_duration,
                split_thinking,
            ))
        } else {
            let Collected {
                text,
                tokens_generated,
                tokens_prompt,
                duration_ms,
                stop_reason,
                stats,
            } = collect_stream(rx).await.map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": e })),
                )
            })?;

            audit.log(
                &model,
                StreamFormat::OllamaChat,
                tokens_prompt,
                tokens_generated,
                duration_ms,
                &stats,
            );

            Ok(Json(audit.extended(with_drafts(
                json!({
                    "model": model,
                    "created_at": chrono::Utc::now().to_rfc3339(),
                    "message": chat_message(text, split_thinking),
                    "done": true,
                    "done_reason": stop_reason.as_api_str(),
                    "total_duration": duration_ms * 1_000_000 + nanos(snap.load_duration),
                    "load_duration": nanos(snap.load_duration),
                    "prompt_eval_count": tokens_prompt,
                    "prompt_eval_duration": nanos(stats.prompt_time),
                    "eval_count": tokens_generated,
                    "eval_duration": nanos(stats.eval_time)
                }),
                &stats,
            )))
            .into_response())
        }
    } else {
        let engine = Arc::clone(snap.engine.as_ref().unwrap());

        if is_streaming(&body) {
            let rx = sequential_to_channel(engine, request);
            let rx = crate::monitor::watch(rx, &snap.model_name, "chat");
            Ok(ndjson_stream_response(
                rx,
                model,
                StreamFormat::OllamaChat,
                audit,
                snap.lease,
                snap.load_duration,
                split_thinking,
            ))
        } else {
            let tracker = crate::monitor::begin(&snap.model_name, "chat");
            let result = tokio::task::spawn_blocking({
                let engine = engine;
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

            tracker.answered(&result);
            audit.log(
                &model,
                StreamFormat::OllamaChat,
                result.tokens_prompt,
                result.tokens_generated,
                result.duration_ms,
                &result.stats,
            );

            Ok(Json(audit.extended(with_drafts(
                json!({
                    "model": model,
                    "created_at": chrono::Utc::now().to_rfc3339(),
                    "message": chat_message(result.text, split_thinking),
                    "done": true,
                    "done_reason": result.stop_reason.as_api_str(),
                    "total_duration": result.duration_ms * 1_000_000 + nanos(snap.load_duration),
                    "load_duration": nanos(snap.load_duration),
                    "prompt_eval_count": result.tokens_prompt,
                    "prompt_eval_duration": nanos(result.stats.prompt_time),
                    "eval_count": result.tokens_generated,
                    "eval_duration": nanos(result.stats.eval_time)
                }),
                &result.stats,
            )))
            .into_response())
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
                entry.name, entry.domain, entry.base(), entry.source_model(), entry.license
            ),
            "parameters": "num_ctx 4096\ntemperature 0.7",
            "template": "{{ .Prompt }}",
            "details": {
                "format": "gguf",
                "family": entry.base(),
                "parameter_size": format!("{:.1}B", entry.params_b),
                "quantization_level": entry.quantization,
                "domain": entry.domain,
                "source_model": entry.source_model()
            }
        }))
    } else {
        Json(json!({ "error": format!("model '{name}' not found") }))
    }
}

/// `POST /api/pull` — fetch a model, streaming progress as NDJSON.
///
/// Ollama's shape, because Ollama clients are the ones that call this:
/// `{"status": "..."}` lines while it works, `{"status","digest","total",
/// "completed"}` while bytes move, `{"status":"success"}` at the end, and
/// `{"error": "..."}` if it fails. Streaming is the default there, so it is
/// the default here; `{"stream": false}` collapses to one final object.
///
/// Until 0.7.5 this endpoint was a stub that answered "not yet implemented"
/// and did nothing, so there is no old behaviour to keep compatible with.
///
/// The work itself is `models::pull`, shared with `eullm pull`. Two copies of
/// a download path would drift the first time a repo layout changes, the way
/// two copies of a prompt builder already did.
async fn pull_model(State(state): State<S>, Json(body): Json<Value>) -> axum::response::Response {
    use crate::models::pull::{PullEvent, pull_from_huggingface};

    let name = body
        .get("name")
        .or_else(|| body.get("model"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    let Some(hf) = crate::registry::parse_hf_ref(&name) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": format!(
                    "`{name}` is not a HuggingFace reference. Use \
                     hf.co/<owner>/<repo>[:<quant>] — the catalog browser \
                     supplies this string for each quantization."
                )
            })),
        )
            .into_response();
    };
    let id = crate::models::pull::hf_ref_to_model_id(&hf);

    if state.store.gguf_path(&id).is_some() {
        let done = json!({ "status": "success", "model": id, "already_present": true });
        return if stream {
            ndjson_lines(futures_util::stream::once(async move { done }))
        } else {
            (StatusCode::OK, Json(done)).into_response()
        };
    }

    // The whole state, not the store: `ModelStore` is owned by `AppState`
    // and the Arc is what can cross into the task.
    let state = Arc::clone(&state);
    let (tx, rx) = mpsc::channel::<PullEvent>(256);
    let worker = tokio::spawn(async move {
        let tx_for_error = tx.clone();
        if let Err(e) = pull_from_huggingface(&state.store, &hf, &id, tx).await {
            let _ = tx_for_error.try_send(PullEvent::Failed(e));
        }
    });

    if stream {
        return ndjson_lines(pull_events_as_json(rx));
    }

    // Non-streaming: drain to the last meaningful line and answer with it.
    let mut rx = rx;
    let mut last = json!({ "status": "success" });
    while let Some(ev) = rx.recv().await {
        if let Some(v) = pull_event_to_json(&ev) {
            if v.get("error").is_some() {
                let _ = worker.await;
                return (StatusCode::BAD_GATEWAY, Json(v)).into_response();
            }
            if v.get("status").and_then(|s| s.as_str()) == Some("success") {
                last = v;
            }
        }
    }
    let _ = worker.await;
    (StatusCode::OK, Json(last)).into_response()
}

/// Render a [`PullEvent`] in Ollama's `/api/pull` vocabulary. `None` for
/// events that carry nothing a client would show.
fn pull_event_to_json(ev: &crate::models::pull::PullEvent) -> Option<Value> {
    use crate::models::pull::PullEvent;
    Some(match ev {
        PullEvent::Status(s) => json!({ "status": s }),
        PullEvent::Progress {
            file,
            completed,
            total,
        } => json!({
            "status": format!("pulling {file}"),
            // Ollama keys progress by layer digest. We have no digest before
            // the download finishes, so the filename stands in — clients use
            // it only to keep one bar per concurrent item.
            "digest": file,
            "total": total,
            "completed": completed,
        }),
        PullEvent::Done { id } => json!({ "status": "success", "model": id }),
        PullEvent::Failed(e) => json!({ "error": e }),
    })
}

fn pull_events_as_json(
    mut rx: mpsc::Receiver<crate::models::pull::PullEvent>,
) -> impl Stream<Item = Value> {
    async_stream::stream! {
        while let Some(ev) = rx.recv().await {
            if let Some(v) = pull_event_to_json(&ev) {
                yield v;
            }
        }
    }
}

/// Wrap a stream of JSON values as `application/x-ndjson` — one object per
/// line, no `data:` prefix, which is what Ollama's endpoints emit and what
/// its clients parse.
fn ndjson_lines(values: impl Stream<Item = Value> + Send + 'static) -> axum::response::Response {
    use futures_util::StreamExt;
    let lines = values.map(|v| {
        let mut line = v.to_string();
        line.push('\n');
        Ok::<_, std::convert::Infallible>(line)
    });
    axum::response::Response::builder()
        .header(header::CONTENT_TYPE, "application/x-ndjson")
        .body(Body::from_stream(lines))
        .unwrap()
}

// ── OpenAI-compatible handlers ───────────────────────────────────────────────

/// Ids of the models actually present in the local store, in listing order.
///
/// Both model-list endpoints answered from the catalog alone, so a model pulled
/// from a URL or a HuggingFace repo — anything not in the 22 curated entries —
/// was invisible to them even while it was loaded and answering. On `/v1/models`
/// that is the difference between usable and not: a coding editor picks the
/// model from that list, so a model it does not name cannot be selected at all
/// (issue #294).
fn local_model_ids(state: &AppState) -> Vec<String> {
    state
        .store
        .list_all()
        .unwrap_or_default()
        .into_iter()
        .filter(|m| !m.id.is_empty())
        .map(|m| m.id)
        .collect()
}

async fn list_models_openai(State(state): State<S>) -> Json<Value> {
    let mut data: Vec<Value> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    // What is on this machine comes first: it can be used right now, and it
    // includes models the catalog has never heard of.
    for id in local_model_ids(&state) {
        if seen.insert(id.clone()) {
            data.push(json!({
                "id": id,
                "object": "model",
                "created": 1700000000_u64,
                "owned_by": "eullm"
            }));
        }
    }

    // A loaded model launched from a path rather than pulled has no manifest,
    // and so is not in the list above.
    let loaded: Vec<String> = state
        .models
        .read()
        .await
        .by_recent_use()
        .iter()
        .map(|m| m.name.clone())
        .collect();
    for name in loaded {
        if seen.insert(name.clone()) {
            data.push(json!({
                "id": name,
                "object": "model",
                "created": 1700000000_u64,
                "owned_by": "eullm"
            }));
        }
    }

    // The decision model, when one is loaded, for System One clients
    // (jev-style's `model_info`, the System One SDKs' `models.list()`): its
    // entry here carries its budgets — added to the one it already has when
    // it is in the store — and the top-level `models` names it. Without
    // this the list showed only generation models, none of them the one
    // `/v1/systemone` answers with. An OpenAI client ignores the extra
    // fields, and still picks from `data`, where it keeps the place of a
    // model on this machine.
    let decision = state
        .decision
        .read()
        .await
        .as_ref()
        .map(|slot| (slot.model_name.clone(), slot.model.info()));
    let mut models = Vec::new();
    if let Some((name, info)) = decision {
        let (fields, model) = super::systemone::decision_model_listing(&name, &info);
        match data.iter_mut().find(|m| m["id"] == name.as_str()) {
            Some(Value::Object(entry)) => entry.extend(fields),
            _ => {
                seen.insert(name.clone());
                let mut entry = json!({
                    "id": name,
                    "object": "model",
                    "created": 1700000000_u64,
                    "owned_by": "eullm"
                });
                if let Value::Object(entry) = &mut entry {
                    entry.extend(fields);
                }
                data.push(entry);
            }
        }
        models.push(model);
    }

    // The OpenAI `id` field is what clients echo back as the `model` parameter
    // in chat requests, so it has to be the addressable catalog id, not the
    // human display name.
    data.extend(
        EU_CATALOG
            .iter()
            .filter(|m| !seen.contains(&m.id))
            .map(|m| {
                json!({
                    "id": m.id,
                    "object": "model",
                    "created": 1700000000_u64,
                    "owned_by": "eullm"
                })
            }),
    );

    // `auto`, when routing is configured, as on `/api/tags`.
    if state.router.is_some() {
        data.push(json!({
            "id": super::route::AUTO,
            "object": "model",
            "created": 1700000000_u64,
            "owned_by": "eullm"
        }));
    }

    Json(json!({ "object": "list", "data": data, "models": models }))
}

async fn chat_completions(
    State(state): State<S>,
    // Present on every request: the auth middleware inserts an anonymous
    // identity when no keys are configured, so this never fails to extract.
    axum::Extension(identity): axum::Extension<super::Identity>,
    Json(body): Json<Value>,
) -> Result<axum::response::Response, Refusal> {
    answer_routed(
        state,
        identity,
        body,
        StreamFormat::OpenAI,
        chat_completions_with,
    )
    .await
}

/// `/v1/chat/completions`, answered by `routed` when the request was routed
/// to it, and otherwise by the model it names.
async fn chat_completions_with(
    state: S,
    audit: AuditCtx,
    routed: Option<SlotSnapshot>,
    body: Value,
) -> Result<axum::response::Response, Refusal> {
    let snap = match routed {
        Some(snap) => snap,
        None => {
            let (override_batch_size, override_ctx_size) = parse_slot_overrides(&body)?;
            // `keep_alive` is an EULLM/Ollama extension to the OpenAI shape,
            // not part of it — accepted here too so the idle-unload timer
            // works the same way regardless of which endpoint a client
            // happens to use.
            let keep_alive = super::parse_keep_alive(body.get("keep_alive"));
            ensure_model(
                &state,
                requested_model(&body),
                override_batch_size,
                override_ctx_size,
                keep_alive,
            )
            .await?
        }
    };
    let model = snap.model_name.clone();

    let messages = body
        .get("messages")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let messages = inject_web_content(
        messages,
        state.web_enabled,
        state.ctx_size,
        state.batch_size,
        &state.web_policy,
    )
    .await;

    let think = body.get("think").and_then(|v| v.as_bool()).unwrap_or(true);
    let model_name_ref: &str = &model;
    // ── Tool calling (issue #334): render through the model's own template
    // with the request's full OpenAI-format JSON, so tool definitions, tool
    // roles and assistant tool_calls in the history all reach the template.
    // Falls back to the plain path — tools ignored, with a warning — when
    // the GGUF embeds no template: there is no reliable way to teach an
    // unknown hardcoded format to call tools.
    let tools_json: Option<String> = body
        .get("tools")
        .filter(|t| t.as_array().is_some_and(|a| !a.is_empty()))
        .map(Value::to_string);
    let tool_choice: Option<String> = body.get("tool_choice").map(|tc| {
        // String forms pass through; the named-function object form is
        // approximated as "required" (the closest common_chat mapping).
        tc.as_str().map_or_else(|| "required".to_string(), str::to_string)
    });
    let oai_template = if tools_json.is_some() {
        serde_json::to_string(&messages).ok().and_then(|messages_json| {
            if let Some(engine) = snap.engine.as_ref() {
                engine.apply_oai_chat_template(
                    &messages_json,
                    tools_json.as_deref(),
                    tool_choice.as_deref(),
                    think,
                )
            } else if let Some(scheduler) = snap.scheduler.as_ref() {
                scheduler.apply_oai_chat_template(
                    &messages_json,
                    tools_json.as_deref(),
                    tool_choice.as_deref(),
                    think,
                )
            } else {
                None
            }
        })
    } else {
        None
    };
    if tools_json.is_some() && oai_template.is_none() {
        tracing::warn!(
            "tools requested but the model has no embedded chat template; proceeding without tools"
        );
    }

    let (prompt, stop_sequences, response_prefix) = match &oai_template {
        Some(tmpl) => (tmpl.prompt.clone(), Vec::new(), String::new()),
        None => build_chat_prompt(&snap, &messages, think, model_name_ref),
    };
    let sp = parse_chat_completions_params(&body);
    let grammar = parse_format_grammar(&body);

    let request = GenerateRequest {
        prompt,
        max_tokens: sp.max_tokens,
        temperature: sp.temperature,
        top_k: sp.top_k,
        top_p: sp.top_p,
        min_p: sp.min_p,
        repeat_penalty: sp.repeat_penalty,
        repeat_last_n: sp.repeat_last_n,
        seed: sp.seed,
        num_ctx: sp.num_ctx,
        stop_sequences,
        // think-aware: with think:false a think tag in the OUTPUT is
        // spurious, since the prompt already carries a closed empty block.
        // On the tool path nothing is filtered: the format-aware parser
        // needs the raw output, thinking blocks included.
        filter_sequences: if oai_template.is_some() {
            Vec::new()
        } else {
            crate::inference::default_filters(think)
        },
        grammar,
        raw: false,
        cache_prompt: sp.cache_prompt,
        response_prefix,
    };

    // ── Tool calling (issue #334): buffered, format-aware path ──────────
    // Structured tool_calls only exist once the output is complete and
    // parsed, and a client that asked for tools cannot act on half a call
    // anyway, so this path always buffers the full generation. A streaming
    // request still gets an SSE response — one delta carrying the whole
    // parsed message, then the finish chunk — which is valid OpenAI
    // streaming, just coarse.
    if let Some(tmpl) = oai_template {
        let rx = if let Some(ref sched) = snap.scheduler {
            sched.try_submit(request)?
        } else {
            sequential_to_channel(Arc::clone(snap.engine.as_ref().unwrap()), request)
        };
        let rx = rx;
        let rx = crate::monitor::watch(rx, &snap.model_name, "chat.completions");
        let Collected {
            text,
            tokens_generated,
            tokens_prompt,
            duration_ms,
            stop_reason,
            stats,
        } = collect_stream(rx).await.map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e })),
            )
        })?;

        audit.log(
            &model,
            StreamFormat::OpenAI,
            tokens_prompt,
            tokens_generated,
            duration_ms,
            &stats,
        );

        let (message, called_tools) = match tmpl.parse_output(&text) {
            Some(parsed) => {
                let mut message = json!({ "role": "assistant" });
                message["content"] =
                    if parsed.content.is_empty() && !parsed.tool_calls.is_empty() {
                        Value::Null
                    } else {
                        Value::String(parsed.content)
                    };
                if let Some(reasoning) = parsed.reasoning_content {
                    message["reasoning_content"] = Value::String(reasoning);
                }
                let called = !parsed.tool_calls.is_empty();
                if called {
                    message["tool_calls"] = parsed
                        .tool_calls
                        .into_iter()
                        .enumerate()
                        .map(|(i, call)| {
                            json!({
                                "id": call.id.unwrap_or_else(|| format!("call_{}", uuid::Uuid::new_v4().simple())),
                                "index": i,
                                "type": "function",
                                "function": { "name": call.name, "arguments": call.arguments },
                            })
                        })
                        .collect();
                }
                (message, called)
            }
            None => (json!({ "role": "assistant", "content": text }), false),
        };
        let finish_reason = if called_tools {
            "tool_calls"
        } else {
            stop_reason.as_api_str()
        };
        let usage = json!({
            "prompt_tokens": tokens_prompt,
            "completion_tokens": tokens_generated,
            "total_tokens": tokens_prompt + tokens_generated,
        });
        let timings = openai_timings(tokens_prompt, tokens_generated, &stats);

        if is_streaming(&body) {
            let stream = buffered_message_sse(
                model,
                message,
                finish_reason.to_string(),
                usage,
                timings,
                snap.lease,
                audit,
            );
            return Ok(Sse::new(stream).into_response());
        }
        return Ok(Json(audit.extended(json!({
            "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            "object": "chat.completion",
            "created": chrono::Utc::now().timestamp(),
            "model": model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
            "usage": usage,
            "timings": timings,
        })))
        .into_response());
    }

    if let Some(ref sched) = snap.scheduler {
        let rx = sched.try_submit(request)?;
        let rx = crate::monitor::watch(rx, &snap.model_name, "chat.completions");

        if is_streaming(&body) {
            let stream = stream_from_channel_sse(
                rx,
                model,
                StreamFormat::OpenAI,
                audit,
                snap.lease,
                snap.load_duration,
            );
            Ok(Sse::new(stream).into_response())
        } else {
            let Collected {
                text,
                tokens_generated,
                tokens_prompt,
                duration_ms,
                stop_reason,
                stats,
            } = collect_stream(rx).await.map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": e })),
                )
            })?;

            audit.log(
                &model,
                StreamFormat::OpenAI,
                tokens_prompt,
                tokens_generated,
                duration_ms,
                &stats,
            );

            Ok(Json(audit.extended(json!({
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
                    "finish_reason": stop_reason.as_api_str()
                }],
                "usage": {
                    "prompt_tokens": tokens_prompt,
                    "completion_tokens": tokens_generated,
                    "total_tokens": tokens_prompt + tokens_generated
                },
                "timings": openai_timings(tokens_prompt, tokens_generated, &stats),
            })))
            .into_response())
        }
    } else {
        let engine = Arc::clone(snap.engine.as_ref().unwrap());

        if is_streaming(&body) {
            let rx = sequential_to_channel(engine, request);
            let rx = crate::monitor::watch(rx, &snap.model_name, "chat.completions");
            let stream = stream_from_channel_sse(
                rx,
                model,
                StreamFormat::OpenAI,
                audit,
                snap.lease,
                snap.load_duration,
            );
            Ok(Sse::new(stream).into_response())
        } else {
            let tracker = crate::monitor::begin(&snap.model_name, "chat.completions");
            let result = tokio::task::spawn_blocking({
                let engine = engine;
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

            tracker.answered(&result);
            audit.log(
                &model,
                StreamFormat::OpenAI,
                result.tokens_prompt,
                result.tokens_generated,
                result.duration_ms,
                &result.stats,
            );

            Ok(Json(audit.extended(json!({
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
                    "finish_reason": result.stop_reason.as_api_str()
                }],
                "usage": {
                    "prompt_tokens": result.tokens_prompt,
                    "completion_tokens": result.tokens_generated,
                    "total_tokens": result.tokens_prompt + result.tokens_generated
                },
                "timings": openai_timings(result.tokens_prompt, result.tokens_generated, &result.stats),
            })))
            .into_response())
        }
    }
}

// ── Streaming helpers ────────────────────────────────────────────────────────

/// SSE for the buffered tool-calling path: the whole parsed message as one
/// delta chunk, then a finish chunk with usage, then `[DONE]`. Coarse but
/// valid OpenAI streaming — structured tool calls cannot be streamed
/// incrementally without the format-aware incremental parser, which is
/// separate future work.
///
/// The stream holds `lease` until it ends, so the model counts as in use for
/// the whole response — as it does in the other stream helpers.
fn buffered_message_sse(
    model: String,
    message: Value,
    finish_reason: String,
    usage: Value,
    timings: Value,
    lease: Lease,
    audit: AuditCtx,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
    let created = chrono::Utc::now().timestamp();
    let completion_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let last = audit.extended(json!({
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{ "index": 0, "delta": {}, "finish_reason": finish_reason }],
        "usage": usage,
        "timings": timings,
    }));
    async_stream::stream! {
        let _lease = lease;
        yield Ok(Event::default().data(
            json!({
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{ "index": 0, "delta": message, "finish_reason": Value::Null }],
            })
            .to_string(),
        ));
        yield Ok(Event::default().data(last.to_string()));
        yield Ok(Event::default().data("[DONE]"));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamFormat {
    OllamaGenerate,
    OllamaChat,
    OpenAI,
}

impl StreamFormat {
    /// The audit trail's `request_type` of a generation answered in this
    /// format: the endpoint it came through.
    fn request_type(self) -> &'static str {
        match self {
            Self::OllamaGenerate => "generate",
            Self::OllamaChat => "chat",
            Self::OpenAI => "chat.completions",
        }
    }
}

/// What a generation's audit entry records besides the model and what it
/// read and wrote: who asked, and the route `"model": "auto"` took to it.
/// One builder for every way a generation can end — buffered or streamed,
/// on the scheduler or the sequential engine, text or images — each of
/// which used to build the entry field by field, ten copies of the same six
/// lines that any new field had to reach in all ten places. The route also
/// goes on the response, in the same places, through [`AuditCtx::extended`].
#[derive(Debug, Clone, Default, PartialEq)]
struct AuditCtx {
    /// The id of the API key the request presented, when it presented one.
    user_id: Option<String>,
    /// How the request was routed, when it asked for `auto`.
    route: Option<super::route::RouteInfo>,
}

impl AuditCtx {
    /// The context of a request from `identity`.
    fn of(identity: &super::Identity) -> Self {
        Self {
            user_id: identity.key_id().map(str::to_string),
            route: None,
        }
    }

    /// `answer` — a response, or the last line or chunk of a stream — with
    /// the route under `eullm.route` when the request was routed.
    fn extended(&self, mut answer: Value) -> Value {
        if let Some(route) = &self.route {
            answer["eullm"] = json!({ "route": route });
        }
        answer
    }

    /// The audit entry of one finished generation on `model`.
    fn entry(
        &self,
        model: &str,
        format: StreamFormat,
        tokens_prompt: u32,
        tokens_generated: u32,
        duration_ms: u64,
        stats: &AnswerStats,
    ) -> AuditEntry {
        let mut entry = AuditEntry::new(model.to_string(), format.request_type().to_string());
        entry.input_tokens = tokens_prompt;
        entry.output_tokens = tokens_generated;
        entry.duration_ms = duration_ms;
        entry.user_id = self.user_id.clone();
        entry.route = self.route.as_ref().map(super::route::RouteInfo::reference);
        if stats.draft_n > 0 {
            entry.draft_n = Some(stats.draft_n);
            entry.draft_n_accepted = Some(stats.draft_n_accepted);
        }
        entry
    }

    /// Write that entry to the audit trail.
    fn log(
        &self,
        model: &str,
        format: StreamFormat,
        tokens_prompt: u32,
        tokens_generated: u32,
        duration_ms: u64,
        stats: &AnswerStats,
    ) {
        AuditLogger::new().log(&self.entry(
            model,
            format,
            tokens_prompt,
            tokens_generated,
            duration_ms,
            stats,
        ));
    }
}

/// Convert an mpsc channel of StreamEvents into an SSE event stream.
///
/// Used for OpenAI-compatible `/v1/chat/completions` streaming only.
///
/// The stream holds `lease` until it ends — the last event sent, or the
/// client gone — so keep_alive counts from the end of the response, not
/// from when the handler returned it.
fn stream_from_channel_sse(
    mut rx: mpsc::Receiver<StreamEvent>,
    model: String,
    format: StreamFormat,
    audit: AuditCtx,
    lease: Lease,
    load_duration: std::time::Duration,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
    async_stream::stream! {
        let _lease = lease;
        let completion_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());

        while let Some(event) = rx.recv().await {
            match event {
                StreamEvent::Token(piece) => {
                    let data = format_token_event(&piece, &model, &completion_id, format);
                    yield Ok(Event::default().data(data.to_string()));
                }
                StreamEvent::Done { tokens_generated, tokens_prompt, duration_ms, stop_reason, stats } => {
                    // Audit log
                    audit.log(&model, format, tokens_prompt, tokens_generated, duration_ms, &stats);

                    let data = audit.extended(format_done_event(
                        &model, &completion_id, format,
                        tokens_generated, tokens_prompt, duration_ms, stop_reason,
                        load_duration, &stats,
                    ));
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

/// Convert an mpsc channel of StreamEvents into an NDJSON stream.
///
/// Ollama uses NDJSON (newline-delimited JSON) for `/api/generate` and
/// `/api/chat` streaming — NOT SSE. Each line is a complete JSON object
/// followed by a newline. No `data:` prefix, no double newlines.
///
/// This is what Ollama clients (RAG Enterprise, Open WebUI, etc.) expect.
///
/// The stream holds `lease` until it ends, as `stream_from_channel_sse` does.
/// `split_thinking`: the request asked for the reasoning apart (`think`, see
/// [`super::thinking`]), so it streams as `thinking` beside the answer.
fn ndjson_stream_response(
    mut rx: mpsc::Receiver<StreamEvent>,
    model: String,
    format: StreamFormat,
    audit: AuditCtx,
    lease: Lease,
    load_duration: std::time::Duration,
    split_thinking: bool,
) -> axum::response::Response {
    let stream = async_stream::stream! {
        let _lease = lease;
        let completion_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        let mut splitter = split_thinking.then(ThinkingSplitter::new);

        while let Some(event) = rx.recv().await {
            match event {
                StreamEvent::Token(piece) => {
                    let parts = match splitter.as_mut() {
                        Some(splitter) => splitter.push(&piece),
                        None => vec![Part::Content(piece)],
                    };
                    for part in parts {
                        let data = format_part_event(&part, &model, &completion_id, format);
                        let mut line = data.to_string();
                        line.push('\n');
                        yield Ok::<_, std::convert::Infallible>(line);
                    }
                }
                StreamEvent::Done { tokens_generated, tokens_prompt, duration_ms, stop_reason, stats } => {
                    // What the splitter still held: the end of the answer, or
                    // of reasoning the token budget cut short.
                    for part in splitter.take().map(ThinkingSplitter::finish).unwrap_or_default() {
                        let data = format_part_event(&part, &model, &completion_id, format);
                        let mut line = data.to_string();
                        line.push('\n');
                        yield Ok(line);
                    }
                    audit.log(&model, format, tokens_prompt, tokens_generated, duration_ms, &stats);

                    let data = audit.extended(format_done_event(
                        &model, &completion_id, format,
                        tokens_generated, tokens_prompt, duration_ms, stop_reason,
                        load_duration, &stats,
                    ));
                    let mut line = data.to_string();
                    line.push('\n');
                    yield Ok(line);
                    break;
                }
                StreamEvent::Error(msg) => {
                    let data = json!({ "error": msg });
                    let mut line = data.to_string();
                    line.push('\n');
                    yield Ok(line);
                    break;
                }
            }
        }
    };

    axum::response::Response::builder()
        .header(header::CONTENT_TYPE, "application/x-ndjson")
        .header(header::TRANSFER_ENCODING, "chunked")
        .body(Body::from_stream(stream))
        .unwrap()
}

/// Start sequential inference in a background thread, returning an mpsc receiver.
///
/// Shared with the terminal REPL: the sequential engine and the scheduler both
/// speak `StreamEvent`, so the chat loop does not care which one produced it.
pub(crate) fn sequential_to_channel(
    engine: Arc<crate::inference::InferenceEngine>,
    request: GenerateRequest,
) -> mpsc::Receiver<StreamEvent> {
    let (tx, rx) = mpsc::channel::<StreamEvent>(32);
    tokio::task::spawn_blocking(move || {
        engine.generate_streaming(&request, tx);
    });
    rx
}

/// One streamed line of an Ollama endpoint: a piece of the answer, or of its
/// reasoning when the request asked for that apart. Reasoning goes where
/// Ollama puts it, with the answer's own field empty: `message.thinking` on
/// `/api/chat`, `thinking` on `/api/generate`.
fn format_part_event(part: &Part, model: &str, completion_id: &str, format: StreamFormat) -> Value {
    match part {
        Part::Content(piece) => format_token_event(piece, model, completion_id, format),
        Part::Thinking(piece) => {
            let mut event = format_token_event("", model, completion_id, format);
            match format {
                StreamFormat::OllamaChat => event["message"]["thinking"] = json!(piece),
                // `/v1` never splits, so this is `/api/generate`.
                StreamFormat::OllamaGenerate | StreamFormat::OpenAI => {
                    event["thinking"] = json!(piece)
                }
            }
            event
        }
    }
}

/// `/api/chat`'s `message`: the answer, and its reasoning in `thinking` when
/// the request asked for that apart (`think`, see [`super::thinking`]).
fn chat_message(text: String, split_thinking: bool) -> Value {
    if !split_thinking {
        return json!({ "role": "assistant", "content": text });
    }
    let (thinking, content) = super::thinking::split(&text);
    let mut message = json!({ "role": "assistant", "content": content });
    if let Some(thinking) = thinking {
        message["thinking"] = Value::String(thinking);
    }
    message
}

/// `/api/generate`'s `response`, and its reasoning apart when the request
/// asked for that (`think`, see [`super::thinking`]).
fn generate_answer(text: String, split_thinking: bool) -> (String, Option<String>) {
    if split_thinking {
        let (thinking, response) = super::thinking::split(&text);
        (response, thinking)
    } else {
        (text, None)
    }
}

/// `answer` with the MTP drafts its model proposed and kept, under
/// llama-server's names, when the head proposed any (`--mtp`).
fn with_drafts(mut answer: Value, stats: &AnswerStats) -> Value {
    if stats.draft_n > 0 {
        answer["draft_n"] = json!(stats.draft_n);
        answer["draft_n_accepted"] = json!(stats.draft_n_accepted);
    }
    answer
}

/// llama-server's `timings` for an OpenAI answer: how long reading its
/// prompt and writing it took, in milliseconds, and the MTP drafts when the
/// head proposed any. Not part of OpenAI's API: clients that do not know it
/// ignore it, and the ones written for llama-server read it.
fn openai_timings(tokens_prompt: u32, tokens_generated: u32, stats: &AnswerStats) -> Value {
    with_drafts(
        json!({
            "prompt_n": tokens_prompt,
            "prompt_ms": stats.prompt_time.as_secs_f64() * 1000.0,
            "predicted_n": tokens_generated,
            "predicted_ms": stats.eval_time.as_secs_f64() * 1000.0,
        }),
        stats,
    )
}

fn format_token_event(
    piece: &str,
    model: &str,
    completion_id: &str,
    format: StreamFormat,
) -> Value {
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

/// `load_duration` is how long the request waited for the model to load
/// (Ollama's field of the name; zero for a model already loaded), and part
/// of `total_duration`, as in Ollama.
#[allow(clippy::too_many_arguments)]
fn format_done_event(
    model: &str,
    completion_id: &str,
    format: StreamFormat,
    tokens_generated: u32,
    tokens_prompt: u32,
    duration_ms: u64,
    stop_reason: StopReason,
    load_duration: std::time::Duration,
    stats: &AnswerStats,
) -> Value {
    let reason = stop_reason.as_api_str();
    let load = nanos(load_duration);
    match format {
        StreamFormat::OllamaGenerate => with_drafts(
            json!({
                "model": model,
                "created_at": chrono::Utc::now().to_rfc3339(),
                "response": "",
                "done": true,
                "done_reason": reason,
                "total_duration": duration_ms * 1_000_000 + load,
                "load_duration": load,
                "prompt_eval_count": tokens_prompt,
                "prompt_eval_duration": nanos(stats.prompt_time),
                "eval_count": tokens_generated,
                "eval_duration": nanos(stats.eval_time),
            }),
            stats,
        ),
        StreamFormat::OllamaChat => with_drafts(
            json!({
                "model": model,
                "created_at": chrono::Utc::now().to_rfc3339(),
                "message": {
                    "role": "assistant",
                    "content": "",
                },
                "done": true,
                "done_reason": reason,
                "total_duration": duration_ms * 1_000_000 + load,
                "load_duration": load,
                "prompt_eval_count": tokens_prompt,
                "prompt_eval_duration": nanos(stats.prompt_time),
                "eval_count": tokens_generated,
                "eval_duration": nanos(stats.eval_time),
            }),
            stats,
        ),
        StreamFormat::OpenAI => json!({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": chrono::Utc::now().timestamp(),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": reason,
            }],
            "usage": {
                "prompt_tokens": tokens_prompt,
                "completion_tokens": tokens_generated,
                "total_tokens": tokens_prompt + tokens_generated,
            },
            "timings": openai_timings(tokens_prompt, tokens_generated, stats),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::model_names_match;

    /// Ollama's `/api/ps` fields, in their types, and a date for a model kept
    /// for good, which Ollama sends far in the future rather than null.
    #[test]
    fn a_ps_entry_has_ollamas_fields() {
        let model = |expires_in| RunningModel {
            name: "qwen3-8b".into(),
            size: 5_000_000_000,
            size_vram: 4_000_000_000,
            size_vram_measured: true,
            context_length: 4096,
            expires_in,
            idle_for: std::time::Duration::from_secs(30),
            digest: "sha256:abc".into(),
            family: "qwen3".into(),
            parameter_size: "8.2B".into(),
            quantization_level: "Q4_K_M".into(),
            slot: "generation",
            in_flight: 1,
            batch_size: Some(4),
            gpu_layers: Some(-1),
            reserved_companion: false,
            launch: true,
        };
        let now = chrono::Utc::now();
        let entry = running_model_entry(&model(Some(std::time::Duration::from_secs(300))), now);
        for field in ["name", "model", "digest", "expires_at"] {
            assert!(entry[field].is_string(), "{field}: {entry}");
        }
        for field in ["size", "size_vram", "context_length"] {
            assert!(entry[field].is_u64(), "{field}: {entry}");
        }
        for field in [
            "parent_model",
            "format",
            "family",
            "parameter_size",
            "quantization_level",
        ] {
            assert!(entry["details"][field].is_string(), "{field}: {entry}");
        }
        assert_eq!(entry["details"]["families"], json!(["qwen3"]));
        let parse = |entry: &Value, field: &str| {
            chrono::DateTime::parse_from_rfc3339(entry[field].as_str().unwrap()).expect("RFC 3339")
        };
        assert_eq!(
            (parse(&entry, "expires_at").to_utc() - now).num_seconds(),
            300
        );
        assert_eq!(
            (now - parse(&entry["eullm"], "last_used").to_utc()).num_seconds(),
            30
        );
        assert_eq!(entry["eullm"]["slot"], "generation");
        assert_eq!(entry["eullm"]["in_flight"], 1);
        assert_eq!(entry["eullm"]["launch"], true);

        let kept = running_model_entry(&model(None), now);
        let year = parse(&kept, "expires_at").format("%Y").to_string();
        assert!(year.parse::<i32>().unwrap() > 2200, "kept for good: {kept}");
    }

    /// The one builder writes what each of the ten copies it replaced wrote:
    /// the model, the endpoint as the request type, the tokens read and
    /// written, the time, and who asked — and nothing else.
    /// Only a routed request's answer gains the `eullm` object, and its
    /// Ollama and OpenAI fields stay as they were.
    #[test]
    fn only_a_routed_answer_carries_its_route() {
        let done = format_done_event(
            "small-m",
            "chatcmpl-x",
            StreamFormat::OpenAI,
            3,
            5,
            10,
            StopReason::Stop,
            std::time::Duration::ZERO,
            &AnswerStats::default(),
        );
        assert_eq!(AuditCtx::default().extended(done.clone()), done);
        let entry = AuditCtx::default().entry(
            "small-m",
            StreamFormat::OpenAI,
            5,
            3,
            10,
            &AnswerStats::default(),
        );
        assert!(entry.route.is_none());
    }

    #[test]
    fn a_generations_audit_entry_has_the_fields_it_always_had() {
        for (format, request_type) in [
            (StreamFormat::OllamaGenerate, "generate"),
            (StreamFormat::OllamaChat, "chat"),
            (StreamFormat::OpenAI, "chat.completions"),
        ] {
            for user_id in [None, Some("ci".to_string())] {
                let ctx = AuditCtx {
                    user_id: user_id.clone(),
                    route: None,
                };
                let entry = ctx.entry("qwen3-8b", format, 12, 40, 900, &AnswerStats::default());
                // As the copies built it.
                let mut old = AuditEntry::new("qwen3-8b".to_string(), request_type.to_string());
                old.input_tokens = 12;
                old.output_tokens = 40;
                old.duration_ms = 900;
                old.user_id = user_id;
                let (mut new, mut old) = (
                    serde_json::to_value(&entry).unwrap(),
                    serde_json::to_value(&old).unwrap(),
                );
                for line in [&mut new, &mut old] {
                    let fields = line.as_object_mut().unwrap();
                    fields.remove("id");
                    fields.remove("timestamp");
                }
                assert_eq!(new, old, "{request_type}");
            }
        }
    }

    /// The MTP drafts go on a generation's audit line only when there were
    /// some: a line without them is the line written before they existed.
    #[test]
    fn an_audit_entry_names_the_mtp_drafts_only_when_there_were_some() {
        let ctx = AuditCtx::default();
        let entry = |stats: &AnswerStats| {
            serde_json::to_value(ctx.entry("m", StreamFormat::OllamaChat, 1, 2, 3, stats)).unwrap()
        };
        let plain = entry(&AnswerStats::default());
        assert!(plain.get("draft_n").is_none() && plain.get("draft_n_accepted").is_none());
        let drafted = entry(&AnswerStats {
            draft_n: 274,
            draft_n_accepted: 117,
            ..AnswerStats::default()
        });
        assert_eq!(
            (
                drafted["draft_n"].as_u64(),
                drafted["draft_n_accepted"].as_u64()
            ),
            (Some(274), Some(117))
        );
    }

    /// The last line of a stream says where the time went: Ollama's
    /// `prompt_eval_duration` and `eval_duration` from the stats (they were 0
    /// and the whole request), the drafts beside them when there were some,
    /// and llama-server's `timings` on the OpenAI chunk.
    #[test]
    fn the_last_line_reports_prompt_and_answer_times_and_the_drafts() {
        use std::time::Duration;
        let stats = AnswerStats {
            prompt_time: Duration::from_millis(250),
            eval_time: Duration::from_millis(1500),
            draft_n: 40,
            draft_n_accepted: 30,
        };
        let done = |format, stats: &AnswerStats| {
            format_done_event(
                "m",
                "c",
                format,
                100,
                20,
                1800,
                StopReason::Stop,
                Duration::ZERO,
                stats,
            )
        };
        let chat = done(StreamFormat::OllamaChat, &stats);
        assert_eq!(chat["prompt_eval_duration"], 250_000_000);
        assert_eq!(chat["eval_duration"], 1_500_000_000);
        assert_eq!(chat["total_duration"], 1_800_000_000);
        assert_eq!(
            (chat["draft_n"].as_u64(), chat["draft_n_accepted"].as_u64()),
            (Some(40), Some(30))
        );
        let plain = done(StreamFormat::OllamaGenerate, &AnswerStats::default());
        assert!(plain.get("draft_n").is_none());
        let openai = done(StreamFormat::OpenAI, &stats);
        assert_eq!(openai["timings"]["prompt_ms"], 250.0);
        assert_eq!(openai["timings"]["predicted_n"], 100);
        assert_eq!(openai["timings"]["draft_n_accepted"], 30);
        assert!(
            done(StreamFormat::OpenAI, &AnswerStats::default())["timings"]
                .get("draft_n")
                .is_none()
        );
    }

    // A reasoning model doing free-text tool-calling can burn through
    // hundreds of tokens of <think> before producing anything else — a
    // small fixed cap here (the old default was 512) truncates mid-block
    // and leaves a text-based tool-calling client (e.g. Cline) holding an
    // unclosed tag. Confirmed on real hardware: see the v0.6.27 changelog.
    #[test]
    fn max_tokens_defaults_to_unbounded_not_a_small_fixed_cap() {
        let sp = parse_generate_params(&json!({}));
        assert_eq!(sp.max_tokens, u32::MAX);
    }

    /// #345: two quants of the same HuggingFace repo are different models.
    /// The old stem comparison cut at the LAST dot, so every `ornith-1.*`
    /// collapsed to `ornith-1` and switching quants was a silent no-op.
    #[test]
    fn different_quants_of_a_repo_are_different_models() {
        assert!(!model_names_match(
            "ornith-1.0-35b-gguf-ud-q5_k_xl",
            "ornith-1.0-35b-gguf-ud-q4_k_m"
        ));
        assert!(!model_names_match("qwen3.6-27b-q8_0", "qwen3.6-27b-q4_k_m"));
    }

    /// The matching a path-loaded model still needs: same model addressed as
    /// a full path, with or without the extension, in any case.
    #[test]
    fn a_loaded_path_still_matches_its_plain_name() {
        assert!(model_names_match(
            "/home/u/.eullm/models/qwen3.6-27b/Qwen3.6-27B-Q4_K_M.gguf",
            "Qwen3.6-27B-Q4_K_M"
        ));
        assert!(model_names_match("ornith-1.0-35b.gguf", "ORNITH-1.0-35B"));
        assert!(model_names_match("qwen3-8b", "qwen3-8b"));
    }

    #[test]
    fn max_tokens_still_honors_an_explicit_client_value() {
        let sp = parse_generate_params(&json!({ "max_tokens": 128 }));
        assert_eq!(sp.max_tokens, 128);

        let sp = parse_generate_params(&json!({ "options": { "num_predict": 256 } }));
        assert_eq!(sp.max_tokens, 256);
    }

    /// `cache_prompt` is llama.cpp's top-level field; an Ollama client puts
    /// what Ollama lacks in `options`. Only an explicit `false` turns reuse
    /// off: a prompt cache is what keeps a long conversation's next turn
    /// cheap, and a typo must not cost every request its whole prompt.
    #[test]
    fn only_an_explicit_false_turns_the_prompt_cache_off() {
        assert!(parse_generate_params(&json!({})).cache_prompt);
        assert!(!parse_generate_params(&json!({ "cache_prompt": false })).cache_prompt);
        assert!(
            !parse_generate_params(&json!({ "options": { "cache_prompt": false } })).cache_prompt
        );
        assert!(
            !parse_chat_completions_params(&json!({ "cache_prompt": false })).cache_prompt
        );
        assert!(parse_generate_params(&json!({ "cache_prompt": true })).cache_prompt);
        assert!(parse_generate_params(&json!({ "cache_prompt": "false" })).cache_prompt);
        assert!(parse_generate_params(&json!({ "cache_prompt": null })).cache_prompt);
    }

    // ── `max_completion_tokens` on /v1/chat/completions ─────────────────
    //
    // OpenAI's current name for the output limit. Clients written against
    // the current API send only it, older ones only `max_tokens`.

    #[test]
    fn chat_completions_read_the_limit_under_either_name() {
        let sp = parse_chat_completions_params(&json!({ "max_completion_tokens": 64 }));
        assert_eq!(sp.max_tokens, 64);
        let sp = parse_chat_completions_params(&json!({ "max_tokens": 128 }));
        assert_eq!(sp.max_tokens, 128);
    }

    /// Both ways round, so what is pinned is "the current name wins", not
    /// "the smaller limit wins".
    #[test]
    fn max_completion_tokens_wins_over_max_tokens() {
        let sp = parse_chat_completions_params(&json!({
            "max_completion_tokens": 64,
            "max_tokens": 128,
        }));
        assert_eq!(sp.max_tokens, 64);
        let sp = parse_chat_completions_params(&json!({
            "max_completion_tokens": 256,
            "max_tokens": 128,
        }));
        assert_eq!(sp.max_tokens, 256);
    }

    #[test]
    fn chat_completions_without_a_limit_stay_unbounded() {
        let sp = parse_chat_completions_params(&json!({}));
        assert_eq!(sp.max_tokens, u32::MAX);
        // Ollama's `options.num_predict`, accepted here before, still is.
        let sp = parse_chat_completions_params(&json!({ "options": { "num_predict": 256 } }));
        assert_eq!(sp.max_tokens, 256);
    }

    #[test]
    fn a_null_max_completion_tokens_leaves_max_tokens_in_force() {
        let sp = parse_chat_completions_params(&json!({
            "max_completion_tokens": null,
            "max_tokens": 128,
        }));
        assert_eq!(sp.max_tokens, 128);
        let sp = parse_chat_completions_params(&json!({ "max_completion_tokens": null }));
        assert_eq!(sp.max_tokens, u32::MAX);
    }

    /// A value that is not a non-negative integer reads under the current
    /// name exactly as it always has under `max_tokens`, as no limit, and
    /// being set it still takes precedence over a `max_tokens`.
    #[test]
    fn an_invalid_max_completion_tokens_reads_as_an_invalid_max_tokens_does() {
        for value in [json!(-1), json!("64"), json!(1.5), json!(true)] {
            let as_max_tokens = parse_generate_params(&json!({ "max_tokens": value })).max_tokens;
            assert_eq!(as_max_tokens, u32::MAX, "max_tokens: {value}");
            for body in [
                json!({ "max_completion_tokens": value }),
                json!({ "max_completion_tokens": value, "max_tokens": 128 }),
            ] {
                let sp = parse_chat_completions_params(&body);
                assert_eq!(sp.max_tokens, as_max_tokens, "{body}");
            }
        }
    }

    /// Ollama's API has no `max_completion_tokens`, so `/api/generate` and
    /// `/api/chat` do not read one.
    #[test]
    fn ollama_endpoints_do_not_read_max_completion_tokens() {
        let sp = parse_generate_params(&json!({ "max_completion_tokens": 64 }));
        assert_eq!(sp.max_tokens, u32::MAX);
        let sp = parse_generate_params(&json!({
            "max_completion_tokens": 64,
            "max_tokens": 128,
        }));
        assert_eq!(sp.max_tokens, 128);
    }

    // ── Slot override validation ────────────────────────────────────────
    //
    // These two fields are the only request-body values that reach a
    // model-load decision (scheduler slot count, KV-cache size). Before
    // validation existed, `{"batch_size": 4294967296}` truncated to zero
    // usable slots — the model reported as loaded while no request could
    // ever be served again — and an absurd `ctx_size` failed the context
    // allocation *after* load_generation_model had already unloaded the previous
    // model, leaving the slot empty.

    fn overrides(body: Value) -> Result<SlotOverrides, StatusCode> {
        parse_slot_overrides(&body).map_err(|(status, _)| status)
    }

    #[test]
    fn absent_overrides_keep_the_launch_time_values() {
        assert_eq!(overrides(json!({})), Ok((None, None)));
        assert_eq!(
            overrides(json!({ "batch_size": null, "ctx_size": null })),
            Ok((None, None))
        );
    }

    #[test]
    fn in_range_overrides_are_accepted() {
        assert_eq!(
            overrides(json!({ "batch_size": 8, "ctx_size": 32768 })),
            Ok((Some(8), Some(32768)))
        );
        // Exact bounds are inclusive.
        assert_eq!(
            overrides(json!({ "batch_size": MAX_BATCH_SIZE_OVERRIDE })),
            Ok((Some(MAX_BATCH_SIZE_OVERRIDE), None))
        );
        assert_eq!(
            overrides(json!({ "ctx_size": MIN_CTX_SIZE_OVERRIDE })),
            Ok((None, Some(MIN_CTX_SIZE_OVERRIDE)))
        );
    }

    #[test]
    fn batch_size_beyond_i32_no_longer_wedges_the_scheduler() {
        // The historical case: 2^32 truncated to 0 slots via `as i32`.
        assert_eq!(
            overrides(json!({ "batch_size": 4_294_967_296u64 })),
            Err(StatusCode::BAD_REQUEST)
        );
    }

    #[test]
    fn out_of_range_overrides_are_rejected() {
        assert_eq!(
            overrides(json!({ "batch_size": 0 })),
            Err(StatusCode::BAD_REQUEST)
        );
        assert_eq!(
            overrides(json!({ "batch_size": MAX_BATCH_SIZE_OVERRIDE + 1 })),
            Err(StatusCode::BAD_REQUEST)
        );
        assert_eq!(
            overrides(json!({ "ctx_size": MIN_CTX_SIZE_OVERRIDE - 1 })),
            Err(StatusCode::BAD_REQUEST)
        );
        assert_eq!(
            overrides(json!({ "ctx_size": u32::MAX })),
            Err(StatusCode::BAD_REQUEST)
        );
    }

    #[test]
    fn a_present_but_unparseable_override_is_an_error_not_an_absence() {
        // Silently treating these as "not supplied" would hide a client bug
        // behind unchanged server behaviour.
        for body in [
            json!({ "batch_size": -1 }),
            json!({ "batch_size": "8" }),
            json!({ "batch_size": 1.5 }),
            json!({ "ctx_size": -4096 }),
            json!({ "ctx_size": "32768" }),
        ] {
            assert_eq!(
                overrides(body.clone()),
                Err(StatusCode::BAD_REQUEST),
                "expected 400 for {body}"
            );
        }
    }
}

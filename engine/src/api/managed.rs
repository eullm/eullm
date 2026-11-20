//! A server whose models only its administrator changes.
//!
//! Without `EULLM_ADMIN_KEY` every caller the API admits may download a model
//! (`/api/pull`, gigabytes of disk and bandwidth), delete one (`/api/delete`),
//! unload the one others are using (`/api/unload`, or `"keep_alive": 0` on a
//! request), and name any model in the folder, which loads it in place of the
//! one in memory. That is right for one person's machine, and wrong for a
//! server shared by an office, where whoever opens the chat may do all of it.
//!
//! With `EULLM_ADMIN_KEY` set, the server is **managed**:
//!
//! - downloading, deleting and unloading models need the admin key, in an
//!   `X-Admin-Key` header (or an in-process [`AdminGrant`]); the chat's Model
//!   hub asks for it;
//! - a request's `keep_alive` is honoured from an administrator only, and
//!   ignored otherwise: Ollama clients send one on every request, so it is
//!   dropped, not refused, and the server's own `--keep-alive` applies.
//!
//! Apart from the admin key, `EULLM_ALLOWED_MODELS` (a comma-separated list
//! of model names) limits the models a request may name, on any server:
//! another name is refused with 403 before anything is loaded, and the model
//! lists show only those. `auto` and a request that names no model are left
//! to the server's own choice. An administrator may name any.
//!
//! Both are read at startup and logged with their source (see `api::serve`).

use std::net::IpAddr;

use axum::extract::{ConnectInfo, State};
use axum::http::{Method, StatusCode};
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

use super::admin::{AdminGrant, require_admin};
use super::{AppState, model_identity_key};

/// The models requests may name, from `EULLM_ALLOWED_MODELS`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AllowedModels {
    /// None: any model. Names as written, for messages and lists.
    names: Option<Vec<String>>,
    source: String,
}

impl AllowedModels {
    /// From the environment, then `.env`.
    pub fn load() -> Self {
        let setting = crate::settings::var_with_source("EULLM_ALLOWED_MODELS");
        Self::resolve(setting.as_ref().map(|(v, s)| (v.as_str(), s.label())))
    }

    /// Pure, so the parsing is tested without the process environment.
    pub fn resolve(setting: Option<(&str, &str)>) -> Self {
        let Some((value, source)) = setting else {
            return Self {
                names: None,
                source: "default".into(),
            };
        };
        let names: Vec<String> = value
            .split(',')
            .map(str::trim)
            .filter(|n| !n.is_empty())
            .map(str::to_string)
            .collect();
        Self {
            names: (!names.is_empty()).then_some(names),
            source: format!("EULLM_ALLOWED_MODELS ({source})"),
        }
    }

    pub fn is_limited(&self) -> bool {
        self.names.is_some()
    }

    /// Whether a request may name `model`: as `model_names_match` compares
    /// names, so `qwen3:8b`, `qwen3-8b` and a path ending in
    /// `qwen3-8b.gguf` are one model.
    pub fn allows(&self, model: &str) -> bool {
        let Some(names) = &self.names else {
            return true;
        };
        let wanted = model_identity_key(&model.replace(':', "-"));
        names
            .iter()
            .any(|n| model_identity_key(&n.replace(':', "-")) == wanted)
    }

    /// For the startup log.
    pub fn describe(&self) -> String {
        match &self.names {
            None => "any model in the folder".into(),
            Some(names) => names.join(", "),
        }
    }

    pub fn source(&self) -> &str {
        &self.source
    }
}

/// What a request does to the server's models, by its route.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Route {
    /// Downloads, deletes or unloads: an administrator's.
    ChangesModels(&'static str),
    /// Names a model to answer with, and may carry a `keep_alive`.
    UsesModel,
    Other,
}

fn route_of(method: &Method, path: &str) -> Route {
    // The routers nest `/api` and `/v1`; the path here is the full one.
    match (method, path) {
        (&Method::POST, "/api/pull") => Route::ChangesModels("download models"),
        (&Method::DELETE, "/api/delete") => Route::ChangesModels("delete models"),
        (&Method::POST, "/api/unload") => Route::ChangesModels("unload models"),
        (
            &Method::POST,
            "/api/generate" | "/api/chat" | "/api/embed" | "/api/route" | "/v1/chat/completions"
            | "/v1/embeddings" | "/v1/systemone",
        ) => Route::UsesModel,
        _ => Route::Other,
    }
}

/// The middleware: see the module documentation. Runs inside the perimeter
/// (keys, addresses, origins), so whoever reaches it was let in.
pub async fn enforce_managed(
    State(state): State<std::sync::Arc<AppState>>,
    ConnectInfo(addr): ConnectInfo<std::net::SocketAddr>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let route = route_of(req.method(), req.uri().path());
    let managed = state.monitor_access.admin_enabled();
    let granted = req.extensions().get::<AdminGrant>().is_some();
    match route {
        Route::Other => next.run(req).await,
        Route::ChangesModels(what) if managed => {
            match require_admin(
                &state.monitor_access,
                addr.ip(),
                req.headers(),
                granted,
                what,
                "",
            ) {
                Ok(()) => next.run(req).await,
                Err(refusal) => refusal.into_response(),
            }
        }
        Route::ChangesModels(_) => next.run(req).await,
        Route::UsesModel => {
            if !managed && !state.allowed_models.is_limited() {
                return next.run(req).await;
            }
            let admin = granted || is_admin(&state, addr.ip(), req.headers());
            if admin {
                return next.run(req).await;
            }
            match vet_body(req, managed, &state.allowed_models).await {
                Ok(req) => next.run(req).await,
                Err(refusal) => refusal,
            }
        }
    }
}

/// Whether the request carries the right admin key. No key is no failure;
/// a wrong one counts towards the lock-out, as anywhere else.
fn is_admin(state: &AppState, ip: IpAddr, headers: &axum::http::HeaderMap) -> bool {
    match headers
        .get(super::admin::ADMIN_HEADER)
        .and_then(|v| v.to_str().ok())
    {
        Some(key) => state.monitor_access.check_admin(ip, Some(key)).is_ok(),
        None => false,
    }
}

/// Read the body, drop its `keep_alive` on a managed server, and refuse a
/// model that is not allowed. A body that is not a JSON object goes through
/// untouched: the handler answers it as it always has.
async fn vet_body(
    req: axum::extract::Request,
    managed: bool,
    allowed: &AllowedModels,
) -> Result<axum::extract::Request, Response> {
    let (mut parts, body) = req.into_parts();
    let bytes = match axum::body::to_bytes(body, super::MAX_BODY_BYTES).await {
        Ok(b) => b,
        Err(e) => {
            return Err((
                StatusCode::PAYLOAD_TOO_LARGE,
                axum::Json(json!({ "error": format!("could not read the body: {e}") })),
            )
                .into_response());
        }
    };
    let Ok(Value::Object(mut object)) = serde_json::from_slice::<Value>(&bytes) else {
        return Ok(axum::extract::Request::from_parts(
            parts,
            axum::body::Body::from(bytes),
        ));
    };
    if let Some(Value::String(model)) = object.get("model") {
        let model = model.trim();
        if !model.is_empty() && model != "auto" && !allowed.allows(model) {
            tracing::warn!(
                "Refused a request for {}: not among EULLM_ALLOWED_MODELS",
                crate::audit::sanitize_for_log(model)
            );
            return Err((
                StatusCode::FORBIDDEN,
                axum::Json(json!({
                    "error": format!(
                        "'{model}' is not available on this server; ask for one of: {}",
                        allowed.describe()
                    )
                })),
            )
                .into_response());
        }
    }
    if managed && object.remove("keep_alive").is_some() {
        tracing::debug!("Ignored a request's keep_alive: only an administrator sets it here");
        let rewritten = serde_json::to_vec(&Value::Object(object)).unwrap_or_else(|_| bytes.to_vec());
        parts.headers.remove(axum::http::header::CONTENT_LENGTH);
        return Ok(axum::extract::Request::from_parts(
            parts,
            axum::body::Body::from(rewritten),
        ));
    }
    Ok(axum::extract::Request::from_parts(
        parts,
        axum::body::Body::from(bytes),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_are_told_apart_by_method_and_path() {
        assert_eq!(route_of(&Method::POST, "/api/pull"), Route::ChangesModels("download models"));
        assert_eq!(route_of(&Method::DELETE, "/api/delete"), Route::ChangesModels("delete models"));
        assert_eq!(route_of(&Method::POST, "/api/unload"), Route::ChangesModels("unload models"));
        assert_eq!(route_of(&Method::POST, "/api/chat"), Route::UsesModel);
        assert_eq!(route_of(&Method::POST, "/v1/chat/completions"), Route::UsesModel);
        assert_eq!(route_of(&Method::GET, "/api/tags"), Route::Other);
        assert_eq!(route_of(&Method::GET, "/api/pull"), Route::Other);
    }

    #[test]
    fn the_allowed_models_match_however_they_are_written() {
        let allowed = AllowedModels::resolve(Some((" qwen3:8b , gemma-4-12b,", "environment")));
        assert!(allowed.is_limited());
        assert!(allowed.allows("qwen3-8b"));
        assert!(allowed.allows("qwen3:8b"));
        assert!(allowed.allows("/models/Qwen3-8B.gguf"));
        assert!(allowed.allows("gemma-4-12b"));
        assert!(!allowed.allows("qwen3-14b"));
        assert_eq!(allowed.source(), "EULLM_ALLOWED_MODELS (environment)");
        let any = AllowedModels::resolve(None);
        assert!(!any.is_limited() && any.allows("anything"));
        assert!(!AllowedModels::resolve(Some((" , ", ".env"))).is_limited());
    }

    fn request(body: &str) -> axum::extract::Request {
        axum::extract::Request::builder()
            .method(Method::POST)
            .uri("/api/chat")
            .header("content-length", body.len())
            .body(axum::body::Body::from(body.to_string()))
            .unwrap()
    }

    async fn body_of(req: axum::extract::Request) -> Value {
        let bytes = axum::body::to_bytes(req.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn a_managed_server_drops_keep_alive_and_keeps_the_rest() {
        let any = AllowedModels::default();
        let req = vet_body(request(r#"{"model":"m","keep_alive":0,"stream":false}"#), true, &any)
            .await
            .unwrap();
        assert!(req.headers().get("content-length").is_none());
        assert_eq!(body_of(req).await, json!({"model": "m", "stream": false}));
        // Not managed: untouched.
        let req = vet_body(request(r#"{"model":"m","keep_alive":0}"#), false, &any)
            .await
            .unwrap();
        assert_eq!(body_of(req).await, json!({"model": "m", "keep_alive": 0}));
    }

    #[tokio::test]
    async fn a_model_not_allowed_is_refused_and_auto_is_not() {
        let allowed = AllowedModels::resolve(Some(("gemma-4-12b", "environment")));
        let refused = vet_body(request(r#"{"model":"qwen3-8b"}"#), false, &allowed)
            .await
            .unwrap_err();
        assert_eq!(refused.status(), StatusCode::FORBIDDEN);
        for ok in [r#"{"model":"auto"}"#, r#"{"messages":[]}"#, r#"{"model":"gemma-4-12b"}"#, "not json"] {
            assert!(vet_body(request(ok), false, &allowed).await.is_ok(), "{ok}");
        }
    }
}

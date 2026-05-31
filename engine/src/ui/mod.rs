//! Embedded chat UI served by the engine on `/` (HTML + CSS + JS).
//!
//! Assets are baked into the binary at compile time via `include_str!` —
//! no filesystem dependency, no external runtime. The UI is a single-page
//! chat that talks to `/v1/chat/completions` (OpenAI-compatible, SSE) and
//! `/api/tags` (Ollama-compatible model list).
//!
//! Design tenets:
//!  - **Sovereign by default**: zero external resources (no CDN, no fonts,
//!    no analytics). Every byte the browser fetches comes from this binary.
//!  - **Minimal footprint**: ~15 KB total, no build step, no framework.
//!  - **CORS-friendly**: served from the same origin as the API, so no
//!    cross-origin headaches for browsers.

use axum::{
    http::header,
    response::{IntoResponse, Response},
    routing::get,
    Router,
};

const INDEX_HTML: &str = include_str!("index.html");
const STYLE_CSS: &str = include_str!("style.css");
const APP_JS: &str = include_str!("app.js");

/// Build the UI sub-router. Mount at `/` on the top-level router.
pub fn router<S: Clone + Send + Sync + 'static>() -> Router<S> {
    Router::new()
        .route("/", get(serve_index))
        .route("/eullm-ui/style.css", get(serve_css))
        .route("/eullm-ui/app.js", get(serve_js))
}

async fn serve_index() -> Response {
    asset(INDEX_HTML, "text/html; charset=utf-8")
}

async fn serve_css() -> Response {
    asset(STYLE_CSS, "text/css; charset=utf-8")
}

async fn serve_js() -> Response {
    asset(APP_JS, "application/javascript; charset=utf-8")
}

fn asset(body: &'static str, content_type: &'static str) -> Response {
    (
        [
            (header::CONTENT_TYPE, content_type),
            // Mild caching: assets change only on engine version bump,
            // but skip the cache during local dev by validating with ETag.
            (header::CACHE_CONTROL, "public, max-age=300"),
        ],
        body,
    )
        .into_response()
}

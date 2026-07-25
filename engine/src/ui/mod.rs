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
    Router,
    http::header,
    response::{IntoResponse, Response},
    routing::get,
};

const INDEX_HTML: &str = include_str!("index.html");
const STYLE_CSS: &str = include_str!("style.css");
const APP_JS: &str = include_str!("app.js");
const LOGO_DARK_PNG: &[u8] = include_bytes!("eullm-logo-dark.png");
const LOGO_LIGHT_PNG: &[u8] = include_bytes!("eullm-logo-light.png");

/// Build the UI sub-router. Mount at `/` on the top-level router.
pub fn router<S: Clone + Send + Sync + 'static>() -> Router<S> {
    Router::new()
        .route("/", get(serve_index))
        .route("/eullm-ui/style.css", get(serve_css))
        .route("/eullm-ui/app.js", get(serve_js))
        .route("/eullm-ui/logo-dark.png", get(serve_logo_dark))
        .route("/eullm-ui/logo-light.png", get(serve_logo_light))
}

async fn serve_index() -> Response {
    text_asset(INDEX_HTML, "text/html; charset=utf-8")
}

async fn serve_css() -> Response {
    text_asset(STYLE_CSS, "text/css; charset=utf-8")
}

async fn serve_js() -> Response {
    text_asset(APP_JS, "application/javascript; charset=utf-8")
}

async fn serve_logo_dark() -> Response {
    binary_asset(LOGO_DARK_PNG, "image/png")
}

async fn serve_logo_light() -> Response {
    binary_asset(LOGO_LIGHT_PNG, "image/png")
}

/// Content-Security-Policy for the chat UI.
///
/// The UI is built to load nothing from outside this binary — no CDN, no fonts,
/// no analytics — so a restrictive policy costs nothing and turns that design
/// property into something the browser enforces rather than something the
/// README asserts. `'unsafe-inline'` covers the small inline `<style>`/handlers
/// in `index.html`; `connect-src 'self'` keeps the page's fetches on the origin
/// that served it.
const CSP: &str = "default-src 'self'; \
                   script-src 'self' 'unsafe-inline'; \
                   style-src 'self' 'unsafe-inline'; \
                   img-src 'self' data:; \
                   connect-src 'self'; \
                   frame-ancestors 'none'; \
                   base-uri 'none'; \
                   form-action 'none'";

fn text_asset(body: &'static str, content_type: &'static str) -> Response {
    (
        [
            (header::CONTENT_TYPE, content_type),
            (header::CACHE_CONTROL, "public, max-age=300"),
            (header::CONTENT_SECURITY_POLICY, CSP),
            (header::X_CONTENT_TYPE_OPTIONS, "nosniff"),
            (header::REFERRER_POLICY, "no-referrer"),
        ],
        body,
    )
        .into_response()
}

fn binary_asset(body: &'static [u8], content_type: &'static str) -> Response {
    (
        [
            (header::CONTENT_TYPE, content_type),
            // Logos change at most on a release tag, safe to cache longer
            (header::CACHE_CONTROL, "public, max-age=86400"),
            (header::X_CONTENT_TYPE_OPTIONS, "nosniff"),
        ],
        body,
    )
        .into_response()
}

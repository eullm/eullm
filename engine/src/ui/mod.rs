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
const MONITOR_JS: &str = include_str!("monitor.js");
const ABOUT_JS: &str = include_str!("about.js");
const PRO_JS: &str = include_str!("pro.js");
const LOGO_DARK_PNG: &[u8] = include_bytes!("eullm-logo-dark.png");
const LOGO_LIGHT_PNG: &[u8] = include_bytes!("eullm-logo-light.png");
const FAVICON_ICO: &[u8] = include_bytes!("favicon.ico");

/// The page's own files, every route of [`router`]: the same for everyone,
/// they hold no data (the models, chats and settings all come from `/api`
/// and `/v1`). With API keys on, the chat's listener serves them without
/// one, since a browser fetching a script or a stylesheet cannot send a
/// header, and the page could not otherwise load (`api::enforce_auth`).
pub(crate) const PAGE_FILES: &[&str] = &[
    "/",
    "/eullm-ui/style.css",
    "/eullm-ui/app.js",
    "/eullm-ui/monitor.js",
    "/eullm-ui/about.js",
    "/eullm-ui/pro.js",
    "/eullm-ui/extension.js",
    "/eullm-ui/logo-dark.png",
    "/eullm-ui/logo-light.png",
    "/favicon.ico",
];

/// Whether `path` is one of [`PAGE_FILES`], exactly.
pub(crate) fn is_page_file(path: &str) -> bool {
    PAGE_FILES.contains(&path)
}

/// Build the UI sub-router. Mount at `/` on the top-level router.
pub fn router<S: Clone + Send + Sync + 'static>() -> Router<S> {
    Router::new()
        .route("/", get(serve_index))
        .route("/eullm-ui/style.css", get(serve_css))
        .route("/eullm-ui/app.js", get(serve_js))
        .route("/eullm-ui/monitor.js", get(serve_monitor_js))
        .route("/eullm-ui/about.js", get(serve_about_js))
        .route("/eullm-ui/pro.js", get(serve_pro_js))
        .route("/eullm-ui/extension.js", get(serve_extension_js))
        .route("/eullm-ui/logo-dark.png", get(serve_logo_dark))
        .route("/eullm-ui/logo-light.png", get(serve_logo_light))
        .route("/favicon.ico", get(serve_favicon))
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

async fn serve_monitor_js() -> Response {
    text_asset(MONITOR_JS, "application/javascript; charset=utf-8")
}

async fn serve_about_js() -> Response {
    text_asset(ABOUT_JS, "application/javascript; charset=utf-8")
}

/// Whether the administrator's menu points to EuLLM Pro, as decided at
/// startup (`set_pro_hints`). By default on in the binaries the release
/// publishes and in the Docker images, off in a build from source (see
/// `build.rs`), so that developers working on the code do not have it in
/// front of them; `EULLM_NO_PRO_HINTS` decides either way (`1` off, `0` on).
/// Always off when a program built on the engine adds a script of its own
/// to the page: the menu is then its own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProHints {
    pub shown: bool,
    /// Why, for the startup log.
    pub reason: String,
}

impl ProHints {
    /// From `EULLM_NO_PRO_HINTS` (environment, then `.env`) and whether an
    /// embedding program installed a chat script.
    pub fn load() -> Self {
        let setting = crate::settings::var_with_source("EULLM_NO_PRO_HINTS");
        Self::resolve(
            setting.as_ref().map(|(v, s)| (v.as_str(), s.label())),
            crate::extensions::chat_script().is_some(),
            env!("EULLM_PRO_HINTS_BUILD") == "on",
        )
    }

    /// Pure, so the precedence is tested without the process environment.
    /// `released`: this build shows them by default (`build.rs`).
    pub fn resolve(no_hints: Option<(&str, &str)>, extension_script: bool, released: bool) -> Self {
        if extension_script {
            return Self {
                shown: false,
                reason: "off: the program serving the page adds its own script".into(),
            };
        }
        let default = || {
            if released {
                Self {
                    shown: true,
                    reason: "on  [source: default for a released build; EULLM_NO_PRO_HINTS=1 \
                             turns them off]"
                        .into(),
                }
            } else {
                Self {
                    shown: false,
                    reason: "off  [source: default for a build from source; EULLM_NO_PRO_HINTS=0 \
                             shows them]"
                        .into(),
                }
            }
        };
        let Some((value, source)) = no_hints else {
            return default();
        };
        match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Self {
                shown: false,
                reason: format!("off  [source: EULLM_NO_PRO_HINTS ({source})]"),
            },
            "0" | "false" | "no" | "off" => Self {
                shown: true,
                reason: format!("on  [source: EULLM_NO_PRO_HINTS ({source})]"),
            },
            _ => {
                let mut fallback = default();
                fallback.reason = format!(
                    "{}  (EULLM_NO_PRO_HINTS ({source}) is neither 1 nor 0, ignored)",
                    fallback.reason
                );
                fallback
            }
        }
    }
}

/// What `/eullm-ui/pro.js` serves: (shown, managed). Not set (a router
/// built outside `api::serve`, as in tests): shown, not managed.
static PRO_HINTS: std::sync::OnceLock<(bool, bool)> = std::sync::OnceLock::new();

/// Called once by `api::serve` before the listeners start. `managed`:
/// `EULLM_ADMIN_KEY` is set, so only a page holding it is the
/// administrator's.
pub(crate) fn set_pro_hints(hints: &ProHints, managed: bool) {
    let _ = PRO_HINTS.set((hints.shown, managed));
}

/// The script, or an empty one when the pointers are off.
fn pro_script(shown: bool, managed: bool) -> String {
    if !shown {
        return "// EuLLM Pro pointers are off (EULLM_NO_PRO_HINTS, or the page has an extension).\n"
            .into();
    }
    PRO_JS.replacen(
        "/*MANAGED*/false",
        if managed { "true" } else { "false" },
        1,
    )
}

async fn serve_pro_js() -> Response {
    let (shown, managed) = PRO_HINTS.get().copied().unwrap_or((true, false));
    let mut response = (
        [(header::CONTENT_TYPE, "application/javascript; charset=utf-8")],
        pro_script(shown, managed),
    )
        .into_response();
    // Never cached: it changes with the server's settings, not its version.
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        axum::http::HeaderValue::from_static("no-store"),
    );
    response
}

/// What a program that builds on the engine adds to the page
/// (`crate::extensions::ExtensionRegistry::chat_script`): nothing in eullm.
async fn serve_extension_js() -> Response {
    let script = crate::extensions::chat_script().unwrap_or("// No extension.\n");
    let mut response = (
        [(header::CONTENT_TYPE, "application/javascript; charset=utf-8")],
        script.to_string(),
    )
        .into_response();
    // Never cached: it changes with the program serving the page, not with
    // this engine's version.
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        axum::http::HeaderValue::from_static("no-store"),
    );
    response
}

async fn serve_logo_dark() -> Response {
    binary_asset(LOGO_DARK_PNG, "image/png")
}

async fn serve_logo_light() -> Response {
    binary_asset(LOGO_LIGHT_PNG, "image/png")
}

async fn serve_favicon() -> Response {
    binary_asset(FAVICON_ICO, "image/x-icon")
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pro_hints_follow_the_build_unless_the_server_says_otherwise() {
        // By default: on in a released build, off in one from source.
        assert!(ProHints::resolve(None, false, true).shown);
        assert!(!ProHints::resolve(None, false, false).shown);
        // The setting decides either way.
        assert!(!ProHints::resolve(Some(("1", "environment")), false, true).shown);
        assert!(!ProHints::resolve(Some(("yes", ".env")), false, true).shown);
        assert!(ProHints::resolve(Some(("0", "environment")), false, false).shown);
        assert!(ProHints::resolve(Some(("off", ".env")), false, false).shown);
        // Anything else is ignored, and the log says so.
        let ignored = ProHints::resolve(Some(("maybe", "environment")), false, true);
        assert!(ignored.shown && ignored.reason.contains("ignored"), "{ignored:?}");
        // A program with its own script on the page: its menu, not ours.
        assert!(!ProHints::resolve(None, true, true).shown);
        assert!(!ProHints::resolve(Some(("0", "environment")), true, true).shown);
    }

    /// Every route the router serves is in `PAGE_FILES`, and nothing else:
    /// a path added to one and not the other is either a page file that
    /// needs a key, or a key-free path that serves nothing.
    #[tokio::test]
    async fn page_files_are_exactly_the_routes() {
        let base = {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            tokio::spawn(async move {
                let _ = axum::serve(listener, router::<()>()).await;
            });
            format!("http://{addr}")
        };
        for path in PAGE_FILES {
            let r = reqwest::get(format!("{base}{path}")).await.unwrap();
            assert_eq!(r.status(), 200, "{path}");
        }
        let source = include_str!("mod.rs");
        let routes = source.matches(".route(\"").count();
        assert_eq!(routes, PAGE_FILES.len(), "a route missing from PAGE_FILES");
        assert!(is_page_file("/eullm-ui/app.js"));
        assert!(!is_page_file("/eullm-ui/app.js/../../api/tags"));
        assert!(!is_page_file("/api/tags"));
        assert!(!is_page_file("/eullm-ui/"));
    }

    #[test]
    fn the_script_learns_whether_the_server_is_managed() {
        assert_eq!(PRO_JS.matches("/*MANAGED*/false").count(), 1);
        assert!(pro_script(true, true).contains("const MANAGED = true;"));
        assert!(pro_script(true, false).contains("const MANAGED = false;"));
        let off = pro_script(false, true);
        assert!(off.starts_with("//") && off.lines().count() == 1, "{off}");
    }
}

//! Which browser origins may talk to the API.
//!
//! # The hole this closes
//!
//! CORS used to be `allow_origin(Any)` on both listeners. Combined with the IP
//! allowlist that is not a small gap: a request issued by JavaScript on any page
//! the user happens to be visiting comes from the user's own machine, so its
//! source address *is* loopback and the allowlist is satisfied by construction.
//! With `Any`, the browser then also hands the response body back to that page.
//! Every endpoint was reachable that way, including `/api/unload` and a model
//! swap — a page could unload the model out from under a running job, or read
//! whatever the user was asking their local model.
//!
//! # Two controls, because CORS alone is not one
//!
//! CORS decides whether the *response* may be read. It does not stop the
//! request from being executed: a cross-origin `POST` with a simple content
//! type is sent, reaches the handler, and takes effect — the browser only
//! withholds the response afterwards. For endpoints with side effects that is
//! precisely backwards. So this module is used twice:
//!
//! * as the CORS allow-origin predicate, which is what makes legitimate
//!   browser frontends (the embedded chat UI, Open WebUI) work; and
//! * as an explicit check in front of every unsafe method, which *refuses*
//!   the request with 403 before the handler runs.
//!
//! Requests with no `Origin` header at all are untouched by both: that is every
//! non-browser client — curl, an Ollama SDK, a RAG pipeline — and CORS has
//! never applied to them. Breaking those would break the compatibility promise
//! for no security gain, since a program that can set arbitrary headers is not
//! constrained by an origin policy anyway.
//!
//! # Configuration
//!
//! `EULLM_ALLOWED_ORIGINS`, from the process environment first and the `.env`
//! file second (same precedence as every other perimeter setting). A
//! comma-separated list of exact origins:
//!
//! ```text
//! EULLM_ALLOWED_ORIGINS=https://chat.example.eu,http://192.168.7.10:8080
//! ```
//!
//! `*` restores the old permissive behaviour explicitly, for whoever needs it.
//!
//! With nothing configured, any **loopback** origin is allowed on any port
//! (`http://localhost:*`, `http://127.0.0.1:*`, `http://[::1]:*`). That keeps
//! the embedded chat UI and a locally hosted frontend working out of the box —
//! the common case — while still refusing `https://some-random-site.example`,
//! which is the case that mattered.

use std::path::Path;

/// A parsed allow-origin policy.
#[derive(Debug, Clone)]
pub struct AllowedOrigins {
    /// Exact origins that are allowed, lowercased. Empty when only the
    /// loopback default applies.
    exact: Vec<String>,
    /// `EULLM_ALLOWED_ORIGINS=*` — allow everything, explicitly requested.
    allow_any: bool,
    /// Allow any loopback origin regardless of port (the default posture).
    allow_loopback: bool,
    source: String,
}

impl AllowedOrigins {
    /// The default: loopback origins on any port, nothing else.
    fn loopback_only() -> Self {
        Self {
            exact: Vec::new(),
            allow_any: false,
            allow_loopback: true,
            source: "default (any loopback origin — nothing configured)".to_string(),
        }
    }

    /// Load from the environment, falling back to the `.env` file at `path`.
    pub fn load(path: &Path) -> Self {
        let env_spec = std::env::var("EULLM_ALLOWED_ORIGINS").ok();
        let file_contents = std::fs::read_to_string(path).ok();
        Self::resolve(
            env_spec.as_deref(),
            file_contents.as_deref(),
            &path.display().to_string(),
        )
    }

    /// Pure resolution step behind [`load`], so precedence is testable without
    /// mutating process environment variables.
    fn resolve(env_spec: Option<&str>, file_contents: Option<&str>, file_label: &str) -> Self {
        if let Some(spec) = env_spec.map(str::trim).filter(|s| !s.is_empty()) {
            return Self::parse(spec, "EULLM_ALLOWED_ORIGINS (environment)".to_string());
        }
        if let Some(contents) = file_contents
            && let Some(spec) = super::ip_allowlist::env_file_var(contents, "EULLM_ALLOWED_ORIGINS")
        {
            return Self::parse(&spec, format!("EULLM_ALLOWED_ORIGINS ({file_label})"));
        }
        Self::loopback_only()
    }

    fn parse(spec: &str, source: String) -> Self {
        let mut out = Self {
            exact: Vec::new(),
            allow_any: false,
            // A configured list replaces the default rather than adding to it,
            // except for loopback, which stays allowed so configuring a remote
            // frontend cannot lock the operator out of their own chat UI. Same
            // rule as the IP allowlist, where loopback also always survives.
            allow_loopback: true,
            source,
        };
        for entry in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            if entry == "*" {
                out.allow_any = true;
                continue;
            }
            out.exact
                .push(entry.trim_end_matches('/').to_ascii_lowercase());
        }
        out
    }

    /// Where the configuration came from — for the startup log.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Human-readable summary for the startup log.
    pub fn describe(&self) -> String {
        if self.allow_any {
            return "* (any origin — explicitly configured)".to_string();
        }
        let mut parts: Vec<String> = Vec::new();
        if self.allow_loopback {
            parts.push("any loopback origin".to_string());
        }
        parts.extend(self.exact.iter().cloned());
        parts.join(", ")
    }

    /// Whether `origin` (a browser `Origin` header value) is allowed.
    pub fn is_allowed(&self, origin: &str) -> bool {
        if self.allow_any {
            return true;
        }
        let origin = origin.trim().trim_end_matches('/').to_ascii_lowercase();
        // "null" is what a browser sends from a sandboxed iframe, a `file://`
        // page, or after some redirects. It identifies nothing, so it can never
        // match an allowlist entry.
        if origin.is_empty() || origin == "null" {
            return false;
        }
        if self.exact.contains(&origin) {
            return true;
        }
        self.allow_loopback && is_loopback_origin(&origin)
    }
}

/// Whether an origin points at the local machine, on any port.
///
/// Parsed by hand rather than with a URL crate: the input grammar is
/// `scheme://host[:port]` with no path, and hand-parsing keeps the check
/// readable — which matters more here than generality. Note the deliberate
/// strictness: `http://localhost.evil.example` must not match, so the host is
/// compared for equality, never by suffix.
fn is_loopback_origin(origin: &str) -> bool {
    let rest = match origin.split_once("://") {
        Some(("http", r)) | Some(("https", r)) => r,
        _ => return false,
    };
    // A path or credentials mean this is not a bare origin — refuse rather
    // than guess.
    if rest.contains('/') || rest.contains('@') {
        return false;
    }
    // An IPv6 literal is bracketed, so the colons inside it are part of the
    // address and only a colon *after* the closing bracket introduces a port.
    // Splitting on the last colon regardless would leave `[::1]:8080` unmatched.
    let host = if let Some(end) = rest.strip_prefix('[').and_then(|_| rest.find(']')) {
        let (host, tail) = rest.split_at(end + 1);
        if !tail.is_empty() && !is_port_suffix(tail) {
            return false;
        }
        host
    } else {
        rest.split_once(':')
            .filter(|(_, port)| is_port_suffix(&format!(":{port}")))
            .map(|(h, _)| h)
            .unwrap_or(rest)
    };
    matches!(host, "localhost" | "127.0.0.1" | "[::1]")
}

/// Whether `s` is exactly `:<digits>` — a port and nothing else.
fn is_port_suffix(s: &str) -> bool {
    s.strip_prefix(':')
        .is_some_and(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_allows_loopback_on_any_port_and_nothing_else() {
        let o = AllowedOrigins::loopback_only();
        assert!(o.is_allowed("http://localhost:11435"));
        assert!(o.is_allowed("http://127.0.0.1:3000"));
        assert!(o.is_allowed("http://[::1]:8080"));
        assert!(o.is_allowed("https://localhost"));
        assert!(!o.is_allowed("https://evil.example"));
        assert!(!o.is_allowed("http://192.168.1.10:8080"));
    }

    #[test]
    fn a_hostname_merely_containing_localhost_is_not_loopback() {
        // The failure mode of a suffix or `contains` check.
        let o = AllowedOrigins::loopback_only();
        assert!(!o.is_allowed("http://localhost.evil.example"));
        assert!(!o.is_allowed("http://notlocalhost"));
        assert!(!o.is_allowed("http://127.0.0.1.evil.example"));
        assert!(!o.is_allowed("http://evil.example#http://localhost"));
    }

    #[test]
    fn a_null_or_empty_origin_is_never_allowed() {
        // Sandboxed iframes and file:// pages send "null"; it identifies
        // nothing, so it must not satisfy any policy.
        let o = AllowedOrigins::loopback_only();
        assert!(!o.is_allowed("null"));
        assert!(!o.is_allowed(""));
        assert!(!o.is_allowed("   "));
        let o = AllowedOrigins::parse("http://localhost:1", "t".into());
        assert!(!o.is_allowed("null"));
    }

    #[test]
    fn non_http_schemes_are_not_loopback() {
        let o = AllowedOrigins::loopback_only();
        assert!(!o.is_allowed("file://"));
        assert!(!o.is_allowed("chrome-extension://abcdef"));
        assert!(!o.is_allowed("ws://localhost:11434"));
    }

    #[test]
    fn configured_origins_are_matched_exactly_and_case_insensitively() {
        let o = AllowedOrigins::parse("https://chat.example.eu", "test".into());
        assert!(o.is_allowed("https://chat.example.eu"));
        assert!(o.is_allowed("https://CHAT.Example.EU"));
        assert!(o.is_allowed("https://chat.example.eu/"));
        // Different scheme, port or subdomain are different origins.
        assert!(!o.is_allowed("http://chat.example.eu"));
        assert!(!o.is_allowed("https://chat.example.eu:8443"));
        assert!(!o.is_allowed("https://other.example.eu"));
        assert!(!o.is_allowed("https://evil-chat.example.eu"));
    }

    #[test]
    fn loopback_survives_a_configured_list() {
        // Configuring a remote frontend must not lock the operator out of the
        // chat UI on their own machine.
        let o = AllowedOrigins::parse("https://chat.example.eu", "test".into());
        assert!(o.is_allowed("http://localhost:11435"));
    }

    #[test]
    fn a_star_restores_the_permissive_behaviour_explicitly() {
        let o = AllowedOrigins::parse("*", "test".into());
        assert!(o.is_allowed("https://evil.example"));
        assert!(o.describe().contains("any origin"));
    }

    #[test]
    fn precedence_is_environment_then_env_file() {
        let o = AllowedOrigins::resolve(
            Some("https://from-env.example"),
            Some("EULLM_ALLOWED_ORIGINS=https://from-file.example\n"),
            ".env",
        );
        assert!(o.is_allowed("https://from-env.example"));
        assert!(!o.is_allowed("https://from-file.example"));
        assert!(o.source().contains("environment"));

        let o = AllowedOrigins::resolve(
            None,
            Some("EULLM_ALLOWED_ORIGINS=https://from-file.example\n"),
            ".env",
        );
        assert!(o.is_allowed("https://from-file.example"));
        assert!(o.source().contains(".env"));

        let o = AllowedOrigins::resolve(None, None, ".env");
        assert!(o.source().contains("nothing configured"));
    }

    #[test]
    fn describe_never_claims_more_than_is_allowed() {
        let o = AllowedOrigins::resolve(Some("https://a.example"), None, ".env");
        let d = o.describe();
        assert!(d.contains("any loopback origin"), "{d}");
        assert!(d.contains("https://a.example"), "{d}");
    }
}

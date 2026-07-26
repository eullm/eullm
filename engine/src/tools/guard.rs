//! Fetch policy for the web tool: what may be requested, and how much of it.
//!
//! # Threat model
//!
//! With `--web`, a URL appearing in a **user message** is fetched by the server
//! and its content is injected into the prompt. The URL is therefore attacker-
//! controlled in every deployment where the person chatting is not the person
//! running the engine — a shared instance, a RAG frontend, anything behind
//! Docker. Three separate problems follow, and the unguarded fetch had all
//! three:
//!
//! 1. **SSRF.** `http://169.254.169.254/latest/meta-data/` on a cloud instance
//!    returns credentials; `http://127.0.0.1:11434/api/unload` reaches the
//!    engine's own admin surface from inside the allowlist; `http://10.0.0.5/`
//!    reaches whatever else is on the private network. The response is then
//!    injected into the prompt, so the model reads it out — the request is not
//!    blind.
//! 2. **Unbounded body.** `resp.text()` buffers whatever the server sends. A
//!    multi-gigabyte response is an out-of-memory kill of the whole engine, so
//!    one chat message takes down every other session on the process.
//! 3. **Redirects.** Validating the URL and then letting the client follow
//!    redirects with the default policy validates nothing: `https://ok.example`
//!    can 302 to `http://169.254.169.254/` and the check has been bypassed.
//!
//! # What this module does
//!
//! * `https` only, unless `EULLM_WEB_ALLOW_HTTP=1`.
//! * The host is resolved, and **every** address it resolves to must be public.
//!   All of them, not just the first: a name with an A record for a public
//!   address and another for `10.0.0.5` would otherwise be a coin flip.
//! * The connection is then pinned to the address that was validated
//!   (`ClientBuilder::resolve`), so the name cannot resolve to something else
//!   between the check and the connection — DNS rebinding.
//! * Redirects are disabled at the client and followed manually, re-running the
//!   whole check on each hop, up to [`MAX_REDIRECTS`].
//! * The body is read in chunks and abandoned at [`MAX_BODY_BYTES`].
//! * Only textual content types are accepted — a 2 MiB binary injected into a
//!   prompt is tokenised noise, so refusing it costs nothing and removes a
//!   class of surprises.
//! * `EULLM_WEB_ALLOWED_DOMAINS`, when set, restricts fetching to those domains
//!   and their subdomains. For an enterprise deployment this is the defensible
//!   shape of the feature: an allowlist of sources, not the whole internet.
//!
//! `EULLM_WEB_ALLOW_PRIVATE_HOSTS=1` disables the address check for the case of
//! an intranet documentation server that is the whole point of the deployment.
//! It is a deliberate, logged opt-out, and it is not implied by anything else.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::time::Duration;

use futures_util::StreamExt;

/// Hard cap on a fetched body. Well above any article and far below what would
/// threaten the process: the content budget in `super::fetch_for_context`
/// truncates to the model's context window anyway, so a larger download could
/// not be used even if it were kept.
pub const MAX_BODY_BYTES: usize = 4 * 1024 * 1024;

/// Redirect hops followed manually, each fully re-validated.
pub const MAX_REDIRECTS: usize = 3;

/// Per-request timeout, applied to the whole exchange including the body.
const FETCH_TIMEOUT: Duration = Duration::from_secs(15);

/// The configured fetch policy.
#[derive(Debug, Clone, Default)]
pub struct WebPolicy {
    /// Domain suffixes that may be fetched. Empty = any public host.
    allowed_domains: Vec<String>,
    allow_http: bool,
    allow_private_hosts: bool,
}

impl WebPolicy {
    /// Read the policy from the process environment.
    pub fn from_env() -> Self {
        Self::resolve(
            std::env::var("EULLM_WEB_ALLOWED_DOMAINS").ok().as_deref(),
            std::env::var("EULLM_WEB_ALLOW_HTTP").ok().as_deref(),
            std::env::var("EULLM_WEB_ALLOW_PRIVATE_HOSTS")
                .ok()
                .as_deref(),
        )
    }

    /// Pure resolution step behind [`from_env`], so the flags are testable
    /// without mutating process environment variables.
    fn resolve(
        domains: Option<&str>,
        allow_http: Option<&str>,
        allow_private: Option<&str>,
    ) -> Self {
        Self {
            allowed_domains: domains
                .unwrap_or("")
                .split(',')
                .map(|d| d.trim().trim_start_matches('.').to_ascii_lowercase())
                .filter(|d| !d.is_empty())
                .collect(),
            allow_http: is_truthy(allow_http),
            allow_private_hosts: is_truthy(allow_private),
        }
    }

    /// One line for the startup log, so the posture is visible rather than
    /// assumed.
    pub fn describe(&self) -> String {
        let scope = if self.allowed_domains.is_empty() {
            "any public host".to_string()
        } else {
            self.allowed_domains.join(", ")
        };
        let mut notes = Vec::new();
        if self.allow_http {
            notes.push("plain http allowed");
        }
        if self.allow_private_hosts {
            notes.push("private/loopback hosts allowed — SSRF protection off");
        }
        if notes.is_empty() {
            scope
        } else {
            format!("{scope}  [{}]", notes.join("; "))
        }
    }

    /// Check the scheme and host of a URL against the policy, without touching
    /// the network.
    fn check_static(&self, url: &reqwest::Url) -> Result<(), String> {
        match url.scheme() {
            "https" => {}
            "http" if self.allow_http => {}
            "http" => {
                return Err(
                    "only https URLs are fetched (set EULLM_WEB_ALLOW_HTTP=1 to allow http)"
                        .to_string(),
                );
            }
            other => return Err(format!("unsupported URL scheme '{other}'")),
        }
        let Some(host) = url.host_str() else {
            return Err("URL has no host".to_string());
        };
        if !self.allowed_domains.is_empty() && !self.domain_allowed(host) {
            return Err(format!(
                "host '{}' is not in EULLM_WEB_ALLOWED_DOMAINS",
                sanitize(host)
            ));
        }
        Ok(())
    }

    /// Whether `host` matches a configured domain exactly or as a subdomain.
    ///
    /// Matched on label boundaries, never by plain suffix: `evil-example.com`
    /// must not match a configured `example.com`.
    fn domain_allowed(&self, host: &str) -> bool {
        let host = host.trim_end_matches('.').to_ascii_lowercase();
        self.allowed_domains
            .iter()
            .any(|d| host == *d || host.ends_with(&format!(".{d}")))
    }
}

/// Whether an environment flag is set to something meaning "yes".
fn is_truthy(v: Option<&str>) -> bool {
    matches!(
        v.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

/// Strip control characters from host/URL text before it goes into an error
/// message that will be logged — same reasoning as `audit::sanitize_for_log`.
fn sanitize(s: &str) -> String {
    s.chars().filter(|c| !c.is_control()).take(200).collect()
}

/// Whether an address is safe to fetch from: globally routable, and not
/// pointing back at the host, the local network, or cloud metadata.
///
/// Written out rather than using the unstable `is_global()`: the ranges that
/// matter here are few and the explicit list is auditable against RFC 6890.
pub fn is_public_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_public_v4(v4),
        IpAddr::V6(v6) => {
            // Judge the address in its own right first. `::1` and `::` sit
            // inside the IPv4-compatible `::/96` range, so unwrapping them
            // first would hand `is_public_v4` the nonsense `0.0.0.1` — which
            // looks public — and let loopback through.
            if !is_public_v6(v6) {
                return false;
            }
            // An IPv6 address carrying an IPv4 one is exactly as dangerous as
            // the address it carries: ::ffff:169.254.169.254 reaches the same
            // metadata service. Unwrap before judging.
            match embedded_v4(v6) {
                Some(v4) => is_public_v4(v4),
                None => true,
            }
        }
    }
}

fn is_public_v4(ip: Ipv4Addr) -> bool {
    let o = ip.octets();
    !(ip.is_unspecified()
        || ip.is_loopback()
        || ip.is_private()
        || ip.is_link_local()
        || ip.is_multicast()
        || ip.is_broadcast()
        // 0.0.0.0/8 — "this network", RFC 6890. Never a valid destination, and
        // it is also what the low 32 bits of `::1` decode to.
        || o[0] == 0
        // 100.64.0.0/10 — carrier-grade NAT, RFC 6598. Not private by the
        // std definition, but it is not the public internet either.
        || (o[0] == 100 && (64..128).contains(&o[1]))
        // 192.0.0.0/24 — IETF protocol assignments.
        || (o[0] == 192 && o[1] == 0 && o[2] == 0)
        // 198.18.0.0/15 — benchmarking, RFC 2544.
        || (o[0] == 198 && (o[1] == 18 || o[1] == 19))
        // 240.0.0.0/4 — reserved.
        || o[0] >= 240)
}

fn is_public_v6(ip: Ipv6Addr) -> bool {
    let s = ip.segments();
    !(ip.is_unspecified()
        || ip.is_loopback()
        || ip.is_multicast()
        // fc00::/7 — unique local addresses.
        || (s[0] & 0xfe00) == 0xfc00
        // fe80::/10 — link-local.
        || (s[0] & 0xffc0) == 0xfe80)
}

/// The IPv4 address embedded in an IPv6 one, for the forms that carry one:
/// IPv4-mapped (`::ffff:0:0/96`), IPv4-compatible (`::/96`), NAT64's
/// well-known prefix (`64:ff9b::/96`) and 6to4 (`2002::/16`).
fn embedded_v4(ip: Ipv6Addr) -> Option<Ipv4Addr> {
    let s = ip.segments();
    let last_two = |s: [u16; 8]| {
        Ipv4Addr::new(
            (s[6] >> 8) as u8,
            (s[6] & 0xff) as u8,
            (s[7] >> 8) as u8,
            (s[7] & 0xff) as u8,
        )
    };
    // ::ffff:a.b.c.d and ::a.b.c.d (excluding :: and ::1, handled elsewhere).
    if s[0..5] == [0, 0, 0, 0, 0] && (s[5] == 0xffff || s[5] == 0) {
        let v4 = last_two(s);
        if !v4.is_unspecified() {
            return Some(v4);
        }
    }
    // 64:ff9b::/96 — NAT64.
    if s[0] == 0x0064 && s[1] == 0xff9b && s[2..6] == [0, 0, 0, 0] {
        return Some(last_two(s));
    }
    // 2002:a.b.c.d::/48 — 6to4 encodes the v4 address in segments 1 and 2.
    if s[0] == 0x2002 {
        return Some(Ipv4Addr::new(
            (s[1] >> 8) as u8,
            (s[1] & 0xff) as u8,
            (s[2] >> 8) as u8,
            (s[2] & 0xff) as u8,
        ));
    }
    None
}

/// Resolve `url`'s host and return the address to connect to, after checking
/// that **every** address it resolves to is acceptable.
async fn resolve_and_check(url: &reqwest::Url, policy: &WebPolicy) -> Result<SocketAddr, String> {
    let host = url.host_str().ok_or("URL has no host")?;
    let port = url
        .port_or_known_default()
        .ok_or_else(|| "URL has no port and no default for its scheme".to_string())?;

    // An address literal needs no DNS, and must not get any: `lookup_host` on a
    // literal is a no-op at best, and the check has to happen either way.
    // `host_str` brackets IPv6 literals, so strip those before parsing.
    let literal = host
        .strip_prefix('[')
        .and_then(|h| h.strip_suffix(']'))
        .unwrap_or(host)
        .parse::<IpAddr>()
        .ok();
    let addrs: Vec<SocketAddr> = match literal {
        Some(ip) => vec![SocketAddr::from((ip, port))],
        None => tokio::net::lookup_host((host, port))
            .await
            .map_err(|e| format!("could not resolve '{}': {e}", sanitize(host)))?
            .collect(),
    };

    if addrs.is_empty() {
        return Err(format!("'{}' resolved to no addresses", sanitize(host)));
    }
    if !policy.allow_private_hosts {
        // Every address must pass. A name resolving to one public and one
        // private address would otherwise succeed or fail depending on
        // ordering, which is the kind of intermittent hole nobody finds.
        for addr in &addrs {
            if !is_public_ip(addr.ip()) {
                return Err(format!(
                    "'{}' resolves to the non-public address {} — refusing to fetch \
                     (set EULLM_WEB_ALLOW_PRIVATE_HOSTS=1 if this is an intranet host \
                     you intend to reach)",
                    sanitize(host),
                    addr.ip()
                ));
            }
        }
    }
    Ok(addrs[0])
}

/// Fetch a URL under `policy` and return the decoded body text.
///
/// Redirects are followed manually so each hop is re-validated; the body is
/// truncated at [`MAX_BODY_BYTES`].
pub async fn fetch_text(url: &str, policy: &WebPolicy) -> Result<String, String> {
    let mut current =
        reqwest::Url::parse(url).map_err(|e| format!("invalid URL '{}': {e}", sanitize(url)))?;

    for hop in 0..=MAX_REDIRECTS {
        policy.check_static(&current)?;
        let addr = resolve_and_check(&current, policy).await?;
        let host = current.host_str().unwrap_or_default().to_string();

        // Pin the connection to the address we just validated, so the name
        // cannot resolve to something else between the check and the connect.
        let client = reqwest::Client::builder()
            .timeout(FETCH_TIMEOUT)
            .redirect(reqwest::redirect::Policy::none())
            .user_agent("Mozilla/5.0 (compatible; EULLM; +https://eullm.eu)")
            .resolve(&host, addr)
            .build()
            .map_err(|e| format!("HTTP client error: {e}"))?;

        let resp = client
            .get(current.clone())
            .send()
            .await
            .map_err(|e| format!("fetch error for {}: {e}", sanitize(current.as_str())))?;

        let status = resp.status();
        if status.is_redirection() {
            if hop == MAX_REDIRECTS {
                return Err(format!("too many redirects (more than {MAX_REDIRECTS})"));
            }
            let location = resp
                .headers()
                .get(reqwest::header::LOCATION)
                .and_then(|v| v.to_str().ok())
                .ok_or_else(|| format!("HTTP {status} with no usable Location header"))?;
            // Resolve relative redirects against the current URL, then loop —
            // the next iteration re-runs every check on the new target.
            current = current
                .join(location)
                .map_err(|e| format!("invalid redirect target: {e}"))?;
            continue;
        }
        if !status.is_success() {
            return Err(format!(
                "HTTP {status} fetching {}",
                sanitize(current.as_str())
            ));
        }

        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        if !is_textual(&content_type) {
            return Err(format!(
                "refusing to inject non-text content (Content-Type: {})",
                sanitize(if content_type.is_empty() {
                    "absent"
                } else {
                    &content_type
                })
            ));
        }

        let bytes = read_capped(resp).await?;
        return Ok(decode(&bytes, &content_type));
    }
    Err("too many redirects".to_string())
}

/// Whether a `Content-Type` is something worth turning into prompt text.
///
/// An absent Content-Type is refused rather than assumed textual: guessing is
/// how a binary ends up tokenised into a prompt.
fn is_textual(content_type: &str) -> bool {
    let mime = content_type
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    mime.starts_with("text/")
        || matches!(
            mime.as_str(),
            "application/xhtml+xml"
                | "application/xml"
                | "application/json"
                | "application/ld+json"
        )
}

/// Read a response body in chunks, stopping at [`MAX_BODY_BYTES`].
///
/// Streaming rather than `resp.text()` is the whole point: `text()` buffers
/// whatever the far end sends, so a hostile or merely broken server could
/// exhaust the process's memory — and it would take every concurrent session
/// with it, not just the one that fetched the URL.
async fn read_capped(resp: reqwest::Response) -> Result<Vec<u8>, String> {
    // Trust the declared length only to pre-allocate, and only up to the cap:
    // Content-Length is attacker-controlled and may be a lie in either
    // direction.
    let hint = resp
        .content_length()
        .map(|n| (n as usize).min(MAX_BODY_BYTES))
        .unwrap_or(64 * 1024);
    let mut out: Vec<u8> = Vec::with_capacity(hint);
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("read error: {e}"))?;
        let remaining = MAX_BODY_BYTES.saturating_sub(out.len());
        if chunk.len() >= remaining {
            out.extend_from_slice(&chunk[..remaining]);
            tracing::warn!(
                "web fetch truncated at {} bytes — the page is larger than the cap",
                MAX_BODY_BYTES
            );
            break;
        }
        out.extend_from_slice(&chunk);
    }
    Ok(out)
}

/// Decode bytes to text using the charset from `Content-Type`, falling back to
/// UTF-8. Never fails: undecodable bytes become replacement characters, which
/// is the right trade for prompt text.
fn decode(bytes: &[u8], content_type: &str) -> String {
    let charset = content_type
        .split(';')
        .skip(1)
        .filter_map(|p| p.trim().strip_prefix("charset="))
        .map(|c| c.trim_matches('"').to_ascii_lowercase())
        .next();
    let encoding = charset
        .as_deref()
        .and_then(|c| encoding_rs::Encoding::for_label(c.as_bytes()))
        .unwrap_or(encoding_rs::UTF_8);
    encoding.decode(bytes).0.into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── SSRF: the address check ─────────────────────────────────────────────

    #[test]
    fn cloud_metadata_and_loopback_are_not_public() {
        for ip in [
            "169.254.169.254", // AWS/GCP/Azure metadata
            "127.0.0.1",
            "127.0.0.53", // systemd-resolved
            "0.0.0.0",
            "10.0.0.5",
            "172.16.0.1",
            "172.31.255.254",
            "192.168.1.1",
            "100.64.0.1", // CGNAT
            "192.0.0.1",  // IETF protocol assignments
            "198.18.0.1", // benchmarking
            "224.0.0.1",  // multicast
            "255.255.255.255",
            "240.0.0.1",
        ] {
            assert!(
                !is_public_ip(ip.parse().unwrap()),
                "{ip} must not be considered public"
            );
        }
    }

    #[test]
    fn ordinary_public_addresses_are_public() {
        for ip in ["1.1.1.1", "93.184.216.34", "203.0.113.5", "2606:4700::1111"] {
            assert!(is_public_ip(ip.parse().unwrap()), "{ip} should be public");
        }
    }

    #[test]
    fn ipv6_forms_carrying_an_ipv4_address_are_judged_by_that_address() {
        // The bypass this guards: ::ffff:169.254.169.254 reaches exactly the
        // same metadata service as the bare IPv4 address.
        for ip in [
            "::ffff:169.254.169.254",
            "::ffff:127.0.0.1",
            "::ffff:10.0.0.1",
            "64:ff9b::169.254.169.254", // NAT64
            "2002:a9fe:a9fe::",         // 6to4 wrapping 169.254.169.254
        ] {
            assert!(
                !is_public_ip(ip.parse().unwrap()),
                "{ip} must not be considered public"
            );
        }
        assert!(is_public_ip("::ffff:1.1.1.1".parse().unwrap()));
    }

    #[test]
    fn ipv6_local_ranges_are_not_public() {
        for ip in ["::1", "::", "fc00::1", "fd12:3456::1", "fe80::1", "ff02::1"] {
            assert!(
                !is_public_ip(ip.parse().unwrap()),
                "{ip} must not be considered public"
            );
        }
    }

    // ── scheme and domain policy ────────────────────────────────────────────

    fn url(u: &str) -> reqwest::Url {
        reqwest::Url::parse(u).unwrap()
    }

    #[test]
    fn plain_http_is_refused_unless_opted_in() {
        let p = WebPolicy::default();
        assert!(p.check_static(&url("http://example.com/")).is_err());
        assert!(p.check_static(&url("https://example.com/")).is_ok());

        let p = WebPolicy::resolve(None, Some("1"), None);
        assert!(p.check_static(&url("http://example.com/")).is_ok());
    }

    #[test]
    fn non_web_schemes_are_refused() {
        let p = WebPolicy::resolve(None, Some("1"), None);
        for u in [
            "ftp://example.com/x",
            "file:///etc/passwd",
            "data:text/plain,x",
        ] {
            let parsed = reqwest::Url::parse(u).unwrap();
            assert!(p.check_static(&parsed).is_err(), "{u} should be refused");
        }
    }

    #[test]
    fn the_domain_allowlist_matches_on_label_boundaries() {
        let p = WebPolicy::resolve(Some("example.com, .docs.example.eu"), None, None);
        assert!(p.check_static(&url("https://example.com/a")).is_ok());
        assert!(p.check_static(&url("https://www.example.com/a")).is_ok());
        assert!(p.check_static(&url("https://docs.example.eu/a")).is_ok());
        assert!(p.check_static(&url("https://it.docs.example.eu/a")).is_ok());
        // The classic suffix-matching bug.
        assert!(p.check_static(&url("https://evil-example.com/a")).is_err());
        assert!(
            p.check_static(&url("https://example.com.evil.net/a"))
                .is_err()
        );
        assert!(p.check_static(&url("https://other.eu/a")).is_err());
    }

    #[test]
    fn an_empty_allowlist_means_any_public_host() {
        let p = WebPolicy::default();
        assert!(p.check_static(&url("https://anything.example/")).is_ok());
        assert!(p.describe().contains("any public host"));
    }

    #[test]
    fn describe_says_when_ssrf_protection_is_off() {
        let p = WebPolicy::resolve(None, None, Some("yes"));
        assert!(
            p.describe().contains("SSRF protection off"),
            "{}",
            p.describe()
        );
    }

    #[test]
    fn truthy_flag_spellings() {
        for v in ["1", "true", "TRUE", "yes", "on", " 1 "] {
            assert!(is_truthy(Some(v)), "{v:?} should be truthy");
        }
        for v in ["0", "false", "no", "", "maybe"] {
            assert!(!is_truthy(Some(v)), "{v:?} should not be truthy");
        }
        assert!(!is_truthy(None));
    }

    // ── an IP literal in the URL must be checked directly ───────────────────

    #[tokio::test]
    async fn a_literal_private_address_is_refused_without_dns() {
        let p = WebPolicy::resolve(None, Some("1"), None);
        for u in [
            "http://127.0.0.1:11434/api/unload",
            "http://169.254.169.254/latest/meta-data/",
            "http://[::1]:11434/api/tags",
            "http://[::ffff:169.254.169.254]/",
        ] {
            let parsed = reqwest::Url::parse(u).unwrap();
            let err = resolve_and_check(&parsed, &p).await.unwrap_err();
            assert!(err.contains("non-public"), "{u} → {err}");
        }
    }

    #[tokio::test]
    async fn the_opt_out_allows_a_private_literal() {
        let p = WebPolicy::resolve(None, Some("1"), Some("1"));
        let parsed = reqwest::Url::parse("http://127.0.0.1:9/x").unwrap();
        assert!(resolve_and_check(&parsed, &p).await.is_ok());
    }

    // ── content type and decoding ───────────────────────────────────────────

    #[test]
    fn only_textual_content_types_are_accepted() {
        for ct in [
            "text/html",
            "text/html; charset=utf-8",
            "text/plain",
            "TEXT/HTML",
            "application/json",
            "application/xhtml+xml",
        ] {
            assert!(is_textual(ct), "{ct} should be accepted");
        }
        for ct in [
            "",
            "application/pdf",
            "application/octet-stream",
            "image/png",
            "video/mp4",
            "application/zip",
        ] {
            assert!(!is_textual(ct), "{ct} should be refused");
        }
    }

    #[test]
    fn the_declared_charset_is_honoured_and_bad_bytes_never_panic() {
        // 'à' in latin-1 is a bare 0xE0, which is not valid UTF-8.
        let latin1 = vec![b'c', b'a', b'f', b'\xe8'];
        assert_eq!(decode(&latin1, "text/html; charset=iso-8859-1"), "cafè");
        // Same bytes without a charset: replacement char, not a panic.
        assert!(decode(&latin1, "text/html").contains('\u{fffd}'));
        assert_eq!(decode(b"ok", "text/plain"), "ok");
        assert_eq!(decode(b"", "text/plain"), "");
        // A quoted charset label is common in the wild.
        assert_eq!(decode(&latin1, "text/html; charset=\"iso-8859-1\""), "cafè");
    }

    #[test]
    fn error_messages_do_not_carry_control_characters_into_logs() {
        // A URL with a newline could otherwise forge a log line.
        let p = WebPolicy::resolve(Some("example.com"), None, None);
        let parsed = reqwest::Url::parse("https://evil.test/").unwrap();
        let err = p.check_static(&parsed).unwrap_err();
        assert!(!err.contains('\n'));
        assert_eq!(sanitize("a\nb\tc"), "abc");
    }
}

//! Who may administer the Monitor's history, and who may scrape `/metrics`.
//!
//! Reading the Monitor is for whoever may use the API, as the chat is. Three
//! things are not:
//!
//! - **Downloading, deleting and unloading models**, and a request's
//!   `keep_alive`, once `EULLM_ADMIN_KEY` is set: see `api::managed`.
//! - **Changing how long the history is kept, and deleting it**
//!   (`PUT /api/monitor/storage`, `DELETE /api/monitor/history`), which need
//!   `EULLM_ADMIN_KEY` in an `X-Admin-Key` header. The API keys cannot be that
//!   control: on a local server they are off, so everyone who opens the chat
//!   is admitted, and where they are on the chat page holds one in the
//!   browser. Without `EULLM_ADMIN_KEY` the history is administered from the
//!   server's own settings only (`EULLM_METRICS_RETENTION_DAYS`, the files).
//! - **`GET /metrics`**, Prometheus's format, which needs
//!   `EULLM_METRICS_TOKEN` as `Authorization: Bearer <token>` and is not
//!   served without it. A token of its own, read-only: the scraper's
//!   configuration lives on another machine, and must not be able to delete
//!   anything.
//!
//! Both are perimeter settings (environment first, then `.env`; or a file
//! named by `…_FILE`, for Docker secrets), at least 16 characters, compared
//! as SHA-256 digests in fixed time. One set but unusable stops the server,
//! as `EULLM_API_KEYS` does: whoever set a control must not find it off.
//! Five wrong attempts from one address in a quarter of an hour lock that
//! address out of both for a quarter of an hour, and every refusal and every
//! change is logged with the address.

use std::collections::HashMap;
use std::net::IpAddr;
use std::path::Path;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

use super::auth::{MIN_SECRET_LEN, constant_time_eq};

/// The header the admin key comes in.
pub const ADMIN_HEADER: &str = "x-admin-key";
/// Wrong attempts from one address before it is locked out.
const MAX_FAILURES: u32 = 5;
/// How long the failures are counted over, and how long a lock lasts.
const WINDOW: Duration = Duration::from_secs(15 * 60);
/// Addresses remembered at most: past this, the ones with nothing to hold
/// against them are forgotten.
const MAX_TRACKED: usize = 10_000;

/// A secret, kept as its digest, and where it came from.
#[derive(Debug, Clone)]
struct Secret {
    sha256: [u8; 32],
    source: String,
}

/// Why a request was not let through.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// The secret is not configured: this is not offered at all.
    Disabled,
    /// None was presented.
    Missing,
    /// The wrong one was.
    Wrong,
    /// Too many wrong ones from this address.
    Locked { retry_after_s: u64 },
}

#[derive(Debug, Default)]
struct Failures {
    count: u32,
    first: Option<Instant>,
    locked_until: Option<Instant>,
}

/// The two secrets, and the wrong attempts of each address.
#[derive(Debug)]
pub struct MonitorAccess {
    admin: Option<Secret>,
    metrics: Option<Secret>,
    failures: Mutex<HashMap<IpAddr, Failures>>,
}

impl MonitorAccess {
    /// Neither secret: administration and `/metrics` off.
    #[cfg(test)]
    pub fn disabled() -> Self {
        Self::new(None, None)
    }

    fn new(admin: Option<Secret>, metrics: Option<Secret>) -> Self {
        Self {
            admin,
            metrics,
            failures: Mutex::new(HashMap::new()),
        }
    }

    /// From the environment, the `…_FILE` it names, then `.env`. `Err` when
    /// one is set but unusable: the server does not start.
    pub fn load(env_file: &Path) -> Result<Self, String> {
        let env_file_contents = std::fs::read_to_string(env_file).ok();
        let label = env_file.display().to_string();
        let read = |name: &str| -> Result<Option<Secret>, String> {
            let file_var = format!("{name}_FILE");
            let file = std::env::var(&file_var)
                .ok()
                .map(|p| p.trim().to_string())
                .filter(|p| !p.is_empty())
                .map(|p| {
                    std::fs::read_to_string(&p)
                        .map(|c| (p.clone(), c))
                        .map_err(|e| format!("{file_var} is set to '{p}' but unreadable: {e}"))
                })
                .transpose()?;
            resolve(
                name,
                std::env::var(name).ok().as_deref(),
                file.as_ref().map(|(p, c)| (p.as_str(), c.as_str())),
                env_file_contents.as_deref(),
                &label,
            )
        };
        Ok(Self::new(
            read("EULLM_ADMIN_KEY")?,
            read("EULLM_METRICS_TOKEN")?,
        ))
    }

    /// With the two secrets as given: for the tests.
    #[cfg(test)]
    pub fn from_secrets(admin: Option<&str>, metrics: Option<&str>) -> Self {
        let secret = |s: &str| Secret {
            sha256: digest(s),
            source: "test".into(),
        };
        Self::new(admin.map(secret), metrics.map(secret))
    }

    pub fn admin_enabled(&self) -> bool {
        self.admin.is_some()
    }

    /// For the startup log: whether each is on, and from where. Never the
    /// secrets.
    pub fn describe(&self) -> (String, String) {
        let on = |s: &Option<Secret>, off: &str| {
            s.as_ref()
                .map_or_else(|| off.to_string(), |s| format!("on  [source: {}]", s.source))
        };
        (
            on(
                &self.admin,
                "off (EULLM_ADMIN_KEY not set): whoever may use the API may download, \
                 delete and unload models; the Monitor's history can be read but not \
                 deleted or its retention changed over the API",
            ),
            on(
                &self.metrics,
                "off (EULLM_METRICS_TOKEN not set): /metrics is not served",
            ),
        )
    }

    /// May `ip`, presenting `key`, administer the history?
    pub fn check_admin(&self, ip: IpAddr, key: Option<&str>) -> Result<(), Refusal> {
        self.check(self.admin.as_ref(), ip, key)
    }

    /// May `ip`, presenting `token`, read `/metrics`?
    pub fn check_metrics(&self, ip: IpAddr, token: Option<&str>) -> Result<(), Refusal> {
        self.check(self.metrics.as_ref(), ip, token)
    }

    fn check(&self, secret: Option<&Secret>, ip: IpAddr, presented: Option<&str>) -> Result<(), Refusal> {
        self.check_at(secret, ip, presented, Instant::now())
    }

    fn check_at(
        &self,
        secret: Option<&Secret>,
        ip: IpAddr,
        presented: Option<&str>,
        now: Instant,
    ) -> Result<(), Refusal> {
        let secret = secret.ok_or(Refusal::Disabled)?;
        let mut failures = self.failures.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(f) = failures.get(&ip)
            && let Some(until) = f.locked_until
            && until > now
        {
            return Err(Refusal::Locked {
                retry_after_s: until.duration_since(now).as_secs().max(1),
            });
        }
        let presented = presented.map(str::trim).filter(|p| !p.is_empty());
        let Some(presented) = presented else {
            return Err(Refusal::Missing);
        };
        if constant_time_eq(&secret.sha256, &digest(presented)) {
            failures.remove(&ip);
            return Ok(());
        }
        if failures.len() >= MAX_TRACKED {
            failures.retain(|_, f| {
                f.locked_until.is_some_and(|u| u > now)
                    || f.first.is_some_and(|t| now.duration_since(t) < WINDOW)
            });
        }
        let f = failures.entry(ip).or_default();
        if f.first.is_none_or(|t| now.duration_since(t) >= WINDOW) {
            *f = Failures {
                count: 0,
                first: Some(now),
                locked_until: None,
            };
        }
        f.count += 1;
        if f.count >= MAX_FAILURES {
            f.locked_until = Some(now + WINDOW);
            tracing::warn!(
                "{ip} locked out of the Monitor's administration and /metrics for {} minutes \
                 after {MAX_FAILURES} wrong keys",
                WINDOW.as_secs() / 60
            );
        }
        Err(Refusal::Wrong)
    }
}

fn digest(secret: &str) -> [u8; 32] {
    Sha256::digest(secret.as_bytes()).into()
}

/// One secret: the environment, then the file `…_FILE` names (its first
/// line that is not blank or a `#` comment), then `.env`. Pure, so the
/// precedence is tested without the process environment.
fn resolve(
    name: &str,
    env: Option<&str>,
    file: Option<(&str, &str)>,
    env_file_contents: Option<&str>,
    env_file_label: &str,
) -> Result<Option<Secret>, String> {
    let found = if let Some(v) = env.map(str::trim).filter(|v| !v.is_empty()) {
        Some((v.to_string(), format!("{name} (environment)")))
    } else if let Some((path, contents)) = file {
        let line = contents
            .lines()
            .map(str::trim)
            .find(|l| !l.is_empty() && !l.starts_with('#'))
            .ok_or_else(|| format!("{name}_FILE ({path}) holds no secret"))?;
        Some((line.to_string(), format!("{name}_FILE ({path})")))
    } else {
        env_file_contents
            .and_then(|c| super::ip_allowlist::env_file_var(c, name))
            .map(|v| (v, format!("{name} ({env_file_label})")))
    };
    let Some((value, source)) = found else {
        return Ok(None);
    };
    if value.chars().count() < MIN_SECRET_LEN {
        return Err(format!(
            "{source} is shorter than {MIN_SECRET_LEN} characters: use a long random secret \
             (openssl rand -hex 24), or unset it"
        ));
    }
    Ok(Some(Secret {
        sha256: digest(&value),
        source,
    }))
}

/// Set on a request by code running in this process — a middleware an
/// embedding program adds around the engine's routers — to say that its
/// caller is an administrator, as the admin key would. A client cannot set
/// it: request extensions do not travel over HTTP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdminGrant;

/// Whether the request may do what only an administrator may: it carries
/// an [`AdminGrant`], or the admin key in [`ADMIN_HEADER`]. `what` names the
/// action for the log, and `disabled` is the answer when no admin key is
/// configured. Every refusal is logged with the address it came from.
pub fn require_admin(
    access: &MonitorAccess,
    ip: IpAddr,
    headers: &axum::http::HeaderMap,
    granted: bool,
    what: &str,
    disabled: &str,
) -> Result<(), (axum::http::StatusCode, axum::Json<serde_json::Value>)> {
    use axum::http::StatusCode;
    if granted {
        return Ok(());
    }
    let key = headers.get(ADMIN_HEADER).and_then(|v| v.to_str().ok());
    let refusal = match access.check_admin(ip, key) {
        Ok(()) => return Ok(()),
        Err(r) => r,
    };
    let (status, reason, message) = match refusal {
        Refusal::Disabled => (StatusCode::FORBIDDEN, "disabled", disabled.to_string()),
        Refusal::Missing => (
            StatusCode::FORBIDDEN,
            "missing",
            format!("Only an administrator may {what} on this server: send the admin key in an X-Admin-Key header."),
        ),
        Refusal::Wrong => (StatusCode::FORBIDDEN, "wrong", "Wrong admin key.".to_string()),
        Refusal::Locked { retry_after_s } => (
            StatusCode::TOO_MANY_REQUESTS,
            "locked",
            format!("Too many wrong admin keys from this address: try again in {retry_after_s} s."),
        ),
    };
    tracing::warn!("Refused to {what} for {ip} ({reason} admin key)");
    Err((
        status,
        axum::Json(serde_json::json!({ "error": message, "admin": reason })),
    ))
}

/// `Authorization: Bearer <token>`'s token.
pub fn bearer(headers: &axum::http::HeaderMap) -> Option<&str> {
    let v = headers
        .get(axum::http::header::AUTHORIZATION)?
        .to_str()
        .ok()?
        .trim();
    let (scheme, rest) = v.split_once(' ')?;
    scheme.eq_ignore_ascii_case("bearer").then(|| rest.trim())
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY: &str = "a-long-admin-secret-0123";
    const IP: IpAddr = IpAddr::V4(std::net::Ipv4Addr::new(10, 0, 0, 7));
    const OTHER: IpAddr = IpAddr::V4(std::net::Ipv4Addr::new(10, 0, 0, 8));

    #[test]
    fn without_a_secret_nothing_is_let_through() {
        let access = MonitorAccess::disabled();
        assert_eq!(access.check_admin(IP, Some(KEY)), Err(Refusal::Disabled));
        assert_eq!(access.check_metrics(IP, Some(KEY)), Err(Refusal::Disabled));
    }

    #[test]
    fn the_right_key_passes_and_the_two_secrets_are_apart() {
        let access = MonitorAccess::from_secrets(Some(KEY), Some("a-metrics-token-for-prometheus"));
        assert_eq!(access.check_admin(IP, Some(KEY)), Ok(()));
        assert_eq!(access.check_admin(IP, Some(&format!(" {KEY} "))), Ok(()));
        assert_eq!(access.check_admin(IP, None), Err(Refusal::Missing));
        assert_eq!(access.check_admin(IP, Some("")), Err(Refusal::Missing));
        // The scraper's token administers nothing.
        assert_eq!(
            access.check_admin(IP, Some("a-metrics-token-for-prometheus")),
            Err(Refusal::Wrong)
        );
        assert_eq!(access.check_metrics(IP, Some("a-metrics-token-for-prometheus")), Ok(()));
    }

    #[test]
    fn five_wrong_keys_lock_the_address_out_for_a_while() {
        let access = MonitorAccess::from_secrets(Some(KEY), None);
        let admin = access.admin.clone();
        let t0 = Instant::now();
        for _ in 0..MAX_FAILURES {
            assert_eq!(access.check_at(admin.as_ref(), IP, Some("guess"), t0), Err(Refusal::Wrong));
        }
        // Even the right key waits, from that address only.
        assert!(matches!(
            access.check_at(admin.as_ref(), IP, Some(KEY), t0),
            Err(Refusal::Locked { .. })
        ));
        assert_eq!(access.check_at(admin.as_ref(), OTHER, Some(KEY), t0), Ok(()));
        let later = t0 + WINDOW + Duration::from_secs(1);
        assert_eq!(access.check_at(admin.as_ref(), IP, Some(KEY), later), Ok(()));
    }

    #[test]
    fn failures_spread_over_more_than_the_window_do_not_lock() {
        let access = MonitorAccess::from_secrets(Some(KEY), None);
        let admin = access.admin.clone();
        let t0 = Instant::now();
        for i in 0..(MAX_FAILURES * 2) {
            let t = t0 + WINDOW / 3 * i;
            assert_eq!(access.check_at(admin.as_ref(), IP, Some("guess"), t), Err(Refusal::Wrong));
        }
    }

    #[test]
    fn a_secret_resolves_from_the_environment_the_file_then_env() {
        let env_file = "EULLM_ADMIN_KEY=from-the-env-file-123456\n";
        let r = |env, file| resolve("EULLM_ADMIN_KEY", env, file, Some(env_file), ".env").unwrap();
        assert_eq!(
            r(Some("from-the-environment-123"), Some(("/run/k", "from-the-file-12345678"))).unwrap().source,
            "EULLM_ADMIN_KEY (environment)"
        );
        let from_file = r(None, Some(("/run/k", "# a comment\n\nfrom-the-file-12345678\n"))).unwrap();
        assert_eq!(from_file.source, "EULLM_ADMIN_KEY_FILE (/run/k)");
        assert_eq!(from_file.sha256, digest("from-the-file-12345678"));
        assert_eq!(r(Some("  "), None).unwrap().source, "EULLM_ADMIN_KEY (.env)");
        assert!(resolve("EULLM_ADMIN_KEY", None, None, None, ".env").unwrap().is_none());
    }

    #[test]
    fn a_short_or_empty_secret_stops_the_server() {
        assert!(resolve("EULLM_ADMIN_KEY", Some("short"), None, None, ".env").is_err());
        assert!(resolve("EULLM_METRICS_TOKEN", None, Some(("/k", "# nothing\n")), None, ".env").is_err());
    }

    #[test]
    fn the_bearer_token_is_read_case_blind() {
        let mut h = axum::http::HeaderMap::new();
        h.insert("authorization", "bearer  tok-123 ".parse().unwrap());
        assert_eq!(bearer(&h), Some("tok-123"));
        h.insert("authorization", "Basic abc".parse().unwrap());
        assert_eq!(bearer(&h), None);
    }
}

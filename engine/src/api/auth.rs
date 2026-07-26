//! Optional bearer-token authentication with per-key rate limits.
//!
//! # Why this exists alongside the IP allowlist
//!
//! [`super::ip_allowlist`] checks the source IP of the TCP connection, which is
//! a meaningful identity only when the client connects directly. Two deployments
//! where it structurally cannot work:
//!
//! * **Docker published ports.** With `ports: "11434:11434"` every external
//!   client is NAT-ed to the bridge gateway address. The operator either can't
//!   reach the service at all, or widens the allowlist to the bridge subnet —
//!   at which point the allowlist no longer discriminates between clients,
//!   because they all arrive as the same address.
//! * **A request from the user's own browser.** It genuinely originates from
//!   loopback, so the allowlist is satisfied by construction and any page the
//!   user visits can reach the API.
//!
//! A token survives address translation: it identifies the *caller*, not the
//! path the packet took. That is the whole point of putting it in front.
//!
//! # Posture when keys are configured
//!
//! Auth runs **outside** the allowlist, and a request presenting a valid key
//! is admitted regardless of its source IP. This is deliberate and is the only
//! ordering that fixes the Docker case: refusing a valid key because the packet
//! arrived from the bridge gateway would leave the operator exactly where they
//! started. Configuring keys therefore *replaces* address-based admission with
//! identity-based admission; it does not stack on top of it. Startup logs which
//! posture is in effect so this is never a surprise, and requests with no key
//! or a wrong key are refused with 401 whatever their origin — including
//! loopback.
//!
//! # Configuration
//!
//! Perimeter configuration is read from the environment rather than from CLI
//! flags — see the rule in `.claude/CLAUDE.md`. Secrets on a command line are
//! visible in `ps` to every local user, and every non-interactive deployment
//! (`docker run -e`, a compose `environment:` block, systemd `Environment=`)
//! configures the environment anyway.
//!
//! In precedence order:
//!
//! 1. `EULLM_API_KEYS` in the process environment — comma-separated entries.
//! 2. `EULLM_API_KEYS_FILE` in the process environment — path to a file with
//!    one entry per line (`#` comments and blank lines ignored). Preferred for
//!    real deployments: a file can be `chmod 600`, an environment variable is
//!    readable through `/proc/<pid>/environ` and often ends up in logs.
//! 3. `EULLM_API_KEYS` in the `.env` file in the working directory — the same
//!    file the IP allowlist reads, for symmetry.
//!
//! An entry is `id:secret` or `id:secret:rpm=N`:
//!
//! ```text
//! EULLM_API_KEYS=ci:8f3b1d9c2e7a4f60b5,rag-prod:1a2b3c4d5e6f7a8b9c:rpm=600
//! ```
//!
//! `id` is what lands in the audit trail's `user_id` — it is not a secret and
//! is safe to log. `secret` is the token the client presents. `rpm` caps
//! requests per minute for that key (0 = unlimited, and the default when the
//! suffix is omitted).
//!
//! **Configuring keys that don't parse is fatal at startup.** An operator who
//! sets `EULLM_API_KEYS` has asked for authentication; degrading silently to an
//! open service because of a typo is the one failure mode this must not have.

use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use sha2::{Digest, Sha256};

/// Minimum secret length. Not a policy flourish: a token short enough to guess
/// makes every other control here decorative, and the operator who typed
/// `key:test` is exactly the one who will not notice.
const MIN_SECRET_LEN: usize = 16;

/// Longest accepted key id, and the character set it may use. Bounded because
/// the id is written into every audit record and into log lines.
const MAX_ID_LEN: usize = 64;

/// One configured key. The secret is kept only as a digest: comparison happens
/// over fixed-length hashes, so it is constant-time by construction and a core
/// dump does not hand over the plaintext token.
#[derive(Debug, Clone)]
struct KeyEntry {
    id: String,
    secret_sha256: [u8; 32],
    /// Requests per minute allowed for this key; 0 = unlimited.
    rpm: u32,
}

/// Who is making the request.
///
/// Inserted into request extensions by [`ApiKeys::authenticate`]'s caller for
/// every request, so downstream layers and handlers never have to distinguish
/// "auth disabled" from "auth not yet run".
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Identity {
    /// The configured key id, or `None` when authentication is disabled.
    key_id: Option<String>,
}

impl Identity {
    /// Identity of a request that arrived while authentication was disabled.
    pub fn anonymous() -> Self {
        Self { key_id: None }
    }

    /// The key id, for the audit trail's `user_id` and for log lines. `None`
    /// means no authentication was configured, which is a legitimate state for
    /// single-user local use — not an error.
    pub fn key_id(&self) -> Option<&str> {
        self.key_id.as_deref()
    }

    /// Whether this request presented a valid key. Used by the IP allowlist
    /// layer to admit an authenticated caller whose source address would
    /// otherwise be rejected — see the module docs on posture.
    pub fn is_authenticated(&self) -> bool {
        self.key_id.is_some()
    }
}

/// Why a request was refused.
#[derive(Debug, PartialEq, Eq)]
pub enum AuthError {
    /// No token presented at all.
    Missing,
    /// A token was presented but matches no configured key.
    Invalid,
    /// Valid key, but over its per-minute quota. Carries the seconds until the
    /// current window rolls over, for `Retry-After`.
    RateLimited { key_id: String, retry_after_s: u64 },
}

/// The configured key set plus its rate-limit state.
pub struct ApiKeys {
    keys: Vec<KeyEntry>,
    /// Human-readable origin of the configuration, for the startup log.
    source: String,
    /// Fixed-window request counters, keyed by key id.
    windows: Mutex<HashMap<String, Window>>,
}

impl std::fmt::Debug for ApiKeys {
    /// Written by hand so that no `{:?}` anywhere — a panic message, a log line,
    /// a test failure — can ever print key material. Only the ids, the quotas
    /// and the configuration source are shown; the digests are omitted
    /// deliberately, hashes though they are.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApiKeys")
            .field("keys", &self.describe())
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

/// A fixed one-minute counting window. Deliberately not a sliding window or a
/// token bucket: the purpose is to stop a runaway client from monopolising the
/// scheduler's slots, and for that a fixed window is exact enough while being
/// small enough to audit by reading it.
#[derive(Debug)]
struct Window {
    started: Instant,
    count: u32,
}

impl ApiKeys {
    /// No authentication configured.
    fn disabled() -> Self {
        Self {
            keys: Vec::new(),
            source: "not configured (no authentication — the IP allowlist is the only control)"
                .to_string(),
            windows: Mutex::new(HashMap::new()),
        }
    }

    /// Load the key set from the environment, then a key file, then `.env`.
    ///
    /// Returns `Err` when configuration is present but unusable, so startup can
    /// refuse to run rather than serve an open API in the belief it is
    /// protected.
    pub fn load(env_file: &Path) -> Result<Self, String> {
        let env_spec = std::env::var("EULLM_API_KEYS").ok();
        let key_file = std::env::var("EULLM_API_KEYS_FILE").ok();
        let key_file_contents = key_file
            .as_deref()
            .map(str::trim)
            .filter(|p| !p.is_empty())
            .map(|p| {
                std::fs::read_to_string(p)
                    .map(|c| (p.to_string(), c))
                    .map_err(|e| format!("EULLM_API_KEYS_FILE is set to '{p}' but unreadable: {e}"))
            })
            .transpose()?;
        let env_file_contents = std::fs::read_to_string(env_file).ok();

        Self::resolve(
            env_spec.as_deref(),
            key_file_contents
                .as_ref()
                .map(|(p, c)| (p.as_str(), c.as_str())),
            env_file_contents.as_deref(),
            &env_file.display().to_string(),
        )
    }

    /// Pure resolution step behind [`load`], split out so precedence is
    /// testable without mutating process environment variables — which would
    /// race against every other test in this binary.
    fn resolve(
        env_spec: Option<&str>,
        key_file: Option<(&str, &str)>,
        env_file_contents: Option<&str>,
        env_file_label: &str,
    ) -> Result<Self, String> {
        // 1. The environment wins outright: an operator passing `-e` is being
        //    more explicit than any file, which may have been baked into an
        //    image.
        if let Some(spec) = env_spec.map(str::trim).filter(|s| !s.is_empty()) {
            let keys = parse_entries(spec, ',')
                .map_err(|e| format!("EULLM_API_KEYS in the environment is invalid: {e}"))?;
            return Ok(Self {
                keys,
                source: "EULLM_API_KEYS (environment)".to_string(),
                windows: Mutex::new(HashMap::new()),
            });
        }

        // 2. A dedicated key file, one entry per line.
        if let Some((path, contents)) = key_file {
            let keys = parse_entries(contents, '\n')
                .map_err(|e| format!("the key file '{path}' is invalid: {e}"))?;
            return Ok(Self {
                keys,
                source: format!("EULLM_API_KEYS_FILE ({path})"),
                windows: Mutex::new(HashMap::new()),
            });
        }

        // 3. `EULLM_API_KEYS` inside the same `.env` the allowlist reads.
        if let Some(contents) = env_file_contents
            && let Some(spec) = super::ip_allowlist::env_file_var(contents, "EULLM_API_KEYS")
        {
            let keys = parse_entries(&spec, ',')
                .map_err(|e| format!("EULLM_API_KEYS in {env_file_label} is invalid: {e}"))?;
            return Ok(Self {
                keys,
                source: format!("EULLM_API_KEYS ({env_file_label})"),
                windows: Mutex::new(HashMap::new()),
            });
        }

        Ok(Self::disabled())
    }

    /// Whether any key is configured. When false, every request is admitted
    /// with an anonymous [`Identity`] and the IP allowlist is the only control.
    pub fn is_enabled(&self) -> bool {
        !self.keys.is_empty()
    }

    /// Where the configuration came from — for the startup log, so an operator
    /// can tell at a glance whether their keys were picked up at all.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// One line describing the configured keys: ids and their quotas. Contains
    /// no secrets.
    pub fn describe(&self) -> String {
        if self.keys.is_empty() {
            return "none".to_string();
        }
        self.keys
            .iter()
            .map(|k| {
                if k.rpm == 0 {
                    k.id.clone()
                } else {
                    format!("{} ({}/min)", k.id, k.rpm)
                }
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Check a presented token and charge one request against its quota.
    ///
    /// `presented` is `None` when the request carried no token at all, which is
    /// reported separately from a wrong token so the 401 body can say which it
    /// was — that distinction helps a misconfigured client and tells an
    /// attacker nothing they could not learn by sending an empty token.
    pub fn authenticate(&self, presented: Option<&str>) -> Result<Identity, AuthError> {
        if !self.is_enabled() {
            return Ok(Identity::anonymous());
        }
        let token = presented.map(str::trim).filter(|t| !t.is_empty());
        let Some(token) = token else {
            return Err(AuthError::Missing);
        };

        let digest: [u8; 32] = Sha256::digest(token.as_bytes()).into();
        // Scan every key and fold the results, instead of returning on the
        // first match: the comparison cost then does not depend on *which* key
        // matched, only on how many are configured.
        let mut matched: Option<&KeyEntry> = None;
        for key in &self.keys {
            if constant_time_eq(&key.secret_sha256, &digest) {
                matched = Some(key);
            }
        }
        let Some(key) = matched else {
            return Err(AuthError::Invalid);
        };

        if let Some(retry_after_s) = self.charge(&key.id, key.rpm) {
            return Err(AuthError::RateLimited {
                key_id: key.id.clone(),
                retry_after_s,
            });
        }

        Ok(Identity {
            key_id: Some(key.id.clone()),
        })
    }

    /// Charge one request against `key_id`'s window. Returns `Some(seconds)`
    /// when the quota is exhausted, where `seconds` is the wait until the
    /// window rolls over.
    fn charge(&self, key_id: &str, rpm: u32) -> Option<u64> {
        if rpm == 0 {
            return None;
        }
        self.charge_at(key_id, rpm, Instant::now())
    }

    /// [`charge`] with the clock injected, so the window behaviour is testable
    /// without sleeping for a minute.
    fn charge_at(&self, key_id: &str, rpm: u32, now: Instant) -> Option<u64> {
        const WINDOW: Duration = Duration::from_secs(60);
        let mut windows = self.windows.lock();
        let w = windows.entry(key_id.to_string()).or_insert(Window {
            started: now,
            count: 0,
        });
        if now.duration_since(w.started) >= WINDOW {
            w.started = now;
            w.count = 0;
        }
        if w.count >= rpm {
            let elapsed = now.duration_since(w.started);
            return Some(WINDOW.saturating_sub(elapsed).as_secs().max(1));
        }
        w.count += 1;
        None
    }
}

/// Fixed-time byte comparison. Written out rather than pulled from a crate: it
/// is four lines, and the alternative is a new dependency on the audited path.
/// `black_box` keeps the optimiser from turning the fold into an early exit.
fn constant_time_eq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut diff = 0u8;
    for i in 0..32 {
        diff |= a[i] ^ b[i];
    }
    std::hint::black_box(diff) == 0
}

/// Parse entries separated by `sep`, ignoring blank lines and `#` comments.
fn parse_entries(spec: &str, sep: char) -> Result<Vec<KeyEntry>, String> {
    let mut keys: Vec<KeyEntry> = Vec::new();
    for raw in spec.split(sep) {
        let entry = raw.trim();
        if entry.is_empty() || entry.starts_with('#') {
            continue;
        }
        let key = parse_entry(entry)?;
        if keys.iter().any(|k| k.id == key.id) {
            return Err(format!("duplicate key id '{}'", key.id));
        }
        keys.push(key);
    }
    if keys.is_empty() {
        return Err("no key entries found (expected id:secret[:rpm=N])".to_string());
    }
    Ok(keys)
}

/// Parse one `id:secret[:rpm=N]` entry.
///
/// Errors deliberately never echo the secret: an invalid-configuration message
/// ends up in logs and in an operator's terminal scrollback.
fn parse_entry(entry: &str) -> Result<KeyEntry, String> {
    let mut parts = entry.split(':');
    let id = parts.next().unwrap_or("").trim();
    let secret = parts.next().unwrap_or("").trim();
    let extra: Vec<&str> = parts.map(str::trim).collect();

    if id.is_empty() || secret.is_empty() {
        return Err("expected id:secret[:rpm=N]".to_string());
    }
    if id.len() > MAX_ID_LEN {
        return Err(format!("key id is longer than {MAX_ID_LEN} characters",));
    }
    if !id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    {
        return Err(format!(
            "key id '{id}' may contain only letters, digits, '-', '_' and '.'"
        ));
    }
    if secret.chars().count() < MIN_SECRET_LEN {
        return Err(format!(
            "the secret for key '{id}' is shorter than {MIN_SECRET_LEN} characters"
        ));
    }

    let mut rpm = 0u32;
    for part in extra {
        if part.is_empty() {
            continue;
        }
        let Some(value) = part.strip_prefix("rpm=") else {
            return Err(format!(
                "unrecognised option '{part}' on key '{id}' (only rpm=N is supported; \
                 note a secret may not contain ':')"
            ));
        };
        rpm = value
            .parse::<u32>()
            .map_err(|_| format!("rpm on key '{id}' is not a number"))?;
    }

    Ok(KeyEntry {
        id: id.to_string(),
        secret_sha256: Sha256::digest(secret.as_bytes()).into(),
        rpm,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const S1: &str = "0123456789abcdef01";
    const S2: &str = "fedcba9876543210ff";

    fn keys(spec: &str) -> ApiKeys {
        ApiKeys::resolve(Some(spec), None, None, ".env").expect("spec should parse")
    }

    #[test]
    fn nothing_configured_disables_auth_and_admits_anonymously() {
        let a = ApiKeys::resolve(None, None, None, ".env").unwrap();
        assert!(!a.is_enabled());
        let id = a.authenticate(None).unwrap();
        assert!(!id.is_authenticated());
        assert_eq!(id.key_id(), None);
    }

    #[test]
    fn a_valid_token_yields_its_key_id() {
        let a = keys(&format!("ci:{S1}"));
        assert!(a.is_enabled());
        let id = a.authenticate(Some(S1)).unwrap();
        assert_eq!(id.key_id(), Some("ci"));
        assert!(id.is_authenticated());
    }

    #[test]
    fn missing_and_wrong_tokens_are_reported_separately() {
        let a = keys(&format!("ci:{S1}"));
        assert_eq!(a.authenticate(None), Err(AuthError::Missing));
        assert_eq!(a.authenticate(Some("")), Err(AuthError::Missing));
        assert_eq!(a.authenticate(Some("   ")), Err(AuthError::Missing));
        assert_eq!(
            a.authenticate(Some("wrong-but-long-enough")),
            Err(AuthError::Invalid)
        );
    }

    #[test]
    fn the_key_id_is_not_accepted_as_the_token() {
        // Guards against ever comparing the wrong field: the id is public (it
        // goes into every audit record) so accepting it would be a total
        // bypass.
        let a = keys(&format!("ci:{S1}"));
        assert_eq!(a.authenticate(Some("ci")), Err(AuthError::Invalid));
    }

    #[test]
    fn each_key_maps_to_its_own_id() {
        let a = keys(&format!("ci:{S1},rag:{S2}"));
        assert_eq!(a.authenticate(Some(S1)).unwrap().key_id(), Some("ci"));
        assert_eq!(a.authenticate(Some(S2)).unwrap().key_id(), Some("rag"));
    }

    #[test]
    fn short_secrets_are_refused_at_load_not_at_request_time() {
        let e = ApiKeys::resolve(Some("ci:hunter2"), None, None, ".env").unwrap_err();
        assert!(e.contains("shorter than"), "{e}");
        // And the message must not leak the secret into logs, where it would
        // outlive the mistake that produced it.
        assert!(
            !e.contains("hunter2"),
            "the error must not echo the secret: {e}"
        );
    }

    #[test]
    fn malformed_configuration_is_an_error_not_a_silently_open_api() {
        // The whole point: an operator who sets EULLM_API_KEYS and gets a typo
        // must not end up with an unauthenticated service.
        for spec in ["ci", "ci:", ":secret-long-enough-x", "="] {
            assert!(
                ApiKeys::resolve(Some(spec), None, None, ".env").is_err(),
                "spec {spec:?} should be rejected"
            );
        }
    }

    #[test]
    fn duplicate_ids_are_refused() {
        let e =
            ApiKeys::resolve(Some(&format!("ci:{S1},ci:{S2}")), None, None, ".env").unwrap_err();
        assert!(e.contains("duplicate"), "{e}");
    }

    #[test]
    fn ids_are_constrained_to_a_safe_character_set() {
        // The id is written into audit records and log lines.
        assert!(ApiKeys::resolve(Some(&format!("a b:{S1}")), None, None, ".env").is_err());
        assert!(ApiKeys::resolve(Some(&format!("a\nb:{S1}")), None, None, ".env").is_err());
        assert!(ApiKeys::resolve(Some(&format!("ok-id_1.2:{S1}")), None, None, ".env").is_ok());
    }

    #[test]
    fn an_unrecognised_suffix_is_refused_rather_than_ignored() {
        // A secret containing ':' would otherwise silently become a truncated
        // secret plus an ignored tail — i.e. a weaker token than configured.
        let e = ApiKeys::resolve(Some(&format!("ci:{S1}:rpx=5")), None, None, ".env").unwrap_err();
        assert!(e.contains("unrecognised option"), "{e}");
    }

    #[test]
    fn precedence_environment_then_key_file_then_env_file() {
        let a = ApiKeys::resolve(
            Some(&format!("from-env:{S1}")),
            Some(("/k", &format!("from-file:{S2}"))),
            Some(&format!("EULLM_API_KEYS=from-dotenv:{S2}\n")),
            ".env",
        )
        .unwrap();
        assert_eq!(a.authenticate(Some(S1)).unwrap().key_id(), Some("from-env"));
        assert_eq!(a.authenticate(Some(S2)), Err(AuthError::Invalid));
        assert!(a.source().contains("environment"));

        let a = ApiKeys::resolve(
            None,
            Some(("/k", &format!("from-file:{S2}"))),
            Some(&format!("EULLM_API_KEYS=from-dotenv:{S1}\n")),
            ".env",
        )
        .unwrap();
        assert_eq!(
            a.authenticate(Some(S2)).unwrap().key_id(),
            Some("from-file")
        );
        assert!(a.source().contains("/k"));

        let a = ApiKeys::resolve(
            None,
            None,
            Some(&format!("EULLM_API_KEYS=from-dotenv:{S1}\n")),
            ".env",
        )
        .unwrap();
        assert_eq!(
            a.authenticate(Some(S1)).unwrap().key_id(),
            Some("from-dotenv")
        );
        assert!(a.source().contains(".env"));
    }

    #[test]
    fn a_key_file_ignores_comments_and_blank_lines() {
        let a = ApiKeys::resolve(
            None,
            Some((
                "/k",
                &format!("# a comment\n\nci:{S1}\n\n# another\nrag:{S2}\n"),
            )),
            None,
            ".env",
        )
        .unwrap();
        assert_eq!(a.authenticate(Some(S1)).unwrap().key_id(), Some("ci"));
        assert_eq!(a.authenticate(Some(S2)).unwrap().key_id(), Some("rag"));
    }

    #[test]
    fn rpm_zero_or_absent_means_unlimited() {
        let a = keys(&format!("ci:{S1}"));
        for _ in 0..1000 {
            assert!(a.authenticate(Some(S1)).is_ok());
        }
        let a = keys(&format!("ci:{S1}:rpm=0"));
        for _ in 0..1000 {
            assert!(a.authenticate(Some(S1)).is_ok());
        }
    }

    #[test]
    fn the_quota_refuses_the_request_after_the_limit() {
        let a = keys(&format!("ci:{S1}:rpm=3"));
        for i in 0..3 {
            assert!(a.authenticate(Some(S1)).is_ok(), "request {i} should pass");
        }
        match a.authenticate(Some(S1)) {
            Err(AuthError::RateLimited {
                key_id,
                retry_after_s,
            }) => {
                assert_eq!(key_id, "ci");
                assert!(
                    (1..=60).contains(&retry_after_s),
                    "retry_after_s = {retry_after_s}"
                );
            }
            other => panic!("expected a rate limit, got {other:?}"),
        }
    }

    #[test]
    fn the_window_rolls_over_after_a_minute() {
        let a = keys(&format!("ci:{S1}:rpm=2"));
        let t0 = Instant::now();
        assert!(a.charge_at("ci", 2, t0).is_none());
        assert!(a.charge_at("ci", 2, t0).is_none());
        assert!(
            a.charge_at("ci", 2, t0).is_some(),
            "third in-window request"
        );
        // Just under a minute: still limited.
        assert!(a.charge_at("ci", 2, t0 + Duration::from_secs(59)).is_some());
        // A full minute later the window resets.
        assert!(a.charge_at("ci", 2, t0 + Duration::from_secs(60)).is_none());
    }

    #[test]
    fn quotas_are_per_key_not_global() {
        let a = keys(&format!("ci:{S1}:rpm=1,rag:{S2}:rpm=1"));
        assert!(a.authenticate(Some(S1)).is_ok());
        // ci is now exhausted, but rag must be unaffected.
        assert!(matches!(
            a.authenticate(Some(S1)),
            Err(AuthError::RateLimited { .. })
        ));
        assert!(a.authenticate(Some(S2)).is_ok());
    }

    #[test]
    fn describe_lists_ids_and_quotas_but_never_secrets() {
        let a = keys(&format!("ci:{S1}:rpm=60,rag:{S2}"));
        let d = a.describe();
        assert!(d.contains("ci (60/min)"), "{d}");
        assert!(d.contains("rag"), "{d}");
        assert!(!d.contains(S1) && !d.contains(S2), "secret leaked into {d}");
    }

    #[test]
    fn constant_time_eq_still_compares_correctly() {
        let a = Sha256::digest(b"x").into();
        let b = Sha256::digest(b"x").into();
        let c: [u8; 32] = Sha256::digest(b"y").into();
        assert!(constant_time_eq(&a, &b));
        assert!(!constant_time_eq(&a, &c));
        // Differences in the last byte must be caught as reliably as in the
        // first — the failure mode of a hand-rolled comparison.
        let mut d = a;
        d[31] ^= 1;
        assert!(!constant_time_eq(&a, &d));
        let mut e = a;
        e[0] ^= 1;
        assert!(!constant_time_eq(&a, &e));
    }
}

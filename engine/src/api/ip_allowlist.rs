//! IP allowlist for the API and chat UI listeners.
//!
//! The server always binds `0.0.0.0` (the engine may legitimately run on a
//! different host than its callers — a RAG pipeline, a LAN client) so the
//! bind address itself can't be the safety boundary. This module is that
//! boundary instead: every request's source IP is checked against an
//! allowlist before it reaches any handler.
//!
//! With nothing configured, the allowlist defaults to loopback only
//! (`127.0.0.1`, `::1`) — the same effective result as binding `127.0.0.1`,
//! without needing a different bind address for the "nothing configured"
//! case. Broader access (a specific RAG host, a LAN subnet) is opt-in via
//! `EULLM_ALLOWED_IPS`, read from the **process environment** first and from a
//! `.env` file in the working directory second — never the unconfigured
//! default.
//!
//! **Known limit, worth understanding before relying on this.** The check
//! looks at the source IP of the TCP connection, which is only a meaningful
//! identity when the client connects directly. Two cases where it isn't:
//! behind Docker's published ports every external client is NAT-ed to the
//! bridge gateway address, so allowing that address allows everyone who can
//! reach the port; and a request originating in the user's own browser
//! genuinely comes from loopback, so a page on any site can reach the API.
//! An allowlist cannot express either case — closing them needs
//! authentication (a token checked before this layer) and an `Origin` check.
//! See `docs/backlog-fix-e-hardening.md`, H1-A and H1-E.

use std::fs;
use std::net::IpAddr;
use std::path::Path;

use ipnet::IpNet;

/// A parsed, ready-to-check IP allowlist.
#[derive(Debug, Clone)]
pub struct IpAllowlist {
    nets: Vec<IpNet>,
    /// Human-readable origin of the non-loopback entries (see `source`).
    source: String,
}

impl IpAllowlist {
    /// Loopback-only — the default when nothing is configured.
    fn loopback_only() -> Self {
        Self {
            nets: vec!["127.0.0.1/32".parse().unwrap(), "::1/128".parse().unwrap()],
            source: "default (loopback only — nothing configured)".to_string(),
        }
    }

    /// Load the allowlist, reading `EULLM_ALLOWED_IPS` from the process
    /// environment first and falling back to a `.env` file at `path`.
    ///
    /// The environment has to be a supported source, not just the file. Every
    /// non-interactive way of running this engine configures it that way:
    /// `docker run -e`, a compose `environment:` block, a systemd
    /// `Environment=`. For several releases only the file was read, so all of
    /// those silently had no effect — and since no image in this repository
    /// ships a `.env`, every containerised deployment was pinned to
    /// loopback-only with no way to widen it and no diagnostic saying so.
    ///
    /// Loopback is always included, so local access can't break by configuring
    /// a remote host, and any failure to *deliberately* widen the list (no
    /// variable, no file, malformed entry) means "nothing was added" — never
    /// "more than loopback".
    pub fn load(path: &Path) -> Self {
        let env_spec = std::env::var("EULLM_ALLOWED_IPS").ok();
        let file_contents = fs::read_to_string(path).ok();
        Self::resolve(
            env_spec.as_deref(),
            file_contents.as_deref(),
            &path.display().to_string(),
        )
    }

    /// Pure resolution step behind [`load`], split out so the precedence rule
    /// is testable without mutating process environment variables (which would
    /// race against every other test in this binary).
    ///
    /// Returns the allowlist and, via [`Self::source`], where the entries came
    /// from, so startup can report it.
    fn resolve(env_spec: Option<&str>, file_contents: Option<&str>, file_label: &str) -> Self {
        let mut allowlist = Self::loopback_only();

        // The environment wins outright when set to something non-empty: an
        // operator passing `-e EULLM_ALLOWED_IPS=...` is being more explicit
        // than a file that may have been baked into an image.
        if let Some(spec) = env_spec.map(str::trim).filter(|s| !s.is_empty()) {
            match parse_allowlist(spec) {
                Ok(nets) => {
                    allowlist.nets.extend(nets);
                    allowlist.source = "EULLM_ALLOWED_IPS (environment)".to_string();
                    return allowlist;
                }
                Err(e) => {
                    tracing::warn!(
                        "EULLM_ALLOWED_IPS in the environment is malformed ({e}) — ignoring it; \
                         falling back to {file_label}"
                    );
                }
            }
        }

        let Some(contents) = file_contents else {
            return allowlist;
        };
        let vars = parse_env_file(contents);
        let Some(spec) = vars
            .get("EULLM_ALLOWED_IPS")
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
        else {
            return allowlist;
        };
        match parse_allowlist(spec) {
            Ok(nets) => {
                allowlist.nets.extend(nets);
                allowlist.source = format!("EULLM_ALLOWED_IPS ({file_label})");
            }
            Err(e) => tracing::warn!(
                "EULLM_ALLOWED_IPS in {file_label} is malformed ({e}) — ignoring, \
                 loopback-only stays in effect"
            ),
        }
        allowlist
    }

    /// File-only load, ignoring the process environment. Kept so the file
    /// parsing path stays directly testable; production startup calls
    /// [`load`].
    #[cfg(test)]
    pub fn load_from_env_file(path: &Path) -> Self {
        let file_contents = fs::read_to_string(path).ok();
        Self::resolve(None, file_contents.as_deref(), &path.display().to_string())
    }

    /// Where the non-loopback entries came from — for the startup log, so an
    /// operator can tell at a glance whether their configuration was picked up
    /// at all.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Whether `ip` is covered by this allowlist.
    pub fn is_allowed(&self, ip: IpAddr) -> bool {
        self.nets.iter().any(|net| net.contains(&ip))
    }

    /// Human-readable summary for the startup banner/logs.
    pub fn describe(&self) -> String {
        self.nets
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Minimal `.env` parser: `KEY=VALUE` lines, blank lines and `#` comments
/// ignored, optional surrounding single/double quotes stripped from the
/// value. Deliberately not a general-purpose dotenv implementation (no
/// variable interpolation, no `export` prefix, no multiline values) — the
/// only consumer is `EULLM_ALLOWED_IPS`, and a tiny parser we can read in
/// one sitting is safer here than pulling in a dependency to set process
/// environment variables we don't actually need mutated.
fn parse_env_file(contents: &str) -> std::collections::HashMap<String, String> {
    let mut vars = std::collections::HashMap::new();
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim().to_string();
        let mut value = value.trim();
        if (value.starts_with('"') && value.ends_with('"') && value.len() >= 2)
            || (value.starts_with('\'') && value.ends_with('\'') && value.len() >= 2)
        {
            value = &value[1..value.len() - 1];
        }
        vars.insert(key, value.to_string());
    }
    vars
}

/// Parse a comma-separated list of bare IPs and/or CIDR subnets, e.g.
/// `"203.0.113.5,192.168.0.0/24,::1"`. A bare IP is treated as an exact
/// match (implicit /32 or /128).
fn parse_allowlist(spec: &str) -> Result<Vec<IpNet>, String> {
    spec.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|entry| {
            if let Ok(net) = entry.parse::<IpNet>() {
                return Ok(net);
            }
            entry
                .parse::<IpAddr>()
                .map(IpNet::from)
                .map_err(|_| format!("'{entry}' is not a valid IP address or CIDR subnet"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_allows_only_loopback() {
        let allowlist = IpAllowlist::loopback_only();
        assert!(allowlist.is_allowed("127.0.0.1".parse().unwrap()));
        assert!(allowlist.is_allowed("::1".parse().unwrap()));
        assert!(!allowlist.is_allowed("203.0.113.5".parse().unwrap()));
        assert!(!allowlist.is_allowed("192.168.1.1".parse().unwrap()));
    }

    #[test]
    fn missing_env_file_falls_back_to_loopback() {
        let allowlist = IpAllowlist::load_from_env_file(Path::new("/nonexistent/.env"));
        assert!(allowlist.is_allowed("127.0.0.1".parse().unwrap()));
        assert!(!allowlist.is_allowed("203.0.113.5".parse().unwrap()));
    }

    #[test]
    fn empty_or_missing_key_falls_back_to_loopback() {
        let dir = std::env::temp_dir().join(format!("eullm-ipallow-test-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join(".env");

        fs::write(&path, "SOME_OTHER_VAR=hello\n").unwrap();
        assert!(!IpAllowlist::load_from_env_file(&path).is_allowed("203.0.113.5".parse().unwrap()));

        fs::write(&path, "EULLM_ALLOWED_IPS=\n").unwrap();
        assert!(!IpAllowlist::load_from_env_file(&path).is_allowed("203.0.113.5".parse().unwrap()));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn parses_single_ip_and_subnet_from_env_file() {
        let dir = std::env::temp_dir().join(format!("eullm-ipallow-test-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join(".env");
        fs::write(
            &path,
            "# comment, should be ignored\n\nEULLM_ALLOWED_IPS=203.0.113.5,192.168.0.0/24\n",
        )
        .unwrap();

        let allowlist = IpAllowlist::load_from_env_file(&path);
        assert!(allowlist.is_allowed("203.0.113.5".parse().unwrap()));
        assert!(allowlist.is_allowed("192.168.0.1".parse().unwrap()));
        assert!(allowlist.is_allowed("192.168.0.254".parse().unwrap()));
        assert!(!allowlist.is_allowed("192.168.1.1".parse().unwrap()));
        assert!(!allowlist.is_allowed("203.0.113.6".parse().unwrap()));
        // Loopback stays allowed on top of whatever the file adds — local
        // access shouldn't break just because a remote host was configured.
        assert!(allowlist.is_allowed("127.0.0.1".parse().unwrap()));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn quoted_values_are_unquoted() {
        let dir = std::env::temp_dir().join(format!("eullm-ipallow-test-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join(".env");
        fs::write(&path, "EULLM_ALLOWED_IPS=\"203.0.113.5\"\n").unwrap();

        let allowlist = IpAllowlist::load_from_env_file(&path);
        assert!(allowlist.is_allowed("203.0.113.5".parse().unwrap()));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn malformed_entry_falls_back_to_loopback() {
        let dir = std::env::temp_dir().join(format!("eullm-ipallow-test-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join(".env");
        fs::write(&path, "EULLM_ALLOWED_IPS=not-an-ip\n").unwrap();

        let allowlist = IpAllowlist::load_from_env_file(&path);
        assert!(allowlist.is_allowed("127.0.0.1".parse().unwrap()));
        assert!(!allowlist.is_allowed("203.0.113.5".parse().unwrap()));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn parse_allowlist_rejects_garbage() {
        assert!(parse_allowlist("not-an-ip").is_err());
        assert!(parse_allowlist("203.0.113.5,also-not-an-ip").is_err());
    }

    #[test]
    fn parse_allowlist_accepts_ipv6() {
        let nets = parse_allowlist("::1,2001:db8::/32").unwrap();
        assert_eq!(nets.len(), 2);
    }
}

#[cfg(test)]
mod env_precedence_tests {
    use super::*;

    // Only the `.env` file used to be read. Every non-interactive deployment
    // configures this through the environment instead (`docker run -e`, compose
    // `environment:`, systemd `Environment=`), so those all silently had no
    // effect — and no image here ships a `.env`, which pinned containers to
    // loopback-only with no way to widen it.

    #[test]
    fn environment_alone_is_honoured() {
        let a = IpAllowlist::resolve(Some("203.0.113.5,192.168.0.0/24"), None, ".env");
        assert!(a.is_allowed("203.0.113.5".parse().unwrap()));
        assert!(a.is_allowed("192.168.0.7".parse().unwrap()));
        assert!(!a.is_allowed("198.51.100.1".parse().unwrap()));
        // Loopback survives whatever is configured.
        assert!(a.is_allowed("127.0.0.1".parse().unwrap()));
        assert!(a.source().contains("environment"));
    }

    #[test]
    fn environment_wins_over_the_file() {
        let a = IpAllowlist::resolve(
            Some("203.0.113.5"),
            Some("EULLM_ALLOWED_IPS=198.51.100.9\n"),
            ".env",
        );
        assert!(a.is_allowed("203.0.113.5".parse().unwrap()));
        assert!(
            !a.is_allowed("198.51.100.9".parse().unwrap()),
            "an explicit environment value must not be merged with a baked-in file"
        );
        assert!(a.source().contains("environment"));
    }

    #[test]
    fn file_is_used_when_the_environment_is_unset_or_blank() {
        for env in [None, Some(""), Some("   ")] {
            let a = IpAllowlist::resolve(env, Some("EULLM_ALLOWED_IPS=198.51.100.9\n"), ".env");
            assert!(a.is_allowed("198.51.100.9".parse().unwrap()), "env={env:?}");
            assert!(a.source().contains(".env"), "env={env:?}");
        }
    }

    #[test]
    fn a_malformed_environment_value_falls_back_to_the_file() {
        let a = IpAllowlist::resolve(
            Some("not-an-ip"),
            Some("EULLM_ALLOWED_IPS=198.51.100.9\n"),
            ".env",
        );
        assert!(a.is_allowed("198.51.100.9".parse().unwrap()));
    }

    #[test]
    fn nothing_configured_anywhere_stays_loopback_only() {
        let a = IpAllowlist::resolve(None, None, ".env");
        assert!(a.is_allowed("127.0.0.1".parse().unwrap()));
        assert!(a.is_allowed("::1".parse().unwrap()));
        assert!(!a.is_allowed("203.0.113.5".parse().unwrap()));
        assert!(a.source().contains("loopback only"));
    }
}

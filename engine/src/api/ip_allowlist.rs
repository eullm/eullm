//! IP allowlist for the API and chat UI listeners.
//!
//! The server always binds `0.0.0.0` (the engine may legitimately run on a
//! different host than its callers — a RAG pipeline, a LAN client) so the
//! bind address itself can't be the safety boundary. This module is that
//! boundary instead: every request's source IP is checked against an
//! allowlist before it reaches any handler.
//!
//! With no `.env` file (or no `EULLM_ALLOWED_IPS` key in it), the allowlist
//! defaults to loopback only (`127.0.0.1`, `::1`) — the same effective
//! result as binding `127.0.0.1`, without needing a different bind address
//! for the "nothing configured" case. Broader access (a specific RAG host,
//! a LAN subnet) is opt-in via `.env`, never the unconfigured default.

use std::fs;
use std::net::IpAddr;
use std::path::Path;

use ipnet::IpNet;

/// A parsed, ready-to-check IP allowlist.
#[derive(Debug, Clone)]
pub struct IpAllowlist {
    nets: Vec<IpNet>,
}

impl IpAllowlist {
    /// Loopback-only — the default when nothing is configured.
    fn loopback_only() -> Self {
        Self {
            nets: vec!["127.0.0.1/32".parse().unwrap(), "::1/128".parse().unwrap()],
        }
    }

    /// Load from a `.env` file at `path`. Loopback is always included —
    /// local access shouldn't silently break just because a RAG host or LAN
    /// subnet was added — and entries from `EULLM_ALLOWED_IPS` (when the
    /// file exists, is readable, and the key is non-empty and well-formed)
    /// are added on top of it. Any failure to deliberately configure a
    /// broader list (missing file, missing key, malformed entry) means
    /// "nothing was added," never "more than loopback."
    pub fn load_from_env_file(path: &Path) -> Self {
        let mut allowlist = Self::loopback_only();

        let Ok(contents) = fs::read_to_string(path) else {
            return allowlist;
        };
        let vars = parse_env_file(&contents);
        let Some(spec) = vars
            .get("EULLM_ALLOWED_IPS")
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
        else {
            return allowlist;
        };
        match parse_allowlist(spec) {
            Ok(nets) => allowlist.nets.extend(nets),
            Err(e) => tracing::warn!(
                "EULLM_ALLOWED_IPS in {} is malformed ({e}) — ignoring, loopback-only stays in effect",
                path.display()
            ),
        }
        allowlist
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

//! Settings that are not command-line flags: read from the process
//! environment first, then from the `.env` file in the folder eullm was
//! started from.
//!
//! That is the rule `engine/CLAUDE.md` gives every `EULLM_*` setting, and the
//! one `.env.example` and the quick start promise. The perimeter settings
//! that `api::serve` resolves with their own validation and log lines
//! (`EULLM_API_KEYS`, `EULLM_ALLOWED_IPS`, `EULLM_ALLOWED_ORIGINS`,
//! `EULLM_AUDIT_DIR`, the decision policy and traces, the models folder and
//! libraries) already kept it. The rest read the environment only, so a line
//! in `.env` was ignored without a word: `EULLM_WEB_ALLOWED_DOMAINS` there
//! left `--web` free to fetch from any host, and `HF_TOKEN` there left a
//! gated download anonymous. They read their value here now.

use std::sync::OnceLock;

/// The file a setting falls back to, in the folder eullm was started from.
pub const ENV_FILE: &str = ".env";

/// Where a setting's value came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// The process environment.
    Environment,
    /// The [`ENV_FILE`].
    EnvFile,
}

impl Source {
    /// How the startup log names it.
    pub fn label(self) -> &'static str {
        match self {
            Self::Environment => "environment",
            Self::EnvFile => ENV_FILE,
        }
    }
}

/// `key`'s value: the environment's, else the `.env` file's, blank counting
/// as unset in both.
pub fn var(key: &str) -> Option<String> {
    var_with_source(key).map(|(value, _)| value)
}

/// [`var`], and where the value came from.
pub fn var_with_source(key: &str) -> Option<(String, Source)> {
    resolve(std::env::var(key).ok().as_deref(), env_file(), key)
}

/// Whether `key` is switched on: `1`, `true`, `yes` or `on`, in any case.
pub fn flag(key: &str) -> Option<Source> {
    var_with_source(key).and_then(|(value, source)| is_truthy(&value).then_some(source))
}

/// The `.env` file's contents, read once: the settings are read again and
/// again while the process runs (every download, every projector loaded),
/// and the file is configuration for the whole run, like the environment.
fn env_file() -> Option<&'static str> {
    static CONTENTS: OnceLock<Option<String>> = OnceLock::new();
    CONTENTS
        .get_or_init(|| std::fs::read_to_string(ENV_FILE).ok())
        .as_deref()
}

/// The precedence rule, pure so that it is tested without touching the
/// process environment: a value set in the environment wins, and a blank one
/// counts as unset, as it does for every other `EULLM_*` setting.
pub fn resolve(env: Option<&str>, file: Option<&str>, key: &str) -> Option<(String, Source)> {
    if let Some(value) = env.map(str::trim).filter(|v| !v.is_empty()) {
        return Some((value.to_string(), Source::Environment));
    }
    crate::api::ip_allowlist::env_file_var(file?, key).map(|value| (value, Source::EnvFile))
}

/// `1`, `true`, `yes` or `on`, in any case and with spaces around.
pub fn is_truthy(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_environment_wins_then_the_env_file() {
        let file = "# a comment\nEULLM_X=from-file\n";
        assert_eq!(
            resolve(Some("from-env"), Some(file), "EULLM_X"),
            Some(("from-env".to_string(), Source::Environment))
        );
        assert_eq!(
            resolve(None, Some(file), "EULLM_X"),
            Some(("from-file".to_string(), Source::EnvFile))
        );
    }

    #[test]
    fn a_blank_value_counts_as_unset() {
        let file = "EULLM_X=from-file\n";
        assert_eq!(
            resolve(Some("  "), Some(file), "EULLM_X"),
            Some(("from-file".to_string(), Source::EnvFile))
        );
        assert_eq!(resolve(None, Some("EULLM_X=\n"), "EULLM_X"), None);
        assert_eq!(resolve(None, None, "EULLM_X"), None);
    }

    #[test]
    fn values_are_trimmed_and_quotes_dropped() {
        assert_eq!(
            resolve(Some(" 12 \n"), None, "EULLM_X"),
            Some(("12".to_string(), Source::Environment))
        );
        assert_eq!(
            resolve(None, Some("EULLM_X=\"a.eu, b.eu\"\n"), "EULLM_X"),
            Some(("a.eu, b.eu".to_string(), Source::EnvFile))
        );
    }

    #[test]
    fn another_key_in_the_file_is_not_this_one() {
        assert_eq!(resolve(None, Some("EULLM_Y=1\n"), "EULLM_X"), None);
    }

    #[test]
    fn what_counts_as_switched_on() {
        for on in ["1", "true", "YES", " on "] {
            assert!(is_truthy(on), "{on:?}");
        }
        for off in ["0", "false", "no", "off", "", "2"] {
            assert!(!is_truthy(off), "{off:?}");
        }
    }
}

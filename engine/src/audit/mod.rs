//! AI Act audit trail module.
//!
//! Logs every inference request with metadata required for
//! EU AI Act (Regulation 2024/1689) compliance.
//!
//! Audit entries are persisted to a JSONL file at `~/.eullm/audit/audit.jsonl`.
//! Each line is a self-contained JSON object that can be queried, exported,
//! or submitted for compliance reviews.

use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub mod redact;

/// A single audit log entry for an inference request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Unique identifier for this inference
    pub id: Uuid,
    /// Timestamp of the request
    pub timestamp: DateTime<Utc>,
    /// Model used for inference
    pub model: String,
    /// Type of request (generate, chat, embedding)
    pub request_type: String,
    /// Number of input tokens
    pub input_tokens: u32,
    /// Number of output tokens
    pub output_tokens: u32,
    /// Duration of inference in milliseconds
    pub duration_ms: u64,
    /// Optional user identifier
    pub user_id: Option<String>,
    /// What a `/v1/systemone` request decided, and what the router's
    /// decision model decided on a `route` line. Absent on every other
    /// request type, and on lines written before decisions existed — both of
    /// which still parse.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decision: Option<DecisionRecord>,
    /// On a `route` line, which model `"model": "auto"` chose for a request
    /// and why. Absent on every other line, and on lines written before
    /// routing existed, which still parse.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing: Option<RoutingRecord>,
    /// On the line of a generation `"model": "auto"` routed, the route it
    /// took: its `route` line's id, and whether the fallback answered in the
    /// chosen model's place. Absent on every other line.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route: Option<RouteRef>,
    /// The MTP drafts the model proposed for this generation and the ones it
    /// kept (`--mtp`), under the API's names for them. Absent without
    /// drafts, and on lines written before they were counted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draft_n: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draft_n_accepted: Option<u32>,
}

impl AuditEntry {
    /// Create a new audit entry for an inference request.
    pub fn new(model: String, request_type: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            model,
            request_type,
            input_tokens: 0,
            output_tokens: 0,
            duration_ms: 0,
            user_id: None,
            decision: None,
            routing: None,
            route: None,
            draft_n: None,
            draft_n_accepted: None,
        }
    }
}

/// The record of one routing decision (`"model": "auto"`, or a dry run on
/// `POST /api/route`): which model answers, why, and what it was chosen
/// from. The line's own `id` is the route's: what the response's
/// `eullm.route.id` and the routed generation's audit line name it by.
///
/// The decision model's answer, with the probabilities it was read from, is
/// the line's `decision`, the record `/v1/systemone` writes; this one holds
/// what is particular to routing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RoutingRecord {
    /// The model the request named: `auto`.
    pub requested: String,
    /// The model chosen to answer.
    pub model: String,
    /// Why that one: `decided` by the decision model, or the reason the
    /// fallback answers instead (`no_decision_model`, `timeout`,
    /// `decision_error`, `no_eligible_candidate`, `only_candidate`).
    pub reason: String,
    /// The model that answers whenever the decision model does not decide.
    pub fallback: String,
    /// The models offered to the decision model, in the order it read them.
    pub candidates: Vec<String>,
    /// The configured models not offered, and why.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub excluded: Vec<ExcludedCandidate>,
    /// The decision model asked, when one was.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decision_model: Option<String>,
    /// How long routing took, the decision included, in milliseconds.
    pub decision_ms: f64,
    /// Asked of `POST /api/route`: decided and recorded, but nothing was
    /// generated. Written only when true.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub dry_run: bool,
    /// Why the decision failed, for `decision_error`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// What a routed generation's audit line says about its route; the request's
/// `route` line has the rest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RouteRef {
    /// The `id` of the request's `route` line.
    pub id: Uuid,
    /// The model the request named: `auto`.
    pub requested: String,
    /// Set when the chosen model could not be loaded and the fallback
    /// answered in its place: `load_failed: ` and why. Written only then.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback: Option<String>,
}

/// A configured model the router did not offer for a request, and why: it
/// cannot read the request's attachments, or its context is too short for it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExcludedCandidate {
    pub model: String,
    pub why: String,
}

/// The record of one `/v1/systemone` request: every answer with the
/// probabilities it was taken from, before and after calibration, so an
/// automated decision can be reconstructed and re-examined later — including
/// under a calibration chosen after the fact, since the raw log-probabilities
/// are kept too.
///
/// The state is not stored, only its SHA-256: it is the part most likely to
/// carry personal data, and the hash is enough to prove which state a
/// decision was made on by anyone who still holds it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionRecord {
    /// SHA-256 (hex) of the state exactly as the model read it.
    pub state_sha256: String,
    /// `codes` or `verdict`: how the answers were read.
    #[serde(default = "codes_readout")]
    pub readout: String,
    /// `shared_prefix`, `batched` or `separate`: the mode the answers were
    /// computed in.
    pub mode: String,
    /// `none` or `content_free`.
    pub calibration: String,
    pub temperature: f64,
    /// How the answers' `confidence` was computed:
    /// `normalized_max_probability` (jev-style's definition), or, on a line
    /// written before that became the definition, `normalized_entropy`.
    #[serde(default = "entropy_confidence")]
    pub confidence_method: String,
    /// The client had disconnected by the time these answers were ready:
    /// they were computed, and are recorded, but were never sent. Written
    /// only when true.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub client_disconnected: bool,
    /// Per question, the options the server's decision policy removed
    /// before the model read them; `labels` holds the ones it read. Written
    /// only when the policy removed something.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub policy_removed: BTreeMap<String, Vec<String>>,
    pub answers: Vec<DecisionAnswerRecord>,
}

/// One answered question inside a [`DecisionRecord`]. The vectors are in
/// the order of `labels`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionAnswerRecord {
    /// The question's id in the request.
    pub id: String,
    /// `noul`, `choice` or `score`.
    #[serde(rename = "type")]
    pub kind: String,
    /// `yes`/`no`, the option names, or the level numbers.
    pub labels: Vec<String>,
    /// Code readout: full-vocabulary log-probability of each label's answer
    /// code.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Vec<f64>>,
    /// Verdict readout: `logit(" yes") - logit(" no")` at each label's slot.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scores: Option<Vec<f64>>,
    /// Renormalized over the labels, before any calibration.
    pub raw_probabilities: Vec<f64>,
    /// After calibration: what the answer was taken from.
    pub probabilities: Vec<f64>,
    /// Code readout: share of the model's probability that went to a valid
    /// answer code.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coverage: Option<f64>,
    /// The answer as returned: the option name, the expected level, or
    /// P(yes).
    pub answer: serde_json::Value,
    /// Absent for `noul`, which reports no confidence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
}

/// `readout` of a decision record written before verdict models existed.
fn codes_readout() -> String {
    "codes".to_string()
}

/// `confidence_method` of a decision record written before it was recorded:
/// the only definition there was then.
fn entropy_confidence() -> String {
    crate::inference::decision::ENTROPY_CONFIDENCE_METHOD.to_string()
}

/// Strip ASCII control characters (newlines included) from client-controlled
/// text before it goes into a plain-text tracing line — unlike the JSONL
/// audit file, tracing's text formatter doesn't escape anything, so a
/// newline in an untrusted field (e.g. a request's `model` name) would let
/// it forge what looks like a separate log line.
pub fn sanitize_for_log(s: &str) -> String {
    s.chars().filter(|c| !c.is_control()).collect()
}

/// Audit trail logger that persists entries to a JSONL file.
///
/// The JSONL format (one JSON object per line) is chosen for:
/// - Append-only writes (crash-safe, no corruption)
/// - Easy to grep, tail, and stream
/// - Compatible with standard log analysis tools
/// - Each line is independently parseable
pub struct AuditLogger {
    log_path: PathBuf,
}

impl AuditLogger {
    /// Create a new audit logger at the default location (`~/.eullm/audit/audit.jsonl`).
    pub fn new() -> Self {
        let log_path = Self::default_path();
        Self { log_path }
    }

    /// Create a logger writing to a custom path.
    // Readers for the audit trail, kept without a caller on purpose: the AI Act
    // story needs a way to inspect what was logged, and H3-J in the hardening
    // backlog is the item that will give them one. Named individually rather
    // than covered by clippy's global `-A dead-code`, which is now off, so the
    // next orphan is a build failure instead of a line nobody reads.
    #[allow(dead_code)]
    pub fn with_path(log_path: PathBuf) -> Self {
        Self { log_path }
    }

    /// Default audit log path: `$EULLM_AUDIT_DIR/audit.jsonl` when that
    /// variable is set (see `configured_dir`), otherwise
    /// `~/.eullm/audit/audit.jsonl`.
    ///
    /// Honouring the environment variable is what makes the audit trail
    /// survive a container's lifetime. `engine/Dockerfile` sets
    /// `EULLM_AUDIT_DIR=/data/audit` and `docker-compose.yml` mounts a volume
    /// there, but for several releases nothing read it: every containerised
    /// deployment wrote its "AI Act audit trail" into the ephemeral container
    /// layer and lost it on the next `docker rm`, while the mounted volume
    /// stayed empty. Mirrors how `EULLM_MODELS_DIR` is handled in
    /// `models::store::ModelStore::default_store`.
    fn default_path() -> PathBuf {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_else(|_| std::env::temp_dir().to_string_lossy().into_owned());
        Self::resolve_path(Self::configured_dir().map(|(dir, _)| dir.as_str()), &home)
    }

    /// `EULLM_AUDIT_DIR` and where it was found: the process environment
    /// first, a `.env` file in the working directory second, like every other
    /// `EULLM_*` setting. The `.env` used to be ignored here although the
    /// documentation listed the variable among the ones read from it.
    ///
    /// Read once per process: a logger is made for every audited request, and
    /// each would otherwise read the `.env` again.
    fn configured_dir() -> Option<&'static (String, String)> {
        static DIR: std::sync::OnceLock<Option<(String, String)>> = std::sync::OnceLock::new();
        DIR.get_or_init(|| {
            let env_dir = std::env::var("EULLM_AUDIT_DIR").ok();
            let file_contents = fs::read_to_string(".env").ok();
            Self::resolve_dir(env_dir.as_deref(), file_contents.as_deref(), ".env")
        })
        .as_ref()
    }

    /// Pure resolution step behind `configured_dir`: the environment, then the
    /// `.env` file contents, blank values counting as unset.
    fn resolve_dir(
        env_dir: Option<&str>,
        file_contents: Option<&str>,
        file_label: &str,
    ) -> Option<(String, String)> {
        if let Some(dir) = env_dir.map(str::trim).filter(|d| !d.is_empty()) {
            return Some((dir.to_string(), "EULLM_AUDIT_DIR".to_string()));
        }
        let dir = crate::api::ip_allowlist::env_file_var(file_contents?, "EULLM_AUDIT_DIR")?;
        Some((dir, format!("EULLM_AUDIT_DIR ({file_label})")))
    }

    /// Where the audit destination came from, for the startup log.
    pub fn source() -> &'static str {
        Self::configured_dir().map_or("default", |(_, source)| source.as_str())
    }

    /// Pure resolution step behind `default_path`, split out so the precedence
    /// rule is testable without mutating process environment variables (which
    /// would race against every other test in the binary).
    fn resolve_path(audit_dir: Option<&str>, home: &str) -> PathBuf {
        match audit_dir.map(str::trim).filter(|d| !d.is_empty()) {
            Some(dir) => PathBuf::from(dir).join("audit.jsonl"),
            None => PathBuf::from(home)
                .join(".eullm")
                .join("audit")
                .join("audit.jsonl"),
        }
    }

    /// Whether the operator explicitly chose an audit destination via
    /// `EULLM_AUDIT_DIR`.
    ///
    /// This is what decides how hard an unwritable destination fails at
    /// startup. Someone who set the variable — or mounted a volume at it, as
    /// `engine/Dockerfile` does — has stated that the trail matters, and
    /// silently serving without one betrays that; whereas refusing to start an
    /// inference server over a *log file* nobody asked for would turn a
    /// read-only home directory into an outage. The strict posture is the
    /// operator's to choose, not ours to impose by default.
    pub fn is_explicitly_configured() -> bool {
        Self::configured_dir().is_some()
    }

    /// Verify the audit log's directory is writable, creating it if needed.
    ///
    /// Called once at startup so a misconfigured audit destination surfaces
    /// immediately instead of as a `warn!` on every request after the fact —
    /// for a component whose purpose is producing a defensible record, silently
    /// degrading to "no record" is the wrong failure mode.
    pub fn check_writable(&self) -> Result<(), String> {
        let parent = self.log_path.parent().ok_or_else(|| {
            format!(
                "audit path {} has no parent directory",
                self.log_path.display()
            )
        })?;
        fs::create_dir_all(parent)
            .map_err(|e| format!("cannot create audit directory {}: {e}", parent.display()))?;
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_path)
            .map(|_| ())
            .map_err(|e| format!("cannot write audit log {}: {e}", self.log_path.display()))
    }

    /// Log an audit entry — writes to tracing AND persists to JSONL file.
    pub fn log(&self, entry: &AuditEntry) {
        // Always log to tracing (visible in console/structured logs). Only
        // the tracing line needs sanitizing — the persisted JSONL below is
        // already safe, serde_json escapes control chars in string values.
        tracing::info!(
            audit_id = %entry.id,
            model = %sanitize_for_log(&entry.model),
            request_type = %entry.request_type,
            input_tokens = entry.input_tokens,
            output_tokens = entry.output_tokens,
            duration_ms = entry.duration_ms,
            draft_n = entry.draft_n,
            draft_n_accepted = entry.draft_n_accepted,
            "Audit: inference logged"
        );

        // Persist to JSONL file
        if let Err(e) = self.persist(entry) {
            tracing::warn!("Failed to persist audit entry: {e}");
        }
    }

    /// Persist an entry to the JSONL file.
    fn persist(&self, entry: &AuditEntry) -> Result<(), Box<dyn std::error::Error>> {
        // Ensure directory exists
        if let Some(parent) = self.log_path.parent() {
            fs::create_dir_all(parent)?;
        }

        // Serialize to single-line JSON
        let json = serde_json::to_string(entry)?;

        // Append to file (create if not exists)
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_path)?;

        // One write, not two. `writeln!` goes through `write_fmt`, which issues
        // a separate syscall for the formatted value and for the newline. Under
        // O_APPEND each individual write is atomic, but two concurrent writers
        // interleave *between* them, producing `{a}{b}\n\n` — one line holding
        // two records, and a JSONL file that no longer parses. Found by the
        // release smoke test's eight-concurrent-request check, on an audit trail
        // whose entire purpose is to be a defensible record.
        let mut line = json;
        line.push('\n');
        file.write_all(line.as_bytes())?;

        Ok(())
    }

    /// Read all audit entries from the log file.
    #[allow(dead_code)]
    pub fn read_all(&self) -> Result<Vec<AuditEntry>, Box<dyn std::error::Error>> {
        if !self.log_path.exists() {
            return Ok(vec![]);
        }

        let content = fs::read_to_string(&self.log_path)?;
        let entries: Vec<AuditEntry> = content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect();

        Ok(entries)
    }

    /// Count total audit entries without loading them all into memory.
    #[allow(dead_code)]
    pub fn count(&self) -> Result<u64, Box<dyn std::error::Error>> {
        if !self.log_path.exists() {
            return Ok(0);
        }

        let content = fs::read_to_string(&self.log_path)?;
        let count = content.lines().filter(|l| !l.trim().is_empty()).count() as u64;
        Ok(count)
    }

    /// Get the path to the audit log file.
    pub fn log_path(&self) -> &PathBuf {
        &self.log_path
    }
}

impl Default for AuditLogger {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_for_log_strips_newlines_and_control_chars() {
        assert_eq!(
            sanitize_for_log("qwen3\n2026-07-19T00:00:00Z INFO forged log line"),
            "qwen32026-07-19T00:00:00Z INFO forged log line"
        );
        assert_eq!(sanitize_for_log("qwen3-14b"), "qwen3-14b");
        assert_eq!(sanitize_for_log("a\rb\tc"), "abc");
    }

    /// `EULLM_AUDIT_DIR` is what makes the trail land on a mounted volume
    /// instead of a container's ephemeral layer — the regression this guards
    /// silently discarded every audit record in Docker deployments.
    #[test]
    fn audit_dir_env_var_takes_precedence_over_home() {
        assert_eq!(
            AuditLogger::resolve_path(Some("/data/audit"), "/home/eullm"),
            PathBuf::from("/data/audit/audit.jsonl")
        );
    }

    #[test]
    fn audit_path_falls_back_to_home_when_unset_or_blank() {
        let expected = PathBuf::from("/home/eullm/.eullm/audit/audit.jsonl");
        assert_eq!(AuditLogger::resolve_path(None, "/home/eullm"), expected);
        // An exported-but-empty variable means "unset", not "write to /".
        assert_eq!(AuditLogger::resolve_path(Some(""), "/home/eullm"), expected);
        assert_eq!(
            AuditLogger::resolve_path(Some("   "), "/home/eullm"),
            expected
        );
    }

    #[test]
    fn audit_dir_comes_from_the_environment_then_the_env_file() {
        let file = "EULLM_AUDIT_DIR=/from/file\n";
        assert_eq!(
            AuditLogger::resolve_dir(Some("/from/env"), Some(file), ".env"),
            Some(("/from/env".to_string(), "EULLM_AUDIT_DIR".to_string()))
        );
        // The `.env` line used to be ignored although it was documented.
        assert_eq!(
            AuditLogger::resolve_dir(Some(" "), Some(file), ".env"),
            Some((
                "/from/file".to_string(),
                "EULLM_AUDIT_DIR (.env)".to_string()
            ))
        );
        assert_eq!(
            AuditLogger::resolve_dir(None, Some("EULLM_AUDIT_DIR=\n"), ".env"),
            None
        );
        assert_eq!(AuditLogger::resolve_dir(None, None, ".env"), None);
    }

    #[test]
    fn check_writable_reports_an_unusable_destination() {
        let dir = std::env::temp_dir().join(format!("eullm-audit-ok-{}", uuid::Uuid::new_v4()));
        let logger = AuditLogger::with_path(dir.join("audit.jsonl"));
        assert!(
            logger.check_writable().is_ok(),
            "should create the directory"
        );
        let _ = fs::remove_dir_all(&dir);

        // A path whose parent is an existing *file* cannot be a directory.
        let file = std::env::temp_dir().join(format!("eullm-audit-bad-{}", uuid::Uuid::new_v4()));
        fs::write(&file, b"x").unwrap();
        let logger = AuditLogger::with_path(file.join("audit.jsonl"));
        assert!(logger.check_writable().is_err());
        let _ = fs::remove_file(&file);
    }

    #[test]
    fn test_audit_entry_serialization() {
        let entry = AuditEntry::new("eullm/legal-it-7b".into(), "chat".into());
        let json = serde_json::to_string(&entry).unwrap();
        let parsed: AuditEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.model, "eullm/legal-it-7b");
        assert_eq!(parsed.request_type, "chat");
        // No decision, no key: every non-decision line stays as it was.
        assert!(!json.contains("decision"));
    }

    #[test]
    fn lines_written_before_decisions_existed_still_parse() {
        let old = r#"{"id":"67e55044-10b1-426f-9247-bb680e5fe0c8","timestamp":"2026-09-01T10:00:00Z","model":"qwen3-8b","request_type":"chat","input_tokens":12,"output_tokens":40,"duration_ms":900,"user_id":null}"#;
        let parsed: AuditEntry = serde_json::from_str(old).unwrap();
        assert!(parsed.decision.is_none());
        assert!(parsed.routing.is_none(), "nor before routing existed");
    }

    /// A route line keeps which model was chosen, from what, and why; a
    /// line that routed nothing has no `routing` key at all, and a dry run
    /// says so only when it is one.
    #[test]
    fn a_routing_record_round_trips() {
        let mut entry = AuditEntry::new("jev-style-0.8b".into(), "route".into());
        assert!(!serde_json::to_string(&entry).unwrap().contains("routing"));
        let record = RoutingRecord {
            requested: "auto".into(),
            model: "qwen3-4b".into(),
            reason: "decided".into(),
            fallback: "qwen3-8b".into(),
            candidates: vec!["qwen3-4b".into(), "qwen3-8b".into()],
            excluded: vec![ExcludedCandidate {
                model: "qwen3-1.7b".into(),
                why: "its context of 2048 tokens per request is shorter than the request".into(),
            }],
            decision_model: Some("jev-style-0.8b".into()),
            decision_ms: 38.2,
            dry_run: false,
            error: None,
        };
        entry.routing = Some(record.clone());
        let json = serde_json::to_string(&entry).unwrap();
        assert!(
            !json.contains("dry_run") && !json.contains("\"error\""),
            "{json}"
        );
        let parsed: AuditEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.routing.as_ref(), Some(&record));

        entry.routing = Some(RoutingRecord {
            dry_run: true,
            excluded: Vec::new(),
            ..record
        });
        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains(r#""dry_run":true"#), "{json}");
        assert!(!json.contains("excluded"), "{json}");
    }

    /// A routed generation's line names its route line, and says the
    /// fallback answered only when it did.
    #[test]
    fn a_route_reference_round_trips() {
        let mut entry = AuditEntry::new("qwen3-4b".into(), "chat".into());
        assert!(!serde_json::to_string(&entry).unwrap().contains("route"));
        let reference = RouteRef {
            id: Uuid::new_v4(),
            requested: "auto".into(),
            fallback: None,
        };
        entry.route = Some(reference.clone());
        let json = serde_json::to_string(&entry).unwrap();
        assert!(!json.contains("fallback"), "{json}");
        let parsed: AuditEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.route.as_ref(), Some(&reference));

        entry.route = Some(RouteRef {
            fallback: Some("load_failed: out of memory".into()),
            ..reference
        });
        let json = serde_json::to_string(&entry).unwrap();
        assert!(
            json.contains(r#""fallback":"load_failed: out of memory""#),
            "{json}"
        );
    }

    #[test]
    fn a_decision_record_round_trips() {
        let mut entry = AuditEntry::new("qwen3-4b".into(), "systemone".into());
        entry.decision = Some(DecisionRecord {
            state_sha256: "ab".repeat(32),
            readout: "codes".into(),
            mode: "shared_prefix".into(),
            calibration: "none".into(),
            temperature: 1.0,
            confidence_method: "normalized_max_probability".into(),
            client_disconnected: false,
            policy_removed: BTreeMap::new(),
            answers: vec![DecisionAnswerRecord {
                id: "area".into(),
                kind: "choice".into(),
                labels: vec!["civile".into(), "penale".into()],
                logprobs: Some(vec![-0.1, -2.4]),
                scores: None,
                raw_probabilities: vec![0.91, 0.09],
                probabilities: vec![0.91, 0.09],
                coverage: Some(0.99),
                answer: serde_json::json!("civile"),
                confidence: Some(0.56),
            }],
        });
        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains(r#""type":"choice""#), "{json}");
        // Delivered, as nearly every decision is: no flag on the line. No
        // policy removed anything: no key either.
        assert!(!json.contains("client_disconnected"), "{json}");
        assert!(!json.contains("policy_removed"), "{json}");
        let parsed: AuditEntry = serde_json::from_str(&json).unwrap();
        let decision = parsed.decision.unwrap();
        assert_eq!(decision.confidence_method, "normalized_max_probability");
        assert!(decision.policy_removed.is_empty());
        let answer = &decision.answers[0];
        assert_eq!(answer.labels, ["civile", "penale"]);
        assert_eq!(answer.answer, "civile");

        // What the policy removed is on the line, next to what the model read.
        let mut record = entry.decision.clone().unwrap();
        record
            .policy_removed
            .insert("area".into(), vec!["amministrativo".into()]);
        entry.decision = Some(record);
        let json = serde_json::to_string(&entry).unwrap();
        assert!(
            json.contains(r#""policy_removed":{"area":["amministrativo"]}"#),
            "{json}"
        );
        let parsed: AuditEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(
            parsed.decision.unwrap().policy_removed["area"],
            ["amministrativo"]
        );
    }

    /// A decision line from before the confidence method was recorded still
    /// reads, and says which definition its confidences follow.
    #[test]
    fn a_decision_record_without_its_confidence_method_is_entropy_based() {
        let old = r#"{"state_sha256":"ab","mode":"shared_prefix","calibration":"none","temperature":1.0,"answers":[]}"#;
        let record: DecisionRecord = serde_json::from_str(old).unwrap();
        assert_eq!(record.confidence_method, "normalized_entropy");
        assert_eq!(record.readout, "codes");
    }

    #[test]
    fn test_audit_logger_persist_and_read() {
        let tmp_dir = std::env::temp_dir().join(format!("eullm-test-{}", uuid::Uuid::new_v4()));
        let log_path = tmp_dir.join("test-audit.jsonl");
        let logger = AuditLogger::with_path(log_path.clone());

        // Write two entries
        let mut entry1 = AuditEntry::new("model-a".into(), "generate".into());
        entry1.input_tokens = 10;
        entry1.output_tokens = 50;
        entry1.duration_ms = 200;
        logger.log(&entry1);

        let mut entry2 = AuditEntry::new("model-b".into(), "chat".into());
        entry2.input_tokens = 25;
        entry2.output_tokens = 100;
        entry2.duration_ms = 500;
        logger.log(&entry2);

        // Read them back
        let entries = logger.read_all().unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].model, "model-a");
        assert_eq!(entries[0].output_tokens, 50);
        assert_eq!(entries[1].model, "model-b");
        assert_eq!(entries[1].duration_ms, 500);

        // Count
        assert_eq!(logger.count().unwrap(), 2);

        // Cleanup
        let _ = fs::remove_dir_all(tmp_dir);
    }
}

#[cfg(test)]
mod concurrent_append_tests {
    use super::*;
    use std::sync::Arc;

    /// Every record must survive concurrent writers, intact and on its own line.
    ///
    /// The regression: `writeln!` on a `File` is two syscalls, so two threads
    /// could interleave between the JSON and its newline and leave a line with
    /// two records on it. `read_all` then either fails or silently drops
    /// entries — on a trail that exists to be a defensible record of what the
    /// system did.
    #[test]
    fn concurrent_writers_never_interleave_a_line() {
        let dir = std::env::temp_dir().join(format!("eullm-audit-race-{}", Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("audit.jsonl");
        let logger = Arc::new(AuditLogger::with_path(path.clone()));

        const THREADS: usize = 8;
        const PER_THREAD: usize = 40;
        let handles: Vec<_> = (0..THREADS)
            .map(|t| {
                let logger = Arc::clone(&logger);
                std::thread::spawn(move || {
                    for i in 0..PER_THREAD {
                        // Vary the length so an interleave cannot be masked by
                        // every record happening to be the same size.
                        let model = format!("model-{t}-{}", "x".repeat(i % 17));
                        let mut e = AuditEntry::new(model, "chat".to_string());
                        e.input_tokens = i as u32;
                        logger.log(&e);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }

        let contents = fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = contents.lines().filter(|l| !l.trim().is_empty()).collect();
        assert_eq!(
            lines.len(),
            THREADS * PER_THREAD,
            "every record must be on exactly one line"
        );
        for (n, line) in lines.iter().enumerate() {
            serde_json::from_str::<AuditEntry>(line)
                .unwrap_or_else(|e| panic!("line {} does not parse: {e}\n{line}", n + 1));
        }
        assert_eq!(logger.read_all().unwrap().len(), THREADS * PER_THREAD);

        fs::remove_dir_all(&dir).ok();
    }
}

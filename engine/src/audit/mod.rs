//! AI Act audit trail module.
//!
//! Logs every inference request with metadata required for
//! EU AI Act (Regulation 2024/1689) compliance.
//!
//! Audit entries are persisted to a JSONL file at `~/.eullm/audit/audit.jsonl`.
//! Each line is a self-contained JSON object that can be queried, exported,
//! or submitted for compliance reviews.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

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
        }
    }
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
    /// variable is set, otherwise `~/.eullm/audit/audit.jsonl`.
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
        let audit_dir = std::env::var("EULLM_AUDIT_DIR").ok();
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_else(|_| std::env::temp_dir().to_string_lossy().into_owned());
        Self::resolve_path(audit_dir.as_deref(), &home)
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
        std::env::var("EULLM_AUDIT_DIR").is_ok_and(|d| !d.trim().is_empty())
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

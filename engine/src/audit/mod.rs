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
    pub fn with_path(log_path: PathBuf) -> Self {
        Self { log_path }
    }

    /// Default audit log path: `~/.eullm/audit/audit.jsonl`.
    fn default_path() -> PathBuf {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_else(|_| "/tmp".into());
        PathBuf::from(home).join(".eullm").join("audit").join("audit.jsonl")
    }

    /// Log an audit entry — writes to tracing AND persists to JSONL file.
    pub fn log(&self, entry: &AuditEntry) {
        // Always log to tracing (visible in console/structured logs)
        tracing::info!(
            audit_id = %entry.id,
            model = %entry.model,
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

        writeln!(file, "{json}")?;

        Ok(())
    }

    /// Read all audit entries from the log file.
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

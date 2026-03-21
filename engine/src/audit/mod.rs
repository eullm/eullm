//! AI Act audit trail module.
//!
//! Logs every inference request with metadata required for
//! EU AI Act (Regulation 2024/1689) compliance.

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

/// Audit trail logger.
pub struct AuditLogger {
    // TODO: configurable storage backend (file, database)
}

impl AuditLogger {
    /// Create a new audit logger.
    pub fn new() -> Self {
        Self {}
    }

    /// Log an audit entry.
    pub fn log(&self, entry: &AuditEntry) {
        tracing::info!(
            audit_id = %entry.id,
            model = %entry.model,
            request_type = %entry.request_type,
            "Audit: inference logged"
        );
        // TODO: persist to audit storage
    }
}

impl Default for AuditLogger {
    fn default() -> Self {
        Self::new()
    }
}

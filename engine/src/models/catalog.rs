//! Model catalog — list of permissively-licensed open-weight models that
//! `eullm` can discover, download, and run.
//!
//! Two layers:
//!   - Embedded catalog (compiled into the binary from `catalog/v1/catalog.json`)
//!     so the CLI always has something to show, even offline or before any
//!     remote service is reachable.
//!   - Remote catalog fetch (best-effort) that pulls the freshest version
//!     from GitHub raw — `fetch_remote()` below. Falls back to embedded on
//!     any error.
//!
//! When the eullm.eu registry goes live, the URL constant moves to it and
//! nothing else changes. Until then, GitHub raw is a free, fast, EU-cached
//! CDN that doesn't require us to stand up infrastructure on day one.

use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

/// Where to fetch the live catalog from. Switched to `registry.eullm.eu`
/// once the proxy on Contabo is live.
pub const REMOTE_CATALOG_URL: &str =
    "https://raw.githubusercontent.com/eullm/eullm/main/catalog/v1/catalog.json";

/// One entry in the EuLLM model catalog.
///
/// All fields are populated from the catalog JSON. Anything marked
/// `#[serde(default)]` is optional in the JSON and gets a sensible default.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogEntry {
    /// Short identifier the user types: `qwen3-8b`, `mistral-nemo-12b`.
    pub id: String,
    /// Human-readable name: "Qwen3 8B Instruct".
    pub name: String,
    pub description: String,
    pub params_b: f32,
    pub quantization: String,
    pub size_bytes: u64,
    pub vram_gb: u32,
    pub languages: Vec<String>,
    /// `general`, `code`, `reasoning`, `business`, `legal`, `medical`, etc.
    pub domain: String,
    /// SPDX license string. Catalog only accepts permissive licenses.
    pub license: String,
    #[serde(default)]
    pub tags: Vec<String>,
    /// HuggingFace repo holding the GGUF: `Qwen/Qwen3-8B-GGUF`.
    pub hf_repo: String,
    /// Filename inside the repo: `Qwen3-8B-Q4_K_M.gguf`.
    pub hf_filename: String,
    /// `sha256:...` digest of the GGUF for integrity verification.
    /// Empty string = unverified (engine downloads anyway and warns).
    #[serde(default)]
    pub digest: String,
    /// Whether this is one of the curated "start here" picks.
    #[serde(default)]
    pub recommended: bool,
    /// Optional multimodal projector — HF repo where the mmproj GGUF lives.
    /// When set (together with `mmproj_filename`), `eullm pull` will download
    /// it alongside the main GGUF so a multimodal engine build can load it.
    /// For text-only models leave it absent.
    #[serde(default)]
    pub mmproj_repo: Option<String>,
    /// Optional multimodal projector filename inside `mmproj_repo`
    /// (e.g. `mmproj-F16.gguf`). Ignored without `mmproj_repo`.
    #[serde(default)]
    pub mmproj_filename: Option<String>,
}

impl CatalogEntry {
    /// Model family derived from the id prefix: `qwen3-8b` → `qwen3`.
    /// Used by Ollama-compatible `/api/show` to fill the `family` field.
    pub fn base(&self) -> String {
        self.id
            .split('-')
            .next()
            .unwrap_or(&self.id)
            .to_string()
    }

    /// Source HF path, used by Ollama-compatible `/api/show`.
    pub fn source_model(&self) -> &str {
        &self.hf_repo
    }
}

/// Top-level catalog document — what `catalog.json` deserializes into.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Catalog {
    pub version: u32,
    #[serde(default)]
    pub updated_at: String,
    #[serde(default)]
    pub notes: String,
    pub models: Vec<CatalogEntry>,
}

/// Embedded catalog — baked into the binary at build time.
/// This is the source of truth when offline or before the remote registry
/// is reachable. The JSON file lives at the repo root: `catalog/v1/catalog.json`.
const EMBEDDED_CATALOG_JSON: &str = include_str!("../../../catalog/v1/catalog.json");

pub static EU_CATALOG: LazyLock<Vec<CatalogEntry>> = LazyLock::new(|| {
    match serde_json::from_str::<Catalog>(EMBEDDED_CATALOG_JSON) {
        Ok(c) => c.models,
        Err(e) => {
            // This would only happen if someone broke catalog.json — fail loud.
            tracing::error!("Embedded catalog JSON is malformed: {e}");
            Vec::new()
        }
    }
});

/// Fetch the live catalog from GitHub raw (or the eullm.eu registry once
/// it's live). Falls back to the embedded catalog on any error.
///
/// Use this in interactive contexts (the picker) where freshness matters.
/// For backward-compatible synchronous lookups, `EU_CATALOG` is fine.
pub async fn fetch_remote() -> Vec<CatalogEntry> {
    let client = match reqwest::Client::builder()
        .user_agent(concat!("eullm/", env!("CARGO_PKG_VERSION")))
        .timeout(std::time::Duration::from_secs(5))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            tracing::debug!("could not build HTTP client: {e}");
            return EU_CATALOG.clone();
        }
    };

    match client.get(REMOTE_CATALOG_URL).send().await {
        Ok(resp) if resp.status().is_success() => match resp.text().await {
            Ok(body) => match serde_json::from_str::<Catalog>(&body) {
                Ok(c) => {
                    tracing::debug!("remote catalog: {} models", c.models.len());
                    c.models
                }
                Err(e) => {
                    tracing::debug!("remote catalog JSON malformed ({e}); falling back");
                    EU_CATALOG.clone()
                }
            },
            Err(e) => {
                tracing::debug!("remote catalog body read failed ({e}); falling back");
                EU_CATALOG.clone()
            }
        },
        Ok(resp) => {
            tracing::debug!(
                "remote catalog HTTP {}; falling back to embedded",
                resp.status()
            );
            EU_CATALOG.clone()
        }
        Err(e) => {
            tracing::debug!("remote catalog unreachable ({e}); falling back");
            EU_CATALOG.clone()
        }
    }
}

/// Find a model in the (embedded) catalog by user-supplied identifier.
///
/// Matches against, in order:
///   1. exact `id` (lowercased)
///   2. exact `name` (lowercased)
///   3. legacy long form `eullm/{id}` (for users of older eullm verticali names)
///
/// Used both by the CLI (`eullm run <id>`) and by the API
/// (Ollama-compatible `/api/show` lookups).
pub fn find_model(name: &str) -> Option<&'static CatalogEntry> {
    let needle = name.trim().to_lowercase();
    let needle_short = needle.strip_prefix("eullm/").unwrap_or(&needle);

    EU_CATALOG.iter().find(|m| {
        m.id.to_lowercase() == needle
            || m.id.to_lowercase() == needle_short
            || m.name.to_lowercase() == needle
            || m.hf_filename.to_lowercase() == needle
    })
}

/// List models filtered by domain (`general`, `code`, `reasoning`, ...).
pub fn find_by_domain(domain: &str) -> Vec<&'static CatalogEntry> {
    EU_CATALOG
        .iter()
        .filter(|m| m.domain.eq_ignore_ascii_case(domain))
        .collect()
}

/// List models filtered by language code (`it`, `de`, `fr`, ...).
pub fn find_by_language(lang: &str) -> Vec<&'static CatalogEntry> {
    EU_CATALOG
        .iter()
        .filter(|m| m.languages.iter().any(|l| l.eq_ignore_ascii_case(lang)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_catalog_parses() {
        // If this fails, the JSON in catalog/v1/catalog.json is malformed.
        assert!(!EU_CATALOG.is_empty(), "embedded catalog must not be empty");
    }

    #[test]
    fn find_model_works_with_short_id() {
        assert!(find_model("qwen3-8b").is_some());
        assert!(find_model("QWEN3-8B").is_some(), "should be case-insensitive");
        assert!(find_model("does-not-exist").is_none());
    }

    #[test]
    fn legacy_eullm_prefix_still_works() {
        // Older eullm verticali used `eullm/legal-it-7b` style — keep it working.
        // We don't ship those models yet, so we just verify the strip logic
        // by checking that a known model still resolves through the prefix path.
        assert!(find_model("eullm/qwen3-8b").is_some());
    }

    #[test]
    fn base_derives_from_id() {
        let q = find_model("qwen3-8b").unwrap();
        assert_eq!(q.base(), "qwen3");
    }
}

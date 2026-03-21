//! EU model catalog — the list of models available from EULLM Hub.
//!
//! Contains both pre-verticalizzati 7B models (the demo lineup)
//! and larger models for users with more powerful hardware.

use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

/// Metadata for a model in the EU catalog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogEntry {
    pub name: String,
    pub description: String,
    pub languages: Vec<String>,
    pub base: String,
    pub vram_gb: u32,
    pub size_bytes: u64,
    pub license: String,
    pub digest: String,
    /// Domain specialization (e.g., "legal", "medical", "finance", "general")
    pub domain: String,
    /// Source model this was verticalizzato from
    pub source_model: String,
    /// HuggingFace repo for downloading the GGUF file (e.g., "eullm/legal-it-7b-GGUF").
    /// Empty if not yet available.
    pub hf_repo: String,
    /// GGUF filename in the HuggingFace repo (e.g., "legal-it-7b-q4_k_m.gguf").
    pub hf_filename: String,
}

/// Built-in catalog of EULLM Hub models.
///
/// In production this will be fetched from the registry API.
/// For now we embed a static catalog for the CLI to work offline.
pub static EU_CATALOG: LazyLock<Vec<CatalogEntry>> = LazyLock::new(|| {
    vec![
        // ── Verticalizzati 7B (consumer GPU / laptop) ──────────────────
        CatalogEntry {
            name: "eullm/legal-it-7b".into(),
            description: "Italian legal domain — civil code, GDPR, Cassazione rulings".into(),
            languages: vec!["it".into(), "en".into()],
            base: "qwen3".into(),
            vram_gb: 6,
            size_bytes: 4_500_000_000,
            license: "Apache-2.0".into(),
            digest: "sha256:le7a1it0000000000000000000000001".into(),
            domain: "legal".into(),
            source_model: "Qwen/Qwen3-14B".into(),
            hf_repo: "eullm/legal-it-7b-GGUF".into(),
            hf_filename: "legal-it-7b-q4_k_m.gguf".into(),
        },
        CatalogEntry {
            name: "eullm/medical-de-7b".into(),
            description: "German medical — clinical guidelines, medical documentation".into(),
            languages: vec!["de".into(), "en".into()],
            base: "qwen3".into(),
            vram_gb: 6,
            size_bytes: 4_500_000_000,
            license: "Apache-2.0".into(),
            digest: "sha256:med1ca1de000000000000000000000001".into(),
            domain: "medical".into(),
            source_model: "Qwen/Qwen3-14B".into(),
            hf_repo: "eullm/medical-de-7b-GGUF".into(),
            hf_filename: "medical-de-7b-q4_k_m.gguf".into(),
        },
        CatalogEntry {
            name: "eullm/finance-fr-7b".into(),
            description: "French finance — AMF regulations, BCE directives, banking".into(),
            languages: vec!["fr".into(), "en".into()],
            base: "qwen3".into(),
            vram_gb: 6,
            size_bytes: 4_500_000_000,
            license: "Apache-2.0".into(),
            digest: "sha256:f1nancefr000000000000000000000001".into(),
            domain: "finance".into(),
            source_model: "Qwen/Qwen3-14B".into(),
            hf_repo: "eullm/finance-fr-7b-GGUF".into(),
            hf_filename: "finance-fr-7b-q4_k_m.gguf".into(),
        },
        // ── General purpose models ─────────────────────────────────────
        CatalogEntry {
            name: "eullm/general-eu-7b".into(),
            description: "General purpose multilingual — runs on any laptop".into(),
            languages: vec!["en", "it", "de", "fr", "es", "pt", "nl"]
                .into_iter()
                .map(String::from)
                .collect(),
            base: "qwen3".into(),
            vram_gb: 6,
            size_bytes: 4_500_000_000,
            license: "Apache-2.0".into(),
            digest: "sha256:genera1eu7b0000000000000000000001".into(),
            domain: "general".into(),
            source_model: "Qwen/Qwen3-14B".into(),
            hf_repo: "eullm/general-eu-7b-GGUF".into(),
            hf_filename: "general-eu-7b-q4_k_m.gguf".into(),
        },
        CatalogEntry {
            name: "eullm/general-eu-14b".into(),
            description: "General purpose multilingual — requires dedicated GPU".into(),
            languages: vec!["en", "it", "de", "fr", "es", "pt", "nl"]
                .into_iter()
                .map(String::from)
                .collect(),
            base: "qwen3".into(),
            vram_gb: 10,
            size_bytes: 8_500_000_000,
            license: "Apache-2.0".into(),
            digest: "sha256:genera1eu14b000000000000000000001".into(),
            domain: "general".into(),
            source_model: "Qwen/Qwen3-30B-A3B".into(),
            hf_repo: "eullm/general-eu-14b-GGUF".into(),
            hf_filename: "general-eu-14b-q4_k_m.gguf".into(),
        },
        // ── Specialized 14B (workstation GPU) ──────────────────────────
        CatalogEntry {
            name: "eullm/legal-it-14b".into(),
            description: "Italian legal domain — full-size, higher quality".into(),
            languages: vec!["it".into(), "en".into()],
            base: "qwen3".into(),
            vram_gb: 10,
            size_bytes: 8_200_000_000,
            license: "Apache-2.0".into(),
            digest: "sha256:le7a1it14b00000000000000000000001".into(),
            domain: "legal".into(),
            source_model: "Qwen/Qwen3-30B-A3B".into(),
            hf_repo: "eullm/legal-it-14b-GGUF".into(),
            hf_filename: "legal-it-14b-q4_k_m.gguf".into(),
        },
        CatalogEntry {
            name: "eullm/code-eu-14b".into(),
            description: "Multilingual coding model".into(),
            languages: vec!["en", "it", "de", "fr", "es"]
                .into_iter()
                .map(String::from)
                .collect(),
            base: "deepseek".into(),
            vram_gb: 10,
            size_bytes: 8_500_000_000,
            license: "MIT".into(),
            digest: "sha256:c0deeu14b000000000000000000000001".into(),
            domain: "code".into(),
            source_model: "deepseek-ai/DeepSeek-V3".into(),
            hf_repo: "eullm/code-eu-14b-GGUF".into(),
            hf_filename: "code-eu-14b-q4_k_m.gguf".into(),
        },
    ]
});

/// Find a model in the catalog by name.
///
/// Supports both full name (`eullm/legal-it-7b`) and short name (`legal-it-7b`).
pub fn find_model(name: &str) -> Option<&CatalogEntry> {
    EU_CATALOG.iter().find(|m| {
        m.name == name || m.name.strip_prefix("eullm/").is_some_and(|short| short == name)
    })
}

/// List models filtered by domain.
pub fn find_by_domain(domain: &str) -> Vec<&CatalogEntry> {
    EU_CATALOG
        .iter()
        .filter(|m| m.domain == domain)
        .collect()
}

/// List models filtered by language.
pub fn find_by_language(lang: &str) -> Vec<&CatalogEntry> {
    EU_CATALOG
        .iter()
        .filter(|m| m.languages.iter().any(|l| l == lang))
        .collect()
}

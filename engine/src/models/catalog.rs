//! EU model catalog — the list of models available from EULLM Hub.

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
}

/// Built-in catalog of EULLM Hub models.
///
/// In production this will be fetched from the registry API.
/// For now we embed a static catalog for the CLI to work offline.
pub static EU_CATALOG: LazyLock<Vec<CatalogEntry>> = LazyLock::new(|| {
    vec![
        CatalogEntry {
            name: "eullm/general-eu-14b".into(),
            description: "General purpose multilingual model for European use".into(),
            languages: vec!["en", "it", "de", "fr", "es", "pt", "nl"]
                .into_iter()
                .map(String::from)
                .collect(),
            base: "qwen3".into(),
            vram_gb: 10,
            size_bytes: 8_500_000_000,
            license: "Apache-2.0".into(),
            digest: "sha256:abcdef1234567890".into(),
        },
        CatalogEntry {
            name: "eullm/legal-it-14b".into(),
            description: "Italian legal domain model".into(),
            languages: vec!["it".into(), "en".into()],
            base: "qwen3".into(),
            vram_gb: 10,
            size_bytes: 8_200_000_000,
            license: "Apache-2.0".into(),
            digest: "sha256:fedcba0987654321".into(),
        },
        CatalogEntry {
            name: "eullm/finance-de-8b".into(),
            description: "German finance domain model".into(),
            languages: vec!["de".into(), "en".into()],
            base: "mistral".into(),
            vram_gb: 6,
            size_bytes: 4_800_000_000,
            license: "Apache-2.0".into(),
            digest: "sha256:112233aabbccddee".into(),
        },
        CatalogEntry {
            name: "eullm/healthcare-fr-14b".into(),
            description: "French healthcare domain model".into(),
            languages: vec!["fr".into(), "en".into()],
            base: "qwen3".into(),
            vram_gb: 10,
            size_bytes: 8_400_000_000,
            license: "Apache-2.0".into(),
            digest: "sha256:445566aabbccddee".into(),
        },
        CatalogEntry {
            name: "eullm/code-eu-32b".into(),
            description: "Multilingual coding model".into(),
            languages: vec!["en", "it", "de", "fr", "es"]
                .into_iter()
                .map(String::from)
                .collect(),
            base: "deepseek".into(),
            vram_gb: 20,
            size_bytes: 18_000_000_000,
            license: "MIT".into(),
            digest: "sha256:778899aabbccddee".into(),
        },
        CatalogEntry {
            name: "eullm/customer-es-8b".into(),
            description: "Spanish customer service model".into(),
            languages: vec!["es".into(), "en".into()],
            base: "mistral".into(),
            vram_gb: 6,
            size_bytes: 4_600_000_000,
            license: "Apache-2.0".into(),
            digest: "sha256:aabb00112233eeff".into(),
        },
    ]
});

/// Find a model in the catalog by name.
///
/// Supports both full name (`eullm/legal-it-14b`) and short name (`legal-it-14b`).
pub fn find_model(name: &str) -> Option<&CatalogEntry> {
    EU_CATALOG.iter().find(|m| {
        m.name == name || m.name.strip_prefix("eullm/").is_some_and(|short| short == name)
    })
}

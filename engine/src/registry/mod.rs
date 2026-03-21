//! EU Model Registry client.
//!
//! Handles pulling and managing models from EU-hosted registries
//! (Hetzner DE, OVH FR).

/// Known EU registry endpoints.
pub const EU_REGISTRIES: &[&str] = &[
    "https://registry.eullm.eu", // Primary (Hetzner, Nuremberg DE)
];

/// Model metadata from the registry.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub size: u64,
    pub digest: String,
    pub format: String,
}

/// Pull a model from the EU registry.
pub async fn pull(_model: &str) -> Result<(), Box<dyn std::error::Error>> {
    // TODO: implement model download from EU registry
    Ok(())
}

/// List models available in the EU registry.
pub async fn list_remote() -> Result<Vec<ModelInfo>, Box<dyn std::error::Error>> {
    // TODO: query registry API
    Ok(vec![])
}

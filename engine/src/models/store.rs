//! Local model storage — manages downloaded models on disk.

use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::catalog::CatalogEntry;

/// Manifest written to disk for each pulled model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    pub name: String,
    pub description: String,
    pub languages: Vec<String>,
    pub base: String,
    pub vram_gb: u32,
    pub size_bytes: u64,
    pub license: String,
    pub digest: String,
    pub pulled_at: String,
    pub status: String,
}

/// Manages the local model directory (`~/.eullm/models/`).
pub struct ModelStore {
    root: PathBuf,
}

impl ModelStore {
    /// Create a store at the default location (`~/.eullm/models/`).
    pub fn default_store() -> Result<Self, Box<dyn std::error::Error>> {
        let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE"))?;
        let root = PathBuf::from(home).join(".eullm").join("models");
        fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    /// Create a store at a custom path.
    pub fn new(root: PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    /// "Pull" a model — writes the manifest to disk.
    ///
    /// In production this will download the GGUF file from the EU registry.
    /// For now it creates a manifest marking the model as available.
    pub fn pull(&self, entry: &CatalogEntry) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let short_name = entry
            .name
            .strip_prefix("eullm/")
            .unwrap_or(&entry.name);
        let model_dir = self.root.join(short_name);
        fs::create_dir_all(&model_dir)?;

        let manifest = ModelManifest {
            name: entry.name.clone(),
            description: entry.description.clone(),
            languages: entry.languages.clone(),
            base: entry.base.clone(),
            vram_gb: entry.vram_gb,
            size_bytes: entry.size_bytes,
            license: entry.license.clone(),
            digest: entry.digest.clone(),
            pulled_at: chrono::Utc::now().to_rfc3339(),
            status: "mock".into(),
        };

        let manifest_path = model_dir.join("manifest.json");
        let json = serde_json::to_string_pretty(&manifest)?;
        fs::write(&manifest_path, json)?;

        Ok(model_dir)
    }

    /// List all locally available models.
    pub fn list(&self) -> Result<Vec<ModelManifest>, Box<dyn std::error::Error>> {
        let mut models = Vec::new();

        if !self.root.exists() {
            return Ok(models);
        }

        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                let manifest_path = entry.path().join("manifest.json");
                if manifest_path.exists() {
                    let data = fs::read_to_string(&manifest_path)?;
                    let manifest: ModelManifest = serde_json::from_str(&data)?;
                    models.push(manifest);
                }
            }
        }

        Ok(models)
    }

    /// Get a single model's manifest by name.
    pub fn get(&self, name: &str) -> Result<Option<ModelManifest>, Box<dyn std::error::Error>> {
        let short_name = name.strip_prefix("eullm/").unwrap_or(name);
        let manifest_path = self.root.join(short_name).join("manifest.json");

        if !manifest_path.exists() {
            return Ok(None);
        }

        let data = fs::read_to_string(&manifest_path)?;
        let manifest: ModelManifest = serde_json::from_str(&data)?;
        Ok(Some(manifest))
    }

    /// Check if a model exists locally.
    pub fn exists(&self, name: &str) -> bool {
        let short_name = name.strip_prefix("eullm/").unwrap_or(name);
        self.root.join(short_name).join("manifest.json").exists()
    }

    /// Return the path to the model directory.
    pub fn model_path(&self, name: &str) -> PathBuf {
        let short_name = name.strip_prefix("eullm/").unwrap_or(name);
        self.root.join(short_name)
    }
}

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
    /// Path to the GGUF file relative to the model directory (if downloaded).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gguf_file: Option<String>,
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

    /// Write a manifest to disk for a pulled model.
    pub fn write_manifest(
        &self,
        entry: &CatalogEntry,
        status: &str,
        gguf_file: Option<&str>,
    ) -> Result<PathBuf, Box<dyn std::error::Error>> {
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
            status: status.into(),
            gguf_file: gguf_file.map(String::from),
        };

        let manifest_path = model_dir.join("manifest.json");
        let json = serde_json::to_string_pretty(&manifest)?;
        fs::write(&manifest_path, json)?;

        Ok(model_dir)
    }

    /// Get the GGUF file path for a locally available model.
    pub fn gguf_path(&self, name: &str) -> Option<PathBuf> {
        let short_name = name.strip_prefix("eullm/").unwrap_or(name);
        let model_dir = self.root.join(short_name);

        // Check manifest for recorded gguf_file
        let manifest_path = model_dir.join("manifest.json");
        if manifest_path.exists()
            && let Ok(data) = fs::read_to_string(&manifest_path)
            && let Ok(manifest) = serde_json::from_str::<ModelManifest>(&data)
            && let Some(ref gguf) = manifest.gguf_file
        {
            let path = model_dir.join(gguf);
            if path.exists() {
                return Some(path);
            }
        }

        // Fallback: look for any .gguf file in the model directory
        if model_dir.is_dir() && let Ok(entries) = fs::read_dir(&model_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "gguf") {
                    return Some(path);
                }
            }
        }

        None
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

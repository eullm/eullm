//! Local model storage — manages downloaded models on disk.

use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::catalog::CatalogEntry;

/// Manifest written to disk for each pulled model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    /// Addressable identifier — the catalog id and the on-disk directory key
    /// (e.g. `gemma-4-12b`). This is the exact string to pass to `eullm run`.
    /// `#[serde(default)]` + directory-name backfill in `list`/`get` keeps
    /// manifests written before this field existed working without a re-pull.
    #[serde(default)]
    pub id: String,
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
    /// Filename of the multimodal projector (mmproj) GGUF inside the same
    /// model directory, when this is a vision/audio-capable model and the
    /// projector was pulled alongside the main weights. Absent for text-only
    /// models, and silently ignored by text-only engine builds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mmproj_file: Option<String>,
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

    /// Write a manifest to disk for a pulled model.
    ///
    /// The on-disk directory is keyed by the catalog `id` (filesystem-safe,
    /// stable across catalog revisions). The manifest still records the
    /// human `name` for display.
    pub fn write_manifest(
        &self,
        entry: &CatalogEntry,
        status: &str,
        gguf_file: Option<&str>,
        mmproj_file: Option<&str>,
    ) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let model_dir = self.root.join(&entry.id);
        fs::create_dir_all(&model_dir)?;

        let manifest = ModelManifest {
            id: entry.id.clone(),
            name: entry.name.clone(),
            description: entry.description.clone(),
            languages: entry.languages.clone(),
            base: entry.base(),
            vram_gb: entry.vram_gb,
            size_bytes: entry.size_bytes,
            license: entry.license.clone(),
            digest: entry.digest.clone(),
            pulled_at: chrono::Utc::now().to_rfc3339(),
            status: status.into(),
            gguf_file: gguf_file.map(String::from),
            mmproj_file: mmproj_file.map(String::from),
        };

        let manifest_path = model_dir.join("manifest.json");
        let json = serde_json::to_string_pretty(&manifest)?;
        fs::write(&manifest_path, json)?;

        Ok(model_dir)
    }

    /// Write a manifest for an "external" model — one pulled from an arbitrary
    /// URL or HuggingFace repo, not present in the EU catalog. Metadata is
    /// minimal (we don't know language/VRAM/license), but the entry is a
    /// first-class local model: it lists, runs, and `rm`s like any other.
    ///
    /// `id` is the filesystem-safe directory key; `gguf_file` is the GGUF
    /// filename inside that directory; `source` is the URL/repo it came from,
    /// recorded as the description for provenance.
    pub fn write_external_manifest(
        &self,
        id: &str,
        gguf_file: &str,
        source: &str,
        size_bytes: u64,
    ) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let model_dir = self.root.join(id);
        fs::create_dir_all(&model_dir)?;

        let manifest = ModelManifest {
            id: id.to_string(),
            name: id.to_string(),
            description: format!("External model pulled from {source}"),
            languages: Vec::new(),
            base: id.to_string(),
            vram_gb: 0,
            size_bytes,
            license: "unknown".into(),
            digest: String::new(),
            pulled_at: chrono::Utc::now().to_rfc3339(),
            status: "ready".into(),
            gguf_file: Some(gguf_file.to_string()),
            mmproj_file: None,
        };

        let manifest_path = model_dir.join("manifest.json");
        let json = serde_json::to_string_pretty(&manifest)?;
        fs::write(&manifest_path, json)?;

        Ok(model_dir)
    }

    /// Get the multimodal projector path for a locally available model, if any.
    /// Mirrors `gguf_path` but reads `manifest.mmproj_file` instead, and falls
    /// back to scanning the directory for any `mmproj*.gguf` file.
    pub fn mmproj_path(&self, name: &str) -> Option<PathBuf> {
        let short_name = name.strip_prefix("eullm/").unwrap_or(name);
        let model_dir = self.root.join(short_name);

        let manifest_path = model_dir.join("manifest.json");
        if manifest_path.exists()
            && let Ok(data) = fs::read_to_string(&manifest_path)
            && let Ok(manifest) = serde_json::from_str::<ModelManifest>(&data)
            && let Some(ref mmproj) = manifest.mmproj_file
        {
            let path = model_dir.join(mmproj);
            if path.exists() {
                return Some(path);
            }
        }

        if model_dir.is_dir() && let Ok(entries) = fs::read_dir(&model_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(name) = path.file_name().and_then(|n| n.to_str())
                    && name.starts_with("mmproj")
                    && path.extension().is_some_and(|e| e == "gguf")
                {
                    return Some(path);
                }
            }
        }
        None
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
                    let mut manifest: ModelManifest = serde_json::from_str(&data)?;
                    // Backfill the addressable id from the directory name for
                    // manifests written before the `id` field existed.
                    if manifest.id.is_empty()
                        && let Some(dir) = entry.file_name().to_str()
                    {
                        manifest.id = dir.to_string();
                    }
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
        let mut manifest: ModelManifest = serde_json::from_str(&data)?;
        // Backfill the addressable id from the directory key for older manifests.
        if manifest.id.is_empty() {
            manifest.id = short_name.to_string();
        }
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

    /// Remove a model's entire directory from disk.
    ///
    /// Returns `Ok(Some(bytes_freed))` if the directory existed and was
    /// removed, `Ok(None)` if there was nothing to remove. Errors only on
    /// real filesystem failures.
    pub fn delete(&self, name: &str) -> Result<Option<u64>, Box<dyn std::error::Error>> {
        let short_name = name.strip_prefix("eullm/").unwrap_or(name);
        let model_dir = self.root.join(short_name);
        if !model_dir.exists() {
            return Ok(None);
        }
        let size = dir_size(&model_dir).unwrap_or(0);
        fs::remove_dir_all(&model_dir)?;
        Ok(Some(size))
    }
}

/// Recursively sum the byte sizes of every regular file under `path`.
/// Used for reporting how much disk space a `rm` actually freed.
fn dir_size(path: &std::path::Path) -> Result<u64, Box<dyn std::error::Error>> {
    let mut total: u64 = 0;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let meta = entry.metadata()?;
        if meta.is_dir() {
            total += dir_size(&entry.path())?;
        } else {
            total += meta.len();
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_store() -> (ModelStore, PathBuf) {
        let root = std::env::temp_dir().join(format!("eullm-test-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        (ModelStore { root: root.clone() }, root)
    }

    /// A manifest written before the `id` field existed must still resolve to
    /// an addressable id, backfilled from the on-disk directory name — so the
    /// user never has to re-pull a model just to see it in `list`.
    #[test]
    fn legacy_manifest_without_id_backfills_from_dir() {
        let (store, root) = temp_store();
        let dir = root.join("gemma-4-12b");
        fs::create_dir_all(&dir).unwrap();
        // Old-format manifest JSON: no "id" key at all.
        let legacy = r#"{
            "name": "Gemma 4 12B Instruct (vision-capable)",
            "description": "x", "languages": [], "base": "gemma",
            "vram_gb": 10, "size_bytes": 7100000000, "license": "Apache-2.0",
            "digest": "", "pulled_at": "2026-01-01T00:00:00Z", "status": "ready"
        }"#;
        fs::write(dir.join("manifest.json"), legacy).unwrap();

        let listed = store.list().unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, "gemma-4-12b", "id backfilled from dir name");

        let got = store.get("gemma-4-12b").unwrap().unwrap();
        assert_eq!(got.id, "gemma-4-12b");

        fs::remove_dir_all(&root).ok();
    }

    /// A manifest written by the current code records the id explicitly, and it
    /// matches the directory key used to address the model.
    #[test]
    fn external_manifest_records_id() {
        let (store, root) = temp_store();
        store
            .write_external_manifest("my-model", "weights.gguf", "https://x/y.gguf", 123)
            .unwrap();
        let listed = store.list().unwrap();
        assert_eq!(listed[0].id, "my-model");
        fs::remove_dir_all(&root).ok();
    }
}

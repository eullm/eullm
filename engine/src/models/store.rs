//! Local model storage — manages downloaded models on disk.

use std::fs;
use std::path::{Path, PathBuf};

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

/// Whether `name` is safe to join onto a model directory as a single filename.
///
/// `Path::join` is not a string concatenation: a component containing `..`
/// climbs out of the base directory, and an absolute component *replaces* the
/// base entirely. So any externally-supplied filename has to be checked before
/// it becomes a path, and two of ours are externally supplied — the
/// `siblings[].rfilename` values from the HuggingFace model API (which were
/// only ever filtered by their `.gguf` suffix), and the `gguf_file` /
/// `mmproj_file` fields read back out of a `manifest.json` on disk.
///
/// Deliberately strict: one path component, no separators on either platform,
/// no `..`, not absolute, no NUL, no leading dot (which would make the file
/// hidden and is never a legitimate model filename). This mirrors what
/// `hub::is_valid_model_slug` does for Hub download slugs, loosened only where
/// real GGUF filenames need it — uppercase and spaces do occur in published
/// quant names.
pub fn is_safe_filename(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 255
        && !name.starts_with('.')
        && !name.contains("..")
        && !name.contains('/')
        && !name.contains('\\')
        && !name.contains('\0')
        // Reject a Windows drive-relative or UNC-ish prefix even when running
        // on Unix: the manifest may have been written on another platform.
        && !name.contains(':')
        && Path::new(name).components().count() == 1
}

/// Manages the local model directory (`~/.eullm/models/`).
pub struct ModelStore {
    root: PathBuf,
}

/// Write a file so a reader never sees it half-written.
///
/// `fs::write` truncates first and writes after, so a process that dies in
/// between leaves a valid path holding invalid content. For `manifest.json`
/// that turns into a parse error on the next `eullm list`, and the model looks
/// as though it were never pulled. Seen in the field.
///
/// Writing to a sibling temporary file and renaming makes the swap atomic on
/// every platform we ship: `std::fs::rename` replaces an existing destination
/// on Unix and on Windows alike. The reader therefore observes either the old
/// manifest or the new one, never a prefix of either.
///
/// The temporary file sits in the destination's own directory on purpose. A
/// rename across filesystems is not atomic and would fall back to a copy,
/// which is the failure mode this exists to avoid.
fn write_atomically(path: &Path, contents: &str) -> Result<(), Box<dyn std::error::Error>> {
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, contents)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

impl ModelStore {
    /// Create a store at the default location (`~/.eullm/models/`), or at
    /// `$EULLM_MODELS_DIR` when set.
    ///
    /// The directory is **not** created here. The store is only needed for
    /// catalog / pull / list operations; running a direct `.gguf` path (e.g.
    /// when eullm is embedded in another binary that manages its own model
    /// directory) must not require `~/.eullm/models` to exist — and must not
    /// fail at startup if that path happens to be a dangling symlink to an
    /// unmounted volume. Creation happens lazily in `write_manifest` /
    /// `write_external_manifest` when a model is actually stored.
    pub fn default_store() -> Result<Self, Box<dyn std::error::Error>> {
        let root = match std::env::var("EULLM_MODELS_DIR") {
            Ok(dir) if !dir.is_empty() => PathBuf::from(dir),
            _ => {
                let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE"))?;
                PathBuf::from(home).join(".eullm").join("models")
            }
        };
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
        create_model_dir(&model_dir)?;

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
        write_atomically(&manifest_path, &json)?;

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
        create_model_dir(&model_dir)?;

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
        write_atomically(&manifest_path, &json)?;

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
            && is_safe_filename(mmproj)
        {
            let path = model_dir.join(mmproj);
            if path.exists() {
                return Some(path);
            }
        }

        if model_dir.is_dir()
            && let Ok(entries) = fs::read_dir(&model_dir)
        {
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
            && is_safe_filename(gguf)
        {
            let path = model_dir.join(gguf);
            if path.exists() {
                return Some(path);
            }
        }

        // Fallback: look for any .gguf file in the model directory
        if model_dir.is_dir()
            && let Ok(entries) = fs::read_dir(&model_dir)
        {
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
                    // One damaged manifest must not hide every other model.
                    // This used to propagate with `?`, so a single truncated
                    // file made `eullm list` print a bare parser position and
                    // nothing else: no list, and no clue which directory was
                    // at fault. Reported from the field as
                    // "expected ',' or '}' at line 24 column 3".
                    //
                    // Note the two sibling readers below already tolerate this
                    // (`if let Ok(manifest) = ...`); the listing was the strict
                    // one, and it is the one people actually run.
                    let data = match fs::read_to_string(&manifest_path) {
                        Ok(d) => d,
                        Err(e) => {
                            tracing::warn!("skipping {}: {e}", manifest_path.display());
                            continue;
                        }
                    };
                    let mut manifest: ModelManifest = match serde_json::from_str(&data) {
                        Ok(m) => m,
                        Err(e) => {
                            tracing::warn!(
                                "skipping {}: {e}. Delete that directory and pull the model \
                                 again to repair it.",
                                manifest_path.display()
                            );
                            continue;
                        }
                    };
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

/// Create a model directory (and its parents, including the store root),
/// attaching a hint when the failure looks like a dangling symlink to an
/// unmounted volume — the one place the store actually has to create the tree.
fn create_model_dir(dir: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(dir).map_err(|e| {
        format!(
            "could not create model directory at {}: {e}. \
             If the store path is a symlink to an unmounted volume (e.g. a Windows/NAS mount), \
             mount it first, remove the dangling link, or set EULLM_MODELS_DIR to a writable path.",
            dir.display()
        )
        .into()
    })
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

#[cfg(test)]
mod safe_filename_tests {
    use super::is_safe_filename;

    #[test]
    fn accepts_real_gguf_filenames() {
        for name in [
            "qwen3-8b-Q4_K_M.gguf",
            "Qwen3-14B-Instruct-Q4_K_M.gguf",
            "mmproj-F16.gguf",
            "model.v2.gguf",
            "gemma 4 12b q4.gguf", // published quants do contain spaces
            "granite-3.3-8b-instruct-Q8_0.gguf",
        ] {
            assert!(is_safe_filename(name), "should accept {name:?}");
        }
    }

    #[test]
    fn rejects_anything_that_escapes_the_model_directory() {
        for name in [
            "../../../../etc/cron.d/payload.gguf",
            "../qwen3-8b/qwen3-8b.gguf",
            "/etc/shadow.gguf",
            "subdir/model.gguf",
            "subdir\\model.gguf",
            "C:\\Windows\\model.gguf",
            "..",
            ".",
            "",
            ".hidden.gguf",
            "model\0.gguf",
        ] {
            assert!(!is_safe_filename(name), "should reject {name:?}");
        }
    }

    #[test]
    fn rejects_an_absurdly_long_name() {
        assert!(!is_safe_filename(&format!("{}.gguf", "a".repeat(300))));
    }

    /// The property that matters: for every accepted name, joining it onto a
    /// base directory stays inside that directory.
    #[test]
    fn accepted_names_never_leave_the_base_directory() {
        let base = std::path::Path::new("/models/qwen3-8b");
        for name in ["a.gguf", "Q4_K_M.gguf", "model.v2.gguf", "with space.gguf"] {
            assert!(is_safe_filename(name));
            let joined = base.join(name);
            assert!(
                joined.starts_with(base),
                "{name:?} joined to {joined:?} left the base directory"
            );
            assert_eq!(joined.parent(), Some(base));
        }
    }
}

#[cfg(test)]
mod manifest_robustness_tests {
    use super::*;
    use uuid::Uuid;

    // Every non-defaulted field of ModelManifest. A fixture missing one is
    // rejected exactly like a truncated file, which would make the test below
    // pass for the wrong reason.
    const GOOD: &str = r#"{
        "id": "good", "name": "Good", "description": "d", "languages": ["en"],
        "base": "b", "vram_gb": 2, "size_bytes": 1, "license": "Apache-2.0",
        "digest": "sha256:0", "pulled_at": "2026-07-27T00:00:00Z",
        "status": "ready", "gguf_file": "g.gguf"
    }"#;
    // The shape reported from the field: a write that stopped partway.
    const TRUNCATED: &str = "{\n  \"id\": \"broken\",\n  \"name\": \"Bro";

    struct Scratch(PathBuf);
    impl Scratch {
        fn new() -> Self {
            let d = std::env::temp_dir().join(format!("eullm-store-{}", Uuid::new_v4()));
            fs::create_dir_all(&d).expect("mkdir");
            Self(d)
        }
        fn model(&self, name: &str, manifest: &str) {
            let d = self.0.join(name);
            fs::create_dir_all(&d).expect("mkdir");
            fs::write(d.join("manifest.json"), manifest).expect("write");
        }
        fn store(&self) -> ModelStore {
            ModelStore {
                root: self.0.clone(),
            }
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn one_damaged_manifest_does_not_hide_the_others() {
        let s = Scratch::new();
        s.model("good", GOOD);
        s.model("broken", TRUNCATED);
        let listed = s
            .store()
            .list()
            .expect("listing must survive a bad manifest");
        let ids: Vec<&str> = listed.iter().map(|m| m.id.as_str()).collect();
        assert!(ids.contains(&"good"), "the healthy model vanished: {ids:?}");
        assert!(
            !ids.contains(&"broken"),
            "the damaged one must be skipped: {ids:?}"
        );
    }

    #[test]
    fn a_directory_without_a_manifest_is_ignored() {
        let s = Scratch::new();
        s.model("good", GOOD);
        fs::create_dir_all(s.0.join("empty")).expect("mkdir");
        assert_eq!(s.store().list().expect("list").len(), 1);
    }

    // The other half of the defect. If the write is not atomic, an interrupted
    // one produces exactly the truncated file the test above has to tolerate.
    #[test]
    fn an_atomic_write_leaves_no_temporary_behind() {
        let s = Scratch::new();
        let target = s.0.join("manifest.json");
        write_atomically(&target, GOOD).expect("write");
        assert_eq!(fs::read_to_string(&target).expect("read"), GOOD);
        let leftovers: Vec<String> = fs::read_dir(&s.0)
            .expect("readdir")
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "temporary left behind: {leftovers:?}");
    }

    #[test]
    fn an_atomic_write_replaces_existing_content_whole() {
        let s = Scratch::new();
        let target = s.0.join("manifest.json");
        write_atomically(&target, "{\"id\":\"old\"}").expect("first");
        write_atomically(&target, GOOD).expect("second");
        assert_eq!(fs::read_to_string(&target).expect("read"), GOOD);
    }
}

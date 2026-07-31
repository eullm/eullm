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

/// Find a multimodal projector sitting next to a GGUF file.
///
/// `ModelStore::mmproj_path` only looks inside the store, keyed by model id, so
/// a model run straight from a path — `eullm run ./gemma-4-12b-Q4.gguf`, or
/// anything pulled from a URL — could never be multimodal however many
/// `mmproj-F16.gguf` files sat in the same folder. That is also the layout
/// every HuggingFace vision repo ships, and what llama.cpp users expect to
/// work.
pub fn mmproj_beside(gguf: &Path) -> Option<PathBuf> {
    let dir = gguf.parent()?;
    let entries = fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.file_name()?.to_str()?.to_ascii_lowercase();
        if name.starts_with("mmproj") && name.ends_with(".gguf") {
            return Some(path);
        }
    }
    None
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

    /// Create a store rooted at an explicit directory.
    ///
    /// `#[cfg(test)]` because that is the only caller: production always goes
    /// through `default_store()`. Tests must not, because it reads
    /// `EULLM_MODELS_DIR` and `HOME`, and setting either from a test races
    /// every other test in the binary — the same rule `CLAUDE.md` states for
    /// perimeter settings, for the same reason.
    #[cfg(test)]
    pub fn at(root: PathBuf) -> Self {
        Self { root }
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
        mmproj_file: Option<&str>,
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
            mmproj_file: mmproj_file.map(String::from),
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

    /// The directory this store reads and writes, and where that came from.
    ///
    /// Printed by `eullm list` and at server startup because the alternative
    /// is what happened in practice: `list` showed a model as installed while
    /// the API answered 404 for the same name, and both were right, because
    /// the two processes had different roots. Nothing anywhere said which
    /// directory was in use. The audit trail already reports its own path at
    /// startup; the model store not doing so was an inconsistency, not a
    /// decision.
    pub fn root_with_source(&self) -> (&Path, &'static str) {
        let source = if std::env::var_os("EULLM_MODELS_DIR").is_some() {
            "EULLM_MODELS_DIR"
        } else {
            "default"
        };
        (&self.root, source)
    }

    /// Whether the GGUF this manifest names is actually on disk.
    ///
    /// `status` is a string copied into the manifest when the model was
    /// pulled, so it keeps saying `ready` after the file is gone or was never
    /// fully written. A listing that repeats it without looking is how a user
    /// ends up trusting a model that cannot load.
    pub fn is_present(&self, id: &str) -> bool {
        self.gguf_path(id).is_some()
    }

    /// Directories that hold weights but produced no entry in `list()`.
    ///
    /// `list()` counts a directory only when it contains a `manifest.json`
    /// that parses. Anything else is skipped, and until this existed it was
    /// skipped in silence: a model with its weights on disk simply stopped
    /// appearing, with nothing on screen connecting the absence to a cause.
    /// A missing manifest is not exotic — an interrupted pull, a restored
    /// backup, a directory copied by hand from another machine.
    ///
    /// Only directories that actually contain a `.gguf` are reported, so an
    /// unrelated folder in the store root does not become a warning.
    ///
    /// Returns `(directory name, why it was skipped)`.
    pub fn unlisted(&self) -> Vec<(String, String)> {
        let mut out = Vec::new();
        let Ok(entries) = fs::read_dir(&self.root) else {
            return out;
        };
        for entry in entries.flatten() {
            if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                continue;
            }
            let dir = entry.path();
            let holds_weights = fs::read_dir(&dir).is_ok_and(|mut it| {
                it.any(|e| e.is_ok_and(|e| e.path().extension().is_some_and(|ext| ext == "gguf")))
            });
            if !holds_weights {
                continue;
            }
            let manifest = dir.join("manifest.json");
            let reason = if !manifest.exists() {
                "no manifest.json".to_string()
            } else {
                match fs::read_to_string(&manifest) {
                    Err(e) => format!("manifest.json cannot be read: {e}"),
                    Ok(data) => match serde_json::from_str::<ModelManifest>(&data) {
                        // Listed normally — nothing to report.
                        Ok(_) => continue,
                        Err(e) => format!("manifest.json is malformed: {e}"),
                    },
                }
            };
            out.push((entry.file_name().to_string_lossy().into_owned(), reason));
        }
        out.sort();
        out
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
                    // The directory name *is* the addressable id, and it
                    // overrides whatever the manifest claims.
                    //
                    // `gguf_path`, `exists`, `rm` and `run` all resolve
                    // `root/<id>`, so the only string that can address a model
                    // is its directory name. The manifest's own `id` was
                    // trusted here, which made the NAME column a label rather
                    // than a handle: two directories carrying a copied
                    // manifest both printed the same id, and only one of them
                    // could be reached by it. Reported from a real store with
                    // `gemma-4-e4b` listed twice.
                    //
                    // This also covers manifests written before the field
                    // existed, which is why the backfill was here originally.
                    if let Some(dir) = entry.file_name().to_str() {
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

    /// Create this model's directory, or explain why it cannot be created.
    ///
    /// A pull creates the directory implicitly, on its way to writing the
    /// first byte of a multi-gigabyte download. When that fails the failure
    /// arrives after the HTTP request has already started and is reported
    /// alongside a suggestion that the model may not be published yet, which
    /// sends the user to look at the wrong machine. Called up front, a local
    /// problem is a local message and nothing is downloaded.
    pub fn ensure_model_dir(&self, id: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let dir = self.model_path(id);
        create_model_dir(&dir)?;
        Ok(dir)
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
///
/// `create_dir_all` reports `AlreadyExists` for two situations that read
/// identically as `File exists (os error 17)` and are the two most likely
/// reasons this fails on a real machine: the path is occupied by a regular
/// file, or it is a symlink whose target is gone. Neither is a plausible
/// guess from that message alone, so both are named here.
fn create_model_dir(dir: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(dir).map_err(|e| {
        let occupant = match fs::symlink_metadata(dir) {
            Ok(md) if md.file_type().is_symlink() => {
                " That path is a symlink; if its target no longer exists, remove the link."
            }
            Ok(md) if md.is_file() => {
                " That path is a regular file, not a directory. Remove or rename it."
            }
            _ => "",
        };
        format!(
            "could not create model directory at {}: {e}.{occupant} \
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
mod addressable_id_tests {
    use super::*;

    /// Two directories carrying the same manifest `id`, which is what a
    /// manifest copied between models produces.
    #[test]
    fn the_listed_name_is_always_the_directory_that_can_be_run() {
        let root = std::env::temp_dir().join(format!("eullm-ids-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        for dir in ["gemma-4-e4b", "gemma-4-e4b-gguf"] {
            let d = root.join(dir);
            fs::create_dir_all(&d).expect("model dir");
            fs::write(d.join("model.gguf"), b"weights").expect("weights");
            // Both manifests claim the same id, as a copied one would.
            let manifest = serde_json::json!({
                "id": "gemma-4-e4b", "name": "Gemma 4 E4B", "description": "",
                "languages": [], "base": "", "vram_gb": 8, "size_bytes": 7,
                "license": "Apache-2.0", "digest": "", "pulled_at": "",
                "status": "ready", "gguf_file": "model.gguf",
            });
            fs::write(d.join("manifest.json"), manifest.to_string()).expect("manifest");
        }
        let store = ModelStore::at(root.clone());

        let mut ids: Vec<String> = store
            .list()
            .expect("list")
            .into_iter()
            .map(|m| m.id)
            .collect();
        ids.sort();
        assert_eq!(ids, vec!["gemma-4-e4b", "gemma-4-e4b-gguf"]);

        // Every name the listing shows must resolve to weights on disk,
        // which is the property that makes it a handle rather than a label.
        for id in &ids {
            assert!(store.is_present(id), "{id} is listed but cannot be run");
        }
        let _ = fs::remove_dir_all(&root);
    }
}

#[cfg(test)]
mod unlisted_tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("eullm-unlisted-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).expect("tmp dir");
        d
    }

    fn weights(dir: &Path, name: &str) -> PathBuf {
        let d = dir.join(name);
        fs::create_dir_all(&d).expect("model dir");
        fs::write(d.join("model.gguf"), b"weights").expect("weights");
        d
    }

    // The case that prompted this: weights present, manifest gone. Before,
    // `list()` skipped the directory in silence and the model simply was not
    // there any more, with nothing saying why.
    #[test]
    fn a_model_without_a_manifest_is_reported_rather_than_vanishing() {
        let root = tmp("no-manifest");
        weights(&root, "orphaned-model");
        let store = ModelStore::at(root.clone());

        assert!(store.list().expect("list").is_empty());
        let unlisted = store.unlisted();
        assert_eq!(unlisted.len(), 1);
        assert_eq!(unlisted[0].0, "orphaned-model");
        assert!(
            unlisted[0].1.contains("no manifest.json"),
            "{:?}",
            unlisted[0]
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_malformed_manifest_says_so_instead_of_being_skipped_quietly() {
        let root = tmp("malformed");
        let d = weights(&root, "half-written");
        fs::write(d.join("manifest.json"), "{\"id\": \"half-written\",").expect("truncated");
        let store = ModelStore::at(root.clone());

        assert!(store.list().expect("list").is_empty());
        let unlisted = store.unlisted();
        assert_eq!(unlisted.len(), 1);
        assert!(unlisted[0].1.contains("malformed"), "{:?}", unlisted[0]);
        let _ = fs::remove_dir_all(&root);
    }

    // A model that lists correctly must not also be reported as broken, and a
    // directory with no weights in it is not a model at all: neither belongs
    // in a warning the user is meant to act on.
    #[test]
    fn nothing_is_reported_when_there_is_nothing_wrong() {
        let root = tmp("healthy");
        let d = weights(&root, "good-model");
        let manifest = serde_json::json!({
            "id": "good-model", "name": "good-model", "description": "",
            "languages": [], "base": "", "vram_gb": 1, "size_bytes": 7,
            "license": "Apache-2.0", "digest": "", "pulled_at": "", "status": "ready",
            "gguf_file": "model.gguf",
        });
        fs::write(d.join("manifest.json"), manifest.to_string()).expect("manifest");
        fs::create_dir_all(root.join("not-a-model")).expect("stray dir");
        let store = ModelStore::at(root.clone());

        assert_eq!(store.list().expect("list").len(), 1);
        assert!(store.unlisted().is_empty());
        let _ = fs::remove_dir_all(&root);
    }
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
            .write_external_manifest("my-model", "weights.gguf", "https://x/y.gguf", 123, None)
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

#[cfg(test)]
mod store_lookup_tests {
    use super::*;
    use uuid::Uuid;

    // A catalog model as it lands on disk: the directory key and the
    // human-readable title are different strings.
    const MANIFEST: &str = r#"{
        "id": "gemma-4-e4b", "name": "Gemma 4 E4B Instruct (vision-capable)",
        "description": "d", "languages": ["en"], "base": "b", "vram_gb": 8,
        "size_bytes": 1, "license": "Gemma Terms of Use", "digest": "sha256:0",
        "pulled_at": "2026-07-27T00:00:00Z", "status": "ready",
        "gguf_file": "g.gguf"
    }"#;

    struct Scratch(PathBuf);
    impl Scratch {
        fn new() -> Self {
            let d = std::env::temp_dir().join(format!("eullm-lookup-{}", Uuid::new_v4()));
            fs::create_dir_all(d.join("gemma-4-e4b")).expect("mkdir");
            fs::write(d.join("gemma-4-e4b").join("manifest.json"), MANIFEST).expect("manifest");
            Self(d)
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

    // The contract the picker was breaking: a model is addressed by `id`, the
    // directory key, never by `name`. Looking up by name asks for a directory
    // called "Gemma 4 E4B Instruct (vision-capable)" and finds nothing, so
    // every catalog model on disk vanished from the picker's LOCAL section and
    // only ones whose title happened to equal their id survived.
    #[test]
    fn a_model_resolves_by_id_and_not_by_display_name() {
        let s = Scratch::new();
        fs::write(s.0.join("gemma-4-e4b").join("g.gguf"), b"x").expect("gguf");
        let store = s.store();
        assert!(
            store.gguf_path("gemma-4-e4b").is_some(),
            "the directory key must resolve"
        );
        assert!(
            store
                .gguf_path("Gemma 4 E4B Instruct (vision-capable)")
                .is_none(),
            "the display name must not be used as a path"
        );
    }

    // What the user actually hit: a manifest present, the GGUF absent. The
    // listing used to repeat `status` from the manifest and call it ready.
    #[test]
    fn a_manifest_without_its_gguf_is_not_present() {
        let s = Scratch::new();
        let store = s.store();
        assert!(!store.is_present("gemma-4-e4b"));
        fs::write(s.0.join("gemma-4-e4b").join("g.gguf"), b"x").expect("gguf");
        assert!(store.is_present("gemma-4-e4b"));
    }

    // A pull reported `File exists (os error 17)` and then suggested the model
    // might not be published yet, which is the wrong machine entirely. The
    // message has to name the path and say what is sitting on it.
    #[test]
    fn a_file_where_the_model_directory_goes_is_explained() {
        let s = Scratch::new();
        fs::write(s.0.join("blocked"), b"not a directory").expect("write");
        let err = s
            .store()
            .ensure_model_dir("blocked")
            .expect_err("a regular file must not pass for a model directory")
            .to_string();
        assert!(err.contains("blocked"), "the path must be named: {err}");
        assert!(
            err.contains("regular file"),
            "the occupant must be named: {err}"
        );
    }

    #[test]
    fn an_existing_model_directory_is_accepted() {
        let s = Scratch::new();
        let dir = s.store().ensure_model_dir("gemma-4-e4b").expect("existing");
        assert_eq!(dir, s.0.join("gemma-4-e4b"));
    }

    #[test]
    fn the_root_reports_where_it_came_from() {
        let s = Scratch::new();
        let store = s.store();
        let (root, source) = store.root_with_source();
        assert_eq!(root, s.0.as_path());
        assert!(matches!(source, "EULLM_MODELS_DIR" | "default"));
    }
}

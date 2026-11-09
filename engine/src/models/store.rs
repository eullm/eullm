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
    /// The HuggingFace repository the weights were downloaded from. With
    /// `hf_filename`, the same two fields a catalog entry names its download
    /// by, and what lets a later pull of the same file under another id find
    /// it here instead of downloading it a second time
    /// (`models::pull::reuse_stored_weights`).
    ///
    /// Absent for weights that did not come from HuggingFace (a plain URL,
    /// or an Ollama import by a release that still had `eullm import-ollama`)
    /// and in manifests written before the field existed.
    /// Those older manifests are still recognised, from what they did record:
    /// see `ModelManifest::holds`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hf_repo: Option<String>,
    /// Path of the GGUF inside `hf_repo`, as the Hub lists it; for a split
    /// model, its first shard. Not the same as `gguf_file`, which is only the
    /// name the file is stored under: a repo that keeps each quantization in
    /// its own directory (`UD-Q4_K_XL/…-00001-of-00004.gguf`) loses that
    /// directory on the way to disk, and it is part of which file this is.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hf_filename: Option<String>,
}

/// A file in a HuggingFace repository: what a pull downloads, and what two
/// pulls have to agree on to be the same download.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfFile {
    /// `owner/repo`.
    pub repo: String,
    /// Path inside the repository, as the Hub lists it.
    pub path: String,
}

impl HfFile {
    pub fn new(repo: &str, path: &str) -> Self {
        Self {
            repo: repo.to_string(),
            path: path.to_string(),
        }
    }

    /// The name the file is stored under locally: the path's last component.
    pub fn file_name(&self) -> &str {
        self.path.rsplit('/').next().unwrap_or(&self.path)
    }
}

/// Whether two repository ids name the same repository. Without regard to
/// case, because the Hub resolves them that way: `unsloth/qwen3-8b-gguf` is
/// `unsloth/Qwen3-8B-GGUF`, and no two repositories may differ only in case.
/// Paths inside a repository are git paths and stay case-sensitive.
fn same_repo(a: &str, b: &str) -> bool {
    a.eq_ignore_ascii_case(b)
}

/// The description `write_external_manifest` writes, up to the source.
/// Shared with the reader in `ModelManifest::recorded_hf_file`, which gets
/// the repository of an `hf.co` pull back out of it for manifests written
/// before `hf_repo` existed, so the two cannot drift apart.
const EXTERNAL_SOURCE_PREFIX: &str = "External model pulled from ";

/// The description `eullm import` wrote in 0.7.95: provenance, not a
/// description of the model, so `written_description` leaves it out like the
/// one above. Such manifests are still on disk.
const IMPORTED_DESCRIPTION: &str = "Imported from files already in the model store";

/// The start of the description of a library model's made-up manifest, the
/// file's path after it.
const LIBRARY_PREFIX: &str = "In a model library: ";

/// The manifest a library model is listed with: made up from the file, never
/// written.
fn library_manifest(m: &super::library::LibraryModel) -> ModelManifest {
    let file_name = |p: &Path| p.file_name().and_then(|n| n.to_str()).map(str::to_string);
    ModelManifest {
        id: m.id.clone(),
        name: m.id.clone(),
        description: format!("{LIBRARY_PREFIX}{}", m.path.display()),
        languages: Vec::new(),
        base: m.id.clone(),
        vram_gb: 0,
        size_bytes: m.size_bytes,
        license: "unknown".into(),
        digest: String::new(),
        pulled_at: String::new(),
        status: "library".into(),
        gguf_file: file_name(&m.path),
        mmproj_file: m.mmproj.as_deref().and_then(file_name),
        hf_repo: None,
        hf_filename: None,
    }
}

/// Which HuggingFace file a manifest's weights are, as far as it says.
struct RecordedHfFile {
    repo: String,
    /// The path inside the repository, or only the file's name when
    /// `name_only`: all an `hf.co` pull recorded before `hf_filename`
    /// existed was the name it stored the file under.
    path: String,
    name_only: bool,
}

impl RecordedHfFile {
    fn is(&self, wanted: &HfFile) -> bool {
        same_repo(&self.repo, &wanted.repo)
            && if self.name_only {
                self.path == wanted.file_name()
            } else {
                self.path == wanted.path
            }
    }
}

impl ModelManifest {
    /// What the description says the model is, for a reader that shows it as
    /// such: `None` when it is empty, or when it only records where an
    /// external pull came from (see `write_external_manifest`).
    pub fn written_description(&self) -> Option<&str> {
        let description = self.description.trim();
        (!description.is_empty()
            && !description.starts_with(EXTERNAL_SOURCE_PREFIX)
            && description != IMPORTED_DESCRIPTION
            && !description.starts_with(LIBRARY_PREFIX))
            .then_some(description)
    }

    /// Whether these weights are the HuggingFace file `wanted` — asked of
    /// every stored model before a pull downloads anything.
    pub fn holds(&self, wanted: &HfFile) -> bool {
        self.recorded_hf_file()
            .is_some_and(|recorded| recorded.is(wanted))
    }

    /// Whether this model's projector is the HuggingFace file `wanted`.
    ///
    /// A projector is taken to live in the repository its weights came from,
    /// because both ways of pulling one put it there: an `hf.co` pull fetches
    /// it from the repository it resolved the weights in, and every catalog
    /// entry that declares a projector declares it beside its weights
    /// (`catalog::tests::every_projector_lives_beside_its_weights` holds the
    /// catalog to that). Only the name is compared, since the name is all a
    /// manifest keeps of it.
    pub fn projector_is(&self, wanted: &HfFile) -> bool {
        self.mmproj_file.as_deref() == Some(wanted.file_name())
            && self
                .recorded_hf_file()
                .is_some_and(|recorded| same_repo(&recorded.repo, &wanted.repo))
    }

    /// Which HuggingFace file the weights are. Recorded in `hf_repo` and
    /// `hf_filename` by every pull since those fields exist; for a manifest
    /// written before, worked out from what it did record, and only where
    /// that is certain enough to share a file on the strength of it:
    ///
    /// - a catalog pull recorded its catalog id and the catalog's SHA-256 of
    ///   the file, which the download was verified against. When the catalog
    ///   entry still carries that digest, it names the very bytes on disk.
    ///   A catalog that has moved on since names a different file, and the
    ///   manifest is then not taken for it;
    /// - an `hf.co` pull recorded its ref in the description and the file's
    ///   name in `gguf_file`, but not the directory the repository kept it
    ///   in. Its name is therefore compared on its own.
    fn recorded_hf_file(&self) -> Option<RecordedHfFile> {
        if let (Some(repo), Some(path)) = (&self.hf_repo, &self.hf_filename) {
            return Some(RecordedHfFile {
                repo: repo.clone(),
                path: path.clone(),
                name_only: false,
            });
        }
        let gguf_file = self.gguf_file.as_deref()?;
        if let Some(entry) = super::catalog::EU_CATALOG.iter().find(|e| e.id == self.id)
            && !entry.hf_repo.is_empty()
            && !entry.digest.is_empty()
            && entry.digest == self.digest
            && entry.hf_filename == gguf_file
        {
            return Some(RecordedHfFile {
                repo: entry.hf_repo.clone(),
                path: entry.hf_filename.clone(),
                name_only: false,
            });
        }
        let source = self.description.strip_prefix(EXTERNAL_SOURCE_PREFIX)?;
        let hf = crate::registry::parse_hf_ref(source)?;
        Some(RecordedHfFile {
            repo: hf.repo,
            path: gguf_file.to_string(),
            name_only: true,
        })
    }
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

/// Whether `path` is safe as a *remote* HuggingFace reference: one or more
/// components, each one safe by `is_safe_filename`.
///
/// Repos that publish many quantizations give each its own subdirectory —
/// `UD-Q4_K_XL/Model-UD-Q4_K_XL-00001-of-00004.gguf` — so `siblings[].rfilename`
/// is a path, not a bare name. Judging those with `is_safe_filename` dropped
/// every weight file in such a repo and left only the projector sitting at the
/// repo root, which surfaced as the nonsensical "this HuggingFace repo contains
/// only a projector (mmproj), no model weights". Found live pulling
/// `unsloth/Qwen3.8-Flash-Next-GGUF`, whose weights are *all* in subdirectories.
///
/// The traversal guarantee is unchanged, because every component still goes
/// through the same rules: `..`, absolute prefixes, backslashes, drive letters
/// and NUL are all rejected per component, and an absolute path fails outright
/// (its leading `/` splits to an empty first component). Callers still reduce
/// this to a basename before touching the filesystem — this validates the value
/// on the way *out* to the network, which is a use of its own.
pub fn is_safe_relative_path(path: &str) -> bool {
    !path.is_empty() && path.len() <= 4096 && path.split('/').all(is_safe_filename)
}

/// Manages the local model directory (`~/.eullm/models/`).
pub struct ModelStore {
    root: PathBuf,
    /// Where `root` came from, for `root_with_source`.
    source: String,
    /// The folders of `EULLM_MODEL_LIBRARIES`: read in place, never written to
    /// (`models::library`).
    libraries: Vec<PathBuf>,
    /// Where `libraries` came from, for `libraries_with_source`.
    libraries_source: String,
    /// The last scan of the libraries and when it was made: `/api/tags` and
    /// `/v1/models` are asked for often, and a scan walks every folder.
    scan_cache: std::sync::Mutex<Option<(std::time::Instant, std::sync::Arc<super::library::Scan>)>>,
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

/// What `--mtp` makes of a model's projector, and the line that says so.
///
/// Only the scheduler's single slot drafts, and a model with a projector is
/// loaded on the sequential engine, since the scheduler does not route
/// images. So `--mtp` on such a model did nothing and said nothing, the
/// banner reading "MTP drafts" all the same: unsloth's Qwen3.5 and Qwen3.6
/// MTP repositories ship `mmproj-*.gguf`, a pull fetches it with the
/// weights, and LUMI-G measured no draft on them in any round (24-10-2026).
///
/// `drafts` is whether this load would draft without the projector (`--mtp`
/// on one slot). A projector found for the model, in the store or beside its
/// weights, then gives way to the drafts asked for; one `named` with
/// `--mmproj` stays, and the line says `--mtp` drafts nothing for the model.
pub fn projector_with_mtp(
    projector: Option<PathBuf>,
    named: bool,
    drafts: bool,
    model: &str,
) -> (Option<PathBuf>, Option<String>) {
    match projector {
        Some(p) if drafts && !named => {
            let line = format!(
                "--mtp: {model} loads without its projector ({}) so that the MTP head can draft; \
                 it takes no images this way (without --mtp it loads with them)",
                p.display()
            );
            (None, Some(line))
        }
        Some(p) if drafts => (
            Some(p),
            Some(format!(
                "--mtp: {model} loads with --mmproj on the sequential engine, which does not \
                 draft: --mtp does nothing for it"
            )),
        ),
        other => (other, None),
    }
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
    /// `EULLM_MODELS_DIR` when set: in the process environment first, in a
    /// `.env` file in the working directory second, like every other
    /// `EULLM_*` setting.
    ///
    /// The `.env` used to be ignored here, while the quick-start guide and
    /// `engine/CLAUDE.md` listed `EULLM_MODELS_DIR` among the settings read
    /// from it: a line `EULLM_MODELS_DIR=/data/models` there had no effect,
    /// and nothing said so.
    ///
    /// The directory is **not** created here. The store is only needed for
    /// catalog / pull / list operations; running a direct `.gguf` path (e.g.
    /// when eullm is embedded in another binary that manages its own model
    /// directory) must not require `~/.eullm/models` to exist — and must not
    /// fail at startup if that path happens to be a dangling symlink to an
    /// unmounted volume. Creation happens lazily in `write_manifest` /
    /// `write_external_manifest` when a model is actually stored.
    pub fn default_store() -> Result<Self, Box<dyn std::error::Error>> {
        let env_dir = std::env::var("EULLM_MODELS_DIR").ok();
        let env_file = Path::new(".env");
        let file_contents = fs::read_to_string(env_file).ok();
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .ok();
        let (root, source) = Self::resolve_root(
            env_dir.as_deref(),
            file_contents.as_deref(),
            &env_file.display().to_string(),
            home.as_deref(),
        )?;
        let env_libraries = std::env::var("EULLM_MODEL_LIBRARIES").ok();
        let (libraries, libraries_source) = Self::resolve_libraries(
            env_libraries.as_deref(),
            file_contents.as_deref(),
            &env_file.display().to_string(),
        );
        Ok(Self {
            root,
            source,
            libraries,
            libraries_source,
            scan_cache: std::sync::Mutex::new(None),
        })
    }

    /// `EULLM_MODEL_LIBRARIES`: the environment, then the `.env` file; none
    /// when neither sets it. Returns the folders and where they came from.
    fn resolve_libraries(
        env: Option<&str>,
        file_contents: Option<&str>,
        file_label: &str,
    ) -> (Vec<PathBuf>, String) {
        if let Some(spec) = env.map(str::trim).filter(|s| !s.is_empty()) {
            return (super::library::parse_libraries(spec), "EULLM_MODEL_LIBRARIES".to_string());
        }
        if let Some(spec) = file_contents
            .and_then(|contents| crate::api::ip_allowlist::env_file_var(contents, "EULLM_MODEL_LIBRARIES"))
        {
            return (
                super::library::parse_libraries(&spec),
                format!("EULLM_MODEL_LIBRARIES ({file_label})"),
            );
        }
        (Vec::new(), "none".to_string())
    }

    /// Pure resolution step behind `default_store`, split out so the
    /// precedence is testable without mutating process environment variables
    /// (which race against every other test in the binary): the environment,
    /// then the `.env` file, then `~/.eullm/models`. Blank values count as
    /// unset. Returns the directory and where it came from.
    fn resolve_root(
        env_dir: Option<&str>,
        file_contents: Option<&str>,
        file_label: &str,
        home: Option<&str>,
    ) -> Result<(PathBuf, String), Box<dyn std::error::Error>> {
        if let Some(dir) = env_dir.map(str::trim).filter(|d| !d.is_empty()) {
            return Ok((PathBuf::from(dir), "EULLM_MODELS_DIR".to_string()));
        }
        if let Some(dir) = file_contents.and_then(|contents| {
            crate::api::ip_allowlist::env_file_var(contents, "EULLM_MODELS_DIR")
        }) {
            return Ok((PathBuf::from(dir), format!("EULLM_MODELS_DIR ({file_label})")));
        }
        let home = home.ok_or("neither HOME nor USERPROFILE is set, and EULLM_MODELS_DIR is not")?;
        Ok((
            PathBuf::from(home).join(".eullm").join("models"),
            "default".to_string(),
        ))
    }

    /// Create a store rooted at an explicit directory.
    ///
    /// `#[cfg(test)]` because that is the only caller: production always goes
    /// through `default_store()`. Tests must not, because it reads
    /// `EULLM_MODELS_DIR`, `HOME` and `.env`, and setting either variable from
    /// a test races every other test in the binary — the same rule `CLAUDE.md`
    /// states for perimeter settings, for the same reason.
    #[cfg(test)]
    pub fn at(root: PathBuf) -> Self {
        Self {
            root,
            source: "test".to_string(),
            libraries: Vec::new(),
            libraries_source: "none".to_string(),
            scan_cache: std::sync::Mutex::new(None),
        }
    }

    /// The same store, reading the model libraries `libraries` too. Tests only,
    /// like `at`.
    #[cfg(test)]
    pub fn with_libraries(mut self, libraries: Vec<PathBuf>) -> Self {
        self.libraries = libraries;
        self
    }

    /// Write a manifest to disk for a pulled model.
    ///
    /// The on-disk directory is keyed by the catalog `id` (filesystem-safe,
    /// stable across catalog revisions). The manifest still records the
    /// human `name` for display.
    ///
    /// `hf_file` is the file the weights are, when the caller knows: the one
    /// it just downloaded or linked. It is not derived from `entry`, because
    /// a manifest rewritten for a model pulled under an older catalog would
    /// then claim the current catalog's file for weights that may predate it.
    pub fn write_manifest(
        &self,
        entry: &CatalogEntry,
        status: &str,
        gguf_file: Option<&str>,
        mmproj_file: Option<&str>,
        hf_file: Option<&HfFile>,
    ) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let model_dir = self.root.join(&entry.id);
        create_model_dir(&model_dir).map_err(|e| -> Box<dyn std::error::Error> { e.to_string().into() })?;

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
            hf_repo: hf_file.map(|f| f.repo.clone()),
            hf_filename: hf_file.map(|f| f.path.clone()),
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
    /// recorded as the description for provenance; `hf_file` is the file the
    /// weights are on HuggingFace, when they came from there.
    pub fn write_external_manifest(
        &self,
        id: &str,
        gguf_file: &str,
        source: &str,
        size_bytes: u64,
        mmproj_file: Option<&str>,
        hf_file: Option<&HfFile>,
    ) -> Result<PathBuf, Box<dyn std::error::Error>> {
        self.write_minimal_manifest(
            id,
            gguf_file,
            format!("{EXTERNAL_SOURCE_PREFIX}{source}"),
            size_bytes,
            mmproj_file,
            hf_file,
        )
    }

    /// The manifest of a model nothing is known about but its files: the
    /// shape `write_external_manifest` and `import` share.
    fn write_minimal_manifest(
        &self,
        id: &str,
        gguf_file: &str,
        description: String,
        size_bytes: u64,
        mmproj_file: Option<&str>,
        hf_file: Option<&HfFile>,
    ) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let model_dir = self.root.join(id);
        create_model_dir(&model_dir).map_err(|e| -> Box<dyn std::error::Error> { e.to_string().into() })?;

        let manifest = ModelManifest {
            id: id.to_string(),
            name: id.to_string(),
            description,
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
            hf_repo: hf_file.map(|f| f.repo.clone()),
            hf_filename: hf_file.map(|f| f.path.clone()),
        };

        let manifest_path = model_dir.join("manifest.json");
        let json = serde_json::to_string_pretty(&manifest)?;
        write_atomically(&manifest_path, &json)?;

        Ok(model_dir)
    }

    /// Get the multimodal projector path for a locally available model, if any.
    /// Resolve a model name to its directory under the store root, refusing
    /// any name that would escape it.
    ///
    /// The `model` field of an API request reaches these lookups directly:
    /// only the launch-model shortcut, the opt-in `EULLM_ALLOW_MODEL_PATHS`
    /// path forms, and the `/models` mount join in `api::resolve_model` are
    /// gated there, and step 5 handed the raw name straight to the store. A
    /// name like `../../../../etc` would then resolve to `root.join("../…")`
    /// and load — or, via `delete`, remove — a directory outside the store,
    /// and double as an existence oracle for `.gguf`/`manifest.json` anywhere
    /// the process can read. `is_safe_filename` (the same check the `/models`
    /// join and the manifest's own `gguf_file` already use) rejects `..`,
    /// separators, and absolute/drive-relative forms, so the join can only
    /// ever stay one component under the root. `eullm/` is stripped first
    /// because that is a legitimate id prefix, not a path segment.
    fn safe_model_dir(&self, name: &str) -> Option<PathBuf> {
        let short_name = name.strip_prefix("eullm/").unwrap_or(name);
        is_safe_filename(short_name).then(|| self.root.join(short_name))
    }

    /// Mirrors `gguf_path` but reads `manifest.mmproj_file` instead, and falls
    /// back to scanning the directory for any `mmproj*.gguf` file.
    pub fn mmproj_path(&self, name: &str) -> Option<PathBuf> {
        let model_dir = self.safe_model_dir(name)?;

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
                    // Case-insensitive to match `mmproj_beside`, the picker and
                    // `registry::is_mmproj`: an `MMProj-F16.gguf` those three
                    // find must not be invisible here.
                    && name.to_ascii_lowercase().starts_with("mmproj")
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
        let model_dir = self.safe_model_dir(name)?;

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
    pub fn root_with_source(&self) -> (&Path, &str) {
        (&self.root, &self.source)
    }

    /// Whether the GGUF this manifest names is actually on disk.
    ///
    /// `status` is a string copied into the manifest when the model was
    /// pulled, so it keeps saying `ready` after the file is gone or was never
    /// fully written. A listing that repeats it without looking is how a user
    /// ends up trusting a model that cannot load.
    pub fn is_present(&self, id: &str) -> bool {
        self.gguf_path(id).is_some() || self.library_model(id).is_some()
    }

    /// The folders of `EULLM_MODEL_LIBRARIES` and where they were set.
    pub fn libraries_with_source(&self) -> (&[PathBuf], &str) {
        (&self.libraries, &self.libraries_source)
    }

    /// The models found without a manifest: in the folders of
    /// `EULLM_MODEL_LIBRARIES`, and in the store's own folder outside its model
    /// directories (files copied in by hand, a model directory whose manifest
    /// is missing). Read in place: nothing is moved or written. A scan is
    /// reused for a few seconds, the time of a client's burst of requests.
    pub fn library(&self) -> std::sync::Arc<super::library::Scan> {
        const FRESH: std::time::Duration = std::time::Duration::from_secs(5);
        let mut cache = self.scan_cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((at, scan)) = cache.as_ref()
            && at.elapsed() < FRESH
        {
            return scan.clone();
        }
        let scan = std::sync::Arc::new(self.scan_now());
        *cache = Some((std::time::Instant::now(), scan.clone()));
        scan
    }

    fn scan_now(&self) -> super::library::Scan {
        let managed = self.list().unwrap_or_default();
        let skip_dirs: std::collections::HashSet<PathBuf> =
            managed.iter().map(|m| self.root.join(&m.id)).collect();
        let taken: std::collections::HashSet<String> = managed.into_iter().map(|m| m.id).collect();
        let mut roots = Vec::with_capacity(1 + self.libraries.len());
        if self.root.is_dir() {
            roots.push(self.root.clone());
        }
        roots.extend(self.libraries.iter().filter(|l| **l != self.root).cloned());
        super::library::scan(&roots, &skip_dirs, &taken)
    }

    /// The model `name` names in a library, if any.
    pub fn library_model(&self, name: &str) -> Option<super::library::LibraryModel> {
        self.library().get(name).cloned()
    }

    /// The weights `name` loads: a model of the store, else one of a library.
    /// `gguf_path` stays the store's alone, because a pull asks it whether a
    /// model is already downloaded, and a library's file is not.
    pub fn resolve_gguf(&self, name: &str) -> Option<PathBuf> {
        self.gguf_path(name).or_else(|| self.library_model(name).map(|m| m.path))
    }

    /// The projector of `name`: the store's, else the one beside a library model.
    pub fn resolve_mmproj(&self, name: &str) -> Option<PathBuf> {
        self.mmproj_path(name)
            .or_else(|| self.library_model(name).and_then(|m| m.mmproj))
    }

    /// The store's models and the libraries' together, for the listings
    /// (`eullm list`, `/api/tags`, `/v1/models`, the picker). A library model
    /// gets a manifest made up on the spot, never written.
    pub fn list_all(&self) -> Result<Vec<ModelManifest>, Box<dyn std::error::Error>> {
        let mut models = self.list()?;
        models.extend(self.library().models.iter().map(library_manifest));
        Ok(models)
    }

    /// `get`, falling back to the libraries.
    pub fn get_any(&self, name: &str) -> Result<Option<ModelManifest>, Box<dyn std::error::Error>> {
        if let Some(m) = self.get(name)? {
            return Ok(Some(m));
        }
        Ok(self.library_model(name).as_ref().map(library_manifest))
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
        let Some(model_dir) = self.safe_model_dir(name) else {
            return Ok(None);
        };
        let manifest_path = model_dir.join("manifest.json");

        if !manifest_path.exists() {
            return Ok(None);
        }

        let data = fs::read_to_string(&manifest_path)?;
        let mut manifest: ModelManifest = serde_json::from_str(&data)?;
        // Backfill the addressable id from the directory key for older manifests.
        if manifest.id.is_empty() {
            manifest.id = name.strip_prefix("eullm/").unwrap_or(name).to_string();
        }
        Ok(Some(manifest))
    }

    /// Check if a model exists locally.
    pub fn exists(&self, name: &str) -> bool {
        self.safe_model_dir(name)
            .is_some_and(|d| d.join("manifest.json").exists())
    }

    /// Return the path to the model directory.
    ///
    /// A name that would escape the store root (see `safe_model_dir`) is
    /// mapped to a fixed non-escaping placeholder under the root rather than
    /// the traversing join, so no caller can be steered outside the store —
    /// every real caller passes a store-owned id, and the placeholder simply
    /// won't exist for a crafted one.
    pub fn model_path(&self, name: &str) -> PathBuf {
        self.safe_model_dir(name)
            .unwrap_or_else(|| self.root.join("__eullm_invalid_model_name__"))
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
        create_model_dir(&dir).map_err(|e| -> Box<dyn std::error::Error> { e.to_string().into() })?;
        Ok(dir)
    }

    /// Remove a model's entire directory from disk.
    ///
    /// Returns `Ok(Some(removed))` if the directory existed and was removed,
    /// `Ok(None)` if there was nothing to remove. Errors only on real
    /// filesystem failures.
    ///
    /// Weights shared with another id through `link_files` stay usable there:
    /// removing a directory removes one name for them, and the data goes only
    /// with the last name.
    pub fn delete(&self, name: &str) -> Result<Option<Removed>, Box<dyn std::error::Error>> {
        // Never let a crafted name steer `remove_dir_all` outside the store —
        // this is the most destructive of the store lookups.
        let Some(model_dir) = self.safe_model_dir(name) else {
            return Ok(None);
        };
        if !model_dir.exists() {
            return Ok(None);
        }
        let removed = removal_size(&model_dir).unwrap_or_default();
        fs::remove_dir_all(&model_dir)?;
        Ok(Some(removed))
    }

    /// Give model `to` the files `files` of model `from`, as hard links: one
    /// copy on disk under two ids.
    ///
    /// A link and not a copy, because the disk space is the point: these are
    /// multi-gigabyte files, and the caller has established they are the same
    /// download. A hard link and not a symlink, because each id has to survive
    /// the other's `eullm rm` — removing a directory drops one name for the
    /// data, which goes only with the last name, where a symlink would be left
    /// pointing at nothing.
    ///
    /// All or nothing. Every file is checked to be there before anything is
    /// linked, and when a link fails — the two directories on different
    /// filesystems, or a filesystem without hard links such as FAT32 or exFAT
    /// — the links already made are removed again, and so is `to`'s directory
    /// if this call created it. Half a split model under a new id would list
    /// as present and fail to load.
    ///
    /// `link` makes one link: `std::fs::hard_link` outside the tests, which
    /// need a link that fails.
    pub fn link_files(
        &self,
        from: &str,
        to: &str,
        files: &[String],
        link: &dyn Fn(&Path, &Path) -> std::io::Result<()>,
    ) -> std::io::Result<()> {
        use std::io::{Error, ErrorKind};
        let src_dir = self.safe_model_dir(from).ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidInput,
                format!("invalid model id {from:?}"),
            )
        })?;
        let dst_dir = self.safe_model_dir(to).ok_or_else(|| {
            Error::new(ErrorKind::InvalidInput, format!("invalid model id {to:?}"))
        })?;
        for name in files {
            if !is_safe_filename(name) {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    format!("unsafe file name {name:?}"),
                ));
            }
            if !src_dir.join(name).is_file() {
                return Err(Error::new(
                    ErrorKind::NotFound,
                    format!("{name} is not in {}", src_dir.display()),
                ));
            }
        }
        let created = !dst_dir.exists();
        create_model_dir(&dst_dir).map_err(Error::other)?;
        let mut made = Vec::with_capacity(files.len());
        for name in files {
            let dst = dst_dir.join(name);
            if let Err(e) = link(&src_dir.join(name), &dst) {
                for path in &made {
                    let _ = fs::remove_file(path);
                }
                if created {
                    let _ = fs::remove_dir(&dst_dir);
                }
                return Err(e);
            }
            made.push(dst);
        }
        Ok(())
    }
}

/// The id a model stored under file stem `stem` gets: lowercase, with
/// anything but ASCII letters, digits, `.`, `_` and `-` replaced by `-`.
/// Shared by a pull from a URL and `import`, so the same file gets the same
/// id both ways.
pub fn model_id_for(stem: &str) -> String {
    let id: String = stem
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '-'
            }
        })
        .collect();
    let id = id.trim_matches('-');
    // A leading dot would make the directory hidden, and `is_safe_filename`
    // refuses it.
    let id = id.trim_start_matches('.');
    if id.is_empty() {
        "model".to_string()
    } else {
        id.to_string()
    }
}

/// What `ModelStore::delete` removed.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Removed {
    /// Bytes given back to the filesystem.
    pub freed: u64,
    /// Bytes that stay on disk because another hard link keeps them: weights
    /// a pull shared with another model id (`ModelStore::link_files`). Always
    /// 0 where the link count cannot be read, which is Windows: it is then
    /// counted as freed, as it always was.
    pub shared: u64,
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
pub(crate) fn create_model_dir(
    dir: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
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

/// What removing everything under `path` gives back, for reporting what a
/// `rm` actually freed. A file with another hard link stays on disk under
/// that other name, so it counts as shared rather than freed: once a pull can
/// link the same weights into a second id, "5.0 GB freed" for a removal that
/// frees nothing would be a false statement about the user's disk.
fn removal_size(path: &std::path::Path) -> Result<Removed, Box<dyn std::error::Error>> {
    let mut total = Removed::default();
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let meta = entry.metadata()?;
        if meta.is_dir() {
            let sub = removal_size(&entry.path())?;
            total.freed += sub.freed;
            total.shared += sub.shared;
        } else if has_other_links(&meta) {
            total.shared += meta.len();
        } else {
            total.freed += meta.len();
        }
    }
    Ok(total)
}

#[cfg(unix)]
fn has_other_links(meta: &fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    meta.nlink() > 1
}

/// The standard library has no stable way to read a link count on Windows
/// (`MetadataExt::number_of_links` is still unstable), so a shared file is
/// reported as freed there. Only the message is affected, not the removal.
#[cfg(not(unix))]
fn has_other_links(_meta: &fs::Metadata) -> bool {
    false
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
mod projector_with_mtp_tests {
    use super::*;

    #[test]
    fn a_found_projector_gives_way_to_the_drafts_asked_for() {
        let p = PathBuf::from("/m/qwen3.5-9b/mmproj-F16.gguf");
        let (kept, line) = projector_with_mtp(Some(p.clone()), false, true, "qwen3.5-9b");
        assert_eq!(kept, None);
        let line = line.expect("said");
        assert!(
            line.contains("qwen3.5-9b loads without its projector"),
            "{line}"
        );
        assert!(line.contains("mmproj-F16.gguf"), "{line}");

        // Named with --mmproj, it stays, and the line says --mtp is idle.
        let (kept, line) = projector_with_mtp(Some(p.clone()), true, true, "qwen3.5-9b");
        assert_eq!(kept, Some(p.clone()));
        assert!(line.expect("said").contains("--mtp does nothing for it"));

        // Without drafts to make room for, nothing changes and nothing is said.
        assert_eq!(
            projector_with_mtp(Some(p.clone()), false, false, "m"),
            (Some(p), None)
        );
        assert_eq!(projector_with_mtp(None, false, true, "m"), (None, None));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_store() -> (ModelStore, PathBuf) {
        let root = std::env::temp_dir().join(format!("eullm-test-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        (ModelStore::at(root.clone()), root)
    }

    /// A store path that is a symlink to an unmounted volume must explain
    /// itself. `create_dir_all` answers EEXIST there — the symlink exists,
    /// it just does not lead anywhere — and reporting that raw reads as
    /// "File exists (os error 17)" in the middle of a download, which sounds
    /// like the model is already present. Reported on a real setup where the
    /// store lived on a volume that was not mounted yet.
    #[cfg(unix)]
    #[test]
    fn a_dangling_store_symlink_explains_itself() {
        let base = std::env::temp_dir().join(format!("eullm-test-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&base).unwrap();
        let link = base.join("models");
        std::os::unix::fs::symlink(base.join("not-mounted"), &link).unwrap();

        let err = create_model_dir(&link.join("some-model"))
            .expect_err("a dangling symlink cannot be created into")
            .to_string();
        assert!(
            err.contains("mount it first") || err.contains("symlink"),
            "error must point at the unmounted volume, got: {err}"
        );
        let _ = fs::remove_dir_all(&base);
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
            .write_external_manifest(
                "my-model",
                "weights.gguf",
                "https://x/y.gguf",
                123,
                None,
                None,
            )
            .unwrap();
        let listed = store.list().unwrap();
        assert_eq!(listed[0].id, "my-model");
        fs::remove_dir_all(&root).ok();
    }
}

#[cfg(test)]
mod safe_relative_path_tests {
    use super::{is_safe_filename, is_safe_relative_path};

    /// The case that motivated this: `unsloth/Qwen3.8-Flash-Next-GGUF` keeps
    /// every quantization in its own subdirectory, so judging the HF API's
    /// `rfilename` values as single filenames dropped all the weights and left
    /// only the repo-root projector — reported to the user as "this repo
    /// contains only a projector, no model weights".
    #[test]
    fn accepts_the_per_quant_subdirectory_layout() {
        for path in [
            "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
            "Q8_0/Qwen3.8-Flash-Next-Q8_0-00006-of-00006.gguf",
            "BF16/Qwen3.8-Flash-Next-BF16-00001-of-00008.gguf",
            "MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf",
        ] {
            assert!(is_safe_relative_path(path), "should accept {path:?}");
            assert!(
                !is_safe_filename(path),
                "{path:?} is exactly the shape the old check rejected"
            );
        }
    }

    /// Widening to paths must not widen what escapes: every component is still
    /// judged by the same rules.
    #[test]
    fn still_rejects_anything_that_escapes() {
        for path in [
            "../../../../etc/cron.d/payload.gguf",
            "UD-Q4_K_XL/../../../etc/shadow.gguf",
            "/etc/shadow.gguf",
            "UD-Q4_K_XL//model.gguf", // empty component
            "UD-Q4_K_XL/",
            "sub\\dir/model.gguf",
            "C:\\Windows\\model.gguf",
            "UD-Q4_K_XL/.hidden.gguf",
            "UD-Q4_K_XL/model\0.gguf",
            "",
        ] {
            assert!(!is_safe_relative_path(path), "should reject {path:?}");
        }
    }

    /// A bare filename is still a valid relative path — the single-component
    /// case must keep working, since most repos are laid out flat.
    #[test]
    fn a_bare_filename_is_still_accepted() {
        for name in ["qwen3-8b-Q4_K_M.gguf", "mmproj-F16.gguf"] {
            assert!(is_safe_relative_path(name));
        }
    }

    /// The property callers rely on: taking the last component of any accepted
    /// path yields a name that is itself safe to join onto the model directory.
    #[test]
    fn the_basename_of_an_accepted_path_is_always_a_safe_filename() {
        for path in [
            "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
            "a/b/c/model.gguf",
            "model.gguf",
        ] {
            assert!(is_safe_relative_path(path));
            let leaf = path.rsplit('/').next().unwrap();
            assert!(is_safe_filename(leaf), "basename of {path:?} must be safe");
        }
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
            ModelStore::at(self.0.clone())
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
            ModelStore::at(self.0.clone())
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

    // A crafted `model` name must never let a lookup escape the store root.
    // Every store lookup reachable from an API request goes through
    // `safe_model_dir`; here we plant a real .gguf OUTSIDE the store and
    // confirm a `../`-laden name cannot reach it, delete it, or confirm its
    // existence.
    #[test]
    fn a_traversing_name_cannot_escape_the_store_root() {
        let s = Scratch::new();
        // A sibling dir next to the store root, holding weights the store must
        // not be able to reach by name.
        let outside = s.0.parent().unwrap().join(format!(
            "eullm-outside-{}",
            s.0.file_name().unwrap().to_string_lossy()
        ));
        fs::create_dir_all(&outside).expect("outside dir");
        fs::write(outside.join("model.gguf"), b"x").expect("outside gguf");
        let store = s.store();

        let escape = format!("../{}", outside.file_name().unwrap().to_string_lossy());
        assert!(
            store.gguf_path(&escape).is_none(),
            "gguf_path must not resolve a ../ name outside the root"
        );
        assert!(
            !store.exists(&escape),
            "exists must not confirm a directory outside the root"
        );
        assert!(store.get(&escape).expect("get").is_none());
        assert!(store.mmproj_path(&escape).is_none());
        // delete must be a no-op, and the outside weights must survive.
        assert!(store.delete(&escape).expect("delete").is_none());
        assert!(
            outside.join("model.gguf").exists(),
            "delete escaped the store root and removed an outside file"
        );

        let _ = fs::remove_dir_all(&outside);
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
        assert_eq!(source, "test");
    }

    #[test]
    fn the_environment_wins_over_the_env_file_and_the_default() {
        let file = "EULLM_MODELS_DIR=/from/file\n";
        let (root, source) =
            ModelStore::resolve_root(Some("/from/env"), Some(file), ".env", Some("/home/u"))
                .unwrap();
        assert_eq!(root, PathBuf::from("/from/env"));
        assert_eq!(source, "EULLM_MODELS_DIR");
    }

    /// The bug this exists for: the `.env` line was documented and ignored.
    #[test]
    fn the_env_file_is_read_when_the_environment_says_nothing() {
        let file = "# models\nEULLM_ALLOWED_IPS=10.0.0.0/8\nEULLM_MODELS_DIR=\"/data/models\"\n";
        for env in [None, Some(""), Some("  ")] {
            let (root, source) =
                ModelStore::resolve_root(env, Some(file), ".env", Some("/home/u")).unwrap();
            assert_eq!(root, PathBuf::from("/data/models"));
            assert_eq!(source, "EULLM_MODELS_DIR (.env)");
        }
    }

    #[test]
    fn nothing_configured_is_the_home_directory() {
        for file in [None, Some("EULLM_MODELS_DIR=\n"), Some("EULLM_API_KEYS=x\n")] {
            let (root, source) =
                ModelStore::resolve_root(None, file, ".env", Some("/home/u")).unwrap();
            assert_eq!(root, PathBuf::from("/home/u").join(".eullm").join("models"));
            assert_eq!(source, "default");
        }
        assert!(ModelStore::resolve_root(None, None, ".env", None).is_err());
    }
}

#[cfg(test)]
mod hf_file_tests {
    use super::*;

    fn manifest(extra: serde_json::Value) -> ModelManifest {
        let mut m = serde_json::json!({
            "id": "some-model", "name": "some-model", "description": "",
            "languages": [], "base": "", "vram_gb": 0, "size_bytes": 1,
            "license": "unknown", "digest": "", "pulled_at": "", "status": "ready",
        });
        for (k, v) in extra.as_object().expect("an object") {
            m[k] = v.clone();
        }
        serde_json::from_value(m).expect("a valid manifest")
    }

    #[test]
    fn a_recorded_file_matches_by_its_whole_path() {
        let m = manifest(serde_json::json!({
            "gguf_file": "M-UD-Q4_K_XL-00001-of-00002.gguf",
            "hf_repo": "someone/M-GGUF",
            "hf_filename": "UD-Q4_K_XL/M-UD-Q4_K_XL-00001-of-00002.gguf",
        }));
        assert!(m.holds(&HfFile::new(
            "someone/M-GGUF",
            "UD-Q4_K_XL/M-UD-Q4_K_XL-00001-of-00002.gguf"
        )));
        assert!(!m.holds(&HfFile::new(
            "someone/M-GGUF",
            "old/M-UD-Q4_K_XL-00001-of-00002.gguf"
        )));
        assert!(
            !m.holds(&HfFile::new(
                "someone/M-GGUF",
                "ud-q4_k_xl/m-ud-q4_k_xl-00001-of-00002.gguf"
            )),
            "a path inside a repository is case-sensitive"
        );
    }

    // All an `hf.co` pull used to record was the ref and the stored name,
    // which has lost the directory. The name is then all there is to compare.
    #[test]
    fn an_older_hub_pull_is_matched_by_the_name_it_stored() {
        let m = manifest(serde_json::json!({
            "description": "External model pulled from hf.co/someone/M-GGUF:UD-Q4_K_XL",
            "gguf_file": "M-UD-Q4_K_XL-00001-of-00002.gguf",
        }));
        assert!(m.holds(&HfFile::new(
            "someone/m-gguf",
            "UD-Q4_K_XL/M-UD-Q4_K_XL-00001-of-00002.gguf"
        )));
        assert!(!m.holds(&HfFile::new(
            "someone/M-GGUF",
            "Q8_0/M-Q8_0-00001-of-00002.gguf"
        )));
        assert!(!m.holds(&HfFile::new(
            "other/M-GGUF",
            "UD-Q4_K_XL/M-UD-Q4_K_XL-00001-of-00002.gguf"
        )));
    }

    // A plain URL and an Ollama import (made by a release that still had the
    // command; its manifests stay on disk) say nothing about which Hub file
    // they are, so they must never be linked in as one.
    #[test]
    fn a_url_pull_or_an_ollama_import_is_never_a_hub_file() {
        let wanted = HfFile::new("unsloth/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf");
        for description in [
            "External model pulled from https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
            "Imported from Ollama: qwen3:8b",
        ] {
            let m = manifest(serde_json::json!({
                "description": description, "gguf_file": "Qwen3-8B-Q4_K_M.gguf",
            }));
            assert!(!m.holds(&wanted), "{description}");
        }
    }

    // The new fields are written when known and left out otherwise, and a
    // manifest from before they existed still reads — with them absent.
    #[test]
    fn the_file_is_recorded_when_known_and_old_manifests_still_read() {
        let root = std::env::temp_dir().join(format!("eullm-hffile-{}", uuid::Uuid::new_v4()));
        let store = ModelStore::at(root.clone());
        let file = HfFile::new("unsloth/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf");
        store
            .write_external_manifest(
                "from-hub",
                "Qwen3-8B-Q4_K_M.gguf",
                "hf.co/x/y",
                1,
                None,
                Some(&file),
            )
            .expect("write");
        store
            .write_external_manifest(
                "from-url",
                "model.gguf",
                "https://host/model.gguf",
                1,
                None,
                None,
            )
            .expect("write");

        let hub = store.get("from-hub").expect("get").expect("manifest");
        assert_eq!(hub.hf_repo.as_deref(), Some("unsloth/Qwen3-8B-GGUF"));
        assert_eq!(hub.hf_filename.as_deref(), Some("Qwen3-8B-Q4_K_M.gguf"));
        let raw = fs::read_to_string(root.join("from-url").join("manifest.json")).expect("read");
        assert!(
            !raw.contains("hf_repo") && !raw.contains("hf_filename"),
            "{raw}"
        );

        let old = manifest(serde_json::json!({ "gguf_file": "model.gguf" }));
        assert_eq!((old.hf_repo, old.hf_filename), (None, None));
        let _ = fs::remove_dir_all(&root);
    }
}


#[cfg(test)]
mod library_tests {
    use super::*;
    use uuid::Uuid;

    struct Scratch(PathBuf);
    impl Scratch {
        fn new() -> Self {
            let d = std::env::temp_dir().join(format!("eullm-store-lib-{}", Uuid::new_v4()));
            fs::create_dir_all(d.join("store")).expect("mkdir");
            fs::create_dir_all(d.join("lib")).expect("mkdir");
            Self(d)
        }
        fn file(&self, rel: &str) -> PathBuf {
            let p = self.0.join(rel);
            fs::create_dir_all(p.parent().unwrap()).expect("mkdir");
            fs::write(&p, b"not a real gguf").expect("write");
            p
        }
        fn store(&self) -> ModelStore {
            ModelStore::at(self.0.join("store")).with_libraries(vec![self.0.join("lib")])
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn a_library_model_is_listed_and_runs_by_name_where_it_lies() {
        let s = Scratch::new();
        let file = s.file("lib/lmstudio-community/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf");
        let projector = s.file("lib/lmstudio-community/Qwen3-8B-GGUF/mmproj-F16.gguf");
        let store = s.store();

        assert!(store.list().unwrap().is_empty(), "the store's own list is the store's");
        let all = store.list_all().unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].id, "qwen3-8b-q4_k_m");
        assert_eq!(all[0].status, "library");
        assert_eq!(all[0].written_description(), None);

        assert_eq!(store.resolve_gguf("qwen3-8b-q4_k_m"), Some(file.clone()));
        assert_eq!(store.resolve_mmproj("qwen3-8b-q4_k_m"), Some(projector));
        assert!(store.is_present("qwen3-8b-q4_k_m"));
        // A pull asks gguf_path whether a model is downloaded: a library's is not.
        assert_eq!(store.gguf_path("qwen3-8b-q4_k_m"), None);
        // Nothing was written next to it.
        let entries = fs::read_dir(file.parent().unwrap()).unwrap().count();
        assert_eq!(entries, 2);
    }

    #[test]
    fn the_store_s_own_folder_is_read_like_a_library_outside_its_models() {
        let s = Scratch::new();
        let loose = s.file("store/copied-in-by-hand.gguf");
        s.file("store/pulled/model.gguf");
        let store = s.store();
        store
            .write_external_manifest("pulled", "model.gguf", "https://example.eu/m.gguf", 1, None, None)
            .unwrap();
        let all = store.list_all().unwrap();
        let ids: Vec<&str> = all.iter().map(|m| m.id.as_str()).collect();
        // The pulled model once, from its manifest; the loose file found in place.
        assert_eq!(ids, vec!["pulled", "copied-in-by-hand"]);
        assert_eq!(store.resolve_gguf("copied-in-by-hand"), Some(loose.clone()));
        assert!(loose.is_file(), "not moved");
    }

    #[test]
    fn a_store_id_wins_over_a_library_file_of_the_same_name() {
        let s = Scratch::new();
        s.file("store/qwen3-8b/model.gguf");
        s.file("lib/some-folder/qwen3-8b.gguf");
        let store = s.store();
        store
            .write_external_manifest("qwen3-8b", "model.gguf", "https://example.eu/m.gguf", 1, None, None)
            .unwrap();
        let ids: Vec<String> = store.list_all().unwrap().into_iter().map(|m| m.id).collect();
        assert_eq!(ids, vec!["qwen3-8b".to_string(), "some-folder-qwen3-8b".to_string()]);
        assert!(store.resolve_gguf("qwen3-8b").unwrap().starts_with(s.0.join("store")));
    }

    #[test]
    fn the_libraries_come_from_the_environment_then_the_env_file() {
        let file = "EULLM_MODEL_LIBRARIES=/from/file;/also\n";
        let (libs, source) = ModelStore::resolve_libraries(Some("/env"), Some(file), ".env");
        assert_eq!((libs, source.as_str()), (vec![PathBuf::from("/env")], "EULLM_MODEL_LIBRARIES"));
        let (libs, source) = ModelStore::resolve_libraries(Some(" "), Some(file), ".env");
        assert_eq!(libs, vec![PathBuf::from("/from/file"), PathBuf::from("/also")]);
        assert_eq!(source, "EULLM_MODEL_LIBRARIES (.env)");
        let (libs, source) = ModelStore::resolve_libraries(None, None, ".env");
        assert!(libs.is_empty());
        assert_eq!(source, "none");
    }
}

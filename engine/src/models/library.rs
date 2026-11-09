//! Model libraries: folders of GGUF files that EuLLM reads in place and never
//! writes to.
//!
//! A user coming from LM Studio, Ollama's GGUF exports, a Hugging Face cache
//! or a folder of their own keeps that folder as it is, usable by the program
//! that made it: EuLLM lists and runs what it finds there, through every
//! level of subfolders, and moves, renames or deletes nothing. The folders
//! are named by `EULLM_MODEL_LIBRARIES` (in the environment or in `.env`),
//! separated by `;` on every platform and also by `:` outside Windows.
//!
//! What is found is worked out again on every scan, so nothing has to be kept
//! in step with the folder: no index, no manifest. A scan reads the start of
//! each `.gguf` only, to tell a projector from a model.

use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File};
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// A model found in a library: one `.gguf`, or every shard of a split model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LibraryModel {
    /// The name `eullm run` and the API's `model` field take.
    pub id: String,
    /// The file llama.cpp opens: the model itself, or a split's first shard.
    pub path: PathBuf,
    /// The projector in the same folder, when it holds exactly one.
    pub mmproj: Option<PathBuf>,
    /// Bytes of the weights, every shard counted.
    pub size_bytes: u64,
    /// The folder of `EULLM_MODEL_LIBRARIES` (or the model store) it was found in.
    pub library: PathBuf,
}

/// A `.gguf` a scan found and did not offer as a model, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skipped {
    pub path: PathBuf,
    pub reason: String,
}

/// What a scan of the libraries found.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Scan {
    pub models: Vec<LibraryModel>,
    pub skipped: Vec<Skipped>,
}

impl Scan {
    /// The model named `id`, if a library holds it.
    pub fn get(&self, id: &str) -> Option<&LibraryModel> {
        self.models.iter().find(|m| m.id == id)
    }
}

/// The folders `EULLM_MODEL_LIBRARIES` names: blank entries dropped, in the
/// order given. `;` separates them everywhere, `:` too outside Windows, where
/// it cannot be part of a path's drive letter.
pub fn parse_libraries(spec: &str) -> Vec<PathBuf> {
    spec.split(|c| c == ';' || (!cfg!(windows) && c == ':'))
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .collect()
}

/// How deep a scan goes below a library's folder: every level a folder of
/// models has (LM Studio: publisher/repository/file; a Hugging Face cache:
/// models--org--repo/snapshots/hash/file), with room to spare. The bound only
/// keeps a library pointed at a whole disk from walking it for minutes.
const MAX_DEPTH: usize = 32;

/// Scan `roots` for models. A folder of `skip_dirs` is not entered (the model
/// store's own model directories, which have manifests), and an id in `taken`
/// is not given to a library model (the store's ids, which win).
///
/// Ids come from the file names (`Qwen3-8B-Q4_K_M.gguf` is `qwen3-8b-q4_k_m`);
/// when two files would have the same id, every one of them takes its folder's
/// name in front (`lmstudio-community-qwen3-8b-q4_k_m`), and a number after
/// that if it is still not unique.
pub fn scan(roots: &[PathBuf], skip_dirs: &HashSet<PathBuf>, taken: &HashSet<String>) -> Scan {
    let mut found: Vec<(PathBuf, Found)> = Vec::new();
    let mut skipped = Vec::new();
    for root in roots {
        let mut visited = HashSet::new();
        walk(root, root, 0, skip_dirs, &mut visited, &mut found, &mut skipped);
    }

    // Ids: the file's stem, or folder and stem where two files share a stem.
    let base_ids: Vec<String> = found.iter().map(|(_, f)| super::store::model_id_for(&f.stem)).collect();
    let mut count: BTreeMap<&str, usize> = BTreeMap::new();
    for id in &base_ids {
        *count.entry(id.as_str()).or_default() += 1;
    }
    let mut used: HashSet<String> = taken.clone();
    let mut models = Vec::with_capacity(found.len());
    for ((library, f), base) in found.into_iter().zip(&base_ids) {
        let mut id = if count[base.as_str()] > 1 || taken.contains(base) {
            let folder = f
                .path
                .parent()
                .and_then(Path::file_name)
                .and_then(|n| n.to_str())
                .unwrap_or("");
            super::store::model_id_for(&format!("{folder}-{}", f.stem))
        } else {
            base.clone()
        };
        if used.contains(&id) {
            let stem = id.clone();
            let mut n = 2;
            while used.contains(&id) {
                id = format!("{stem}-{n}");
                n += 1;
            }
        }
        used.insert(id.clone());
        models.push(LibraryModel {
            id,
            path: f.path,
            mmproj: f.mmproj,
            size_bytes: f.size_bytes,
            library,
        });
    }
    Scan { models, skipped }
}

/// A model before it has an id.
struct Found {
    path: PathBuf,
    stem: String,
    mmproj: Option<PathBuf>,
    size_bytes: u64,
}

fn walk(
    root: &Path,
    dir: &Path,
    depth: usize,
    skip_dirs: &HashSet<PathBuf>,
    visited: &mut HashSet<PathBuf>,
    found: &mut Vec<(PathBuf, Found)>,
    skipped: &mut Vec<Skipped>,
) {
    if depth > MAX_DEPTH || skip_dirs.contains(dir) {
        return;
    }
    // A folder reached twice (a link back up the tree) is walked once.
    if let Ok(canonical) = fs::canonicalize(dir)
        && !visited.insert(canonical)
    {
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut subdirs = Vec::new();
    let mut ggufs = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        // Hidden folders and files: other programs' state, partial downloads.
        if name.starts_with('.') {
            continue;
        }
        let path = entry.path();
        // `metadata` follows links: a Hugging Face cache's snapshot files are
        // links to its blobs, and they are the files to list.
        let Ok(meta) = fs::metadata(&path) else {
            continue;
        };
        if meta.is_dir() {
            subdirs.push(path);
        } else if meta.is_file() && is_gguf_name(name) {
            ggufs.push((name.to_string(), path, meta.len()));
        }
    }
    subdirs.sort();
    ggufs.sort();

    let mut projectors = Vec::new();
    let mut weights = Vec::new();
    for (name, path, size) in ggufs {
        if is_projector(&name, &path) {
            projectors.push(path);
        } else {
            weights.push((name, path, size));
        }
    }
    let mmproj = match projectors.as_slice() {
        [one] => Some(one.clone()),
        _ => None,
    };
    if weights.is_empty() {
        for p in &projectors {
            skipped.push(Skipped {
                path: p.clone(),
                reason: "a projector (mmproj), with no model beside it".to_string(),
            });
        }
    }

    for group in group_weights(weights) {
        match group {
            Group::Model { stem, path, size_bytes } => found.push((
                root.to_path_buf(),
                Found {
                    path,
                    stem,
                    mmproj: mmproj.clone(),
                    size_bytes,
                },
            )),
            Group::Incomplete { path, present, total } => skipped.push(Skipped {
                path,
                reason: format!("split model with {present} of its {total} parts"),
            }),
        }
    }

    for sub in subdirs {
        walk(root, &sub, depth + 1, skip_dirs, visited, found, skipped);
    }
}

enum Group {
    Model { stem: String, path: PathBuf, size_bytes: u64 },
    Incomplete { path: PathBuf, present: usize, total: u32 },
}

/// One model per file, and one per split: `X-00001-of-00003.gguf` and its
/// siblings are one model, opened from the first.
fn group_weights(weights: Vec<(String, PathBuf, u64)>) -> Vec<Group> {
    // (index, path, size) of each part of a split found
    type Parts = Vec<(u32, PathBuf, u64)>;
    let mut out = Vec::new();
    // stem -> (total the names claim, its parts)
    let mut splits: BTreeMap<String, (u32, Parts)> = BTreeMap::new();
    for (name, path, size) in weights {
        match crate::registry::parse_shard(&name) {
            Some((stem, index, total)) => splits
                .entry(stem.to_string())
                .or_insert_with(|| (total, Vec::new()))
                .1
                .push((index, path, size)),
            None => out.push(Group::Model {
                stem: name[..name.len() - ".gguf".len()].to_string(),
                path,
                size_bytes: size,
            }),
        }
    }
    for (stem, (total, mut parts)) in splits {
        parts.sort_by_key(|(i, _, _)| *i);
        parts.dedup_by_key(|(i, _, _)| *i);
        if parts.len() as u32 != total || parts[0].0 != 1 {
            out.push(Group::Incomplete {
                path: parts[0].1.clone(),
                present: parts.len(),
                total,
            });
            continue;
        }
        out.push(Group::Model {
            stem,
            path: parts[0].1.clone(),
            size_bytes: parts.iter().map(|(_, _, s)| s).sum(),
        });
    }
    out
}

fn is_gguf_name(name: &str) -> bool {
    name.len() > 5 && name[name.len() - 5..].eq_ignore_ascii_case(".gguf")
}

/// Whether a `.gguf` is a multimodal projector rather than a model: by its
/// name (`mmproj-*`, how every published projector is named), or else by its
/// header, which says `general.architecture = clip` for a projector whatever
/// the file is called.
fn is_projector(name: &str, path: &Path) -> bool {
    crate::registry::is_mmproj(name)
        || gguf_architecture(path).is_ok_and(|arch| arch.as_deref() == Some("clip"))
}

/// `general.architecture` from a GGUF's header, `None` when the file does not
/// declare one before its larger entries. Reads the metadata entries in order
/// and stops at that key; the header puts it first in every GGUF llama.cpp's
/// tools write, so this reads a few hundred bytes.
fn gguf_architecture(path: &Path) -> io::Result<Option<String>> {
    let mut r = BufReader::with_capacity(16 * 1024, File::open(path)?);
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        return Ok(None);
    }
    let version = read_u32(&mut r)?;
    if version < 2 {
        return Ok(None);
    }
    let _n_tensors = read_u64(&mut r)?;
    let n_kv = read_u64(&mut r)?;
    // The general.* keys come first; past these many entries the file is in
    // its vocabulary and the key is not there.
    for _ in 0..n_kv.min(64) {
        let key = read_string(&mut r, 1 << 16)?;
        let ty = read_u32(&mut r)?;
        if key == "general.architecture" && ty == GGUF_STRING {
            return Ok(Some(read_string(&mut r, 1 << 16)?));
        }
        skip_value(&mut r, ty)?;
    }
    Ok(None)
}

const GGUF_STRING: u32 = 8;
const GGUF_ARRAY: u32 = 9;

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_string(r: &mut impl Read, max: u64) -> io::Result<String> {
    let len = read_u64(r)?;
    if len > max {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "GGUF string too long"));
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// Bytes of one value of a fixed-size GGUF type, `None` for strings and arrays.
fn fixed_size(ty: u32) -> Option<i64> {
    match ty {
        0 | 1 | 7 => Some(1),  // u8, i8, bool
        2 | 3 => Some(2),      // u16, i16
        4..=6 => Some(4),      // u32, i32, f32
        10..=12 => Some(8),    // u64, i64, f64
        _ => None,
    }
}

fn skip_value(r: &mut (impl Read + Seek), ty: u32) -> io::Result<()> {
    if let Some(n) = fixed_size(ty) {
        r.seek(SeekFrom::Current(n))?;
        return Ok(());
    }
    match ty {
        GGUF_STRING => {
            let len = read_u64(r)?;
            r.seek(SeekFrom::Current(i64::try_from(len).unwrap_or(i64::MAX)))?;
        }
        GGUF_ARRAY => {
            let elem = read_u32(r)?;
            let n = read_u64(r)?;
            if let Some(size) = fixed_size(elem) {
                r.seek(SeekFrom::Current(size.saturating_mul(i64::try_from(n).unwrap_or(i64::MAX))))?;
            } else if elem == GGUF_STRING {
                for _ in 0..n {
                    let len = read_u64(r)?;
                    r.seek(SeekFrom::Current(i64::try_from(len).unwrap_or(i64::MAX)))?;
                }
            } else {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "nested GGUF array"));
            }
        }
        _ => return Err(io::Error::new(io::ErrorKind::InvalidData, "unknown GGUF type")),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    struct Scratch(PathBuf);
    impl Scratch {
        fn new() -> Self {
            let d = std::env::temp_dir().join(format!("eullm-library-{}", Uuid::new_v4()));
            fs::create_dir_all(&d).expect("mkdir");
            Self(d)
        }
        fn file(&self, rel: &str, bytes: &[u8]) -> PathBuf {
            let p = self.0.join(rel);
            fs::create_dir_all(p.parent().unwrap()).expect("mkdir");
            fs::write(&p, bytes).expect("write");
            p
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// A GGUF header with `general.architecture` after a few other entries.
    fn header(arch: &str) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&3u64.to_le_bytes());
        let key = |b: &mut Vec<u8>, k: &str| {
            b.extend_from_slice(&(k.len() as u64).to_le_bytes());
            b.extend_from_slice(k.as_bytes());
        };
        key(&mut b, "general.alignment");
        b.extend_from_slice(&4u32.to_le_bytes());
        b.extend_from_slice(&32u32.to_le_bytes());
        key(&mut b, "general.tags");
        b.extend_from_slice(&9u32.to_le_bytes());
        b.extend_from_slice(&8u32.to_le_bytes());
        b.extend_from_slice(&2u64.to_le_bytes());
        for t in ["a", "bc"] {
            key(&mut b, t);
        }
        key(&mut b, "general.architecture");
        b.extend_from_slice(&8u32.to_le_bytes());
        key(&mut b, arch);
        b
    }

    fn scan_one(root: &Path) -> Scan {
        scan(&[root.to_path_buf()], &HashSet::new(), &HashSet::new())
    }

    #[test]
    fn every_level_of_subfolders_is_read_and_nothing_is_moved() {
        let s = Scratch::new();
        let a = s.file("lmstudio-community/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf", &header("qwen3"));
        let b = s.file("loose.gguf", &header("llama"));
        let c = s.file("a/b/c/d/deep.gguf", &header("gemma3"));
        let found = scan_one(&s.0);
        let ids: Vec<&str> = found.models.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, vec!["loose", "deep", "qwen3-8b-q4_k_m"]);
        for p in [a, b, c] {
            assert!(p.is_file(), "left where it was");
        }
        assert!(found.skipped.is_empty());
    }

    #[test]
    fn a_projector_is_not_a_model_and_goes_with_the_model_beside_it() {
        let s = Scratch::new();
        s.file("vision/gemma-3-4b-it-Q4_K_M.gguf", &header("gemma3"));
        let named = s.file("vision/mmproj-F16.gguf", &header("clip"));
        // Named like a model, but its header says it is a projector.
        s.file("other/projector-f16.gguf", &header("clip"));
        let found = scan_one(&s.0);
        assert_eq!(found.models.len(), 1);
        assert_eq!(found.models[0].id, "gemma-3-4b-it-q4_k_m");
        assert_eq!(found.models[0].mmproj.as_deref(), Some(named.as_path()));
        assert_eq!(found.skipped.len(), 1);
        assert!(found.skipped[0].path.ends_with("other/projector-f16.gguf"));
    }

    #[test]
    fn a_split_model_is_one_model_and_an_incomplete_one_is_reported() {
        let s = Scratch::new();
        s.file("big/Big-Q4-00002-of-00002.gguf", b"x");
        let first = s.file("big/Big-Q4-00001-of-00002.gguf", &header("qwen3moe"));
        s.file("half/Half-00001-of-00003.gguf", &header("qwen3moe"));
        let found = scan_one(&s.0);
        assert_eq!(found.models.len(), 1);
        assert_eq!(found.models[0].id, "big-q4");
        assert_eq!(found.models[0].path, first);
        assert!(found.skipped[0].reason.contains("1 of its 3 parts"));
    }

    #[test]
    fn the_same_file_name_twice_takes_its_folder_s_name() {
        let s = Scratch::new();
        s.file("bartowski/model.gguf", &header("qwen3"));
        s.file("unsloth/model.gguf", &header("qwen3"));
        s.file("solo.gguf", &header("qwen3"));
        let taken: HashSet<String> = ["solo".to_string()].into();
        let found = scan(std::slice::from_ref(&s.0), &HashSet::new(), &taken);
        let mut ids: Vec<&str> = found.models.iter().map(|m| m.id.as_str()).collect();
        ids.sort();
        let root_name = model_id_of_root(&s.0);
        let mut want = [
            "bartowski-model".to_string(),
            "unsloth-model".to_string(),
            format!("{root_name}-solo"),
        ];
        want.sort();
        assert_eq!(ids, want.iter().map(String::as_str).collect::<Vec<_>>());
    }

    fn model_id_of_root(root: &Path) -> String {
        super::super::store::model_id_for(root.file_name().unwrap().to_str().unwrap())
    }

    #[test]
    fn hidden_folders_and_skipped_folders_are_not_entered() {
        let s = Scratch::new();
        s.file(".cache/partial.gguf", &header("qwen3"));
        s.file("managed/x.gguf", &header("qwen3"));
        s.file("kept.gguf", &header("qwen3"));
        let skip: HashSet<PathBuf> = [s.0.join("managed")].into();
        let found = scan(std::slice::from_ref(&s.0), &skip, &HashSet::new());
        let ids: Vec<&str> = found.models.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, vec!["kept"]);
    }

    #[test]
    fn the_libraries_are_split_on_semicolons_everywhere() {
        assert_eq!(
            parse_libraries(" /a/b ; ;/c "),
            vec![PathBuf::from("/a/b"), PathBuf::from("/c")]
        );
        #[cfg(not(windows))]
        assert_eq!(parse_libraries("/a:/b"), vec![PathBuf::from("/a"), PathBuf::from("/b")]);
    }

    #[test]
    fn a_header_is_read_past_arrays_and_a_non_gguf_is_no_projector() {
        let s = Scratch::new();
        let p = s.file("m.gguf", &header("clip"));
        assert_eq!(gguf_architecture(&p).unwrap().as_deref(), Some("clip"));
        let junk = s.file("junk.gguf", b"not a gguf at all");
        assert!(!is_projector("junk.gguf", &junk));
    }
}

//! Pulling a model from HuggingFace, once, for both callers.
//!
//! The CLI (`eullm pull`) and the API (`POST /api/pull`) need exactly the same
//! sequence — resolve the repo, fetch every shard, fetch the projector beside
//! them, write the manifest — and differ only in how they report it: one
//! prints, the other streams NDJSON. So the sequence lives here and the
//! reporting is a channel.
//!
//! Written this way on purpose rather than implemented twice. Two copies of a
//! prompt builder is how the multimodal path spent a month sending Gemma turn
//! markers to every model, because the fix landed on one copy; two copies of a
//! download path would go the same way the first time a repo layout changes.

use tokio::sync::mpsc;

use crate::models::store::ModelStore;
use crate::registry::{self, HfRef};

/// Derive a filesystem-safe model id from a HuggingFace ref. Uses the repo
/// name (last path segment), lowercased and sanitized like `url_to_model_id`,
/// with the quant appended when one was requested so different quants of the
/// same repo coexist:  `hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M` → `qwen3-8b-gguf-q4_k_m`.
pub fn hf_ref_to_model_id(hf: &registry::HfRef) -> String {
    let repo_name = hf.repo.rsplit('/').next().unwrap_or(&hf.repo);
    let base = match hf.quant.as_deref() {
        Some(q) => format!("{repo_name}-{q}"),
        None => repo_name.to_string(),
    };
    let id: String = base
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
    let id = id.trim_matches('-').to_string();
    if id.is_empty() {
        "model".to_string()
    } else {
        id
    }
}


/// What a pull reports as it goes.
#[derive(Debug, Clone)]
pub enum PullEvent {
    /// A step began. Human-readable, and the `status` field Ollama clients
    /// display.
    Status(String),
    /// Bytes moved for one file. `total` is 0 when the server sent no
    /// Content-Length, which is the same convention the download layer uses.
    Progress {
        file: String,
        completed: u64,
        total: u64,
    },
    /// The model is on disk and usable, stored under this id.
    Done { id: String },
    /// The pull failed and nothing was left behind.
    Failed(String),
}

/// Send and ignore a closed receiver: a client that hangs up mid-download
/// should not turn into an error inside the download.
fn emit(tx: &mpsc::Sender<PullEvent>, ev: PullEvent) {
    let _ = tx.try_send(ev);
}

/// Fetch `hf` into `store`, reporting through `tx`.
///
/// Returns the stored model id on success. Every failure path removes the
/// model directory first: a partial split cannot be loaded, and half a model
/// that `eullm list` shows as present is worse than no model at all.
pub async fn pull_from_huggingface(
    store: &ModelStore,
    hf: &HfRef,
    id: &str,
    tx: mpsc::Sender<PullEvent>,
) -> Result<String, String> {
    emit(
        &tx,
        PullEvent::Status(format!("resolving {} on HuggingFace", hf.repo)),
    );
    let filenames = registry::resolve_hf_gguf(hf)
        .await
        .map_err(|e| format!("could not resolve a GGUF to download: {e}"))?;

    let model_dir = store.model_path(id);
    // Each name is the repo-relative path and can carry a subdirectory when
    // the repo groups quantizations (`UD-Q4_K_XL/Model-…-00001-of-00004.gguf`).
    // The remote path is what the download needs; locally the model already
    // has its own directory, so that prefix is redundant and the file is
    // stored under its bare name. Shards keep their `-NNNNN-of-TOTAL` suffix:
    // only the directory prefix is dropped.
    let leaves: Vec<String> = filenames
        .iter()
        .map(|f| f.rsplit('/').next().unwrap_or(f).to_string())
        .collect();
    // The manifest names the first shard. llama.cpp reads the split count from
    // its header and opens the siblings itself, which is why they all have to
    // land in the same directory.
    let leaf = leaves[0].clone();

    if filenames.len() > 1 {
        emit(
            &tx,
            PullEvent::Status(format!(
                "pulling {} in {} shards",
                hf.repo,
                filenames.len()
            )),
        );
    }

    for (remote, local) in filenames.iter().zip(leaves.iter()) {
        emit(&tx, PullEvent::Status(format!("pulling {local}")));
        let progress = file_progress(&tx, local);
        if let Err(e) = registry::download_from_huggingface(
            &hf.repo,
            remote,
            &model_dir.join(local),
            None,
            Some(progress),
        )
        .await
        {
            let _ = store.delete(id);
            return Err(format!("download failed on {local}: {e}"));
        }
    }

    // A vision repo ships the projector beside the weights, and without it the
    // model loads but cannot see. llama.cpp's own `-hf` fetches both, and a
    // user who has to notice the second file and pass `--mmproj` by hand is
    // being asked to know something the repo layout already says.
    let mmproj_name = match registry::list_hf_ggufs(&hf.repo).await {
        Ok(files) => files.into_iter().find(|f| registry::is_mmproj(f)),
        Err(_) => None,
    };
    let mut mmproj_stored: Option<String> = None;
    if let Some(name) = mmproj_name {
        let projector = name.rsplit('/').next().unwrap_or(&name).to_string();
        emit(
            &tx,
            PullEvent::Status(format!("pulling projector {projector}")),
        );
        let progress = file_progress(&tx, &projector);
        match registry::download_from_huggingface(
            &hf.repo,
            &name,
            &model_dir.join(&projector),
            None,
            Some(progress),
        )
        .await
        {
            Ok(()) => mmproj_stored = Some(projector),
            // The weights are already on disk and usable for text. Losing the
            // projector costs image and audio input, not the model, so it is
            // reported and the pull still succeeds.
            Err(e) => emit(
                &tx,
                PullEvent::Status(format!(
                    "projector download failed ({e}); text still works, \
                     re-run the pull or pass --mmproj"
                )),
            ),
        }
    }

    // Every shard, not just the first: the recorded size is what the model
    // costs on disk, and showing 30 GB for a 111 GB split would be worse than
    // showing nothing.
    let size: u64 = leaves
        .iter()
        .filter_map(|l| std::fs::metadata(model_dir.join(l)).ok())
        .map(|m| m.len())
        .sum();

    store
        .write_external_manifest(id, &leaf, &hf.original, size, mmproj_stored.as_deref())
        .map_err(|e| format!("download succeeded but manifest write failed: {e}"))?;

    emit(&tx, PullEvent::Done { id: id.to_string() });
    Ok(id.to_string())
}

/// A progress callback that forwards into the event channel.
///
/// `try_send` on a bounded channel, so a slow or vanished consumer drops
/// progress ticks instead of stalling the download. Losing a tick costs a
/// smoother bar; blocking a download on a browser that stopped reading costs
/// the download.
fn file_progress(tx: &mpsc::Sender<PullEvent>, file: &str) -> registry::ProgressCallback {
    let tx = tx.clone();
    let file = file.to_string();
    Box::new(move |completed, total| {
        let _ = tx.try_send(PullEvent::Progress {
            file: file.clone(),
            completed,
            total,
        });
    })
}

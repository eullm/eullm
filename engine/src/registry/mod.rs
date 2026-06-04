//! EU Model Registry client.
//!
//! Downloads GGUF model files from EU-hosted registries or HuggingFace.
//! In production, models are served from Hetzner DE / OVH FR.
//! During early development, models are fetched from HuggingFace.

use std::fs;
use std::io::Write;
use std::path::Path;

/// Known EU registry endpoints.
pub const EU_REGISTRIES: &[&str] = &[
    "https://registry.eullm.eu", // Primary (Hetzner, Nuremberg DE) — coming soon
];

/// Where to download a model from.
#[derive(Debug, Clone)]
pub enum DownloadSource {
    /// Direct URL to a GGUF file (EU registry or any HTTPS endpoint).
    Url(String),
    /// HuggingFace repo + filename (e.g., repo="eullm/legal-it-7b-GGUF", file="legal-it-7b-q4_k_m.gguf").
    HuggingFace { repo: String, filename: String },
}

/// Model metadata from the registry.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub size: u64,
    pub digest: String,
    pub format: String,
}

/// Progress callback: (bytes_downloaded, total_bytes).
/// `total_bytes` is 0 if the server didn't send Content-Length.
pub type ProgressCallback = Box<dyn Fn(u64, u64) + Send>;

/// Download a GGUF file from a URL to a local path.
///
/// Shows download progress via the callback. Streams to disk
/// to avoid loading multi-GB files into memory.
///
/// If the download fails at any point, the partial `.part` file is
/// removed before returning the error, so failed pulls don't leave
/// gigabytes of orphaned bytes on disk.
pub async fn download_file(
    url: &str,
    dest: &Path,
    on_progress: Option<ProgressCallback>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Ensure parent directory exists up-front so we can put the .part there.
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp_path = dest.with_extension("gguf.part");

    let result = download_file_inner(url, dest, &tmp_path, on_progress).await;
    if result.is_err() {
        // Best-effort cleanup. Ignore the error: we already have a real error
        // to report, and not being able to delete a transient file shouldn't
        // mask it.
        let _ = fs::remove_file(&tmp_path);
    }
    result
}

async fn download_file_inner(
    url: &str,
    dest: &Path,
    tmp_path: &Path,
    on_progress: Option<ProgressCallback>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let client = reqwest::Client::builder()
        .user_agent("eullm/0.1.0")
        .build()?;

    let response = client.get(url).send().await?;

    if !response.status().is_success() {
        return Err(format!(
            "Download failed: HTTP {} from {}",
            response.status(),
            url
        )
        .into());
    }

    let total = response.content_length().unwrap_or(0);

    // Stream to a temporary file, then rename (atomic-ish)
    let mut file = fs::File::create(tmp_path)?;
    let mut downloaded: u64 = 0;

    let mut stream = response.bytes_stream();
    use futures_util::StreamExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk)?;
        downloaded += chunk.len() as u64;

        if let Some(ref cb) = on_progress {
            cb(downloaded, total);
        }
    }

    file.flush()?;
    drop(file);

    // Rename temp file to final path
    fs::rename(tmp_path, dest)?;

    Ok(())
}

/// Download a GGUF from HuggingFace Hub.
///
/// Uses the HuggingFace CDN: `https://huggingface.co/{repo}/resolve/main/{filename}`
pub async fn download_from_huggingface(
    repo: &str,
    filename: &str,
    dest: &Path,
    on_progress: Option<ProgressCallback>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        repo, filename
    );
    tracing::info!("Downloading from HuggingFace: {url}");
    download_file(&url, dest, on_progress).await
}

/// Download a model from the EU registry.
///
/// Tries the EU registry first, falls back to HuggingFace if configured.
pub async fn pull_model(
    source: &DownloadSource,
    dest: &Path,
    on_progress: Option<ProgressCallback>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    match source {
        DownloadSource::Url(url) => {
            tracing::info!("Downloading from: {url}");
            download_file(url, dest, on_progress).await
        }
        DownloadSource::HuggingFace { repo, filename } => {
            download_from_huggingface(repo, filename, dest, on_progress).await
        }
    }
}

/// Query the EU registry API for available models.
///
/// Falls back to the built-in catalog if the registry is not reachable.
pub async fn list_remote() -> Result<Vec<ModelInfo>, Box<dyn std::error::Error + Send + Sync>> {
    let client = reqwest::Client::builder()
        .user_agent("eullm/0.1.0")
        .timeout(std::time::Duration::from_secs(5))
        .build()?;

    for registry in EU_REGISTRIES {
        let url = format!("{registry}/v1/models");
        match client.get(&url).send().await {
            Ok(response) if response.status().is_success() => {
                #[derive(serde::Deserialize)]
                struct ModelsResponse {
                    models: Vec<ModelInfo>,
                }
                if let Ok(body) = response.json::<ModelsResponse>().await {
                    return Ok(body.models);
                }
            }
            _ => {
                tracing::debug!("Registry {registry} not reachable, trying next...");
            }
        }
    }

    // Registry not available — return empty (caller should use local catalog)
    tracing::info!("EU registry not reachable. Using local catalog.");
    Ok(vec![])
}

/// Format download progress as a human-readable string.
pub fn format_progress(downloaded: u64, total: u64) -> String {
    let dl = format_size(downloaded);
    if total > 0 {
        let pct = (downloaded as f64 / total as f64 * 100.0) as u32;
        let tot = format_size(total);
        format!("{dl} / {tot} ({pct}%)")
    } else {
        dl
    }
}

fn format_size(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1_000_000_000.0)
    } else if bytes >= 1_000_000 {
        format!("{:.1} MB", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.1} KB", bytes as f64 / 1_000.0)
    } else {
        format!("{bytes} B")
    }
}

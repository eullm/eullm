//! EU Model Registry client.
//!
//! Downloads GGUF model files from EU-hosted registries or HuggingFace.
//! In production, models are served from Hetzner DE / OVH FR.
//! During early development, models are fetched from HuggingFace.

use std::fs;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Progress callback: (bytes_downloaded, total_bytes).
/// `total_bytes` is 0 if the server didn't send Content-Length.
pub type ProgressCallback = Box<dyn Fn(u64, u64) + Send>;

/// Size of each byte range fetched by a parallel download worker.
const PARALLEL_CHUNK_SIZE: u64 = 16 * 1024 * 1024;

/// How many times a single chunk is retried (with exponential backoff) before
/// the whole download is considered failed. A retry re-fetches only that chunk,
/// so a transient drop (e.g. a Starlink satellite handover) costs one chunk,
/// not the entire file.
const CHUNK_MAX_ATTEMPTS: u32 = 5;

/// Per-request wall-clock cap. Bounds a stalled connection so it errors and the
/// chunk is retried instead of hanging forever. Generous enough for one
/// `PARALLEL_CHUNK_SIZE` chunk on a slow link.
const CHUNK_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// Number of concurrent range requests for a parallel download.
///
/// A single TCP stream cannot saturate a high-latency link (the
/// bandwidth-delay product is large on e.g. Starlink), so we fan out. Default
/// 8 — the same ballpark `hf_transfer`/`aria2` use, and well within what the
/// HuggingFace CDN tolerates. Override with `EULLM_DOWNLOAD_CONNECTIONS`
/// (clamped to 1..=16; 1 forces the legacy single-stream path).
pub fn default_connections() -> usize {
    std::env::var("EULLM_DOWNLOAD_CONNECTIONS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(8)
        .clamp(1, 16)
}

/// Download a GGUF file from a URL to a local path.
///
/// Uses parallel HTTP Range requests when the server supports them (most CDNs,
/// including HuggingFace, do), which both saturates high-latency links and
/// survives transient drops by retrying individual chunks. Falls back to a
/// single streaming GET when ranges or a content length aren't available.
///
/// Shows download progress via the callback. Streams to disk to avoid loading
/// multi-GB files into memory.
///
/// If the download fails at any point, the partial `.part` file is removed
/// before returning the error, so failed pulls don't leave gigabytes of
/// orphaned bytes on disk.
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

    let result = download_file_smart(url, dest, &tmp_path, on_progress).await;
    if result.is_err() {
        // Best-effort cleanup. Ignore the error: we already have a real error
        // to report, and not being able to delete a transient file shouldn't
        // mask it.
        let _ = fs::remove_file(&tmp_path);
    }
    result
}

/// Probe the URL for Range support and either fan out into parallel chunk
/// workers or fall back to a single streaming download.
async fn download_file_smart(
    url: &str,
    dest: &Path,
    tmp_path: &Path,
    on_progress: Option<ProgressCallback>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let client = reqwest::Client::builder()
        .user_agent("eullm/0.1.0")
        .connect_timeout(Duration::from_secs(15))
        .build()?;

    // Probe: a one-byte ranged GET tells us, in a single round-trip, both
    // whether ranges are supported (206 Partial Content) and the total size
    // (the `/<total>` tail of Content-Range).
    let (supports_range, total) = probe_range(&client, url).await;

    let connections = default_connections();

    if !supports_range || total == 0 || connections <= 1 {
        // Legacy path: one connection, sequential stream.
        return download_stream(&client, url, tmp_path, dest, on_progress).await;
    }

    // Pre-allocate the destination so chunk workers can write at their offsets.
    let file = fs::File::create(tmp_path)?;
    file.set_len(total)?;
    drop(file);

    let ranges = split_ranges(total, PARALLEL_CHUNK_SIZE);
    let downloaded = Arc::new(AtomicU64::new(0));

    use futures_util::stream::{self, StreamExt};
    let mut workers = stream::iter(ranges.into_iter().map(|(s, e)| {
        let client = client.clone();
        let url = url.to_string();
        let tmp_path = tmp_path.to_path_buf();
        let downloaded = Arc::clone(&downloaded);
        async move { fetch_chunk(&client, &url, &tmp_path, s, e, &downloaded).await }
    }))
    .buffer_unordered(connections);

    while let Some(res) = workers.next().await {
        // Propagate the first chunk that fails even after its own retries; the
        // caller removes the partial file.
        res?;
        if let Some(ref cb) = on_progress {
            cb(downloaded.load(Ordering::Relaxed), total);
        }
    }
    drop(workers);

    fs::rename(tmp_path, dest)?;
    Ok(())
}

/// Split a total byte count into inclusive `[start, end]` ranges of at most
/// `chunk` bytes each. The last range covers whatever remains.
fn split_ranges(total: u64, chunk: u64) -> Vec<(u64, u64)> {
    let mut ranges = Vec::new();
    let mut start = 0u64;
    while start < total {
        let end = (start + chunk - 1).min(total - 1);
        ranges.push((start, end));
        start = end + 1;
    }
    ranges
}

/// Send a `bytes=0-0` ranged GET. Returns `(supports_range, total_bytes)`.
/// `supports_range` is true only on a 206 with a parseable `Content-Range`.
async fn probe_range(client: &reqwest::Client, url: &str) -> (bool, u64) {
    let resp = match client
        .get(url)
        .header(reqwest::header::RANGE, "bytes=0-0")
        .timeout(CHUNK_REQUEST_TIMEOUT)
        .send()
        .await
    {
        Ok(r) => r,
        Err(_) => return (false, 0),
    };
    if resp.status() != reqwest::StatusCode::PARTIAL_CONTENT {
        return (false, 0);
    }
    let total = resp
        .headers()
        .get(reqwest::header::CONTENT_RANGE)
        .and_then(|v| v.to_str().ok())
        // "bytes 0-0/123456" → take the part after the final '/'.
        .and_then(|s| s.rsplit('/').next())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0);
    (true, total)
}

/// Fetch one byte range with retries. On success the chunk's byte count is
/// added to `downloaded` exactly once (a failed attempt adds nothing, so the
/// progress counter never overshoots).
async fn fetch_chunk(
    client: &reqwest::Client,
    url: &str,
    tmp_path: &Path,
    start: u64,
    end: u64,
    downloaded: &AtomicU64,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut attempt = 0u32;
    loop {
        match fetch_chunk_once(client, url, tmp_path, start, end).await {
            Ok(()) => {
                downloaded.fetch_add(end - start + 1, Ordering::Relaxed);
                return Ok(());
            }
            Err(e) => {
                attempt += 1;
                if attempt >= CHUNK_MAX_ATTEMPTS {
                    return Err(format!(
                        "range {start}-{end} failed after {attempt} attempts: {e}"
                    )
                    .into());
                }
                // Exponential backoff: 1s, 2s, 4s, 8s.
                tokio::time::sleep(Duration::from_secs(1u64 << (attempt - 1))).await;
            }
        }
    }
}

/// One attempt at fetching a byte range and writing it at its file offset.
async fn fetch_chunk_once(
    client: &reqwest::Client,
    url: &str,
    tmp_path: &Path,
    start: u64,
    end: u64,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let resp = client
        .get(url)
        .header(reqwest::header::RANGE, format!("bytes={start}-{end}"))
        .timeout(CHUNK_REQUEST_TIMEOUT)
        .send()
        .await?;

    let status = resp.status();
    if status != reqwest::StatusCode::PARTIAL_CONTENT && !status.is_success() {
        return Err(format!("HTTP {status} for range {start}-{end}").into());
    }

    // Each worker opens its own handle and seeks to the chunk's offset; within
    // a chunk the body arrives in order, so a plain sequential write is correct.
    let mut file = fs::OpenOptions::new().write(true).open(tmp_path)?;
    file.seek(SeekFrom::Start(start))?;

    use futures_util::StreamExt;
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk)?;
    }
    file.flush()?;
    Ok(())
}

/// Single-connection streaming download (the fallback when Range isn't
/// supported, the size is unknown, or `EULLM_DOWNLOAD_CONNECTIONS=1`).
async fn download_stream(
    client: &reqwest::Client,
    url: &str,
    tmp_path: &Path,
    dest: &Path,
    on_progress: Option<ProgressCallback>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
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

    let mut file = fs::File::create(tmp_path)?;
    let mut downloaded: u64 = 0;

    use futures_util::StreamExt;
    let mut stream = response.bytes_stream();
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

/// A parsed HuggingFace repo reference, e.g. `hf.co/owner/repo:Q4_K_M`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfRef {
    /// `owner/repo` — the path used to address the model on the Hub.
    pub repo: String,
    /// Optional quantization token the user asked for (e.g. `Q4_K_M`),
    /// preserved in its original case. Matched case-insensitively later.
    pub quant: Option<String>,
    /// The original reference string, used as the recorded `source`.
    pub original: String,
}

/// Parse a HuggingFace repo shorthand into an [`HfRef`].
///
/// Accepted forms (case-insensitive prefix):
///   - `hf.co/<owner>/<repo>` and `hf.co/<owner>/<repo>:<quant>`
///   - `huggingface.co/<owner>/<repo>[:<quant>]`
///   - `hf:<owner>/<repo>[:<quant>]`
///
/// Returns `None` if `s` is not a recognised HF shorthand. A direct
/// `https://.../*.gguf` URL is intentionally NOT an HF ref — those go down
/// the existing direct-download path untouched.
pub fn parse_hf_ref(s: &str) -> Option<HfRef> {
    let trimmed = s.trim();
    let lower = trimmed.to_lowercase();

    // Strip the recognised prefix, keeping the remainder in its original case
    // (owner/repo paths and quant tokens are case-sensitive on the Hub).
    let rest = if let Some(stripped) = lower.strip_prefix("hf.co/") {
        &trimmed[trimmed.len() - stripped.len()..]
    } else if let Some(stripped) = lower.strip_prefix("huggingface.co/") {
        &trimmed[trimmed.len() - stripped.len()..]
    } else if let Some(stripped) = lower.strip_prefix("hf://") {
        &trimmed[trimmed.len() - stripped.len()..]
    } else {
        let stripped = lower.strip_prefix("hf:")?;
        &trimmed[trimmed.len() - stripped.len()..]
    };

    // Split off an optional ":<quant>" suffix. The repo itself is `owner/repo`
    // and must contain exactly one slash with non-empty segments.
    let (path, quant) = match rest.split_once(':') {
        Some((p, q)) if !q.is_empty() => (p, Some(q.to_string())),
        Some((p, _)) => (p, None),
        None => (rest, None),
    };

    let path = path.trim_matches('/');
    let mut segments = path.split('/');
    let owner = segments.next().filter(|s| !s.is_empty())?;
    let repo = segments.next().filter(|s| !s.is_empty())?;
    // Reject extra path segments (e.g. a resolve URL): we only address repos.
    if segments.next().is_some() {
        return None;
    }

    Some(HfRef {
        repo: format!("{owner}/{repo}"),
        quant,
        original: trimmed.to_string(),
    })
}

/// Choose the GGUF filename to download from the list of `.gguf` siblings in
/// a HuggingFace repo.
///
/// `requested_quant` is matched case-insensitively as a substring of the
/// filename. With no requested quant we prefer `Q4_K_M`, then `Q4_0`, then
/// fall back to the first gguf.
///
/// Returns `Err(message)` when the choice is ambiguous (multiple candidates
/// for the requested quant) or sharded (`*-00001-of-0000N.gguf`); the message
/// lists the available filenames so the caller can print it and ask the user
/// to re-run with an explicit `:<quant>`.
fn select_gguf(ggufs: &[String], requested_quant: Option<&str>) -> Result<String, String> {
    if ggufs.is_empty() {
        return Err("no .gguf files found in this HuggingFace repo".to_string());
    }

    // Sharded multi-file ggufs (e.g. `model-00001-of-00003.gguf`) cannot be
    // loaded from a single file — refuse to guess and let the user pick.
    let is_shard = |name: &str| {
        let lower = name.to_lowercase();
        lower.contains("-of-") && lower.contains(".gguf")
    };

    let candidates: Vec<&String> = match requested_quant {
        Some(q) => {
            let q_lower = q.to_lowercase();
            ggufs
                .iter()
                .filter(|f| f.to_lowercase().contains(&q_lower))
                .collect()
        }
        None => ggufs.iter().collect(),
    };

    if candidates.is_empty() {
        return Err(format!(
            "no .gguf file matches quant '{}'. Available files:\n{}",
            requested_quant.unwrap_or(""),
            list_for_error(ggufs),
        ));
    }

    // If a quant was requested and it disambiguates to a single file, use it
    // even if other (non-matching) shards exist.
    if requested_quant.is_some() {
        let non_shard: Vec<&&String> = candidates.iter().filter(|f| !is_shard(f.as_str())).collect();
        if non_shard.len() == 1 {
            return Ok(non_shard[0].to_string());
        }
        if candidates.len() == 1 {
            // Single match, even if it looks like a shard — let llama.cpp try.
            return Ok(candidates[0].to_string());
        }
        return Err(format!(
            "multiple .gguf files match quant '{}'. Re-run with a more specific :<quant>. Available files:\n{}",
            requested_quant.unwrap_or(""),
            list_for_error(ggufs),
        ));
    }

    // No quant requested: refuse if the repo only contains shards (we'd have
    // to guess across them), otherwise prefer the standard quants.
    let single_file: Vec<&String> = ggufs.iter().filter(|f| !is_shard(f.as_str())).collect();
    if single_file.is_empty() {
        return Err(format!(
            "this repo only contains sharded .gguf files; pass an explicit :<quant>. Available files:\n{}",
            list_for_error(ggufs),
        ));
    }

    let prefer = |needle: &str| {
        single_file
            .iter()
            .find(|f| f.to_lowercase().contains(needle))
            .map(|f| f.to_string())
    };
    Ok(prefer("q4_k_m")
        .or_else(|| prefer("q4_0"))
        .unwrap_or_else(|| single_file[0].to_string()))
}

/// Render a bullet list of filenames for an error message.
fn list_for_error(files: &[String]) -> String {
    files
        .iter()
        .map(|f| format!("  - {f}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Query the HuggingFace model API and return the list of `.gguf` filenames
/// in the repo's `siblings` array.
///
/// Hits `https://huggingface.co/api/models/{repo}` and parses the JSON. Only
/// the `siblings[].rfilename` values are read; everything else is ignored.
pub async fn list_hf_ggufs(
    repo: &str,
) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
    let url = format!("https://huggingface.co/api/models/{repo}");
    let client = reqwest::Client::builder()
        .user_agent(concat!("eullm/", env!("CARGO_PKG_VERSION")))
        .timeout(std::time::Duration::from_secs(15))
        .build()?;

    let response = client.get(&url).send().await?;
    if !response.status().is_success() {
        return Err(format!(
            "HuggingFace API returned HTTP {} for {repo}",
            response.status()
        )
        .into());
    }

    let body: serde_json::Value = response.json().await?;
    let mut ggufs = Vec::new();
    if let Some(siblings) = body.get("siblings").and_then(|s| s.as_array()) {
        for sib in siblings {
            if let Some(name) = sib.get("rfilename").and_then(|n| n.as_str())
                && name.to_lowercase().ends_with(".gguf")
            {
                ggufs.push(name.to_string());
            }
        }
    }
    Ok(ggufs)
}

/// Resolve a HuggingFace ref to a single GGUF filename to download.
///
/// Combines [`list_hf_ggufs`] and [`select_gguf`]. The returned string is the
/// `rfilename` to fetch from `{repo}/resolve/main/{filename}`.
pub async fn resolve_hf_gguf(
    hf: &HfRef,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let ggufs = list_hf_ggufs(&hf.repo).await?;
    select_gguf(&ggufs, hf.quant.as_deref()).map_err(|e| e.into())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_ranges_exact_multiple() {
        // 30 bytes in 10-byte chunks → three full ranges, no gaps/overlap.
        let r = split_ranges(30, 10);
        assert_eq!(r, vec![(0, 9), (10, 19), (20, 29)]);
    }

    #[test]
    fn split_ranges_with_remainder() {
        // Last range is short and ends exactly at total-1.
        let r = split_ranges(25, 10);
        assert_eq!(r, vec![(0, 9), (10, 19), (20, 24)]);
        // Ranges tile the whole file contiguously.
        let covered: u64 = r.iter().map(|(s, e)| e - s + 1).sum();
        assert_eq!(covered, 25);
    }

    #[test]
    fn split_ranges_smaller_than_chunk() {
        assert_eq!(split_ranges(5, 16), vec![(0, 4)]);
    }

    #[test]
    fn split_ranges_empty() {
        assert!(split_ranges(0, 16).is_empty());
    }

    #[test]
    fn parses_hf_co_prefix() {
        let r = parse_hf_ref("hf.co/Qwen/Qwen3-8B-GGUF").unwrap();
        assert_eq!(r.repo, "Qwen/Qwen3-8B-GGUF");
        assert_eq!(r.quant, None);
    }

    #[test]
    fn parses_quant_suffix() {
        let r = parse_hf_ref("hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M").unwrap();
        assert_eq!(r.repo, "Qwen/Qwen3-8B-GGUF");
        assert_eq!(r.quant.as_deref(), Some("Q4_K_M"));
    }

    #[test]
    fn parses_huggingface_co_and_hf_colon() {
        assert_eq!(
            parse_hf_ref("huggingface.co/owner/repo").unwrap().repo,
            "owner/repo"
        );
        assert_eq!(parse_hf_ref("hf:owner/repo").unwrap().repo, "owner/repo");
        assert_eq!(parse_hf_ref("hf://owner/repo").unwrap().repo, "owner/repo");
    }

    #[test]
    fn rejects_non_hf_and_plain_urls() {
        assert!(parse_hf_ref("https://example.com/model.gguf").is_none());
        assert!(parse_hf_ref("qwen3-8b").is_none());
        // Missing repo segment.
        assert!(parse_hf_ref("hf.co/owner").is_none());
        // Extra path segments (e.g. a resolve URL) are not a repo ref.
        assert!(parse_hf_ref("hf.co/owner/repo/resolve/main/x.gguf").is_none());
    }

    #[test]
    fn select_prefers_q4_k_m_by_default() {
        let files = vec![
            "model-Q8_0.gguf".to_string(),
            "model-Q4_K_M.gguf".to_string(),
            "model-Q4_0.gguf".to_string(),
        ];
        assert_eq!(select_gguf(&files, None).unwrap(), "model-Q4_K_M.gguf");
    }

    #[test]
    fn select_matches_requested_quant_case_insensitive() {
        let files = vec![
            "model-Q8_0.gguf".to_string(),
            "model-Q4_K_M.gguf".to_string(),
        ];
        assert_eq!(
            select_gguf(&files, Some("q8_0")).unwrap(),
            "model-Q8_0.gguf"
        );
    }

    #[test]
    fn select_refuses_shards_without_quant() {
        let files = vec![
            "model-00001-of-00003.gguf".to_string(),
            "model-00002-of-00003.gguf".to_string(),
            "model-00003-of-00003.gguf".to_string(),
        ];
        assert!(select_gguf(&files, None).is_err());
    }

    #[test]
    fn select_errors_on_unknown_quant() {
        let files = vec!["model-Q4_K_M.gguf".to_string()];
        assert!(select_gguf(&files, Some("Q2_K")).is_err());
    }
}

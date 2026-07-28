//! EU Model Registry client.
//!
//! Downloads GGUF model files from EU-hosted registries or HuggingFace.
//! In production, models are served from Hetzner DE / OVH FR.
//! During early development, models are fetched from HuggingFace.

use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use sha2::{Digest, Sha256};

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
/// orphaned bytes on disk. This includes a SHA-256 mismatch against
/// `expected_sha256` (when given) — verification runs on the `.part` file
/// before it's renamed into place, so a corrupted or tampered download is
/// never left at `dest` under any name. Pass `None` when no digest is known
/// (e.g. an arbitrary user-supplied URL) — the file is downloaded but not
/// verified, with a warning logged.
pub async fn download_file(
    url: &str,
    dest: &Path,
    expected_sha256: Option<&str>,
    on_progress: Option<ProgressCallback>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Ensure parent directory exists up-front so we can put the .part there.
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp_path = dest.with_extension("gguf.part");

    let result = download_file_smart(url, dest, &tmp_path, expected_sha256, on_progress).await;
    if result.is_err() {
        // Best-effort cleanup. Ignore the error: we already have a real error
        // to report, and not being able to delete a transient file shouldn't
        // mask it.
        let _ = fs::remove_file(&tmp_path);
    }
    result
}

/// Hash `path` with SHA-256, reading in fixed-size chunks so a multi-GB file
/// is never loaded into memory at once.
fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex_encode(&hasher.finalize()))
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Verify `path`'s SHA-256 against `expected` (accepts either a bare hex
/// digest or a `sha256:`-prefixed one, matching the catalog's format).
/// Does nothing when `expected` is `None` or empty — most download call
/// sites (arbitrary URLs, off-catalog pulls) have no digest to check.
fn verify_digest_if_present(
    path: &Path,
    expected: Option<&str>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let Some(expected) = expected.map(str::trim).filter(|s| !s.is_empty()) else {
        tracing::warn!(
            "No integrity digest recorded for {} — downloaded without SHA-256 verification",
            path.display()
        );
        return Ok(());
    };
    let expected = expected.strip_prefix("sha256:").unwrap_or(expected);

    let actual = sha256_file(path)?;
    if !actual.eq_ignore_ascii_case(expected) {
        return Err(format!(
            "SHA-256 mismatch for {}: expected {expected}, got {actual}. \
             The download may be corrupted or the source may have changed.",
            path.display()
        )
        .into());
    }
    Ok(())
}

/// Probe the URL for Range support and either fan out into parallel chunk
/// workers or fall back to a single streaming download.
async fn download_file_smart(
    url: &str,
    dest: &Path,
    tmp_path: &Path,
    expected_sha256: Option<&str>,
    on_progress: Option<ProgressCallback>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let client = reqwest::Client::builder()
        .user_agent(concat!("eullm/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(15))
        .build()?;

    // Probe: a one-byte ranged GET tells us, in a single round-trip, both
    // whether ranges are supported (206 Partial Content) and the total size
    // (the `/<total>` tail of Content-Range).
    let (supports_range, total) = probe_range(&client, url).await;

    let connections = default_connections();

    if !supports_range || total == 0 || connections <= 1 {
        // Legacy path: one connection, sequential stream.
        return download_stream(&client, url, tmp_path, dest, expected_sha256, on_progress).await;
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

    verify_digest_if_present(tmp_path, expected_sha256)?;
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
    expected_sha256: Option<&str>,
    on_progress: Option<ProgressCallback>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let response = client.get(url).send().await?;

    if !response.status().is_success() {
        return Err(format!("Download failed: HTTP {} from {}", response.status(), url).into());
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

    verify_digest_if_present(tmp_path, expected_sha256)?;
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
    expected_sha256: Option<&str>,
    on_progress: Option<ProgressCallback>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let url = format!("https://huggingface.co/{}/resolve/main/{}", repo, filename);
    tracing::info!("Downloading from HuggingFace: {url}");
    download_file(&url, dest, expected_sha256, on_progress).await
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
/// Whether a repo filename is a multimodal projector rather than model weights.
///
/// Published vision repos name it `mmproj-<something>.gguf`, sometimes with a
/// directory prefix, which is the convention llama.cpp itself relies on.
pub fn is_mmproj(name: &str) -> bool {
    name.rsplit('/')
        .next()
        .unwrap_or(name)
        .to_lowercase()
        .starts_with("mmproj")
}

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

    // A projector is a `.gguf` in the same repo but never the model, and it
    // has to come out of the candidate set before anything else looks at it.
    // Left in, it makes a vision repo ambiguous for a plain pull, and
    // `:F16` on such a repo can select `mmproj-F16.gguf` as the weights,
    // which then fails to load with an error about the file rather than
    // about the choice. It is downloaded separately, alongside whatever
    // model is picked.
    let ggufs: Vec<String> = ggufs.iter().filter(|f| !is_mmproj(f)).cloned().collect();
    let ggufs = &ggufs[..];
    if ggufs.is_empty() {
        return Err(
            "this HuggingFace repo contains only a projector (mmproj), no model weights"
                .to_string(),
        );
    }

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
        let non_shard: Vec<&&String> = candidates
            .iter()
            .filter(|f| !is_shard(f.as_str()))
            .collect();
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
            let Some(name) = sib.get("rfilename").and_then(|n| n.as_str()) else {
                continue;
            };
            if !name.to_lowercase().ends_with(".gguf") {
                continue;
            }
            // The suffix used to be the only check, and this value becomes a
            // path component in `cmd_pull_hf` (`model_dir.join(&filename)`).
            // A repo is free to declare whatever `rfilename` it likes, so
            // anything that isn't a single safe filename is dropped here, at
            // the boundary, rather than being validated at each use site.
            if !crate::models::store::is_safe_filename(name) {
                tracing::warn!(
                    "Ignoring unsafe filename from the HuggingFace API for {repo}: {}",
                    crate::audit::sanitize_for_log(name),
                );
                continue;
            }
            ggufs.push(name.to_string());
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

    fn write_temp_file(contents: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("eullm-sha256-test-{}", uuid::Uuid::new_v4()));
        fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn sha256_file_matches_a_known_vector() {
        // sha256("") — the empty-string test vector everyone can check by hand.
        let path = write_temp_file(b"");
        let hash = sha256_file(&path).unwrap();
        fs::remove_file(&path).ok();
        assert_eq!(
            hash,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn verify_digest_if_present_accepts_a_match() {
        let path = write_temp_file(b"hello eullm");
        let hash = sha256_file(&path).unwrap();
        let result = verify_digest_if_present(&path, Some(&format!("sha256:{hash}")));
        fs::remove_file(&path).ok();
        assert!(result.is_ok());
    }

    #[test]
    fn verify_digest_if_present_rejects_a_mismatch() {
        let path = write_temp_file(b"hello eullm");
        let wrong_digest = format!("sha256:{}", "0".repeat(64));
        let result = verify_digest_if_present(&path, Some(&wrong_digest));
        fs::remove_file(&path).ok();
        assert!(result.is_err());
    }

    #[test]
    fn verify_digest_if_present_skips_when_none_or_empty() {
        let path = write_temp_file(b"hello eullm");
        assert!(verify_digest_if_present(&path, None).is_ok());
        assert!(verify_digest_if_present(&path, Some("")).is_ok());
        fs::remove_file(&path).ok();
    }

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

    // A vision repo carries the projector next to the weights. It is a .gguf
    // like any other, so before it was excluded a plain pull saw two
    // candidates and refused as ambiguous, and `:F16` on such a repo selected
    // `mmproj-F16.gguf` as the model — which then failed to load with an
    // error about the file rather than about the choice.
    #[test]
    fn a_projector_is_never_chosen_as_the_model() {
        let files = vec![
            "gemma-4-12b-it-Q4_K_M.gguf".to_string(),
            "mmproj-F16.gguf".to_string(),
        ];
        // Without the exclusion this repo had two candidates and a plain pull
        // could land on the projector.
        assert_eq!(
            select_gguf(&files, None).expect("one model, one projector"),
            "gemma-4-12b-it-Q4_K_M.gguf"
        );
        // `:F16` used to match `mmproj-F16.gguf` and download it as the
        // weights, which then failed to load with an error about the file
        // rather than about the choice. Now it reports that no *model*
        // matches, and lists what there is.
        let err = select_gguf(&files, Some("F16")).expect_err("no F16 weights in this repo");
        assert!(err.contains("no .gguf file matches quant"), "{err}");
        assert!(
            !err.contains("mmproj"),
            "the projector must not be offered as an alternative: {err}"
        );
    }

    #[test]
    fn a_repo_with_only_a_projector_is_refused_by_name() {
        let files = vec!["mmproj-F16.gguf".to_string()];
        let err = select_gguf(&files, None).expect_err("no weights to pick");
        assert!(err.contains("projector"), "unhelpful message: {err}");
    }

    #[test]
    fn is_mmproj_matches_the_published_naming() {
        assert!(is_mmproj("mmproj-F16.gguf"));
        assert!(is_mmproj("MMPROJ-model-f32.gguf"));
        assert!(is_mmproj("some/dir/mmproj-F16.gguf"));
        assert!(!is_mmproj("gemma-4-12b-it-Q4_K_M.gguf"));
    }

    #[test]
    fn select_errors_on_unknown_quant() {
        let files = vec!["model-Q4_K_M.gguf".to_string()];
        assert!(select_gguf(&files, Some("Q2_K")).is_err());
    }
}

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
    // Through the store's helper, for its diagnosis: a bare `create_dir_all`
    // reports EEXIST ("File exists (os error 17)") when a path component is
    // a symlink to an unmounted volume, which reads as "already downloaded"
    // and sent a user hunting for a file that was never there.
    if let Some(parent) = dest.parent() {
        crate::models::store::create_model_dir(parent)?;
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

/// One split GGUF: every shard of `stem-NNNNN-of-TOTAL.gguf`, in order.
#[derive(Debug)]
struct ShardSet {
    /// Everything before `-NNNNN-of-TOTAL.gguf`, which identifies the split.
    stem: String,
    /// How many shards the filenames themselves claim exist.
    total: u32,
    /// The shards present in the repo listing, ordered by index.
    files: Vec<String>,
}

impl ShardSet {
    fn is_complete(&self) -> bool {
        self.files.len() as u32 == self.total
    }
}

/// Decompose `…/Model-Q4_K_XL-00002-of-00004.gguf` into its stem, index and
/// total. `None` for anything that is not a split shard.
fn parse_shard(name: &str) -> Option<(&str, u32, u32)> {
    let base = name.get(..name.len().checked_sub(5)?)?;
    if !name[name.len() - 5..].eq_ignore_ascii_case(".gguf") {
        return None;
    }
    let (rest, total) = base.rsplit_once("-of-")?;
    let (stem, index) = rest.rsplit_once('-')?;
    let total: u32 = total.parse().ok()?;
    let index: u32 = index.parse().ok()?;
    // A zero index or a shard numbered past the total is not a split we
    // understand; treat the name as an ordinary file rather than guess.
    if index == 0 || total == 0 || index > total {
        return None;
    }
    Some((stem, index, total))
}

/// Split a candidate list into standalone files and the split GGUFs they
/// belong to. Shard order follows the index in the filename, not the order
/// the API happened to list them in.
fn group_shards(files: &[&String]) -> (Vec<String>, Vec<ShardSet>) {
    let mut singles = Vec::new();
    // Insertion-ordered so the result does not depend on hash iteration order,
    // which would make an ambiguity error vary between runs.
    let mut sets: Vec<(ShardSet, Vec<(u32, String)>)> = Vec::new();
    for f in files {
        let Some((stem, index, total)) = parse_shard(f) else {
            singles.push((*f).clone());
            continue;
        };
        match sets.iter_mut().find(|(s, _)| s.stem == stem) {
            Some((_, indexed)) => indexed.push((index, (*f).clone())),
            None => sets.push((
                ShardSet {
                    stem: stem.to_string(),
                    total,
                    files: Vec::new(),
                },
                vec![(index, (*f).clone())],
            )),
        }
    }
    let sets = sets
        .into_iter()
        .map(|(mut set, mut indexed)| {
            indexed.sort_by_key(|(i, _)| *i);
            indexed.dedup_by_key(|(i, _)| *i);
            set.files = indexed.into_iter().map(|(_, f)| f).collect();
            set
        })
        .collect();
    (singles, sets)
}

/// Every file that has to be downloaded for one model, in the order to
/// fetch them.
///
/// A split GGUF is `N` files and llama.cpp opens the rest from the first, but
/// only once they are all on disk — so returning one name was never enough.
/// Until 0.7.5-rc7 this returned a single `String`, which is why pulling a
/// large quantization from a repo that ships it split failed: with four
/// shards matching the requested quant, none of the single-file branches
/// applied and the pull ended on `multiple .gguf files match quant`.
fn select_gguf(ggufs: &[String], requested_quant: Option<&str>) -> Result<Vec<String>, String> {
    if ggufs.is_empty() {
        return Err("no .gguf files found in this HuggingFace repo".to_string());
    }

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

    let (singles, sets) = group_shards(&candidates);

    // Refusing to download a split we know to be incomplete beats downloading
    // three quarters of a model and failing at load with an error about the
    // file rather than about the repo.
    let take = |set: ShardSet| -> Result<Vec<String>, String> {
        if set.is_complete() {
            return Ok(set.files);
        }
        Err(format!(
            "'{}' is split into {} shards but the repo lists only {}. \
             The upload looks incomplete; nothing was downloaded.",
            set.stem,
            set.total,
            set.files.len(),
        ))
    };

    if requested_quant.is_some() {
        // A quant that disambiguates to a single standalone file wins even
        // when other (non-matching) shards are present in the repo.
        if singles.len() == 1 {
            return Ok(vec![singles.into_iter().next().unwrap()]);
        }
        if singles.is_empty() && sets.len() == 1 {
            return take(sets.into_iter().next().unwrap());
        }
        if candidates.len() == 1 {
            return Ok(vec![candidates[0].to_string()]);
        }
        return Err(format!(
            "multiple .gguf files match quant '{}'. Re-run with a more specific :<quant>. Available files:\n{}",
            requested_quant.unwrap_or(""),
            list_for_error(ggufs),
        ));
    }

    // No quant requested: prefer a standalone file, and fall back to a split
    // when it is the only thing the repo offers. Guessing across *several*
    // splits is still refused — that is a real choice for the user to make.
    let prefer = |needle: &str| {
        singles
            .iter()
            .find(|f| f.to_lowercase().contains(needle))
            .cloned()
    };
    if !singles.is_empty() {
        return Ok(vec![
            prefer("q4_k_m")
                .or_else(|| prefer("q4_0"))
                .unwrap_or_else(|| singles[0].clone()),
        ]);
    }
    if sets.len() == 1 {
        return take(sets.into_iter().next().unwrap());
    }
    Err(format!(
        "this repo contains several sharded .gguf models; pass an explicit :<quant>. Available files:\n{}",
        list_for_error(ggufs),
    ))
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
            // The suffix used to be the only check. A repo is free to declare
            // whatever `rfilename` it likes, so anything unsafe is dropped
            // here, at the boundary, rather than being validated at each use
            // site.
            //
            // A *relative path* is allowed, not just a bare filename: repos
            // with many quantizations put each in its own subdirectory, and
            // requiring a single component rejected every weight file in such
            // a repo. Callers store the basename, so nothing joins this onto a
            // local directory as-is.
            if !crate::models::store::is_safe_relative_path(name) {
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
/// Combines [`list_hf_ggufs`] and [`select_gguf`]. Each returned string is an
/// `rfilename` to fetch from `{repo}/resolve/main/{filename}`.
///
/// More than one file comes back when the model is a split GGUF: all of its
/// shards, in order, all of which have to be on disk before llama.cpp can
/// open the first.
pub async fn resolve_hf_gguf(
    hf: &HfRef,
) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
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
        assert_eq!(select_gguf(&files, None).unwrap(), ["model-Q4_K_M.gguf"]);
    }

    #[test]
    fn select_matches_requested_quant_case_insensitive() {
        let files = vec![
            "model-Q8_0.gguf".to_string(),
            "model-Q4_K_M.gguf".to_string(),
        ];
        assert_eq!(select_gguf(&files, Some("q8_0")).unwrap(), ["model-Q8_0.gguf"]);
    }

    // A repo whose only model is one split GGUF used to be refused with
    // "pass an explicit :<quant>" -- and passing one refused again, because
    // several files matched it. Both halves failed, so the repo could not be
    // pulled at all. The whole split now comes back, in shard order.
    #[test]
    fn select_returns_a_whole_split_without_quant() {
        let files = vec![
            "model-00002-of-00003.gguf".to_string(),
            "model-00003-of-00003.gguf".to_string(),
            "model-00001-of-00003.gguf".to_string(),
        ];
        assert_eq!(
            select_gguf(&files, None).unwrap(),
            [
                "model-00001-of-00003.gguf",
                "model-00002-of-00003.gguf",
                "model-00003-of-00003.gguf",
            ]
        );
    }

    // The real layout that started this: unsloth/Qwen3.8-Flash-Next-GGUF, one
    // subdirectory per quantization and every large quant split. rc5 made the
    // files visible; the pull still ended on "multiple .gguf files match
    // quant" because four shards matched and none of the single-file branches
    // applied.
    #[test]
    fn select_returns_every_shard_of_the_requested_quant() {
        let files: Vec<String> = (1..=4)
            .map(|i| format!("UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-0000{i}-of-00004.gguf"))
            .chain(std::iter::once("mmproj-F16.gguf".to_string()))
            .chain((1..=2).map(|i| {
                format!("UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-0000{i}-of-00002.gguf")
            }))
            .collect();
        let picked = select_gguf(&files, Some("UD-Q4_K_XL")).unwrap();
        assert_eq!(picked.len(), 4);
        assert!(picked[0].ends_with("00001-of-00004.gguf"));
        assert!(picked[3].ends_with("00004-of-00004.gguf"));
        // The other quantization's shards, and the projector, stay out of it.
        assert!(picked.iter().all(|f| f.contains("UD-Q4_K_XL")));
    }

    // Downloading three quarters of a model and failing at load with an error
    // about the file is worse than refusing up front.
    #[test]
    fn select_refuses_an_incomplete_split() {
        let files = vec![
            "model-00001-of-00003.gguf".to_string(),
            "model-00003-of-00003.gguf".to_string(),
        ];
        let err = select_gguf(&files, None).expect_err("shard 2 is missing");
        assert!(err.contains("3 shards"), "{err}");
        assert!(err.contains("only 2"), "{err}");
    }

    // Two splits and no way to tell which is wanted is still the user's
    // choice to make, not ours to guess.
    #[test]
    fn select_refuses_to_guess_between_two_splits() {
        let files = vec![
            "model-Q4_K_M-00001-of-00002.gguf".to_string(),
            "model-Q4_K_M-00002-of-00002.gguf".to_string(),
            "model-Q8_0-00001-of-00002.gguf".to_string(),
            "model-Q8_0-00002-of-00002.gguf".to_string(),
        ];
        assert!(select_gguf(&files, None).is_err());
        // Naming one of them resolves it.
        assert_eq!(select_gguf(&files, Some("q8_0")).unwrap().len(), 2);
    }

    #[test]
    fn a_name_that_only_looks_sharded_is_an_ordinary_file() {
        // No `-of-` at all, and a `-of-` that is not a shard counter.
        assert_eq!(parse_shard("model-Q4_K_M.gguf"), None);
        assert_eq!(parse_shard("best-of-breed.gguf"), None);
        assert_eq!(parse_shard("model-00000-of-00003.gguf"), None);
        assert_eq!(parse_shard("model-00004-of-00003.gguf"), None);
        assert_eq!(
            parse_shard("dir/model-00002-of-00003.gguf"),
            Some(("dir/model", 2, 3))
        );
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
            ["gemma-4-12b-it-Q4_K_M.gguf"]
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

// ── Catalog browsing: search the Hub, and price up a repo's quantizations ───

/// One row of a Hub search result.
#[derive(Debug, Clone, serde::Serialize)]
pub struct HfModelSummary {
    /// `owner/repo`, which is also what `eullm pull hf.co/<id>` takes.
    pub id: String,
    pub downloads: u64,
    pub likes: u64,
    /// ISO-8601 last-modified timestamp, as the Hub reports it.
    pub updated: Option<String>,
    /// Gated repos need an accepted licence agreement and a token, neither of
    /// which the engine has, so a pull will fail. Surfaced rather than hidden:
    /// "you must accept the terms" is a better answer than an empty list.
    pub gated: bool,
}

/// Search the HuggingFace Hub for repos containing GGUF files.
///
/// Server-side on purpose. The browser never talks to the Hub, so a user's
/// address is not handed to it by opening the catalog, and
/// `EULLM_WEB_ALLOWED_DOMAINS` stays the single place the perimeter is
/// decided. It also means the catalog works from an HPC login node, which is
/// where model downloads have to happen when compute nodes have no route out.
/// The `expand[]` parameters are not decoration: without them the Hub omits
/// `lastModified` and `gated` from a search response entirely, so a row could
/// not say how old a model is or that it needs an accepted licence.
pub async fn search_hf_models(
    query: &str,
    limit: usize,
) -> Result<Vec<HfModelSummary>, Box<dyn std::error::Error + Send + Sync>> {
    let limit = limit.clamp(1, 100);
    let url = format!(
        "https://huggingface.co/api/models?search={}&filter=gguf&sort=downloads&direction=-1&limit={limit}\
         &expand%5B%5D=lastModified&expand%5B%5D=downloads&expand%5B%5D=likes&expand%5B%5D=gated",
        urlencode(query),
    );
    let client = reqwest::Client::builder()
        .user_agent(concat!("eullm/", env!("CARGO_PKG_VERSION")))
        .timeout(std::time::Duration::from_secs(15))
        .build()?;
    let response = client.get(&url).send().await?;
    if !response.status().is_success() {
        return Err(format!("HuggingFace search returned HTTP {}", response.status()).into());
    }
    let body: serde_json::Value = response.json().await?;
    let rows = body.as_array().map(Vec::as_slice).unwrap_or(&[]);
    Ok(rows
        .iter()
        .filter_map(|m| {
            let id = m.get("id").or_else(|| m.get("modelId"))?.as_str()?;
            // The Hub is free to return whatever it likes here, and this id
            // becomes part of a URL and a local directory name.
            if !is_plausible_repo_id(id) {
                tracing::warn!(
                    "Ignoring implausible repo id from the HuggingFace search: {}",
                    crate::audit::sanitize_for_log(id),
                );
                return None;
            }
            Some(HfModelSummary {
                id: id.to_string(),
                downloads: m.get("downloads").and_then(|v| v.as_u64()).unwrap_or(0),
                likes: m.get("likes").and_then(|v| v.as_u64()).unwrap_or(0),
                updated: m
                    .get("lastModified")
                    .and_then(|v| v.as_str())
                    .map(str::to_string),
                // `gated` is `false`, `"auto"` or `"manual"`.
                gated: !matches!(m.get("gated"), None | Some(serde_json::Value::Bool(false))),
            })
        })
        .collect())
}

/// `owner/repo`, each part a path segment we would be willing to put in a URL.
pub fn is_plausible_repo_id(id: &str) -> bool {
    let mut parts = id.split('/');
    let (Some(owner), Some(repo), None) = (parts.next(), parts.next(), parts.next()) else {
        return false;
    };
    let ok = |s: &str| {
        !s.is_empty()
            && s.len() <= 96
            && s != "."
            && s != ".."
            && s.chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    };
    ok(owner) && ok(repo)
}

/// Minimal percent-encoding for a query string value.
fn urlencode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            b' ' => "+".to_string(),
            other => format!("%{other:02X}"),
        })
        .collect()
}

/// One downloadable quantization of a repo: its files and what they weigh.
#[derive(Debug, Clone, serde::Serialize)]
pub struct QuantOption {
    /// What to show: the subdirectory when the repo groups by quantization,
    /// otherwise the part of the filename that varies between them.
    pub label: String,
    /// What to pass as `:<quant>`. Usually the same as `label`, but falls
    /// back to the full stem when the label alone would match more than one
    /// entry — a directory holding several quantizations makes that happen.
    pub quant: String,
    /// Repo-relative paths, in shard order. One entry unless it is a split.
    pub files: Vec<String>,
    /// Sum of every file's size, which is what the download costs and what
    /// the model occupies on disk.
    pub total_bytes: u64,
}

/// Everything the catalog needs to show about one repo.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RepoContents {
    pub quants: Vec<QuantOption>,
    /// The multimodal projector, if the repo ships one. Downloaded alongside
    /// whichever quantization is chosen — it is the same file for all of them.
    pub mmproj: Option<String>,
    pub mmproj_bytes: u64,
}

/// List a repo's quantizations with their sizes.
///
/// Uses the tree endpoint rather than `siblings`, because only the tree
/// carries `size` — and a catalog that cannot say how big a download is
/// cannot say whether it will run either.
pub async fn list_hf_repo_contents(
    repo: &str,
) -> Result<RepoContents, Box<dyn std::error::Error + Send + Sync>> {
    let url = format!("https://huggingface.co/api/models/{repo}/tree/main?recursive=1");
    let client = reqwest::Client::builder()
        .user_agent(concat!("eullm/", env!("CARGO_PKG_VERSION")))
        .timeout(std::time::Duration::from_secs(20))
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
    let entries = body.as_array().map(Vec::as_slice).unwrap_or(&[]);

    let mut sizes: Vec<(String, u64)> = Vec::new();
    for e in entries {
        let Some(path) = e.get("path").and_then(|p| p.as_str()) else {
            continue;
        };
        if !path.to_lowercase().ends_with(".gguf") {
            continue;
        }
        // Same boundary check the pull path applies — anything that could
        // escape a directory is dropped here rather than at each use site.
        if !crate::models::store::is_safe_relative_path(path) {
            tracing::warn!(
                "Ignoring unsafe path from the HuggingFace tree for {repo}: {}",
                crate::audit::sanitize_for_log(path),
            );
            continue;
        }
        let size = e.get("size").and_then(|s| s.as_u64()).unwrap_or(0);
        sizes.push((path.to_string(), size));
    }

    let (mmproj, mmproj_bytes) = sizes
        .iter()
        .find(|(p, _)| is_mmproj(p))
        .map(|(p, s)| (Some(p.clone()), *s))
        .unwrap_or((None, 0));

    let weights: Vec<String> = sizes
        .iter()
        .filter(|(p, _)| !is_mmproj(p))
        .map(|(p, _)| p.clone())
        .collect();
    let byte_of = |p: &str| sizes.iter().find(|(q, _)| q == p).map(|(_, s)| *s).unwrap_or(0);

    Ok(RepoContents {
        quants: group_quantizations(&weights, byte_of),
        mmproj,
        mmproj_bytes,
    })
}

/// Group a repo's weight files into one entry per quantization.
///
/// Two layouts, and both appear in the wild:
///
/// - one subdirectory per quantization (`UD-Q4_K_XL/Model-…-00001-of-00004.gguf`),
///   where the directory name *is* the label; and
/// - everything flat (`Qwen3-8B-Q4_K_M.gguf`, `Qwen3-8B-Q8_0.gguf`), where
///   the label is what differs between the names.
///
/// The flat case is handled by stripping the longest prefix common to the
/// names, backed off to the last `-`, which is the model name and leaves the
/// quantization. It needs no list of known quantization tokens, so it does
/// not go stale when a new one is invented.
///
/// A directory does not always hold exactly one quantization, and assuming it
/// did was wrong on a real repo: `unsloth/Qwen3.8-Flash-Next-GGUF` keeps six
/// separate MTP draft models under `MTP/`, which collapsed into a single
/// 22.9 GiB entry that was not any downloadable thing. So the directory name
/// is used only when the directory holds one entry; otherwise the same
/// prefix-stripping runs *within* that directory.
fn group_quantizations(files: &[String], size_of: impl Fn(&str) -> u64) -> Vec<QuantOption> {
    // Key by shard stem: every shard of one split shares it, and it already
    // carries the directory, so it separates two quantizations that happen to
    // have the same filename in different directories.
    let mut groups: Vec<(String, Vec<String>)> = Vec::new();
    for f in files {
        let stem = match parse_shard(f) {
            Some((stem, _, _)) => stem.to_string(),
            None => f.strip_suffix(".gguf").unwrap_or(f).to_string(),
        };
        match groups.iter_mut().find(|(s, _)| *s == stem) {
            Some((_, fs)) => fs.push(f.clone()),
            None => groups.push((stem, vec![f.clone()])),
        }
    }

    let dir_of = |stem: &str| match stem.rsplit_once('/') {
        Some((dir, _)) => dir.to_string(),
        None => String::new(),
    };
    let leaf_of = |stem: &str| stem.rsplit('/').next().unwrap_or(stem).to_string();

    let mut out: Vec<QuantOption> = Vec::new();
    for (stem, mut group_files) in groups.iter().cloned() {
        let dir = dir_of(&stem);
        let siblings: Vec<String> = groups
            .iter()
            .filter(|(s, _)| dir_of(s) == dir)
            .map(|(s, _)| leaf_of(s))
            .collect();

        let label = if siblings.len() == 1 && !dir.is_empty() {
            // One quantization in its own directory: the directory names it.
            dir.rsplit('/').next().unwrap_or(&dir).to_string()
        } else if siblings.len() == 1 {
            // A single file at the repo root: there is nothing to contrast
            // it with, so its whole name is the most informative label.
            leaf_of(&stem)
        } else {
            let common = longest_common_prefix(&siblings);
            let common = match common.rfind('-') {
                Some(i) => common[..=i].to_string(),
                None => String::new(),
            };
            let leaf = leaf_of(&stem);
            let trimmed = leaf.strip_prefix(&common).unwrap_or(&leaf).trim_matches('-');
            if trimmed.is_empty() {
                leaf.clone()
            } else {
                trimmed.to_string()
            }
        };

        group_files.sort_by_key(|f| parse_shard(f).map(|(_, i, _)| i).unwrap_or(0));
        let total_bytes = group_files.iter().map(|f| size_of(f)).sum();
        out.push(QuantOption {
            label,
            // Filled in below, once every label is known.
            quant: String::new(),
            files: group_files,
            total_bytes,
        });
    }

    // A label can still repeat across directories -- `MTP/` holds a draft
    // `Q8_0` and the repo root holds the real one -- and two identical rows
    // are worse than a long one. Qualify the duplicates by their directory,
    // and give them a pull token that is unambiguous too: `:<quant>` selects
    // by substring, so a repeated label would resolve to "multiple .gguf
    // files match". The stem is unique by construction, the groups being
    // keyed by it.
    let duplicated: Vec<String> = out
        .iter()
        .filter(|q| out.iter().filter(|o| o.label == q.label).count() > 1)
        .map(|q| q.label.clone())
        .collect();
    for entry in &mut out {
        let label = entry.label.clone();
        let unique = !duplicated.contains(&label);
        if !unique
            && let Some((dir, _)) = entry.files[0].rsplit_once('/')
            // `Q8_0/Q8_0` says nothing `Q8_0` did not: when the directory is
            // already the label, the other entry is the one that needs
            // qualifying, and it gets it on its own pass.
            && dir != label
        {
            entry.label = format!("{dir}/{label}");
        }
        entry.quant = if unique {
            label
        } else {
            match parse_shard(&entry.files[0]) {
                Some((stem, _, _)) => stem.to_string(),
                None => entry.files[0]
                    .strip_suffix(".gguf")
                    .unwrap_or(&entry.files[0])
                    .to_string(),
            }
        };
    }

    out.sort_by_key(|q| q.total_bytes);
    out
}
fn longest_common_prefix(items: &[String]) -> String {
    let Some(first) = items.first() else {
        return String::new();
    };
    let mut len = first.len();
    for other in &items[1..] {
        len = len.min(
            first
                .bytes()
                .zip(other.bytes())
                .take_while(|(a, b)| a == b)
                .count(),
        );
    }
    // Never split a UTF-8 character: back off to the nearest boundary.
    while len > 0 && !first.is_char_boundary(len) {
        len -= 1;
    }
    first[..len].to_string()
}

#[cfg(test)]
mod catalog_tests {
    use super::*;

    fn sizes(files: &[(&str, u64)]) -> (Vec<String>, impl Fn(&str) -> u64 + use<>) {
        let owned: Vec<(String, u64)> = files
            .iter()
            .map(|(f, s)| ((*f).to_string(), *s))
            .collect();
        let names = owned.iter().map(|(f, _)| f.clone()).collect();
        let lookup = move |p: &str| {
            owned
                .iter()
                .find(|(f, _)| f == p)
                .map(|(_, s)| *s)
                .unwrap_or(0)
        };
        (names, lookup)
    }

    // The layout that started all of this: a directory per quantization,
    // every large one split into shards.
    #[test]
    fn a_directory_per_quantization_is_labelled_by_the_directory() {
        let (files, size_of) = sizes(&[
            ("UD-Q4_K_XL/M-UD-Q4_K_XL-00001-of-00002.gguf", 50),
            ("UD-Q4_K_XL/M-UD-Q4_K_XL-00002-of-00002.gguf", 61),
            ("UD-Q2_K_XL/M-UD-Q2_K_XL-00001-of-00001.gguf", 30),
        ]);
        let q = group_quantizations(&files, size_of);
        assert_eq!(q.len(), 2);
        // Sorted smallest first.
        assert_eq!(q[0].label, "UD-Q2_K_XL");
        assert_eq!(q[1].label, "UD-Q4_K_XL");
        assert_eq!(q[1].files.len(), 2);
        assert_eq!(q[1].total_bytes, 111);
    }

    // The other common layout, and the one where the label has to be worked
    // out rather than read off a directory.
    #[test]
    fn a_flat_repo_is_labelled_by_what_differs_between_the_names() {
        let (files, size_of) = sizes(&[
            ("Qwen3-8B-Q4_K_M.gguf", 5),
            ("Qwen3-8B-Q8_0.gguf", 9),
            ("Qwen3-8B-Q2_K.gguf", 3),
        ]);
        let q = group_quantizations(&files, size_of);
        let labels: Vec<&str> = q.iter().map(|q| q.label.as_str()).collect();
        assert_eq!(labels, ["Q2_K", "Q4_K_M", "Q8_0"]);
    }

    // Every quantization starting with the same letters is the case that
    // broke the naive common prefix: `Q4_K_M` and `Q4_K_S` share `Q4_K_`, and
    // stripping that leaves `M` and `S`, which name nothing.
    #[test]
    fn a_shared_leading_token_does_not_get_eaten_from_the_label() {
        let (files, size_of) = sizes(&[
            ("Qwen3-8B-Q4_K_M.gguf", 5),
            ("Qwen3-8B-Q4_K_S.gguf", 4),
        ]);
        let q = group_quantizations(&files, size_of);
        let labels: Vec<&str> = q.iter().map(|q| q.label.as_str()).collect();
        assert_eq!(labels, ["Q4_K_S", "Q4_K_M"]);
    }

    // With one file there is no "what differs", and stripping the common
    // prefix would leave nothing at all.
    #[test]
    fn a_single_flat_file_keeps_its_whole_name_as_the_label() {
        let (files, size_of) = sizes(&[("gemma-4-12b-it-Q4_K_M.gguf", 7)]);
        let q = group_quantizations(&files, size_of);
        assert_eq!(q.len(), 1);
        assert_eq!(q[0].label, "gemma-4-12b-it-Q4_K_M");
        assert_eq!(q[0].total_bytes, 7);
    }

    #[test]
    fn shards_are_ordered_by_index_not_by_listing_order() {
        let (files, size_of) = sizes(&[
            ("M-Q4-00003-of-00003.gguf", 1),
            ("M-Q4-00001-of-00003.gguf", 1),
            ("M-Q4-00002-of-00003.gguf", 1),
        ]);
        let q = group_quantizations(&files, size_of);
        assert!(q[0].files[0].contains("00001"));
        assert!(q[0].files[2].contains("00003"));
    }

    // A directory is not always one quantization. unsloth/Qwen3.8-Flash-Next
    // keeps six separate MTP draft models under `MTP/`, and labelling by the
    // directory collapsed them into one 22.9 GiB entry that was not any
    // downloadable thing.
    #[test]
    fn a_directory_holding_several_models_is_not_one_entry() {
        let (files, size_of) = sizes(&[
            ("MTP/mtp-Model-Q4_K_M.gguf", 3),
            ("MTP/mtp-Model-Q8_0.gguf", 4),
            ("MTP/mtp-Model-shared-Q4_K_M.gguf", 2),
            ("UD-Q2_K_XL/Model-UD-Q2_K_XL-00001-of-00002.gguf", 10),
            ("UD-Q2_K_XL/Model-UD-Q2_K_XL-00002-of-00002.gguf", 11),
        ]);
        let q = group_quantizations(&files, size_of);
        assert_eq!(q.len(), 4, "three MTP files plus one split, got {q:?}");
        // The directory with one entry still gets the directory as its label.
        let split = q.iter().find(|q| q.label == "UD-Q2_K_XL").expect("the split");
        assert_eq!(split.files.len(), 2);
        assert_eq!(split.total_bytes, 21);
        // The crowded directory falls back to what differs inside it.
        let labels: Vec<&str> = q.iter().map(|q| q.label.as_str()).collect();
        assert!(labels.contains(&"Q8_0"), "got {labels:?}");
        assert!(labels.contains(&"shared-Q4_K_M"), "got {labels:?}");
    }

    // `:<quant>` matches by substring, so a label that is not unique across
    // the repo would resolve to "multiple .gguf files match". The pull token
    // falls back to the stem, which is unique by construction.
    #[test]
    fn a_duplicated_label_gets_an_unambiguous_pull_token() {
        let (files, size_of) = sizes(&[
            ("a/Model-Q4_K_M.gguf", 1),
            ("a/Model-Q8_0.gguf", 1),
            ("b/Model-Q4_K_M.gguf", 1),
            ("b/Model-Q8_0.gguf", 1),
        ]);
        let q = group_quantizations(&files, size_of);
        assert_eq!(q.len(), 4);
        for entry in &q {
            if entry.label == "Q4_K_M" {
                assert!(
                    entry.quant.contains('/'),
                    "an ambiguous label must fall back to the stem, got {:?}",
                    entry.quant
                );
                // And the token must actually pick out this one file.
                let picked = select_gguf(&files, Some(&entry.quant)).unwrap();
                assert_eq!(picked, entry.files);
            }
        }
    }

    #[test]
    fn a_repo_id_has_exactly_two_plausible_segments() {
        assert!(is_plausible_repo_id("unsloth/Qwen3.8-Flash-Next-GGUF"));
        assert!(!is_plausible_repo_id("unsloth"));
        assert!(!is_plausible_repo_id("a/b/c"));
        assert!(!is_plausible_repo_id("../etc"));
        assert!(!is_plausible_repo_id("owner/"));
        assert!(!is_plausible_repo_id("own er/repo"));
    }

    #[test]
    fn a_search_query_is_encoded_not_pasted_into_the_url() {
        assert_eq!(urlencode("qwen 3 8b"), "qwen+3+8b");
        assert_eq!(urlencode("a&b=c"), "a%26b%3Dc");
        assert_eq!(urlencode("../x"), "..%2Fx");
    }
}

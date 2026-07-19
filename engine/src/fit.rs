//! Auto GPU-layer fitting for `--fit`.
//!
//! Opt-in helper that decides how many model layers to offload to the GPU so
//! the model fits in available VRAM. CUDA-first: VRAM is probed through a tiny
//! cudart FFI call, available only on `--features cuda` builds. On every other
//! build (or when the probe fails) VRAM is reported as unknown and the caller
//! falls back to the user-provided `--gpu-layers`.
//!
//! The layer count is read from the GGUF header (`<arch>.block_count`) with a
//! small, bounds-checked binary parser, and the on-disk file size is used as a
//! proxy for total weight bytes.

use std::io::IsTerminal;
use std::path::Path;

/// GGUF magic: ASCII "GGUF" stored little-endian as the u32 0x46554747.
const GGUF_MAGIC: u32 = 0x4655_4747;

/// GGUF metadata value type tags (subset we need to parse/skip).
const GGUF_TYPE_UINT8: u32 = 0;
const GGUF_TYPE_INT8: u32 = 1;
const GGUF_TYPE_UINT16: u32 = 2;
const GGUF_TYPE_INT16: u32 = 3;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_UINT64: u32 = 10;
const GGUF_TYPE_INT64: u32 = 11;
const GGUF_TYPE_FLOAT64: u32 = 12;

/// What the parser extracted from a GGUF header that we care about for fitting.
///
/// `n_layers` is required; the attention dimensions are optional — when present
/// they let `compute_fit` size the KV cache exactly for the chosen cache type
/// (so quantizing the KV frees room for more GPU layers). When absent, the
/// sizer falls back to a coarse per-token KV reserve.
#[derive(Debug, Clone)]
pub struct GgufInfo {
    /// Number of transformer blocks/layers (`<arch>.block_count`).
    pub n_layers: u32,
    /// Embedding dimension (`<arch>.embedding_length`), if present.
    pub n_embd: Option<u32>,
    /// Number of attention heads (`<arch>.attention.head_count`), if present.
    pub n_head: Option<u32>,
    /// Number of key/value heads (`<arch>.attention.head_count_kv`) — the GQA
    /// group count that actually sizes the KV cache. If present.
    pub n_head_kv: Option<u32>,
}

impl GgufInfo {
    /// KV elements per token per layer = `n_head_kv × head_dim`, where
    /// `head_dim = n_embd / n_head`. This mirrors exactly the runtime estimate
    /// in the scheduler. Returns `None` when any dimension is missing or zero,
    /// so the sizer can fall back to a coarse reserve.
    fn kv_elems_per_token_per_layer(&self) -> Option<f64> {
        let n_embd = self.n_embd.filter(|&v| v > 0)? as f64;
        let n_head = self.n_head.filter(|&v| v > 0)? as f64;
        let n_head_kv = self.n_head_kv.filter(|&v| v > 0)? as f64;
        let head_dim = n_embd / n_head;
        Some(n_head_kv * head_dim)
    }
}

/// A tiny forward-only cursor over a byte slice. Every read is bounds-checked
/// and returns `None` past the end, so a truncated or malformed header can
/// never panic — the caller treats `None` as "couldn't parse, fall back".
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(n)?;
        let slice = self.data.get(self.pos..end)?;
        self.pos = end;
        Some(slice)
    }

    fn u32(&mut self) -> Option<u32> {
        let b = self.take(4)?;
        Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Option<u64> {
        let b = self.take(8)?;
        Some(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    /// Read a GGUF string: u64 length prefix followed by that many raw bytes.
    /// Returns the bytes without copying (caller decides on UTF-8).
    fn gguf_string(&mut self) -> Option<&'a [u8]> {
        let len = self.u64()?;
        // Guard against absurd lengths from a corrupt header.
        let len = usize::try_from(len).ok()?;
        self.take(len)
    }

    /// Advance past a scalar value of the given type tag without interpreting
    /// it. Returns `None` for unknown tags so parsing stops cleanly.
    fn skip_scalar(&mut self, type_tag: u32) -> Option<()> {
        let size = match type_tag {
            GGUF_TYPE_UINT8 | GGUF_TYPE_INT8 | GGUF_TYPE_BOOL => 1,
            GGUF_TYPE_UINT16 | GGUF_TYPE_INT16 => 2,
            GGUF_TYPE_UINT32 | GGUF_TYPE_INT32 | GGUF_TYPE_FLOAT32 => 4,
            GGUF_TYPE_UINT64 | GGUF_TYPE_INT64 | GGUF_TYPE_FLOAT64 => 8,
            GGUF_TYPE_STRING => {
                self.gguf_string()?;
                return Some(());
            }
            _ => return None,
        };
        self.take(size).map(|_| ())
    }

    /// Skip a whole metadata value (scalar or array) of the given type tag.
    fn skip_value(&mut self, type_tag: u32) -> Option<()> {
        if type_tag == GGUF_TYPE_ARRAY {
            let elem_type = self.u32()?;
            let count = usize::try_from(self.u64()?).ok()?;
            for _ in 0..count {
                self.skip_scalar(elem_type)?;
            }
            Some(())
        } else {
            self.skip_scalar(type_tag)
        }
    }

    /// Read a scalar integer value of the given type tag as u64, when the tag
    /// is one of the unsigned/signed integer types. Used for `block_count`,
    /// which different exporters store as u32 or u64.
    fn read_uint_as_u64(&mut self, type_tag: u32) -> Option<u64> {
        match type_tag {
            GGUF_TYPE_UINT8 | GGUF_TYPE_INT8 => self.take(1).map(|b| b[0] as u64),
            GGUF_TYPE_UINT16 | GGUF_TYPE_INT16 => {
                let b = self.take(2)?;
                Some(u16::from_le_bytes([b[0], b[1]]) as u64)
            }
            GGUF_TYPE_UINT32 | GGUF_TYPE_INT32 => self.u32().map(|v| v as u64),
            GGUF_TYPE_UINT64 | GGUF_TYPE_INT64 => self.u64(),
            _ => None,
        }
    }
}

/// Parse the GGUF header bytes and extract the layer count.
///
/// `data` should be a prefix of the file large enough to cover the metadata
/// (the caller reads a few MB). Returns `None` on any malformed/truncated
/// input or if no `*.block_count` key is present.
pub fn parse_gguf_header(data: &[u8]) -> Option<GgufInfo> {
    let mut c = Cursor::new(data);

    if c.u32()? != GGUF_MAGIC {
        return None;
    }
    let _version = c.u32()?;
    let _tensor_count = c.u64()?;
    let metadata_kv_count = c.u64()?;

    let mut n_layers: Option<u32> = None;
    let mut n_embd: Option<u32> = None;
    let mut n_head: Option<u32> = None;
    let mut n_head_kv: Option<u32> = None;

    // The metadata keys we want, each an integer stored as u32 or u64 depending
    // on the exporter. The `_kv` head count is checked before the plain head
    // count because the former does NOT end with `.attention.head_count`.
    for _ in 0..metadata_kv_count {
        let key_bytes = c.gguf_string()?;
        let value_type = c.u32()?;

        // Pick which field (if any) this key feeds. All are scalar integers;
        // an array-typed match is ignored (skipped) to stay aligned.
        let target: Option<&mut Option<u32>> = if value_type == GGUF_TYPE_ARRAY {
            None
        } else if key_bytes.ends_with(b".block_count") {
            Some(&mut n_layers)
        } else if key_bytes.ends_with(b".embedding_length") {
            Some(&mut n_embd)
        } else if key_bytes.ends_with(b".attention.head_count_kv") {
            Some(&mut n_head_kv)
        } else if key_bytes.ends_with(b".attention.head_count") {
            Some(&mut n_head)
        } else {
            None
        };

        match target {
            Some(slot) => {
                if let Some(v) = c.read_uint_as_u64(value_type) {
                    // Clamp into u32; all of these counts are small.
                    *slot = u32::try_from(v).ok().or(Some(u32::MAX));
                } else {
                    // Unexpected type for a key we wanted; skip to stay aligned.
                    c.skip_value(value_type)?;
                }
            }
            // Not a key we care about (or an array) — skip its value.
            None => c.skip_value(value_type)?,
        }
    }

    n_layers.map(|n_layers| GgufInfo {
        n_layers,
        n_embd,
        n_head,
        n_head_kv,
    })
}

/// Read the leading bytes of a GGUF file and parse its header.
///
/// Reads up to 8 MiB — generous for the metadata block, which sits before the
/// tensor data. Returns `None` on I/O error or parse failure (caller warns and
/// falls back to the provided `--gpu-layers`).
pub fn read_gguf_info(path: &Path) -> Option<GgufInfo> {
    use std::io::Read;

    let mut file = std::fs::File::open(path).ok()?;
    let mut buf = vec![0u8; 8 * 1024 * 1024];
    let n = file.read(&mut buf).ok()?;
    buf.truncate(n);
    parse_gguf_header(&buf)
}

/// Free VRAM in bytes, as reported by the active GPU backend.
///
/// CUDA-only: calls `cudaMemGetInfo` (cudart is already linked in `--features
/// cuda` builds). On non-CUDA builds, or when the call fails / reports zero,
/// returns `None` to signal "VRAM unknown".
#[cfg(feature = "cuda")]
pub fn free_vram_bytes() -> Option<u64> {
    // cudart is linked by the CUDA build of llama.cpp. We bind only the one
    // symbol we need rather than pulling in a CUDA crate.
    unsafe extern "C" {
        fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
    }
    let mut free: usize = 0;
    let mut total: usize = 0;
    // SAFETY: cudaMemGetInfo writes two usize out-params and returns a status
    // code. We pass valid, initialized pointers and read the values only on
    // success. No CUDA context state is mutated.
    let rc = unsafe { cudaMemGetInfo(&mut free as *mut usize, &mut total as *mut usize) };
    if rc != 0 || free == 0 {
        return None;
    }
    Some(free as u64)
}

/// Non-CUDA builds cannot probe VRAM: always "unknown".
#[cfg(not(feature = "cuda"))]
pub fn free_vram_bytes() -> Option<u64> {
    None
}

/// The outcome of a fit computation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FitDecision {
    /// The model fits fully on the GPU → offload all layers (`gpu_layers = -1`).
    FitsFully,
    /// Partial fit: offload `n` of `n_layers` layers, rest stay on CPU.
    Partial { layers: i32, n_layers: u32 },
    /// VRAM or the GGUF header could not be read → fall back to the
    /// user-provided `--gpu-layers`. `reason` is a human-readable explanation.
    Unknown { reason: String },
}

/// Fraction of free VRAM we're willing to use, reserving a slice for
/// allocator fragmentation and miscellaneous driver overhead. Together with
/// `COMPUTE_BUFFER_RESERVE_BYTES` this reproduces the ~0.8 GiB headroom
/// measured on an RTX 5070 Ti loading qwq-32b at 45/64 layers (f16 KV,
/// 4096 ctx) — i.e. the sizer lands the same safe split that was validated by
/// hand, then offloads strictly more layers as the KV cache is quantized.
const VRAM_SAFETY_FRACTION: f64 = 0.97;

/// Flat reserve for the CUDA context + the prefill/decode compute buffer,
/// which does not scale per offloaded layer. The original 320 MiB here was
/// calibrated against an observed ~307 MiB CUDA0 compute buffer — but at the
/// time, `n_ubatch` (the actual physical prefill micro-batch size, which is
/// what the compute buffer scales with) was never set explicitly and
/// defaulted to llama.cpp's own 512 regardless of `n_batch`. Now that
/// `n_ubatch` is explicitly set to 1024 (see `inference::build_ctx_params_with_cache`),
/// this reserve is doubled as a conservative linear estimate — NOT yet
/// confirmed against a real measurement at n_ubatch=1024. Re-measure the
/// actual CUDA0 compute buffer size (nvidia-smi, or the loader's own
/// buffer-size log line) before trusting this at a tight fit.
const COMPUTE_BUFFER_RESERVE_BYTES: f64 = 640.0 * 1024.0 * 1024.0;

/// Coarse KV reserve used only when the GGUF header doesn't expose the
/// attention dims: ~128 B per token per layer (a rough F16 ballpark for
/// 7-8B-class models). The exact path below supersedes this whenever the
/// dims are present.
const FALLBACK_KV_BYTES_PER_TOKEN_PER_LAYER: f64 = 128.0;

/// Compute the fit decision from probed VRAM, the GGUF info, the on-disk file
/// size (a proxy for total weight bytes), and the chosen KV cache element
/// sizes.
///
/// The cost charged for each GPU-offloaded layer is its share of the weights
/// plus its KV-cache slice for the requested context. Because the KV term is
/// now sized from the real cache type, quantizing the KV (e.g. `--cache-type
/// q4_0`) lowers the per-layer cost and lets more layers land on the GPU — the
/// effect grows with context length, where the KV dominates.
///
/// `kv_bytes_per_elem_k` / `_v` are the per-element byte costs of the K and V
/// caches (e.g. 2.0 for F16, 0.5625 for Q4_0).
pub fn compute_fit(
    free_vram: Option<u64>,
    info: Option<&GgufInfo>,
    file_size: u64,
    ctx_size: u32,
    kv_bytes_per_elem_k: f64,
    kv_bytes_per_elem_v: f64,
) -> FitDecision {
    let free_vram = match free_vram {
        Some(v) => v,
        None => {
            return FitDecision::Unknown {
                reason: "could not read free VRAM (needs a CUDA build with a working CUDA device)"
                    .to_string(),
            };
        }
    };
    let info = match info {
        Some(i) if i.n_layers > 0 => i,
        _ => {
            return FitDecision::Unknown {
                reason: "could not parse layer count from the GGUF header".to_string(),
            };
        }
    };
    if file_size == 0 {
        return FitDecision::Unknown {
            reason: "model file size is zero".to_string(),
        };
    }

    let n_layers = info.n_layers as u64;
    let per_layer_weight = file_size as f64 / info.n_layers as f64;
    if per_layer_weight <= 0.0 {
        return FitDecision::Unknown {
            reason: "degenerate per-layer size".to_string(),
        };
    }

    // KV bytes per offloaded layer for this context. Exact when the attention
    // dims are known (mirrors the scheduler's runtime estimate); coarse
    // fallback otherwise.
    let kv_per_layer = match info.kv_elems_per_token_per_layer() {
        Some(elems) => (ctx_size as f64) * elems * (kv_bytes_per_elem_k + kv_bytes_per_elem_v),
        None => (ctx_size as f64) * FALLBACK_KV_BYTES_PER_TOKEN_PER_LAYER,
    };

    let cost_per_layer = per_layer_weight + kv_per_layer;

    // Budget = a fraction of free VRAM, minus the flat compute-buffer reserve.
    let usable = (free_vram as f64 * VRAM_SAFETY_FRACTION - COMPUTE_BUFFER_RESERVE_BYTES).max(0.0);

    let max_layers = (usable / cost_per_layer).floor();
    let max_layers = if max_layers < 0.0 {
        0
    } else {
        max_layers as u64
    };

    if max_layers >= n_layers {
        FitDecision::FitsFully
    } else {
        FitDecision::Partial {
            layers: max_layers as i32,
            n_layers: info.n_layers,
        }
    }
}

/// Both stdin and stdout connected to a terminal — the same gate the picker
/// uses. A non-TTY invocation (Docker, systemd, piped) must never block on a
/// prompt, so the decision logic checks this before asking anything.
fn interactive() -> bool {
    std::io::stdin().is_terminal() && std::io::stdout().is_terminal()
}

/// Format a byte count as GiB for human-facing log lines.
fn gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
}

/// Result of running the full `--fit` flow: the effective `gpu_layers` to use,
/// or `Abort` when the user (or strict mode) declined to load.
pub enum FitOutcome {
    /// Proceed with this `gpu_layers` value.
    Proceed(i32),
    /// Do not load the model (strict mode failure, or user chose abort).
    Abort,
}

/// Run the `--fit` decision flow and return the effective `gpu_layers`.
///
/// `fallback_gpu_layers` is the user-provided `--gpu-layers`, used whenever
/// fit cannot probe. `strict` is `--fit-strict`.
///
/// Headless safety: this only ever prompts when BOTH stdin and stdout are
/// terminals. A non-interactive invocation proceeds with the computed split
/// (printing a one-line warning) and never blocks.
pub fn run_fit(
    model_path: &Path,
    fallback_gpu_layers: i32,
    ctx_size: u32,
    strict: bool,
    kv_bytes_per_elem_k: f64,
    kv_bytes_per_elem_v: f64,
) -> FitOutcome {
    let free_vram = free_vram_bytes();
    let info = read_gguf_info(model_path);
    let file_size = std::fs::metadata(model_path).map(|m| m.len()).unwrap_or(0);

    let decision = compute_fit(
        free_vram,
        info.as_ref(),
        file_size,
        ctx_size,
        kv_bytes_per_elem_k,
        kv_bytes_per_elem_v,
    );

    match decision {
        FitDecision::Unknown { reason } => {
            eprintln!("[EULLM] --fit could not size the model: {reason}.");
            if strict {
                eprintln!(
                    "[EULLM] --fit-strict set: refusing to load without a reliable estimate."
                );
                return FitOutcome::Abort;
            }
            eprintln!(
                "[EULLM] Falling back to --gpu-layers {}.",
                if fallback_gpu_layers < 0 {
                    "all".to_string()
                } else {
                    fallback_gpu_layers.to_string()
                }
            );
            FitOutcome::Proceed(fallback_gpu_layers)
        }
        FitDecision::FitsFully => {
            if let Some(v) = free_vram {
                println!(
                    "[EULLM] --fit: model ({}) fits fully in {} free VRAM → offloading all layers.",
                    gib(file_size),
                    gib(v),
                );
            }
            FitOutcome::Proceed(-1)
        }
        FitDecision::Partial { layers, n_layers } => {
            let free = free_vram.unwrap_or(0);
            if strict {
                eprintln!(
                    "[EULLM] --fit-strict: model needs ~{} but only {} VRAM is free; not loading.",
                    gib(file_size),
                    gib(free),
                );
                eprintln!(
                    "[EULLM] Retry without --fit-strict to offload {layers}/{n_layers} layers (rest in RAM)."
                );
                return FitOutcome::Abort;
            }

            if interactive() {
                println!("[EULLM] --fit: model does not fit fully on the GPU.");
                println!("  Free VRAM:  {}", gib(free));
                println!("  Model size: {}", gib(file_size));
                println!(
                    "  Computed split: {layers}/{n_layers} layers on GPU, rest in RAM (slower)."
                );
                loop {
                    print!("  Continue with this split? [c]ontinue / [a]bort > ");
                    let _ = std::io::Write::flush(&mut std::io::stdout());
                    let mut input = String::new();
                    if std::io::stdin().read_line(&mut input).is_err() {
                        return FitOutcome::Abort;
                    }
                    match input.trim().to_lowercase().as_str() {
                        "c" | "continue" | "y" | "yes" => return FitOutcome::Proceed(layers),
                        "a" | "abort" | "n" | "no" | "q" | "" => return FitOutcome::Abort,
                        _ => println!("  ! Type 'c' to continue or 'a' to abort."),
                    }
                }
            } else {
                // Non-interactive (Docker/systemd/piped): never prompt. Proceed
                // with the computed split so the service starts unattended.
                eprintln!(
                    "[EULLM] --fit: model does not fit fully ({} free VRAM, model {}). \
                     Offloading {layers}/{n_layers} layers, rest in RAM (non-interactive: not prompting).",
                    gib(free),
                    gib(file_size),
                );
                FitOutcome::Proceed(layers)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid GGUF header with a single `<arch>.block_count`
    /// metadata key of the given type, so the parser has something to read.
    fn make_gguf(block_count: u64, as_u64: bool) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes()); // version
        b.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        b.extend_from_slice(&1u64.to_le_bytes()); // metadata_kv_count

        let key = b"qwen3.block_count";
        b.extend_from_slice(&(key.len() as u64).to_le_bytes());
        b.extend_from_slice(key);
        if as_u64 {
            b.extend_from_slice(&GGUF_TYPE_UINT64.to_le_bytes());
            b.extend_from_slice(&block_count.to_le_bytes());
        } else {
            b.extend_from_slice(&GGUF_TYPE_UINT32.to_le_bytes());
            b.extend_from_slice(&(block_count as u32).to_le_bytes());
        }
        b
    }

    #[test]
    fn parses_block_count_u32() {
        let data = make_gguf(28, false);
        let info = parse_gguf_header(&data).unwrap();
        assert_eq!(info.n_layers, 28);
    }

    #[test]
    fn parses_block_count_u64() {
        let data = make_gguf(36, true);
        let info = parse_gguf_header(&data).unwrap();
        assert_eq!(info.n_layers, 36);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut data = make_gguf(28, false);
        data[0] = 0;
        assert!(parse_gguf_header(&data).is_none());
    }

    #[test]
    fn handles_truncated_input() {
        let data = make_gguf(28, false);
        // Cut the value off mid-way; parser must return None, not panic.
        let truncated = &data[..data.len() - 2];
        assert!(parse_gguf_header(truncated).is_none());
    }

    /// F16 KV element sizes (2 bytes each for K and V) — the common case.
    const F16: (f64, f64) = (2.0, 2.0);

    /// A bare `GgufInfo` with only the layer count (no attention dims) — the
    /// fallback path where KV is sized by the coarse constant.
    fn info_layers(n: u32) -> GgufInfo {
        GgufInfo {
            n_layers: n,
            n_embd: None,
            n_head: None,
            n_head_kv: None,
        }
    }

    #[test]
    fn fit_unknown_when_no_vram() {
        let info = info_layers(28);
        let d = compute_fit(None, Some(&info), 5_000_000_000, 4096, F16.0, F16.1);
        assert!(matches!(d, FitDecision::Unknown { .. }));
    }

    #[test]
    fn fit_full_when_vram_ample() {
        let info = info_layers(28);
        // 40 GiB free, 5 GB model → fits fully.
        let d = compute_fit(
            Some(40 * 1024 * 1024 * 1024),
            Some(&info),
            5_000_000_000,
            4096,
            F16.0,
            F16.1,
        );
        assert_eq!(d, FitDecision::FitsFully);
    }

    #[test]
    fn fit_partial_when_vram_tight() {
        let info = info_layers(32);
        // 4 GB free, 16 GB model → only some layers fit.
        let d = compute_fit(
            Some(4_000_000_000),
            Some(&info),
            16_000_000_000,
            4096,
            F16.0,
            F16.1,
        );
        match d {
            FitDecision::Partial { layers, n_layers } => {
                assert!(layers > 0 && (layers as u32) < n_layers);
                assert_eq!(n_layers, 32);
            }
            other => panic!("expected partial, got {other:?}"),
        }
    }

    /// With the attention dims present, quantizing the KV cache (smaller
    /// per-element bytes) must let at least as many — and in a tight fit,
    /// strictly more — layers onto the GPU than F16. This is the headline of
    /// the KV-aware sizer.
    #[test]
    fn quantized_kv_offloads_more_layers() {
        // qwq-32b-ish dims: 64 layers, n_embd 5120, 40 heads, 8 KV heads.
        let info = GgufInfo {
            n_layers: 64,
            n_embd: Some(5120),
            n_head: Some(40),
            n_head_kv: Some(8),
        };
        let free = 15 * 1024 * 1024 * 1024; // ~15 GiB free, 18.5 GB model
        let file = 18_500_000_000;
        // Large context so the KV term is significant.
        let ctx = 32768;

        let f16 = compute_fit(Some(free), Some(&info), file, ctx, 2.0, 2.0);
        // Q4_0 ≈ 0.5625 B/elem for both K and V.
        let q4 = compute_fit(Some(free), Some(&info), file, ctx, 0.5625, 0.5625);

        let layers = |d: &FitDecision| match d {
            FitDecision::Partial { layers, .. } => *layers,
            FitDecision::FitsFully => i32::MAX,
            FitDecision::Unknown { .. } => -1,
        };
        assert!(
            layers(&q4) > layers(&f16),
            "q4_0 KV should offload more layers than f16 at long context: q4={:?} f16={:?}",
            q4,
            f16
        );
    }

    /// The parser must pick up the attention dims when present, and keep the
    /// `_kv` head count distinct from the plain head count.
    #[test]
    fn parses_attention_dims() {
        let mut b = Vec::new();
        b.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes()); // version
        b.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        b.extend_from_slice(&4u64.to_le_bytes()); // metadata_kv_count

        let put_u32 = |key: &[u8], val: u32, buf: &mut Vec<u8>| {
            buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
            buf.extend_from_slice(key);
            buf.extend_from_slice(&GGUF_TYPE_UINT32.to_le_bytes());
            buf.extend_from_slice(&val.to_le_bytes());
        };
        put_u32(b"qwen3.block_count", 64, &mut b);
        put_u32(b"qwen3.embedding_length", 5120, &mut b);
        put_u32(b"qwen3.attention.head_count", 40, &mut b);
        put_u32(b"qwen3.attention.head_count_kv", 8, &mut b);

        let info = parse_gguf_header(&b).unwrap();
        assert_eq!(info.n_layers, 64);
        assert_eq!(info.n_embd, Some(5120));
        assert_eq!(info.n_head, Some(40));
        assert_eq!(info.n_head_kv, Some(8));
        // head_dim = 5120/40 = 128 → 8 × 128 = 1024 elems/token/layer.
        assert_eq!(info.kv_elems_per_token_per_layer(), Some(1024.0));
    }
}

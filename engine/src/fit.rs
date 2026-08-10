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
    /// Explicit key head dimension (`<arch>.attention.key_length`), if the
    /// exporter declared one. Authoritative when present — see
    /// `kv_elems_per_token_per_layer`.
    pub key_length: Option<u32>,
    /// Explicit value head dimension (`<arch>.attention.value_length`).
    pub value_length: Option<u32>,
    /// Hybrid-SSM attention cadence (`<arch>.full_attention_interval`): only
    /// one layer in every `interval` is full attention and pays per-token KV
    /// cache; the rest carry fixed-size recurrent state. Absent (or 1) on
    /// classic transformers, where every layer pays. Some hybrid exporters
    /// omit the key entirely — see `effective_attention_interval` for the
    /// architecture default that covers them.
    pub full_attention_interval: Option<u32>,
    /// `general.architecture`, e.g. `qwen35moe`. Used to apply upstream's
    /// per-architecture attention-cadence default when the explicit key is
    /// missing.
    pub architecture: Option<String>,
}

impl GgufInfo {
    /// KV elements per token per layer, as `(key_elems, value_elems)`.
    ///
    /// Each is `n_head_kv × head_dim`. The head dimension comes from the
    /// explicit `attention.key_length` / `attention.value_length` metadata
    /// when the exporter declared it, and only falls back to `n_embd / n_head`
    /// otherwise.
    ///
    /// That precedence is the whole point of this function. `n_embd / n_head`
    /// is an *assumption* — that the attention heads exactly tile the
    /// embedding — and a growing number of architectures break it by declaring
    /// a head dimension independent of `n_embd`. Qwen3-4B is in our own
    /// catalog: `n_embd` 2560 / `n_head` 32 = 80, while its real `head_dim` is
    /// 128. Sizing the KV cache from 80 under-estimates it by 37%, so `--fit`
    /// offloads more layers than actually fit and the load dies with an
    /// out-of-VRAM error — the failure mode `--fit` exists to prevent.
    ///
    /// Returns `None` only when there is no way to derive a head dimension at
    /// all, so the sizer can fall back to its coarse per-token reserve.
    fn kv_elems_per_token_per_layer(&self) -> Option<(f64, f64)> {
        let n_head_kv = self.n_head_kv.filter(|&v| v > 0)? as f64;

        // Fallback head dim, used per-side only when that side has no explicit
        // length. Computed lazily so a model that declares key_length but not
        // n_head still resolves.
        let derived = || -> Option<f64> {
            let n_embd = self.n_embd.filter(|&v| v > 0)? as f64;
            let n_head = self.n_head.filter(|&v| v > 0)? as f64;
            Some(n_embd / n_head)
        };

        let k_dim = match self.key_length.filter(|&v| v > 0) {
            Some(v) => v as f64,
            None => derived()?,
        };
        let v_dim = match self.value_length.filter(|&v| v > 0) {
            Some(v) => v as f64,
            // Virtually every architecture uses the same dimension for K and V,
            // so reuse whatever K resolved to (explicit or derived) rather than
            // reaching back to the n_embd assumption independently.
            None => k_dim,
        };

        Some((n_head_kv * k_dim, n_head_kv * v_dim))
    }

    /// How many of `n` offloaded layers pay per-token KV cache.
    ///
    /// Charging every layer the full KV slice is exactly right on a classic
    /// transformer and wildly wrong on a hybrid-SSM model. Measured on real
    /// hardware (Qwen3.6-35B-A3B, `full_attention_interval=4`, 64 layers,
    /// `--ctx-size 262144`): the uniform charge priced every layer at ~1.35
    /// GiB, the sizer stopped at a handful of layers and left 8 GiB of VRAM
    /// idle, and the MoE path's total-KV estimate came out 4× the real ~20
    /// GiB. Three out of four layers actually cost only their weights.
    ///
    /// Rounded UP, not averaged. llama.cpp offloads a contiguous block of
    /// layers, and how many attention layers fall inside it depends on the
    /// alignment (upstream marks layer `i` as attention when
    /// `(i+1) % interval == 0` — see `models/qwen35.cpp`): a block of 22
    /// with interval 4 holds 6 paying layers, not 5.5. The average
    /// under-charged by half a KV slice (~0.5 GiB at 262k context) and ate
    /// into the safety margin — observed live as a load sitting ~600 MiB
    /// from the VRAM ceiling. The ceiling division is alignment-independent
    /// and at worst one layer conservative.
    fn kv_paying_layers(&self, n: u64) -> u64 {
        match self.effective_attention_interval() {
            Some(interval) if interval > 1 => n.div_ceil(u64::from(interval)),
            _ => n,
        }
    }

    /// The attention cadence actually in effect: the explicit header key
    /// when present, else upstream's per-architecture default. llama.cpp
    /// hardcodes `full_attn_interval = 4` for the qwen35 family and
    /// qwen3next BEFORE the optional key read (`models/qwen35.cpp`,
    /// `qwen35moe.cpp`, `qwen3next.cpp`), and real GGUFs rely on it:
    /// Ornith-1.0-35B (arch `qwen35moe`) ships without the key at all, so
    /// keying the discount on the header alone silently re-charged every
    /// layer full KV on that model.
    fn effective_attention_interval(&self) -> Option<u32> {
        if self.full_attention_interval.is_some() {
            return self.full_attention_interval;
        }
        match self.architecture.as_deref() {
            Some(arch) if arch.starts_with("qwen35") || arch == "qwen3next" => Some(4),
            _ => None,
        }
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
    let mut key_length: Option<u32> = None;
    let mut value_length: Option<u32> = None;
    let mut full_attention_interval: Option<u32> = None;
    let mut architecture: Option<String> = None;

    // The metadata keys we want, each an integer stored as u32 or u64 depending
    // on the exporter. The `_kv` head count is checked before the plain head
    // count because the former does NOT end with `.attention.head_count`.
    //
    // Running out of buffer mid-metadata breaks out with whatever was parsed
    // so far instead of returning `None`. Exporters write the hyperparameter
    // keys before the tokenizer block, and a big-vocab model's tokenizer
    // arrays alone can overrun any fixed read budget — found on real
    // hardware with Qwen3.6-35B-A3B (248k-token vocabulary): its
    // `block_count` sat 20 keys before the truncation point, and discarding
    // it made `--fit` report "could not parse layer count", fall back to
    // `--gpu-layers all`, and OOM on a model the sizer exists to handle.
    for _ in 0..metadata_kv_count {
        let Some(key_bytes) = c.gguf_string() else {
            break;
        };
        let Some(value_type) = c.u32() else {
            break;
        };

        // `general.architecture` is the one string-typed key we want — it is
        // conventionally the first key in the file, and it selects the
        // per-architecture attention-cadence default when the explicit
        // interval key is absent (see `effective_attention_interval`).
        if key_bytes == b"general.architecture" && value_type == GGUF_TYPE_STRING {
            match c.gguf_string() {
                Some(s) => {
                    architecture = Some(String::from_utf8_lossy(s).into_owned());
                    continue;
                }
                None => break,
            }
        }

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
        } else if key_bytes.ends_with(b".attention.key_length") {
            Some(&mut key_length)
        } else if key_bytes.ends_with(b".attention.value_length") {
            Some(&mut value_length)
        } else if key_bytes.ends_with(b".full_attention_interval") {
            Some(&mut full_attention_interval)
        } else {
            None
        };

        let advanced = match target {
            Some(slot) => match c.read_uint_as_u64(value_type) {
                Some(v) => {
                    // Clamp into u32; all of these counts are small.
                    *slot = u32::try_from(v).ok().or(Some(u32::MAX));
                    true
                }
                // Unexpected type for a key we wanted; skip to stay aligned.
                None => c.skip_value(value_type).is_some(),
            },
            // Not a key we care about (or an array) — skip its value.
            None => c.skip_value(value_type).is_some(),
        };
        if !advanced {
            break;
        }

        // Everything wanted is in hand — stop here instead of paying to
        // skip through the tokenizer arrays (and instead of depending on
        // the read budget covering them at all).
        // `full_attention_interval` is in the early-stop set even though only
        // hybrid-SSM models have it: on those it sits AFTER the attention
        // dims (qwen35 writes it at key 33, value_length at 27), so stopping
        // without it would always miss it. Dense models simply scan on to the
        // buffer's end — every unwanted value is skipped by cursor
        // arithmetic, and the truncation tolerance above already covers the
        // case where the buffer ends first.
        if n_layers.is_some()
            && n_embd.is_some()
            && n_head.is_some()
            && n_head_kv.is_some()
            && key_length.is_some()
            && value_length.is_some()
            && full_attention_interval.is_some()
        {
            break;
        }
    }

    n_layers.map(|n_layers| GgufInfo {
        n_layers,
        n_embd,
        n_head,
        n_head_kv,
        key_length,
        value_length,
        full_attention_interval,
        architecture,
    })
}

/// Read the leading bytes of a GGUF file and parse its header.
///
/// Reads up to 8 MiB — generous for the metadata block, which sits before the
/// tensor data. Returns `None` on I/O error or parse failure (caller warns and
/// falls back to the provided `--gpu-layers`).
pub fn read_gguf_info(path: &Path) -> Option<GgufInfo> {
    use std::io::Read;

    let file = std::fs::File::open(path).ok()?;
    // `Read::read` is allowed to return fewer bytes than asked for even when
    // more are available, so a single call could hand the parser a truncated
    // header, fail, and silently fall back to `--gpu-layers` — non
    // deterministically. `take(..).read_to_end(..)` reads until the limit or EOF.
    let mut buf = Vec::with_capacity(8 * 1024 * 1024);
    file.take(8 * 1024 * 1024).read_to_end(&mut buf).ok()?;
    parse_gguf_header(&buf)
}

/// Per-layer split of a GGUF's tensor bytes into "expert" (the MoE
/// feed-forward tensors `--cpu-moe`/`--n-cpu-moe` can move to CPU RAM) and
/// everything else (attention, norms, embeddings, output head, and — for a
/// MoE model — the router/gate weights, all of which stay GPU-resident
/// regardless of expert offload).
///
/// Sizes come from the GGUF tensor-info section's `offset` field, not from
/// the tensor's declared type/shape: consecutive tensors' offsets bound each
/// other's real on-disk byte size exactly, with no need to know every ggml
/// quantization format's block size.
#[derive(Debug, Clone)]
pub struct MoeLayout {
    /// Bytes of every tensor that is not part of any layer's expert set.
    pub non_expert_bytes: u64,
    /// Expert-tensor bytes for each transformer block, indexed by layer
    /// number parsed from the tensor name (`blk.<i>...`). A dense
    /// (non-MoE) model has every entry `0`.
    pub expert_bytes_per_layer: Vec<u64>,
}

impl MoeLayout {
    /// Whether any layer actually has expert tensors.
    pub fn is_moe(&self) -> bool {
        self.expert_bytes_per_layer.iter().any(|&b| b > 0)
    }
}

/// The exact tensor-name families `--cpu-moe`/`--n-cpu-moe` already move to
/// CPU RAM (`inference::mod.rs`'s `add_cpu_moe_override` / the per-layer
/// `blk\.{i}\.ffn_(up|down|gate|gate_up)_(ch|)exps` pattern). Matched here by
/// substring rather than a regex engine — the vocabulary is small and fixed,
/// so a new dependency isn't worth it for eight literal strings.
const EXPERT_TENSOR_MARKERS: [&str; 8] = [
    "ffn_up_exps",
    "ffn_down_exps",
    "ffn_gate_exps",
    "ffn_gate_up_exps",
    "ffn_up_chexps",
    "ffn_down_chexps",
    "ffn_gate_chexps",
    "ffn_gate_up_chexps",
];

fn is_expert_tensor_name(name: &str) -> bool {
    EXPERT_TENSOR_MARKERS.iter().any(|marker| name.contains(marker))
}

/// Parse the layer index out of a tensor name shaped `blk.<N>.<rest>`.
/// Returns `None` for the handful of tensors with no layer (embeddings,
/// output head, output norm) — the caller counts those as non-expert.
fn tensor_layer_index(name: &str) -> Option<u32> {
    let rest = name.strip_prefix("blk.")?;
    let end = rest.find('.')?;
    rest[..end].parse().ok()
}

/// Round `pos` up to the next multiple of `alignment`. GGUF's tensor data
/// section starts at the first such boundary after the tensor-info table;
/// `alignment` defaults to 32 and is only ever overridden by a
/// `general.alignment` metadata key.
fn align_up(pos: u64, alignment: u64) -> Option<u64> {
    if alignment == 0 {
        return Some(pos);
    }
    let rem = pos % alignment;
    if rem == 0 {
        Some(pos)
    } else {
        pos.checked_add(alignment - rem)
    }
}

/// Parse a GGUF's tensor-info section into a [`MoeLayout`].
///
/// `data` must start at the file's first byte (magic included) and cover the
/// metadata block *and* the tensor-info table — both sit before the tensor
/// data itself, so the same leading chunk `read_gguf_info` already reads is
/// enough. `file_size` is the real on-disk size, needed only to size the
/// *last* tensor (every other tensor's size is the gap to the next one's
/// offset). `n_layers` sizes the returned per-layer vector; pass the value
/// already read via [`parse_gguf_header`] so the two never disagree.
///
/// Returns `None` on any malformed/truncated input, exactly like
/// [`parse_gguf_header`] — the caller treats that as "not a MoE model" and
/// falls back to the ordinary dense `--fit` path.
pub fn parse_gguf_moe_layout(data: &[u8], file_size: u64, n_layers: u32) -> Option<MoeLayout> {
    let mut c = Cursor::new(data);

    if c.u32()? != GGUF_MAGIC {
        return None;
    }
    let _version = c.u32()?;
    let tensor_count = c.u64()?;
    let metadata_kv_count = c.u64()?;

    let mut alignment: u64 = 32;
    for _ in 0..metadata_kv_count {
        let key_bytes = c.gguf_string()?;
        let value_type = c.u32()?;
        if key_bytes == b"general.alignment" {
            match c.read_uint_as_u64(value_type) {
                Some(v) if v > 0 => alignment = v,
                Some(_) => {}
                None => c.skip_value(value_type)?,
            }
        } else {
            c.skip_value(value_type)?;
        }
    }

    // Tensor-info record: name, n_dimensions, dimensions[n_dimensions] (u64
    // each), ggml type (u32), offset (u64). Only name and offset matter here;
    // the rest is consumed purely to keep the cursor aligned with the next
    // record. `tensor_count` is untrusted (a corrupt file could claim
    // billions) — no `with_capacity` on it, so a bad count just runs the
    // bounds-checked cursor out of bytes and returns `None`, never a huge
    // allocation.
    let mut tensors: Vec<(String, u64)> = Vec::new();
    for _ in 0..tensor_count {
        let name = String::from_utf8_lossy(c.gguf_string()?).into_owned();
        let n_dims = c.u32()?;
        for _ in 0..n_dims {
            c.u64()?;
        }
        let _ggml_type = c.u32()?;
        let offset = c.u64()?;
        tensors.push((name, offset));
    }
    if tensors.is_empty() {
        return None;
    }

    let data_start = align_up(c.pos as u64, alignment)?;
    tensors.sort_by_key(|(_, offset)| *offset);

    let mut non_expert_bytes: u64 = 0;
    let mut expert_bytes_per_layer = vec![0u64; n_layers as usize];

    for (idx, (name, offset)) in tensors.iter().enumerate() {
        let size = match tensors.get(idx + 1) {
            Some((_, next_offset)) => next_offset.checked_sub(*offset)?,
            None => file_size.checked_sub(data_start)?.checked_sub(*offset)?,
        };

        match (is_expert_tensor_name(name), tensor_layer_index(name)) {
            (true, Some(layer)) if (layer as usize) < expert_bytes_per_layer.len() => {
                expert_bytes_per_layer[layer as usize] += size;
            }
            _ => non_expert_bytes += size,
        }
    }

    Some(MoeLayout {
        non_expert_bytes,
        expert_bytes_per_layer,
    })
}

/// Read a GGUF file's leading bytes and parse its tensor layout.
///
/// Unlike [`read_gguf_info`] — whose parser can stop early because the keys
/// it wants come first — the tensor-info table sits *after* every metadata
/// entry, so the whole metadata block must fit in the buffer. A big-vocab
/// model overruns the first budget with tokenizer arrays alone (Qwen3.6's
/// 248k-token vocabulary plus merges is ~10 MiB of metadata by itself), so
/// on a parse failure the read retries with geometrically larger budgets
/// before giving up. The cap stays far below any real model's weights, so
/// this never reads tensor data. Same `take().read_to_end()` pattern as
/// `read_gguf_info` — see there for why a single `Read::read` isn't enough.
pub fn read_gguf_moe_layout(path: &Path, file_size: u64, n_layers: u32) -> Option<MoeLayout> {
    use std::io::Read;

    for cap in [8u64 << 20, 32 << 20, 128 << 20] {
        let cap = cap.min(file_size);
        let file = std::fs::File::open(path).ok()?;
        let mut buf = Vec::with_capacity(cap as usize);
        file.take(cap).read_to_end(&mut buf).ok()?;
        if let Some(layout) = parse_gguf_moe_layout(&buf, file_size, n_layers) {
            return Some(layout);
        }
        // The whole file is already in the buffer — a bigger budget cannot
        // see anything more.
        if (buf.len() as u64) >= file_size {
            break;
        }
    }
    None
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

    // KV bytes for one PAYING layer at this context. Exact when the
    // attention dims are known (mirrors the scheduler's runtime estimate);
    // coarse fallback otherwise. On hybrid-SSM models only some layers pay
    // (see `kv_paying_layers`), so the cost of offloading n layers is
    // per-layer weights times n plus this slice times the paying count —
    // counted exactly, not averaged, because the average under-charges
    // whenever the offloaded block holds one more attention layer than the
    // mean (measured live: ~0.5 GiB at 262k context).
    let kv_per_paying_layer = match info.kv_elems_per_token_per_layer() {
        Some((k_elems, v_elems)) => {
            (ctx_size as f64) * (k_elems * kv_bytes_per_elem_k + v_elems * kv_bytes_per_elem_v)
        }
        None => (ctx_size as f64) * FALLBACK_KV_BYTES_PER_TOKEN_PER_LAYER,
    };

    // Budget = a fraction of free VRAM, minus the flat compute-buffer reserve.
    let usable = (free_vram as f64 * VRAM_SAFETY_FRACTION - COMPUTE_BUFFER_RESERVE_BYTES).max(0.0);

    // Largest layer count whose exact cost fits the budget. A few hundred
    // iterations at most; the closed-form division stopped being exact the
    // moment the KV charge became per-paying-layer instead of uniform.
    let mut max_layers = 0u64;
    for n in (0..=n_layers).rev() {
        let cost = (n as f64) * per_layer_weight
            + (info.kv_paying_layers(n) as f64) * kv_per_paying_layer;
        if cost <= usable {
            max_layers = n;
            break;
        }
    }

    if max_layers >= n_layers {
        FitDecision::FitsFully
    } else {
        FitDecision::Partial {
            layers: max_layers as i32,
            n_layers: info.n_layers,
        }
    }
}

/// The outcome of composing `--fit` with MoE expert offload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MoeFitDecision {
    /// Not a MoE model (or its layout couldn't be read) — the caller's
    /// existing dense `--fit` decision, from [`compute_fit`], stands
    /// unchanged.
    NotMoe,
    /// Apply directly: keep the first `n_cpu_moe` layers' expert tensors on
    /// CPU RAM (same convention `--n-cpu-moe` already uses), and offload
    /// every whole layer to the GPU (`gpu_layers = -1`) — pushing this many
    /// layers' experts off is enough for everything else to fit.
    /// `n_cpu_moe == 0` means the model is MoE but already fits fully as-is.
    Proceed { n_cpu_moe: u32 },
    /// Even with every layer's experts on CPU RAM, the non-expert weights
    /// plus KV cache alone don't fit fully on GPU either. Apply blanket
    /// `--cpu-moe` (all experts on CPU) *and* this `gpu_layers` split for
    /// the non-expert weights — the same reduced-offload fallback `--fit`
    /// already uses for a dense model, charged against non-expert bytes
    /// only. `gpu_layers` can be as low as `0` (fully CPU): the guarantee
    /// this composes toward is that the model loads, not that it's fast.
    ProceedCpuMoeAndPartial { gpu_layers: i32 },
}

/// Compute the MoE-aware fit decision from probed VRAM, GGUF info, and the
/// tensor layout parsed by [`parse_gguf_moe_layout`]/[`read_gguf_moe_layout`].
///
/// Pure decision logic — no I/O — mirroring [`compute_fit`]'s split from
/// [`run_fit`]. See [`run_moe_fit`] for the I/O-performing wrapper.
pub fn compute_moe_fit(
    free_vram: Option<u64>,
    info: Option<&GgufInfo>,
    layout: Option<&MoeLayout>,
    ctx_size: u32,
    kv_bytes_per_elem_k: f64,
    kv_bytes_per_elem_v: f64,
) -> MoeFitDecision {
    let (Some(free_vram), Some(info), Some(layout)) = (free_vram, info, layout) else {
        return MoeFitDecision::NotMoe;
    };
    if !layout.is_moe() {
        return MoeFitDecision::NotMoe;
    }

    let kv_per_paying_layer = match info.kv_elems_per_token_per_layer() {
        Some((k_elems, v_elems)) => {
            (ctx_size as f64) * (k_elems * kv_bytes_per_elem_k + v_elems * kv_bytes_per_elem_v)
        }
        None => (ctx_size as f64) * FALLBACK_KV_BYTES_PER_TOKEN_PER_LAYER,
    };
    let total_kv =
        kv_per_paying_layer * info.kv_paying_layers(info.n_layers as u64) as f64;
    let usable = (free_vram as f64 * VRAM_SAFETY_FRACTION - COMPUTE_BUFFER_RESERVE_BYTES).max(0.0);
    let fixed_cost = layout.non_expert_bytes as f64 + total_kv;

    if fixed_cost >= usable {
        // Every expert already assumed off-GPU here; charge only the
        // non-expert bytes against the ordinary dense sizer to find how many
        // whole layers of *those* still fit.
        let decision = compute_fit(
            Some(free_vram),
            Some(info),
            layout.non_expert_bytes,
            ctx_size,
            kv_bytes_per_elem_k,
            kv_bytes_per_elem_v,
        );
        let gpu_layers = match decision {
            FitDecision::FitsFully => -1,
            FitDecision::Partial { layers, .. } => layers,
            // free_vram/info/file_size are already known valid at this
            // point, so this arm is unreachable in practice — 0 (fully CPU)
            // is the safe floor if it ever fires anyway.
            FitDecision::Unknown { .. } => 0,
        };
        return MoeFitDecision::ProceedCpuMoeAndPartial { gpu_layers };
    }

    // Budget left over for expert tensors once non-expert weights + KV are
    // paid for. Keep layers on GPU from the END backward — `--n-cpu-moe`
    // only supports evicting a *contiguous* prefix (`blk.0 .. blk.N-1`), so
    // the moment one layer (scanned from the last) doesn't fit, it and every
    // earlier layer must go, not just that one.
    let budget_left = usable - fixed_cost;
    let n_layers = layout.expert_bytes_per_layer.len();
    let mut kept_bytes = 0.0f64;
    let mut n_cpu_moe = n_layers as u32;
    for (rev_idx, expert_bytes) in layout.expert_bytes_per_layer.iter().rev().enumerate() {
        let tentative = kept_bytes + *expert_bytes as f64;
        if tentative > budget_left {
            break;
        }
        kept_bytes = tentative;
        n_cpu_moe = (n_layers - (rev_idx + 1)) as u32;
    }

    MoeFitDecision::Proceed { n_cpu_moe }
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

/// Apply a user-set `--gpu-layers` ceiling to a computed offload.
///
/// `--gpu-layers` states how many layers the user wants on the GPU, and that
/// upper bound is honoured — but it cannot raise the offload above what the
/// sizer says fits. A layer count chosen for one model is not a fact about
/// the next one: that is the same mistake as reusing a launch model's split
/// for a swapped-in model, which is how a 27B loaded with `all` layers and
/// died out of memory. Forcing past the estimate is what `--no-fit` is for.
///
/// Negative means "no ceiling" (`-1` = all layers) on either side.
pub fn apply_gpu_layers_ceiling(computed: i32, ceiling: i32) -> i32 {
    match (computed, ceiling) {
        (_, c) if c < 0 => computed,
        (comp, c) if comp < 0 => c,
        (comp, c) => comp.min(c),
    }
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
    run_fit_impl(
        model_path,
        fallback_gpu_layers,
        ctx_size,
        strict,
        kv_bytes_per_elem_k,
        kv_bytes_per_elem_v,
        /* allow_prompt */ true,
        /* announce */ true,
    )
}

/// [`run_fit`] for server contexts: never prompts, even on an interactive
/// terminal. A daemon (or a model swap serving an API request) has nobody
/// at the keyboard on the other end of stdin — `serve` started from a shell
/// IS a TTY, so the [`run_fit`] gate alone would block it on the first
/// partial-fit load. A partial split proceeds with a logged one-liner;
/// `strict` still refuses, and the caller turns that into an API error.
pub fn run_fit_headless(
    model_path: &Path,
    fallback_gpu_layers: i32,
    ctx_size: u32,
    strict: bool,
    kv_bytes_per_elem_k: f64,
    kv_bytes_per_elem_v: f64,
) -> FitOutcome {
    run_fit_impl(
        model_path,
        fallback_gpu_layers,
        ctx_size,
        strict,
        kv_bytes_per_elem_k,
        kv_bytes_per_elem_v,
        /* allow_prompt */ false,
        /* announce */ false,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_fit_impl(
    model_path: &Path,
    fallback_gpu_layers: i32,
    ctx_size: u32,
    strict: bool,
    kv_bytes_per_elem_k: f64,
    kv_bytes_per_elem_v: f64,
    allow_prompt: bool,
    announce: bool,
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
            if strict {
                eprintln!("[EULLM] --fit could not size the model: {reason}.");
                eprintln!(
                    "[EULLM] --fit-strict set: refusing to load without a reliable estimate."
                );
                return FitOutcome::Abort;
            }
            // Silent when sizing is the automatic default: every non-CUDA
            // build lands here on every launch (there is no free-VRAM probe
            // to read), and an unrequested warning about a flag the user
            // never typed is noise. A `--fit` that was actually asked for
            // still gets its explanation.
            if announce {
                eprintln!("[EULLM] --fit could not size the model: {reason}.");
                eprintln!(
                    "[EULLM] Falling back to --gpu-layers {}.",
                    if fallback_gpu_layers < 0 {
                        "all".to_string()
                    } else {
                        fallback_gpu_layers.to_string()
                    }
                );
            }
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

            if allow_prompt && interactive() {
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
                // No prompt here: either the caller forbids it (a server, or
                // automatic sizing) or this is not a terminal (Docker,
                // systemd, piped). Say what was decided and why — a partial
                // split costs speed, and the user is entitled to know it
                // happened even when nobody asked a question.
                eprintln!(
                    "[EULLM] Model larger than free VRAM ({} free, model {}): \
                     offloading {layers}/{n_layers} layers, the rest runs in RAM (slower). \
                     Set --gpu-layers to choose yourself, or --no-fit to disable sizing.",
                    gib(free),
                    gib(file_size),
                );
                FitOutcome::Proceed(layers)
            }
        }
    }
}

/// Run the MoE-aware fit flow: probe VRAM, read the GGUF header and tensor
/// layout, and delegate to [`compute_moe_fit`].
///
/// Always non-interactive — unlike [`run_fit`], this never prompts. The
/// caller runs it *before* `run_fit`: a MoE decision here always resolves to
/// a loadable configuration (expert offload, in the worst case combined with
/// a reduced layer split down to fully-CPU), so there is no "doesn't fit,
/// continue anyway?" question left to ask. Only when this returns
/// [`MoeFitDecision::NotMoe`] does the dense `run_fit` flow — with its
/// prompt and its `--fit-strict` handling — take over.
pub fn run_moe_fit(
    model_path: &Path,
    ctx_size: u32,
    kv_bytes_per_elem_k: f64,
    kv_bytes_per_elem_v: f64,
) -> MoeFitDecision {
    let free_vram = free_vram_bytes();
    let info = read_gguf_info(model_path);
    let file_size = std::fs::metadata(model_path).map(|m| m.len()).unwrap_or(0);
    let layout = match (&info, file_size) {
        (Some(i), size) if size > 0 => read_gguf_moe_layout(model_path, size, i.n_layers),
        _ => None,
    };

    compute_moe_fit(
        free_vram,
        info.as_ref(),
        layout.as_ref(),
        ctx_size,
        kv_bytes_per_elem_k,
        kv_bytes_per_elem_v,
    )
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
            key_length: None,
            value_length: None,
            full_attention_interval: None,
            architecture: None,
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
            key_length: None,
            value_length: None,
            full_attention_interval: None,
            architecture: None,
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
        // head_dim = 5120/40 = 128 → 8 × 128 = 1024 elems/token/layer, K and V alike.
        assert_eq!(info.kv_elems_per_token_per_layer(), Some((1024.0, 1024.0)));
    }
}

#[cfg(test)]
mod key_length_tests {
    use super::*;

    /// Append `key`/`value` as a u32 metadata entry.
    fn put_u32(key: &[u8], val: u32, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key);
        buf.extend_from_slice(&GGUF_TYPE_UINT32.to_le_bytes());
        buf.extend_from_slice(&val.to_le_bytes());
    }

    fn put_str(key: &[u8], val: &[u8], buf: &mut Vec<u8>) {
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key);
        buf.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
        buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
        buf.extend_from_slice(val);
    }

    fn header(entries: &[(&[u8], u32)]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&(entries.len() as u64).to_le_bytes());
        for (k, v) in entries {
            put_u32(k, *v, &mut b);
        }
        b
    }

    /// Qwen3-4B — in our own catalog — declares a head dimension that is NOT
    /// `n_embd / n_head`: 2560/32 would give 80, the real value is 128. This is
    /// the regression that made `--fit` under-size the KV cache by 37% and
    /// offload more layers than fit, turning `--fit` into the cause of the
    /// out-of-VRAM error it exists to prevent.
    #[test]
    fn explicit_key_length_overrides_the_n_embd_over_n_head_assumption() {
        let data = header(&[
            (b"qwen3.block_count", 36),
            (b"qwen3.embedding_length", 2560),
            (b"qwen3.attention.head_count", 32),
            (b"qwen3.attention.head_count_kv", 8),
            (b"qwen3.attention.key_length", 128),
            (b"qwen3.attention.value_length", 128),
        ]);
        let info = parse_gguf_header(&data).expect("header parses");
        assert_eq!(info.key_length, Some(128));
        assert_eq!(info.value_length, Some(128));
        // 8 KV heads × 128 = 1024, not 8 × 80 = 640.
        assert_eq!(info.kv_elems_per_token_per_layer(), Some((1024.0, 1024.0)));
    }

    #[test]
    fn without_explicit_lengths_the_derived_head_dim_is_still_used() {
        let data = header(&[
            (b"qwen3.block_count", 64),
            (b"qwen3.embedding_length", 5120),
            (b"qwen3.attention.head_count", 40),
            (b"qwen3.attention.head_count_kv", 8),
        ]);
        let info = parse_gguf_header(&data).expect("header parses");
        assert_eq!(info.kv_elems_per_token_per_layer(), Some((1024.0, 1024.0)));
    }

    /// A declared key_length also covers V when only K is declared — closer to
    /// the truth than falling back to the n_embd assumption for that side.
    #[test]
    fn key_length_alone_covers_both_sides() {
        let data = header(&[
            (b"arch.block_count", 32),
            (b"arch.embedding_length", 2560),
            (b"arch.attention.head_count", 32),
            (b"arch.attention.head_count_kv", 8),
            (b"arch.attention.key_length", 128),
        ]);
        let info = parse_gguf_header(&data).expect("header parses");
        assert_eq!(info.kv_elems_per_token_per_layer(), Some((1024.0, 1024.0)));
    }

    /// Differing K and V dimensions are charged separately rather than being
    /// collapsed onto one figure.
    #[test]
    fn asymmetric_key_and_value_lengths_are_charged_separately() {
        let data = header(&[
            (b"arch.block_count", 30),
            (b"arch.attention.head_count_kv", 4),
            (b"arch.attention.key_length", 256),
            (b"arch.attention.value_length", 128),
        ]);
        let info = parse_gguf_header(&data).expect("header parses");
        assert_eq!(info.kv_elems_per_token_per_layer(), Some((1024.0, 512.0)));
    }

    /// Qwen3.6-35B-A3B's tokenizer block (248k-token vocabulary) overruns
    /// the 8 MiB read budget on its own, so the buffer ends mid-array. The
    /// hyperparameter keys all precede it — a truncation there must degrade
    /// to "return what was parsed", not discard an already-read layer count
    /// (which made `--fit` fall back to `--gpu-layers all` and OOM on real
    /// hardware).
    #[test]
    fn tolerates_truncation_inside_the_tokenizer_arrays() {
        let mut b = Vec::new();
        b.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&5u64.to_le_bytes()); // claims 5 metadata entries
        put_u32(b"qwen35moe.block_count", 40, &mut b);
        put_u32(b"qwen35moe.embedding_length", 2048, &mut b);
        put_u32(b"qwen35moe.attention.head_count", 16, &mut b);
        put_u32(b"qwen35moe.attention.head_count_kv", 2, &mut b);
        // Fifth entry: a tokenizer array whose contents lie past the end of
        // the buffer — only the array header made it in.
        let key = b"tokenizer.ggml.tokens";
        b.extend_from_slice(&(key.len() as u64).to_le_bytes());
        b.extend_from_slice(key);
        b.extend_from_slice(&GGUF_TYPE_ARRAY.to_le_bytes());
        b.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
        b.extend_from_slice(&248_320u64.to_le_bytes());

        let info = parse_gguf_header(&b).expect("partial parse must succeed");
        assert_eq!(info.n_layers, 40);
        assert_eq!(info.n_head_kv, Some(2));
    }

    /// Once every wanted key is filled the parser must stop reading — a
    /// later duplicate that would overwrite an already-parsed value proves
    /// the stop happened by design, not because the buffer ran out. The
    /// wanted set includes `full_attention_interval`, which hybrid models
    /// write AFTER the attention dims, so the fixture is hybrid-shaped.
    #[test]
    fn stops_reading_once_every_wanted_key_is_in_hand() {
        let data = header(&[
            (b"arch.block_count", 40),
            (b"arch.embedding_length", 2048),
            (b"arch.attention.head_count", 16),
            (b"arch.attention.head_count_kv", 2),
            (b"arch.attention.key_length", 256),
            (b"arch.attention.value_length", 256),
            (b"arch.full_attention_interval", 4),
            (b"arch.block_count", 99),
        ]);
        let info = parse_gguf_header(&data).expect("parses");
        assert_eq!(info.n_layers, 40);
        assert_eq!(info.full_attention_interval, Some(4));
    }

    /// The Ornith case: a `qwen35moe` GGUF that ships WITHOUT the explicit
    /// interval key. Upstream hardcodes the default 4 before the optional
    /// read, so the discount must apply from the architecture alone.
    #[test]
    fn qwen35_architectures_default_the_attention_interval() {
        let mut b = Vec::new();
        b.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&2u64.to_le_bytes());
        put_str(b"general.architecture", b"qwen35moe", &mut b);
        put_u32(b"qwen35moe.block_count", 64, &mut b);

        let info = parse_gguf_header(&b).expect("parses");
        assert_eq!(info.architecture.as_deref(), Some("qwen35moe"));
        assert_eq!(info.full_attention_interval, None);
        // 64 layers at the defaulted cadence of 4: 16 pay KV, not 64.
        assert_eq!(info.kv_paying_layers(64), 16);

        // An unknown architecture without the key stays uniform.
        let dense = GgufInfo {
            architecture: Some("llama".to_string()),
            ..info.clone()
        };
        assert_eq!(dense.kv_paying_layers(64), 64);
    }

    /// A dense model never writes `full_attention_interval`; the parser
    /// scans to the end (skipping unwanted values) and everything else is
    /// still read correctly.
    #[test]
    fn dense_models_parse_without_an_attention_interval() {
        let data = header(&[
            (b"arch.block_count", 40),
            (b"arch.embedding_length", 2048),
            (b"arch.attention.head_count", 16),
            (b"arch.attention.head_count_kv", 2),
            (b"arch.attention.key_length", 256),
            (b"arch.attention.value_length", 256),
        ]);
        let info = parse_gguf_header(&data).expect("parses");
        assert_eq!(info.n_layers, 40);
        assert_eq!(info.full_attention_interval, None);
    }

    /// The hybrid-SSM discount: with `full_attention_interval=4` only one
    /// layer in four pays KV, so at a large context the sizer offloads far
    /// more layers than the uniform charge would allow. Measured live at
    /// `--ctx-size 262144` on Qwen3.6-35B-A3B: the uniform math stopped
    /// with 8 GiB of VRAM idle.
    #[test]
    fn hybrid_ssm_models_offload_more_layers_at_large_context() {
        let dense = GgufInfo {
            n_layers: 64,
            n_embd: Some(5120),
            n_head: Some(24),
            n_head_kv: Some(4),
            key_length: Some(256),
            value_length: Some(256),
            full_attention_interval: None,
            architecture: None,
        };
        let hybrid = GgufInfo {
            full_attention_interval: Some(4),
            ..dense.clone()
        };
        let free = Some(16u64 * 1024 * 1024 * 1024);
        let file_size = 22u64 * 1024 * 1024 * 1024;
        let ctx = 131072;
        let dense_layers = match compute_fit(free, Some(&dense), file_size, ctx, 2.0, 2.0) {
            FitDecision::Partial { layers, .. } => layers,
            other => panic!("expected partial, got {other:?}"),
        };
        let hybrid_layers = match compute_fit(free, Some(&hybrid), file_size, ctx, 2.0, 2.0) {
            FitDecision::Partial { layers, .. } => layers,
            other => panic!("expected partial, got {other:?}"),
        };
        assert!(
            hybrid_layers > dense_layers,
            "hybrid must offload more: {hybrid_layers} vs {dense_layers}"
        );
    }

    /// The whole point: with the real head dimension the sizer charges more
    /// per layer, so it offloads fewer layers — and the load succeeds instead
    /// of running out of VRAM.
    #[test]
    fn real_head_dim_offloads_no_more_layers_than_the_assumption_did() {
        let with_explicit = GgufInfo {
            n_layers: 36,
            n_embd: Some(2560),
            n_head: Some(32),
            n_head_kv: Some(8),
            key_length: Some(128),
            value_length: Some(128),
            full_attention_interval: None,
            architecture: None,
        };
        let assumed = GgufInfo {
            key_length: None,
            value_length: None,
            ..with_explicit.clone()
        };
        // A tight-but-not-hopeless fit, so both variants land in `Partial` and
        // the layer counts are actually comparable (with ample VRAM both would
        // report `FitsFully` and the comparison would be vacuous).
        let free = 4 * 1024 * 1024 * 1024;
        let file = 2_500_000_000;
        let ctx = 32768;

        let layers = |i: &GgufInfo| match compute_fit(Some(free), Some(i), file, ctx, 2.0, 2.0) {
            FitDecision::Partial { layers, .. } => layers,
            FitDecision::FitsFully => i32::MAX,
            FitDecision::Unknown { .. } => -1,
        };
        assert!(
            layers(&with_explicit) < layers(&assumed),
            "the real head_dim must be charged as more expensive than the \
             n_embd/n_head under-estimate: explicit={} assumed={}",
            layers(&with_explicit),
            layers(&assumed),
        );
    }
}

#[cfg(test)]
mod moe_layout_tests {
    use super::*;

    /// Build a synthetic GGUF: `tensors` is `(name, byte_size)` in on-disk
    /// order; offsets are assigned sequentially starting at 0. `alignment`,
    /// when `Some`, is written as a `general.alignment` metadata key so the
    /// parser is exercised on a non-default value instead of always falling
    /// through to its own default. Returns `(buffer, file_size)` where the
    /// buffer covers the *whole* synthetic file (dummy tensor-data bytes
    /// included), unlike production's 8 MiB-capped read — `file_size` is
    /// still passed separately, exactly as `read_gguf_moe_layout` does, so
    /// the "last tensor sized from `file_size`" path is exercised the same
    /// way either way.
    fn make_gguf_with_tensors(alignment: Option<u64>, tensors: &[(&str, u64)]) -> (Vec<u8>, u64) {
        let mut b = Vec::new();
        b.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes()); // version
        b.extend_from_slice(&(tensors.len() as u64).to_le_bytes()); // tensor_count
        b.extend_from_slice(&(if alignment.is_some() { 1u64 } else { 0u64 }).to_le_bytes());

        if let Some(a) = alignment {
            let key = b"general.alignment";
            b.extend_from_slice(&(key.len() as u64).to_le_bytes());
            b.extend_from_slice(key);
            b.extend_from_slice(&GGUF_TYPE_UINT32.to_le_bytes());
            b.extend_from_slice(&(a as u32).to_le_bytes());
        }

        let mut offset = 0u64;
        for (name, size) in tensors {
            let name_bytes = name.as_bytes();
            b.extend_from_slice(&(name_bytes.len() as u64).to_le_bytes());
            b.extend_from_slice(name_bytes);
            b.extend_from_slice(&1u32.to_le_bytes()); // n_dimensions
            b.extend_from_slice(&1u64.to_le_bytes()); // dims[0] (unused by the parser)
            b.extend_from_slice(&GGUF_TYPE_FLOAT32.to_le_bytes()); // ggml type (unused)
            b.extend_from_slice(&offset.to_le_bytes());
            offset += size;
        }
        let total_tensor_bytes = offset;

        let align = alignment.unwrap_or(32);
        let data_start = align_up(b.len() as u64, align).unwrap();
        b.resize(data_start as usize, 0);
        b.resize((data_start + total_tensor_bytes) as usize, 0xAA);

        let file_size = b.len() as u64;
        (b, file_size)
    }

    #[test]
    fn separates_expert_and_non_expert_bytes_per_layer() {
        let (data, file_size) = make_gguf_with_tensors(
            None,
            &[
                ("token_embd.weight", 1000),
                ("blk.0.attn_q.weight", 100),
                ("blk.0.ffn_gate_exps.weight", 3000),
                ("blk.0.ffn_up_exps.weight", 3000),
                ("blk.0.ffn_down_exps.weight", 3000),
                ("blk.1.attn_q.weight", 100),
                ("blk.1.ffn_gate_exps.weight", 4000),
                ("blk.1.ffn_up_exps.weight", 4000),
                ("blk.1.ffn_down_exps.weight", 4000),
                ("output_norm.weight", 50),
            ],
        );

        let layout = parse_gguf_moe_layout(&data, file_size, 2).expect("parses");
        assert!(layout.is_moe());
        assert_eq!(layout.non_expert_bytes, 1000 + 100 + 100 + 50);
        assert_eq!(layout.expert_bytes_per_layer, vec![9000, 12000]);
        assert_eq!(
            layout.expert_bytes_per_layer.iter().sum::<u64>(),
            21000
        );
    }

    #[test]
    fn dense_model_has_no_experts() {
        let (data, file_size) = make_gguf_with_tensors(
            None,
            &[
                ("token_embd.weight", 1000),
                ("blk.0.attn_q.weight", 100),
                ("blk.0.ffn_gate.weight", 300),
                ("blk.0.ffn_up.weight", 300),
                ("blk.0.ffn_down.weight", 300),
            ],
        );
        let layout = parse_gguf_moe_layout(&data, file_size, 1).expect("parses");
        assert!(!layout.is_moe());
        assert_eq!(layout.non_expert_bytes, 2000);
        assert_eq!(layout.expert_bytes_per_layer, vec![0]);
    }

    #[test]
    fn respects_a_custom_alignment_key() {
        // A single short tensor name keeps the cursor position after the
        // tensor-info table off any 32-byte boundary, so this only passes if
        // the parser actually reads `general.alignment` (64 here) instead of
        // silently defaulting to 32.
        let (data, file_size) =
            make_gguf_with_tensors(Some(64), &[("blk.0.ffn_gate_exps.weight", 500)]);
        let layout = parse_gguf_moe_layout(&data, file_size, 1).expect("parses");
        assert_eq!(layout.expert_bytes_per_layer, vec![500]);
        assert_eq!(layout.non_expert_bytes, 0);
    }

    #[test]
    fn truncated_tensor_info_returns_none() {
        let (data, file_size) = make_gguf_with_tensors(
            None,
            &[("blk.0.ffn_gate_exps.weight", 500), ("blk.1.attn_q.weight", 100)],
        );
        // Cut a few bytes into the first tensor-info record (past the
        // 24-byte header, mid-way through the name), not the trailing dummy
        // tensor-data bytes — the parser never reads that far, so trimming
        // only the tail wouldn't actually exercise a truncated *header*.
        let truncated = &data[..30];
        assert!(parse_gguf_moe_layout(truncated, file_size, 2).is_none());
    }

    #[test]
    fn layer_index_and_expert_marker_parsing() {
        assert_eq!(tensor_layer_index("blk.0.attn_q.weight"), Some(0));
        assert_eq!(tensor_layer_index("blk.17.ffn_gate_exps.weight"), Some(17));
        assert_eq!(tensor_layer_index("token_embd.weight"), None);
        assert_eq!(tensor_layer_index("output_norm.weight"), None);

        assert!(is_expert_tensor_name("blk.0.ffn_gate_exps.weight"));
        assert!(is_expert_tensor_name("blk.0.ffn_up_chexps.weight"));
        assert!(is_expert_tensor_name("blk.0.ffn_gate_up_exps.weight"));
        assert!(!is_expert_tensor_name("blk.0.ffn_gate.weight"));
        assert!(!is_expert_tensor_name("blk.0.attn_q.weight"));
    }
}

#[cfg(test)]
mod moe_fit_tests {
    use super::*;

    const F16: (f64, f64) = (2.0, 2.0);

    fn info(n_layers: u32) -> GgufInfo {
        GgufInfo {
            n_layers,
            n_embd: None,
            n_head: None,
            n_head_kv: None,
            key_length: None,
            value_length: None,
            full_attention_interval: None,
            architecture: None,
        }
    }

    #[test]
    fn not_moe_when_layout_has_no_experts() {
        let layout = MoeLayout {
            non_expert_bytes: 5_000_000_000,
            expert_bytes_per_layer: vec![0, 0, 0],
        };
        let d = compute_moe_fit(
            Some(40 * 1024 * 1024 * 1024),
            Some(&info(3)),
            Some(&layout),
            4096,
            F16.0,
            F16.1,
        );
        assert_eq!(d, MoeFitDecision::NotMoe);
    }

    #[test]
    fn not_moe_when_free_vram_unknown() {
        let layout = MoeLayout {
            non_expert_bytes: 1_000_000_000,
            expert_bytes_per_layer: vec![1_000_000_000],
        };
        let d = compute_moe_fit(None, Some(&info(1)), Some(&layout), 4096, F16.0, F16.1);
        assert_eq!(d, MoeFitDecision::NotMoe);
    }

    #[test]
    fn everything_fits_needs_no_eviction() {
        // 4 layers, tiny non-expert + expert bytes, huge free VRAM.
        let layout = MoeLayout {
            non_expert_bytes: 1_000_000_000,
            expert_bytes_per_layer: vec![500_000_000; 4],
        };
        let d = compute_moe_fit(
            Some(40 * 1024 * 1024 * 1024),
            Some(&info(4)),
            Some(&layout),
            4096,
            F16.0,
            F16.1,
        );
        assert_eq!(d, MoeFitDecision::Proceed { n_cpu_moe: 0 });
    }

    /// Budget only has room for one layer's worth of experts after
    /// non-expert+KV: the LAST layer (highest index) must be the one kept,
    /// and eviction must cover the contiguous prefix `0..n_cpu_moe`, not an
    /// arbitrary subset — this is the one real invariant `--n-cpu-moe`
    /// requires from this function.
    #[test]
    fn evicts_a_contiguous_prefix_from_the_lowest_layers() {
        let per_layer_expert = 2_000_000_000u64; // 2 GB/layer
        let layout = MoeLayout {
            non_expert_bytes: 1_000_000_000, // 1 GB
            expert_bytes_per_layer: vec![per_layer_expert; 4],
        };
        // usable ≈ free*0.97 - 640MiB. Pick free VRAM so that after non-expert
        // (1GB) + a small KV term, there's room for exactly ~1 layer of
        // experts (2GB) but not 2 (4GB).
        let free = 4_200_000_000u64;
        let d = compute_moe_fit(Some(free), Some(&info(4)), Some(&layout), 512, F16.0, F16.1);
        match d {
            MoeFitDecision::Proceed { n_cpu_moe } => {
                assert_eq!(n_cpu_moe, 3, "expected layers 0..=2 evicted, layer 3 kept");
            }
            other => panic!("expected Proceed, got {other:?}"),
        }
    }

    #[test]
    fn experts_only_not_enough_falls_back_to_partial_dense() {
        // Non-expert weights alone already exceed the VRAM budget, even with
        // every expert assumed off-GPU.
        let layout = MoeLayout {
            non_expert_bytes: 20_000_000_000, // 20 GB
            expert_bytes_per_layer: vec![1_000_000_000; 2],
        };
        let free = 8_000_000_000u64; // 8 GB free
        let d = compute_moe_fit(Some(free), Some(&info(2)), Some(&layout), 4096, F16.0, F16.1);
        match d {
            MoeFitDecision::ProceedCpuMoeAndPartial { gpu_layers } => {
                assert!(gpu_layers >= 0, "must still resolve to a loadable split");
                assert!(
                    gpu_layers < 2,
                    "20GB of non-expert weight can't fit fully in 8GB free"
                );
            }
            other => panic!("expected ProceedCpuMoeAndPartial, got {other:?}"),
        }
    }

    #[test]
    fn extreme_case_still_resolves_to_a_startable_split() {
        // Pathological: enormous non-expert weight, almost no free VRAM.
        // Must still resolve to *something* loadable (gpu_layers as low as
        // 0 — fully CPU — is an acceptable, expected outcome here), never an
        // unresolvable decision.
        let layout = MoeLayout {
            non_expert_bytes: 500_000_000_000, // 500 GB (absurd on purpose)
            expert_bytes_per_layer: vec![50_000_000_000; 8],
        };
        let free = 2_000_000_000u64; // 2 GB free
        let d = compute_moe_fit(Some(free), Some(&info(8)), Some(&layout), 4096, F16.0, F16.1);
        match d {
            MoeFitDecision::ProceedCpuMoeAndPartial { gpu_layers } => {
                assert!(gpu_layers >= 0, "even the worst case must resolve, not abort");
            }
            other => panic!("expected ProceedCpuMoeAndPartial even in the extreme case, got {other:?}"),
        }
    }
}

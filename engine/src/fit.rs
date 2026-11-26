//! Sizing a load to the GPU: what `--fit` decides, on by default.
//!
//! Before every load — at launch, and before each model `serve` swaps in —
//! this decides how many layers go on the GPU, which MoE experts stay in
//! system RAM (`--cpu-moe`/`--n-cpu-moe`), how large an expert cache comes out
//! (`--moe-cache`), and whether the micro-batch and memory mapping change with
//! it, so that the model, its KV cache and the compute buffers fit in the VRAM
//! that is actually free. `--gpu-layers` and the MoE flags are ceilings it
//! keeps to, never values it overrides upward; `--no-fit` turns it off.
//!
//! Free and total VRAM come from ggml's device registry ([`vram_bytes`]), on
//! every GPU backend we ship. Where nothing can be read (no GPU, or before the
//! backend is initialised) the user's own `--gpu-layers` is used as given.
//!
//! The layer count and attention dimensions are read from the GGUF header
//! (`<arch>.block_count`, …) with a small, bounds-checked binary parser, and
//! the on-disk file size is used as a proxy for total weight bytes.

use std::io::IsTerminal;
use std::path::{Path, PathBuf};

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
    /// Layers of a multi-token-prediction head (`<arch>.nextn_predict_layers`),
    /// counted in `n_layers`, when the model carries one: what `--mtp`'s draft
    /// context runs (see [`mtp_reserve_bytes`]).
    pub nextn_layers: Option<u32>,
    /// The context the model was trained for (`<arch>.context_length`), when
    /// declared: the ceiling of `--ctx-size auto`.
    pub context_length: Option<u32>,
    /// Its sliding-window layers, on a model that has them and says which
    /// (Gemma 2, 3 and 4, gpt-oss): see [`SlidingWindow`].
    pub window: Option<SlidingWindow>,
    /// A MoE model's experts per layer (`<arch>.expert_count`) and those
    /// each token uses (`<arch>.expert_used_count`): what writing a token
    /// reads of them (see `measured_fit`).
    pub n_expert: Option<u32>,
    pub n_expert_used: Option<u32>,
}

/// A model's sliding-window attention layers, which keep only the last
/// `n_swa` tokens of each slot (`swa_full = false`, as llama-server and the
/// engine load them), and how many cells llama.cpp allocates for them.
///
/// Without it every layer is charged the whole context. On Gemma 4 31B that
/// was not even the worst of it: its KV heads are given per layer (16 on the
/// 50 window layers, 4 on the 10 global ones), an array the header reader
/// skipped, so sizing fell back to 128 bytes per token per layer and charged
/// about 0.4 GB of KV where llama.cpp allocates 13 GB with 8 slots at
/// `-c 49152`: "fits fully", then out of memory (Leonardo, 19-11-2026).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SlidingWindow {
    /// The window, in tokens (`<arch>.attention.sliding_window`).
    pub n_swa: u32,
    /// Per layer, whether it is a window layer
    /// (`<arch>.attention.sliding_window_pattern`, or the architecture's
    /// period, as llama.cpp's `load_swa_pattern`).
    pub layers: Vec<bool>,
    /// Per layer, its KV heads, when the header gives them per layer.
    pub head_count_kv: Vec<u32>,
    /// The window layers' head dimensions, when they differ from the others'
    /// (`key_length_swa`, `value_length_swa`).
    pub key_length: Option<u32>,
    pub value_length: Option<u32>,
    /// How the load being sized lays out its KV cache (see [`KvLayout`]);
    /// `None` when sizing does not know, and then each window layer is
    /// charged the whole context, which is never less than llama.cpp
    /// allocates.
    pub layout: Option<KvLayout>,
}

/// How a load lays out its KV cache, which decides what the window layers
/// take: its slots (`--batch-size`, 1 for the sequential engine), its
/// micro-batch (`--n-ubatch`) and `--kv-unified`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvLayout {
    pub slots: u32,
    pub n_ubatch: u32,
    pub unified: bool,
}

impl KvLayout {
    /// The layout a load of `batch_size` slots (0: the sequential engine,
    /// one sequence) takes.
    pub fn new(batch_size: usize, n_ubatch: u32, unified: bool) -> Self {
        Self {
            slots: u32::try_from(batch_size.max(1)).unwrap_or(u32::MAX),
            n_ubatch: n_ubatch.max(1),
            unified,
        }
    }
}

impl SlidingWindow {
    /// The cells a window layer has in a context of `ctx` tokens, as
    /// llama.cpp's `llama_kv_cache_iswa` sizes them: per sequence (one
    /// stream each) the window and a micro-batch, padded to 256 and never
    /// more than the sequence's share of the context; with `--kv-unified`,
    /// one stream holding every sequence's window.
    fn cells(&self, ctx: u32) -> u64 {
        let pad = |n: u64| n.div_ceil(256) * 256;
        let ctx = u64::from(ctx);
        let Some(layout) = self.layout else {
            return ctx;
        };
        let n_swa = u64::from(self.n_swa);
        let slots = u64::from(layout.slots.max(1));
        let ubatch = u64::from(layout.n_ubatch);
        if layout.unified {
            pad(ctx.min(n_swa * slots + ubatch))
        } else {
            let per_seq = pad(ctx / slots);
            slots * pad(per_seq.min(n_swa + ubatch))
        }
    }
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

    /// Bytes of KV cache `ctx` tokens take on a model with sliding-window
    /// layers, every layer counted with its own heads, head dimensions and
    /// cells (see [`SlidingWindow`]); `None` without a window, or without the
    /// dimensions to count one with, and the uniform estimate applies.
    pub(crate) fn windowed_kv_bytes(&self, ctx: u32, bpe_k: f64, bpe_v: f64) -> Option<f64> {
        let w = self.window.as_ref()?;
        let n = self.n_layers as usize;
        if w.n_swa == 0 || w.layers.len() < n {
            return None;
        }
        let derived = || -> Option<u32> {
            let n_embd = self.n_embd.filter(|&v| v > 0)?;
            let n_head = self.n_head.filter(|&v| v > 0)?;
            Some(n_embd / n_head)
        };
        let k_full = self.key_length.filter(|&v| v > 0).or_else(derived)?;
        let v_full = self.value_length.filter(|&v| v > 0).unwrap_or(k_full);
        let k_swa = w.key_length.filter(|&v| v > 0).unwrap_or(k_full);
        let v_swa = w.value_length.filter(|&v| v > 0).unwrap_or(v_full);
        let window_cells = w.cells(ctx) as f64;
        let mut bytes = 0.0;
        for (i, &is_swa) in w.layers.iter().take(n).enumerate() {
            let heads = match w.head_count_kv.get(i) {
                Some(&h) => h,
                None => self.n_head_kv.filter(|&v| v > 0)?,
            };
            let (k, v, cells) = if is_swa {
                (k_swa, v_swa, window_cells)
            } else {
                (k_full, v_full, f64::from(ctx))
            };
            bytes += cells * f64::from(heads) * (f64::from(k) * bpe_k + f64::from(v) * bpe_v);
        }
        Some(bytes)
    }

    /// Bytes of KV cache one paying layer takes at `ctx` tokens, for sizers
    /// that charge layers one at a time: on a model with sliding-window
    /// layers, its whole KV over its layers (window and global layers
    /// alternate through the model, so a block of them pays about its share).
    fn kv_bytes_per_paying_layer(&self, ctx: u32, bpe_k: f64, bpe_v: f64) -> f64 {
        if let Some(total) = self.windowed_kv_bytes(ctx, bpe_k, bpe_v) {
            return total / f64::from(self.n_layers.max(1));
        }
        match self.kv_elems_per_token_per_layer() {
            Some((k_elems, v_elems)) => f64::from(ctx) * (k_elems * bpe_k + v_elems * bpe_v),
            None => f64::from(ctx) * FALLBACK_KV_BYTES_PER_TOKEN_PER_LAYER,
        }
    }

    /// The same model, sized for a load laid out as `layout`: what its window
    /// layers take depends on it (see [`SlidingWindow::cells`]).
    pub fn with_kv_layout(mut self, layout: KvLayout) -> Self {
        if let Some(window) = self.window.as_mut() {
            window.layout = Some(layout);
        }
        self
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

    /// Read an array of integers or flags (bool, any integer type), after its
    /// `GGUF_TYPE_ARRAY` tag: one per layer, so a count past `MAX_LAYERS` is
    /// not one of the arrays wanted, and is skipped with `None`... after
    /// which parsing stops, as it does for any malformed value.
    fn uint_array(&mut self) -> Option<Vec<u64>> {
        let elem_type = self.u32()?;
        let count = usize::try_from(self.u64()?).ok()?;
        if count > MAX_LAYERS as usize {
            return None;
        }
        (0..count)
            .map(|_| match elem_type {
                GGUF_TYPE_BOOL => self.take(1).map(|b| u64::from(b[0])),
                other => self.read_uint_as_u64(other),
            })
            .collect()
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
    let mut nextn_layers: Option<u32> = None;
    let mut context_length: Option<u32> = None;
    let mut n_expert: Option<u32> = None;
    let mut n_expert_used: Option<u32> = None;
    let mut architecture: Option<String> = None;
    // Sliding-window layers (see `SlidingWindow`): the window, which layers
    // have it (an array of flags, or a period), their head dimensions, and
    // the KV heads when given per layer, as arrays the loop below reads
    // rather than skips.
    let mut n_swa: Option<u32> = None;
    let mut swa_period: Option<u32> = None;
    let mut swa_layers: Option<Vec<bool>> = None;
    let mut head_count_kv_layers: Option<Vec<u32>> = None;
    let mut key_length_swa: Option<u32> = None;
    let mut value_length_swa: Option<u32> = None;

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

        // The two arrays wanted: the KV heads per layer (Gemma 4) and the
        // window layers' flags. Any other array is skipped below.
        if value_type == GGUF_TYPE_ARRAY
            && (key_bytes.ends_with(b".attention.head_count_kv")
                || key_bytes.ends_with(b".attention.sliding_window_pattern"))
        {
            let Some(values) = c.uint_array() else {
                break;
            };
            if key_bytes.ends_with(b".attention.head_count_kv") {
                // The most any layer has stands for all where one number is
                // read: never less than a layer takes.
                if n_head_kv.is_none() {
                    n_head_kv = values.iter().max().and_then(|&v| u32::try_from(v).ok());
                }
                head_count_kv_layers = values.iter().map(|&v| u32::try_from(v).ok()).collect();
            } else {
                swa_layers = Some(values.iter().map(|&v| v != 0).collect());
            }
            continue;
        }

        // Pick which field (if any) this key feeds. All are scalar integers;
        // an array-typed match is ignored (skipped) to stay aligned.
        let target: Option<&mut Option<u32>> = if value_type == GGUF_TYPE_ARRAY {
            None
        } else if key_bytes.ends_with(b".attention.sliding_window") {
            Some(&mut n_swa)
        } else if key_bytes.ends_with(b".attention.sliding_window_pattern") {
            Some(&mut swa_period)
        } else if key_bytes.ends_with(b".attention.key_length_swa") {
            Some(&mut key_length_swa)
        } else if key_bytes.ends_with(b".attention.value_length_swa") {
            Some(&mut value_length_swa)
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
        } else if key_bytes.ends_with(b".nextn_predict_layers") {
            Some(&mut nextn_layers)
        } else if key_bytes.ends_with(b".context_length") {
            Some(&mut context_length)
        } else if key_bytes.ends_with(b".expert_count") {
            Some(&mut n_expert)
        } else if key_bytes.ends_with(b".expert_used_count") {
            Some(&mut n_expert_used)
        } else {
            None
        };

        let advanced = match target {
            Some(slot) => match c.read_uint_as_u64(value_type) {
                Some(v) => {
                    // No saturation: these counts come from the file, and a
                    // value that does not fit u32 is corrupt — recording
                    // u32::MAX for it would send the sizer into a
                    // ~4-billion-iteration loop below.
                    *slot = u32::try_from(v).ok();
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
        // case where the buffer ends first. `nextn_predict_layers` is in the
        // set for the same reason: a model with an MTP head writes it after
        // `full_attention_interval` (qwen35, key 32 against 30).
        if n_layers.is_some()
            && n_embd.is_some()
            && n_head.is_some()
            && n_head_kv.is_some()
            && key_length.is_some()
            && value_length.is_some()
            && full_attention_interval.is_some()
            && nextn_layers.is_some()
            && context_length.is_some()
        {
            // Not waiting for the expert counts, which a dense model never
            // has: a MoE writes them with its attention dims, before
            // `full_attention_interval` and `nextn_predict_layers`.
            break;
        }
    }

    let n_layers = n_layers?;
    let window = n_swa.filter(|&n| n > 0).and_then(|n_swa| {
        let layers = swa_layers.or_else(|| {
            let (period, dense_first) = match swa_period {
                Some(period) => (period, false),
                None => swa_period_of(architecture.as_deref()?)?,
            };
            Some(swa_pattern(n_layers, period, dense_first))
        })?;
        Some(SlidingWindow {
            n_swa,
            layers,
            head_count_kv: head_count_kv_layers.unwrap_or_default(),
            key_length: key_length_swa,
            value_length: value_length_swa,
            layout: None,
        })
    });
    Some(GgufInfo {
        n_layers,
        n_embd,
        n_head,
        n_head_kv,
        key_length,
        value_length,
        full_attention_interval,
        architecture,
        nextn_layers,
        context_length,
        window,
        n_expert,
        n_expert_used,
    })
}

/// The period of an architecture's sliding-window layers when its header
/// gives none, and whether the period starts with a global layer: the
/// defaults llama.cpp's models pass to `load_swa_pattern`, for the
/// architectures EuLLM users run.
fn swa_period_of(architecture: &str) -> Option<(u32, bool)> {
    match architecture {
        "gemma2" | "gpt-oss" => Some((2, false)),
        "gemma3" | "gemma-embedding" => Some((6, false)),
        "gemma3n" => Some((5, false)),
        "cohere2" | "exaone4" => Some((4, false)),
        _ => None,
    }
}

/// llama.cpp's `set_swa_pattern`: with a period of `n`, the last layer of
/// every `n` is global and the others have a window (or, `dense_first`, the
/// first); 0 makes every layer a window layer.
fn swa_pattern(n_layers: u32, period: u32, dense_first: bool) -> Vec<bool> {
    (0..n_layers)
        .map(|il| {
            period == 0
                || if dense_first {
                    il % period != 0
                } else {
                    il % period < period - 1
                }
        })
        .collect()
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
#[derive(Debug, Clone, Default)]
pub struct MoeLayout {
    /// Bytes of every tensor that is not part of any layer's expert set.
    pub non_expert_bytes: u64,
    /// The part of `non_expert_bytes` in each transformer block (attention,
    /// norms, routers, shared experts), indexed as `expert_bytes_per_layer`;
    /// the rest is the input and output layers. Empty when not read: a
    /// split across GPUs then spreads `non_expert_bytes` evenly.
    pub non_expert_bytes_per_layer: Vec<u64>,
    /// Expert-tensor bytes for each transformer block, indexed by layer
    /// number parsed from the tensor name (`blk.<i>...`). A dense
    /// (non-MoE) model has every entry `0`.
    pub expert_bytes_per_layer: Vec<u64>,
    /// The largest single expert tensor, in bytes: what one slot of the
    /// prefetch has to hold (see [`prefetch_slot_bytes`]). `0` for a dense
    /// model.
    pub largest_expert_tensor_bytes: u64,
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
    if n_layers > MAX_LAYERS {
        // Same corrupt-header class `compute_fit` refuses above: without
        // this the scratch vector below prices ~8 bytes per layer off one
        // untrusted integer.
        return None;
    }
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
        // A part that carries only metadata (Unsloth's UD-* splits put the
        // tokenizer and hyperparameters alone in part 1): it holds no
        // weights, which is a layout of zero bytes, not an unreadable file.
        // Returning `None` here made `read_model_moe_layout` give up on the
        // whole model, which then looked dense to `--fit`.
        return Some(MoeLayout {
            non_expert_bytes: 0,
            non_expert_bytes_per_layer: vec![0u64; n_layers as usize],
            expert_bytes_per_layer: vec![0u64; n_layers as usize],
            largest_expert_tensor_bytes: 0,
        });
    }

    let data_start = align_up(c.pos as u64, alignment)?;
    tensors.sort_by_key(|(_, offset)| *offset);

    let mut non_expert_bytes: u64 = 0;
    let mut non_expert_bytes_per_layer = vec![0u64; n_layers as usize];
    let mut expert_bytes_per_layer = vec![0u64; n_layers as usize];
    let mut largest_expert_tensor_bytes: u64 = 0;

    for (idx, (name, offset)) in tensors.iter().enumerate() {
        let size = match tensors.get(idx + 1) {
            Some((_, next_offset)) => next_offset.checked_sub(*offset)?,
            None => file_size.checked_sub(data_start)?.checked_sub(*offset)?,
        };

        match (is_expert_tensor_name(name), tensor_layer_index(name)) {
            (true, Some(layer)) if (layer as usize) < expert_bytes_per_layer.len() => {
                expert_bytes_per_layer[layer as usize] += size;
                largest_expert_tensor_bytes = largest_expert_tensor_bytes.max(size);
            }
            // llama.cpp keeps a model's input embeddings in host memory and reads
            // them a row at a time, whatever is offloaded; Qwen3.8-Flash-Next
            // carries a 28.8 GB one per layer (`per_layer_token_embd`), which
            // counted as VRAM left no room for the expert cache at all.
            _ if is_host_only_tensor_name(name) => {}
            (false, Some(layer)) if (layer as usize) < non_expert_bytes_per_layer.len() => {
                non_expert_bytes_per_layer[layer as usize] += size;
                non_expert_bytes += size;
            }
            _ => non_expert_bytes += size,
        }
    }

    Some(MoeLayout {
        non_expert_bytes,
        non_expert_bytes_per_layer,
        expert_bytes_per_layer,
        largest_expert_tensor_bytes,
    })
}

/// Tensors that stay in host memory however many layers are offloaded: the
/// per-layer input embedding, a lookup table (Gemma 3n, Qwen3.8-Flash-Next).
fn is_host_only_tensor_name(name: &str) -> bool {
    name == "per_layer_token_embd.weight"
}

/// Every file of a model on disk, in order: all the parts of a split GGUF
/// (`<name>-00001-of-00009.gguf`, as llama.cpp's gguf-split names them) when
/// `path` is one of them and every part is there, else `path` alone.
///
/// llama.cpp loads every part from the path of the first, and every size
/// taken from that one file was a ninth of the model: `--fit` called
/// DeepSeek-V3.1 Q4_K_M "45.14 GiB, fits fully" for 378 GiB in nine parts,
/// and `--fit-strict` let it start a load that ran an hour on LUMI
/// (06-10-2026) instead of refusing it in a second.
pub fn gguf_parts(path: &Path) -> Vec<PathBuf> {
    if let Some((prefix, count)) = path
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(split_gguf_name)
    {
        let dir = path.parent().unwrap_or_else(|| Path::new(""));
        let parts: Vec<PathBuf> = (1..=count)
            .map(|i| dir.join(format!("{prefix}-{i:05}-of-{count:05}.gguf")))
            .collect();
        if parts.iter().all(|p| p.is_file()) {
            return parts;
        }
    }
    vec![path.to_path_buf()]
}

/// `("<name>", count)` for `<name>-NNNNN-of-MMMMM.gguf`, gguf-split's naming.
fn split_gguf_name(name: &str) -> Option<(&str, u32)> {
    let stem = name.strip_suffix(".gguf")?;
    let (rest, count) = stem.rsplit_once("-of-")?;
    let (prefix, index) = rest.rsplit_once('-')?;
    let five_digits = |s: &str| s.len() == 5 && s.bytes().all(|b| b.is_ascii_digit());
    if prefix.is_empty() || !five_digits(index) || !five_digits(count) {
        return None;
    }
    let count: u32 = count.parse().ok()?;
    let index: u32 = index.parse().ok()?;
    (count >= 1 && (1..=count).contains(&index)).then_some((prefix, count))
}

/// The bytes of the whole model: every part of a split GGUF.
pub fn model_file_bytes(path: &Path) -> u64 {
    gguf_parts(path)
        .iter()
        .filter_map(|p| std::fs::metadata(p).ok())
        .map(|m| m.len())
        .sum()
}

/// [`read_gguf_moe_layout`] over every part of a model: each part's header
/// describes the tensors in that part, so the layout is their sum. `None`
/// if any part cannot be read, as for one file.
pub fn read_model_moe_layout(path: &Path, n_layers: u32) -> Option<MoeLayout> {
    let mut total: Option<MoeLayout> = None;
    for part in gguf_parts(path) {
        let size = std::fs::metadata(&part).ok()?.len();
        let layout = read_gguf_moe_layout(&part, size, n_layers)?;
        total = Some(match total {
            None => layout,
            Some(mut t) => {
                t.non_expert_bytes += layout.non_expert_bytes;
                for (sum, part_bytes) in t
                    .non_expert_bytes_per_layer
                    .iter_mut()
                    .zip(layout.non_expert_bytes_per_layer)
                {
                    *sum += part_bytes;
                }
                for (sum, part_bytes) in t
                    .expert_bytes_per_layer
                    .iter_mut()
                    .zip(layout.expert_bytes_per_layer)
                {
                    *sum += part_bytes;
                }
                t
            }
        });
    }
    total
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

/// The GPUs llama.cpp places a model on, in its order: ggml's devices of
/// type GPU, less any whose `device_id` (its PCI address) a device before it
/// already has. A build with two backends lists one card under both: CUDA and
/// Vulkan each see an NVIDIA card, and llama.cpp loads on the first only
/// ("skipping device … with the same id", `llama_model_load`), where a sum
/// over every device would size against its VRAM twice.
pub(crate) fn gpu_devices() -> Vec<llama_cpp_sys_2::ggml_backend_dev_t> {
    use llama_cpp_sys_2::{
        GGML_BACKEND_DEVICE_TYPE_GPU, ggml_backend_dev_count, ggml_backend_dev_get,
        ggml_backend_dev_get_props, ggml_backend_dev_props, ggml_backend_dev_type,
    };

    let mut devices = Vec::new();
    let mut ids = Vec::new();
    // SAFETY: the registry is a process-global initialised by the ggml
    // backends when they load; every index below the count is valid, the
    // properties are written into a struct we own, and `device_id` is a C
    // string the device owns for the life of the process.
    unsafe {
        for i in 0..ggml_backend_dev_count() {
            let dev = ggml_backend_dev_get(i);
            if dev.is_null() || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU {
                continue;
            }
            let mut props: ggml_backend_dev_props = std::mem::zeroed();
            ggml_backend_dev_get_props(dev, &raw mut props);
            ids.push((!props.device_id.is_null()).then(|| {
                std::ffi::CStr::from_ptr(props.device_id)
                    .to_string_lossy()
                    .into_owned()
            }));
            devices.push(dev);
        }
    }
    first_of_each_device(&ids)
        .into_iter()
        .map(|i| devices[i])
        .collect()
}

/// The indices of `ids` llama.cpp keeps: each device unless one before it
/// has the same id. A device without an id is always kept, as llama.cpp
/// keeps it.
pub(crate) fn first_of_each_device(ids: &[Option<String>]) -> Vec<usize> {
    (0..ids.len())
        .filter(|&i| {
            ids[i]
                .as_ref()
                .is_none_or(|id| !ids[..i].iter().any(|other| other.as_ref() == Some(id)))
        })
        .collect()
}

/// Free VRAM in bytes, as reported by the active GPU backend.
///
/// `(free, total)` in bytes, or `None` when there is no GPU or nothing can be
/// read from it. The total is for the callers that report it; the loader's
/// context probe requires `--vram-reserve` ([`DEFAULT_VRAM_RESERVE_MIB`]) to
/// remain free after allocation, a floor the sizer has to respect or it
/// produces splits the loader then refuses.
///
/// Asked through ggml's device registry rather than `cudaMemGetInfo`, so it
/// answers on **every** GPU backend we ship — Vulkan, Metal and ROCm as well
/// as CUDA — instead of only the CUDA builds. The old probe was
/// `#[cfg(feature = "cuda")]` and returned `None` everywhere else, which was
/// invisible while VRAM only fed `--fit` (falling back to the user's own
/// `--gpu-layers` is a reasonable non-answer) and became wrong the moment the
/// model catalog started colouring downloads by it: a Vulkan build on a 16 GB
/// card was told "no GPU detected" and judged every model against system RAM
/// alone. Reported from the field.
///
/// Multiple GPUs are summed, because that is what a layer split can use;
/// each card once, as llama.cpp uses it (see [`gpu_devices`]).
///
/// Returns `None` before `llama_backend_init` has run: the device registry is
/// empty until then, and reporting zero VRAM would read as "a GPU with no
/// memory" rather than "not asked yet".
pub fn vram_bytes() -> Option<(u64, u64)> {
    let mut free_total: u64 = 0;
    let mut total_total: u64 = 0;
    for (free, total) in device_vram() {
        free_total += free;
        total_total += total;
    }

    // A device that reports nothing is a device we cannot size against, and
    // saying so beats sizing a model to zero bytes of VRAM.
    if total_total == 0 || free_total == 0 {
        return None;
    }
    Some((free_total, total_total))
}

/// Free and total VRAM of each GPU, in bytes, in the order llama.cpp numbers
/// the devices a model is split across: what [`vram_bytes`] sums. Empty
/// before `llama_backend_init`, or without a GPU.
pub fn device_vram() -> Vec<(u64, u64)> {
    gpu_devices()
        .into_iter()
        .map(|dev| {
            let mut free: usize = 0;
            let mut total: usize = 0;
            // SAFETY: a device of the registry (`gpu_devices`); the call
            // writes two out-params we own and mutates no backend state.
            unsafe {
                llama_cpp_sys_2::ggml_backend_dev_memory(dev, &raw mut free, &raw mut total);
            }
            (free as u64, total as u64)
        })
        .collect()
}

/// Where the layers of a MoE model go across several GPUs when the experts of
/// its first layers stay in RAM: see [`balance_moe_split`].
#[derive(Debug, Clone, PartialEq)]
pub struct BalancedSplit {
    /// The proportions for llama.cpp's `tensor_split`, one a GPU, that give
    /// each GPU exactly `layers` of them, the output layer to the last.
    pub tensor_split: Vec<f32>,
    /// The transformer blocks each GPU gets, in order.
    pub layers: Vec<u32>,
    /// The layers whose experts stay in RAM: the plan's, or more when the
    /// GPUs could not hold it even split this way.
    pub n_cpu_moe: u32,
}

/// Split a MoE model whose first `n_cpu_moe` layers keep their experts in RAM
/// across GPUs whose free VRAM is `free_per_device`, so that each GPU holds
/// what it can, and not as llama.cpp does by default: by count of layers,
/// in proportion to free VRAM.
///
/// `--fit` counts the experts it keeps on the GPU against the VRAM of all the
/// GPUs together. The layers whose experts stay in RAM are the first, which
/// llama.cpp's split by count gives the first GPU; the last GPU gets layers
/// whole. GLM-5.3-Flash on two GCDs of LUMI-G: the experts of 18 layers in
/// RAM, and 92,164.77 MiB asked of the second GCD, 64 GB (11-11-2026);
/// DeepSeek-V4-Flash the same. Here every block is weighed (its tensors
/// other than experts, its share of the KV cache, its experts when on the
/// GPU), and the blocks are cut into runs whose bytes follow each GPU's free
/// VRAM less `per_device_reserve`, the output layer and the other tensors
/// outside the blocks counted on the last GPU. If no such cut holds, the
/// experts of one more layer go to RAM, and so on.
///
/// `None` with fewer than two GPUs, a model without experts, or where no
/// number of layers in RAM makes a split hold; the load then goes as before.
#[allow(clippy::too_many_arguments)]
pub fn balance_moe_split(
    layout: &MoeLayout,
    info: &GgufInfo,
    ctx_size: u32,
    kv_bytes_per_elem_k: f64,
    kv_bytes_per_elem_v: f64,
    n_cpu_moe: u32,
    free_per_device: &[u64],
    per_device_reserve: u64,
) -> Option<BalancedSplit> {
    let n = layout.expert_bytes_per_layer.len();
    if free_per_device.len() < 2 || n == 0 || !layout.is_moe() {
        return None;
    }
    let budgets: Vec<f64> = free_per_device
        .iter()
        .map(|&free| free.saturating_sub(per_device_reserve) as f64)
        .collect();
    let total_kv = kv_cache_bytes(
        Some(info),
        ctx_size,
        kv_bytes_per_elem_k,
        kv_bytes_per_elem_v,
    );
    let kv_per_layer = total_kv as f64 / n as f64;
    let (non_expert, outside): (Vec<f64>, f64) = if layout.non_expert_bytes_per_layer.len() == n {
        let in_blocks: u64 = layout.non_expert_bytes_per_layer.iter().sum();
        (
            layout
                .non_expert_bytes_per_layer
                .iter()
                .map(|&b| b as f64)
                .collect(),
            layout.non_expert_bytes.saturating_sub(in_blocks) as f64,
        )
    } else {
        (vec![layout.non_expert_bytes as f64 / n as f64; n], 0.0)
    };
    for in_ram in n_cpu_moe..=n as u32 {
        let weights: Vec<f64> = (0..n)
            .map(|l| {
                let experts = if (l as u32) < in_ram {
                    0.0
                } else {
                    layout.expert_bytes_per_layer[l] as f64
                };
                non_expert[l] + kv_per_layer + experts
            })
            .collect();
        if let Some(layers) = split_layers(&weights, outside, &budgets) {
            return Some(BalancedSplit {
                tensor_split: tensor_split_for(&layers),
                layers,
                n_cpu_moe: in_ram,
            });
        }
    }
    None
}

/// Cut `weights`, one a block, into a run of blocks per GPU that each GPU's
/// `budgets` holds, the last also holding `outside`: first in proportion to
/// the budgets, which leaves every GPU the same share of room to spare, else
/// filling each GPU in turn. `None` when neither holds, or a GPU would get no
/// block.
fn split_layers(weights: &[f64], outside: f64, budgets: &[f64]) -> Option<Vec<u32>> {
    let (n, d) = (weights.len(), budgets.len());
    let total: f64 = weights.iter().sum::<f64>() + outside;
    let room: f64 = budgets.iter().sum();
    if d == 0 || total > room {
        return None;
    }
    let holds = |counts: &[u32]| {
        let mut start = 0usize;
        counts.iter().enumerate().all(|(dev, &count)| {
            let end = start + count as usize;
            let mut bytes: f64 = weights[start..end].iter().sum();
            if dev == d - 1 {
                bytes += outside;
            }
            start = end;
            count > 0 && bytes <= budgets[dev]
        })
    };
    // In proportion: each cut where the running sum of bytes is nearest the
    // GPUs' running share of the total.
    let mut counts = vec![0u32; d];
    let (mut i, mut sum, mut target) = (0usize, 0.0f64, 0.0f64);
    for (dev, budget) in budgets.iter().enumerate().take(d - 1) {
        target += total * budget / room;
        let start = i;
        while i < n && sum + weights[i] / 2.0 <= target {
            sum += weights[i];
            i += 1;
        }
        counts[dev] = (i - start) as u32;
    }
    counts[d - 1] = (n - i) as u32;
    if holds(&counts) {
        return Some(counts);
    }
    // Each GPU filled in turn, the last taking the rest.
    let mut counts = vec![0u32; d];
    let mut i = 0usize;
    for (dev, budget) in budgets.iter().enumerate().take(d - 1) {
        let (start, mut used) = (i, 0.0f64);
        while i < n && used + weights[i] <= *budget {
            used += weights[i];
            i += 1;
        }
        counts[dev] = (i - start) as u32;
    }
    counts[d - 1] = (n - i) as u32;
    holds(&counts).then_some(counts)
}

/// llama.cpp's `tensor_split` for these counts of blocks. It sends block `il`
/// of `n` to the first GPU whose running share of the proportions is above
/// `il / (n + 1)` (the output layer is the `n + 1`-th), so the proportions
/// are the counts, the output layer added to the last GPU, with the first
/// less half a block and the last more: every cut falls half a block from
/// any block, never on one, where rounding could move it.
fn tensor_split_for(counts: &[u32]) -> Vec<f32> {
    let mut split: Vec<f32> = counts.iter().map(|&c| c as f32).collect();
    if let (Some(first), true) = (split.first_mut(), counts.len() > 1) {
        *first -= 0.5;
    }
    if let Some(last) = split.last_mut() {
        *last += if counts.len() > 1 { 1.5 } else { 1.0 };
    }
    split
}

/// [`balance_moe_split`] for the model at `model_path` on the GPUs as they are
/// now, the first less `reserve_bytes` (what companions and resident models
/// will need), each less `vram_reserve`, the compute buffer and what a
/// micro-batch of `n_ubatch` adds to it.
#[allow(clippy::too_many_arguments)]
pub fn run_balance_moe_split(
    model_path: &Path,
    ctx_size: u32,
    kv_bytes_per_elem_k: f64,
    kv_bytes_per_elem_v: f64,
    n_cpu_moe: u32,
    n_ubatch: u32,
    reserve_bytes: u64,
    vram_reserve: u64,
    kv: KvLayout,
) -> Option<BalancedSplit> {
    let info = read_gguf_info(model_path)?.with_kv_layout(kv);
    let layout = read_model_moe_layout(model_path, info.n_layers)?;
    let mut free: Vec<u64> = device_vram().into_iter().map(|(free, _)| free).collect();
    if let Some(first) = free.first_mut() {
        *first = first.saturating_sub(reserve_bytes);
    }
    balance_moe_split(
        &layout,
        &info,
        ctx_size,
        kv_bytes_per_elem_k,
        kv_bytes_per_elem_v,
        n_cpu_moe,
        &free,
        per_device_reserve(vram_reserve, n_ubatch),
    )
}

/// What each GPU of a split keeps free beside its share of the model: the
/// reserve, and its own compute buffer, which every GPU of a split has.
///
/// The buffer's growth with the micro-batch is counted twice over here.
/// `ubatch_reserve_bytes` was measured on Qwen3.8-Flash-Next, and a wider
/// model asks more: GLM-5.3-Flash split over two GCDs of LUMI-G asked
/// 2,795.76 MiB of the first at 2,048 tokens and 32 slots, where the flat
/// reserve and the growth came to 1,472 MiB and the load failed
/// (17-11-2026); twice the growth, 2,624 MiB, and the reserve beside it,
/// hold it. A single GPU has the cache's own way back
/// (`inference::grow_moe_cache`); a split, made at load, has none.
pub fn per_device_reserve(vram_reserve: u64, n_ubatch: u32) -> u64 {
    vram_reserve + COMPUTE_BUFFER_RESERVE_BYTES as u64 + 2 * ubatch_reserve_bytes(n_ubatch)
}

/// The line the load log gets for a balanced split.
pub fn describe_balanced_split(split: &BalancedSplit, planned_n_cpu_moe: u32) -> String {
    let layers = split
        .layers
        .iter()
        .map(|c| c.to_string())
        .collect::<Vec<_>>()
        .join("/");
    let more = if split.n_cpu_moe > planned_n_cpu_moe {
        format!(
            "; the experts of {} layers in RAM instead of {planned_n_cpu_moe}, for each GPU to hold its share",
            split.n_cpu_moe
        )
    } else {
        String::new()
    };
    format!(
        "--fit: {} GPUs, layers {layers}: more to the GPU(s) holding the layers whose experts \
         are in RAM, so that each holds its share (--tensor-split {}){more}",
        split.layers.len(),
        split
            .tensor_split
            .iter()
            .map(|x| format!("{x}"))
            .collect::<Vec<_>>()
            .join(","),
    )
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

/// How much VRAM must still be free after the model and its context are
/// resident, in MiB (`--vram-reserve`). One number for the sizer
/// ([`compute_fit`], [`moe_budget`]), the loader's context probe
/// (`inference::probe_and_shrink_context`) and the companion-model check
/// (`api::usable_vram`): sizing to leave one amount while the loader demands
/// another produced splits the loader then refused — the model's weights
/// went in, and the context allocation failed all the way down to 512
/// tokens ("allocation succeeded but left only 10% of GPU memory free").
/// Found on a swap into a 27B on a 16 GiB card. That is why every reader
/// takes the value from here and none keeps a constant of its own.
///
/// It was a share of the card (12% of the total, with a further 3% of the
/// free memory) until 0.7.80, which is the wrong shape: what has to stay
/// free is the scratch the card needs while generating and what other
/// processes take, and none of it grows with the card. 1.9 GB on a 16 GB card
/// cost a dense 8B two of its 36 layers (132 tokens/s on one request became
/// 52); 7.7 GB on a 64 GB one and 11.5 GB on a 96 GB one bought nothing.
///
/// 1 GiB covers 25 times the growth measured while generating (about 40 MiB
/// on Qwen3-8B Q4_K_M with 16 requests, 17-10-2026) and leaves room for a
/// desktop on the same card, the case the old 12% was found for
/// (`inference::probe_and_shrink_context`). The compute buffer is not in
/// it: that is allocated at load and charged by
/// [`COMPUTE_BUFFER_RESERVE_BYTES`] and [`ubatch_reserve_bytes`].
pub const DEFAULT_VRAM_RESERVE_MIB: u32 = 1024;

/// The least `--vram-reserve` accepted, in MiB. The CUDA context and the
/// scratch a decode step asks for after the context is allocated: the
/// compute buffer measured on a 27B at the default micro-batch was 164 MiB
/// (see [`COMPUTE_BUFFER_RESERVE_BYTES`]), and a CUDA context takes about
/// 100 MiB more, so below 256 even a dense model on a card nothing else
/// uses has nothing left for the allocator to work in.
pub const MIN_VRAM_RESERVE_MIB: u32 = 256;

/// `--vram-reserve`: MiB that must stay free, [`MIN_VRAM_RESERVE_MIB`] or
/// more.
pub fn parse_vram_reserve(value: &str) -> Result<u32, String> {
    let mib: u32 = value
        .trim()
        .parse()
        .map_err(|_| format!("`{value}` is not a number of MiB"))?;
    if mib < MIN_VRAM_RESERVE_MIB {
        return Err(format!(
            "{mib} MiB is below the {MIN_VRAM_RESERVE_MIB} MiB minimum: the CUDA context and \
             the scratch a decode step asks for after loading need that much"
        ));
    }
    Ok(mib)
}

/// What `--ctx-size` carries for `auto` (and for `0`, which llama.cpp also
/// reads as "whatever the model and the card allow"): resolved at launch by
/// [`auto_context`], never past it.
pub const CTX_AUTO: u32 = 0;

/// The smallest context `auto` settles for, which is also the default of
/// `--ctx-size`: below it a chat does not hold its own prompt, so the layers
/// go to the CPU before the window shrinks further.
pub const AUTO_CTX_FLOOR: u32 = 4096;

/// The largest context `auto` asks for, whatever the model was trained for
/// and the card could hold: past it the cache takes the card for a window
/// almost nobody fills, and `--ctx-size N` says so when somebody does.
pub const AUTO_CTX_CEILING: u32 = 131_072;

/// `--ctx-size`: a number of tokens, or `auto`.
pub fn parse_ctx_size(value: &str) -> Result<u32, String> {
    let value = value.trim();
    if value.eq_ignore_ascii_case("auto") {
        return Ok(CTX_AUTO);
    }
    value
        .parse()
        .map_err(|_| format!("`{value}` is not a number of tokens or `auto`"))
}

/// The context `--ctx-size auto` comes to: the largest window, up to what the
/// model was trained for and [`AUTO_CTX_CEILING`], that still keeps every
/// layer on the GPU, in steps of `align` tokens (the slots divide it, so the
/// caller passes a multiple of them). The window gives way before a layer
/// does, as llama.cpp's `--fit` does for a context nobody set.
///
/// `extra_reserve` is what is already spoken for besides the weights and the
/// cache (companions, a projector, a larger micro-batch). When even
/// [`AUTO_CTX_FLOOR`] leaves layers in RAM, that is what comes back, with
/// `whole` false, and the usual split takes over. `None` when the card or the
/// header cannot be read.
#[allow(clippy::too_many_arguments)]
pub fn auto_context(
    free_vram: Option<(u64, u64)>,
    info: Option<&GgufInfo>,
    file_size: u64,
    kv_bytes_per_elem_k: f64,
    kv_bytes_per_elem_v: f64,
    vram_reserve: u64,
    extra_reserve: u64,
    align: u32,
) -> Option<AutoContext> {
    let info = info?;
    let align = align.max(1);
    let floor = AUTO_CTX_FLOOR.div_ceil(align) * align;
    let trained = info
        .context_length
        .filter(|&c| c > 0)
        .unwrap_or(AUTO_CTX_CEILING);
    let ceiling = (trained.min(AUTO_CTX_CEILING) / align * align).max(floor);
    let whole_at = |ctx: u32| {
        compute_fit(
            free_vram,
            Some(info),
            file_size,
            ctx,
            kv_bytes_per_elem_k,
            kv_bytes_per_elem_v,
            vram_reserve.saturating_add(extra_reserve),
        )
    };
    if matches!(whole_at(floor), FitDecision::Unknown { .. }) {
        return None;
    }
    if whole_at(floor) != FitDecision::FitsFully {
        return Some(AutoContext {
            ctx: floor,
            whole: false,
        });
    }
    // The cost grows with the window, so the answer is the last step that
    // still fits: bisect over the steps between the floor and the ceiling.
    let (mut lo, mut hi) = (0u32, (ceiling - floor) / align);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if whole_at(floor + mid * align) == FitDecision::FitsFully {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    Some(AutoContext {
        ctx: floor + lo * align,
        whole: true,
    })
}

/// What [`auto_context`] settled on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AutoContext {
    /// The window, in tokens, all slots together.
    pub ctx: u32,
    /// Whether every layer fits on the GPU at that window.
    pub whole: bool,
}

/// A `--vram-reserve` in bytes.
pub const fn vram_reserve_bytes(mib: u32) -> u64 {
    (mib as u64) << 20
}

/// [`DEFAULT_VRAM_RESERVE_MIB`] in bytes.
pub const DEFAULT_VRAM_RESERVE_BYTES: u64 = vram_reserve_bytes(DEFAULT_VRAM_RESERVE_MIB);

/// Flat reserve for the CUDA context + the prefill/decode compute buffer,
/// which does not scale per offloaded layer. H2-H (docs/backlog-fix-e-hardening.md)
/// tracked this as "almost certainly oversized" since the `n_outputs_max` cap
/// landed, but left at 640 pending an on-hardware re-measurement — this is
/// that re-measurement. `LlamaContext::memory_breakdown_print` on a 27B
/// (RTX 5070 Ti, 16384 ctx, `n_ubatch` at its real default of 512, the
/// `n_outputs_max` cap in effect) reported the actual CUDA0 compute buffer at
/// **164 MiB**. 320 keeps roughly 2× that measurement as margin for
/// architectures not yet re-measured (wider intermediate tensors, more attention
/// heads) rather than pinning the reserve to one data point from one model.
///
/// Getting this wrong in the low direction OOMs at load, which is why it isn't
/// pinned to the bare 164 — but this constant only feeds `--fit`'s *automatic*
/// sizing, the guardrail for someone who launched with no `--gpu-layers` at
/// all. Anyone who wants to steer past what it picks still can, with
/// `--gpu-layers` (a ceiling on top of fit) or `--no-fit` (fit out of the
/// loop entirely) — so a split that turns out mildly optimistic on some other
/// architecture costs that person a manual `--gpu-layers` step down, not a
/// silent failure with no recourse.
///
/// Re-measure via the breakdown table before lowering further; 320 is this
/// session's floor, not a ceiling nothing will ever beat.
const COMPUTE_BUFFER_RESERVE_BYTES: f64 = 320.0 * 1024.0 * 1024.0;

/// What a micro-batch larger than the default adds to the compute buffer,
/// on top of `COMPUTE_BUFFER_RESERVE_BYTES`, which every fit already
/// charges and which was measured at the default 512 tokens. The buffer
/// holds the activations of one micro-batch, so it grows with it, by
/// `UBATCH_RESERVE_BYTES_PER_TOKEN` for every token past the default. `0`
/// at or below the default, so a run without `--n-ubatch` is sized exactly
/// as before.
///
/// The VRAM it takes comes out of what the fit hands the model: for an
/// MoE model, fewer layers' experts on the GPU. That is the trade
/// `--n-ubatch` makes — prompts read in fewer, larger passes, answers
/// written with a little more of the model in RAM.
pub(crate) fn ubatch_reserve_bytes(n_ubatch: u32) -> u64 {
    let extra = n_ubatch.saturating_sub(crate::inference::DEFAULT_N_UBATCH);
    (f64::from(extra) * UBATCH_RESERVE_BYTES_PER_TOKEN) as u64
}

/// How much the compute buffer grows for each token a micro-batch adds.
/// Measured with `LlamaContext::memory_breakdown_print` on Qwen3.8-Flash-Next
/// IQ2_XS (RTX 5070 Ti, 40,960 context, the experts of 37 and 40 of its 48
/// layers in RAM): the CUDA0 compute buffer was 648 MiB at a 512-token
/// micro-batch and 3,165 MiB at 4,096, 0.70 MiB per token. 0.75 covers
/// that with 7% to spare. A dense 27B's activations cost about half as
/// much per token, so this reserves more than such a model needs — and a
/// dense model gains little from a larger micro-batch anyway.
///
/// The same measurement shows the flat reserve short for this model at
/// 512: 648 MiB against `COMPUTE_BUFFER_RESERVE_BYTES`' 320, the rest
/// being the copy of a layer's RAM-resident experts the GPU computes them
/// from. The fit's other margins absorbed it (1.8 GiB was still free);
/// re-measure more MoE models with experts in RAM before raising the flat
/// reserve for every model.
const UBATCH_RESERVE_BYTES_PER_TOKEN: f64 = 0.75 * 1024.0 * 1024.0;

/// The same kind of flat reserve as `COMPUTE_BUFFER_RESERVE_BYTES`, but for
/// an embedding model. A `--embedding-model` companion's context is built at
/// launch, at the size of its longest input, and kept for every request, so
/// when free VRAM is read it is in use already, beside the weights: this is
/// the margin for what a decode allocates beside them. An embedder loaded on
/// demand builds its context when its first input comes, and this is all the
/// coexistence check counts for that context, which a long input outgrows —
/// a context for 2,048 tokens of Qwen3-Embedding holds about 1.4 GB.
///
/// `pub(crate)`: shared by two different reservations that must agree on
/// what "the embedder's own overhead" costs — `api::fits_in_free_vram`'s
/// runtime coexistence check for an ad-hoc `/api/embed`-loaded model, and
/// `run_fit`/`run_fit_headless`/`run_moe_fit_plan`'s `reserve_bytes` parameter,
/// which protects a launch-time `--embedding-model` companion's footprint
/// before the generation model's own sizing ever sees the VRAM it needs.
pub(crate) const EMBEDDING_COMPUTE_RESERVE_BYTES: u64 = 256 * 1024 * 1024;

/// The flat compute-buffer part of a decision request's footprint
/// (`inference::decision`), on top of its KV cache (see
/// [`decision_reserve_bytes`]). Twice the embedder's: the micro-batch is the
/// same 512, but the context is sized to a state plus up to 64 questions,
/// and the output holds one vocabulary-sized logits row per question
/// (~39 MiB for 64 questions on a 150k vocabulary). Not yet measured on
/// hardware the way `COMPUTE_BUFFER_RESERVE_BYTES` was; re-measure with the
/// context's memory breakdown before lowering it.
pub(crate) const DECISION_COMPUTE_RESERVE_BYTES: u64 = 512 * 1024 * 1024;

/// Bytes of KV cache `ctx` tokens take in the model `info` describes, at
/// the given per-element sizes — every layer that pays for KV (see
/// `GgufInfo::kv_paying_layers`), with the same coarse fallback as
/// [`compute_fit`] when the header has no attention dims. `0` when the
/// layer count is missing or not believable.
pub(crate) fn kv_cache_bytes(
    info: Option<&GgufInfo>,
    ctx: u32,
    kv_bytes_per_elem_k: f64,
    kv_bytes_per_elem_v: f64,
) -> u64 {
    let Some(info) = info.filter(|i| i.n_layers > 0 && i.n_layers <= MAX_LAYERS) else {
        return 0;
    };
    if let Some(bytes) = info.windowed_kv_bytes(ctx, kv_bytes_per_elem_k, kv_bytes_per_elem_v) {
        return bytes as u64;
    }
    let per_token_per_layer = match info.kv_elems_per_token_per_layer() {
        Some((k_elems, v_elems)) => k_elems * kv_bytes_per_elem_k + v_elems * kv_bytes_per_elem_v,
        None => FALLBACK_KV_BYTES_PER_TOKEN_PER_LAYER,
    };
    let paying = info.kv_paying_layers(u64::from(info.n_layers)) as f64;
    (per_token_per_layer * paying * f64::from(ctx)) as u64
}

/// What one decision request can take on top of the decision model's
/// weights: an F16 KV cache of `max_ctx` tokens (the most a request may
/// ask for, `--decision-ctx`) plus [`DECISION_COMPUTE_RESERVE_BYTES`]. The
/// decision model keeps its context between requests but releases it
/// before a generation model is sized (`api::AppState`'s
/// `release_decision_context`), so unlike the weights it never shows up in
/// the free-VRAM figure that sizing reads — this is what the decision slot
/// asks to be kept free, when it loads and when a generation model is sized
/// next to it.
pub(crate) fn decision_reserve_bytes(path: &Path, max_ctx: u32) -> u64 {
    let info = read_gguf_info(path);
    kv_cache_bytes(info.as_ref(), max_ctx, 2.0, 2.0).saturating_add(DECISION_COMPUTE_RESERVE_BYTES)
}

/// What `--mtp`'s draft context takes beside the model's own (see
/// `inference::scheduler::start_mtp`): the KV cache of the model's MTP
/// layers at `ctx` — attention layers, each paying for every token — and a
/// compute buffer for a micro-batch of `n_ubatch` tokens through them. `0`
/// for a model without MTP layers, where `--mtp` loads no head.
///
/// The draft context is built once the model has loaded, so the free VRAM a
/// load is sized against still holds this memory: without the reserve, a
/// model sized to fill the card left the head no room, and the load that
/// created it was the one that failed.
///
/// Measured with the draft context's memory breakdown on Qwen3.5-0.8B-MTP
/// (4,096 tokens of context, micro-batch 512): 8 MiB of KV, exactly this
/// formula's, and 27 MiB of compute, 54 bytes per token of micro-batch and
/// per unit of `n_embd`. [`MTP_COMPUTE_BYTES_PER_TOKEN_EMBD`] covers that
/// with a fifth to spare. Not yet measured on an MoE model's head, whose
/// experts the compute buffer may have to hold a copy of.
pub(crate) fn mtp_reserve_bytes(
    info: Option<&GgufInfo>,
    ctx: u32,
    kv_bytes_per_elem_k: f64,
    kv_bytes_per_elem_v: f64,
    n_ubatch: u32,
) -> u64 {
    let Some(info) = info else {
        return 0;
    };
    let Some(nextn) = info.nextn_layers.filter(|&n| n > 0 && n < info.n_layers) else {
        return 0;
    };
    mtp_context_bytes(info, nextn, ctx, kv_bytes_per_elem_k, kv_bytes_per_elem_v, n_ubatch)
}

/// What `--mtp-model` takes from the GPU: the head's file, loaded whole, and
/// the draft context it runs on (one layer, sized like the model's own
/// attention layers, which `info` describes). The model's own GGUF has no
/// MTP layers then, so [`mtp_reserve_bytes`] counts nothing for it.
pub(crate) fn mtp_file_reserve_bytes(
    file_bytes: u64,
    info: Option<&GgufInfo>,
    ctx: u32,
    kv_bytes_per_elem_k: f64,
    kv_bytes_per_elem_v: f64,
    n_ubatch: u32,
) -> u64 {
    let Some(info) = info else {
        return file_bytes;
    };
    file_bytes.saturating_add(mtp_context_bytes(
        info,
        1,
        ctx,
        kv_bytes_per_elem_k,
        kv_bytes_per_elem_v,
        n_ubatch,
    ))
}

/// The KV cache of `nextn` layers over `ctx` tokens plus the compute buffer
/// of a micro-batch, for the MTP head's context.
fn mtp_context_bytes(
    info: &GgufInfo,
    nextn: u32,
    ctx: u32,
    kv_bytes_per_elem_k: f64,
    kv_bytes_per_elem_v: f64,
    n_ubatch: u32,
) -> u64 {
    let per_token_per_layer = match info.kv_elems_per_token_per_layer() {
        Some((k_elems, v_elems)) => k_elems * kv_bytes_per_elem_k + v_elems * kv_bytes_per_elem_v,
        None => FALLBACK_KV_BYTES_PER_TOKEN_PER_LAYER,
    };
    let kv = per_token_per_layer * f64::from(nextn) * f64::from(ctx);
    let n_embd = f64::from(info.n_embd.filter(|&n| n > 0).unwrap_or(4096));
    let compute = (f64::from(n_ubatch) * n_embd * MTP_COMPUTE_BYTES_PER_TOKEN_EMBD)
        .max(MTP_COMPUTE_FLOOR_BYTES);
    (kv + compute) as u64
}

/// Compute buffer of `--mtp`'s draft context, per token of micro-batch and
/// per unit of `n_embd` (see [`mtp_reserve_bytes`]).
const MTP_COMPUTE_BYTES_PER_TOKEN_EMBD: f64 = 64.0;

/// The least compute buffer [`mtp_reserve_bytes`] counts, for a small model
/// on a small micro-batch.
const MTP_COMPUTE_FLOOR_BYTES: f64 = 32.0 * 1024.0 * 1024.0;

/// Coarse KV reserve used only when the GGUF header doesn't expose the
/// attention dims: ~128 B per token per layer (a rough F16 ballpark for
/// 7-8B-class models). The exact path below supersedes this whenever the
/// dims are present.
const FALLBACK_KV_BYTES_PER_TOKEN_PER_LAYER: f64 = 128.0;

/// Sanity ceiling for a layer count read from a GGUF header. Real
/// architectures ship well under 200 layers; anything past this is a corrupt
/// file, and without a ceiling it prices loop iterations and scratch vectors
/// off one untrusted integer (see `compute_fit` and `parse_gguf_moe_layout`).
const MAX_LAYERS: u32 = 4096;

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
    vram: Option<(u64, u64)>,
    info: Option<&GgufInfo>,
    file_size: u64,
    ctx_size: u32,
    kv_bytes_per_elem_k: f64,
    kv_bytes_per_elem_v: f64,
    vram_reserve: u64,
) -> FitDecision {
    let (free_vram, _) = match vram {
        Some(v) => v,
        None => {
            return FitDecision::Unknown {
                reason: "could not read free VRAM (needs a CUDA build with a working CUDA device)"
                    .to_string(),
            };
        }
    };
    let info = match info {
        Some(i) if i.n_layers > 0 && i.n_layers <= MAX_LAYERS => i,
        Some(i) if i.n_layers > MAX_LAYERS => {
            return FitDecision::Unknown {
                reason: format!(
                    "absurd layer count {} in the GGUF header (past the {MAX_LAYERS} sanity ceiling)",
                    i.n_layers
                ),
            };
        }
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
    // mean (measured live: ~0.5 GiB at 262k context). A model with
    // sliding-window layers is charged its whole KV over its layers (see
    // `GgufInfo::kv_bytes_per_paying_layer`).
    let kv_per_paying_layer =
        info.kv_bytes_per_paying_layer(ctx_size, kv_bytes_per_elem_k, kv_bytes_per_elem_v);

    // Budget = free VRAM, minus the reserve that must stay free (the same
    // amount the loader's context probe will demand, so sizing past it only
    // produces splits it refuses), minus the flat compute-buffer reserve.
    let usable =
        (free_vram as f64 - vram_reserve as f64 - COMPUTE_BUFFER_RESERVE_BYTES).max(0.0);

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

/// The VRAM a MoE model may use and what the part that cannot leave the GPU
/// costs of it, in bytes: `(usable, fixed_cost)`. `usable` follows
/// `compute_fit`'s budget rule, never past the headroom the loader's context
/// probe will require; `fixed_cost` is the non-expert weights plus the KV
/// cache. What is left between the two is what the experts can have.
fn moe_budget(
    free_vram: u64,
    info: &GgufInfo,
    layout: &MoeLayout,
    ctx_size: u32,
    kv_bytes_per_elem_k: f64,
    kv_bytes_per_elem_v: f64,
    vram_reserve: u64,
) -> (f64, f64) {
    let kv_per_paying_layer =
        info.kv_bytes_per_paying_layer(ctx_size, kv_bytes_per_elem_k, kv_bytes_per_elem_v);
    let total_kv = kv_per_paying_layer * info.kv_paying_layers(info.n_layers as u64) as f64;
    let usable =
        (free_vram as f64 - vram_reserve as f64 - COMPUTE_BUFFER_RESERVE_BYTES).max(0.0);
    (usable, layout.non_expert_bytes as f64 + total_kv)
}

/// What `--fit` decided for a MoE model, with the load settings that go with
/// an expert layout kept in RAM.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoeFitPlan {
    pub decision: MoeFitDecision,
    /// The micro-batch the plan was sized for: [`MOE_CACHE_N_UBATCH`] when no
    /// `--n-ubatch` was given and the split still holds beside the larger
    /// compute buffer, `None` to keep the caller's.
    pub n_ubatch: Option<u32>,
    /// Bytes of experts the split keeps in RAM: the load reads the model into
    /// memory when they fit there (see [`plan_read_into_memory`]).
    pub host_bytes: u64,
}

/// The bytes of the experts of the first `n_cpu_moe` layers.
pub fn cpu_expert_bytes(layout: &MoeLayout, n_cpu_moe: u32) -> u64 {
    layout
        .expert_bytes_per_layer
        .iter()
        .take(n_cpu_moe as usize)
        .sum()
}

/// [`compute_moe_fit`], and what a split with experts in RAM wants of the
/// load beside it. Prompts are read in micro-batches, and each one pays for
/// the experts it touches: with a 512-token micro-batch and most experts in
/// RAM, a 31,000-token prompt moves them about 60 times (Qwen3.8-Flash-Next
/// UD-Q4_K_XL on an A100: 379 tokens/s read at 512, 1,876 at 8,192). So,
/// as for an expert cache, a split sized with `auto_n_ubatch` takes
/// [`MOE_CACHE_N_UBATCH`] if it still holds with the larger compute buffer.
#[allow(clippy::too_many_arguments)]
pub fn compute_moe_fit_plan(
    vram: Option<(u64, u64)>,
    info: Option<&GgufInfo>,
    layout: Option<&MoeLayout>,
    ctx_size: u32,
    kv_bytes_per_elem_k: f64,
    kv_bytes_per_elem_v: f64,
    vram_reserve: u64,
    auto_n_ubatch: bool,
) -> MoeFitPlan {
    let fit = |vram| {
        compute_moe_fit(
            vram,
            info,
            layout,
            ctx_size,
            kv_bytes_per_elem_k,
            kv_bytes_per_elem_v,
            vram_reserve,
        )
    };
    let mut decision = fit(vram);
    let mut n_ubatch = None;
    if auto_n_ubatch && matches!(decision, MoeFitDecision::Proceed { n_cpu_moe } if n_cpu_moe > 0)
    {
        let larger = vram.map(|(free, total)| {
            (
                free.saturating_sub(ubatch_reserve_bytes(MOE_CACHE_N_UBATCH)),
                total,
            )
        });
        let again = fit(larger);
        if matches!(again, MoeFitDecision::Proceed { n_cpu_moe } if n_cpu_moe > 0) {
            decision = again;
            n_ubatch = Some(MOE_CACHE_N_UBATCH);
        }
    }
    let host_bytes = match (&decision, layout) {
        (MoeFitDecision::Proceed { n_cpu_moe }, Some(layout)) => {
            cpu_expert_bytes(layout, *n_cpu_moe)
        }
        _ => 0,
    };
    MoeFitPlan {
        decision,
        n_ubatch,
        host_bytes,
    }
}

/// Compute the MoE-aware fit decision from probed VRAM, GGUF info, and the
/// tensor layout parsed by [`parse_gguf_moe_layout`]/[`read_gguf_moe_layout`].
///
/// Pure decision logic — no I/O — mirroring [`compute_fit`]'s split from
/// [`run_fit`]. See [`run_moe_fit_plan`] for the I/O-performing wrapper.
pub fn compute_moe_fit(
    vram: Option<(u64, u64)>,
    info: Option<&GgufInfo>,
    layout: Option<&MoeLayout>,
    ctx_size: u32,
    kv_bytes_per_elem_k: f64,
    kv_bytes_per_elem_v: f64,
    vram_reserve: u64,
) -> MoeFitDecision {
    let (Some((free_vram, total_vram)), Some(info), Some(layout)) = (vram, info, layout) else {
        return MoeFitDecision::NotMoe;
    };
    if !layout.is_moe() {
        return MoeFitDecision::NotMoe;
    }

    let (usable, fixed_cost) = moe_budget(
        free_vram,
        info,
        layout,
        ctx_size,
        kv_bytes_per_elem_k,
        kv_bytes_per_elem_v,
        vram_reserve,
    );

    if fixed_cost >= usable {
        // Every expert already assumed off-GPU here; charge only the
        // non-expert bytes against the ordinary dense sizer to find how many
        // whole layers of *those* still fit.
        let decision = compute_fit(
            Some((free_vram, total_vram)),
            Some(info),
            layout.non_expert_bytes,
            ctx_size,
            kv_bytes_per_elem_k,
            kv_bytes_per_elem_v,
            vram_reserve,
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

/// `--moe-cache`: keep a MoE model's experts in RAM and give the VRAM they
/// would have had to a cache of the ones the model uses most. The cache is
/// llama.cpp's (PR #29887, carried on top of the pinned release until it is
/// merged): experts are copied in as the model asks for them and the least
/// recently used ones leave. It serves batches of up to 32 tokens, so it
/// speeds up writing, not the reading of a long prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeCache {
    /// As large as the VRAM left once the rest of the model is placed.
    Auto,
    /// This many MiB, or less when there is less room.
    Mib(u32),
    /// No cache: the usual split decides, which is what `--moe-cache` did
    /// not being given meant before it came on by itself.
    Off,
    /// No flag: `Auto` where the cache gains, the usual split elsewhere
    /// (see [`MOE_CACHE_DEFAULT_OUT`]). Never parsed: what `--moe-cache` left
    /// out comes to.
    Default,
}

impl MoeCache {
    /// Sized from the VRAM left, as `auto` is, asked for or not.
    pub fn is_auto(self) -> bool {
        matches!(self, MoeCache::Auto | MoeCache::Default)
    }
}

/// Parse `--moe-cache`: `auto`, `off`, or a size in MiB above 0.
pub fn parse_moe_cache(s: &str) -> Result<MoeCache, String> {
    if s.eq_ignore_ascii_case("auto") {
        return Ok(MoeCache::Auto);
    }
    if s.eq_ignore_ascii_case("off") {
        return Ok(MoeCache::Off);
    }
    match s.parse::<u32>() {
        Ok(mib) if mib > 0 => Ok(MoeCache::Mib(mib)),
        _ => Err(format!(
            "expected `auto`, `off` or a size in MiB above 0, got `{s}`"
        )),
    }
}

/// What `--moe-cache` comes to once the launch flags are in: the user's own
/// word when there is one, and `auto` when there is none and nothing else
/// says where the experts go.
///
/// The cache is on by itself for a MoE model whose experts do not fit, since
/// the usual split (the experts of the first layers in RAM, computed on the
/// CPU) writes less than half as fast: on an RTX 5070 Ti, Qwen3.6-35B-A3B
/// UD-Q4_K_M wrote 55 tokens/s with the split and 133 with the cache, and
/// read a prompt 25% faster. It stays off where the user placed the experts
/// (`--cpu-moe`, `--n-cpu-moe`), where there is no sizing to place a cache
/// (`--no-fit`), with `--mtp` (not measured to gain) and wherever the
/// machine cannot run one, all without a word: only an asked-for cache says
/// why it did not run. Where it comes on by itself it is
/// [`MoeCache::Default`], which `plan_moe_cache` plans only where the share
/// of the experts the usual split leaves in RAM says the cache gains
/// ([`MOE_CACHE_DEFAULT_OUT`]), and never for a dense model or one whose
/// experts all fit.
pub fn effective_moe_cache(
    asked: Option<MoeCache>,
    fit: bool,
    cpu_moe: bool,
    n_cpu_moe: u32,
    mtp: u32,
) -> Option<MoeCache> {
    effective_moe_cache_on(asked, fit, cpu_moe, n_cpu_moe, mtp, &gpu_names())
}

/// [`effective_moe_cache`] for the GPUs ggml lists, by name.
///
/// `off` is no cache at all, so the caller neither sizes one nor says why it
/// could not run one. By itself the cache comes on for one GPU of the CUDA
/// backend, CUDA or ROCm, as [`MoeCache::Default`]: where LUMI's MI250X
/// measured it behind the usual split (gpt-oss-120b at 16 requests, 110
/// against 184 tokens/s; DeepSeek-V3.1 at 4 and 16, 3.2-3.7 against
/// 4.1-5.3), the share of the experts left in RAM was outside the band the
/// default keeps to, and where it was inside the cache won on the same GCD
/// (Qwen3-235B 1.6 to 3.2 times).
pub fn effective_moe_cache_on(
    asked: Option<MoeCache>,
    fit: bool,
    cpu_moe: bool,
    n_cpu_moe: u32,
    mtp: u32,
    gpus: &[String],
) -> Option<MoeCache> {
    match asked {
        Some(MoeCache::Off) => None,
        Some(asked) => Some(asked),
        None => (fit
            && !cpu_moe
            && n_cpu_moe == 0
            && mtp == 0
            && moe_cache_support_on(gpus).is_ok())
        .then_some(MoeCache::Default),
    }
}

/// The share of a MoE model's expert bytes that the usual split leaves in RAM
/// (the experts that do not fit once the rest of the model is placed),
/// between which the cache comes on by itself ([`MoeCache::Default`]). With
/// no flag placing the experts the cache gets the VRAM those that fit would
/// have had, so it holds `1 - share` of them. Measured on one GCD of LUMI-G's
/// MI250X (64 GB, a 64-core host; `c10` in round `r-next21`, 21-10-2026),
/// tokens/s with the split and with the cache, at 1 / 4-8 / 16-32 requests:
///
/// * gpt-oss-120b MXFP4, about 5% in RAM: 107/108, -, 184/110 at 16 and
///   168/170 at 32. The split already keeps nearly every expert on the GPU,
///   and the cache adds copies.
/// * Qwen3-235B Q4_K_M, about 60%: 6.9/21.8, 13.2/31.5, 15.9/25.2.
/// * Qwen3-Coder-480B Q4_K_M, about 80%: 3.4/7.4, 7.3/8.1, 8.9/9.1.
/// * DeepSeek-V3.1 Q4_K_M, about 90%: 3.4/3.4, 4.1/3.2 at 4, 5.3/3.7 at 16.
///   A cache holding a tenth of the experts misses most of them, and the
///   host's cores compute them faster than the link copies them.
///
/// On an RTX 5070 Ti, where the default was decided, Qwen3.6-35B-A3B has
/// about half of its experts out (55 to 133 tokens/s with the cache). The
/// bounds come from a host with 64 cores: on a desktop CPU the cache may
/// gain nearer them, which is measured next. `--moe-cache auto` asks for the
/// cache whatever the share.
pub const MOE_CACHE_DEFAULT_OUT: (f64, f64) = (0.25, 0.85);

/// The share of the experts' bytes that do not fit once the rest of the
/// model is placed: what the usual split leaves in RAM, 0 to 1.
pub fn moe_out_share(usable: f64, fixed_cost: f64, expert_bytes: u64) -> f64 {
    if expert_bytes == 0 {
        return 0.0;
    }
    ((fixed_cost + expert_bytes as f64 - usable) / expert_bytes as f64).clamp(0.0, 1.0)
}

/// Below this an automatic cache holds too few experts per layer to be worth
/// the copies, and the usual sizing, which keeps whole layers of experts on
/// the GPU, applies instead.
pub(crate) const MOE_CACHE_MIN_BYTES: u64 = 512 * 1024 * 1024;

/// The step a cache is sized in, so that a few megabytes more or less of free
/// VRAM between two loads do not change the size the log reports.
const MOE_CACHE_STEP_BYTES: u64 = 256 * 1024 * 1024;

/// `--moe-prefetch`'s default: while a prompt is read, the experts kept in
/// RAM are copied into this many slots of VRAM ahead of the layer that reads
/// them, on a second stream of the GPU, instead of in between that layer's
/// computations (llama.cpp patch `0003`, phase 6 of
/// `docs/moe-offload-plan.md`). It applies to micro-batches of 512 tokens or
/// more, on one CUDA or ROCm GPU, to experts in the GPU's pinned host memory: a model
/// read into memory. Measured on Qwen3.8-Flash-Next IQ2_XS on an RTX 5070 Ti:
/// with an `auto` cache 1 GiB smaller for four slots of 256 MiB, at the
/// default micro-batch of 2048, a 33,200-token prompt read 42% faster (968
/// to 1,373 tokens/s), to the same answer, and answers were written as fast
/// (53.3 and 54.3 tokens/s). Two slots gained 10-15% and three 18%, six and
/// eight no more than four (at micro-batch 4096 and a fixed cache).
pub const MOE_PREFETCH_SLOTS: u32 = 4;

/// Parse `--moe-prefetch`: `0` (off), or 2 to 8 slots. With one, every copy
/// would wait for the layer before it to be read.
pub fn parse_moe_prefetch(s: &str) -> Result<u32, String> {
    match s.parse::<u32>() {
        Ok(slots) if slots == 0 || (2..=8).contains(&slots) => Ok(slots),
        _ => Err(format!("expected 0 (off) or 2 to 8 slots, got `{s}`")),
    }
}

/// What `--moe-prefetch` asks of a load with an expert cache, and what
/// decides whether the slots can be used at all: only experts in pinned
/// memory are copied ahead, and those of a load with a cache are pinned when
/// the model is read into memory (see [`plan_read_into_memory`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MoePrefetch {
    /// The slots asked for; `0` without the prefetch, or where it cannot run
    /// (one GPU of the CUDA backend, as for the expert cache: see
    /// [`moe_cache_support`]).
    pub slots: u32,
    /// `--no-mmap`.
    pub no_mmap: bool,
    /// `--mmap`.
    pub keep_mapped: bool,
    /// The machine's RAM, `None` where it cannot be read.
    pub ram_total: Option<u64>,
}

impl MoePrefetch {
    /// The slots a load whose cache copies from `host_bytes` of experts can
    /// use: all of them where those experts are pinned, none otherwise.
    pub fn slots_for(&self, host_bytes: u64) -> u32 {
        let (pinned, _) =
            plan_read_into_memory(self.no_mmap, self.keep_mapped, host_bytes, self.ram_total);
        if pinned {
            self.slots
        } else {
            0
        }
    }
}

/// The VRAM one slot of the prefetch takes: the largest expert tensor, the
/// most a slot is asked to hold, rounded up to a MiB for what llama.cpp adds
/// past the end of a copy (part of a row of padding, and the alignment).
pub fn prefetch_slot_bytes(layout: &MoeLayout) -> u64 {
    layout.largest_expert_tensor_bytes.div_ceil(1 << 20) << 20
}

/// The slots a load asks llama.cpp for: `slots`, `--moe-prefetch`'s, where
/// some experts are kept in RAM (`cpu_moe`, `n_cpu_moe`) and pinned (the
/// model read into memory, `read_into_memory`); none elsewhere, where there
/// is nothing they could copy. llama.cpp makes them at the first long prompt,
/// if the VRAM left has room; a load with an expert cache has kept that room
/// for them already where it could (see [`MoeCachePlan::prefetch_bytes`]).
pub fn prefetch_slots(slots: u32, read_into_memory: bool, cpu_moe: bool, n_cpu_moe: u32) -> u32 {
    if read_into_memory && (cpu_moe || n_cpu_moe > 0) {
        slots
    } else {
        0
    }
}

/// What the load may do with the VRAM that a cache sized by `--moe-cache auto`
/// leaves over: the plan charges the compute buffer of a large micro-batch
/// from a flat figure that is generous for most contexts, and the load can
/// measure what is really free once the context exists
/// (`inference::grow_moe_cache`). Only an `auto` cache carries it; a size
/// given in MiB is used as given.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeCacheGrow {
    /// VRAM that stays free for what is made after the probe: the prefetch's
    /// slots, which llama.cpp allocates at the first long prompt.
    pub keep_bytes: u64,
    /// The most the cache can hold: the experts kept in RAM.
    pub max_bytes: u64,
}

/// Where a load with an expert cache puts its experts, and the cache's size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeCachePlan {
    /// Every layer's experts in RAM (`--cpu-moe`).
    pub cpu_moe: bool,
    /// Otherwise the first `n_cpu_moe` layers', as the user's `--n-cpu-moe`
    /// asked.
    pub n_cpu_moe: u32,
    /// VRAM for the cache, in bytes.
    pub bytes: u64,
    /// The size `--moe-cache` asked for, when there was less room than that.
    pub asked_bytes: Option<u64>,
    /// Bytes of the experts kept in RAM, which the cache copies from.
    pub host_bytes: u64,
    /// The micro-batch the load takes: [`MOE_CACHE_N_UBATCH`] when the
    /// caller left it to the plan and the cache keeps its size beside the
    /// larger compute buffer, `None` to keep the caller's.
    pub n_ubatch: Option<u32>,
    /// VRAM kept out of the cache for the prefetch's slots: `0` where the
    /// experts are not pinned, and where keeping it would have cost the cache
    /// its minimum, the size the user asked for or the larger micro-batch.
    /// The slots are asked for all the same where the experts are pinned, and
    /// llama.cpp makes them at the first long prompt if there is room then.
    pub prefetch_bytes: u64,
    /// Set for `--moe-cache auto`: see [`MoeCacheGrow`].
    pub grow: Option<MoeCacheGrow>,
}

/// The micro-batch a load with an expert cache takes when `--n-ubatch` was
/// not given. A prompt copies the experts kept in RAM to the GPU once per
/// micro-batch, so at 2,048 tokens it copies them a quarter as often as at
/// llama.cpp's 512, and the compute buffer that grows with it comes out of
/// the cache. Measured on Qwen3.8-Flash-Next IQ2_XS with the experts pinned
/// (RTX 5070 Ti, 40,960-token context): a 33,200-token prompt read at 964
/// tokens/s instead of 452, answers written at 54.9 instead of 58.1, the
/// cache 6.75 GiB instead of 8. At 4,096 reading reached 1,240 and writing
/// fell to 47.1.
pub const MOE_CACHE_N_UBATCH: u32 = 2048;

/// Size a load's expert cache. `None` when it gets none: not a MoE model,
/// one whose experts all fit on the GPU anyway, or no room for a cache once
/// the rest of the model is placed. The usual MoE sizing applies then.
///
/// With a cache, every expert stays in RAM, unless the user's `--n-cpu-moe N`
/// keeps those of layer `N` onward on the GPU, which are then paid for first.
/// The rest of the model goes on the GPU whole. Measured on
/// Qwen3.8-Flash-Next IQ2_XS on an RTX 5070 Ti: every expert in RAM with an
/// 8,000 MiB cache wrote 49.4 tokens/s, 4,000 MiB 33.1, and the split that
/// keeps the last layers' experts on the GPU 22.4.
///
/// Beside the cache the room may hold the larger micro-batch's compute
/// buffer (`auto_n_ubatch`) and the prefetch's slots (`prefetch`, where the
/// experts are pinned): see the body for which gives way first.
#[allow(clippy::too_many_arguments)]
pub fn plan_moe_cache(
    request: MoeCache,
    vram: Option<(u64, u64)>,
    info: Option<&GgufInfo>,
    layout: Option<&MoeLayout>,
    ctx_size: u32,
    kv_bytes_per_elem_k: f64,
    kv_bytes_per_elem_v: f64,
    vram_reserve: u64,
    cpu_moe: bool,
    n_cpu_moe: u32,
    auto_n_ubatch: bool,
    prefetch: MoePrefetch,
) -> Option<MoeCachePlan> {
    if request == MoeCache::Off {
        return None;
    }
    let (Some((free_vram, _)), Some(info), Some(layout)) = (vram, info, layout) else {
        return None;
    };
    if !layout.is_moe() {
        return None;
    }
    let (usable, fixed_cost) = moe_budget(
        free_vram,
        info,
        layout,
        ctx_size,
        kv_bytes_per_elem_k,
        kv_bytes_per_elem_v,
        vram_reserve,
    );
    let expert_bytes: u64 = layout.expert_bytes_per_layer.iter().sum();
    // Without a flag of the user's sending experts to RAM, a model that fits
    // whole has none there to cache.
    if !cpu_moe && n_cpu_moe == 0 && fixed_cost + expert_bytes as f64 <= usable {
        return None;
    }
    // Not asked for: on only where the cache gains.
    if request == MoeCache::Default {
        let out = moe_out_share(usable, fixed_cost, expert_bytes);
        let (low, high) = MOE_CACHE_DEFAULT_OUT;
        if !(low..=high).contains(&out) {
            tracing::info!(
                "MoE: {:.0}% of the experts do not fit in VRAM; --fit's split instead of the \
                 expert cache, which comes on by itself between {:.0}% and {:.0}% \
                 (--moe-cache auto asks for it)",
                out * 100.0,
                low * 100.0,
                high * 100.0
            );
            return None;
        }
        tracing::info!(
            "MoE: {:.0}% of the experts do not fit in VRAM; the expert cache is on by \
             default (--moe-cache off turns it off)",
            out * 100.0
        );
    }
    let kept_on_gpu: u64 = if cpu_moe || n_cpu_moe == 0 {
        0
    } else {
        layout
            .expert_bytes_per_layer
            .iter()
            .skip(n_cpu_moe as usize)
            .sum()
    };
    let in_ram = expert_bytes.saturating_sub(kept_on_gpu);
    let room = usable - fixed_cost - kept_on_gpu as f64;
    if room <= 0.0 || in_ram == 0 {
        return None;
    }
    let step_down = |bytes: u64| bytes / MOE_CACHE_STEP_BYTES * MOE_CACHE_STEP_BYTES;
    // The cache a given room holds, as the request asks: (bytes, asked).
    let size = |room: f64| -> Option<(u64, Option<u64>)> {
        if room <= 0.0 {
            return None;
        }
        let room = (room as u64).min(in_ram);
        let (bytes, asked_bytes) = match request {
            MoeCache::Auto | MoeCache::Default => (step_down(room), None),
            MoeCache::Off => return None,
            MoeCache::Mib(mib) => {
                let asked = u64::from(mib) * 1024 * 1024;
                if asked <= room {
                    (asked, None)
                } else {
                    (step_down(room), Some(asked))
                }
            }
        };
        if bytes == 0 || (request.is_auto() && bytes < MOE_CACHE_MIN_BYTES) {
            return None;
        }
        Some((bytes, asked_bytes))
    };
    // What the room holds beside the cache, in the order tried: the first
    // that leaves a cache wins, and only the last may cut a size the user
    // asked for. The larger micro-batch comes in when the caller sized with
    // the default and left the choice here: it reads a prompt twice as fast
    // (963.8 tokens/s against 451.5 on Qwen3.8-Flash-Next IQ2_XS), the slots
    // of the prefetch a quarter faster again, so the slots give way first.
    let larger = auto_n_ubatch.then_some(MOE_CACHE_N_UBATCH);
    let slots_bytes = u64::from(prefetch.slots_for(in_ram)) * prefetch_slot_bytes(layout);
    let mut beside = Vec::with_capacity(3);
    if slots_bytes > 0 {
        beside.push((larger, slots_bytes));
    }
    if larger.is_some() {
        beside.push((larger, 0));
    }
    beside.push((None, 0));
    let last = beside.len() - 1;
    let ((bytes, asked_bytes), n_ubatch, prefetch_bytes) = beside
        .into_iter()
        .enumerate()
        .find_map(|(i, (n_ubatch, slots_bytes))| {
            let buffer = n_ubatch.map_or(0, ubatch_reserve_bytes);
            let sized = size(room - buffer as f64 - slots_bytes as f64)?;
            (i == last || sized.1.is_none()).then_some((sized, n_ubatch, slots_bytes))
        })?;
    Some(MoeCachePlan {
        cpu_moe: cpu_moe || n_cpu_moe == 0,
        n_cpu_moe,
        bytes,
        asked_bytes,
        host_bytes: in_ram,
        n_ubatch,
        prefetch_bytes,
        grow: request.is_auto().then_some(MoeCacheGrow {
            keep_bytes: prefetch_bytes,
            max_bytes: in_ram,
        }),
    })
}

/// Whether pinning `host_bytes` of experts leaves the rest of the machine
/// enough RAM: a quarter of it, and at least 8 GiB. Pinned pages can be
/// neither swapped out nor dropped. The same rule the expert cache's own
/// pinning applies (llama.cpp patch `0002`).
pub fn pin_fits_in_ram(host_bytes: u64, ram_total: u64) -> bool {
    host_bytes <= ram_total.saturating_sub(pinned_ram_reserve(ram_total))
}

fn pinned_ram_reserve(ram_total: u64) -> u64 {
    (ram_total / 4).max(8 << 30)
}

/// Whether a load reads the model into memory instead of mapping its file,
/// and the line that says why when the expert cache decided it. `--no-mmap`
/// always reads it in. With an expert cache (`cache_host_bytes` of experts
/// in RAM), reading it in is what puts those experts in pinned memory, which
/// the GPU copies from at the bus's speed: done whenever the RAM can spare
/// them, unless `--mmap` keeps the file mapped.
pub fn plan_read_into_memory(
    no_mmap: bool,
    keep_mapped: bool,
    cache_host_bytes: u64,
    ram_total: Option<u64>,
) -> (bool, Option<String>) {
    if no_mmap {
        return (true, None);
    }
    if keep_mapped || cache_host_bytes == 0 {
        return (false, None);
    }
    match ram_total {
        Some(ram) if pin_fits_in_ram(cache_host_bytes, ram) => (
            true,
            Some(format!(
                "experts in RAM: reading the model into memory, so that its {} of them \
                 are pinned for the GPU's copies (--mmap keeps the file mapped)",
                gib(cache_host_bytes)
            )),
        ),
        Some(ram) => (
            false,
            Some(format!(
                "experts in RAM: keeping the model file mapped: pinning its {} of experts in RAM \
                 would leave less than {} of the {} of RAM to the rest (--no-mmap reads it \
                 into memory anyway)",
                gib(cache_host_bytes),
                gib(pinned_ram_reserve(ram)),
                gib(ram)
            )),
        ),
        None => (
            false,
            Some(
                "experts in RAM: keeping the model file mapped: the size of the RAM is not known \
                 here (--no-mmap reads it into memory anyway)"
                    .to_string(),
            ),
        ),
    }
}

/// Whether every layer of the model, and every expert, goes to the GPUs:
/// `gpu_layers` of `-1`, or at least one more than the model's
/// `n_layers` (the output layer is the extra one), with no expert kept in
/// RAM. An unknown layer count counts as all only for `-1`.
pub fn every_layer_on_gpu(
    gpu_layers: i32,
    n_layers: Option<u32>,
    cpu_moe: bool,
    n_cpu_moe: u32,
    moe_cache_bytes: u64,
) -> bool {
    let all_layers = gpu_layers < 0
        || n_layers.is_some_and(|n| i64::from(gpu_layers) > i64::from(n));
    all_layers && !cpu_moe && n_cpu_moe == 0 && moe_cache_bytes == 0
}

/// Whether a model that goes to the GPUs whole is read into memory instead
/// of mapped, and the line that says why when it is: when its files are more
/// than half of the memory this process may fill (`memory`, the RAM or a
/// Slurm job's `--mem`), or are on a network file system (`network`, as
/// `readahead::network_file_system` names it), unless `--mmap` keeps them
/// mapped.
///
/// Measured on LUMI-G (c09, 07-10-2026 and 08-10-2026), loading cold from
/// Lustre: Qwen3-Coder-480B Q4_K_M, 270 GiB on 8 GCDs of a 480 GiB job,
/// never loaded mapped (six tries; one watched for 70 minutes, the file read
/// into the page cache and then copied to the GPUs by one thread at 90 MB/s)
/// and loaded read in, in 160 and 208 s; DeepSeek-V3.1 Q4_K_M the same, 226 s
/// read in. One server alone below half gained nothing: Qwen3-235B, 132 GiB,
/// 110 s mapped and 108 s read in; Qwen3.8-27B, 29 GiB, 46 s mapped and 51 s
/// read in. The tokens per second were the same both ways.
///
/// Below half, servers mapping one file at once from a network file system
/// were the slow case (c02, 09 and 10-10-2026, Lustre): two servers of the
/// 235B starting together took 667 to 1,901 s to load, and once failed after
/// the hour the runner waits; four of gpt-oss-120b, 59 GiB, failed the hour
/// and then took 761 s, 57-61 s once the file was in the page cache; the
/// 233 GiB Q8_0, alone but at the edge of half, failed the hour and then took
/// 1,161 s. Read in by the rule above, the 480B and the 671B loaded in 91 to
/// 284 s in the same job. A file on a local disk stays mapped, as before.
///
/// Cold, alone on a node (c09 `load-together-*`, 13-10-2026), mapped against
/// read in, seconds: two servers of the 235B Q4 2,448 against 113; four of
/// gpt-oss-120b 1,789 against 53; eight of Qwen3-14B (9 GiB) 343 against 14.
/// One server of the 235B Q8_0 took 204 mapped and 225 read in: alone, the
/// file maps fine. With this rule and no flag: 121, 63, 16 and 209 s.
///
/// GPFS (IBM Storage Scale) is the exception, measured on JUPITER's GH200
/// nodes (`tools/jupiter/sbatch_load.slurm`, 14-10-2026), where reading in is
/// the slow way: Qwen3.6-27B Q4_K_M (16.8 GB) on one GPU took 53 s read in,
/// cold and again warm, against 10 s mapped (llama-server: 48 s read in, 4 s
/// mapped); four servers of Qwen3.6-35B-A3B (22.1 GB), one per GPU, loading
/// the file at once took 16 s each mapped, cold, and 34 s read in. Its files
/// stay mapped below half the memory. The other network file systems keep
/// the rule: none of them has been measured.
pub fn read_in_whole_on_gpu(
    model_bytes: u64,
    whole_on_gpu: bool,
    memory: Option<u64>,
    network: Option<&str>,
    keep_mapped: bool,
) -> Option<String> {
    if keep_mapped || !whole_on_gpu {
        return None;
    }
    if let Some(memory) = memory.filter(|&m| model_bytes > m / 2) {
        return Some(format!(
            "reading the model ({}) into memory instead of mapping its file: it goes to the \
             GPUs whole and is more than half of the {} of memory this process may use, where \
             a mapped file is read twice and copied to the GPUs a page at a time (--mmap keeps \
             it mapped)",
            gib(model_bytes),
            gib(memory)
        ));
    }
    // GPFS maps faster than it reads in, alone or several at once (above).
    let network = network.filter(|&fs| fs != "GPFS")?;
    Some(format!(
        "reading the model ({}) into memory instead of mapping its file: it goes to the GPUs \
         whole and its file is on {network}, where a mapped file is copied to the GPUs a page \
         at a time as it is read, and servers mapping one file at once took up to half an hour \
         to load (--mmap keeps it mapped)",
        gib(model_bytes)
    ))
}

/// Whether llama.cpp's expert cache can run on this machine: one GPU of
/// ggml's CUDA backend (see [`moe_cache_support_on`]). `Err` says why not,
/// for the log.
pub fn moe_cache_support() -> Result<(), String> {
    moe_cache_support_on(&gpu_names())
}

/// The names ggml gives its GPUs (`CUDA0`, `ROCm0`, ...), in its order.
fn gpu_names() -> Vec<String> {
    gpu_devices()
        .into_iter()
        .map(|dev| {
            // SAFETY: a device of the registry (`gpu_devices`);
            // `ggml_backend_dev_name` returns a NUL-terminated string owned by
            // the backend.
            let name = unsafe { llama_cpp_sys_2::ggml_backend_dev_name(dev) };
            if name.is_null() {
                String::new()
            } else {
                unsafe { std::ffi::CStr::from_ptr(name) }
                    .to_string_lossy()
                    .into_owned()
            }
        })
        .collect()
}

/// [`moe_cache_support`] for the GPUs ggml lists, by name. The cache
/// refuses more than one device. What it runs on is ggml's CUDA backend
/// (pinned host memory, a second stream for the prefetch, patches 0001 to
/// 0004), which ggml also builds for HIP: its GPUs are `CUDA0` built with
/// CUDA and `ROCm0` built for AMD's. ROCm is allowed since 14-10-2026, to be
/// measured on LUMI's MI250X; other backends are refused.
pub fn moe_cache_support_on(gpus: &[String]) -> Result<(), String> {
    match gpus {
        [] => Err("no GPU".to_string()),
        [name] if name.starts_with("CUDA") || name.starts_with("ROCm") => Ok(()),
        [name] => Err(format!(
            "{name} is not a GPU of the CUDA backend (CUDA or ROCm), the only one it runs on"
        )),
        _ => Err(format!(
            "{} GPUs, and llama.cpp's cache works with one",
            gpus.len()
        )),
    }
}

/// Both stdin and stdout connected to a terminal — the same gate the picker
/// uses. A non-TTY invocation (Docker, systemd, piped) must never block on a
/// prompt, so the decision logic checks this before asking anything.
fn interactive() -> bool {
    std::io::stdin().is_terminal() && std::io::stdout().is_terminal()
}

/// Format a byte count as GiB for human-facing log lines.
pub(crate) fn gib(bytes: u64) -> String {
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

/// Layers a split may be short of the whole model for [`partial_split_hint`]
/// to bother naming a lever: a few layers in RAM cost a lot of speed (two of
/// 36 made a dense 8B four times slower) and are within reach of a setting,
/// where half a model in RAM is not.
const HINT_MAX_MISSING_LAYERS: u32 = 4;

/// The smallest `--ctx-size` [`partial_split_hint`] suggests: a window below
/// this is not what anyone asked a long context for.
const HINT_MIN_CTX: u32 = 2048;

/// When a split leaves only a few layers in RAM, what would put them all on
/// the GPU, each worked out with [`compute_fit`] itself so that following it
/// does what it says: the `--vram-reserve` that would do it (when still
/// above [`MIN_VRAM_RESERVE_MIB`]), the `--ctx-size`, and the q8_0 KV cache
/// (when the caches are larger than that). `None` when the split is not
/// near-whole, or none of them gets there.
#[allow(clippy::too_many_arguments)]
pub fn partial_split_hint(
    free_vram: u64,
    info: &GgufInfo,
    file_size: u64,
    ctx_size: u32,
    kv_bytes_per_elem_k: f64,
    kv_bytes_per_elem_v: f64,
    vram_reserve: u64,
    layers: i32,
    n_layers: u32,
) -> Option<String> {
    let on_gpu = u32::try_from(layers).ok()?;
    if on_gpu >= n_layers || n_layers - on_gpu > HINT_MAX_MISSING_LAYERS {
        return None;
    }
    let fits = |ctx: u32, k: f64, v: f64, reserve: u64| {
        compute_fit(
            Some((free_vram, free_vram)),
            Some(info),
            file_size,
            ctx,
            k,
            v,
            reserve,
        ) == FitDecision::FitsFully
    };
    let mut levers = Vec::new();

    // The reserve: the layers' bytes missing come out of it one for one.
    // Walk down in 64 MiB steps from the current one, no further than the
    // minimum.
    let mut reserve_mib = (vram_reserve >> 20) as u32;
    while reserve_mib > MIN_VRAM_RESERVE_MIB {
        reserve_mib = reserve_mib.saturating_sub(64).max(MIN_VRAM_RESERVE_MIB);
        if fits(
            ctx_size,
            kv_bytes_per_elem_k,
            kv_bytes_per_elem_v,
            vram_reserve_bytes(reserve_mib),
        ) {
            levers.push(format!(
                "--vram-reserve {reserve_mib} (now {})",
                vram_reserve >> 20
            ));
            break;
        }
    }

    // The window, in steps of 1,024 tokens.
    let mut ctx = ctx_size / 1024 * 1024;
    if ctx == ctx_size {
        ctx = ctx.saturating_sub(1024);
    }
    while ctx >= HINT_MIN_CTX {
        if fits(ctx, kv_bytes_per_elem_k, kv_bytes_per_elem_v, vram_reserve) {
            levers.push(format!("--ctx-size {ctx} (now {ctx_size})"));
            break;
        }
        ctx -= 1024;
    }

    // The KV cache at 8 bits, where it is larger than that now.
    let q8 = crate::inference::cache_type_bytes_per_elem(&crate::inference::KvCacheType::Q8_0);
    if (kv_bytes_per_elem_k > q8 || kv_bytes_per_elem_v > q8)
        && fits(ctx_size, q8.min(kv_bytes_per_elem_k), q8.min(kv_bytes_per_elem_v), vram_reserve)
    {
        levers.push(
            "--cache-type-k q8_0 --cache-type-v q8_0 (less exact attention over a long window)"
                .to_string(),
        );
    }

    if levers.is_empty() {
        return None;
    }
    Some(format!(
        "All {n_layers} layers would be on the GPU with any one of: {}.",
        levers.join("; ")
    ))
}

/// Run the `--fit` decision flow and return the effective `gpu_layers`.
///
/// `fallback_gpu_layers` is the user-provided `--gpu-layers`, used whenever
/// fit cannot probe. `strict` is `--fit-strict`.
///
/// Headless safety: this only ever prompts when BOTH stdin and stdout are
/// terminals. A non-interactive invocation proceeds with the computed split
/// (printing a one-line warning) and never blocks.
#[allow(clippy::too_many_arguments)]
pub fn run_fit(
    model_path: &Path,
    fallback_gpu_layers: i32,
    ctx_size: u32,
    strict: bool,
    kv_bytes_per_elem_k: f64,
    kv_bytes_per_elem_v: f64,
    reserve_bytes: u64,
    vram_reserve: u64,
    kv: KvLayout,
) -> FitOutcome {
    run_fit_impl(
        model_path,
        fallback_gpu_layers,
        ctx_size,
        strict,
        kv_bytes_per_elem_k,
        kv_bytes_per_elem_v,
        reserve_bytes,
        vram_reserve,
        kv,
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
#[allow(clippy::too_many_arguments)]
pub fn run_fit_headless(
    model_path: &Path,
    fallback_gpu_layers: i32,
    ctx_size: u32,
    strict: bool,
    kv_bytes_per_elem_k: f64,
    kv_bytes_per_elem_v: f64,
    reserve_bytes: u64,
    vram_reserve: u64,
    kv: KvLayout,
) -> FitOutcome {
    run_fit_impl(
        model_path,
        fallback_gpu_layers,
        ctx_size,
        strict,
        kv_bytes_per_elem_k,
        kv_bytes_per_elem_v,
        reserve_bytes,
        vram_reserve,
        kv,
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
    // Subtracted from free VRAM before this model is sized, protecting a
    // launch-time `--embedding-model` companion's margin
    // (`EMBEDDING_COMPUTE_RESERVE_BYTES`) so it is never counted as space
    // available to this load — the companion's weights and its kept context
    // need no separate bookkeeping here, since they are built before this
    // runs and so already show up as used VRAM in the free-VRAM figure this
    // reads.
    // Zero from every call site except the initial `eullm run`/`eullm
    // serve` launch when that flag was given — a later `load_generation_model`
    // reserves it too, but only while the resident embedder is the
    // reserved companion and not an ad-hoc one (see
    // `EmbeddingSlot::is_reserved_companion`).
    reserve_bytes: u64,
    vram_reserve: u64,
    kv: KvLayout,
    allow_prompt: bool,
    announce: bool,
) -> FitOutcome {
    let vram = vram_bytes().map(|(free, total)| (free.saturating_sub(reserve_bytes), total));
    let free_vram = vram.map(|(free, _)| free);
    let info = read_gguf_info(model_path).map(|info| info.with_kv_layout(kv));
    let file_size = model_file_bytes(model_path);

    let decision = compute_fit(
        vram,
        info.as_ref(),
        file_size,
        ctx_size,
        kv_bytes_per_elem_k,
        kv_bytes_per_elem_v,
        vram_reserve,
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
            let hint = info.as_ref().and_then(|info| {
                partial_split_hint(
                    free,
                    info,
                    file_size,
                    ctx_size,
                    kv_bytes_per_elem_k,
                    kv_bytes_per_elem_v,
                    vram_reserve,
                    layers,
                    n_layers,
                )
            });
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
                if let Some(hint) = &hint {
                    println!("  {hint}");
                }
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
                if let Some(hint) = &hint {
                    eprintln!("[EULLM] {hint}");
                }
                FitOutcome::Proceed(layers)
            }
        }
    }
}

/// Run the MoE-aware fit flow: probe VRAM, read the GGUF header and tensor
/// layout, and delegate to [`compute_moe_fit_plan`]: the decision, and the
/// micro-batch and host bytes that go with it.
///
/// Always non-interactive — unlike [`run_fit`], this never prompts. The
/// caller runs it *before* `run_fit`: a MoE decision here always resolves to
/// a loadable configuration (expert offload, in the worst case combined with
/// a reduced layer split down to fully-CPU), so there is no "doesn't fit,
/// continue anyway?" question left to ask. Only when its decision is
/// [`MoeFitDecision::NotMoe`] does the dense `run_fit` flow — with its
/// prompt and its `--fit-strict` handling — take over.
///
/// `reserve_bytes`: see the identical parameter on `run_fit_impl` — same
/// purpose (protect a launch-time embedding companion's footprint), applied
/// here before the MoE decision instead of the dense one.
#[allow(clippy::too_many_arguments)]
pub fn run_moe_fit_plan(
    model_path: &Path,
    ctx_size: u32,
    kv_bytes_per_elem_k: f64,
    kv_bytes_per_elem_v: f64,
    reserve_bytes: u64,
    vram_reserve: u64,
    auto_n_ubatch: bool,
    kv: KvLayout,
) -> MoeFitPlan {
    let info = read_gguf_info(model_path).map(|info| info.with_kv_layout(kv));
    let file_size = model_file_bytes(model_path);
    let layout = match (&info, file_size) {
        (Some(i), size) if size > 0 => read_model_moe_layout(model_path, i.n_layers),
        _ => None,
    };
    let vram = vram_bytes().map(|(free, total)| (free.saturating_sub(reserve_bytes), total));
    compute_moe_fit_plan(
        vram,
        info.as_ref(),
        layout.as_ref(),
        ctx_size,
        kv_bytes_per_elem_k,
        kv_bytes_per_elem_v,
        vram_reserve,
        auto_n_ubatch,
    )
}

/// [`plan_moe_cache`] for the model at `model_path`, against the VRAM free
/// right now less `reserve_bytes` — [`run_moe_fit_plan`]'s counterpart for the
/// expert cache.
#[allow(clippy::too_many_arguments)]
pub fn run_moe_cache(
    model_path: &Path,
    ctx_size: u32,
    kv_bytes_per_elem_k: f64,
    kv_bytes_per_elem_v: f64,
    reserve_bytes: u64,
    vram_reserve: u64,
    request: MoeCache,
    cpu_moe: bool,
    n_cpu_moe: u32,
    auto_n_ubatch: bool,
    prefetch: MoePrefetch,
    kv: KvLayout,
) -> Option<MoeCachePlan> {
    let info = read_gguf_info(model_path).map(|info| info.with_kv_layout(kv));
    let file_size = model_file_bytes(model_path);
    let layout = match (&info, file_size) {
        (Some(i), size) if size > 0 => read_model_moe_layout(model_path, i.n_layers),
        _ => None,
    };
    let vram = vram_bytes().map(|(free, total)| (free.saturating_sub(reserve_bytes), total));
    plan_moe_cache(
        request,
        vram,
        info.as_ref(),
        layout.as_ref(),
        ctx_size,
        kv_bytes_per_elem_k,
        kv_bytes_per_elem_v,
        vram_reserve,
        cpu_moe,
        n_cpu_moe,
        auto_n_ubatch,
        prefetch,
    )
}

/// A multimodal projector's compute buffer, reserved alongside its weights
/// whenever the projector is sized onto the GPU.
///
/// One measurement so far, and it is offered as one: the BF16 projector of a
/// Qwen3.8 27B vision model reported a CUDA0 compute buffer of 248.10 MiB
/// warming up at 1472×1472, on an RTX 5070 Ti. The buffer is sized from the
/// largest image the projector accepts, so it moves with the model, not the
/// card. 320 MiB matches `COMPUTE_BUFFER_RESERVE_BYTES`, the text side's own
/// flat reserve, and covers that figure with about 30% to spare; it wants a
/// second vision model measured before it is trusted further than that.
pub(crate) const MMPROJ_COMPUTE_RESERVE_BYTES: u64 = 320 * 1024 * 1024;

/// VRAM a multimodal projector occupies once loaded: its file, standing in
/// for its weights, plus its compute buffer.
///
/// `0` for no projector, and for one whose file cannot be read — the load
/// fails on that before any VRAM is spent, so there is nothing to reserve.
pub fn mmproj_footprint_bytes(path: Option<&Path>) -> u64 {
    path.and_then(|p| std::fs::metadata(p).ok())
        .map(|m| m.len().saturating_add(MMPROJ_COMPUTE_RESERVE_BYTES))
        .unwrap_or(0)
}

/// Where a multimodal projector runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmprojPlacement {
    /// Beside the text model on the GPU: the whole text model still fits
    /// with the projector's footprint taken out of free VRAM first.
    Gpu,
    /// In system RAM, so the VRAM it would have held goes to text layers.
    Cpu,
    /// VRAM or the model's header could not be read, so there is nothing to
    /// decide with — keep the rule that predates the choice (the projector
    /// follows the text model onto the GPU whenever any layer goes there).
    FollowText,
}

impl MmprojPlacement {
    /// What `--mmproj-offload` / `--no-mmproj-offload` asked for, before any
    /// sizing: a forced placement, or `FollowText` when neither was given
    /// (which sizing then replaces with a decision of its own).
    pub fn from_flag(forced: Option<bool>) -> Self {
        match forced {
            Some(true) => Self::Gpu,
            Some(false) => Self::Cpu,
            None => Self::FollowText,
        }
    }

    /// The `use_gpu` override the projector loader takes: `None` keeps its
    /// own rule, which is what `FollowText` means.
    pub fn on_gpu(self) -> Option<bool> {
        match self {
            Self::Gpu => Some(true),
            Self::Cpu => Some(false),
            Self::FollowText => None,
        }
    }

    /// VRAM to hold back for the projector before the text model is sized.
    ///
    /// Everything except `Cpu`. `FollowText` puts the projector on the GPU
    /// whenever a single text layer goes there, so counting it is the only
    /// reading that cannot size the text model into the space the projector
    /// is about to take — which is the failure this whole reservation fixes.
    pub fn reserve(self, mmproj_bytes: u64) -> u64 {
        match self {
            Self::Cpu => 0,
            Self::Gpu | Self::FollowText => mmproj_bytes,
        }
    }

    /// One line for the load log. A projector moved to RAM makes images
    /// slower than they were, and the reason has to be on screen where
    /// someone will look for it.
    pub fn describe(self) -> &'static str {
        match self {
            Self::Gpu => "projector on the GPU: the whole text model fits beside it",
            Self::Cpu => {
                "projector in system RAM, so its VRAM goes to the text model — images \
                 are encoded on the CPU; --mmproj-offload keeps it on the GPU instead"
            }
            Self::FollowText => "projector follows the text model's offload",
        }
    }
}

/// Decide where a multimodal projector goes. Pure — see
/// [`decide_mmproj_placement`] for the wrapper that reads VRAM and the file.
///
/// The two things competing for the card are not paid for the same way. A
/// projector runs once per image, while the prompt is being read, and is idle
/// for every token after that and for every request with no image at all. A
/// text layer runs once per token, prefill and decode alike, and a layer left
/// in RAM makes every one of those tokens cross the bus. So the projector
/// gets the GPU only when that costs the text model nothing: if the whole
/// text model still fits with the projector beside it, both go on the card;
/// if it would not, the projector moves to RAM first and the text model gets
/// every byte, and only then does sizing start cutting text layers.
///
/// Measured before this existed, on a 16 GiB card: a 27B left 17% of VRAM
/// free by its own sizing, its projector then took 888 MiB of weights and a
/// 248 MiB compute buffer out of that, and the context probe found 10% where
/// it requires 12% — at every size down to its 512-token floor.
#[allow(clippy::too_many_arguments)]
pub fn place_mmproj(
    vram: Option<(u64, u64)>,
    info: Option<&GgufInfo>,
    layout: Option<&MoeLayout>,
    file_size: u64,
    ctx_size: u32,
    kv_bytes_per_elem_k: f64,
    kv_bytes_per_elem_v: f64,
    vram_reserve: u64,
    mmproj_bytes: u64,
) -> MmprojPlacement {
    if mmproj_bytes == 0 {
        return MmprojPlacement::FollowText;
    }
    let Some((free, total)) = vram else {
        return MmprojPlacement::FollowText;
    };
    let with_projector = Some((free.saturating_sub(mmproj_bytes), total));

    // A MoE that fits fully needs no expert offload at all; any other MoE
    // decision is the text model already paying. A dense model is decided by
    // the dense sizer, and one it cannot read is a model we cannot reason
    // about — which is not the same as one that does not fit.
    let text_fits_fully = match compute_moe_fit(
        with_projector,
        info,
        layout,
        ctx_size,
        kv_bytes_per_elem_k,
        kv_bytes_per_elem_v,
        vram_reserve,
    ) {
        MoeFitDecision::Proceed { n_cpu_moe: 0 } => true,
        MoeFitDecision::NotMoe => match compute_fit(
            with_projector,
            info,
            file_size,
            ctx_size,
            kv_bytes_per_elem_k,
            kv_bytes_per_elem_v,
            vram_reserve,
        ) {
            FitDecision::FitsFully => true,
            FitDecision::Partial { .. } => false,
            FitDecision::Unknown { .. } => return MmprojPlacement::FollowText,
        },
        _ => false,
    };

    if text_fits_fully {
        MmprojPlacement::Gpu
    } else {
        MmprojPlacement::Cpu
    }
}

/// [`place_mmproj`] against the live card and the model on disk.
///
/// Reads exactly what [`run_moe_fit_plan`] reads, so the two agree on what the
/// model is. `reserve_bytes` is every other reservation already in force (a
/// launch-time embedding companion), taken out of free VRAM before the
/// projector is weighed against it; `mmproj_bytes` comes from
/// [`mmproj_footprint_bytes`]; `kv` is how the load lays its KV cache out.
#[allow(clippy::too_many_arguments)]
pub fn decide_mmproj_placement(
    model_path: &Path,
    mmproj_bytes: u64,
    ctx_size: u32,
    kv_bytes_per_elem_k: f64,
    kv_bytes_per_elem_v: f64,
    reserve_bytes: u64,
    vram_reserve: u64,
    kv: KvLayout,
) -> MmprojPlacement {
    if mmproj_bytes == 0 {
        return MmprojPlacement::FollowText;
    }
    // Laid out as the load lays its KV cache out, as every other sizer here
    // reads the model: without it a sliding-window layer was charged the
    // whole context, and Gemma 4 E4B at 256k tokens weighed more than 15 GiB of
    // KV cache instead of 4, so its projector went to RAM beside 5.4 GiB of
    // free VRAM and images were encoded on the CPU (24-11-2026).
    let info = read_gguf_info(model_path).map(|info| info.with_kv_layout(kv));
    let file_size = model_file_bytes(model_path);
    let layout = match (&info, file_size) {
        (Some(i), size) if size > 0 => read_model_moe_layout(model_path, i.n_layers),
        _ => None,
    };
    let vram = vram_bytes().map(|(free, total)| (free.saturating_sub(reserve_bytes), total));
    place_mmproj(
        vram,
        info.as_ref(),
        layout.as_ref(),
        file_size,
        ctx_size,
        kv_bytes_per_elem_k,
        kv_bytes_per_elem_v,
        vram_reserve,
        mmproj_bytes,
    )
}

/// The flags a load is sized under: the user's own, never a split worked
/// out for some other model (see `engine/CLAUDE.md`).
#[derive(Debug, Clone, Copy)]
pub struct OffloadFlags {
    /// `--gpu-layers`: a ceiling on whatever sizing decides; `-1` for none.
    pub gpu_layers: i32,
    /// `--cpu-moe`.
    pub cpu_moe: bool,
    /// `--n-cpu-moe`.
    pub n_cpu_moe: u32,
    /// `--mmproj-offload` / `--no-mmproj-offload`, or `None` to let sizing
    /// place the projector.
    pub mmproj_offload: Option<bool>,
    /// `--moe-cache`, or `None` without the flag or where the cache cannot
    /// run (see [`moe_cache_support`]).
    pub moe_cache: Option<MoeCache>,
    /// No `--n-ubatch` was given: the reserve was sized for the default
    /// micro-batch, and an expert cache may raise it (see
    /// [`MOE_CACHE_N_UBATCH`]).
    pub auto_n_ubatch: bool,
    /// `--moe-prefetch`, and what decides whether an expert cache keeps room
    /// for its slots.
    pub moe_prefetch: MoePrefetch,
    /// `--vram-reserve` in bytes: what must stay free after the load (see
    /// [`DEFAULT_VRAM_RESERVE_MIB`]).
    pub vram_reserve: u64,
}

/// What sizing decided, before the `--gpu-layers` ceiling: what the load
/// log reports, and what `--fit-strict` judges.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OffloadBasis {
    /// A MoE model whose first `n_cpu_moe` layers keep their experts in RAM,
    /// every whole layer on the GPU.
    MoeExperts { n_cpu_moe: u32 },
    /// A MoE model with every expert in RAM, and still only this split of
    /// the rest on the GPU.
    MoeAllExpertsAndPartial { gpu_layers: i32 },
    /// A MoE model with its experts in RAM, the rest on the GPU, and this
    /// many bytes of VRAM caching the experts it uses most (`--moe-cache`);
    /// `asked_bytes` is the size the flag asked for when less was left.
    MoeCache {
        bytes: u64,
        asked_bytes: Option<u64>,
    },
    /// The dense sizer's decision, which also stands for a MoE model that
    /// fits whole and for one whose expert offload the user set.
    Dense(FitDecision),
}

/// How one model is to be loaded: one plan, decided before the load, which
/// the load then uses as it is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OffloadPlan {
    /// Layers on the GPU, after the `--gpu-layers` ceiling; `-1` for all.
    pub gpu_layers: i32,
    pub cpu_moe: bool,
    pub n_cpu_moe: u32,
    pub mmproj: MmprojPlacement,
    pub basis: OffloadBasis,
    /// Sizing cut nothing below what the flags ask for: the text model whole
    /// on the GPU — or with as many layers as `--gpu-layers` allows — beside
    /// whatever experts the user sent to RAM, and the projector on the GPU
    /// unless the user sent it to RAM. A model loads beside other resident
    /// models only on a full plan: a second model gets what the first one
    /// left, and a split it never asked for would quietly divide the card.
    pub full: bool,
    /// The free VRAM the model was sized against, every reservation taken
    /// out — the figure the load log reports.
    pub free_vram: Option<u64>,
    /// What sizing put on the GPU when the `--gpu-layers` ceiling lowered
    /// it, for the log line that says so.
    pub capped_from: Option<i32>,
    /// VRAM for the expert cache, in bytes; `0` for none.
    pub moe_cache_bytes: u64,
    /// Bytes of the experts the cache copies from, kept in RAM; `0` for none.
    pub moe_cache_host_bytes: u64,
    /// The micro-batch the expert cache chose (see [`MoeCachePlan::n_ubatch`]).
    pub n_ubatch: Option<u32>,
    /// VRAM the expert cache left for the prefetch's slots (see
    /// [`MoeCachePlan::prefetch_bytes`]); `0` for none.
    pub moe_prefetch_bytes: u64,
    /// Whether the load may grow the cache into the VRAM left over (see
    /// [`MoeCacheGrow`]).
    pub moe_cache_grow: Option<MoeCacheGrow>,
    /// The VRAM the plan kept free beside the text model: the reserve it was
    /// given and a projector going to the GPU.
    pub kept_bytes: u64,
}

/// Size one load: where the projector goes, then MoE expert offload, then
/// the dense split, then the `--gpu-layers` ceiling — the order every load
/// of the API server has always followed, now as one pure function over the
/// probed VRAM and the model's header, so that a later step can ask whether
/// the plan fits without loading anything. Prints nothing; see
/// [`OffloadPlan::print_decision`].
///
/// `reserve_bytes` is taken out of free VRAM first: the memory reserved
/// companions and resident models will need that the free-VRAM figure does
/// not show yet. The projector's own footprint is added to it when the
/// projector goes on the GPU, before the text model is sized.
#[allow(clippy::too_many_arguments)]
pub fn plan_offload(
    vram: Option<(u64, u64)>,
    info: Option<&GgufInfo>,
    layout: Option<&MoeLayout>,
    file_size: u64,
    ctx_size: u32,
    kv_bytes_per_elem_k: f64,
    kv_bytes_per_elem_v: f64,
    reserve_bytes: u64,
    mmproj_bytes: u64,
    flags: OffloadFlags,
) -> OffloadPlan {
    let without = |reserved: u64| vram.map(|(free, total)| (free.saturating_sub(reserved), total));
    let mmproj = match flags.mmproj_offload {
        None if mmproj_bytes > 0 => place_mmproj(
            without(reserve_bytes),
            info,
            layout,
            file_size,
            ctx_size,
            kv_bytes_per_elem_k,
            kv_bytes_per_elem_v,
            flags.vram_reserve,
            mmproj_bytes,
        ),
        forced => MmprojPlacement::from_flag(forced),
    };
    let vram = without(reserve_bytes.saturating_add(mmproj.reserve(mmproj_bytes)));

    let cache = flags.moe_cache.and_then(|request| {
        plan_moe_cache(
            request,
            vram,
            info,
            layout,
            ctx_size,
            kv_bytes_per_elem_k,
            kv_bytes_per_elem_v,
            flags.vram_reserve,
            flags.cpu_moe,
            flags.n_cpu_moe,
            flags.auto_n_ubatch,
            flags.moe_prefetch,
        )
    });
    if let Some(cache) = cache {
        // Every layer on the GPU but for the experts: the cache was sized
        // with the rest of the model paid for.
        let capped = apply_gpu_layers_ceiling(-1, flags.gpu_layers);
        return OffloadPlan {
            gpu_layers: capped,
            cpu_moe: cache.cpu_moe,
            n_cpu_moe: cache.n_cpu_moe,
            mmproj,
            full: false,
            basis: OffloadBasis::MoeCache {
                bytes: cache.bytes,
                asked_bytes: cache.asked_bytes,
            },
            free_vram: vram.map(|(free, _)| free),
            capped_from: (capped != -1).then_some(-1),
            moe_cache_bytes: cache.bytes,
            moe_cache_host_bytes: cache.host_bytes,
            n_ubatch: cache.n_ubatch,
            moe_prefetch_bytes: cache.prefetch_bytes,
            moe_cache_grow: cache.grow,
            kept_bytes: reserve_bytes.saturating_add(mmproj.reserve(mmproj_bytes)),
        };
    }

    let MoeFitPlan {
        decision: moe,
        n_ubatch: moe_n_ubatch,
        host_bytes: moe_host_bytes,
    } = if flags.cpu_moe || flags.n_cpu_moe > 0 {
        MoeFitPlan {
            decision: MoeFitDecision::NotMoe,
            n_ubatch: None,
            host_bytes: 0,
        }
    } else {
        compute_moe_fit_plan(
            vram,
            info,
            layout,
            ctx_size,
            kv_bytes_per_elem_k,
            kv_bytes_per_elem_v,
            flags.vram_reserve,
            flags.auto_n_ubatch,
        )
    };
    let (gpu_layers, cpu_moe, n_cpu_moe, basis) = match moe {
        MoeFitDecision::Proceed { n_cpu_moe } if n_cpu_moe > 0 => (
            -1,
            flags.cpu_moe,
            n_cpu_moe,
            OffloadBasis::MoeExperts { n_cpu_moe },
        ),
        MoeFitDecision::ProceedCpuMoeAndPartial { gpu_layers } => (
            gpu_layers,
            true,
            flags.n_cpu_moe,
            OffloadBasis::MoeAllExpertsAndPartial { gpu_layers },
        ),
        _ => {
            let decision = compute_fit(
                vram,
                info,
                file_size,
                ctx_size,
                kv_bytes_per_elem_k,
                kv_bytes_per_elem_v,
                flags.vram_reserve,
            );
            let gpu_layers = match decision {
                FitDecision::FitsFully => -1,
                FitDecision::Partial { layers, .. } => layers,
                FitDecision::Unknown { .. } => flags.gpu_layers,
            };
            (
                gpu_layers,
                flags.cpu_moe,
                flags.n_cpu_moe,
                OffloadBasis::Dense(decision),
            )
        }
    };
    let capped = apply_gpu_layers_ceiling(gpu_layers, flags.gpu_layers);
    let text_full = match basis {
        OffloadBasis::Dense(FitDecision::FitsFully) => true,
        // Short of the whole model, but not of the user's own ceiling.
        OffloadBasis::Dense(FitDecision::Partial { layers, .. }) => {
            flags.gpu_layers >= 0 && layers >= flags.gpu_layers
        }
        _ => false,
    };
    let projector_full =
        mmproj_bytes == 0 || mmproj != MmprojPlacement::Cpu || flags.mmproj_offload == Some(false);
    OffloadPlan {
        gpu_layers: capped,
        cpu_moe,
        n_cpu_moe,
        mmproj,
        full: text_full && projector_full,
        basis,
        free_vram: vram.map(|(free, _)| free),
        capped_from: (capped != gpu_layers).then_some(gpu_layers),
        moe_cache_bytes: 0,
        moe_cache_host_bytes: moe_host_bytes,
        n_ubatch: moe_n_ubatch,
        moe_prefetch_bytes: 0,
        moe_cache_grow: None,
        kept_bytes: reserve_bytes.saturating_add(mmproj.reserve(mmproj_bytes)),
    }
}

impl OffloadPlan {
    /// Whether this plan rests on an estimate at all: free VRAM and the
    /// model's header were both read. Without one, whether a model fits
    /// beside others cannot be told, and only the count of residents is
    /// enforced.
    pub fn sized(&self) -> bool {
        !matches!(self.basis, OffloadBasis::Dense(FitDecision::Unknown { .. }))
    }

    /// `--fit-strict` loads only a model the dense sizer put wholly on the
    /// GPU: it refuses a split, and a model it could not size. A MoE plan
    /// always resolves to a loadable configuration and is never refused.
    pub fn refused_by_strict(&self) -> bool {
        matches!(
            self.basis,
            OffloadBasis::Dense(FitDecision::Partial { .. } | FitDecision::Unknown { .. })
        )
    }

    /// The lines sizing has always printed for its decision, on the streams
    /// it has always used, so a load logs what it did as before: nothing for
    /// a model it could not size unless `--fit-strict` refuses it then.
    /// `file_size` is the model's, for the sizes it states.
    pub fn print_decision(&self, file_size: u64, strict: bool, hint: Option<&str>) {
        let free = self.free_vram.unwrap_or(0);
        match &self.basis {
            OffloadBasis::MoeExperts { n_cpu_moe } => tracing::info!(
                "--fit: MoE model — keeping expert tensors on CPU RAM for the \
                 first {n_cpu_moe} layers so the rest fits in VRAM"
            ),
            OffloadBasis::MoeAllExpertsAndPartial { gpu_layers } => tracing::info!(
                "--fit: MoE model — even with every expert tensor on CPU RAM the \
                 rest doesn't fit fully; offloading a reduced layer split ({gpu_layers})"
            ),
            OffloadBasis::MoeCache { bytes, asked_bytes } => tracing::info!(
                "--fit: MoE model — expert tensors in CPU RAM, {} of VRAM caching the \
                 ones it uses most{}{}",
                gib(*bytes),
                asked_bytes
                    .map(|asked| format!(
                        " (--moe-cache asked for {}, that is what was left)",
                        gib(asked)
                    ))
                    .unwrap_or_default(),
                prefetch_room(self.moe_prefetch_bytes)
            ),
            OffloadBasis::Dense(FitDecision::FitsFully) => {
                if let Some(v) = self.free_vram {
                    println!(
                        "[EULLM] --fit: model ({}) fits fully in {} free VRAM → offloading all layers.",
                        gib(file_size),
                        gib(v),
                    );
                }
            }
            OffloadBasis::Dense(FitDecision::Partial { layers, n_layers }) if strict => {
                eprintln!(
                    "[EULLM] --fit-strict: model needs ~{} but only {} VRAM is free; not loading.",
                    gib(file_size),
                    gib(free),
                );
                eprintln!(
                    "[EULLM] Retry without --fit-strict to offload {layers}/{n_layers} layers (rest in RAM)."
                );
            }
            OffloadBasis::Dense(FitDecision::Partial { layers, n_layers }) => {
                eprintln!(
                    "[EULLM] Model larger than free VRAM ({} free, model {}): \
                     offloading {layers}/{n_layers} layers, the rest runs in RAM (slower). \
                     Set --gpu-layers to choose yourself, or --no-fit to disable sizing.",
                    gib(free),
                    gib(file_size),
                );
                if let Some(hint) = hint {
                    eprintln!("[EULLM] {hint}");
                }
            }
            OffloadBasis::Dense(FitDecision::Unknown { reason }) if strict => {
                eprintln!("[EULLM] --fit could not size the model: {reason}.");
                eprintln!(
                    "[EULLM] --fit-strict set: refusing to load without a reliable estimate."
                );
            }
            OffloadBasis::Dense(FitDecision::Unknown { .. }) => {}
        }
    }
}

/// The part of the line about an expert cache that says what it left the
/// prefetch's slots: nothing when it left them nothing.
pub fn prefetch_room(prefetch_bytes: u64) -> String {
    if prefetch_bytes == 0 {
        String::new()
    } else {
        format!(
            ", and {} beside it for the slots of --moe-prefetch",
            gib(prefetch_bytes)
        )
    }
}

/// What a sequential engine's context takes when a request creates it: its
/// KV cache at `ctx_size` plus the flat compute reserve. The engine builds a
/// context per request, so while it is idle this memory is free in the
/// figure every other load is sized against — and an embedder or a second
/// model sized into it leaves the next request without room for its context.
/// It is kept reserved instead. A micro-batch above the default
/// (`--n-ubatch`) grows the compute buffer, and the reserve with it.
pub(crate) fn context_reserve_bytes(
    info: Option<&GgufInfo>,
    ctx_size: u32,
    kv_bytes_per_elem_k: f64,
    kv_bytes_per_elem_v: f64,
    n_ubatch: u32,
) -> u64 {
    kv_cache_bytes(info, ctx_size, kv_bytes_per_elem_k, kv_bytes_per_elem_v)
        .saturating_add(COMPUTE_BUFFER_RESERVE_BYTES as u64)
        .saturating_add(ubatch_reserve_bytes(n_ubatch))
}

#[cfg(test)]
mod gpu_device_tests {
    use super::first_of_each_device;

    #[test]
    fn a_card_two_backends_list_is_counted_once() {
        let id = |s: &str| Some(s.to_string());
        // CUDA0 and Vulkan0 are the same RTX 5070 Ti, Vulkan1 the Radeon.
        let ids = [id("0000:01:00.0"), id("0000:01:00.0"), id("0000:05:00.0")];
        assert_eq!(first_of_each_device(&ids), [0, 2]);
        // A device without an id is kept, as llama.cpp keeps it.
        assert_eq!(first_of_each_device(&[None, None, id("a")]), [0, 1, 2]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_expert_cache_runs_on_one_gpu_of_the_cuda_backend_built_for_cuda_or_rocm() {
        let gpus = |names: &[&str]| names.iter().map(|n| n.to_string()).collect::<Vec<_>>();
        assert!(moe_cache_support_on(&gpus(&["CUDA0"])).is_ok());
        assert!(moe_cache_support_on(&gpus(&["ROCm0"])).is_ok());
        assert!(moe_cache_support_on(&gpus(&["Vulkan0"])).unwrap_err().contains("Vulkan0"));
        assert_eq!(moe_cache_support_on(&gpus(&[])).unwrap_err(), "no GPU");
        assert!(moe_cache_support_on(&gpus(&["ROCm0", "ROCm1"])).unwrap_err().starts_with("2 GPUs"));
    }

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

    /// A block_count that does not fit u32 must not saturate to u32::MAX:
    /// that value would send compute_fit into a ~4-billion-iteration loop
    /// and size a ~32 GiB MoE scratch vector.
    #[test]
    fn overflowing_block_count_is_unknown_not_max() {
        let data = make_gguf(u64::MAX, true);
        assert!(parse_gguf_header(&data).is_none());
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
            nextn_layers: None,
            context_length: None,
            window: None,
            n_expert: None,
            n_expert_used: None,
        }
    }

    const GIB: u64 = 1024 * 1024 * 1024;
    const MIB: u64 = 1024 * 1024;

    #[test]
    fn kv_cache_bytes_counts_every_paying_layer() {
        // Qwen3-0.6B: 28 layers, 8 KV heads of 128 → 2 × 1024 elements per
        // token per layer, 2 bytes each in F16: 112 KiB per token.
        let info = GgufInfo {
            n_head_kv: Some(8),
            key_length: Some(128),
            value_length: Some(128),
            ..info_layers(28)
        };
        assert_eq!(kv_cache_bytes(Some(&info), 1, F16.0, F16.1), 112 * 1024);
        assert_eq!(kv_cache_bytes(Some(&info), 8192, F16.0, F16.1), 896 * MIB);
        // A hybrid model pays on one layer in four.
        let hybrid = GgufInfo {
            full_attention_interval: Some(4),
            ..info.clone()
        };
        assert_eq!(kv_cache_bytes(Some(&hybrid), 8192, F16.0, F16.1), 224 * MIB);
        assert_eq!(kv_cache_bytes(None, 8192, F16.0, F16.1), 0);
    }

    /// The projector from the report that motivated `place_mmproj`: 888 MiB
    /// of BF16 weights, plus the flat compute reserve.
    const PROJECTOR: u64 = 888 * MIB + MMPROJ_COMPUTE_RESERVE_BYTES;

    /// The least free VRAM, in 64 MiB steps, at which the dense sizer puts
    /// this whole model on the GPU. Found rather than hard-coded, so the
    /// placement tests below keep landing where they mean to whatever the
    /// sizer's margins become.
    fn fits_fully_threshold(info: &GgufInfo, file_size: u64, total: u64) -> u64 {
        let mut free = 0;
        loop {
            let d = compute_fit(
                Some((free, total)),
                Some(info),
                file_size,
                4096,
                F16.0,
                F16.1,
                DEFAULT_VRAM_RESERVE_BYTES);
            if d == FitDecision::FitsFully {
                return free;
            }
            free += 64 * MIB;
            assert!(free <= total, "the fixture never fits fully on this card");
        }
    }

    fn place(
        free: u64,
        total: u64,
        info: &GgufInfo,
        file_size: u64,
        projector: u64,
    ) -> MmprojPlacement {
        place_mmproj(
            Some((free, total)),
            Some(info),
            None,
            file_size,
            4096,
            F16.0,
            F16.1,
            DEFAULT_VRAM_RESERVE_BYTES,
            projector)
    }

    /// The case from the report, and the one that has to discriminate: enough
    /// VRAM for the text model alone, not for the text model and its
    /// projector. Half a projector past the threshold is inside that window by
    /// construction, and the first assertion checks the window is real — the
    /// text model on its own does fit there — so a pass cannot come from a
    /// card too small for anything.
    #[test]
    fn the_projector_moves_to_ram_rather_than_cost_the_text_model_a_layer() {
        let (total, file_size) = (16 * GIB, 12 * GIB);
        let info = info_layers(62);
        let t = fits_fully_threshold(&info, file_size, total);
        let free = t + PROJECTOR / 2;

        assert_eq!(
            compute_fit(
                Some((free, total)),
                Some(&info),
                file_size,
                4096,
                F16.0,
                F16.1,
                DEFAULT_VRAM_RESERVE_BYTES
            ),
            FitDecision::FitsFully,
            "the text model alone must fit here, or this proves nothing"
        );
        assert_eq!(
            place(free, total, &info, file_size, PROJECTOR),
            MmprojPlacement::Cpu
        );
    }

    #[test]
    fn the_projector_stays_on_the_gpu_when_both_fit() {
        let (total, file_size) = (16 * GIB, 12 * GIB);
        let info = info_layers(62);
        let free = fits_fully_threshold(&info, file_size, total) + PROJECTOR + 64 * MIB;
        assert!(
            free <= total,
            "the fixture must leave room for both on this card"
        );
        assert_eq!(
            place(free, total, &info, file_size, PROJECTOR),
            MmprojPlacement::Gpu
        );
    }

    /// Once text layers have to leave the GPU anyway, every byte is worth
    /// more to them than to a projector that runs once per image.
    #[test]
    fn a_text_model_that_does_not_fit_alone_sends_the_projector_to_ram() {
        let (total, file_size) = (16 * GIB, 12 * GIB);
        let info = info_layers(62);
        let free = fits_fully_threshold(&info, file_size, total) - 256 * MIB;
        assert_eq!(
            place(free, total, &info, file_size, PROJECTOR),
            MmprojPlacement::Cpu
        );
    }

    /// Nothing to decide with is not the same as a decision: without VRAM, a
    /// projector or a readable header, the projector keeps following the text
    /// model, as it did before there was a choice.
    #[test]
    fn without_vram_a_projector_or_a_header_nothing_is_decided() {
        let info = info_layers(62);
        let unknown = place_mmproj(
            None,
            Some(&info),
            None,
            12 * GIB,
            4096,
            F16.0,
            F16.1,
            DEFAULT_VRAM_RESERVE_BYTES,
            PROJECTOR);
        assert_eq!(unknown, MmprojPlacement::FollowText);
        assert_eq!(
            place(15 * GIB, 16 * GIB, &info, 12 * GIB, 0),
            MmprojPlacement::FollowText
        );
        let headerless = place_mmproj(
            Some((15 * GIB, 16 * GIB)),
            None,
            None,
            12 * GIB,
            4096,
            F16.0,
            F16.1,
            DEFAULT_VRAM_RESERVE_BYTES,
            PROJECTOR);
        assert_eq!(headerless, MmprojPlacement::FollowText);

        assert_eq!(MmprojPlacement::Gpu.on_gpu(), Some(true));
        assert_eq!(MmprojPlacement::Cpu.on_gpu(), Some(false));
        assert_eq!(MmprojPlacement::FollowText.on_gpu(), None);
    }

    /// The flags win over sizing, and only `Cpu` gives the projector's VRAM
    /// back to the text model: `FollowText` still lands the projector on the
    /// GPU as soon as one text layer does, so it has to be counted.
    #[test]
    fn a_forced_placement_and_the_reserve_it_implies() {
        assert_eq!(MmprojPlacement::from_flag(Some(true)), MmprojPlacement::Gpu);
        assert_eq!(
            MmprojPlacement::from_flag(Some(false)),
            MmprojPlacement::Cpu
        );
        assert_eq!(
            MmprojPlacement::from_flag(None),
            MmprojPlacement::FollowText
        );

        assert_eq!(MmprojPlacement::Gpu.reserve(PROJECTOR), PROJECTOR);
        assert_eq!(MmprojPlacement::FollowText.reserve(PROJECTOR), PROJECTOR);
        assert_eq!(MmprojPlacement::Cpu.reserve(PROJECTOR), 0);
    }

    #[test]
    fn the_footprint_is_the_file_plus_its_compute_buffer() {
        assert_eq!(mmproj_footprint_bytes(None), 0);
        let missing =
            std::env::temp_dir().join(format!("eullm-mmproj-missing-{}", uuid::Uuid::new_v4()));
        assert_eq!(mmproj_footprint_bytes(Some(&missing)), 0);

        let file = std::env::temp_dir().join(format!("eullm-mmproj-{}.gguf", uuid::Uuid::new_v4()));
        std::fs::write(&file, vec![0u8; 4096]).unwrap();
        let got = mmproj_footprint_bytes(Some(&file));
        let _ = std::fs::remove_file(&file);
        assert_eq!(got, 4096 + MMPROJ_COMPUTE_RESERVE_BYTES);
    }

    #[test]
    fn fit_unknown_when_no_vram() {
        let info = info_layers(28);
        let d = compute_fit(None, Some(&info), 5_000_000_000, 4096, F16.0, F16.1, DEFAULT_VRAM_RESERVE_BYTES);
        assert!(matches!(d, FitDecision::Unknown { .. }));
    }

    /// An absurd layer count returns Unknown immediately (no million-iteration
    /// loop): real architectures ship under 200 layers, so anything past the
    /// sanity ceiling is a corrupt header, not a model to size.
    #[test]
    fn absurd_layer_count_is_unknown_not_a_hang() {
        let info = info_layers(1_000_000);
        let d = compute_fit(
            Some((8 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024)),
            Some(&info),
            20_000_000_000,
            4096,
            F16.0,
            F16.1,
            DEFAULT_VRAM_RESERVE_BYTES);
        assert!(matches!(d, FitDecision::Unknown { .. }));
    }

    #[test]
    fn fit_full_when_vram_ample() {
        let info = info_layers(28);
        // 40 GiB free, 5 GB model → fits fully.
        let d = compute_fit(
            Some((40 * 1024 * 1024 * 1024, 40 * 1024 * 1024 * 1024)),
            Some(&info),
            5_000_000_000,
            4096,
            F16.0,
            F16.1,
            DEFAULT_VRAM_RESERVE_BYTES);
        assert_eq!(d, FitDecision::FitsFully);
    }

    #[test]
    fn fit_partial_when_vram_tight() {
        let info = info_layers(32);
        // 4 GB free, 16 GB model → only some layers fit.
        let d = compute_fit(
            Some((4_000_000_000, 4_000_000_000)),
            Some(&info),
            16_000_000_000,
            4096,
            F16.0,
            F16.1,
            DEFAULT_VRAM_RESERVE_BYTES);
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
            nextn_layers: None,
            context_length: None,
            window: None,
            n_expert: None,
            n_expert_used: None,
        };
        let free = 15 * 1024 * 1024 * 1024; // ~15 GiB free, 18.5 GB model
        let file = 18_500_000_000;
        // Large context so the KV term is significant.
        let ctx = 32768;

        let f16 = compute_fit(Some((free, free)), Some(&info), file, ctx, 2.0, 2.0, DEFAULT_VRAM_RESERVE_BYTES);
        // Q4_0 ≈ 0.5625 B/elem for both K and V.
        let q4 = compute_fit(Some((free, free)), Some(&info), file, ctx, 0.5625, 0.5625, DEFAULT_VRAM_RESERVE_BYTES);

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

    // ── Properties ───────────────────────────────────────────────────────
    //
    // A GGUF is bytes we did not write. `eullm pull hf.co/...` records no
    // digest, so a corrupt or hostile file reaches this parser intact, and
    // #490 showed what one integer in it can buy: a four-billion-iteration
    // loop and a 32 GiB allocation, both under the lock that serialises model
    // swaps.
    //
    // Uniformly random bytes would fail the magic check and never get inside,
    // so the strategy keeps a valid prelude and randomises what follows —
    // header fields first, then the metadata block. That is where the numbers
    // the sizer trusts actually live. (Reaching deeper than this is what a
    // coverage-guided fuzzer is for; a property test cannot guess its way
    // into a nested branch.)
    use proptest::prelude::*;

    /// A GGUF prelude that passes the magic check, followed by arbitrary bytes.
    fn gguf_shaped() -> impl Strategy<Value = Vec<u8>> {
        (
            any::<u32>(),
            any::<u64>(),
            any::<u64>(),
            proptest::collection::vec(any::<u8>(), 0..512),
        )
            .prop_map(|(version, tensor_count, kv_count, rest)| {
                let mut b = Vec::new();
                b.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
                b.extend_from_slice(&version.to_le_bytes());
                b.extend_from_slice(&tensor_count.to_le_bytes());
                b.extend_from_slice(&kv_count.to_le_bytes());
                b.extend_from_slice(&rest);
                b
            })
    }

    proptest! {
        /// No sequence of bytes makes the header parser panic. It answers or
        /// gives up; it never takes the process with it.
        #[test]
        fn parse_gguf_header_never_panics(data in proptest::collection::vec(any::<u8>(), 0..1024)) {
            let _ = parse_gguf_header(&data);
        }

        /// Same, for bytes that get past the magic and into the parsing proper.
        #[test]
        fn parse_gguf_header_never_panics_on_well_formed_prelude(data in gguf_shaped()) {
            let _ = parse_gguf_header(&data);
        }

        /// #490 as a rule rather than three examples. The parser deliberately
        /// reports whatever the file claims; the ceiling lives in the
        /// consumers, and there are two of them, which is the whole reason to
        /// state this over every absurd count instead of trusting one call
        /// site to remember. A regression does not fail this assertion — it
        /// hangs the test in a four-billion-iteration loop, which is the
        /// loudest signal this class allows.
        #[test]
        fn no_absurd_layer_count_ever_reaches_the_sizing_loop(n in (MAX_LAYERS + 1)..=u32::MAX) {
            let d = compute_fit(
                Some((8 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024)),
                Some(&info_layers(n)),
                20_000_000_000,
                4096,
                F16.0,
                F16.1,
                DEFAULT_VRAM_RESERVE_BYTES);
            prop_assert!(matches!(d, FitDecision::Unknown { .. }), "{n} layers gave {d:?}");
        }
    }
}

#[cfg(test)]
mod plan_offload_tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;
    const MIB: u64 = 1024 * 1024;
    const F16: (f64, f64) = (2.0, 2.0);

    fn attention(n_layers: u32, n_head_kv: u32) -> GgufInfo {
        GgufInfo {
            n_layers,
            n_embd: None,
            n_head: None,
            n_head_kv: Some(n_head_kv),
            key_length: Some(128),
            value_length: Some(128),
            full_attention_interval: None,
            architecture: Some("qwen3".into()),
            nextn_layers: None,
            context_length: None,
            window: None,
            n_expert: None,
            n_expert_used: None,
        }
    }

    /// A dense 14B: 40 layers, 160 KiB of F16 KV per token.
    fn dense() -> (GgufInfo, u64) {
        (attention(40, 8), 8 * GIB + 400 * MIB)
    }

    /// `auto_context` on the dense 14B above, 160 KiB of F16 KV per token,
    /// with `free` bytes free, a 1 GiB reserve, and the window the model was
    /// trained for.
    fn auto_on_dense(free: u64, trained: Option<u32>, align: u32) -> Option<AutoContext> {
        let (mut info, size) = dense();
        info.context_length = trained;
        auto_context(
            Some((free, free)),
            Some(&info),
            size,
            F16.0,
            F16.1,
            GIB,
            0,
            align,
        )
    }

    #[test]
    fn auto_context_stops_at_what_the_model_was_trained_for_when_the_card_has_room() {
        // 80 GiB free: the cache of 32768 tokens (5 GiB) is nothing to it.
        let auto = auto_on_dense(80 * GIB, Some(32_768), 256).unwrap();
        assert_eq!(
            auto,
            AutoContext {
                ctx: 32_768,
                whole: true
            }
        );
        // No header value: the ceiling of the key, not a guess.
        let auto = auto_on_dense(400 * GIB, None, 256).unwrap();
        assert_eq!(auto.ctx, AUTO_CTX_CEILING);
        // A 1M-token model on a big card is still held to it.
        let auto = auto_on_dense(400 * GIB, Some(1_048_576), 256).unwrap();
        assert_eq!(auto.ctx, AUTO_CTX_CEILING);
    }

    #[test]
    fn auto_context_is_the_largest_window_that_keeps_every_layer_on_the_card() {
        // 16 GiB free: weights 8.4 GiB, 1 GiB reserve and the compute buffer
        // leave about 6.3 GiB for the cache, some 41,000 tokens.
        let free = 16 * GIB;
        for align in [256, 768, 4096] {
            let auto = auto_on_dense(free, Some(131_072), align).unwrap();
            assert!(auto.whole);
            assert_eq!(auto.ctx % align, 0, "align {align}");
            let (info, size) = dense();
            let whole_at = |ctx: u32| {
                compute_fit(
                    Some((free, free)),
                    Some(&info),
                    size,
                    ctx,
                    F16.0,
                    F16.1,
                    GIB,
                ) == FitDecision::FitsFully
            };
            assert!(whole_at(auto.ctx), "align {align}: {} must fit", auto.ctx);
            assert!(
                !whole_at(auto.ctx + align),
                "align {align}: {} would fit as well",
                auto.ctx + align
            );
        }
    }

    #[test]
    fn auto_context_gives_way_to_the_weights_only_below_the_floor() {
        // 9.6 GiB free: the weights and the reserve are there, a cache of
        // 4096 tokens (640 MiB) is not.
        let auto = auto_on_dense(9 * GIB + 600 * MIB, Some(131_072), 256).unwrap();
        assert_eq!(
            auto,
            AutoContext {
                ctx: AUTO_CTX_FLOOR,
                whole: false
            }
        );
        // The floor is a multiple of the slots' block, rounded up.
        let auto = auto_on_dense(9 * GIB + 600 * MIB, Some(131_072), 768).unwrap();
        assert_eq!(auto.ctx, 4608);
    }

    #[test]
    fn auto_context_counts_what_is_already_spoken_for() {
        let (info, size) = dense();
        let ask = |extra: u64| {
            auto_context(
                Some((20 * GIB, 20 * GIB)),
                Some(&info),
                size,
                F16.0,
                F16.1,
                GIB,
                extra,
                256,
            )
            .unwrap()
            .ctx
        };
        // 3 GiB of companions are 3 GiB less cache: 160 KiB a token.
        let (without, with) = (ask(0), ask(3 * GIB));
        assert!(without > with);
        let lost = (without - with) as u64 * 160 * 1024;
        assert!(lost.abs_diff(3 * GIB) < 512 * 1024 * 1024, "lost {lost}");
    }

    #[test]
    fn auto_context_says_nothing_without_a_card_or_a_header() {
        let (info, size) = dense();
        assert_eq!(
            auto_context(None, Some(&info), size, F16.0, F16.1, GIB, 0, 256),
            None
        );
        assert_eq!(
            auto_context(
                Some((16 * GIB, 16 * GIB)),
                None,
                size,
                F16.0,
                F16.1,
                GIB,
                0,
                256
            ),
            None
        );
    }

    #[test]
    fn ctx_size_takes_a_number_or_auto() {
        assert_eq!(parse_ctx_size("8192"), Ok(8192));
        assert_eq!(parse_ctx_size("auto"), Ok(CTX_AUTO));
        assert_eq!(parse_ctx_size(" AUTO "), Ok(CTX_AUTO));
        assert_eq!(parse_ctx_size("0"), Ok(CTX_AUTO));
        assert!(parse_ctx_size("big").is_err());
        assert!(parse_ctx_size("-1").is_err());
    }

    /// A 30B-A3B-shaped MoE: 48 layers, 2 GiB outside the experts and
    /// 400 MiB of experts per layer, in three tensors of which the largest
    /// is 150 MiB.
    fn moe() -> (GgufInfo, MoeLayout, u64) {
        let layout = MoeLayout {
            non_expert_bytes: 2 * GIB,
            expert_bytes_per_layer: vec![400 * MIB; 48],
            largest_expert_tensor_bytes: 150 * MIB,
            ..Default::default()
        };
        (attention(48, 4), layout, 2 * GIB + 48 * 400 * MIB)
    }

    fn flags() -> OffloadFlags {
        OffloadFlags {
            gpu_layers: -1,
            cpu_moe: false,
            n_cpu_moe: 0,
            mmproj_offload: None,
            moe_cache: None,
            auto_n_ubatch: false,
            moe_prefetch: MoePrefetch::default(),
            vram_reserve: DEFAULT_VRAM_RESERVE_BYTES,
        }
    }

    /// Qwen3-8B Q4_K_M with a 65,536-token window on a 16 GB RTX 5070 Ti
    /// (15,841 MiB, 15,595 free; measured 17-10-2026). The reserve was 12%
    /// of the card, 1,901 MiB, and the fit kept two of its 36 layers in RAM:
    /// 52 tokens/s on one request against 132 with all 36 on the GPU.
    #[test]
    fn a_dense_8b_with_a_long_window_fits_whole_on_a_16_gb_card() {
        let info = attention(36, 8);
        let card = (15_595 * MIB, 15_841 * MIB);
        let fit = |reserve| {
            compute_fit(Some(card), Some(&info), 4_792 * MIB, 65_536, F16.0, F16.1, reserve)
        };
        assert_eq!(fit(DEFAULT_VRAM_RESERVE_BYTES), FitDecision::FitsFully);
        // The share of the card it used to leave free.
        assert_eq!(
            fit(1_901 * MIB),
            FitDecision::Partial {
                layers: 34,
                n_layers: 36
            }
        );
    }

    /// The split that cost the 8B two layers names what would have kept them,
    /// each of which works when followed; a split far from whole names none.
    #[test]
    fn a_split_a_few_layers_short_names_what_would_fit_the_rest() {
        let info = attention(36, 8);
        let free = 15_595 * MIB;
        let hint = |reserve: u64, layers, k: f64, v: f64| {
            partial_split_hint(free, &info, 4_792 * MIB, 65_536, k, v, reserve, layers, 36)
        };
        let text = hint(1_901 * MIB, 34, F16.0, F16.1).expect("two layers short");
        assert!(text.contains("--vram-reserve 1261 (now 1901"), "{text}");
        assert!(text.contains("--ctx-size 60416 (now 65536)"), "{text}");
        assert!(text.contains("--cache-type-k q8_0 --cache-type-v q8_0"), "{text}");
        // Each one does what it says.
        let fits = |ctx, k, v, reserve| {
            compute_fit(Some((free, free)), Some(&info), 4_792 * MIB, ctx, k, v, reserve)
                == FitDecision::FitsFully
        };
        assert!(fits(65_536, F16.0, F16.1, 1_261 * MIB));
        assert!(fits(60_416, F16.0, F16.1, 1_901 * MIB));
        assert!(fits(65_536, 1.0625, 1.0625, 1_901 * MIB));
        assert!(!fits(61_440 + 1024, F16.0, F16.1, 1_901 * MIB));
        // Already at q8_0: no lever there. Whole, or half the model short:
        // nothing to say.
        let q8 = hint(1_901 * MIB, 34, 1.0625, 1.0625).unwrap_or_default();
        assert!(!q8.contains("--cache-type-k"), "{q8}");
        assert_eq!(hint(1_901 * MIB, 36, F16.0, F16.1), None);
        assert_eq!(hint(1_901 * MIB, 18, F16.0, F16.1), None);
        // A reserve already at the minimum cannot be lowered.
        let floor = hint(256 * MIB, 34, F16.0, F16.1).unwrap_or_default();
        assert!(!floor.contains("--vram-reserve"), "{floor}");
    }

    /// The reserve is the same on a small card and a large one: that is what
    /// moving from a share of the card to MiB changes.
    #[test]
    fn the_reserve_does_not_grow_with_the_card() {
        let info = attention(36, 8);
        let file = 4_792 * MIB;
        // A model that fits whole with a gigabyte to spare, on a 16 GB card
        // and on a 96 GB one.
        for total in [16 * GIB, 96 * GIB] {
            let free = total - GIB / 4;
            assert_eq!(
                compute_fit(
                    Some((free, total)),
                    Some(&info),
                    file,
                    4096,
                    F16.0,
                    F16.1,
                    DEFAULT_VRAM_RESERVE_BYTES
                ),
                FitDecision::FitsFully
            );
        }
        // And the reserve bites exactly where it says: free memory that
        // leaves less than the reserve beside the compute buffer, the KV
        // cache and the weights keeps a layer back.
        let kv = 4096 * 36 * (8 * 128 * 2 * 2);
        let needed = file + kv + COMPUTE_BUFFER_RESERVE_BYTES as u64 + DEFAULT_VRAM_RESERVE_BYTES;
        let at = |free| {
            compute_fit(
                Some((free, 16 * GIB)),
                Some(&info),
                file,
                4096,
                F16.0,
                F16.1,
                DEFAULT_VRAM_RESERVE_BYTES,
            )
        };
        assert_eq!(at(needed), FitDecision::FitsFully);
        assert!(matches!(at(needed - 1), FitDecision::Partial { .. }));
    }

    /// `--vram-reserve` below the minimum is refused with the reason.
    #[test]
    fn a_reserve_below_the_minimum_is_refused() {
        assert_eq!(parse_vram_reserve("1024"), Ok(1024));
        assert_eq!(parse_vram_reserve(" 256 "), Ok(MIN_VRAM_RESERVE_MIB));
        let low = parse_vram_reserve("255").unwrap_err();
        assert!(low.contains("256 MiB minimum"), "{low}");
        assert!(parse_vram_reserve("0").is_err());
        assert!(parse_vram_reserve("1 GiB").unwrap_err().contains("not a number"));
        assert_eq!(vram_reserve_bytes(1024), DEFAULT_VRAM_RESERVE_BYTES);
    }

    /// A split with experts in RAM reads prompts in a larger micro-batch when
    /// none was asked for, sized with the larger compute buffer, and reports
    /// the experts it keeps in RAM; a fixed micro-batch, or a model that fits
    /// whole, is left as it was.
    #[test]
    fn a_split_of_experts_takes_a_larger_micro_batch_unless_one_was_asked() {
        let (info, layout, size) = moe();
        let vram = Some((16 * GIB, 16 * GIB));
        let plan = |auto_n_ubatch, vram| {
            plan_offload(
                vram,
                Some(&info),
                Some(&layout),
                size,
                16_384,
                2.0,
                2.0,
                0,
                0,
                OffloadFlags {
                    auto_n_ubatch,
                    ..flags()
                },
            )
        };

        let fixed = plan(false, vram);
        let OffloadBasis::MoeExperts { n_cpu_moe: kept } = fixed.basis else {
            panic!("a 20 GiB MoE on 16 GiB splits its experts: {:?}", fixed.basis);
        };
        assert_eq!(fixed.n_ubatch, None);
        assert_eq!(fixed.moe_cache_host_bytes, u64::from(kept) * 400 * MIB);

        let auto = plan(true, vram);
        let OffloadBasis::MoeExperts { n_cpu_moe: more } = auto.basis else {
            panic!("still a split: {:?}", auto.basis);
        };
        assert_eq!(auto.n_ubatch, Some(MOE_CACHE_N_UBATCH));
        // The larger compute buffer is paid for by keeping more in RAM, never less.
        assert!(more >= kept);
        assert_eq!(auto.moe_cache_host_bytes, u64::from(more) * 400 * MIB);

        let roomy = plan(true, Some((80 * GIB, 80 * GIB)));
        assert_eq!(roomy.n_ubatch, None);
        assert_eq!(roomy.moe_cache_host_bytes, 0);
    }

    /// The sizing every API load ran before `plan_offload`, step for step:
    /// `decide_mmproj_placement`, `run_moe_fit_plan`, `run_fit_headless` and the
    /// ceiling, each against free VRAM less the reservations so far.
    #[allow(clippy::too_many_arguments)]
    fn sized_as_before(
        vram: Option<(u64, u64)>,
        info: Option<&GgufInfo>,
        layout: Option<&MoeLayout>,
        file_size: u64,
        ctx: u32,
        reserve: u64,
        mmproj_bytes: u64,
        flags: OffloadFlags,
    ) -> (i32, bool, u32, MmprojPlacement) {
        let less = |r: u64| vram.map(|(free, total)| (free.saturating_sub(r), total));
        let mut cpu_moe = flags.cpu_moe;
        let mut n_cpu_moe = flags.n_cpu_moe;
        let mut placement = MmprojPlacement::from_flag(flags.mmproj_offload);
        let mut reserve = reserve;
        if flags.mmproj_offload.is_none() {
            placement = if mmproj_bytes == 0 {
                MmprojPlacement::FollowText
            } else {
                place_mmproj(
                    less(reserve),
                    info,
                    layout,
                    file_size,
                    ctx,
                    F16.0,
                    F16.1,
                    DEFAULT_VRAM_RESERVE_BYTES,
                    mmproj_bytes)
            };
        }
        reserve += placement.reserve(mmproj_bytes);
        let moe = if !cpu_moe && n_cpu_moe == 0 {
            compute_moe_fit(less(reserve), info, layout, ctx, F16.0, F16.1, DEFAULT_VRAM_RESERVE_BYTES)
        } else {
            MoeFitDecision::NotMoe
        };
        let gpu_layers = match moe {
            MoeFitDecision::Proceed {
                n_cpu_moe: computed,
            } if computed > 0 => {
                n_cpu_moe = computed;
                -1
            }
            MoeFitDecision::ProceedCpuMoeAndPartial { gpu_layers: gl } => {
                cpu_moe = true;
                gl
            }
            _ => match compute_fit(less(reserve), info, file_size, ctx, F16.0, F16.1, DEFAULT_VRAM_RESERVE_BYTES) {
                FitDecision::FitsFully => -1,
                FitDecision::Partial { layers, .. } => layers,
                FitDecision::Unknown { .. } => flags.gpu_layers,
            },
        };
        (
            apply_gpu_layers_ceiling(gpu_layers, flags.gpu_layers),
            cpu_moe,
            n_cpu_moe,
            placement,
        )
    }

    #[test]
    fn plan_offload_decides_what_every_load_decided_before() {
        let (dense_info, dense_size) = dense();
        let (moe_info, moe_layout, moe_size) = moe();
        let models: [(Option<&GgufInfo>, Option<&MoeLayout>, u64); 3] = [
            (Some(&dense_info), None, dense_size),
            (Some(&moe_info), Some(&moe_layout), moe_size),
            (None, None, dense_size),
        ];
        let flag_sets = [
            flags(),
            OffloadFlags {
                gpu_layers: 20,
                ..flags()
            },
            OffloadFlags {
                cpu_moe: true,
                ..flags()
            },
            OffloadFlags {
                n_cpu_moe: 10,
                ..flags()
            },
            OffloadFlags {
                mmproj_offload: Some(true),
                ..flags()
            },
            OffloadFlags {
                mmproj_offload: Some(false),
                ..flags()
            },
        ];
        let mut cases = 0;
        for (info, layout, file_size) in models {
            for flags in flag_sets {
                for mmproj_bytes in [0, 1200 * MIB] {
                    for reserve in [0, GIB] {
                        let mut cards = vec![None];
                        cards.extend((1..=96).map(|q| Some((q * 256 * MIB, 24 * GIB))));
                        for vram in cards {
                            for ctx in [4096, 32768] {
                                let plan = plan_offload(
                                    vram,
                                    info,
                                    layout,
                                    file_size,
                                    ctx,
                                    F16.0,
                                    F16.1,
                                    reserve,
                                    mmproj_bytes,
                                    flags,
                                );
                                let before = sized_as_before(
                                    vram,
                                    info,
                                    layout,
                                    file_size,
                                    ctx,
                                    reserve,
                                    mmproj_bytes,
                                    flags,
                                );
                                assert_eq!(
                                    (plan.gpu_layers, plan.cpu_moe, plan.n_cpu_moe, plan.mmproj),
                                    before,
                                    "{vram:?} {flags:?} mmproj {mmproj_bytes} reserve {reserve} \
                                     ctx {ctx} moe {}",
                                    layout.is_some()
                                );
                                cases += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(cases > 10_000, "{cases}");
    }

    #[test]
    fn the_plan_says_what_it_decided_and_what_strict_refuses() {
        let (info, file_size) = dense();
        let plan = |vram, flags| {
            plan_offload(
                vram,
                Some(&info),
                None,
                file_size,
                4096,
                F16.0,
                F16.1,
                0,
                0,
                flags,
            )
        };

        let whole = plan(Some((15 * GIB, 16 * GIB)), flags());
        assert_eq!(whole.basis, OffloadBasis::Dense(FitDecision::FitsFully));
        assert_eq!((whole.gpu_layers, whole.capped_from), (-1, None));
        assert_eq!(whole.free_vram, Some(15 * GIB));
        assert!(!whole.refused_by_strict());

        let split = plan(Some((6 * GIB, 16 * GIB)), flags());
        assert!(matches!(
            split.basis,
            OffloadBasis::Dense(FitDecision::Partial { .. })
        ));
        assert!(split.gpu_layers > 0 && split.gpu_layers < 40);
        assert!(split.refused_by_strict());

        let unknown = plan(
            None,
            OffloadFlags {
                gpu_layers: 12,
                ..flags()
            },
        );
        assert!(matches!(
            unknown.basis,
            OffloadBasis::Dense(FitDecision::Unknown { .. })
        ));
        assert_eq!(unknown.gpu_layers, 12, "the user's --gpu-layers, as given");
        assert!(unknown.refused_by_strict());

        let capped = plan(
            Some((15 * GIB, 16 * GIB)),
            OffloadFlags {
                gpu_layers: 20,
                ..flags()
            },
        );
        assert_eq!((capped.gpu_layers, capped.capped_from), (20, Some(-1)));
    }

    #[test]
    fn a_moe_plan_moves_experts_before_layers_and_is_never_refused() {
        let (info, layout, file_size) = moe();
        let plan = |free| {
            plan_offload(
                Some((free, 16 * GIB)),
                Some(&info),
                Some(&layout),
                file_size,
                4096,
                F16.0,
                F16.1,
                0,
                0,
                flags(),
            )
        };
        let experts = plan(14 * GIB);
        match experts.basis {
            OffloadBasis::MoeExperts { n_cpu_moe } => {
                assert!(n_cpu_moe > 0 && n_cpu_moe < 48);
                assert_eq!((experts.gpu_layers, experts.n_cpu_moe), (-1, n_cpu_moe));
            }
            other => panic!("expected expert offload, got {other:?}"),
        }
        assert!(!experts.refused_by_strict());

        let starved = plan(3 * GIB);
        assert!(matches!(
            starved.basis,
            OffloadBasis::MoeAllExpertsAndPartial { .. }
        ));
        assert!(starved.cpu_moe);
        assert!(!starved.refused_by_strict());
    }

    #[test]
    fn moe_cache_takes_auto_or_a_size_in_mib() {
        assert_eq!(parse_moe_cache("auto"), Ok(MoeCache::Auto));
        assert_eq!(parse_moe_cache("AUTO"), Ok(MoeCache::Auto));
        assert_eq!(parse_moe_cache("6000"), Ok(MoeCache::Mib(6000)));
        assert_eq!(parse_moe_cache("off"), Ok(MoeCache::Off));
        assert_eq!(parse_moe_cache("OFF"), Ok(MoeCache::Off));
        for bad in ["0", "-1", "6GB", ""] {
            assert!(parse_moe_cache(bad).is_err(), "accepted {bad:?}");
        }
    }

    /// The cache is on by itself only where nothing else says where the
    /// experts go, and on one GPU of the CUDA backend; what the user asked
    /// for is kept, `off` aside, which is no cache at all.
    #[test]
    fn the_expert_cache_comes_on_by_itself_only_where_nothing_places_the_experts() {
        let cuda = vec!["CUDA0".to_string()];
        let on = |asked, fit, cpu_moe, n_cpu_moe, mtp, gpus: &[String]| {
            effective_moe_cache_on(asked, fit, cpu_moe, n_cpu_moe, mtp, gpus)
        };
        for asked in [MoeCache::Auto, MoeCache::Mib(4000)] {
            // The user's word wins whatever the other flags and the GPUs are.
            assert_eq!(on(Some(asked), false, true, 7, 2, &[]), Some(asked));
        }
        // `off` plans nothing and says nothing, whatever the machine.
        for gpus in [vec![], cuda.clone(), vec!["CUDA0".into(), "CUDA1".into()]] {
            assert_eq!(on(Some(MoeCache::Off), true, false, 0, 0, &gpus), None);
        }
        // No sizing (`--no-fit`), experts placed by hand, or drafts: no cache.
        assert_eq!(on(None, false, false, 0, 0, &cuda), None);
        assert_eq!(on(None, true, true, 0, 0, &cuda), None);
        assert_eq!(on(None, true, false, 20, 0, &cuda), None);
        assert_eq!(on(None, true, false, 0, 2, &cuda), None);
        // Otherwise the default, which the planner keeps to where the cache
        // gains, on one GPU of the CUDA backend, CUDA or ROCm; not on two
        // GPUs, on none or on another backend.
        assert_eq!(on(None, true, false, 0, 0, &cuda), Some(MoeCache::Default));
        assert_eq!(
            on(None, true, false, 0, 0, &["ROCm0".to_string()]),
            Some(MoeCache::Default)
        );
        assert_eq!(
            on(None, true, false, 0, 0, &["CUDA0".into(), "CUDA1".into()]),
            None
        );
        assert_eq!(on(None, true, false, 0, 0, &["Vulkan0".to_string()]), None);
        assert_eq!(on(None, true, false, 0, 0, &[]), None);
    }

    /// The cache gets the room the experts would have had on the GPU, and the
    /// experts all go to RAM; a model that fits, or one with no room left
    /// once the rest is placed, gets none.
    #[test]
    fn an_expert_cache_takes_the_room_the_experts_would_have_had() {
        let (info, layout, _) = moe();
        let cache = |request, vram, cpu_moe, n_cpu_moe| {
            plan_moe_cache(
                request,
                Some(vram),
                Some(&info),
                Some(&layout),
                4096,
                F16.0,
                F16.1,
                DEFAULT_VRAM_RESERVE_BYTES,
                cpu_moe,
                n_cpu_moe,
                false,
                MoePrefetch::default())
        };
        let card = (14 * GIB, 16 * GIB);
        let (usable, fixed) = moe_budget(card.0, &info, &layout, 4096, F16.0, F16.1, DEFAULT_VRAM_RESERVE_BYTES);
        let room = (usable - fixed) as u64;

        let auto = cache(MoeCache::Auto, card, false, 0).expect("the experts do not fit");
        assert!(auto.cpu_moe && auto.n_cpu_moe == 0 && auto.asked_bytes.is_none());
        assert_eq!(auto.bytes % (256 * MIB), 0);
        assert!(auto.bytes <= room && room - auto.bytes < 256 * MIB);

        // `off` plans nothing, even though the experts do not fit.
        assert_eq!(cache(MoeCache::Off, card, false, 0), None);

        // A size that fits is used as given, one that does not is cut to the room.
        let asked = cache(MoeCache::Mib(2000), card, false, 0).unwrap();
        assert_eq!((asked.bytes, asked.asked_bytes), (2000 * MIB, None));
        let too_big = cache(MoeCache::Mib(64_000), card, false, 0).unwrap();
        assert_eq!(
            (too_big.bytes, too_big.asked_bytes),
            (auto.bytes, Some(64_000 * MIB))
        );

        // `--n-cpu-moe 40` keeps the last 8 layers' experts on the GPU, paid
        // for before the cache.
        let split = cache(MoeCache::Auto, card, false, 40).unwrap();
        assert!(!split.cpu_moe && split.n_cpu_moe == 40);
        assert!(split.bytes <= room - 8 * 400 * MIB);

        // Only an automatic cache may grow into the VRAM left over, up to the
        // experts in RAM; a size given in MiB is used as given.
        let grow = auto.grow.expect("an automatic cache may grow");
        assert_eq!(grow.max_bytes, auto.host_bytes);
        assert_eq!(grow.keep_bytes, auto.prefetch_bytes);
        assert_eq!(asked.grow, None);

        // The whole model fits on a 48 GB card: no expert is in RAM to cache.
        assert_eq!(cache(MoeCache::Auto, (40 * GIB, 48 * GIB), false, 0), None);
        // On 3 GB the rest of the model leaves nothing.
        assert_eq!(cache(MoeCache::Auto, (3 * GIB, 16 * GIB), false, 0), None);

        // 384 MiB of room: too little for an automatic cache, enough for 300 MiB asked for.
        let tight = fixed as u64 + DEFAULT_VRAM_RESERVE_BYTES + 320 * MIB + 384 * MIB;
        assert_eq!(cache(MoeCache::Auto, (tight, 16 * GIB), false, 0), None);
        let small = cache(MoeCache::Mib(300), (tight, 16 * GIB), false, 0).unwrap();
        assert_eq!(small.bytes, 300 * MIB);

        // Every expert is in RAM for the cache to copy from; with
        // `--n-cpu-moe 40`, the 40 layers' worth.
        let expert_bytes: u64 = layout.expert_bytes_per_layer.iter().sum();
        assert_eq!(auto.host_bytes, expert_bytes);
        assert_eq!(split.host_bytes, 40 * 400 * MIB);
        // Without `auto_n_ubatch` the plan keeps the caller's micro-batch.
        assert_eq!(auto.n_ubatch, None);
    }

    /// Not asked for, the cache is planned only where the share of the
    /// experts the usual split leaves in RAM is in the band it gains in:
    /// not for a model that nearly fits (gpt-oss-120b on a 64 GB GCD), not
    /// where the cache would hold a tenth of them (DeepSeek-V3.1 there).
    /// Asked for, it is planned whatever the share.
    #[test]
    fn the_default_expert_cache_keeps_to_the_band_it_gains_in() {
        let (info, layout, _) = moe();
        let expert_bytes: u64 = layout.expert_bytes_per_layer.iter().sum();
        let cache = |request, free| {
            plan_moe_cache(
                request,
                Some((free, 64 * GIB)),
                Some(&info),
                Some(&layout),
                4096,
                F16.0,
                F16.1,
                DEFAULT_VRAM_RESERVE_BYTES,
                false,
                0,
                false,
                MoePrefetch::default())
        };
        let share = |free| {
            let (usable, fixed) =
                moe_budget(free, &info, &layout, 4096, F16.0, F16.1, DEFAULT_VRAM_RESERVE_BYTES);
            moe_out_share(usable, fixed, expert_bytes)
        };
        // The free VRAM that leaves about this share of the experts out.
        let free_for = |out: f64| {
            (1..=256u64)
                .map(|i| i * 256 * MIB)
                .min_by(|a, b| (share(*a) - out).abs().total_cmp(&(share(*b) - out).abs()))
                .unwrap()
        };
        let (low, high) = MOE_CACHE_DEFAULT_OUT;
        for (out, gains) in [(0.05, false), (0.5, true), (0.9, false)] {
            let free = free_for(out);
            assert!((share(free) - out).abs() < 0.03, "no free VRAM leaves {out} out");
            assert_eq!(
                cache(MoeCache::Default, free).is_some(),
                gains,
                "{out} of the experts out, band {low}-{high}"
            );
            // Asked for, the band does not apply.
            assert!(cache(MoeCache::Auto, free).is_some(), "auto at {out}");
        }
        // In the band the default is `auto`, size and growth alike.
        let free = free_for(0.5);
        let (default, auto) = (cache(MoeCache::Default, free), cache(MoeCache::Auto, free));
        assert_eq!(default, auto);
        assert!(default.unwrap().grow.is_some());
        // A model that fits has nothing to cache, by default or not.
        assert_eq!(cache(MoeCache::Default, 60 * GIB), None);
        assert_eq!(moe_out_share(10.0, 2.0, 0), 0.0);
    }

    /// Left to choose, a load with an expert cache reads prompts with the
    /// larger micro-batch, whose compute buffer comes out of the cache, as
    /// long as the cache still gets its minimum, or the size asked for.
    #[test]
    fn an_expert_cache_reads_prompts_with_a_larger_micro_batch_when_left_to_it() {
        let (info, layout, _) = moe();
        let cache = |request, vram: (u64, u64), auto_n_ubatch| {
            plan_moe_cache(
                request,
                Some(vram),
                Some(&info),
                Some(&layout),
                4096,
                F16.0,
                F16.1,
                DEFAULT_VRAM_RESERVE_BYTES,
                false,
                0,
                auto_n_ubatch,
                MoePrefetch::default())
        };
        let card = (14 * GIB, 16 * GIB);
        let default = cache(MoeCache::Auto, card, false).unwrap();
        let larger = cache(MoeCache::Auto, card, true).unwrap();
        assert_eq!(larger.n_ubatch, Some(MOE_CACHE_N_UBATCH));
        let extra = ubatch_reserve_bytes(MOE_CACHE_N_UBATCH);
        assert!(larger.bytes < default.bytes);
        assert!(default.bytes - larger.bytes <= extra + 256 * MIB);
        assert_eq!(larger.bytes % (256 * MIB), 0);

        // A size asked for that still fits beside the larger buffer: both.
        let asked = cache(MoeCache::Mib(2000), card, true).unwrap();
        assert_eq!((asked.bytes, asked.n_ubatch), (2000 * MIB, Some(MOE_CACHE_N_UBATCH)));
        // One that fits only with the default micro-batch keeps it, whole.
        let (usable, fixed) = moe_budget(card.0, &info, &layout, 4096, F16.0, F16.1, DEFAULT_VRAM_RESERVE_BYTES);
        let room_mib = ((usable - fixed) as u64 / MIB) as u32;
        let snug = cache(MoeCache::Mib(room_mib - 64), card, true).unwrap();
        assert_eq!(snug.bytes, u64::from(room_mib - 64) * MIB);
        assert_eq!((snug.asked_bytes, snug.n_ubatch), (None, None));

        // Too little room for a cache beside the larger buffer: the default
        // micro-batch, and the cache it leaves room for.
        let fixed_and_floor = fixed as u64 + DEFAULT_VRAM_RESERVE_BYTES + 320 * MIB;
        let tight = (fixed_and_floor + GIB, 16 * GIB);
        let kept = cache(MoeCache::Auto, tight, true).unwrap();
        assert_eq!(kept.n_ubatch, None);
        assert_eq!(kept.bytes, cache(MoeCache::Auto, tight, false).unwrap().bytes);
    }

    #[test]
    fn the_prefetch_takes_0_or_2_to_8_slots() {
        assert_eq!(parse_moe_prefetch("0"), Ok(0));
        assert_eq!(parse_moe_prefetch("2"), Ok(2));
        assert_eq!(parse_moe_prefetch("8"), Ok(8));
        for bad in ["1", "9", "-1", "", "auto"] {
            assert!(parse_moe_prefetch(bad).is_err(), "accepted {bad:?}");
        }
        assert_eq!(MOE_PREFETCH_SLOTS, 4);
    }

    /// A slot holds the largest expert tensor, with a MiB at most to spare
    /// for what llama.cpp adds past the end of a copy.
    #[test]
    fn a_prefetch_slot_is_the_largest_expert_tensor_to_the_mib() {
        let slot = |bytes| {
            prefetch_slot_bytes(&MoeLayout {
                largest_expert_tensor_bytes: bytes,
                ..Default::default()
            })
        };
        assert_eq!(slot(256 * MIB), 256 * MIB);
        assert_eq!(slot(256 * MIB + 1), 257 * MIB);
        assert_eq!(slot(1), MIB);
        assert_eq!(slot(0), 0);
    }

    /// The slots go only where experts are kept in RAM and pinned: there is
    /// nothing else they could copy, and from pageable memory llama.cpp turns
    /// them off.
    #[test]
    fn the_prefetch_is_asked_for_only_where_experts_in_ram_are_pinned() {
        assert_eq!(prefetch_slots(4, true, true, 0), 4);
        assert_eq!(prefetch_slots(4, true, false, 12), 4);
        assert_eq!(prefetch_slots(4, false, true, 0), 0, "a mapped file");
        assert_eq!(prefetch_slots(4, true, false, 0), 0, "no expert in RAM");
        assert_eq!(prefetch_slots(0, true, true, 0), 0, "--moe-prefetch 0");

        // With an expert cache, pinned is what reading the model into memory
        // decides: whenever the RAM can spare the experts, unless --mmap.
        let ram = Some(62 * GIB);
        let asked = |no_mmap, keep_mapped, ram_total| MoePrefetch {
            slots: 4,
            no_mmap,
            keep_mapped,
            ram_total,
        };
        assert_eq!(asked(false, false, ram).slots_for(33 * GIB), 4);
        assert_eq!(asked(false, false, ram).slots_for(50 * GIB), 0);
        assert_eq!(asked(false, false, None).slots_for(33 * GIB), 0);
        assert_eq!(asked(false, true, ram).slots_for(33 * GIB), 0);
        assert_eq!(asked(true, false, ram).slots_for(50 * GIB), 4);
    }

    /// An automatic cache leaves the prefetch's slots their VRAM where the
    /// experts are pinned, and only there; a size the user asked for, the
    /// cache's minimum and the larger micro-batch each come before the slots.
    #[test]
    fn an_expert_cache_leaves_the_prefetch_its_slots_where_it_can() {
        let (info, layout, _) = moe();
        let ram = Some(62 * GIB);
        let four = MoePrefetch {
            slots: 4,
            ram_total: ram,
            ..Default::default()
        };
        let cache = |request, vram: (u64, u64), auto_n_ubatch, prefetch| {
            plan_moe_cache(
                request,
                Some(vram),
                Some(&info),
                Some(&layout),
                4096,
                F16.0,
                F16.1,
                DEFAULT_VRAM_RESERVE_BYTES,
                false,
                0,
                auto_n_ubatch,
                prefetch)
        };
        let card = (14 * GIB, 16 * GIB);
        let (usable, fixed) = moe_budget(card.0, &info, &layout, 4096, F16.0, F16.1, DEFAULT_VRAM_RESERVE_BYTES);
        let room = (usable - fixed) as u64;
        let step_down = |bytes: u64| bytes / (256 * MIB) * (256 * MIB);
        let slots = 4 * 150 * MIB;

        // The 18.75 GiB of experts fit the 62 GiB of RAM pinned: the slots'
        // 600 MiB come out of the cache.
        let without = cache(MoeCache::Auto, card, false, MoePrefetch::default()).unwrap();
        let with = cache(MoeCache::Auto, card, false, four).unwrap();
        assert_eq!((without.prefetch_bytes, with.prefetch_bytes), (0, slots));
        assert_eq!(without.bytes, step_down(room));
        assert_eq!(with.bytes, step_down(room - slots));
        assert_eq!(with.n_ubatch, None);

        // Experts left mapped, by --mmap or for want of RAM, cannot be copied
        // ahead: the cache keeps all its room.
        for unpinned in [
            MoePrefetch {
                keep_mapped: true,
                ..four
            },
            MoePrefetch {
                ram_total: Some(16 * GIB),
                ..four
            },
        ] {
            let kept = cache(MoeCache::Auto, card, false, unpinned).unwrap();
            assert_eq!((kept.bytes, kept.prefetch_bytes), (without.bytes, 0));
        }

        // Left to choose, both the larger micro-batch and the slots, when the
        // room holds both beside a cache.
        let both = cache(MoeCache::Auto, card, true, four).unwrap();
        let extra = ubatch_reserve_bytes(MOE_CACHE_N_UBATCH);
        assert_eq!(
            (both.n_ubatch, both.prefetch_bytes),
            (Some(MOE_CACHE_N_UBATCH), slots)
        );
        assert_eq!(both.bytes, step_down(room - extra - slots));

        // A size asked for that fits only without the slots stays whole.
        let room_mib = (room / MIB) as u32;
        let snug = cache(MoeCache::Mib(room_mib - 64), card, false, four).unwrap();
        assert_eq!(snug.bytes, u64::from(room_mib - 64) * MIB);
        assert_eq!((snug.asked_bytes, snug.prefetch_bytes), (None, 0));
        // One that fits beside them gets them.
        let asked = cache(MoeCache::Mib(2000), card, false, four).unwrap();
        assert_eq!((asked.bytes, asked.prefetch_bytes), (2000 * MIB, slots));

        // Room for the larger micro-batch and a cache of 768 MiB (and a bit),
        // which the slots would cut below the minimum: the micro-batch stays,
        // the slots give way.
        let fixed_and_floor = fixed as u64 + DEFAULT_VRAM_RESERVE_BYTES + 320 * MIB;
        let tight = (fixed_and_floor + extra + 832 * MIB, 16 * GIB);
        let larger = cache(MoeCache::Auto, tight, true, four).unwrap();
        assert_eq!(larger.n_ubatch, Some(MOE_CACHE_N_UBATCH));
        assert_eq!((larger.bytes, larger.prefetch_bytes), (768 * MIB, 0));
    }

    /// A plan with an expert cache carries what it left the slots, and says it.
    #[test]
    fn a_plan_with_an_expert_cache_carries_the_prefetchs_room() {
        let (info, layout, file_size) = moe();
        let plan = plan_offload(
            Some((14 * GIB, 16 * GIB)),
            Some(&info),
            Some(&layout),
            file_size,
            4096,
            F16.0,
            F16.1,
            0,
            0,
            OffloadFlags {
                moe_cache: Some(MoeCache::Auto),
                moe_prefetch: MoePrefetch {
                    slots: 4,
                    ram_total: Some(62 * GIB),
                    ..Default::default()
                },
                ..flags()
            },
        );
        assert_eq!(plan.moe_prefetch_bytes, 4 * 150 * MIB);
        assert_eq!(
            prefetch_room(plan.moe_prefetch_bytes),
            ", and 0.59 GiB beside it for the slots of --moe-prefetch"
        );
        assert_eq!(prefetch_room(0), "");
        // Without a cache there is nothing to take the room from.
        let usual = plan_offload(
            Some((14 * GIB, 16 * GIB)),
            Some(&info),
            Some(&layout),
            file_size,
            4096,
            F16.0,
            F16.1,
            0,
            0,
            OffloadFlags {
                moe_prefetch: MoePrefetch {
                    slots: 4,
                    no_mmap: true,
                    ..Default::default()
                },
                ..flags()
            },
        );
        assert_eq!(usual.moe_prefetch_bytes, 0);
    }

    /// Pinned experts must leave a quarter of the RAM, and at least 8 GiB.
    #[test]
    fn experts_are_pinned_only_when_the_ram_can_spare_them() {
        // The reference PC: 62.7 GiB of RAM, 33.02 GiB of experts.
        let ram = 62 * GIB + 700 * MIB;
        assert!(pin_fits_in_ram(33 * GIB, ram));
        assert!(!pin_fits_in_ram(48 * GIB, ram));
        // On 32 GiB the floor is 8 GiB, a quarter.
        assert!(pin_fits_in_ram(24 * GIB, 32 * GIB));
        assert!(!pin_fits_in_ram(24 * GIB + MIB, 32 * GIB));
        // On 16 GiB the 8 GiB floor is more than a quarter.
        assert!(pin_fits_in_ram(8 * GIB, 16 * GIB));
        assert!(!pin_fits_in_ram(9 * GIB, 16 * GIB));
        assert!(!pin_fits_in_ram(GIB, 4 * GIB));
    }

    /// `--no-mmap` always reads the model in; an expert cache does when the
    /// RAM can spare its experts, and says so either way; `--mmap` keeps the
    /// mapping.
    #[test]
    fn an_expert_cache_reads_the_model_into_memory_when_the_ram_allows() {
        let ram = Some(62 * GIB);
        assert_eq!(plan_read_into_memory(true, false, 0, None), (true, None));
        assert_eq!(plan_read_into_memory(false, false, 0, ram), (false, None));
        assert_eq!(plan_read_into_memory(false, true, 33 * GIB, ram), (false, None));

        let (read, why) = plan_read_into_memory(false, false, 33 * GIB, ram);
        assert!(read);
        assert!(why.unwrap().contains("--mmap keeps"));
        let (read, why) = plan_read_into_memory(false, false, 50 * GIB, ram);
        assert!(!read);
        assert!(why.unwrap().contains("--no-mmap reads"));
        let (read, why) = plan_read_into_memory(false, false, 33 * GIB, None);
        assert!(!read);
        assert!(why.unwrap().contains("not known"));
    }

    /// A model that goes to the GPUs whole is read in when its files are more
    /// than half the memory (LUMI-G's 480B: 270 GiB of a 480 GiB job), and
    /// on a local disk stays mapped below (the 235B's 132 GiB), when part of
    /// it stays on the CPU, when the memory is not known, or with `--mmap`.
    #[test]
    fn a_model_on_the_gpus_larger_than_half_the_memory_is_read_in() {
        let job = Some(480 * GIB);
        let why = read_in_whole_on_gpu(270 * GIB, true, job, None, false).expect("read in");
        assert!(why.contains("270.00 GiB") && why.contains("--mmap keeps"), "{why}");
        assert!(why.contains("half of the 480.00 GiB"), "{why}");
        assert_eq!(read_in_whole_on_gpu(132 * GIB, true, job, None, false), None);
        assert_eq!(read_in_whole_on_gpu(240 * GIB, true, job, None, false), None);
        assert_eq!(read_in_whole_on_gpu(270 * GIB, false, job, None, false), None);
        assert_eq!(read_in_whole_on_gpu(270 * GIB, true, None, None, false), None);
        assert_eq!(read_in_whole_on_gpu(270 * GIB, true, job, None, true), None);
    }

    /// On a network file system any model that goes to the GPUs whole is
    /// read in, whatever its size and whether or not the memory is known
    /// (c02 on LUMI-G: servers mapping the 235B and gpt-oss-120b from Lustre
    /// at once took up to half an hour); not one that leaves part of itself
    /// on the CPU, and not with `--mmap`. Half the memory still says why
    /// first.
    #[test]
    fn a_model_on_the_gpus_from_a_network_file_system_is_read_in() {
        let job = Some(480 * GIB);
        let lustre = Some("Lustre");
        for size in [GIB, 59 * GIB, 132 * GIB, 233 * GIB] {
            let why = read_in_whole_on_gpu(size, true, job, lustre, false).expect("read in");
            assert!(why.contains("on Lustre") && why.contains("--mmap keeps"), "{why}");
            assert!(read_in_whole_on_gpu(size, true, None, lustre, false).is_some());
            assert_eq!(read_in_whole_on_gpu(size, false, job, lustre, false), None);
            assert_eq!(read_in_whole_on_gpu(size, true, job, lustre, true), None);
        }
        let why = read_in_whole_on_gpu(270 * GIB, true, job, lustre, false).expect("read in");
        assert!(why.contains("more than half"), "{why}");
    }

    /// GPFS is the network file system that stays mapped (JUPITER, 14-10-2026:
    /// a 16.8 GB model 53 s read in against 10 s mapped, four servers of a
    /// 22.1 GB one at once 34 s against 16), unless the model is more than
    /// half the memory; the file systems nobody measured keep the rule.
    #[test]
    fn a_model_on_gpfs_stays_mapped_below_half_the_memory() {
        let job = Some(480 * GIB);
        let gpfs = Some("GPFS");
        for size in [GIB, 16 * GIB, 22 * GIB, 233 * GIB] {
            assert_eq!(read_in_whole_on_gpu(size, true, job, gpfs, false), None);
            assert_eq!(read_in_whole_on_gpu(size, true, None, gpfs, false), None);
        }
        let why = read_in_whole_on_gpu(270 * GIB, true, job, gpfs, false).expect("read in");
        assert!(why.contains("more than half"), "{why}");
        for fs in ["NFS", "SMB", "BeeGFS", "CephFS", "9p"] {
            let why = read_in_whole_on_gpu(16 * GIB, true, job, Some(fs), false).expect("read in");
            assert!(why.contains(&format!("on {fs}")), "{why}");
        }
    }

    /// All layers is `-1`, or one more than the model's blocks (the output
    /// layer); any expert in RAM is not all on the GPUs.
    #[test]
    fn every_layer_on_the_gpu_counts_the_output_layer_and_the_experts() {
        assert!(every_layer_on_gpu(-1, None, false, 0, 0));
        assert!(every_layer_on_gpu(-1, Some(62), false, 0, 0));
        assert!(every_layer_on_gpu(63, Some(62), false, 0, 0));
        assert!(every_layer_on_gpu(999, Some(62), false, 0, 0));
        assert!(!every_layer_on_gpu(62, Some(62), false, 0, 0));
        assert!(!every_layer_on_gpu(999, None, false, 0, 0));
        assert!(!every_layer_on_gpu(-1, Some(62), true, 0, 0));
        assert!(!every_layer_on_gpu(-1, Some(62), false, 30, 0));
        assert!(!every_layer_on_gpu(-1, Some(62), false, 0, GIB));
    }

    #[test]
    fn a_plan_with_an_expert_cache_puts_every_layer_on_the_gpu() {
        let (info, layout, file_size) = moe();
        let plan = |flags| {
            plan_offload(
                Some((14 * GIB, 16 * GIB)),
                Some(&info),
                Some(&layout),
                file_size,
                4096,
                F16.0,
                F16.1,
                0,
                0,
                flags,
            )
        };
        let auto = OffloadFlags {
            moe_cache: Some(MoeCache::Auto),
            ..flags()
        };
        let cached = plan(auto);
        assert!(matches!(
            cached.basis,
            OffloadBasis::MoeCache {
                asked_bytes: None,
                ..
            }
        ));
        assert_eq!(
            (cached.gpu_layers, cached.cpu_moe, cached.n_cpu_moe),
            (-1, true, 0)
        );
        assert!(cached.moe_cache_bytes > 0 && !cached.full && !cached.refused_by_strict());

        // Without the flag the same card gets the usual split, and no cache.
        let usual = plan(flags());
        assert!(matches!(usual.basis, OffloadBasis::MoeExperts { .. }));
        assert_eq!(usual.moe_cache_bytes, 0);

        // A --gpu-layers ceiling still holds.
        let capped = plan(OffloadFlags {
            gpu_layers: 30,
            ..auto
        });
        assert_eq!((capped.gpu_layers, capped.capped_from), (30, Some(-1)));
    }

    /// When a plan counts as whole, which is what a second resident model
    /// has to be.
    #[test]
    fn a_plan_is_full_only_when_sizing_cut_nothing_the_flags_asked_for() {
        let (info, file_size) = dense();
        let (moe_info, moe_layout, moe_size) = moe();
        let card = |free| Some((free, 16 * GIB));
        let dense_plan = |vram, mmproj_bytes, flags| {
            plan_offload(
                vram,
                Some(&info),
                None,
                file_size,
                4096,
                F16.0,
                F16.1,
                0,
                mmproj_bytes,
                flags,
            )
        };
        let whole = dense_plan(card(15 * GIB), 0, flags());
        assert!(whole.full && whole.sized());

        let split = dense_plan(card(6 * GIB), 0, flags());
        assert!(!split.full && split.sized());
        // The same split under a ceiling it meets: what the flags asked for.
        let ceiling = match split.basis {
            OffloadBasis::Dense(FitDecision::Partial { layers, .. }) => layers,
            ref other => panic!("{other:?}"),
        };
        let at_ceiling = dense_plan(
            card(6 * GIB),
            0,
            OffloadFlags {
                gpu_layers: ceiling,
                ..flags()
            },
        );
        assert!(at_ceiling.full);
        let above_it = dense_plan(
            card(6 * GIB),
            0,
            OffloadFlags {
                gpu_layers: ceiling + 1,
                ..flags()
            },
        );
        assert!(!above_it.full);

        // The text model fits, but not with its projector beside it: sizing
        // sent the projector to RAM, and that is a cut...
        let threshold = (1..=64)
            .map(|q| q * 256 * MIB)
            .find(|&free| dense_plan(card(free), 0, flags()).full)
            .expect("fits on a 16 GiB card");
        let crowded = dense_plan(card(threshold), 1200 * MIB, flags());
        assert_eq!(crowded.mmproj, MmprojPlacement::Cpu);
        assert!(!crowded.full);
        // ...unless the user sent it there.
        let asked = dense_plan(
            card(threshold),
            1200 * MIB,
            OffloadFlags {
                mmproj_offload: Some(false),
                ..flags()
            },
        );
        assert!(asked.full);

        let unknown = dense_plan(None, 0, flags());
        assert!(!unknown.sized() && !unknown.full);

        let moe_plan = |free, flags| {
            plan_offload(
                card(free),
                Some(&moe_info),
                Some(&moe_layout),
                moe_size,
                4096,
                F16.0,
                F16.1,
                0,
                0,
                flags,
            )
        };
        assert!(!moe_plan(14 * GIB, flags()).full, "experts moved to RAM");
        assert!(!moe_plan(3 * GIB, flags()).full);
        let roomy = plan_offload(
            Some((40 * GIB, 48 * GIB)),
            Some(&moe_info),
            Some(&moe_layout),
            moe_size,
            4096,
            F16.0,
            F16.1,
            0,
            0,
            flags(),
        );
        assert!(roomy.full, "a MoE that fits whole: {:?}", roomy.basis);
    }

    /// Qwen3.5-0.8B-MTP's draft context, as its memory breakdown measured
    /// it at a 4,096-token context and a 512-token micro-batch: 8 MiB of KV
    /// for its one MTP layer (2 KV heads of 256, F16), which the formula
    /// gives exactly, and 27 MiB of compute, which the floor covers.
    #[test]
    fn the_mtp_head_context_is_reserved_as_measured() {
        let qwen35_08b = GgufInfo {
            n_embd: Some(1024),
            key_length: Some(256),
            value_length: Some(256),
            full_attention_interval: Some(4),
            architecture: Some("qwen35".into()),
            nextn_layers: Some(1),
            context_length: None,
            ..attention(25, 2)
        };
        let reserve = |info: &GgufInfo, ctx, n_ubatch| {
            mtp_reserve_bytes(Some(info), ctx, F16.0, F16.1, n_ubatch)
        };
        assert_eq!(reserve(&qwen35_08b, 4096, 512), 8 * MIB + 32 * MIB);
        assert!(reserve(&qwen35_08b, 4096, 512) >= 8 * MIB + 27 * MIB);
        // The KV grows with the context, the compute with the micro-batch
        // and the model's width.
        assert_eq!(reserve(&qwen35_08b, 32768, 512), 64 * MIB + 32 * MIB);
        let wider = GgufInfo {
            n_embd: Some(4096),
            ..qwen35_08b.clone()
        };
        assert_eq!(reserve(&wider, 4096, 2048), 8 * MIB + 512 * MIB);
        // No MTP layers, no head: nothing to reserve, and a count the
        // model's own layers cannot hold is a corrupt header.
        for nextn_layers in [None, Some(0), Some(25)] {
            let info = GgufInfo {
                nextn_layers,
                ..qwen35_08b.clone()
            };
            assert_eq!(reserve(&info, 4096, 512), 0, "{nextn_layers:?}");
        }
        assert_eq!(mtp_reserve_bytes(None, 4096, F16.0, F16.1, 512), 0);
    }

    /// `--mtp-model` reserves the head's file whole plus the draft context,
    /// sized as one layer, whatever the model's own GGUF says about MTP.
    #[test]
    fn a_head_in_its_own_file_is_reserved_whole_with_its_context() {
        let info = GgufInfo {
            n_embd: Some(1024),
            key_length: Some(256),
            value_length: Some(256),
            full_attention_interval: Some(4),
            architecture: Some("qwen35".into()),
            nextn_layers: None,
            context_length: None,
            ..attention(25, 2)
        };
        let file = 4 * GIB;
        // The model's own GGUF has no MTP layers: nothing from that side.
        assert_eq!(mtp_reserve_bytes(Some(&info), 4096, F16.0, F16.1, 512), 0);
        // The file plus one layer's KV and the compute floor.
        assert_eq!(
            mtp_file_reserve_bytes(file, Some(&info), 4096, F16.0, F16.1, 512),
            file + 8 * MIB + 32 * MIB
        );
        // The KV grows with the context.
        assert_eq!(
            mtp_file_reserve_bytes(file, Some(&info), 32768, F16.0, F16.1, 512),
            file + 64 * MIB + 32 * MIB
        );
        // Without the model's header only the file can be counted.
        assert_eq!(mtp_file_reserve_bytes(file, None, 4096, F16.0, F16.1, 512), file);
    }

    /// A model sized to fill the card left `--mtp`'s head no room: its
    /// context is built after the load, from the VRAM sizing had handed out.
    /// Reserved, it costs the model the layers it needs instead.
    #[test]
    fn a_model_that_fills_the_card_leaves_the_mtp_head_its_room() {
        let (info, file_size) = dense();
        let info = GgufInfo {
            n_embd: Some(5120),
            nextn_layers: Some(1),
            context_length: None,
            ..info
        };
        let total = 16 * GIB;
        // The least free VRAM, in 64 MiB steps, at which the whole model
        // goes on the GPU.
        let fits_whole = |free| {
            compute_fit(
                Some((free, total)),
                Some(&info),
                file_size,
                4096,
                F16.0,
                F16.1,
                DEFAULT_VRAM_RESERVE_BYTES) == FitDecision::FitsFully
        };
        let free = (1..=total / (64 * MIB))
            .map(|steps| steps * 64 * MIB)
            .find(|&free| fits_whole(free))
            .expect("the fixture fits whole on this card");
        let plan = |reserve| {
            plan_offload(
                Some((free, total)),
                Some(&info),
                None,
                file_size,
                4096,
                F16.0,
                F16.1,
                reserve,
                0,
                flags(),
            )
        };
        assert!(plan(0).full);
        let head = mtp_reserve_bytes(Some(&info), 4096, F16.0, F16.1, 512);
        assert!(head > 0);
        let with_head = plan(head);
        assert!(!with_head.full, "{:?}", with_head.basis);
        assert!(with_head.gpu_layers >= 0 && (with_head.gpu_layers as u32) < info.n_layers);
    }

    #[test]
    fn a_sequential_engine_reserves_its_kv_cache_and_compute_buffer() {
        // Qwen3-0.6B: 28 layers, 8 KV heads of 128 — 112 KiB per token.
        let info = attention(28, 8);
        let default = crate::inference::DEFAULT_N_UBATCH;
        assert_eq!(
            context_reserve_bytes(Some(&info), 8192, F16.0, F16.1, default),
            896 * MIB + 320 * MIB
        );
        assert_eq!(
            context_reserve_bytes(None, 8192, F16.0, F16.1, default),
            320 * MIB
        );
        // A 4096-token micro-batch: 3584 tokens past the default, at
        // 0.75 MiB each.
        assert_eq!(
            context_reserve_bytes(Some(&info), 8192, F16.0, F16.1, 4096),
            896 * MIB + 320 * MIB + 2688 * MIB
        );
    }

    #[test]
    fn a_micro_batch_above_the_default_reserves_its_compute_buffer() {
        assert_eq!(ubatch_reserve_bytes(crate::inference::DEFAULT_N_UBATCH), 0);
        assert_eq!(
            ubatch_reserve_bytes(64),
            0,
            "below the default reserves nothing extra"
        );
        assert_eq!(ubatch_reserve_bytes(1024), 384 * MIB);
        // Qwen3.8-Flash-Next's compute buffer grew by 2,517 MiB from 512 to
        // 4096 tokens; the reserve covers it.
        assert_eq!(ubatch_reserve_bytes(4096), 2688 * MIB);
        assert!(ubatch_reserve_bytes(4096) >= (3165 - 648) * MIB);
    }

    #[test]
    fn a_larger_micro_batch_keeps_fewer_experts_on_the_gpu() {
        let (info, layout, size) = moe();
        let plan = |reserve| {
            plan_offload(
                Some((14 * GIB, 16 * GIB)),
                Some(&info),
                Some(&layout),
                size,
                4096,
                F16.0,
                F16.1,
                reserve,
                0,
                flags(),
            )
        };
        let default = plan(ubatch_reserve_bytes(crate::inference::DEFAULT_N_UBATCH));
        let large = plan(ubatch_reserve_bytes(4096));
        assert_eq!(default, plan(0), "the default is sized exactly as before");
        // 2,688 MiB of compute buffer is 6.7 layers of 400 MiB experts: the
        // larger micro-batch sends 6 or 7 more layers' experts to RAM.
        let moved = large.n_cpu_moe - default.n_cpu_moe;
        assert!(
            (6..=7).contains(&moved),
            "{} → {} layers' experts in RAM",
            default.n_cpu_moe,
            large.n_cpu_moe
        );
        assert_eq!(
            large.gpu_layers, -1,
            "every layer's other weights stay on the GPU"
        );
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

    /// A model with an MTP head writes `nextn_predict_layers` after
    /// `full_attention_interval` (Qwen3.5-0.8B-MTP: keys 30 and 32), where the
    /// parser used to stop once every other key it wanted was in hand.
    #[test]
    fn the_mtp_layer_count_is_read_after_the_attention_interval() {
        let data = header(&[
            (b"qwen35.block_count", 25),
            (b"qwen35.embedding_length", 1024),
            (b"qwen35.attention.head_count", 8),
            (b"qwen35.attention.head_count_kv", 2),
            (b"qwen35.attention.key_length", 256),
            (b"qwen35.attention.value_length", 256),
            (b"qwen35.full_attention_interval", 4),
            (b"qwen35.nextn_predict_layers", 1),
        ]);
        let info = parse_gguf_header(&data).expect("header parses");
        assert_eq!(info.full_attention_interval, Some(4));
        assert_eq!(info.nextn_layers, Some(1));
        let without = header(&[(b"qwen3.block_count", 36)]);
        assert_eq!(
            parse_gguf_header(&without)
                .expect("header parses")
                .nextn_layers,
            None
        );
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
    /// write AFTER the attention dims, and `nextn_predict_layers`, which a
    /// model with an MTP head writes after that, so the fixture is shaped
    /// like one. A model without either reads on to the buffer's end.
    /// A MoE model's expert counts: what writing a token reads of its
    /// experts (`measured_fit`, roadmap MG-D).
    #[test]
    fn reads_the_expert_counts() {
        let data = header(&[
            (b"arch.block_count", 24),
            (b"arch.expert_count", 128),
            (b"arch.expert_used_count", 8),
            (b"arch.expert_shared_count", 1),
        ]);
        let info = parse_gguf_header(&data).expect("parses");
        assert_eq!(info.n_expert, Some(128));
        assert_eq!(info.n_expert_used, Some(8));
    }

    #[test]
    fn stops_reading_once_every_wanted_key_is_in_hand() {
        let data = header(&[
            (b"arch.context_length", 40960),
            (b"arch.block_count", 40),
            (b"arch.embedding_length", 2048),
            (b"arch.attention.head_count", 16),
            (b"arch.attention.head_count_kv", 2),
            (b"arch.attention.key_length", 256),
            (b"arch.attention.value_length", 256),
            (b"arch.full_attention_interval", 4),
            (b"arch.nextn_predict_layers", 1),
            (b"arch.block_count", 99),
        ]);
        let info = parse_gguf_header(&data).expect("parses");
        assert_eq!(info.n_layers, 40);
        assert_eq!(info.context_length, Some(40960));
        assert_eq!(info.full_attention_interval, Some(4));
        assert_eq!(info.nextn_layers, Some(1));
    }

    /// The sizer must never hand the loader a split the loader will refuse.
    /// `inference::probe_and_shrink_context` requires the reserve to still
    /// be free once the model and its context are resident; sizing to leave
    /// a different amount than the probe asks (3% of *free* VRAM, against 12%
    /// of the card's total) broke that on a swap into a 27B on a 16 GiB card
    /// — the weights loaded, then the context allocation failed all the way
    /// down to 512 tokens.
    #[test]
    fn sizing_leaves_the_headroom_the_loader_requires() {
        const GIB: u64 = 1024 * 1024 * 1024;
        let info = GgufInfo {
            n_layers: 64,
            n_embd: Some(5120),
            n_head: Some(24),
            n_head_kv: Some(4),
            key_length: Some(256),
            value_length: Some(256),
            full_attention_interval: Some(4),
            architecture: Some("qwen35".to_string()),
            nextn_layers: None,
            context_length: None,
            window: None,
            n_expert: None,
            n_expert_used: None,
        };
        // The reported case: 14.38 GiB free on a 15.92 GiB card, 15.66 GiB
        // of weights, 4096 context, F16 cache.
        let free = (14.38 * GIB as f64) as u64;
        let total = (15.92 * GIB as f64) as u64;
        let file = (15.66 * GIB as f64) as u64;
        let layers = match compute_fit(Some((free, total)), Some(&info), file, 4096, 2.0, 2.0, DEFAULT_VRAM_RESERVE_BYTES) {
            FitDecision::Partial { layers, .. } => layers,
            other => panic!("expected a partial split, got {other:?}"),
        };

        // Recompute what that split costs and check the leftover clears the
        // loader's floor, which is the property that actually matters.
        let per_layer_weight = file as f64 / 64.0;
        let kv_per_paying = 4096.0 * (4.0 * 256.0 * 2.0 + 4.0 * 256.0 * 2.0);
        let paying = info.kv_paying_layers(layers as u64) as f64;
        let used = layers as f64 * per_layer_weight + paying * kv_per_paying;
        let left = free as f64 - used - COMPUTE_BUFFER_RESERVE_BYTES;
        assert!(
            left >= DEFAULT_VRAM_RESERVE_BYTES as f64,
            "split of {layers} layers leaves {:.2} GiB, below the loader's {:.2} GiB floor",
            left / GIB as f64,
            DEFAULT_VRAM_RESERVE_BYTES as f64 / GIB as f64
        );
        // And it is still a useful split, not a collapse to CPU.
        assert!(layers > 40, "expected a substantial offload, got {layers}");
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
            nextn_layers: None,
            context_length: None,
            window: None,
            n_expert: None,
            n_expert_used: None,
        };
        let hybrid = GgufInfo {
            full_attention_interval: Some(4),
            ..dense.clone()
        };
        let free = Some((16u64 * 1024 * 1024 * 1024, 16u64 * 1024 * 1024 * 1024));
        let file_size = 22u64 * 1024 * 1024 * 1024;
        let ctx = 131072;
        let dense_layers = match compute_fit(free, Some(&dense), file_size, ctx, 2.0, 2.0, DEFAULT_VRAM_RESERVE_BYTES) {
            FitDecision::Partial { layers, .. } => layers,
            other => panic!("expected partial, got {other:?}"),
        };
        let hybrid_layers = match compute_fit(free, Some(&hybrid), file_size, ctx, 2.0, 2.0, DEFAULT_VRAM_RESERVE_BYTES) {
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
            nextn_layers: None,
            context_length: None,
            window: None,
            n_expert: None,
            n_expert_used: None,
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

        let layers = |i: &GgufInfo| match compute_fit(Some((free, free)), Some(i), file, ctx, 2.0, 2.0, DEFAULT_VRAM_RESERVE_BYTES) {
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
    fn split_names_are_read_as_gguf_split_writes_them() {
        assert_eq!(
            split_gguf_name("DeepSeek-V3.1-Q4_K_M-00001-of-00009.gguf"),
            Some(("DeepSeek-V3.1-Q4_K_M", 9))
        );
        assert_eq!(split_gguf_name("m-00009-of-00009.gguf"), Some(("m", 9)));
        for not_split in [
            "Qwen3-8B-Q4_K_M.gguf",
            "m-1-of-2.gguf",
            "m-00010-of-00009.gguf",
            "m-00000-of-00009.gguf",
            "-00001-of-00002.gguf",
            "m-00001-of-00002.bin",
        ] {
            assert_eq!(split_gguf_name(not_split), None, "{not_split}");
        }
    }

    // Qwen3.8-Flash-Next's second part is one 28.8 GB table, read from host
    // memory: it is no VRAM cost, and counting it left `--moe-cache` no room.
    #[test]
    fn the_per_layer_embedding_table_is_not_a_vram_cost() {
        let (bytes, size) = make_gguf_with_tensors(
            None,
            &[
                ("token_embd.weight", 1000),
                ("blk.0.attn_q.weight", 100),
                ("blk.0.ffn_up_exps.weight", 3000),
                ("per_layer_token_embd.weight", 90000),
            ],
        );
        let layout = parse_gguf_moe_layout(&bytes, size, 1).expect("parses");
        assert_eq!(layout.non_expert_bytes, 1000 + 100);
        assert_eq!(layout.expert_bytes_per_layer, vec![3000]);
    }

    #[test]
    fn a_split_model_is_sized_and_laid_out_from_every_part() {
        let dir = std::env::temp_dir().join(format!("eullm-split-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let (one, one_size) = make_gguf_with_tensors(
            None,
            &[
                ("token_embd.weight", 1000),
                ("blk.0.attn_q.weight", 100),
                ("blk.0.ffn_up_exps.weight", 3000),
            ],
        );
        let (two, two_size) = make_gguf_with_tensors(
            None,
            &[
                ("blk.1.attn_q.weight", 100),
                ("blk.1.ffn_up_exps.weight", 4000),
                ("output.weight", 500),
            ],
        );
        let first = dir.join("m-00001-of-00002.gguf");
        std::fs::write(&first, &one).unwrap();
        // One part missing: llama.cpp could not load it either, so the file
        // given is taken alone.
        assert_eq!(gguf_parts(&first), vec![first.clone()]);
        assert_eq!(model_file_bytes(&first), one_size);

        std::fs::write(dir.join("m-00002-of-00002.gguf"), &two).unwrap();
        assert_eq!(gguf_parts(&first).len(), 2);
        assert_eq!(model_file_bytes(&first), one_size + two_size);
        let layout = read_model_moe_layout(&first, 2).expect("both parts parse");
        assert_eq!(layout.non_expert_bytes, 1000 + 100 + 100 + 500);
        assert_eq!(layout.expert_bytes_per_layer, vec![3000, 4000]);

        // A file that is not split is itself, as before.
        let single = dir.join("single.gguf");
        std::fs::write(&single, &one).unwrap();
        assert_eq!(gguf_parts(&single), vec![single.clone()]);
        assert_eq!(model_file_bytes(&single), one_size);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_split_model_whose_first_part_holds_only_metadata_is_still_laid_out() {
        let dir = std::env::temp_dir().join(format!("eullm-split-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let (meta, _) = make_gguf_with_tensors(None, &[]);
        let (body, _) = make_gguf_with_tensors(
            None,
            &[
                ("token_embd.weight", 1000),
                ("blk.0.attn_q.weight", 100),
                ("blk.0.ffn_up_exps.weight", 3000),
            ],
        );
        let first = dir.join("m-00001-of-00002.gguf");
        std::fs::write(&first, &meta).unwrap();
        std::fs::write(dir.join("m-00002-of-00002.gguf"), &body).unwrap();
        let layout = read_model_moe_layout(&first, 1).expect("both parts parse");
        assert!(layout.is_moe());
        assert_eq!(layout.non_expert_bytes, 1000 + 100);
        assert_eq!(layout.expert_bytes_per_layer, vec![3000]);
        std::fs::remove_dir_all(&dir).ok();
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
        // Each block's own, the embeddings and the output norm outside them.
        assert_eq!(layout.non_expert_bytes_per_layer, vec![100, 100]);
        assert_eq!(layout.expert_bytes_per_layer, vec![9000, 12000]);
        assert_eq!(
            layout.expert_bytes_per_layer.iter().sum::<u64>(),
            21000
        );
        // The largest expert tensor, not the largest tensor: the embeddings
        // are not copied into a slot.
        assert_eq!(layout.largest_expert_tensor_bytes, 4000);
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

    /// An absurd layer count returns None before sizing any scratch vector:
    /// with valid tensor data the old code allocated ~8 bytes per layer off
    /// one untrusted integer.
    #[test]
    fn absurd_layer_count_returns_none_without_allocating() {
        let (data, file_size) =
            make_gguf_with_tensors(None, &[("blk.0.ffn_gate_exps.weight", 500)]);
        assert!(parse_gguf_moe_layout(&data, file_size, 1_000_000).is_none());
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

    // ── Properties ───────────────────────────────────────────────────────
    //
    // The other half of #490. This entry point is reached from `run_moe_fit_plan`
    // without passing through `compute_fit`, so the ceiling checked there
    // covers nothing here — which is exactly the kind of gap a rule closes
    // and an example does not.
    use proptest::prelude::*;

    proptest! {
        /// No layer count past the ceiling ever gets as far as sizing the
        /// per-layer vector. The data is valid on purpose: with a real tensor
        /// block the parser would otherwise reach the allocation, and at
        /// `u32::MAX` that is 32 GiB. A regression here does not fail the
        /// assertion, it takes the runner's memory with it.
        #[test]
        fn no_absurd_layer_count_ever_sizes_the_per_layer_vector(
            n in (MAX_LAYERS + 1)..=u32::MAX,
        ) {
            let (data, file_size) =
                make_gguf_with_tensors(None, &[("blk.0.ffn_gate_exps.weight", 500)]);
            prop_assert!(parse_gguf_moe_layout(&data, file_size, n).is_none());
        }

        /// And no sequence of bytes makes it panic, whatever layer count it
        /// is handed alongside them.
        #[test]
        fn parse_gguf_moe_layout_never_panics(
            data in proptest::collection::vec(any::<u8>(), 0..1024),
            file_size in any::<u64>(),
            n in 0u32..=MAX_LAYERS,
        ) {
            let _ = parse_gguf_moe_layout(&data, file_size, n);
        }
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
            nextn_layers: None,
            context_length: None,
            window: None,
            n_expert: None,
            n_expert_used: None,
        }
    }

    #[test]
    fn not_moe_when_layout_has_no_experts() {
        let layout = MoeLayout {
            non_expert_bytes: 5_000_000_000,
            expert_bytes_per_layer: vec![0, 0, 0],
            ..Default::default()
        };
        let d = compute_moe_fit(
            Some((40 * 1024 * 1024 * 1024, 40 * 1024 * 1024 * 1024)),
            Some(&info(3)),
            Some(&layout),
            4096,
            F16.0,
            F16.1,
            DEFAULT_VRAM_RESERVE_BYTES);
        assert_eq!(d, MoeFitDecision::NotMoe);
    }

    #[test]
    fn not_moe_when_free_vram_unknown() {
        let layout = MoeLayout {
            non_expert_bytes: 1_000_000_000,
            expert_bytes_per_layer: vec![1_000_000_000],
            ..Default::default()
        };
        let d = compute_moe_fit(None, Some(&info(1)), Some(&layout), 4096, F16.0, F16.1, DEFAULT_VRAM_RESERVE_BYTES);
        assert_eq!(d, MoeFitDecision::NotMoe);
    }

    #[test]
    fn everything_fits_needs_no_eviction() {
        // 4 layers, tiny non-expert + expert bytes, huge free VRAM.
        let layout = MoeLayout {
            non_expert_bytes: 1_000_000_000,
            expert_bytes_per_layer: vec![500_000_000; 4],
            ..Default::default()
        };
        let d = compute_moe_fit(
            Some((40 * 1024 * 1024 * 1024, 40 * 1024 * 1024 * 1024)),
            Some(&info(4)),
            Some(&layout),
            4096,
            F16.0,
            F16.1,
            DEFAULT_VRAM_RESERVE_BYTES);
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
            ..Default::default()
        };
        // usable = free - 1 GiB reserve - 320 MiB compute buffer. Pick free
        // VRAM so that after non-expert (1GB) + a small KV term, there's room
        // for exactly ~1 layer of experts (2GB) but not 2 (4GB). 4.2 GB
        // before the reserve was absolute (it was 12% of the card, 3% of free).
        let free = 4_800_000_000u64;
        let d = compute_moe_fit(Some((free, free)), Some(&info(4)), Some(&layout), 512, F16.0, F16.1, DEFAULT_VRAM_RESERVE_BYTES);
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
            ..Default::default()
        };
        let free = 8_000_000_000u64; // 8 GB free
        let d = compute_moe_fit(Some((free, free)), Some(&info(2)), Some(&layout), 4096, F16.0, F16.1, DEFAULT_VRAM_RESERVE_BYTES);
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
            ..Default::default()
        };
        let free = 2_000_000_000u64; // 2 GB free
        let d = compute_moe_fit(Some((free, free)), Some(&info(8)), Some(&layout), 4096, F16.0, F16.1, DEFAULT_VRAM_RESERVE_BYTES);
        match d {
            MoeFitDecision::ProceedCpuMoeAndPartial { gpu_layers } => {
                assert!(gpu_layers >= 0, "even the worst case must resolve, not abort");
            }
            other => panic!("expected ProceedCpuMoeAndPartial even in the extreme case, got {other:?}"),
        }
    }
}

// ── Pre-download sizing: "will this model run here?" ────────────────────────

/// Total physical RAM in bytes, or `None` where it cannot be read.
///
/// Only needed to answer a question the VRAM probe cannot: whether a model
/// too big for the GPU would at least run with layers on the CPU, or not run
/// at all. `sysconf` covers Linux and macOS; Windows has no libc equivalent
/// and reports unknown, which downgrades a red verdict to amber rather than
/// producing a wrong one.
#[cfg(unix)]
pub fn system_ram_bytes() -> Option<u64> {
    // SAFETY: `sysconf` takes an int and returns a long. No pointers, no
    // allocation, no global state touched.
    let pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) };
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if pages <= 0 || page_size <= 0 {
        return None;
    }
    Some(pages as u64 * page_size as u64)
}

/// Non-Unix: no portable way to read total RAM without pulling in a
/// platform crate. See [`system_ram_bytes`].
#[cfg(not(unix))]
pub fn system_ram_bytes() -> Option<u64> {
    None
}

/// How a model of a known download size is expected to run on this machine.
///
/// This is an **estimate made before downloading**, from the file size alone.
/// It is not [`compute_fit`], which reads the GGUF header and returns an
/// exact layer split — that needs the file. The two answer different
/// questions: this one decides whether the download is worth starting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunVerdict {
    /// Expected to run entirely on the GPU, or comfortably in RAM on a
    /// machine without one.
    Fits,
    /// Expected to run, but not comfortably: layers spilling to the CPU, or
    /// filling most of the available RAM. Usable, considerably slower.
    Tight,
    /// Not expected to run: larger than VRAM and RAM together.
    TooLarge,
    /// Neither VRAM nor RAM could be read, so no honest answer is available.
    Unknown,
}

/// What a model needs resident beyond its weights: KV cache, compute
/// buffers, the context. A fraction of the weights plus a floor, because the
/// fixed costs dominate for a small model and scale for a large one.
///
/// Deliberately rough. The exact number depends on context size, cache
/// types and architecture, none of which are known before the file is
/// downloaded — and being roughly right in the right direction is the whole
/// job of a traffic light.
const OVERHEAD_FRACTION: f64 = 0.15;
const OVERHEAD_FLOOR: u64 = 512 * 1024 * 1024;

/// Fraction of total RAM treated as available to a model. The rest is the
/// operating system, the page cache and everything else already running:
/// a model sized to 100% of RAM swaps rather than runs.
const RAM_USABLE_FRACTION: f64 = 0.70;

/// Classify a download of `file_bytes` against the memory this machine has.
///
/// `vram` is `(free, total)` as [`vram_bytes`] reports it, `vram_reserve` the
/// `--vram-reserve` in bytes, `ram` is total
/// system RAM. Both optional: a CPU-only build has no VRAM to report, and
/// Windows has no RAM figure here.
pub fn classify_download(
    file_bytes: u64,
    vram: Option<(u64, u64)>,
    ram: Option<u64>,
    vram_reserve: u64,
) -> RunVerdict {
    let needed =
        file_bytes + ((file_bytes as f64 * OVERHEAD_FRACTION) as u64).max(OVERHEAD_FLOOR);

    // Same headroom the real sizer applies, so a green light here does not
    // turn into a partial split there: the reserve that must stay free
    // (`--vram-reserve`).
    let usable_vram = vram
        .map(|(free, _)| free.saturating_sub(vram_reserve))
        .unwrap_or(0);
    let usable_ram = ram
        .map(|r| (r as f64 * RAM_USABLE_FRACTION) as u64)
        .unwrap_or(0);

    if vram.is_none() && ram.is_none() {
        return RunVerdict::Unknown;
    }
    if needed <= usable_vram {
        return RunVerdict::Fits;
    }
    // No GPU: a model that fits in the usable slice of RAM runs fine, one
    // that only fits in nearly all of it runs badly.
    if vram.is_none() {
        return if needed <= usable_ram {
            RunVerdict::Fits
        } else if let Some(r) = ram {
            if needed <= r {
                RunVerdict::Tight
            } else {
                RunVerdict::TooLarge
            }
        } else {
            RunVerdict::Unknown
        };
    }
    // A GPU that cannot hold it all: the rest goes to the CPU, which works
    // and is slow. Without a RAM figure, say so rather than guess.
    if ram.is_none() {
        return RunVerdict::Tight;
    }
    if needed <= usable_vram + usable_ram {
        RunVerdict::Tight
    } else {
        RunVerdict::TooLarge
    }
}

#[cfg(test)]
mod download_verdict_tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn a_small_model_on_a_big_card_fits() {
        // 4 GiB of weights on a 24 GiB card with 23 free.
        assert_eq!(
            classify_download(4 * GIB, Some((23 * GIB, 24 * GIB)), Some(64 * GIB), DEFAULT_VRAM_RESERVE_BYTES),
            RunVerdict::Fits
        );
    }

    #[test]
    fn a_model_past_the_card_but_inside_ram_is_tight_not_impossible() {
        // 40 GiB of weights, 24 GiB card, 128 GiB of RAM: this runs, slowly.
        assert_eq!(
            classify_download(40 * GIB, Some((23 * GIB, 24 * GIB)), Some(128 * GIB), DEFAULT_VRAM_RESERVE_BYTES),
            RunVerdict::Tight
        );
    }

    #[test]
    fn a_model_past_both_is_too_large() {
        assert_eq!(
            classify_download(400 * GIB, Some((23 * GIB, 24 * GIB)), Some(64 * GIB), DEFAULT_VRAM_RESERVE_BYTES),
            RunVerdict::TooLarge
        );
    }

    // The loader needs the reserve to stay free, so a card reporting
    // almost all of its memory free still cannot take a model sized to it.
    // Sizing against `free` alone is what produced splits the loader then
    // refused; the same headroom has to apply here or a green light turns
    // into a partial offload after a 20 GB download.
    #[test]
    fn the_loader_floor_is_respected_not_just_free_memory() {
        // 21 GiB of weights on a 24 GiB card with 23.5 free. Free memory
        // alone says yes; the reserve says no.
        assert_ne!(
            classify_download(21 * GIB, Some((23 * GIB, 24 * GIB)), Some(64 * GIB), DEFAULT_VRAM_RESERVE_BYTES),
            RunVerdict::Fits
        );
    }

    #[test]
    fn a_cpu_only_machine_is_judged_on_ram_alone() {
        assert_eq!(
            classify_download(4 * GIB, None, Some(32 * GIB), DEFAULT_VRAM_RESERVE_BYTES),
            RunVerdict::Fits
        );
        // Fits in RAM, but only by filling nearly all of it.
        assert_eq!(
            classify_download(25 * GIB, None, Some(32 * GIB), DEFAULT_VRAM_RESERVE_BYTES),
            RunVerdict::Tight
        );
        assert_eq!(
            classify_download(64 * GIB, None, Some(32 * GIB), DEFAULT_VRAM_RESERVE_BYTES),
            RunVerdict::TooLarge
        );
    }

    // Better to say "unknown" than to paint a red light from no data.
    #[test]
    fn nothing_measurable_is_unknown_not_a_guess() {
        assert_eq!(classify_download(8 * GIB, None, None, DEFAULT_VRAM_RESERVE_BYTES), RunVerdict::Unknown);
    }

    // Windows reports no RAM figure. A model past the card is still known
    // to spill to the CPU, so amber is honest; red would not be.
    #[test]
    fn a_missing_ram_figure_never_produces_a_red_light() {
        assert_eq!(
            classify_download(400 * GIB, Some((23 * GIB, 24 * GIB)), None, DEFAULT_VRAM_RESERVE_BYTES),
            RunVerdict::Tight
        );
    }

    // The overhead floor matters for small models: a 200 MB model does not
    // need 30 MB of context, it needs a few hundred.
    #[test]
    fn small_models_still_carry_the_fixed_overhead() {
        let tiny = 200 * 1024 * 1024;
        // 512 MiB floor + 200 MiB of weights does not fit in 600 MiB.
        assert_ne!(
            classify_download(tiny, Some((600 * 1024 * 1024, 700 * 1024 * 1024)), None, DEFAULT_VRAM_RESERVE_BYTES),
            RunVerdict::Fits
        );
    }
}

#[cfg(test)]
mod balanced_split_tests {
    use super::*;

    const GB: u64 = 1_000_000_000;

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
            nextn_layers: None,
            context_length: None,
            window: None,
            n_expert: None,
            n_expert_used: None,
        }
    }

    /// 40 blocks of 0.5 GB of attention and 3 GB of experts, 1 GB outside
    /// them: a MoE model of 141 GB, for two 64 GB GPUs.
    fn layout() -> MoeLayout {
        MoeLayout {
            non_expert_bytes: 21 * GB,
            non_expert_bytes_per_layer: vec![GB / 2; 40],
            expert_bytes_per_layer: vec![3 * GB; 40],
            largest_expert_tensor_bytes: GB,
        }
    }

    /// The GPU llama.cpp gives each block, and the output layer last, for a
    /// `tensor_split`, as `llama_model::load_tensors` computes it in f32.
    fn llama_devices(split: &[f32], n_layers: usize) -> Vec<usize> {
        let mut cum = Vec::with_capacity(split.len());
        let mut sum = 0.0f32;
        for &s in split {
            sum += s;
            cum.push(sum);
        }
        for c in &mut cum {
            *c /= sum;
        }
        (0..=n_layers)
            .map(|il| {
                let x = il as f32 / (n_layers + 1) as f32;
                cum.iter().position(|&c| c > x).unwrap_or(split.len())
            })
            .collect()
    }

    /// Blocks a GPU gets: the count of each device in `llama_devices`, the
    /// output layer left out.
    fn counts(devices: &[usize], n_gpus: usize) -> Vec<u32> {
        let mut c = vec![0u32; n_gpus];
        for &d in &devices[..devices.len() - 1] {
            c[d] += 1;
        }
        c
    }

    /// GLM-5.3-Flash on two GCDs (11-11-2026), in small: the experts of the
    /// first layers in RAM, counted against both GPUs together, and llama.cpp
    /// giving each GPU half the blocks overflowed the second. The balanced
    /// split gives the first GPU more blocks, and each holds its share.
    #[test]
    fn the_gpus_holding_light_blocks_take_more_of_them() {
        let (layout, info) = (layout(), info(40));
        // 67 GB free each, 2 GB kept: 65 GB a GPU. On the sum, 35 blocks of
        // experts stay: those of the first 5 in RAM.
        let free = [67 * GB, 67 * GB];
        let reserve = 2 * GB;
        // llama.cpp by count: blocks 0-20 to the first, 21-39 and the output
        // to the second, 19 * 3.5 + 1 = 67.5 GB for 65.
        let by_count = llama_devices(&[1.0, 1.0], 40);
        let second: f64 = (0..40)
            .filter(|&l| by_count[l] == 1)
            .map(|l| if l < 5 { 0.5 } else { 3.5 })
            .sum::<f64>()
            + 1.0;
        assert!(second > 65.0, "{second}");

        let split = balance_moe_split(&layout, &info, 4096, 2.0, 2.0, 5, &free, reserve).unwrap();
        assert_eq!(split.n_cpu_moe, 5);
        assert_eq!(split.layers, vec![22, 18]);
        // llama.cpp sends exactly those blocks, and the output layer to the last.
        let devices = llama_devices(&split.tensor_split, 40);
        assert_eq!(counts(&devices, 2), split.layers);
        assert_eq!(devices[40], 1);
        let line = describe_balanced_split(&split, 5);
        assert!(
            line.contains("layers 22/18") && !line.contains("instead of"),
            "{line}"
        );
    }

    /// Each GPU keeps its own compute buffer and reserve, which the sum did
    /// not count: where no cut holds, the experts of one more layer go to RAM.
    #[test]
    fn where_no_cut_holds_one_more_layer_of_experts_goes_to_ram() {
        let (layout, info) = (layout(), info(40));
        let split = balance_moe_split(
            &layout,
            &info,
            4096,
            2.0,
            2.0,
            6,
            &[64 * GB, 64 * GB],
            2 * GB,
        )
        .unwrap();
        assert_eq!(split.n_cpu_moe, 7);
        assert_eq!(split.layers, vec![23, 17]);
        assert_eq!(
            counts(&llama_devices(&split.tensor_split, 40), 2),
            split.layers
        );
        assert!(describe_balanced_split(&split, 6).contains("7 layers in RAM instead of 6"));
    }

    /// Each GPU of a split keeps its reserve, the flat compute buffer and
    /// twice the growth of a large micro-batch: GLM-5.3-Flash asked
    /// 2,795.76 MiB at 2,048 tokens, which once the growth did not hold.
    #[test]
    fn a_split_gpu_keeps_room_for_a_wide_models_compute_buffer() {
        const MIB: u64 = 1024 * 1024;
        let reserve = per_device_reserve(1024 * MIB, 2048);
        assert_eq!(reserve, 1024 * MIB + 320 * MIB + 2 * 1152 * MIB);
        // GLM's 2,795.76 MiB, with half a GiB to spare.
        assert!(reserve > 2796 * MIB + 512 * MIB);
        // At the default micro-batch, the flat buffer alone.
        assert_eq!(per_device_reserve(1024 * MIB, 512), 1344 * MIB);
    }

    /// The proportions land each block where meant for any counts, also on
    /// eight GPUs, with no cut on a block for f32 to round across.
    #[test]
    fn the_proportions_give_llama_cpp_exactly_the_counts() {
        for layers in [
            vec![1u32, 1],
            vec![30, 49],
            vec![60, 18],
            vec![10, 10, 10, 10, 10, 10, 10, 11],
            vec![20, 9, 9, 9, 9, 9, 9, 4],
        ] {
            let n: u32 = layers.iter().sum();
            let devices = llama_devices(&tensor_split_for(&layers), n as usize);
            assert_eq!(counts(&devices, layers.len()), layers);
            assert_eq!(
                devices[n as usize],
                layers.len() - 1,
                "the output layer to the last"
            );
        }
    }

    /// Nothing to balance: one GPU, a dense model, or a model the GPUs cannot
    /// hold however many experts go to RAM.
    #[test]
    fn one_gpu_a_dense_model_or_too_little_vram_split_nothing() {
        let (layout, info) = (layout(), info(40));
        assert_eq!(
            balance_moe_split(&layout, &info, 4096, 2.0, 2.0, 5, &[66 * GB], GB),
            None
        );
        let dense = MoeLayout {
            non_expert_bytes: 100 * GB,
            expert_bytes_per_layer: vec![0; 40],
            ..Default::default()
        };
        assert_eq!(
            balance_moe_split(&dense, &info, 4096, 2.0, 2.0, 0, &[66 * GB, 66 * GB], GB),
            None
        );
        // 21 GB of attention and the rest for 2 * 8 GB.
        assert_eq!(
            balance_moe_split(
                &layout,
                &info,
                4096,
                2.0,
                2.0,
                5,
                &[10 * GB, 10 * GB],
                2 * GB
            ),
            None
        );
    }

    /// A layout read before blocks were weighed spreads the tensors outside
    /// the experts evenly, and still balances.
    #[test]
    fn without_per_block_weights_the_rest_is_spread_evenly() {
        let mut layout = layout();
        layout.non_expert_bytes_per_layer.clear();
        let split = balance_moe_split(
            &layout,
            &info(40),
            4096,
            2.0,
            2.0,
            5,
            &[66 * GB, 66 * GB],
            2 * GB,
        )
        .unwrap();
        assert!(split.layers[0] > split.layers[1], "{:?}", split.layers);
    }
}

#[cfg(test)]
mod sliding_window_tests {
    use super::*;

    const MIB: u64 = 1024 * 1024;

    fn key(k: &[u8], tag: u32, b: &mut Vec<u8>) {
        b.extend_from_slice(&(k.len() as u64).to_le_bytes());
        b.extend_from_slice(k);
        b.extend_from_slice(&tag.to_le_bytes());
    }
    fn put_u32(k: &[u8], v: u32, b: &mut Vec<u8>) {
        key(k, GGUF_TYPE_UINT32, b);
        b.extend_from_slice(&v.to_le_bytes());
    }
    fn put_u32s(k: &[u8], vs: &[u32], b: &mut Vec<u8>) {
        key(k, GGUF_TYPE_ARRAY, b);
        b.extend_from_slice(&GGUF_TYPE_UINT32.to_le_bytes());
        b.extend_from_slice(&(vs.len() as u64).to_le_bytes());
        for v in vs {
            b.extend_from_slice(&v.to_le_bytes());
        }
    }
    fn put_bools(k: &[u8], vs: &[bool], b: &mut Vec<u8>) {
        key(k, GGUF_TYPE_ARRAY, b);
        b.extend_from_slice(&GGUF_TYPE_BOOL.to_le_bytes());
        b.extend_from_slice(&(vs.len() as u64).to_le_bytes());
        b.extend(vs.iter().map(|&v| u8::from(v)));
    }

    /// The header of ggml-org's gemma-4-31B-it Q8_0, as far as KV sizing
    /// reads it: 60 layers, every sixth global with 4 KV heads of 512, the
    /// rest a window of 1024 with 16 KV heads of 256.
    fn gemma_4_31b() -> GgufInfo {
        let global = |il: u32| il % 6 == 5;
        let mut b = Vec::new();
        b.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&11u64.to_le_bytes());
        key(b"general.architecture", GGUF_TYPE_STRING, &mut b);
        b.extend_from_slice(&6u64.to_le_bytes());
        b.extend_from_slice(b"gemma4");
        put_u32(b"gemma4.block_count", 60, &mut b);
        put_u32(b"gemma4.embedding_length", 5376, &mut b);
        put_u32(b"gemma4.attention.head_count", 32, &mut b);
        let heads: Vec<u32> = (0..60).map(|il| if global(il) { 4 } else { 16 }).collect();
        put_u32s(b"gemma4.attention.head_count_kv", &heads, &mut b);
        put_u32(b"gemma4.attention.key_length", 512, &mut b);
        put_u32(b"gemma4.attention.value_length", 512, &mut b);
        put_u32(b"gemma4.attention.sliding_window", 1024, &mut b);
        let pattern: Vec<bool> = (0..60).map(|il| !global(il)).collect();
        put_bools(b"gemma4.attention.sliding_window_pattern", &pattern, &mut b);
        put_u32(b"gemma4.attention.key_length_swa", 256, &mut b);
        put_u32(b"gemma4.attention.value_length_swa", 256, &mut b);
        parse_gguf_header(&b).expect("parsed")
    }

    #[test]
    fn gemma_4s_per_layer_heads_and_window_are_read() {
        let info = gemma_4_31b();
        assert_eq!(info.n_layers, 60);
        assert_eq!(info.n_head_kv, Some(16), "the most any layer has");
        let w = info.window.as_ref().expect("a window");
        assert_eq!(w.n_swa, 1024);
        assert_eq!(w.layers.iter().filter(|&&s| s).count(), 50);
        assert_eq!(w.head_count_kv.len(), 60);
        assert_eq!((w.key_length, w.value_length), (Some(256), Some(256)));
    }

    /// Leonardo's load (19-11-2026): `-c 49152`, 8 slots, F16. llama.cpp
    /// allocates 8 streams of 1536 cells in each window layer (9,600 MiB)
    /// and the whole context in each global one (3,840 MiB). Sizing charged
    /// about 360 MiB before, and said "fits fully".
    #[test]
    fn gemma_4s_kv_is_what_llama_cpp_allocates() {
        let info = gemma_4_31b().with_kv_layout(KvLayout::new(8, 512, false));
        assert_eq!(kv_cache_bytes(Some(&info), 49152, 2.0, 2.0), (9600 + 3840) * MIB);
        // Not knowing the slots, every window layer is charged the whole
        // context: what llama.cpp's own default (`swa_full`) allocates.
        let unknown = gemma_4_31b();
        assert_eq!(kv_cache_bytes(Some(&unknown), 49152, 2.0, 2.0), (38400 + 3840) * MIB);
        // One slot: 1536 cells of window; the whole --kv-unified stream for 8.
        let one = gemma_4_31b().with_kv_layout(KvLayout::new(1, 512, false));
        assert_eq!(kv_cache_bytes(Some(&one), 49152, 2.0, 2.0), (1200 + 3840) * MIB);
        let unified = gemma_4_31b().with_kv_layout(KvLayout::new(8, 512, true));
        let cells = (1024 * 8 + 512) as u64; // 8704, a multiple of 256
        assert_eq!(
            kv_cache_bytes(Some(&unified), 49152, 2.0, 2.0),
            50 * cells * 16 * 1024 + 3840 * MIB
        );
    }

    /// A context shorter than the window: every layer holds it all.
    #[test]
    fn a_short_context_is_held_whole_by_the_window_layers_too() {
        let info = gemma_4_31b().with_kv_layout(KvLayout::new(1, 512, false));
        let all_full = 50 * 1024 * 16 * 1024 + 10 * 1024 * 4 * 2048;
        assert_eq!(kv_cache_bytes(Some(&info), 1024, 2.0, 2.0), all_full);
    }

    /// Gemma 3 gives no pattern in its header: llama.cpp's period of 6 for
    /// the architecture stands in for it.
    #[test]
    fn gemma_3s_pattern_is_the_architectures_period() {
        assert_eq!(
            swa_pattern(12, 6, false),
            [true, true, true, true, true, false, true, true, true, true, true, false]
        );
        assert_eq!(swa_period_of("gemma3"), Some((6, false)));
        assert_eq!(swa_period_of("llama"), None);
        assert_eq!(swa_pattern(4, 2, true), [false, true, false, true]);
    }
}

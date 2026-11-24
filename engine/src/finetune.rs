//! `eullm finetune`: train a model's weights on a text, inside the engine.
//!
//! The same binary that serves a model can adapt it, on the hardware it
//! serves on: llama.cpp's own trainer (`ggml-opt`), the backend the engine
//! already links — CPU, CUDA, or ROCm/HIP. What that trainer supports bounds
//! what this does (see `llama_cpp_2::opt`): F32 weights, full fine-tuning
//! or a subset of tensors chosen by name, no flash attention, the whole
//! context in one micro-batch. Upstream calls it work in progress; this
//! command is honest about the same limits, and checks them before it
//! spends the time to load a model.
//!
//! A run reads a GGUF whose every tensor is F32, trains on a text file (or
//! the `text` field of a JSONL corpus) for some epochs, measures validation
//! loss before and after each one, and writes the trained weights as a new
//! F32 GGUF — to be quantized for deployment as any other model is. The last
//! line it prints is `FINETUNE_RESULT {json}`, for benchmark harnesses.

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::time::Instant;

use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams};
use llama_cpp_2::gguf::GgufContext;
use llama_cpp_2::model::params::{LlamaModelParams, LlamaSplitMode};
use crate::model_tokens::{AddBos, ModelTokens};
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::opt::{
    LrSchedule, NEVER_TRAINED, OptDataset, Optimizer, Pass, TensorFilter, Trainer,
};
use llama_cpp_2::token::LlamaToken;
use serde::Serialize;

/// Report format, for whatever reads `FINETUNE_RESULT` lines.
const SCHEMA: &str = "eullm.finetune/1";

/// The token embeddings: never trained (see `NEVER_TRAINED`), and the
/// table the vocabulary size is read from.
const TOKEN_EMBD: &str = "token_embd.weight";

/// llama.cpp rounds every context up to a multiple of this, and training
/// needs the window, the batch and the context to be one length.
const CTX_MULTIPLE: u32 = 256;

/// `eullm finetune` flags.
#[derive(clap::Args, Debug, Clone)]
pub struct FinetuneOpts {
    /// Model to train: a local `.gguf` path or a model id in the store. Every
    /// tensor must be F32 (convert with `convert_hf_to_gguf.py --outtype f32`).
    pub model: String,

    /// Training text: a plain-text file, or JSONL with a `text` field per line
    /// (the format of Forge's corpora). Documents are separated by the
    /// model's end-of-sequence token.
    #[arg(long)]
    pub data: PathBuf,

    /// Where to write the trained model (an F32 GGUF). Default:
    /// `<model name>-finetuned.gguf` in the current directory.
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Passes over the training data.
    #[arg(long, default_value_t = 2)]
    pub epochs: u32,

    /// Tokens per training window, a multiple of 256. Also the batch and
    /// micro-batch size: the trainer needs all three equal for the K and V
    /// projections to learn, and llama.cpp rounds a context up to 256.
    #[arg(long, default_value_t = 512)]
    pub ctx: u32,

    /// Tokens between the starts of consecutive windows. Default: half the
    /// window, as llama.cpp's own finetune.
    #[arg(long)]
    pub stride: Option<u32>,

    /// Learning rate of the first epoch. Every window is one optimizer
    /// step, a far smaller batch than a pretrained model was trained with,
    /// so its rate is lower than usual: on Qwen3-0.6B-Base at `--ctx 256`,
    /// nine steps at 1e-6 took held-out loss from 1.21 to 0.94, at 1e-5 up
    /// to 1.32. Raise it for a model trained from scratch or a tiny one.
    #[arg(long, default_value_t = 1e-6)]
    pub lr: f32,

    /// Learning-rate floor: halve from `--lr` down to this over
    /// `--decay-epochs`. Unset (or <= 0): constant.
    #[arg(long, default_value_t = -1.0, allow_hyphen_values = true)]
    pub lr_min: f32,

    /// Epochs over which the rate decays to `--lr-min`. Default: all of them.
    #[arg(long, default_value_t = -1.0, allow_hyphen_values = true)]
    pub decay_epochs: f32,

    /// Weight decay.
    #[arg(long, default_value_t = 0.0)]
    pub wd: f32,

    /// Optimizer: adamw (two moments per parameter) or sgd (none).
    #[arg(long, default_value = "adamw")]
    pub optimizer: Optimizer,

    /// Share of the tokens, from the end, kept for validation. 0 trains on
    /// everything and measures nothing.
    #[arg(long, default_value_t = 0.05)]
    pub val_split: f64,

    /// Train only the tensors whose GGUF name matches one of these
    /// comma-separated patterns (`*` is any run of characters), e.g.
    /// `blk.*.attn_*,output.weight`. Saves their gradients and optimizer
    /// state; the frozen weights still take their F32 memory.
    #[arg(long, value_delimiter = ',')]
    pub train_tensors: Vec<String>,

    /// Use at most this many tokens of the data (0: all). For quick runs and
    /// benchmarks.
    #[arg(long, default_value_t = 0)]
    pub limit_tokens: usize,

    /// Layers on the GPU (-1: all). Training wants all of them; a partial
    /// offload is untested upstream.
    #[arg(long, default_value_t = -1, allow_hyphen_values = true)]
    pub gpu_layers: i32,

    /// The GPU to train on, when several are visible. Training uses one.
    #[arg(long, default_value_t = 0)]
    pub device: i32,

    /// CPU threads. Default: the physical cores.
    #[arg(long)]
    pub threads: Option<u32>,

    /// Also write the report as JSON to this file.
    #[arg(long)]
    pub report: Option<PathBuf>,

    /// Check the model and estimate the memory, then stop: nothing is loaded.
    #[arg(long)]
    pub dry_run: bool,

    /// Start even when the memory estimate exceeds what is free.
    #[arg(long)]
    pub force: bool,

    /// No progress bar (for logs that keep every carriage return).
    #[arg(long)]
    pub no_progress: bool,
}

/// One tensor of the GGUF, as stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorFact {
    pub name: String,
    pub type_name: String,
    pub bytes: u64,
    pub is_f32: bool,
}

/// What the model file says, read without loading it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelFacts {
    pub arch: String,
    pub context_length: Option<u32>,
    pub n_layer: u32,
    pub n_embd: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub key_length: u32,
    pub value_length: u32,
    pub tensors: Vec<TensorFact>,
}

impl ModelFacts {
    /// Every tensor's parameter count, for an all-F32 model.
    pub fn params(&self) -> u64 {
        self.tensors.iter().map(|t| t.bytes / 4).sum()
    }

    /// Bytes of all tensors.
    pub fn weight_bytes(&self) -> u64 {
        self.tensors.iter().map(|t| t.bytes).sum()
    }

    /// Parameters that `filter` would train (the token embeddings never do).
    pub fn trainable_params(&self, filter: &TensorFilter) -> u64 {
        self.tensors
            .iter()
            .filter(|t| !NEVER_TRAINED.contains(&t.name.as_str()) && filter.accepts(&t.name))
            .map(|t| t.bytes / 4)
            .sum()
    }

    /// Vocabulary size, from the token-embedding table.
    pub fn n_vocab(&self) -> u64 {
        self.tensors
            .iter()
            .find(|t| t.name == TOKEN_EMBD)
            .map_or(0, |t| t.bytes / 4 / u64::from(self.n_embd.max(1)))
    }

    /// The tensors that are not F32: the reason training cannot start.
    pub fn not_f32(&self) -> Vec<&TensorFact> {
        self.tensors.iter().filter(|t| !t.is_f32).collect()
    }
}

/// Read what training needs to know from the GGUF header.
pub fn inspect(path: &Path) -> Result<ModelFacts, String> {
    let gguf = GgufContext::from_file(path)
        .ok_or_else(|| format!("{} is not a readable GGUF file", path.display()))?;
    // ggml asserts on a key that is absent or of another type, so both are
    // checked before a value is read.
    let arch = {
        let i = gguf.find_key("general.architecture");
        if i >= 0 && gguf.kv_type(i) == llama_cpp_sys_2::GGUF_TYPE_STRING {
            gguf.val_str(i).map(str::to_string).unwrap_or_default()
        } else {
            String::new()
        }
    };
    let context_length = {
        let i = gguf.find_key(&format!("{arch}.context_length"));
        (i >= 0 && gguf.kv_type(i) == llama_cpp_sys_2::GGUF_TYPE_UINT32).then(|| gguf.val_u32(i))
    };
    let mut tensors = Vec::new();
    for i in 0..gguf.n_tensors() {
        let name = gguf.tensor_name(i).unwrap_or("").to_string();
        let ty = gguf
            .tensor_type(i)
            .unwrap_or(llama_cpp_sys_2::GGML_TYPE_COUNT);
        let type_name = unsafe {
            let p = llama_cpp_sys_2::ggml_type_name(ty);
            if p.is_null() {
                format!("type {ty}")
            } else {
                std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        };
        tensors.push(TensorFact {
            name,
            type_name,
            bytes: gguf.tensor_size(i).unwrap_or(0) as u64,
            is_f32: ty == llama_cpp_sys_2::GGML_TYPE_F32,
        });
    }
    let info = crate::fit::read_gguf_info(path).ok_or_else(|| {
        format!(
            "{}: the GGUF header does not describe the layers",
            path.display()
        )
    })?;
    let n_embd = info.n_embd.unwrap_or(0);
    let n_head = info.n_head.unwrap_or(1).max(1);
    let head_dim = n_embd / n_head;
    Ok(ModelFacts {
        arch,
        context_length,
        n_layer: info.n_layers,
        n_embd,
        n_head,
        n_head_kv: info.n_head_kv.unwrap_or(n_head),
        key_length: info.key_length.unwrap_or(head_dim),
        value_length: info.value_length.unwrap_or(head_dim),
        tensors,
    })
}

/// Memory a run needs, by component, in bytes. An estimate: llama.cpp
/// allocates the backward graph itself and reuses buffers where it can, so
/// the activation term is an upper bound in the spirit of Korthikanti et al.
/// (2022) — 34·s·h + 5·a·s² per layer in 16 bits, doubled for F32 — not a
/// measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct MemoryEstimate {
    pub weights: u64,
    pub gradients: u64,
    pub optimizer_state: u64,
    pub kv_cache: u64,
    pub activations: u64,
    pub logits: u64,
    pub total: u64,
}

pub fn estimate_memory(
    facts: &ModelFacts,
    trainable_params: u64,
    optimizer: Optimizer,
    n_ctx: u32,
) -> MemoryEstimate {
    let s = u64::from(n_ctx);
    let layers = u64::from(facts.n_layer);
    let h = u64::from(facts.n_embd);
    let heads = u64::from(facts.n_head);
    let kv_per_token =
        u64::from(facts.n_head_kv) * (u64::from(facts.key_length) + u64::from(facts.value_length));
    let weights = facts.weight_bytes();
    let gradients = trainable_params * 4;
    let optimizer_state = trainable_params * optimizer.state_bytes_per_param();
    let kv_cache = layers * s * kv_per_token * 4;
    let activations = layers * (68 * s * h + 10 * heads * s * s);
    // Logits, their softmax and its gradient.
    let logits = 3 * s * facts.n_vocab() * 4;
    MemoryEstimate {
        weights,
        gradients,
        optimizer_state,
        kv_cache,
        activations,
        logits,
        total: weights + gradients + optimizer_state + kv_cache + activations + logits,
    }
}

/// The documents in `path`: the `text` of each JSONL line, or the whole file.
pub fn read_documents(path: &Path) -> Result<Vec<String>, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let is_jsonl = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("jsonl"));
    let docs: Vec<String> = if is_jsonl {
        let mut docs = Vec::new();
        for (n, line) in raw.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let value: serde_json::Value = serde_json::from_str(line)
                .map_err(|e| format!("{}:{}: not JSON: {e}", path.display(), n + 1))?;
            let text = value
                .get("text")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| format!("{}:{}: no \"text\" string", path.display(), n + 1))?;
            if !text.trim().is_empty() {
                docs.push(text.to_string());
            }
        }
        docs
    } else if raw.trim().is_empty() {
        Vec::new()
    } else {
        vec![raw]
    };
    if docs.is_empty() {
        return Err(format!("{} holds no text to train on", path.display()));
    }
    Ok(docs)
}

/// Where the token stream splits: training before, validation after.
///
/// Validation takes the last `val_split` share, but at least one window's
/// worth, and training must keep at least one window too. Without
/// validation (`val_split <= 0`) everything trains.
pub fn split_point(n_tokens: usize, n_ctx: usize, val_split: f64) -> Result<usize, String> {
    let window = n_ctx + 1;
    if val_split <= 0.0 {
        return if n_tokens >= window {
            Ok(n_tokens)
        } else {
            Err(format!(
                "{n_tokens} tokens, and one window of {n_ctx} needs {window}"
            ))
        };
    }
    if val_split.is_nan() || val_split >= 1.0 {
        return Err(format!("--val-split must be below 1, got {val_split}"));
    }
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let val = ((n_tokens as f64 * val_split).ceil() as usize).max(window);
    if n_tokens < val + window {
        return Err(format!(
            "{n_tokens} tokens: with validation, training needs at least {} \
             (one window of {n_ctx} for each side); shorten --ctx or add data",
            2 * window
        ));
    }
    Ok(n_tokens - val)
}

/// Bytes the next run can use: free VRAM for a GPU run, available RAM for a
/// CPU one; `None` when the platform does not say.
fn available_memory(on_gpu: bool) -> Option<u64> {
    if on_gpu {
        return crate::fit::vram_bytes().map(|(free, _)| free);
    }
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    meminfo.lines().find_map(|l| {
        let rest = l.strip_prefix("MemAvailable:")?;
        rest.trim()
            .trim_end_matches("kB")
            .trim()
            .parse::<u64>()
            .ok()
            .map(|kb| kb * 1024)
    })
}

fn gib(bytes: u64) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    {
        bytes as f64 / f64::from(1u32 << 30)
    }
}

/// Bytes the way a person reads them: 7.4 MiB, 63.9 GiB.
fn size(bytes: u64) -> String {
    #[allow(clippy::cast_precision_loss)]
    let b = bytes as f64;
    if b >= f64::from(1u32 << 30) {
        format!("{:.1} GiB", b / f64::from(1u32 << 30))
    } else if b >= f64::from(1u32 << 20) {
        format!("{:.1} MiB", b / f64::from(1u32 << 20))
    } else {
        format!("{:.1} KiB", b / 1024.0)
    }
}

/// One pass, as reported.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct PassReport {
    pub tokens: i64,
    pub loss: f64,
    pub loss_unc: f64,
    pub perplexity: f64,
    pub accuracy: f64,
    pub accuracy_unc: f64,
    pub seconds: f64,
}

impl From<Pass> for PassReport {
    fn from(p: Pass) -> Self {
        Self {
            tokens: p.tokens,
            loss: p.loss,
            loss_unc: p.loss_unc,
            perplexity: p.perplexity(),
            accuracy: p.accuracy,
            accuracy_unc: p.accuracy_unc,
            seconds: p.seconds,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct EpochReport {
    pub epoch: u32,
    pub lr: f32,
    pub train: PassReport,
    /// Training tokens per second.
    pub train_tok_s: f64,
    pub validation: Option<PassReport>,
}

/// Everything a run did, for the `FINETUNE_RESULT` line and `--report`.
#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub schema: &'static str,
    pub engine: &'static str,
    pub backend: &'static str,
    pub model: String,
    pub data: String,
    pub output: Option<String>,
    pub arch: String,
    pub params: u64,
    pub trainable_params: u64,
    pub trainable_tensors: Option<u32>,
    pub train_tensors: Vec<String>,
    pub optimizer: String,
    pub lr: f32,
    pub lr_min: f32,
    pub decay_epochs: f32,
    pub weight_decay: f32,
    pub epochs: u32,
    pub n_ctx: u32,
    pub stride: u32,
    pub val_split: f64,
    pub gpu_layers: i32,
    pub threads: u32,
    pub tokens: usize,
    pub tokens_train: usize,
    pub tokens_validation: usize,
    pub windows_train: usize,
    pub windows_validation: usize,
    pub memory_estimate: MemoryEstimate,
    pub memory_available: Option<u64>,
    pub vram_used_after: Option<u64>,
    pub load_seconds: f64,
    pub baseline: Option<PassReport>,
    pub per_epoch: Vec<EpochReport>,
    pub total_seconds: f64,
    pub output_bytes: Option<u64>,
    pub context_length_restored: Option<u32>,
    pub dry_run: bool,
}

fn backend_name() -> &'static str {
    if cfg!(all(feature = "cuda", feature = "vulkan")) {
        "cuda+vulkan"
    } else if cfg!(all(feature = "rocm", feature = "vulkan")) {
        "rocm+vulkan"
    } else if cfg!(feature = "cuda") {
        "cuda"
    } else if cfg!(feature = "rocm") {
        "rocm"
    } else if cfg!(feature = "vulkan") {
        "vulkan"
    } else if cfg!(feature = "metal") {
        "metal"
    } else {
        "cpu"
    }
}

fn default_output(model: &Path) -> PathBuf {
    let stem = model
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    PathBuf::from(format!("{stem}-finetuned.gguf"))
}

/// ` ± x`, or nothing when there is no uncertainty to give: ggml has none
/// for a pass over a single window.
fn plus_minus(x: f64, decimals: usize, unit: &str) -> String {
    if x.is_finite() {
        format!(" ± {x:.decimals$}{unit}")
    } else {
        String::new()
    }
}

/// Whether `a` and `b` name one existing file (through links, `..`, or
/// relative paths).
fn is_same_file(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

fn print_pass(label: &str, p: &Pass) {
    println!(
        "  {label:<12} loss {:.4}{}  ppl {:>9.2}  acc {:.2}%{}  ({} tokens, {:.1} s, {:.0} tok/s)",
        p.loss,
        plus_minus(p.loss_unc, 4, ""),
        p.perplexity(),
        100.0 * p.accuracy,
        plus_minus(100.0 * p.accuracy_unc, 2, "%"),
        p.tokens,
        p.seconds,
        p.tokens_per_second()
    );
}

/// Run `eullm finetune` on the model at `model_path`.
pub fn run(opts: &FinetuneOpts, model_path: &Path) -> Result<Report, String> {
    run_on(opts, model_path, None)
}

/// [`run`], on `backend` when given — the tests share one, because llama.cpp
/// allows a single live backend per process — or on a new one.
fn run_on(
    opts: &FinetuneOpts,
    model_path: &Path,
    backend: Option<std::sync::Arc<llama_cpp_2::llama_backend::LlamaBackend>>,
) -> Result<Report, String> {
    let started = Instant::now();
    if opts.ctx == 0 || !opts.ctx.is_multiple_of(CTX_MULTIPLE) {
        let up = opts.ctx.div_ceil(CTX_MULTIPLE).max(1) * CTX_MULTIPLE;
        return Err(format!(
            "--ctx {} is not a multiple of {CTX_MULTIPLE}: llama.cpp would round the context \
             up to {up} and the windows would no longer match it. Use --ctx {up}.",
            opts.ctx
        ));
    }
    if opts.epochs == 0 && !opts.dry_run {
        return Err("--epochs 0 trains nothing".into());
    }
    let output = opts
        .output
        .clone()
        .unwrap_or_else(|| default_output(model_path));
    if is_same_file(&output, model_path) {
        return Err(format!(
            "--output {} is the model being trained: write the result to another file",
            output.display()
        ));
    }
    let stride = opts.stride.unwrap_or(opts.ctx / 2).max(1);
    let filter = TensorFilter::new(opts.train_tensors.clone());
    // Before anything is loaded: ggml checks these with assertions, which
    // would abort the run at its first optimizer step.
    let schedule = LrSchedule::new(
        opts.lr,
        opts.lr_min,
        opts.decay_epochs,
        opts.wd,
        opts.epochs,
    );
    schedule
        .check()
        .map_err(|e| format!("{e} (--lr, --lr-min, --wd)"))?;

    // ── The model file, before anything is loaded ────────────────────────
    let facts = inspect(model_path)?;
    let not_f32 = facts.not_f32();
    if !not_f32.is_empty() {
        let shown: Vec<String> = not_f32
            .iter()
            .take(4)
            .map(|t| format!("{} ({})", t.name, t.type_name))
            .collect();
        return Err(format!(
            "{} is not an F32 model: {} of {} tensors are not F32 — {}{}. llama.cpp's trainer \
             computes weight gradients only in F32; convert the original weights with \
             `convert_hf_to_gguf.py --outtype f32`.",
            model_path.display(),
            not_f32.len(),
            facts.tensors.len(),
            shown.join(", "),
            if not_f32.len() > 4 { ", …" } else { "" }
        ));
    }
    let planned_trainable = facts.trainable_params(&filter);
    if planned_trainable == 0 {
        return Err(format!(
            "--train-tensors {:?} matches no tensor of {}",
            opts.train_tensors,
            model_path.display()
        ));
    }
    let gpu_layers = crate::inference::check_gpu_support(opts.gpu_layers);
    let on_gpu = gpu_layers != 0;
    let estimate = estimate_memory(&facts, planned_trainable, opts.optimizer, opts.ctx);
    let available = available_memory(on_gpu);
    let threads = opts
        .threads
        .unwrap_or_else(crate::inference::default_thread_count);

    println!("eullm finetune — {}", model_path.display());
    println!(
        "  model        {} · {} layers · {} parameters, {} to train ({})",
        facts.arch,
        facts.n_layer,
        count(facts.params()),
        count(planned_trainable),
        if filter.patterns().is_empty() {
            "all but the token embeddings".to_string()
        } else {
            filter.patterns().join(",")
        }
    );
    println!(
        "  training     {} · lr {} {} · wd {} · {} epochs · window {} · stride {} · {}",
        opts.optimizer,
        opts.lr,
        if opts.lr_min > 0.0 {
            format!("→ {}", opts.lr_min)
        } else {
            "constant".into()
        },
        opts.wd,
        opts.epochs,
        opts.ctx,
        stride,
        if on_gpu {
            format!("{} on device {}", backend_name(), opts.device)
        } else {
            format!("CPU, {threads} threads")
        }
    );
    println!(
        "  memory       ~{} estimated (weights {}, gradients {}, optimizer {}, activations {}, \
         logits {}, KV {}){}",
        size(estimate.total),
        size(estimate.weights),
        size(estimate.gradients),
        size(estimate.optimizer_state),
        size(estimate.activations),
        size(estimate.logits),
        size(estimate.kv_cache),
        available.map_or(String::new(), |a| format!(" · {} free", size(a)))
    );

    let mut report = Report {
        schema: SCHEMA,
        engine: crate::VERSION_STRING,
        backend: backend_name(),
        model: model_path.display().to_string(),
        data: opts.data.display().to_string(),
        output: None,
        arch: facts.arch.clone(),
        params: facts.params(),
        trainable_params: planned_trainable,
        trainable_tensors: None,
        train_tensors: opts.train_tensors.clone(),
        optimizer: opts.optimizer.to_string(),
        lr: opts.lr,
        lr_min: opts.lr_min,
        decay_epochs: opts.decay_epochs,
        weight_decay: opts.wd,
        epochs: opts.epochs,
        n_ctx: opts.ctx,
        stride,
        val_split: opts.val_split,
        gpu_layers,
        threads,
        tokens: 0,
        tokens_train: 0,
        tokens_validation: 0,
        windows_train: 0,
        windows_validation: 0,
        memory_estimate: estimate,
        memory_available: available,
        vram_used_after: None,
        load_seconds: 0.0,
        baseline: None,
        per_epoch: Vec::new(),
        total_seconds: 0.0,
        output_bytes: None,
        context_length_restored: None,
        dry_run: opts.dry_run,
    };
    if let Some(free) = available
        && estimate.total > free
        && !opts.force
        && !opts.dry_run
    {
        return Err(format!(
            "the run is estimated at {:.1} GiB and {:.1} GiB is free. Shorten --ctx, train fewer \
             tensors (--train-tensors), use --optimizer sgd, or pass --force to try anyway.",
            gib(estimate.total),
            gib(free)
        ));
    }
    if opts.dry_run {
        report.total_seconds = started.elapsed().as_secs_f64();
        return Ok(report);
    }
    if on_gpu && gpu_layers > 0 && (gpu_layers as u32) < facts.n_layer {
        eprintln!(
            "warning: --gpu-layers {gpu_layers} of {} puts part of the model on the CPU; \
             llama.cpp's trainer is only tested fully offloaded",
            facts.n_layer
        );
    }

    // ── Load: writable weights, one device ───────────────────────────────
    let backend = match backend {
        Some(b) => b,
        None => crate::inference::init_shared_backend().map_err(|e| e.to_string())?,
    };
    let load_started = Instant::now();
    let model_params = LlamaModelParams::default()
        .with_use_mmap(false)
        .with_n_gpu_layers(if gpu_layers < 0 {
            1000
        } else {
            gpu_layers as u32
        })
        .with_split_mode(LlamaSplitMode::None)
        .with_main_gpu(opts.device);
    let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
        .map_err(|e| format!("failed to load {}: {e}", model_path.display()))?;
    report.load_seconds = load_started.elapsed().as_secs_f64();

    // ── Data ─────────────────────────────────────────────────────────────
    let docs = read_documents(&opts.data)?;
    let mut tokens: Vec<LlamaToken> = Vec::new();
    for doc in &docs {
        tokens.extend(
            model
                .str_to_token(doc, AddBos::Always)
                .map_err(|e| format!("tokenizing {}: {e}", opts.data.display()))?,
        );
        tokens.push(model.token_eos());
        if opts.limit_tokens > 0 && tokens.len() >= opts.limit_tokens {
            tokens.truncate(opts.limit_tokens);
            break;
        }
    }
    let n_ctx = opts.ctx as usize;
    let split = split_point(tokens.len(), n_ctx, opts.val_split)?;
    let train_data = OptDataset::from_tokens(&tokens[..split], n_ctx, stride as usize)
        .map_err(|e| e.to_string())?;
    let val_data = if split < tokens.len() {
        Some(
            OptDataset::from_tokens(&tokens[split..], n_ctx, stride as usize)
                .map_err(|e| e.to_string())?,
        )
    } else {
        None
    };
    report.tokens = tokens.len();
    report.tokens_train = split;
    report.tokens_validation = tokens.len() - split;
    report.windows_train = train_data.len();
    report.windows_validation = val_data.as_ref().map_or(0, OptDataset::len);
    println!(
        "  data         {} document{} · {} tokens: {} windows to train, {} to validate",
        docs.len(),
        if docs.len() == 1 { "" } else { "s" },
        tokens.len(),
        report.windows_train,
        report.windows_validation
    );

    // ── Train ────────────────────────────────────────────────────────────
    let n = NonZeroU32::new(opts.ctx).expect("checked above");
    #[allow(clippy::cast_possible_wrap)]
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(n))
        .with_n_batch(opts.ctx)
        .with_n_ubatch(opts.ctx)
        .with_type_k(KvCacheType::F32)
        .with_type_v(KvCacheType::F32)
        .with_flash_attention_policy(llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_DISABLED)
        .with_n_threads(threads as i32)
        .with_n_threads_batch(threads as i32);
    let mut ctx = model
        .new_context(&backend, ctx_params)
        .map_err(|e| format!("cannot create the training context: {e}"))?;
    let progress = !opts.no_progress;
    {
        let mut trainer =
            Trainer::new(&mut ctx, opts.optimizer, schedule, filter).map_err(|e| e.to_string())?;
        report.trainable_params = trainer.trainable_params();
        report.trainable_tensors = Some(trainer.trainable_tensors());
        println!(
            "  trainer      {} tensors, {} parameters",
            trainer.trainable_tensors(),
            count(trainer.trainable_params())
        );
        if let Some(val) = &val_data {
            let p = trainer.evaluate(val, progress).map_err(|e| e.to_string())?;
            print_pass("before", &p);
            report.baseline = Some(p.into());
        }
        for epoch in 0..opts.epochs {
            let lr = trainer.lr(epoch);
            println!("epoch {}/{} — lr {lr:.3e}", epoch + 1, opts.epochs);
            let t = trainer
                .train(epoch, &train_data, progress)
                .map_err(|e| e.to_string())?;
            print_pass("train", &t);
            let tok_s = t.tokens_per_second();
            let v = match &val_data {
                Some(val) => {
                    let p = trainer.evaluate(val, progress).map_err(|e| e.to_string())?;
                    print_pass("validation", &p);
                    Some(PassReport::from(p))
                }
                None => None,
            };
            report.per_epoch.push(EpochReport {
                epoch,
                lr,
                train: t.into(),
                train_tok_s: tok_s,
                validation: v,
            });
        }
    }
    if on_gpu {
        report.vram_used_after =
            crate::fit::vram_bytes().map(|(free, total)| total.saturating_sub(free));
    }
    drop(ctx);

    // ── Save, and give the file back its context length ───────────────────
    model
        .save_to_file(&output)
        .map_err(|e| format!("writing {}: {e}", output.display()))?;
    if let Some(original) = facts.context_length {
        let key = format!("{}.context_length", facts.arch);
        match crate::gguf_patch::set_u32_in_place(&output, &key, original) {
            Ok(Some(_)) => report.context_length_restored = Some(original),
            Ok(None) => {}
            Err(e) => eprintln!(
                "warning: {} keeps the training context length as its {key}: {e}",
                output.display()
            ),
        }
    }
    report.output_bytes = std::fs::metadata(&output).ok().map(|m| m.len());
    report.output = Some(output.display().to_string());
    report.total_seconds = started.elapsed().as_secs_f64();
    println!(
        "  wrote        {} ({}, F32 — quantize it for deployment as any other model)",
        output.display(),
        size(report.output_bytes.unwrap_or(0))
    );
    Ok(report)
}

/// A parameter count the way people say it: 292.8 K, 1.72 B.
fn count(n: u64) -> String {
    #[allow(clippy::cast_precision_loss)]
    let x = n as f64;
    if x >= 1e9 {
        format!("{:.2} B", x / 1e9)
    } else if x >= 1e6 {
        format!("{:.1} M", x / 1e6)
    } else if x >= 1e3 {
        format!("{:.1} K", x / 1e3)
    } else {
        n.to_string()
    }
}

/// `eullm finetune`: run, print the report line, write `--report`.
pub fn cmd(opts: &FinetuneOpts, model_path: &Path) -> Result<(), String> {
    let report = run(opts, model_path)?;
    let json = serde_json::to_string(&report).map_err(|e| e.to_string())?;
    if let Some(path) = &opts.report {
        let pretty = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
        std::fs::write(path, pretty).map_err(|e| format!("writing {}: {e}", path.display()))?;
    }
    println!("FINETUNE_RESULT {json}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts(tensors: &[(&str, u64)]) -> ModelFacts {
        ModelFacts {
            arch: "llama".into(),
            context_length: Some(4096),
            n_layer: 2,
            n_embd: 64,
            n_head: 4,
            n_head_kv: 2,
            key_length: 16,
            value_length: 16,
            tensors: tensors
                .iter()
                .map(|(n, b)| TensorFact {
                    name: (*n).into(),
                    type_name: "f32".into(),
                    bytes: *b,
                    is_f32: true,
                })
                .collect(),
        }
    }

    #[test]
    fn token_embeddings_never_count_as_trainable() {
        let f = facts(&[
            ("token_embd.weight", 4 * 64 * 100),
            ("rope_freqs.weight", 4 * 8),
            ("blk.0.attn_q.weight", 4 * 64 * 64),
            ("blk.0.ffn_up.weight", 4 * 64 * 128),
            ("output.weight", 4 * 64 * 100),
        ]);
        assert_eq!(f.params(), 64 * 100 * 2 + 8 + 64 * 64 + 64 * 128);
        assert_eq!(
            f.trainable_params(&TensorFilter::all()),
            64 * 64 + 64 * 128 + 64 * 100
        );
        let attn = TensorFilter::new(vec!["blk.*.attn_*".into()]);
        assert_eq!(f.trainable_params(&attn), 64 * 64);
        assert_eq!(f.n_vocab(), 100);
    }

    #[test]
    fn the_output_may_not_be_the_model_trained() {
        let dir = std::env::temp_dir().join(format!("eullm-ft-same-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("m.gguf");
        std::fs::write(&model, b"GGUF").unwrap();
        assert!(is_same_file(&dir.join(".").join("m.gguf"), &model));
        assert!(!is_same_file(&dir.join("out.gguf"), &model));
        #[derive(clap::Parser)]
        struct Cli {
            #[command(flatten)]
            opts: FinetuneOpts,
        }
        let Cli { opts } = clap::Parser::parse_from([
            "finetune",
            model.to_str().unwrap(),
            "--data",
            "d.txt",
            "--output",
            dir.join("..")
                .join(dir.file_name().unwrap())
                .join("m.gguf")
                .to_str()
                .unwrap(),
        ]);
        let err = run_on(&opts, &model, None).unwrap_err();
        assert!(err.contains("is the model being trained"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_single_window_has_no_uncertainty_to_print() {
        assert_eq!(plus_minus(0.0125, 4, ""), " ± 0.0125");
        assert_eq!(plus_minus(3.0, 2, "%"), " ± 3.00%");
        assert_eq!(plus_minus(f64::NAN, 4, ""), "");
    }

    #[test]
    fn memory_estimate_adds_up_and_sgd_carries_no_state() {
        let f = facts(&[
            ("token_embd.weight", 4 * 64 * 100),
            ("blk.0.attn_q.weight", 4 * 64 * 64),
        ]);
        let adam = estimate_memory(&f, 4096, Optimizer::AdamW, 128);
        let sgd = estimate_memory(&f, 4096, Optimizer::Sgd, 128);
        assert_eq!(adam.gradients, 4096 * 4);
        assert_eq!(adam.optimizer_state, 4096 * 8);
        assert_eq!(sgd.optimizer_state, 0);
        assert_eq!(adam.kv_cache, 2 * 128 * (2 * 32) * 4);
        assert_eq!(adam.logits, 3 * 128 * 100 * 4);
        assert_eq!(
            adam.total,
            adam.weights
                + adam.gradients
                + adam.optimizer_state
                + adam.kv_cache
                + adam.activations
                + adam.logits
        );
        // Attention activations grow with the square of the window.
        let long = estimate_memory(&f, 4096, Optimizer::AdamW, 256);
        assert!(long.activations > 2 * adam.activations);
    }

    #[test]
    fn split_keeps_a_window_on_each_side() {
        assert_eq!(split_point(1000, 100, 0.05), Ok(899)); // 5% is 50 < 101
        assert_eq!(split_point(10_000, 100, 0.2), Ok(8000));
        assert_eq!(split_point(150, 100, 0.0), Ok(150));
        assert!(split_point(150, 100, 0.05).is_err()); // 101 + 101 > 150
        assert!(split_point(50, 100, 0.0).is_err());
        assert!(split_point(10_000, 100, 1.0).is_err());
    }

    #[test]
    fn documents_from_text_and_jsonl() {
        let dir = std::env::temp_dir().join(format!("eullm-ft-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let txt = dir.join("a.txt");
        std::fs::write(&txt, "uno due tre").unwrap();
        assert_eq!(
            read_documents(&txt).unwrap(),
            vec!["uno due tre".to_string()]
        );

        let jsonl = dir.join("b.jsonl");
        std::fs::write(
            &jsonl,
            "{\"text\": \"primo\"}\n\n{\"text\": \"secondo\", \"id\": 2}\n",
        )
        .unwrap();
        assert_eq!(read_documents(&jsonl).unwrap(), vec!["primo", "secondo"]);

        std::fs::write(&jsonl, "{\"text\": \"ok\"}\n{\"body\": \"no\"}\n").unwrap();
        let err = read_documents(&jsonl).unwrap_err();
        assert!(err.contains(":2:") && err.contains("text"), "{err}");

        let empty = dir.join("c.txt");
        std::fs::write(&empty, "  \n").unwrap();
        assert!(read_documents(&empty).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn sizes_read_like_people_say_them() {
        assert_eq!(size(7_797_760), "7.4 MiB");
        assert_eq!(size(68_702_699_520), "64.0 GiB");
        assert_eq!(size(2048), "2.0 KiB");
    }

    #[test]
    fn counts_read_like_people_say_them() {
        assert_eq!(count(292_800), "292.8 K");
        assert_eq!(count(1_720_000_000), "1.72 B");
        assert_eq!(count(7_600_000), "7.6 M");
        assert_eq!(count(12), "12");
    }

    /// An F32 GGUF small enough to train on a CPU in a test:
    /// `stories260K-f32.gguf` from `ggml-org/test-model-stories260K`.
    ///
    /// ```text
    /// EULLM_FINETUNE_TEST_MODEL=/path/to/stories260K-f32.gguf \
    ///     cargo test --release -p eullm-engine -- --ignored real_model_finetune
    /// ```
    #[test]
    #[ignore = "needs EULLM_FINETUNE_TEST_MODEL, an F32 GGUF"]
    fn real_model_finetune_learns_saves_and_keeps_its_context_length() {
        let model: PathBuf = std::env::var("EULLM_FINETUNE_TEST_MODEL")
            .expect("set EULLM_FINETUNE_TEST_MODEL to an F32 GGUF")
            .into();
        let dir = std::env::temp_dir().join(format!("eullm-ft-real-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        // One template, many fillings: learnable in a few epochs, which a
        // real corpus is not.
        let names = ["Lily", "Tom", "Sam", "Mia"];
        let things = ["a red ball", "a big box", "a small cat", "a shiny stone"];
        let text: String = (0..400)
            .map(|i| {
                let (n, t) = (names[i % 4], things[(i / 4) % 4]);
                format!("Once upon a time, {n} found {t}. {n} took {t} home. The end.\n")
            })
            .collect();
        let data = dir.join("stories.txt");
        std::fs::write(&data, text).unwrap();
        let output = dir.join("out.gguf");
        let opts = FinetuneOpts {
            model: model.display().to_string(),
            data,
            output: Some(output.clone()),
            epochs: 2,
            ctx: 256,
            stride: None,
            lr: 1e-4,
            lr_min: -1.0,
            decay_epochs: -1.0,
            wd: 0.0,
            optimizer: Optimizer::AdamW,
            val_split: 0.05,
            train_tensors: Vec::new(),
            limit_tokens: 0,
            gpu_layers: 0,
            device: 0,
            threads: Some(2),
            report: None,
            dry_run: false,
            force: false,
            no_progress: true,
        };
        let report = run_on(&opts, &model, Some(crate::inference::test_backend())).unwrap();

        let before = report.baseline.expect("a validation baseline").loss;
        let after = report.per_epoch.last().unwrap().validation.unwrap().loss;
        assert!(after < before, "validation loss {before} → {after}");
        assert!(report.trainable_params > 0 && report.trainable_params < report.params);
        // What the trainer was given is what the GGUF says it would be.
        let facts = inspect(&model).unwrap();
        assert_eq!(
            report.trainable_params,
            facts.trainable_params(&TensorFilter::all())
        );
        assert!(report.per_epoch.iter().all(|e| e.train_tok_s > 0.0));

        // The file trained at 256 tokens declares the original context again.
        let original = inspect(&model).unwrap().context_length;
        let saved = inspect(&output).unwrap();
        assert_eq!(saved.context_length, original);
        assert_eq!(report.context_length_restored, original);
        assert!(saved.not_f32().is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn default_output_sits_beside_the_name() {
        assert_eq!(
            default_output(Path::new("/m/stories260K.gguf")),
            PathBuf::from("stories260K-finetuned.gguf")
        );
    }
}

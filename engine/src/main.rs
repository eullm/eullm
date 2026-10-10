mod api;
mod audit;
mod banner;
mod chat_template;
mod finetune;
mod model_tokens;
mod fit;
mod gguf_patch;
mod inference;
mod lineedit;
mod llama_archs;
mod models;
mod picker;
mod readahead;
mod registry;
mod tools;
mod ui;
mod update;

use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use models::{ModelStore, catalog};
use crate::models::pull::hf_ref_to_model_id;

use crate::inference::{BatchScheduler, InferenceConfig, InferenceEngine, SchedulerConfig};
use crate::lineedit::{Line, LineReader};

// Cross-platform default pidfile location. On Unix `/tmp` always exists and
// is the canonical place; on Windows there is no `/tmp`, so we fall back to
// the current working directory (the daemon writer will create the file
// there). The path is materialised at clap-parse time, so it must be a
// compile-time `&'static str`.
#[cfg(unix)]
const DEFAULT_PIDFILE: &str = "/tmp/eullm.pid";
#[cfg(not(unix))]
const DEFAULT_PIDFILE: &str = "eullm.pid";

// `eullm -V` output reflects the build variant so users immediately know
// which backend they are running, e.g.
//   eullm 0.5.8 (CUDA)
//   eullm 0.5.8 (Metal)
//   eullm 0.5.8 (CPU)
// Only one branch matches per build because feature flags are mutually
// exclusive (set by the release matrix).
// The git commit this binary was built from (see build.rs) — the crate
// version alone doesn't say which commit on an unreleased branch a build
// came from, and "did you actually rebuild with the fix" was costing real
// back-and-forth during hands-on debugging on real hardware.
#[cfg(feature = "cuda")]
const VERSION_STRING: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    " (CUDA) [",
    env!("EULLM_GIT_HASH"),
    "]"
);
#[cfg(feature = "metal")]
const VERSION_STRING: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    " (Metal) [",
    env!("EULLM_GIT_HASH"),
    "]"
);
#[cfg(feature = "rocm")]
const VERSION_STRING: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    " (ROCm) [",
    env!("EULLM_GIT_HASH"),
    "]"
);
#[cfg(feature = "vulkan")]
const VERSION_STRING: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    " (Vulkan) [",
    env!("EULLM_GIT_HASH"),
    "]"
);
#[cfg(not(any(
    feature = "cuda",
    feature = "metal",
    feature = "rocm",
    feature = "vulkan"
)))]
const VERSION_STRING: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    " (CPU) [",
    env!("EULLM_GIT_HASH"),
    "]"
);

/// Build the `run` command implied by a choice made in the interactive picker.
///
/// This was three hand-written `Commands::Run { .. }` literals, one per picker
/// outcome, each restating all twenty-seven defaults. They were a fourth copy
/// of the same list: a default changed in the `#[arg]` attribute stayed wrong
/// here, silently, because nothing compares the two. Asking clap to parse
/// `eullm run <model>` returns the same value with the defaults clap actually
/// documents, so there is one source for them again.
///
/// `--fit` is the single deliberate difference, and since 0.6.80 it means
/// less than it used to: sizing is on by default everywhere, so what the
/// picker adds is only the *explicit* form — the one that asks for
/// confirmation before a partial split. The picker only opens on an
/// interactive terminal, where there is someone to answer.
fn picker_run(model: &str) -> Commands {
    // `--` first: a model name or path beginning with a dash would otherwise
    // be read as a flag.
    let mut cmd = Cli::parse_from(["eullm", "run", "--", model])
        .command
        .expect("`eullm run <model>` always parses to a subcommand");
    if let Commands::Run { opts, .. } = &mut cmd {
        opts.fit = true;
    }
    cmd
}

#[derive(Parser)]
#[command(name = "eullm")]
#[command(about = "eullm — sovereign LLM runtime for Europe")]
#[command(version = VERSION_STRING)]
struct Cli {
    /// Subcommand to run. When omitted in an interactive terminal, an
    /// interactive picker opens so the user can choose a local model, a
    /// catalog model, or paste a custom path/URL.
    #[command(subcommand)]
    command: Option<Commands>,
}

/// Flags shared by `run` and `serve`.
///
/// `CLAUDE.md` makes it mandatory that a flag added to `Run` is added to
/// `Serve` in the same change, because the two field lists were maintained by
/// hand and had already drifted in production: `cache_type_k`, `cache_type_v`
/// and `gpu_layers` existed only on `Run`, so every model `serve` loaded was
/// forced into fixed defaults with no way to override them. That rule is a
/// human guard over a structural problem — it holds exactly as long as the
/// next person remembers it.
///
/// Flattening one struct into both subcommands makes the divergence impossible
/// rather than forbidden: a flag added here exists on both, with the same
/// default and the same help text, and there is no second place to forget.
///
/// Nothing stays out any more. The two flags that once did are both here:
/// `batch_size`, whose two defaults (1 for `run`, 8 for `serve`) became one
/// when eight slots were found to give each request an eighth of
/// `--ctx-size`; and `--fit`/`--fit-strict`, since every load `serve` makes
/// is sized too (`api::AppState::load_generation_model`). A flag that
/// reaches a model `serve` loads must still be carried there through
/// `api::ServeConfig` by hand — see `engine/CLAUDE.md`.
#[derive(clap::Args, Debug, PartialEq)]
struct RuntimeOpts {
    /// Port for the API server
    #[arg(short, long, default_value_t = 11434)]
    port: u16,

    /// Replace existing service on the port
    #[arg(long)]
    replace: bool,

    /// Maximum GPU layers to offload (-1 = all, 0 = CPU only). This is an
    /// upper bound, not a fixed count: automatic sizing still runs and may
    /// offload fewer if that is all that fits, so a number chosen for one
    /// model cannot run the next one out of VRAM. Use --no-fit to force a
    /// count past the estimate.
    #[arg(long, allow_hyphen_values = true)]
    gpu_layers: Option<i32>,

    /// Auto-fit GPU layers to available VRAM. Probes free VRAM and the
    /// model's layer count, then offloads as many layers as fit.
    ///
    /// **On by default** since 0.6.80: without sizing, a model larger than
    /// the free VRAM dies with an out-of-memory error at load, while with
    /// it the worst case is a slower partial split — a default that picks
    /// the crash is the wrong default. Passing --fit explicitly changes
    /// one thing: on an interactive terminal a partial split asks for
    /// confirmation, because you asked to be involved in the decision.
    /// Automatic sizing never asks; it applies the split and logs it.
    ///
    /// Turned off by --no-fit. A --gpu-layers of your own is a ceiling it
    /// keeps to, not an off switch. When VRAM cannot be read (no GPU)
    /// automatic sizing stays silent and --gpu-layers is used as-is.
    #[arg(long)]
    fit: bool,

    /// Never size the GPU offload automatically; use --gpu-layers as given
    /// (the pre-0.6.80 behaviour). Wins over --fit.
    #[arg(long, conflicts_with = "fit")]
    no_fit: bool,

    /// With --fit, refuse to load (instead of offloading a partial split
    /// or falling back) when the model does not fully fit on the GPU. On
    /// `serve`, a refused load surfaces as an error to the API caller.
    #[arg(long)]
    fit_strict: bool,

    /// For MoE models (e.g. Qwen3-30B-A3B): keep expert tensors
    /// (`*.ffn_(up|down|gate)_exps`) on CPU RAM while attention,
    /// embeddings, and the KV cache stay on GPU. Only a few experts fire
    /// per token, so this trades a small compute cost for VRAM headroom
    /// far beyond what --gpu-layers' whole-layer offload can reach — a
    /// 20+ GB MoE model can run mostly-GPU-speed on a 12 GB card. No
    /// effect on dense (non-MoE) models. Combines with --gpu-layers/--fit
    /// (which still control the non-expert tensors) and --ctx-size.
    #[arg(long)]
    cpu_moe: bool,

    /// For MoE models: keep expert tensors on CPU RAM for only the
    /// first N transformer layers, leaving the rest on GPU. Finer
    /// grained than --cpu-moe — use this when the blanket flag leaves
    /// VRAM idle (all experts to CPU) but the model doesn't fully fit
    /// with --gpu-layers alone. Mutually exclusive with --cpu-moe.
    #[arg(long, default_value_t = 0)]
    n_cpu_moe: u32,

    /// Recurrent-state rollback window for hybrid/recurrent
    /// architectures (Mamba/Gated-DeltaNet-style SSM layers, e.g.
    /// Qwen3.5/3.6's hybrid attention+SSM design). 0 (default, strongly
    /// recommended) leaves it off. NOT a conversation/KV-cache-reuse
    /// knob: upstream llama.cpp reserves n_rs_seq for bounded
    /// speculative-decoding draft-token rollback and hard-zeroes it
    /// outside that path (`cparams_dft.n_rs_seq = 0`); it is not what
    /// the official server uses for cross-turn prompt caching on these
    /// architectures (that's the separate, bounded `--ctx-checkpoints`
    /// snapshot mechanism). Every recurrent-state tensor scales by
    /// `(1 + N)`, so nonzero values can multiply resident memory by
    /// tens of GB and are not yet validated upstream past a small
    /// synthetic test model. On hybrid/recurrent architectures without
    /// this set, expect KV-cache prefix reuse to fall back to a full
    /// re-prefill on every turn — this is a known, still-open upstream
    /// limitation (llama.cpp's own server logs the identical
    /// "forcing full prompt re-processing due to lack of cache data
    /// (likely due to SWA or hybrid/recurrent memory)" fallback), not
    /// an eullm-specific gap.
    #[arg(long, default_value_t = 0)]
    rs_seq: u32,

    /// Speculative decoding with the model's own multi-token prediction
    /// (MTP) head: after each token the model writes, the head drafts up to
    /// N more, and one decode checks them all. Every draft the model agrees
    /// with is kept, so the answer is the one it would have written anyway,
    /// in fewer steps. 0 (default) turns it off; 2 measured best on a GPU
    /// (see docs/engine-guide.md). Needs a
    /// model whose GGUF carries its MTP layers (unsloth's `*-MTP-GGUF`
    /// Qwen3.5/3.6, for instance) and one request at a time (`--batch-size
    /// 1`, the default): otherwise the load says why and runs without it.
    /// On a hybrid model (Qwen3.5/3.6) it raises the recurrent-state
    /// rollback window to N, the drafts it may have to take back.
    #[arg(
        long,
        value_name = "N",
        default_value_t = 0,
        value_parser = clap::value_parser!(u32).range(0..=8)
    )]
    mtp: u32,

    /// With `--mtp`: stop drafting once the MTP head's own probability for
    /// its next draft falls below P (0 to 1). Drafts it is unsure of are
    /// mostly rejected, and each one costs a pass of the head and a position
    /// in the check, so a threshold lets the draft length follow the text:
    /// long where the text is predictable, none where it is not. 0 (default,
    /// llama.cpp's own) always drafts the full N.
    #[arg(long, value_name = "P", default_value_t = 0.0, value_parser = parse_probability)]
    mtp_p_min: f32,

    /// With `--mtp`: a GGUF holding the MTP head alone, for a model whose own
    /// GGUF has none (Qwen3.8-Flash-Next's, for instance: the head is a
    /// separate file, such as unsloth's `mtp-Qwen3.8-Flash-Next-Q8_0.gguf`).
    /// It is loaded onto the GPU whole, so it costs that much VRAM (3.85 GB
    /// for that one), which `--fit` keeps out of the layers and the expert
    /// cache. Without `--mtp` it is ignored.
    #[arg(long, value_name = "FILE")]
    mtp_model: Option<PathBuf>,

    /// For MoE models whose experts do not all fit in VRAM: keep every
    /// expert in RAM and give the VRAM they would have taken to a cache of
    /// the ones the model uses most. `auto` sizes it from the VRAM left once
    /// the rest of the model is placed (with --fit, on by default); a number
    /// asks for that many MiB. It speeds up writing, not the reading of a
    /// long prompt: on an RTX 5070 Ti, Qwen3.8-Flash-Next (IQ2_XS) wrote 49.4
    /// tokens/s with an 8,000 MiB cache against 22.4 without, and read a
    /// prompt 17% slower. One CUDA GPU only; elsewhere the load says why and
    /// runs without it. Experimental: the cache is llama.cpp PR #29887, which
    /// this build carries ahead of a llama.cpp release. Unless told
    /// otherwise, a load with a cache reads prompts 2048 tokens at a time
    /// (--n-ubatch) and reads the model into memory when the RAM can spare
    /// the experts (--no-mmap, --mmap): on that model, 55 tokens/s writing
    /// and 960 reading a prompt.
    #[arg(long, value_name = "auto|MIB", value_parser = fit::parse_moe_cache)]
    moe_cache: Option<fit::MoeCache>,

    /// Read the model into memory instead of mapping its file. Expert
    /// tensors kept in RAM (by --moe-cache, --cpu-moe, --n-cpu-moe or the
    /// --fit split) then go to memory the GPU driver has pinned, which the
    /// card copies from directly; from a mapped file each copy goes through
    /// a staging buffer of the driver's, which the expert cache's copies
    /// measured at 9 GB/s on an RTX 5070 Ti over PCIe 4.0 x16. Loading reads
    /// the whole file up front, and memory pinned for the experts cannot be
    /// swapped out: the RAM has to hold them. --moe-cache does this by itself
    /// when the RAM can spare the experts.
    #[arg(long)]
    no_mmap: bool,

    /// Keep the model file mapped where --moe-cache would read it into
    /// memory to pin its experts, or where a model that goes to the GPUs whole
    /// is more than half the memory this process may use (the RAM, or a Slurm
    /// job's --mem), which is otherwise read in: mapped, such a model did not
    /// load in an hour on LUMI-G, read in it loaded in 3 to 4 minutes.
    #[arg(long, conflicts_with = "no_mmap")]
    mmap: bool,

    /// For MoE models with experts in RAM: while a prompt is read, copy the
    /// experts of the layers ahead into N slots of VRAM on a second stream
    /// of the GPU, while the layers before them compute, instead of halting
    /// the computing for every copy. 2 to 8 slots, 4 by default; 0 turns it
    /// off. It works on one CUDA GPU, with the experts in pinned memory (a
    /// model read into memory: --no-mmap, which --moe-cache chooses when the
    /// RAM can spare the experts), on micro-batches of 512 tokens or more.
    /// Each slot holds the largest expert tensor, and --moe-cache keeps that
    /// VRAM out of the cache where there is room: on an RTX 5070 Ti,
    /// Qwen3.8-Flash-Next (IQ2_XS) took four slots of 256 MiB out of its
    /// cache, read a 33,200-token prompt 42% faster, to the same answer, and
    /// wrote as fast.
    #[arg(
        long,
        value_name = "N",
        default_value_t = fit::MOE_PREFETCH_SLOTS,
        value_parser = fit::parse_moe_prefetch
    )]
    moe_prefetch: u32,

    /// Threads that read the model file ahead of the load, into the page
    /// cache, so the load finds it in memory. Off by default (0): on Lustre
    /// it made loads slower, not faster (LUMI-G, cold: a 132 GiB model 110 s
    /// without readers, 128 s with 16): llama.cpp alone read the file at
    /// 0.7-1.3 GB/s there.
    /// Kept for file systems where it has not been measured. `auto` uses 16
    /// on Lustre, NFS, SMB, GPFS, BeeGFS, CephFS and 9p and none on a local
    /// disk. Never for a model larger than the memory free for the page cache
    /// (the RAM, or a Slurm job's --mem).
    #[arg(
        long,
        value_name = "N|auto",
        default_value = "0",
        value_parser = readahead::parse_load_threads
    )]
    load_threads: readahead::LoadThreads,

    /// One KV cache for every sequence (llama.cpp's `--kv-unified`) instead of
    /// one per sequence, with several slots (`--batch-size` above 1).
    /// Experimental, off by default. With one cache per sequence llama.cpp
    /// splits a decode step into one pass of the model for every run of
    /// consecutive slot numbers among the answering sequences, so a slot left
    /// out of a step (waiting for its prompt, or idle) splits it in two; one
    /// cache takes any slots in one pass, and its attention reads every
    /// sequence's cells, masked. Measured on an MI250X: 1-4% faster with
    /// short requests that come and go, 2-7% slower with sixteen at once,
    /// 38-45% slower with 32k-token contexts.
    #[arg(long)]
    kv_unified: bool,

    /// Max full-sequence-state checkpoints kept for prompt-prefix
    /// restore (bounded alternative to --rs-seq for hybrid/recurrent
    /// architectures — see the README's "--ctx-checkpoints" section).
    /// 0 (default) disables checkpointing: no snapshot is ever taken,
    /// matching pre-checkpoint behavior exactly. Mirrors llama.cpp
    /// server's flag of the same name (default there: 32); kept off
    /// here since each checkpoint costs one sequence's full state
    /// size. Only useful together with continuous batching
    /// (--batch-size > 0, the default for `run`).
    #[arg(long, default_value_t = 0)]
    ctx_checkpoints: usize,

    /// Minimum new tokens since the closest existing checkpoint of the
    /// same conversation before taking another one. Mirrors llama.cpp
    /// server's `--checkpoint-min-step` (default there: 8192). Only
    /// consulted when --ctx-checkpoints > 0.
    #[arg(long, default_value_t = 8192)]
    checkpoint_min_step: u32,

    /// Context window size
    #[arg(short, long, default_value_t = 4096)]
    ctx_size: u32,

    /// Maximum concurrent requests served by the continuous-batching scheduler.
    ///
    /// `--ctx-size` is the *total* KV budget and is split evenly across these
    /// slots, so per-sequence context is `ctx_size / batch_size`. One slot is
    /// therefore the only default that cannot surprise anyone: the request
    /// gets the whole window that was asked for.
    ///
    /// `serve` used to default to 8. With the 4096 default context that is 512
    /// tokens per request — which a reasoning model spends before it finishes
    /// thinking, so the answer stops mid-sentence and is reported as
    /// `done_reason="length"`. Nothing about that points back at a flag nobody
    /// set. Concurrency is worth having and worth asking for: raise this to
    /// 4–16 when using the engine as a backend for simultaneous users, and
    /// raise `--ctx-size` with it.
    #[arg(long, default_value_t = 1)]
    batch_size: usize,

    /// Number of CPU threads (default: all available)
    #[arg(short, long)]
    threads: Option<u32>,

    /// Disable flash attention (enabled by default for faster inference)
    #[arg(long)]
    no_flash_attn: bool,

    /// Prompt processing batch size (tokens per eval during prefill)
    #[arg(long, default_value_t = 2048)]
    n_batch: u32,

    /// Physical micro-batch: how many prompt tokens the GPU processes in
    /// one pass (llama.cpp's `n_ubatch`; default 512, llama.cpp's own, and
    /// 2048 for a model loaded with an expert cache, see --moe-cache).
    /// Raise it, to 2048-8192, for an MoE model whose experts do not all
    /// fit in VRAM: the experts kept in RAM are copied to the GPU once per
    /// micro-batch of a prompt, so a 32k-token prompt read 512 tokens at a
    /// time copies them 64 times, and 4096 at a time, 8. The compute buffer
    /// grows with it, and --fit makes room for it by keeping fewer layers'
    /// experts on the GPU: answers are written a little slower. Raises
    /// --n-batch to the same value when that is smaller.
    #[arg(
        long,
        value_name = "N",
        value_parser = clap::value_parser!(u32).range(32..=16_384)
    )]
    n_ubatch: Option<u32>,

    /// KV cache type for keys. Options: f16 (default, best GPU compat), q8_0, q4_0
    #[arg(long, default_value = "f16")]
    cache_type_k: String,

    /// KV cache type for values. Options: f16 (default, best GPU compat), q8_0, q4_0
    #[arg(long, default_value = "f16")]
    cache_type_v: String,

    /// Enable transparent web browsing: URLs in user messages are fetched
    /// and their content is injected into the prompt before inference.
    /// Dynamic budget: available context = ctx_size - prompt - 512 reserve.
    #[arg(long)]
    web: bool,

    /// Port for the embedded chat UI (separate from the API port so
    /// the API surface on --port stays pure). Default 11435.
    #[arg(long, default_value_t = 11435)]
    ui_port: u16,

    /// Run as a background daemon (writes PID to --pidfile)
    #[arg(long)]
    daemon: bool,

    /// PID file path (used with --daemon)
    #[arg(long, default_value = DEFAULT_PIDFILE)]
    pidfile: String,

    /// Log file for the background daemon (used with --daemon).
    ///
    /// Defaults to `~/.eullm/logs/eullm.log`. It used to be derived from
    /// `--pidfile`, which put it in `/tmp` by default: a directory that is
    /// small on many systems and gets cleared exactly when something is
    /// filling it up or the machine is rebooted after a crash — the two
    /// moments the log is the only record of what happened (#354). The
    /// PID file stays in `/tmp`, where a file that is meaningless across
    /// reboots belongs.
    ///
    /// Setting `--pidfile` without `--logfile` still puts the log next to
    /// the PID file, so an existing deployment that redirects one keeps
    /// both together.
    #[arg(long, value_name = "PATH")]
    logfile: Option<String>,

    /// Enable extra internal diagnostics for the Rust engine layer. Off
    /// by default (zero added per-token cost, matches upstream
    /// llama.cpp). Today this enables a NaN/Inf scan of every generated
    /// token's logits before sampling — added to help diagnose garbage
    /// output (issue #140) — at the cost of one extra linear scan over
    /// the vocab per token. Not a general log-level flag; use RUST_LOG
    /// for that.
    #[arg(long)]
    rust_debug: bool,

    /// Path to a multimodal projector (mmproj GGUF) for this model.
    ///
    /// Normally unnecessary: a model pulled from the catalog gets its
    /// projector alongside the weights, and one sitting next to the GGUF
    /// as `mmproj*.gguf` is picked up on its own. Needed when the two
    /// live apart, which is common for a model assembled by hand from a
    /// HuggingFace repo. A projector belongs to the model it was trained
    /// with: pairing it with different weights produces confident
    /// nonsense rather than an error.
    #[arg(long, value_name = "PATH")]
    mmproj: Option<PathBuf>,

    /// Keep a multimodal model's projector on the GPU, whatever sizing
    /// would decide. The name is llama.cpp's.
    ///
    /// By default, with sizing on, the projector goes on the GPU only when
    /// the whole text model still fits beside it, and to system RAM
    /// otherwise — before any text layer is moved off the card. A projector
    /// runs once per image and sits idle for every token after it; a text
    /// layer in RAM slows every token of every request. Worth forcing when
    /// nearly every request carries an image, at the price of text layers.
    #[arg(long, conflicts_with = "no_mmproj_offload")]
    mmproj_offload: bool,

    /// Keep a multimodal model's projector in system RAM, whatever sizing
    /// would decide: the most VRAM for the text model, and images encoded
    /// on the CPU. The name is llama.cpp's.
    #[arg(long)]
    no_mmproj_offload: bool,

    /// How long to keep a model resident after its last use, before
    /// unloading it to free VRAM/RAM. Accepts a duration ("5m", "30s",
    /// "2h") or a bare number of seconds. Applies to both the generation
    /// model and the embedding model (loaded on demand by naming it in a
    /// `/v1/embeddings` or `/api/embed` request), independently, and is a
    /// *default*: a request's own `keep_alive` field overrides it for that
    /// load.
    ///
    /// Unset by default — matches every release before this flag existed,
    /// where nothing unloaded a model on its own. Set it to make an
    /// idle GPU actually go idle: without it, a model loaded once and left
    /// untouched keeps the card's memory clocks up indefinitely (see
    /// `docs/arm-cix-p1-cpu-profile.md`'s note on why VRAM occupancy itself
    /// isn't the cost — an active CUDA context is).
    #[arg(long, value_name = "DURATION")]
    keep_alive: Option<String>,

    /// Load a text-embedding model (BGE, E5, and similar) at startup and
    /// reserve VRAM for it before sizing the generation model, so both stay
    /// resident together instead of leaving it to chance whether `--fit`'s
    /// own safety margin happens to be enough.
    ///
    /// Without this flag, an embedding model named in a `/v1/embeddings` or
    /// `/api/embed` request still loads on demand — see `docs/engine.md`'s
    /// "Text Embeddings and the Embedding Slot" — but nothing is reserved
    /// for it ahead of time: on a card where the generation model's own
    /// `--fit` sizing consumes past what a small embedder needs, the first
    /// embedding request evicts the generation model to make room, and it
    /// reloads on the next generation request. Fine for a card too small
    /// for both anyway; wasteful churn for one that has room for both but
    /// where `--fit`'s margin, sized for the generation model alone, was
    /// never told to leave any extra behind.
    ///
    /// With this flag, the embedding model is treated as a **reserved
    /// companion**: it loads first, with the context its inputs are embedded
    /// in (built for its longest input and kept for every request), so both
    /// already show up as used VRAM by the time `--fit` reads free VRAM to
    /// size the generation model — no separate bookkeeping needed there —
    /// and `--fit` also keeps a small margin free on top of that, for what a
    /// decode allocates beside them. A later chat-model swap (a different
    /// model named in a request)
    /// protects the same margin again rather than evicting the companion —
    /// it keeps its place across the process's lifetime, not just at
    /// launch.
    ///
    /// Accepts a GGUF path or a name already in the model store, the same
    /// two forms `--mmproj` accepts — trusted like any other CLI argument,
    /// unlike a request's `model` field, which is gated by
    /// `EULLM_ALLOW_MODEL_PATHS` (see `docs/engine.md`). If reserving its
    /// space would leave the generation model no room at all, the launch
    /// proceeds anyway with a warning: the embedder falls back to loading
    /// on demand, exactly as if this flag had not been given, rather than
    /// refusing to start over a sizing decision automatic sizing already
    /// makes gracefully everywhere else.
    #[arg(long, value_name = "PATH_OR_NAME")]
    embedding_model: Option<String>,

    /// Load a decision model for `POST /v1/systemone` at startup.
    ///
    /// `/v1/systemone` answers typed questions about a state — yes/no,
    /// one-of-N, a level on a scale — from the model's next-token
    /// probabilities instead of generated text (see `docs/engine.md`). It
    /// runs on its own model slot, like embeddings, and this flag gives that
    /// slot the same reserved-companion treatment as `--embedding-model`:
    /// the model loads first, `--fit` keeps free the VRAM a request's
    /// context needs (its KV cache at `--decision-ctx` plus a compute
    /// buffer), and a later chat-model swap does not evict it. Without the
    /// flag, the first request that names a model loads it on demand.
    ///
    /// A small instruction-tuned model is the intended shape — Qwen3 0.6B
    /// to 4B from the catalog. Accepts a GGUF path or a store name, like
    /// `--embedding-model`.
    #[arg(long, value_name = "PATH_OR_NAME")]
    decision_model: Option<String>,

    /// Most tokens of context one `/v1/systemone` request may use: the state
    /// plus its longest question (plus every other question too in
    /// `batched` mode).
    ///
    /// The decision model keeps one context between requests, sized by the
    /// largest request so far, so this is a ceiling, not memory held from
    /// the start — but it is what the decision slot keeps free in VRAM
    /// (for Qwen3-0.6B, 112 KiB per token: 896 MiB at the default). 8192
    /// fits a state of nearly 8k tokens, or in `batched` mode a 4k-token
    /// state with about fifty short questions.
    #[arg(
        long,
        value_name = "N",
        default_value_t = inference::decision::DEFAULT_DECISION_CTX,
        value_parser = clap::value_parser!(u32).range(512..=131_072)
    )]
    decision_ctx: u32,

    /// How many generation models to keep loaded at once, 1 to 16. The
    /// embedding and decision models have slots of their own and are not
    /// counted.
    ///
    /// A model a request names that is not loaded is loaded beside the
    /// others while fewer than this many are; past that, the least recently
    /// used idle one is unloaded first. A model answering requests is never
    /// unloaded to make room: the load waits up to 120 s for one to finish,
    /// then answers 503 with Retry-After.
    ///
    /// A model loads beside others only if it fits whole on the GPU in what
    /// they leave free — every layer, and its projector — and otherwise
    /// models are unloaded until it does, or until it is alone, when it is
    /// sized like any model loaded by itself. Without automatic sizing
    /// (--no-fit, or a build that cannot read free VRAM) only the count is
    /// kept.
    ///
    /// Default 1: one model at a time, as before — a request for another
    /// model replaces it, even mid-answer. A default above 1 would not be a
    /// default but a decision to divide the card, because a second model
    /// only ever gets what the first one left. Ollama's
    /// OLLAMA_MAX_LOADED_MODELS counts every model and is read from the
    /// environment; this is a flag, and counts generation models only.
    #[arg(
        long,
        value_name = "N",
        default_value_t = 1,
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..=16)
    )]
    max_loaded_models: usize,

    /// The generation model a request that names none is answered by: a
    /// store name or a GGUF path, like --decision-model.
    ///
    /// A request with no `model` field, or an empty one, goes to this model,
    /// which is loaded for it when it is not loaded, as if the request had
    /// named it. Without the flag such a request is answered by the most
    /// recently used generation model — with one model at a time, the one
    /// there is — and refused when none is loaded.
    ///
    /// `auto`, with --auto-model, routes such a request as one naming
    /// `"model": "auto"` is.
    ///
    /// Checked at startup: a name that is no model stops the server there,
    /// rather than letting every request that names none fail.
    #[arg(long, value_name = "NAME_OR_PATH")]
    default_model: Option<String>,

    /// A model `"model": "auto"` may choose, as NAME or NAME=DESCRIPTION.
    /// Give it two to eight times, smallest model first.
    ///
    /// A request naming the model `auto` is answered by one of these: those
    /// that cannot take it — it carries an image and the model has no
    /// projector, or it is longer than the model's context — are left out,
    /// and the decision model (--decision-model) chooses among the rest from
    /// a digest of the request. The description is what it reads about each
    /// option: say which requests the model should get ("Short everyday
    /// requests", "Multi-step reasoning, maths and code"), in at most 400
    /// characters. Without one, the store's or the catalog's description is
    /// used, and the startup log warns.
    ///
    /// Whenever the decision model does not decide — none is loaded, it did
    /// not answer within --auto-timeout-ms, it failed — the fallback
    /// answers: --default-model when it is one of these, the last otherwise.
    /// `POST /api/route` shows the choice for a request without generating.
    #[arg(long, value_name = "NAME[=DESCRIPTION]", action = clap::ArgAction::Append)]
    auto_model: Vec<String>,

    /// The most routing (`"model": "auto"`) may add to a request, in
    /// milliseconds, 10 to 60000: a decision not made by then is abandoned,
    /// and the fallback answers.
    #[arg(
        long,
        value_name = "MS",
        default_value_t = api::DEFAULT_AUTO_TIMEOUT_MS,
        value_parser = clap::value_parser!(u64).range(10..=60_000)
    )]
    auto_timeout_ms: u64,
}

#[derive(Subcommand)]
enum Commands {
    /// Pull a model from the EU catalog
    ///
    /// With no argument in an interactive terminal, opens the picker
    /// filtered to catalog models only.
    Pull {
        /// Model id (e.g., qwen3-8b) — see `eullm catalog` or the picker
        model: Option<String>,
    },
    /// Run a model locally (starts API server)
    ///
    /// With no argument in an interactive terminal, opens the picker so
    /// the user can choose a local model, a catalog model, or paste a
    /// custom path/URL.
    Run {
        /// Model id (catalog), path to a local GGUF file, URL to one, or a
        /// HuggingFace repo shorthand (`hf.co/<owner>/<repo>[:<quant>]`)
        model: Option<String>,

        #[command(flatten)]
        opts: RuntimeOpts,

        /// Disable the embedded chat UI (otherwise served on --ui-port).
        /// Use this for headless / backend / RAG deployments where you only
        /// want the OpenAI/Ollama API surface exposed.
        #[arg(long)]
        no_ui: bool,

        /// Terminal-only: don't auto-open the browser chat on startup, just
        /// drop into the CLI REPL. (The chat UI is still served on --ui-port
        /// unless --no-ui is also given.) Alias: --no-chat.
        #[arg(long, visible_alias = "no-chat")]
        cli: bool,

        /// (Multimodal MVP, --features multimodal builds only.) Path to an
        /// image or audio file to send together with the first prompt.
        /// Triggers the multimodal inference path (mtmd) which requires the
        /// model's mmproj projector to be available; for catalog models it
        /// is auto-downloaded during `pull`. The HTTP API also routes media
        /// (web chat / `/api/chat` `images`); this flag is the CLI one-shot.
        #[arg(long, value_name = "PATH")]
        image: Option<PathBuf>,
    },
    /// List locally available models
    List,
    /// Show model information
    Show {
        /// Model name
        model: String,
    },
    /// Remove a locally downloaded model (frees disk space)
    ///
    /// Examples:
    ///   eullm rm qwen3-14b
    ///   eullm rm qwen3-14b --force      (skip the confirmation prompt)
    #[command(visible_alias = "remove")]
    Rm {
        /// Model id (as shown by `eullm list`)
        model: String,

        /// Skip the confirmation prompt
        #[arg(short, long)]
        force: bool,
    },
    /// Start the API server without loading a model
    Serve {
        #[command(flatten)]
        opts: RuntimeOpts,

        /// Enable the embedded chat UI (off by default for headless serve).
        /// Pass --ui to also expose the chat at http://localhost:<ui-port>/.
        #[arg(long)]
        ui: bool,
    },
    /// Unload generation models from a running eullm server, freeing their
    /// VRAM — without restarting the server.
    ///
    /// Every loaded generation model, or with --model only that one, the
    /// others staying loaded. Requests still running on an unloaded model
    /// are cut off. A later request with a `model` field (or another `eullm
    /// run <model>`) loads a model back in. Useful for temporarily handing
    /// GPU memory to another process — e.g. an embedding model needed during
    /// RAG document ingestion — then reloading the LLM once it's done.
    ///
    /// Examples:
    ///   eullm unload
    ///   eullm unload --model qwen3-8b
    ///   eullm unload --port 11500
    Unload {
        /// Port of the running eullm API server
        #[arg(short, long, default_value_t = 11434)]
        port: u16,

        /// Unload only this model (as `/api/tags` names it), and keep the
        /// others loaded
        #[arg(long, value_name = "NAME")]
        model: Option<String>,
    },
    /// Import a model from a local Ollama installation
    ///
    /// Copies the GGUF blob from Ollama's storage into EULLM's model store,
    /// so you can test both engines with the exact same model file.
    ///
    /// Examples:
    ///   eullm import-ollama llama3.2
    ///   eullm import-ollama qwen3:14b
    ///   eullm import-ollama gemma3 --ollama-dir /custom/ollama/path
    ImportOllama {
        /// Ollama model name (e.g., llama3.2, qwen3:14b)
        model: String,

        /// Custom Ollama data directory (default: ~/.ollama)
        #[arg(long)]
        ollama_dir: Option<String>,
    },
    /// Update EuLLM to the latest release
    ///
    /// Asks github.com which release is the latest: the only time EuLLM
    /// looks, since it never checks on its own. When there is a newer one,
    /// downloads the same build as this one (CPU, CUDA, Vulkan, ROCm),
    /// checks it against the release's checksums, makes sure it starts, and
    /// puts it in place of this one. A build from source is not replaced.
    ///
    /// Examples:
    ///   eullm update --check    (only say whether a newer release exists)
    ///   eullm update
    Update {
        /// Only say whether a newer release exists; change nothing
        #[arg(long)]
        check: bool,
    },
    /// Train a model's weights on a text, on this machine's CPU or GPU
    ///
    /// llama.cpp's own trainer, so its limits: an F32 GGUF, flash attention
    /// off, the whole window in one micro-batch. Validation loss is measured
    /// before and after each epoch; the result is a new F32 GGUF.
    ///
    /// Examples:
    ///   eullm finetune ./model-f32.gguf --data corpus.txt --epochs 2
    ///   eullm finetune ./model-f32.gguf --data train.jsonl --optimizer sgd --dry-run
    Finetune {
        #[command(flatten)]
        opts: finetune::FinetuneOpts,
    },
    /// Verticalize a model: compress, specialize, and brand it
    ///
    /// Examples:
    ///   eullm forge Qwen/Qwen3-14B --profile legal-it
    ///   eullm forge Qwen/Qwen3-30B --profile medical-de --identity "MedAI"
    Forge {
        /// Source model (HuggingFace ID or local path)
        source: String,

        /// Verticalizzazione profile (legal-it, medical-de, finance-fr)
        #[arg(short, long)]
        profile: Option<String>,

        /// Model identity name (e.g., "LegalAI di Studio Rossi")
        #[arg(long)]
        identity: Option<String>,

        /// Comma-separated language codes (e.g., it,en)
        #[arg(long)]
        lang: Option<String>,

        /// Output directory or model name
        #[arg(short, long)]
        output: Option<String>,

        /// Target VRAM in GB
        #[arg(long)]
        target_vram: Option<u16>,

        /// Only estimate costs, don't run pipeline
        #[arg(long)]
        estimate_only: bool,

        /// Skip structural pruning
        #[arg(long)]
        skip_pruning: bool,

        /// Skip knowledge distillation
        #[arg(long)]
        skip_distillation: bool,

        /// Skip quantization
        #[arg(long)]
        skip_quantization: bool,

        /// Skip identity fine-tuning
        #[arg(long)]
        skip_identity: bool,
    },
}

#[tokio::main]
async fn main() {
    // Check --daemon BEFORE initializing tracing/tokio internals.
    // The daemon spawns a child process, so must happen early.
    {
        let args: Vec<String> = std::env::args().collect();
        if args.contains(&"--daemon".to_string()) {
            // Read straight from argv: clap has not run yet, and it cannot,
            // because the whole point is to re-exec before the runtime and
            // the model are touched.
            let pidfile = arg_value(&args, "--pidfile");
            let logfile = arg_value(&args, "--logfile");
            let home = std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE"))
                .ok();
            let log_path = resolve_daemon_log_path(logfile, pidfile, home.as_deref());
            daemonize(pidfile.unwrap_or(DEFAULT_PIDFILE), &log_path);
            // daemonize exits the parent — child continues below without --daemon.
        }
    }

    // The Windows console shows ANSI colour codes as literal text (`←[32m`)
    // unless virtual terminal processing is switched on for it, which the
    // classic Windows PowerShell / cmd console does not do by default:
    // every log line, the model picker and the terminal chat came out
    // littered with escape codes there, while Windows Terminal hid the
    // problem. Switch it on once, before anything is logged; where it
    // cannot be (output redirected to a file, no console at all) keep the
    // log lines plain instead of writing escape codes into them. On every
    // other platform this is `None` and the subscriber keeps its own default
    // (which honours NO_COLOR).
    let logs = tracing_subscriber::fmt();
    let logs = if anstyle_query::windows::enable_ansi_colors() == Some(false) {
        logs.with_ansi(false)
    } else {
        logs
    };

    logs.with_env_filter(
        tracing_subscriber::EnvFilter::try_from_default_env()
            // Module paths are rooted at the [[bin]] name ("eullm" in
            // Cargo.toml), not the package name ("eullm-engine") - there's
            // no separate lib.rs, so every tracing::info!/warn! call site
            // resolves its target under "eullm::...". "eullm_engine=info"
            // never matched anything, silently disabling all engine
            // logging (KV-reuse diagnostics, context/scheduler startup
            // info, etc.) unless RUST_LOG was set explicitly.
            .unwrap_or_else(|_| "eullm=info".into()),
    )
    .init();

    // Install signal handler for SIGABRT — llama.cpp calls abort() on
    // GGML_ASSERT failures, which kills the process with no diagnostic info.
    // This handler prints a helpful message before the default action runs.
    install_abort_handler();

    let cli = Cli::parse();

    let store = match ModelStore::default_store() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error: could not initialize model store: {e}");
            std::process::exit(1);
        }
    };

    // No subcommand at all → open the interactive picker (if TTY) and
    // dispatch what the user chose into the regular Run flow with default
    // settings. Non-interactive (pipe/redirect) prints a usage hint instead.
    let cli_command = match cli.command {
        Some(c) => c,
        // The picker only opens on an interactive terminal, so `--fit` is the
        // default here (unlike the scriptable `eullm run`, where it stays
        // opt-in): a user choosing a model from the menu gets GPU layers
        // auto-sized to free VRAM instead of an out-of-memory abort.
        None => match picker::pick(&store).await {
            Some(picker::Picked::Local(path)) => picker_run(&path.to_string_lossy()),
            Some(picker::Picked::Catalog(entry)) => picker_run(&entry.id),
            Some(picker::Picked::Url(_url)) => {
                eprintln!(
                    "URL launch from picker not yet supported. \
                     Workaround: `eullm pull <id>` from the catalog, or \
                     download the .gguf manually and pass its path."
                );
                std::process::exit(2);
            }
            Some(picker::Picked::Quit) => return,
            None => {
                eprintln!("eullm — sovereign LLM runtime for Europe");
                eprintln!();
                eprintln!("Usage:");
                eprintln!("  eullm run <model.gguf | catalog-id>   Run a model");
                eprintln!("  eullm list                            List local models");
                eprintln!("  eullm pull <catalog-id>               Download a catalog model");
                eprintln!("  eullm --help                          Show full help");
                eprintln!();
                eprintln!(
                    "Tip: launch `eullm` from an interactive terminal to pick a model from a menu."
                );
                std::process::exit(1);
            }
        },
    };

    match cli_command {
        Commands::Pull { model } => cmd_pull_maybe(&store, model.as_deref()).await,
        Commands::Run {
            model,
            no_ui,
            cli,
            image,
            opts,
        } => {
            // One `let` re-binds every shared flag under the name the body
            // already uses, so extracting `RuntimeOpts` cost nothing below
            // this line.
            let RuntimeOpts {
                port,
                batch_size,
                replace,
                gpu_layers,
                fit,
                no_fit,
                fit_strict,
                cpu_moe,
                n_cpu_moe,
                rs_seq,
                mtp,
                mtp_p_min,
                mtp_model,
                moe_cache,
                no_mmap,
                mmap,
                moe_prefetch,
                load_threads,
                kv_unified,
                ctx_checkpoints,
                checkpoint_min_step,
                ctx_size,
                threads,
                no_flash_attn,
                n_batch,
                n_ubatch,
                cache_type_k,
                cache_type_v,
                web,
                ui_port,
                daemon,
                pidfile,
                logfile,
                rust_debug,
                mmproj,
                mmproj_offload,
                no_mmproj_offload,
                keep_alive,
                embedding_model,
                decision_model,
                decision_ctx,
                max_loaded_models,
                default_model,
                auto_model,
                auto_timeout_ms,
            } = opts;
            let n_batch = launch_n_batch(n_batch, n_ubatch.unwrap_or(inference::DEFAULT_N_UBATCH));
            let residency = residency_config(
                &store,
                mmproj.as_deref(),
                api::ResidencyFlags {
                    max_loaded_models,
                    default_model,
                    auto_models: auto_model,
                    auto_timeout_ms,
                },
            );
            // `None` lets sizing decide; either flag decides instead.
            let mmproj_offload = match (mmproj_offload, no_mmproj_offload) {
                (true, _) => Some(true),
                (_, true) => Some(false),
                _ => None,
            };
            let keep_alive = keep_alive.as_deref().map(|s| {
                api::parse_keep_alive_flag(s).unwrap_or_else(|e| {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                })
            });
            // Automatic sizing resolves here, once, for both subcommands.
            // `--no-fit` is the only thing that turns it off: an explicit
            // `--gpu-layers` becomes a CEILING instead (see
            // `fit::apply_gpu_layers_ceiling`), because a layer count picked
            // for one model is not a fact about the next one and would take
            // the guardrail away exactly when the user is steering.
            // `fit_explicit` (only when the user typed --fit) is what allows
            // the interactive confirmation; automatic sizing never prompts.
            let fit_explicit = fit;
            let fit = !no_fit;
            let gpu_layers = gpu_layers.unwrap_or(-1);
            // `eullm run` with no model → picker, dispatch back through the same Run.
            let model = match model {
                Some(m) => m,
                None => match picker::pick(&store).await {
                    Some(picker::Picked::Local(p)) => p.to_string_lossy().into_owned(),
                    Some(picker::Picked::Catalog(entry)) => entry.id.clone(),
                    Some(picker::Picked::Url(_)) => {
                        eprintln!("URL launch from picker not yet supported.");
                        std::process::exit(2);
                    }
                    Some(picker::Picked::Quit) => return,
                    None => {
                        eprintln!("Error: missing <MODEL> argument.");
                        eprintln!("Usage: eullm run <model.gguf | catalog-id>");
                        std::process::exit(1);
                    }
                },
            };
            // --daemon is handled at the top of main() before tokio starts.
            let _ = (daemon, pidfile, logfile);
            let mut ctk = inference::parse_cache_type(&cache_type_k).unwrap_or_else(|e| {
                eprintln!("Error: {e}");
                std::process::exit(1);
            });
            let mut ctv = inference::parse_cache_type(&cache_type_v).unwrap_or_else(|e| {
                eprintln!("Error: {e}");
                std::process::exit(1);
            });
            // Gemma 4 requires f16 KV cache (mixed SWA architecture) — see
            // `inference::correct_kv_cache_for_model` for the rationale. The
            // same correction also applies inside `load_generation_model` so it can't
            // be bypassed by swapping models after startup.
            let (corrected_k, corrected_v, corrected) =
                inference::correct_kv_cache_for_model(&model, ctk, ctv);
            if corrected {
                eprintln!(
                    "[EULLM] Gemma 4 detected with non-f16 KV cache ({cache_type_k}/{cache_type_v})."
                );
                eprintln!("[EULLM] Mixed SWA architecture (D=512/256) requires f16 KV cache.");
                eprintln!("[EULLM] Auto-correcting to f16/f16.");
                ctk = corrected_k;
                ctv = corrected_v;
            }
            // After the Gemma correction, so this sees what will actually be
            // used rather than what was typed.
            let (fa_k, fa_corrected) =
                inference::correct_kv_cache_for_flash_attn(ctk, !no_flash_attn);
            if fa_corrected {
                inference::report_flash_attn_kv_correction(ctk);
                ctk = fa_k;
            }
            if cpu_moe && n_cpu_moe > 0 {
                eprintln!("Error: --cpu-moe and --n-cpu-moe are mutually exclusive.");
                eprintln!(
                    "Use --cpu-moe to offload all experts, or --n-cpu-moe N to offload only the first N layers."
                );
                std::process::exit(1);
            }
            let ui_port_opt = if no_ui { None } else { Some(ui_port) };
            // Auto-open the browser chat unless the user asked for terminal-only
            // (--cli / --no-chat) or disabled the UI entirely (--no-ui).
            let open_chat = !cli && ui_port_opt.is_some();
            cmd_run(
                &store,
                &model,
                port,
                replace,
                gpu_layers,
                fit,
                fit_explicit,
                fit_strict,
                cpu_moe,
                n_cpu_moe,
                rs_seq,
                mtp,
                mtp_p_min,
                mtp_model,
                moe_cache,
                no_mmap,
                mmap,
                moe_prefetch,
                load_threads,
                kv_unified,
                ctx_checkpoints,
                checkpoint_min_step,
                ctx_size,
                threads,
                batch_size,
                !no_flash_attn,
                n_batch,
                n_ubatch,
                ctk,
                ctv,
                web,
                ui_port_opt,
                open_chat,
                image,
                rust_debug,
                mmproj,
                mmproj_offload,
                keep_alive,
                embedding_model,
                decision_model,
                decision_ctx,
                residency,
            )
            .await;
        }
        Commands::List => cmd_list(&store),
        Commands::Show { model } => cmd_show(&store, &model),
        Commands::Rm { model, force } => cmd_rm(&store, &model, force),
        Commands::Serve { ui, opts } => {
            // One `let` re-binds every shared flag under the name the body
            // already uses, so extracting `RuntimeOpts` cost nothing below
            // this line.
            let RuntimeOpts {
                port,
                batch_size,
                replace,
                gpu_layers,
                // `--fit` alone changes nothing on serve (no prompt to allow).
                fit: _,
                no_fit,
                fit_strict,
                cpu_moe,
                n_cpu_moe,
                rs_seq,
                mtp,
                mtp_p_min,
                mtp_model,
                moe_cache,
                no_mmap,
                mmap,
                moe_prefetch,
                load_threads,
                kv_unified,
                ctx_checkpoints,
                checkpoint_min_step,
                ctx_size,
                threads,
                no_flash_attn,
                n_batch,
                n_ubatch,
                cache_type_k,
                cache_type_v,
                web,
                ui_port,
                daemon,
                pidfile,
                logfile,
                rust_debug,
                mmproj,
                mmproj_offload,
                no_mmproj_offload,
                keep_alive,
                embedding_model,
                decision_model,
                decision_ctx,
                max_loaded_models,
                default_model,
                auto_model,
                auto_timeout_ms,
            } = opts;
            let n_batch = launch_n_batch(n_batch, n_ubatch.unwrap_or(inference::DEFAULT_N_UBATCH));
            let residency = residency_config(
                &store,
                mmproj.as_deref(),
                api::ResidencyFlags {
                    max_loaded_models,
                    default_model,
                    auto_models: auto_model,
                    auto_timeout_ms,
                },
            );
            // `None` lets sizing decide; either flag decides instead.
            let mmproj_offload = match (mmproj_offload, no_mmproj_offload) {
                (true, _) => Some(true),
                (_, true) => Some(false),
                _ => None,
            };
            let keep_alive = keep_alive.as_deref().map(|s| {
                api::parse_keep_alive_flag(s).unwrap_or_else(|e| {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                })
            });
            // Automatic sizing resolves here, once, for both subcommands:
            // an explicit --gpu-layers is an explicit choice and turns it
            // off, --no-fit turns it off outright, and `fit` stays true for
            // the default path so the engine sizes the offload itself.
            // No `fit_explicit` here: `serve` never prompts either way
            // (see `api::load_generation_model`), so the distinction has no meaning.
            // An explicit `--gpu-layers` is a ceiling, not an off switch.
            let fit = !no_fit;
            let gpu_layers = gpu_layers.unwrap_or(-1);
            let _ = (daemon, pidfile, logfile);
            if cpu_moe && n_cpu_moe > 0 {
                eprintln!("Error: --cpu-moe and --n-cpu-moe are mutually exclusive.");
                eprintln!(
                    "Use --cpu-moe to offload all experts, or --n-cpu-moe N to offload only the first N layers."
                );
                std::process::exit(1);
            }
            let ctk = inference::parse_cache_type(&cache_type_k).unwrap_or_else(|e| {
                eprintln!("Error: {e}");
                std::process::exit(1);
            });
            let ctv = inference::parse_cache_type(&cache_type_v).unwrap_or_else(|e| {
                eprintln!("Error: {e}");
                std::process::exit(1);
            });
            let (ctk, _) = {
                let (k, corrected) =
                    inference::correct_kv_cache_for_flash_attn(ctk, !no_flash_attn);
                if corrected {
                    inference::report_flash_attn_kv_correction(ctk);
                }
                (k, corrected)
            };
            let ui_port_opt = if ui { Some(ui_port) } else { None };
            cmd_serve(
                port,
                replace,
                ui_port_opt,
                batch_size,
                gpu_layers,
                fit,
                fit_strict,
                ctx_size,
                threads,
                !no_flash_attn,
                n_batch,
                n_ubatch,
                ctk,
                ctv,
                web,
                cpu_moe,
                n_cpu_moe,
                rs_seq,
                mtp,
                mtp_p_min,
                mtp_model,
                moe_cache,
                no_mmap,
                mmap,
                moe_prefetch,
                load_threads,
                kv_unified,
                ctx_checkpoints,
                checkpoint_min_step,
                rust_debug,
                mmproj,
                mmproj_offload,
                keep_alive,
                embedding_model,
                decision_model,
                decision_ctx,
                residency,
            )
            .await;
        }
        Commands::Unload { port, model } => cmd_unload(port, model.as_deref()).await,
        Commands::ImportOllama { model, ollama_dir } => {
            cmd_import_ollama(&store, &model, ollama_dir.as_deref())
        }
        Commands::Update { check } => {
            if let Err(e) = update::run(check).await {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        }
        Commands::Finetune { opts } => {
            let Some(path) = resolve_model_path(&opts.model, &store) else {
                eprintln!(
                    "Error: {} is neither a .gguf file nor a model in the store",
                    opts.model
                );
                std::process::exit(1);
            };
            if let Err(e) = finetune::cmd(&opts, &path) {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        }
        Commands::Forge {
            source,
            profile,
            identity,
            lang,
            output,
            target_vram,
            estimate_only,
            skip_pruning,
            skip_distillation,
            skip_quantization,
            skip_identity,
        } => cmd_forge(
            &source,
            profile.as_deref(),
            identity.as_deref(),
            lang.as_deref(),
            output.as_deref(),
            target_vram,
            estimate_only,
            skip_pruning,
            skip_distillation,
            skip_quantization,
            skip_identity,
        ),
    }

    // Exit deterministically instead of returning and letting the runtime
    // drop do it. `#[tokio::main]`'s drop waits for every in-flight
    // `spawn_blocking` task, and inference runs in exactly those: after
    // Ctrl+C the shutdown message printed, `main` returned, and the process
    // then sat there — invisibly — until a prefill that had minutes left
    // finished. Nothing here needs the wait: the audit trail is written per
    // request, models are read-only, and the scheduler holds no state worth
    // draining. Returning from `main` exits with 0 anyway, so this changes
    // only the hang.
}

/// `eullm pull` entry point: if `model` is `None`, open the picker filtered
/// to catalog selections; otherwise just call `cmd_pull` synchronously.
async fn cmd_pull_maybe(store: &ModelStore, model: Option<&str>) {
    if let Some(name) = model {
        cmd_pull(store, name).await;
        return;
    }
    match picker::pick(store).await {
        Some(picker::Picked::Catalog(entry)) => {
            cmd_pull(store, &entry.id).await;
        }
        Some(picker::Picked::Local(p)) => {
            println!("That model is already local: {}", p.display());
        }
        Some(picker::Picked::Url(url)) => {
            cmd_pull_url(store, &url).await;
        }
        Some(picker::Picked::Quit) => {}
        None => {
            eprintln!("Error: missing <MODEL> argument.");
            eprintln!("Usage: eullm pull <catalog-id>");
            std::process::exit(1);
        }
    }
}

/// True if `s` looks like an HTTP(S) URL we should download directly rather
/// than resolve against the catalog.
fn is_url(s: &str) -> bool {
    s.starts_with("http://") || s.starts_with("https://")
}

/// Already-downloaded quants of a HuggingFace repo, as store ids.
///
/// A ref without `:quant` resolves to the bare repo id, which is not what is
/// on disk once any quant has been pulled — the ids carry the quant suffix.
/// Without this lookup, `eullm run hf.co/owner/repo` after pulling
/// `repo:UD-Q4_K_M` re-downloaded the same file under the bare id (#345).
fn local_quants_of_repo(store: &ModelStore, hf: &registry::HfRef) -> Vec<String> {
    let repo_name = hf.repo.rsplit('/').next().unwrap_or(&hf.repo);
    let bare = hf_ref_to_model_id(&registry::HfRef {
        repo: repo_name.to_string(),
        quant: None,
        original: hf.original.clone(),
    });
    let prefix = format!("{bare}-");
    let mut ids: Vec<String> = store
        .list()
        .unwrap_or_default()
        .into_iter()
        .map(|m| m.id)
        .filter(|id| id.starts_with(&prefix) || *id == bare)
        .collect();
    ids.sort();
    ids
}


/// `eullm pull hf.co/owner/repo[:quant]`.
///
/// The sequence itself lives in `models::pull`, shared with `POST /api/pull`,
/// so a repo layout the download path learns to handle is learned by both at
/// once. All this adds is printing.
///
/// Returns the id the model can be run as: the ref's own, or — when the same
/// file was already stored under another id and could not be linked — that
/// other id.
async fn cmd_pull_hf(store: &ModelStore, hf: &registry::HfRef) -> String {
    use crate::models::pull::{PullEvent, pull_from_huggingface};

    let id = hf_ref_to_model_id(hf);

    if let Some(gguf) = store.gguf_path(&id) {
        println!("Model '{id}' is already downloaded.");
        println!("  GGUF: {}", gguf.display());
        println!("\nRun with: eullm run {id}");
        return id;
    }

    println!("  Storing as: {id}");
    println!("  (off-catalog model — no license/VRAM metadata available)");
    println!();

    // Bounded, and the sender drops ticks rather than blocking: a download
    // must not be paced by how fast a terminal repaints.
    let (tx, mut rx) = tokio::sync::mpsc::channel::<PullEvent>(256);
    let printer = tokio::spawn(async move {
        let mut last_line_was_progress = false;
        while let Some(ev) = rx.recv().await {
            match ev {
                PullEvent::Status(s) => {
                    if last_line_was_progress {
                        eprintln!();
                        last_line_was_progress = false;
                    }
                    println!("  {s}");
                }
                PullEvent::Progress {
                    completed, total, ..
                } => {
                    eprint!("\r  {}", registry::format_progress(completed, total));
                    let _ = std::io::Write::flush(&mut std::io::stderr());
                    last_line_was_progress = true;
                }
                PullEvent::Done { .. } | PullEvent::Failed(_) => {}
            }
        }
        if last_line_was_progress {
            eprintln!();
        }
    });

    let outcome = pull_from_huggingface(store, hf, &id, tx).await;
    let _ = printer.await;

    match outcome {
        Ok(stored) => {
            if stored == id {
                println!("  Done. Model ready.");
            }
            println!("\nRun with: eullm run {stored}");
            stored
        }
        Err(e) => {
            eprintln!("Pull failed: {e}");
            std::process::exit(1);
        }
    }
}

/// Derive a filesystem-safe model id and the GGUF filename from a download
/// URL. `https://example.com/models/Gemma4.gguf?token=x` →
/// id `gemma4`, filename `Gemma4.gguf`.
///
/// The id is the lowercased filename stem with anything outside
/// `[a-z0-9._-]` collapsed to `-`, so it nests cleanly under the store root
/// and can be typed back as `eullm run <id>`.
fn url_to_model_id(url: &str) -> (String, String) {
    // Strip query/fragment, take the last path segment.
    let path = url.split(['?', '#']).next().unwrap_or(url);
    let filename = path
        .rsplit('/')
        .find(|s| !s.is_empty())
        .unwrap_or("model.gguf")
        .to_string();

    let stem = filename.strip_suffix(".gguf").unwrap_or(&filename);
    let id: String = stem
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
    let id = if id.is_empty() {
        "model".to_string()
    } else {
        id
    };

    // Ensure the stored filename ends in .gguf so gguf_path() finds it.
    let filename = if filename.to_lowercase().ends_with(".gguf") {
        filename
    } else {
        format!("{id}.gguf")
    };
    (id, filename)
}

/// Pull a GGUF directly from an arbitrary URL, outside the catalog.
///
/// `eullm pull https://host/path/model.gguf` — downloads into the store
/// under an id derived from the filename, writes an external manifest, and
/// the model then behaves like any catalog model (`run`, `list`, `rm`).
async fn cmd_pull_url(store: &ModelStore, url: &str) -> String {
    let (id, filename) = url_to_model_id(url);

    if let Some(gguf) = store.gguf_path(&id) {
        println!("Model '{id}' is already downloaded.");
        println!("  GGUF: {}", gguf.display());
        println!("\nRun with: eullm run {id}");
        return id;
    }

    println!("Pulling from URL: {url}");
    println!("  Storing as: {id}");
    println!("  (off-catalog model — no license/VRAM metadata available)");
    println!();

    let model_dir = store.model_path(&id);
    let gguf_dest = model_dir.join(&filename);

    let result = {
        use crate::registry::{download_file, format_progress};
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU64, Ordering};

        let last_printed = Arc::new(AtomicU64::new(0));
        let progress: registry::ProgressCallback = Box::new(move |downloaded, total| {
            let last = last_printed.load(Ordering::Relaxed);
            if downloaded - last > 10_000_000 || (total > 0 && downloaded >= total) {
                last_printed.store(downloaded, Ordering::Relaxed);
                eprint!("\r  {}", format_progress(downloaded, total));
                let _ = std::io::Write::flush(&mut std::io::stderr());
            }
        });
        download_file(url, &gguf_dest, None, Some(progress)).await
    };
    eprintln!();

    match result {
        Ok(()) => {
            let size = std::fs::metadata(&gguf_dest).map(|m| m.len()).unwrap_or(0);
            match store.write_external_manifest(&id, &filename, url, size, None, None) {
                Ok(_) => {
                    println!("  Done. Model ready.");
                    println!("\nRun with: eullm run {id}");
                }
                Err(e) => eprintln!("Warning: download succeeded but manifest write failed: {e}"),
            }
            id
        }
        Err(e) => {
            eprintln!("Download failed: {e}");
            // The .gguf.part is removed by the downloader; drop the dir too so
            // a failed pull leaves no trace (mirrors catalog-pull cleanup).
            let _ = store.delete(&id);
            std::process::exit(1);
        }
    }
}

/// `eullm pull <catalog-id | hf.co/… | https://…>`.
///
/// Returns the id the model can be run as. That is the one asked for, except
/// when the same file was already stored under another id and could not be
/// linked (see `models::pull::reuse_stored_weights`): nothing is downloaded
/// then, and the id returned is the one that has it — which is what lets
/// `eullm run` go on to load it instead of failing on an id with no weights.
async fn cmd_pull(store: &ModelStore, model: &str) -> String {
    if is_url(model) {
        return cmd_pull_url(store, model).await;
    }

    if let Some(hf) = registry::parse_hf_ref(model) {
        return cmd_pull_hf(store, &hf).await;
    }

    let entry = match catalog::find_model(model) {
        Some(e) => e,
        None => {
            eprintln!("Error: model '{model}' not found in EU catalog.");
            eprintln!("Run `eullm list` to see available models.");
            eprintln!();
            eprintln!("You can also pull any GGUF by URL:");
            eprintln!("  eullm pull https://host/path/model.gguf");
            std::process::exit(1);
        }
    };

    // Check if already downloaded with GGUF
    if let Some(gguf) = store.gguf_path(&entry.id) {
        println!("Model '{}' is already downloaded.", entry.name);
        println!("  GGUF: {}", gguf.display());

        // If the catalog declares a multimodal projector (mmproj) and it's not
        // present on disk, fetch it now. Happens when the user pulled before
        // the catalog gained mmproj fields — without this branch the only fix
        // would be to delete the model and re-download the full ~7 GB.
        if entry.mmproj_repo.is_some()
            && entry.mmproj_filename.is_some()
            && store.mmproj_path(&entry.id).is_none()
        {
            let model_dir = store.model_path(&entry.id);
            if let Some(mmproj_filename) = download_mmproj(entry, &model_dir).await {
                // Refresh the manifest so its mmproj_file matches disk reality.
                // Which file the weights are is carried over as it was, not
                // taken from today's catalog entry: they may predate it.
                let gguf_name = gguf.file_name().and_then(|s| s.to_str());
                let existing = store.get(&entry.id).ok().flatten();
                let hf_file = existing.as_ref().and_then(|m| {
                    m.hf_repo
                        .as_deref()
                        .zip(m.hf_filename.as_deref())
                        .map(|(repo, path)| models::store::HfFile::new(repo, path))
                });
                if let Err(e) = store.write_manifest(
                    entry,
                    "ready",
                    gguf_name,
                    Some(&mmproj_filename),
                    hf_file.as_ref(),
                ) {
                    eprintln!("  Warning: mmproj downloaded but manifest update failed: {e}");
                }
            }
        }

        println!("\nRun with: eullm run {model}");
        return entry.id.clone();
    }

    println!("Pulling {} ...", entry.name);
    println!(
        "  {} | {} | ~{}GB VRAM | {}",
        entry.description,
        entry.base(),
        entry.vram_gb,
        entry.license
    );

    if entry.hf_repo.is_empty() {
        println!("  Warning: no download source configured for this model.");
        println!("  Writing manifest only (no GGUF file).");

        match store.write_manifest(entry, "metadata_only", None, None, None) {
            Ok(path) => println!("  Manifest saved to {}", path.display()),
            Err(e) => {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        }
        return entry.id.clone();
    }

    // The same file may already be here under another id: pulled as
    // `hf.co/<its repo>:<its quant>`, which is stored under a name of its own.
    let weights = models::store::HfFile::new(&entry.hf_repo, &entry.hf_filename);
    match models::pull::reuse_stored_weights(
        store,
        &weights,
        &entry.id,
        std::slice::from_ref(&entry.hf_filename),
    ) {
        Some(models::pull::AlreadyStored::Linked { from }) => {
            println!(
                "  {} is already downloaded as '{from}': linked it as '{}' instead of \
                 downloading it again (no extra disk space).",
                entry.hf_filename, entry.id
            );
            // The projector comes along when `from` has the same one; when it
            // does not, `download_mmproj` fetches it as on a fresh pull (and
            // finds it already there otherwise).
            if let (Some(repo), Some(file)) = (&entry.mmproj_repo, &entry.mmproj_filename) {
                models::pull::link_stored_projector(
                    store,
                    &from,
                    &entry.id,
                    &models::store::HfFile::new(repo, file),
                );
            }
            let mmproj = download_mmproj(entry, &store.model_path(&entry.id)).await;
            if let Err(e) = store.write_manifest(
                entry,
                "ready",
                Some(&entry.hf_filename),
                mmproj.as_deref(),
                Some(&weights),
            ) {
                // Only links were made, so undoing them loses nothing.
                let _ = store.delete(&entry.id);
                eprintln!(
                    "Error: could not write the manifest for '{}': {e}",
                    entry.id
                );
                eprintln!("The model is still available as '{from}'.");
                std::process::exit(1);
            }
            println!("  Done. Model ready.");
            println!("\nRun with: eullm run {}", entry.id);
            return entry.id.clone();
        }
        Some(models::pull::AlreadyStored::Unlinked { from, reason }) => {
            println!(
                "  {} is already downloaded as '{from}', but could not be linked as '{}' \
                 ({reason}).",
                entry.hf_filename, entry.id
            );
            println!("  Nothing was downloaded: it is the same file, use '{from}'.");
            println!("\nRun with: eullm run {from}");
            return from;
        }
        None => {}
    }

    // Download GGUF from HuggingFace
    let short_name = entry.id.as_str();
    // Before anything is fetched: the directory has to be creatable. Doing
    // this here rather than letting the downloader hit it means a local
    // problem is reported as one, instead of surfacing mid-download under a
    // message about the model not being published.
    let model_dir = match store.ensure_model_dir(&entry.id) {
        Ok(dir) => dir,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };
    let gguf_dest = model_dir.join(&entry.hf_filename);

    println!(
        "  Downloading {} from HuggingFace ({})...",
        entry.hf_filename, entry.hf_repo
    );
    println!("  Destination: {}", gguf_dest.display());
    println!("  Size: ~{}", format_bytes(entry.size_bytes));
    println!();

    let hf_repo = entry.hf_repo.clone();
    let hf_filename = entry.hf_filename.clone();
    let entry_clone = entry.clone();

    // Download directly on the current async runtime — we're already inside
    // `#[tokio::main]`, so spawning a nested `Runtime::new().block_on()` here
    // panics with "Cannot start a runtime from within a runtime".
    let result = {
        use crate::registry::{download_from_huggingface, format_progress};
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU64, Ordering};

        let last_printed = Arc::new(AtomicU64::new(0));

        let progress: registry::ProgressCallback = Box::new(move |downloaded, total| {
            let last = last_printed.load(Ordering::Relaxed);
            // Print every 10MB or at completion
            if downloaded - last > 10_000_000 || (total > 0 && downloaded >= total) {
                last_printed.store(downloaded, Ordering::Relaxed);
                eprint!("\r  {}", format_progress(downloaded, total));
                let _ = std::io::Write::flush(&mut std::io::stderr());
            }
        });

        let expected_sha256 = if entry_clone.digest.is_empty() {
            None
        } else {
            Some(entry_clone.digest.as_str())
        };
        download_from_huggingface(
            &hf_repo,
            &hf_filename,
            &gguf_dest,
            expected_sha256,
            Some(progress),
        )
        .await
    };

    eprintln!(); // newline after progress

    // Optional: download the multimodal projector (mmproj) alongside the
    // GGUF. Multimodal models in the catalog declare `mmproj_repo` and
    // `mmproj_filename`; for everyone else this is a no-op.
    let mmproj_filename_stored: Option<String> = if result.is_ok() {
        download_mmproj(&entry_clone, &model_dir).await
    } else {
        None
    };

    match result {
        Ok(()) => {
            // Write manifest with GGUF file reference (and mmproj if pulled)
            match store.write_manifest(
                &entry_clone,
                "ready",
                Some(&entry_clone.hf_filename),
                mmproj_filename_stored.as_deref(),
                Some(&weights),
            ) {
                Ok(_) => {
                    println!("  Done. Model ready.");
                    println!("\nRun with: eullm run {}", short_name);
                }
                Err(e) => {
                    eprintln!("Warning: download succeeded but manifest write failed: {e}");
                }
            }
            entry_clone.id.clone()
        }
        Err(e) => {
            eprintln!("Download failed: {e}");
            eprintln!();
            eprintln!("This may be because the model hasn't been published yet.");
            eprintln!("You can also use a local GGUF file: eullm run ./path/to/model.gguf");

            // Clean up: the partial .gguf.part file is already removed by the
            // downloader. Any empty model directory the pull created stays
            // out of `eullm list` — we explicitly remove it so a failed pull
            // leaves no trace on disk. A subsequent `eullm pull <model>` will
            // re-attempt cleanly.
            let _ = store.delete(&entry_clone.id);
            std::process::exit(1);
        }
    }
}

/// Download the multimodal projector for `entry` into `model_dir`, if the
/// catalog declares one. Returns the on-disk filename when the file ended up
/// present (either freshly downloaded or already there), `None` otherwise.
/// Non-fatal: on failure we print a warning and let the caller proceed.
async fn download_mmproj(
    entry: &models::CatalogEntry,
    model_dir: &std::path::Path,
) -> Option<String> {
    let (Some(mmproj_repo), Some(mmproj_filename)) =
        (entry.mmproj_repo.as_ref(), entry.mmproj_filename.as_ref())
    else {
        return None;
    };
    let mmproj_dest = model_dir.join(mmproj_filename);

    // Idempotency: if the file is already on disk and non-empty, just record
    // it. Lets this helper be called from both the first pull and the
    // mmproj-recovery branch without re-downloading 800+ MB.
    if mmproj_dest.is_file()
        && let Ok(meta) = std::fs::metadata(&mmproj_dest)
        && meta.len() > 0
    {
        return Some(mmproj_filename.clone());
    }

    println!();
    println!(
        "  Downloading multimodal projector {} from {}...",
        mmproj_filename, mmproj_repo
    );
    println!("  Destination: {}", mmproj_dest.display());

    use crate::registry::{download_from_huggingface, format_progress};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    let last_printed = Arc::new(AtomicU64::new(0));
    let progress: registry::ProgressCallback = Box::new(move |downloaded, total| {
        let last = last_printed.load(Ordering::Relaxed);
        if downloaded - last > 10_000_000 || (total > 0 && downloaded >= total) {
            last_printed.store(downloaded, Ordering::Relaxed);
            eprint!("\r  {}", format_progress(downloaded, total));
            let _ = std::io::Write::flush(&mut std::io::stderr());
        }
    });

    match download_from_huggingface(
        mmproj_repo,
        mmproj_filename,
        &mmproj_dest,
        None,
        Some(progress),
    )
    .await
    {
        Ok(()) => {
            eprintln!();
            println!("  mmproj ready ({}).", mmproj_filename);
            Some(mmproj_filename.clone())
        }
        Err(e) => {
            eprintln!();
            // Non-fatal: the GGUF is on disk, the model is usable in text-only
            // mode. Multimodal builds will refuse image input until the user
            // re-runs `pull` (which will retry this download) or drops the
            // file in manually.
            eprintln!("  Warning: mmproj download failed ({e}). Model will run text-only.");
            None
        }
    }
}

/// Report model directories that hold weights but that `list()` could not
/// read, so they do not vanish from the listing without explanation.
///
/// Printed after the table rather than mixed into it: these are not usable
/// models, and putting them in the same list would suggest they can be run.
fn print_unlisted(store: &ModelStore) {
    let unlisted = store.unlisted();
    if unlisted.is_empty() {
        return;
    }
    println!("\nNot listed above — weights on disk, manifest unusable:");
    for (name, reason) in &unlisted {
        println!("  {name:<24} {reason}");
    }
    println!(
        "\n  Repair with `eullm pull <name>` (re-downloads and rewrites the manifest),\n           or run the file directly: `eullm run <path-to-the-.gguf>`."
    );
}

fn cmd_list(store: &ModelStore) {
    match store.list() {
        Ok(models) if models.is_empty() => {
            let (root, source) = store.root_with_source();
            println!("No models installed in {} [{}].", root.display(), source);
            // Before the catalog: "nothing installed" is misleading when a
            // model's weights are right there and only its manifest is gone.
            print_unlisted(store);
            println!("\nAvailable models in EU catalog:");
            for entry in catalog::EU_CATALOG.iter() {
                println!(
                    "  {:<25} {:>3}GB  {}",
                    entry.id, entry.vram_gb, entry.description
                );
            }
            println!("\nPull with: eullm pull <model-name>");
            println!("Or run a local GGUF: eullm run ./path/to/model.gguf");
        }
        Ok(models) => {
            // NAME is the addressable id — exactly what you pass to `eullm run`.
            // The human-readable name is shown as a trailing description.
            let (root, source) = store.root_with_source();
            println!("Models in {} [{}]\n", root.display(), source);
            println!(
                "{:<24} {:>8} {:>6} {:<16} DESCRIPTION",
                "NAME", "SIZE", "VRAM", "STATUS"
            );
            for m in &models {
                let size = format_bytes(m.size_bytes);
                let id = if m.id.is_empty() {
                    m.name.strip_prefix("eullm/").unwrap_or(&m.name)
                } else {
                    &m.id
                };
                // Prefer the current catalog name over the manifest's frozen
                // copy, so a stale display name (e.g. an old "(text-only)" tag)
                // self-corrects without a re-pull. External models fall back to
                // the manifest name.
                let desc = catalog::find_model(id)
                    .map(|e| e.name.as_str())
                    .unwrap_or(m.name.as_str());
                // `status` is whatever was written into the manifest at pull
                // time, so it keeps saying `ready` for a model whose GGUF is
                // missing. Check the disk instead of repeating the file.
                let status = if store.is_present(id) {
                    m.status.clone()
                } else {
                    format!("{} (file missing)", m.status)
                };
                println!(
                    "{:<24} {:>8} {:>4}GB  {:<16} {}",
                    id, size, m.vram_gb, status, desc
                );
            }
            println!(
                "\nRun one with: eullm run <NAME>   (e.g. eullm run {})",
                models
                    .first()
                    .map(|m| if m.id.is_empty() {
                        m.name.as_str()
                    } else {
                        m.id.as_str()
                    })
                    .unwrap_or("<NAME>")
            );
            print_unlisted(store);
        }
        Err(e) => {
            eprintln!("Error listing models: {e}");
            std::process::exit(1);
        }
    }
}

fn cmd_show(store: &ModelStore, model: &str) {
    // First check local store
    match store.get(model) {
        Ok(Some(manifest)) => {
            // Show the addressable id and the fresh catalog name when known.
            let id = if manifest.id.is_empty() {
                model
            } else {
                &manifest.id
            };
            let display_name = catalog::find_model(id)
                .map(|e| e.name.clone())
                .unwrap_or_else(|| manifest.name.clone());
            println!("Name:        {id}");
            println!("Model:       {display_name}");
            println!("Description: {}", manifest.description);
            println!("Base:        {}", manifest.base);
            println!("Languages:   {}", manifest.languages.join(", "));
            println!("VRAM:        {}GB", manifest.vram_gb);
            println!("Size:        {}", format_bytes(manifest.size_bytes));
            println!("License:     {}", manifest.license);
            println!("Digest:      {}", manifest.digest);
            println!("Pulled:      {}", manifest.pulled_at);
            println!("Status:      {}", manifest.status);
        }
        Ok(None) => {
            // Check catalog
            if let Some(entry) = catalog::find_model(model) {
                println!("Model:       {} (not pulled)", entry.name);
                println!("Description: {}", entry.description);
                println!("Base:        {}", entry.base());
                println!("Languages:   {}", entry.languages.join(", "));
                println!("VRAM:        {}GB", entry.vram_gb);
                println!("Size:        {}", format_bytes(entry.size_bytes));
                println!("License:     {}", entry.license);
                println!("\nPull with: eullm pull {}", entry.id);
            } else {
                eprintln!("Error: model '{model}' not found.");
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("Error reading model: {e}");
            std::process::exit(1);
        }
    }
}

fn cmd_rm(store: &ModelStore, model: &str, force: bool) {
    // Resolve the model to its on-disk manifest so we can show name + size
    // in the confirmation prompt. If there's no manifest, refuse.
    let manifest = match store.get(model) {
        Ok(Some(m)) => m,
        Ok(None) => {
            eprintln!("Error: model '{model}' is not installed locally.");
            eprintln!("Run `eullm list` to see installed models.");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("Error reading model: {e}");
            std::process::exit(1);
        }
    };

    if !force {
        // Be explicit about what is going away — the confirmation has to
        // carry enough info that the user can't fat-finger it on a 45 GB
        // download they actually wanted to keep.
        print!(
            "Remove '{}' ({})? [y/N] ",
            manifest.name,
            format_bytes(manifest.size_bytes)
        );
        let _ = std::io::Write::flush(&mut std::io::stdout());
        let mut input = String::new();
        if std::io::stdin().read_line(&mut input).is_err() {
            eprintln!("Cancelled.");
            std::process::exit(1);
        }
        let answer = input.trim().to_lowercase();
        if answer != "y" && answer != "yes" {
            println!("Cancelled.");
            return;
        }
    }

    match store.delete(model) {
        // Weights a pull linked into a second id stay with that id.
        Ok(Some(removed)) if removed.shared > 0 => println!(
            "Removed '{}' ({} freed; {} stays on disk, shared with another model).",
            manifest.name,
            format_bytes(removed.freed),
            format_bytes(removed.shared)
        ),
        Ok(Some(removed)) => println!(
            "Removed '{}' ({} freed).",
            manifest.name,
            format_bytes(removed.freed)
        ),
        Ok(None) => println!("Nothing to remove (already gone)."),
        Err(e) => {
            eprintln!("Error removing model: {e}");
            std::process::exit(1);
        }
    }
}

/// Resolve a model argument to a GGUF file path.
///
/// Supports:
/// - Local GGUF file path: `./model.gguf` or `/path/to/model.gguf`
/// - Downloaded catalog model: `legal-it-7b` → `~/.eullm/models/legal-it-7b/*.gguf`
fn resolve_model_path(model: &str, store: &ModelStore) -> Option<PathBuf> {
    let path = PathBuf::from(model);

    // Direct GGUF file path
    if path.exists() && path.extension().is_some_and(|e| e == "gguf") {
        return Some(path);
    }

    // Check model store for downloaded GGUF files
    store.gguf_path(model)
}

/// `--max-loaded-models`, `--default-model` and `--auto-model`, resolved
/// against the store; exits when one of the models named is no model, as for
/// `--decision-model`, or when `--auto-model` is refused. `mmproj` is
/// `--mmproj`, which gives every model a projector.
fn residency_config(
    store: &ModelStore,
    mmproj: Option<&std::path::Path>,
    flags: api::ResidencyFlags,
) -> api::ResidencyConfig {
    api::ResidencyConfig::resolve(&flags, |arg| candidate_facts(arg, store, mmproj)).unwrap_or_else(
        |e| {
            eprintln!("Error: {e}");
            std::process::exit(1);
        },
    )
}

/// A generation model named on the command line, with what the store and
/// the catalog say it is, and whether it can read images: what
/// `--default-model` and `--auto-model` are resolved with. The store's
/// description is taken only when someone wrote one: not an external pull's
/// provenance, and not the catalog's text a catalog pull copied there.
fn candidate_facts(
    arg: &str,
    store: &ModelStore,
    mmproj: Option<&std::path::Path>,
) -> Option<api::CandidateFacts> {
    let model = named_model(arg, store)?;
    let entry = catalog::find_model(&model.name);
    let store_description = store
        .get(&model.name)
        .ok()
        .flatten()
        .and_then(|manifest| manifest.written_description().map(str::to_string))
        .filter(|text| entry.is_none_or(|entry| entry.description.trim() != text.as_str()));
    let has_projector = mmproj.is_some()
        || store.mmproj_path(&model.name).is_some()
        || models::store::mmproj_beside(&model.path).is_some();
    Some(api::CandidateFacts {
        catalog: entry.map(|entry| api::CatalogFacts {
            description: entry.description.clone(),
            params_b: entry.params_b,
            domain: entry.domain.clone(),
        }),
        model,
        store_description,
        has_projector,
    })
}

/// A probability on the command line: a number from 0 to 1.
fn parse_probability(s: &str) -> Result<f32, String> {
    match s.parse::<f32>() {
        Ok(p) if (0.0..=1.0).contains(&p) => Ok(p),
        Ok(p) => Err(format!("{p} is not between 0 and 1")),
        Err(e) => Err(e.to_string()),
    }
}

/// `--n-batch` as `--n-ubatch` leaves it (see `inference::batch_for_ubatch`),
/// said out loud when the micro-batch raised it: a flag that changes another
/// flag's value without a word is the confusion the raise exists to avoid.
fn launch_n_batch(n_batch: u32, n_ubatch: u32) -> u32 {
    let raised = inference::batch_for_ubatch(n_batch, n_ubatch);
    if raised != n_batch {
        println!(
            "[EULLM] --n-batch raised from {n_batch} to {raised} to hold --n-ubatch {n_ubatch}."
        );
    }
    raised
}

/// A generation model named on the command line, under the name requests
/// will use for it (see `launch_companion_name`), and its GGUF. An Ollama
/// tag (`qwen3:8b`) is taken as the store name it stands for.
fn named_model(arg: &str, store: &ModelStore) -> Option<api::NamedModel> {
    let normalized = arg.replace(':', "-");
    let (arg, path) = resolve_model_path(arg, store)
        .map(|path| (arg, path))
        .or_else(|| {
            resolve_model_path(&normalized, store).map(|path| (normalized.as_str(), path))
        })?;
    Some(api::NamedModel {
        name: launch_companion_name(arg, &path),
        path,
    })
}

/// The name a request would use to ask for a model loaded at launch by
/// `--embedding-model` or `--decision-model`, or named by `--default-model`:
/// what was typed for a store name, the file name for a path.
///
/// The embedder used to take its file name in both cases. A stored model is
/// asked for by its store name, as `eullm list` shows it, and its file is
/// named after the release (`Qwen3-Embedding-0.6B-Q8_0.gguf` in
/// `qwen3-embedding-0.6b-gguf-q8_0`), so the first request naming it found
/// no model of that name loaded and loaded it again — as a model no longer
/// reserved, which the next generation load could evict.
fn launch_companion_name(arg: &str, path: &std::path::Path) -> String {
    if arg.ends_with(".gguf") || arg.contains(['/', '\\']) {
        path.file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| arg.to_string())
    } else {
        arg.to_string()
    }
}

/// Build the `--embedding-model` companion's kept context now, at the size of
/// its longest input, and say what was loaded. Built before the generation
/// model is sized, it is memory already in use when `--fit` reads what is
/// free: a long input then never needs room the generation model took. When
/// it cannot be built — a card already full — the embedder still starts, and
/// its first input builds the context in whatever room is left.
fn keep_launch_embedding_context(
    model: &inference::embedding::EmbeddingModel,
    path: &std::path::Path,
    weights_bytes: u64,
) {
    match model.reserve_context() {
        Ok(()) => println!(
            "Embedding model loaded: {} ({weights_bytes} bytes, and a context for {} tokens kept \
             for every request; {} MiB kept free beside them)",
            path.display(),
            model.largest_context(),
            fit::EMBEDDING_COMPUTE_RESERVE_BYTES / (1024 * 1024)
        ),
        Err(e) => println!(
            "Embedding model loaded: {} ({weights_bytes} bytes). Its context could not be kept \
             yet ({e}): the first input builds it, in the room left then.",
            path.display()
        ),
    }
}

#[cfg(test)]
mod launch_companion_name_tests {
    use super::launch_companion_name;
    use std::path::Path;

    #[test]
    fn a_store_name_is_kept_as_typed() {
        let path = Path::new(
            "/home/u/.eullm/models/qwen3-embedding-0.6b-gguf-q8_0/Qwen3-Embedding-0.6B-Q8_0.gguf",
        );
        assert_eq!(
            launch_companion_name("qwen3-embedding-0.6b-gguf-q8_0", path),
            "qwen3-embedding-0.6b-gguf-q8_0"
        );
    }

    #[test]
    fn a_path_is_named_by_its_file() {
        let path = Path::new("/models/Qwen3-Embedding-0.6B-Q8_0.gguf");
        assert_eq!(
            launch_companion_name("/models/Qwen3-Embedding-0.6B-Q8_0.gguf", path),
            "Qwen3-Embedding-0.6B-Q8_0"
        );
        assert_eq!(
            launch_companion_name("Qwen3-Embedding-0.6B-Q8_0.gguf", path),
            "Qwen3-Embedding-0.6B-Q8_0"
        );
    }

    /// A model named on the command line goes by its store name, an Ollama
    /// tag of it included, or by its file's stem; one not there is none.
    #[test]
    fn a_named_model_is_found_under_the_name_requests_use() {
        let dir = std::env::temp_dir().join(format!("eullm-named-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(dir.join("qwen3-8b")).unwrap();
        let weights = dir.join("qwen3-8b").join("Qwen3-8B-Q4_K_M.gguf");
        std::fs::write(&weights, b"GGUF").unwrap();
        let store = super::ModelStore::at(dir.clone());
        for arg in ["qwen3-8b", "qwen3:8b"] {
            let named = super::named_model(arg, &store).expect(arg);
            assert_eq!(named.name, "qwen3-8b", "{arg}");
            assert_eq!(named.path, weights);
        }
        let by_path = super::named_model(weights.to_str().unwrap(), &store).expect("a path");
        assert_eq!(by_path.name, "Qwen3-8B-Q4_K_M");
        assert!(super::named_model("qwen3-14b", &store).is_none());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// What `--auto-model` reads about a model: the description someone
    /// wrote in its manifest, not the catalog's text a pull copied there;
    /// the catalog's facts; and a projector beside its weights.
    #[test]
    fn a_candidates_facts_come_from_its_manifest_the_catalog_and_its_directory() {
        let dir = std::env::temp_dir().join(format!("eullm-facts-{}", uuid::Uuid::new_v4()));
        let catalog = super::catalog::find_model("qwen3-8b").expect("in the catalog");
        let stored = |id: &str, description: &str, projector: bool| {
            std::fs::create_dir_all(dir.join(id)).unwrap();
            std::fs::write(dir.join(id).join("model.gguf"), b"GGUF").unwrap();
            if projector {
                std::fs::write(dir.join(id).join("mmproj-F16.gguf"), b"GGUF").unwrap();
            }
            let manifest = serde_json::json!({
                "id": id, "name": id, "description": description, "languages": [],
                "base": "x", "vram_gb": 0, "size_bytes": 0, "license": "MIT", "digest": "",
                "pulled_at": "2026-10-01T00:00:00Z", "status": "ready", "gguf_file": "model.gguf",
            });
            std::fs::write(dir.join(id).join("manifest.json"), manifest.to_string()).unwrap();
        };
        stored("qwen3-8b", &catalog.description, false);
        stored("mine", "Italian contracts and case law", true);
        stored("pulled", "External model pulled from hf.co/x/y", false);
        let store = super::ModelStore::at(dir.clone());
        let facts = |id| super::candidate_facts(id, &store, None).expect(id);

        let qwen = facts("qwen3-8b");
        assert_eq!(qwen.store_description, None, "the catalog's own text");
        assert_eq!(qwen.catalog.map(|c| c.params_b), Some(catalog.params_b));
        let mine = facts("mine");
        assert_eq!(
            mine.store_description.as_deref(),
            Some("Italian contracts and case law")
        );
        assert!(mine.has_projector && !qwen.has_projector);
        assert_eq!(facts("pulled").store_description, None, "only provenance");
        let projector = std::path::Path::new("/models/mmproj.gguf");
        assert!(
            super::candidate_facts("qwen3-8b", &store, Some(projector))
                .unwrap()
                .has_projector,
            "--mmproj gives every model one"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

/// Check what service is running on a given port.
async fn detect_port_service(port: u16) -> Option<String> {
    use tokio::net::TcpStream;

    let addr = format!("127.0.0.1:{port}");
    if TcpStream::connect(&addr).await.is_err() {
        return None;
    }

    let url = format!("http://127.0.0.1:{port}/api/version");
    if let Ok(resp) = reqwest::get(&url).await
        && let Ok(body) = resp.text().await
        && body.contains("version")
    {
        if body.contains("eullm") {
            return Some("eullm (already running)".into());
        }
        return Some(format!("another service (response: {body})"));
    }

    Some("unknown service".into())
}

/// Ensure the port is available, or exit with a helpful message.
async fn ensure_port_available(port: u16, replace: bool) {
    if let Some(service) = detect_port_service(port).await {
        if replace {
            eprintln!("Port {port} is in use by {service}.");
            eprintln!("Attempting to take over...");
            eprintln!("Error: --replace is not yet implemented. Stop the service manually.");
            std::process::exit(1);
        } else {
            eprintln!("Error: port {port} is already in use by {service}.");
            eprintln!();
            eprintln!("Options:");
            eprintln!("  1. Stop the existing service on port {port}");
            eprintln!(
                "  2. Use a different port:  eullm serve --port {}",
                port + 1
            );
            std::process::exit(1);
        }
    }
}

/// Open `url` in the user's default browser, cross-platform. Fire-and-forget:
/// spawns the OS handler and returns immediately (the engine keeps running).
fn open_browser(url: &str) -> std::io::Result<()> {
    use std::process::Command;
    #[cfg(target_os = "windows")]
    let mut cmd = {
        // `start` is a cmd builtin; the empty "" is the window-title argument.
        let mut c = Command::new("cmd");
        c.args(["/C", "start", "", url]);
        c
    };
    #[cfg(target_os = "macos")]
    let mut cmd = {
        let mut c = Command::new("open");
        c.arg(url);
        c
    };
    #[cfg(all(unix, not(target_os = "macos")))]
    let mut cmd = {
        let mut c = Command::new("xdg-open");
        c.arg(url);
        c
    };
    // The handler's own diagnostics are not ours to print. On a machine with
    // no graphical browser, xdg-open walks its fallback list and reports each
    // miss, so the engine's startup ended in seven `command not found` lines
    // followed by `no method available` — noise that reads like the engine
    // failing, right after the banner said it was ready. Reported from a real
    // session (issue #286). The spawn result still decides which of the two
    // messages the caller prints, so a failure is still visible, in one line
    // and in our own words.
    cmd.stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map(|_| ())
}

/// Valid launch values for `--batch-size`.
///
/// 0 is sequential mode (also forced for multimodal models); 1..=64 are the
/// scheduler slots, the same ceiling request overrides already enforce.
/// Anything past that truncates through `as u32` into values that panic
/// divisions (`ctx_size / batch_size`) or size absurd queues, so refuse up
/// front instead of crashing after the model loaded.
fn validate_launch_batch_size(n: usize) -> Result<usize, String> {
    const MAX_LAUNCH_BATCH_SIZE: usize = 64;
    if n <= MAX_LAUNCH_BATCH_SIZE {
        Ok(n)
    } else {
        Err(format!(
            "--batch-size must be between 0 and {MAX_LAUNCH_BATCH_SIZE} (0 = sequential), got {n}"
        ))
    }
}

#[cfg(test)]
mod launch_batch_size_tests {
    use super::validate_launch_batch_size;

    #[test]
    fn sequential_and_normal_sizes_pass() {
        for n in [0, 1, 8, 64] {
            assert_eq!(validate_launch_batch_size(n), Ok(n));
        }
    }

    #[test]
    fn absurd_sizes_fail_including_u32_truncation() {
        // 2^32 is the sharp one: `as u32` turns it into zero slots, which
        // panics the per-sequence division at startup.
        for n in [65, 1_000_000, 4_294_967_296usize, usize::MAX] {
            assert!(validate_launch_batch_size(n).is_err(), "accepted {n}");
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn cmd_run(
    store: &ModelStore,
    model: &str,
    port: u16,
    replace: bool,
    gpu_layers: i32,
    fit: bool,
    // True only when the user typed `--fit`. Automatic sizing (the
    // default) must never block on a prompt: see `run_fit_headless`.
    fit_explicit: bool,
    fit_strict: bool,
    cpu_moe: bool,
    n_cpu_moe: u32,
    rs_seq: u32,
    mtp: u32,
    mtp_p_min: f32,
    mtp_model: Option<PathBuf>,
    moe_cache: Option<fit::MoeCache>,
    no_mmap: bool,
    mmap: bool,
    moe_prefetch: u32,
    load_threads: readahead::LoadThreads,
    kv_unified: bool,
    ctx_checkpoints: usize,
    checkpoint_min_step: u32,
    mut ctx_size: u32,
    threads: Option<u32>,
    batch_size: usize,
    flash_attn: bool,
    n_batch: u32,
    n_ubatch: Option<u32>,
    cache_type_k: inference::KvCacheType,
    cache_type_v: inference::KvCacheType,
    web: bool,
    ui_port: Option<u16>,
    open_chat: bool,
    image: Option<PathBuf>,
    rust_debug: bool,
    mmproj: Option<PathBuf>,
    mmproj_offload: Option<bool>,
    keep_alive: Option<std::time::Duration>,
    embedding_model: Option<String>,
    decision_model: Option<String>,
    decision_ctx: u32,
    residency: api::ResidencyConfig,
) {
    let batch_size = validate_launch_batch_size(batch_size).unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        std::process::exit(1);
    });
    // `--image` is a one-shot multimodal probe: load, send the bytes + prompt,
    // print the output, exit. Forces sequential mode (the scheduler does not
    // yet route media) and skips port binding because we won't serve an API.
    let multimodal_oneshot = image.is_some();
    let batch_size = if multimodal_oneshot { 0 } else { batch_size };
    // What the API server is told, kept separate from the value this
    // launch resolves for its own model (see the multimodal fallback below).
    let launch_batch_size = batch_size;
    let ui_port = if multimodal_oneshot { None } else { ui_port };

    // `--fit` may override this below once the GGUF file is resolved; until
    // then it is exactly the user-provided `--gpu-layers`.
    let mut gpu_layers = gpu_layers;
    // The MoE auto-sizing step (below, alongside `--fit`) may override these
    // too — only when the user hasn't already chosen one explicitly.
    let mut cpu_moe = cpu_moe;
    let mut n_cpu_moe = n_cpu_moe;
    // The expert cache `--moe-cache` comes to for the launch model, once sized.
    let mut moe_cache_bytes: u64 = 0;
    // The slots of `--moe-prefetch` the launch model gets, once it is known
    // whether its experts are in RAM and pinned.
    let mut moe_prefetch_slots: u32 = 0;
    // `--n-batch`, `--n-ubatch` and `--no-mmap` as given, for the API server,
    // which sizes every model it loads from them; the launch model's own may
    // change below with an expert cache (`fit::MOE_CACHE_N_UBATCH`,
    // `fit::plan_read_into_memory`).
    let (n_batch_flag, n_ubatch_flag, no_mmap_flag) = (n_batch, n_ubatch, no_mmap);
    let mut n_batch = n_batch;
    let mut n_ubatch = n_ubatch.unwrap_or(inference::DEFAULT_N_UBATCH);
    let mut no_mmap = no_mmap;
    // What the user actually asked for, before --fit specializes the mut
    // bindings above to the LAUNCH model. The API server must inherit these
    // originals: with --fit it re-sizes each model it loads against them,
    // and handing it the launch model's computed split instead is exactly
    // the bug where a dense 27B's 43/64 split was reused to load a 22 GB
    // MoE and OOM'd.
    let flag_gpu_layers = gpu_layers;
    let flag_cpu_moe = cpu_moe;
    let flag_n_cpu_moe = n_cpu_moe;

    if !multimodal_oneshot {
        ensure_port_available(port, replace).await;
        if let Some(p) = ui_port {
            ensure_port_available(p, replace).await;
        }
    }

    let model_name: String;
    // The resolved GGUF path, kept for the API's launch-model allowance (see
    // `api::AppState::launch_model`) because `gguf_path` itself is moved into
    // the loader.
    let launch_gguf_path: Option<PathBuf>;
    // Carried out of the load block so the API can keep it as the fallback
    // projector for a later model swap.
    let mut api_mmproj: Option<PathBuf> = None;
    let mut engine: Option<Arc<InferenceEngine>> = None;
    let mut scheduler: Option<inference::SchedulerHandle> = None;
    let mut n_ctx_train: u32 = 0;
    let mut kv_k_mib: f64 = 0.0;
    let mut kv_v_mib: f64 = 0.0;

    let resolved_threads = threads.unwrap_or_else(inference::default_thread_count);

    // The one `LlamaBackend` this process will ever create — shared by the
    // embedding model below and by the generation model further down (and
    // by any later swap of either). See `inference::init_shared_backend`
    // for why: two independently-initialized backends can never coexist in
    // one process, which used to make bge and a loaded chat model fail on
    // the first request that tried to use both.
    let backend = inference::init_shared_backend().unwrap_or_else(|e| {
        eprintln!("Error initializing llama.cpp backend: {e}");
        std::process::exit(1);
    });

    // --embedding-model: load the companion embedding model now, before the
    // generation model's own --fit sizing runs below, and reserve its VRAM
    // off the top — see the flag's doc comment on `RuntimeOpts` for the
    // rationale. `embedding_reserve_bytes` feeds the fit calls further down;
    // `launch_embedding` becomes `ServeConfig.launch_embedding`.
    let mut launch_embedding: Option<api::EmbeddingSlot> = None;
    let mut embedding_reserve_bytes: u64 = 0;
    if let Some(ref emb_arg) = embedding_model {
        let emb_path = resolve_model_path(emb_arg, store).unwrap_or_else(|| {
            eprintln!("Error: embedding model '{emb_arg}' not found.");
            std::process::exit(1);
        });
        match inference::embedding::EmbeddingModel::load(
            &emb_path,
            resolved_threads,
            inference::embedding::DEFAULT_EMBEDDING_CTX,
            backend.clone(),
        ) {
            Ok(model) => {
                let weights_bytes = std::fs::metadata(&emb_path).map(|m| m.len()).unwrap_or(0);
                let emb_name = launch_companion_name(emb_arg, &emb_path);
                keep_launch_embedding_context(&model, &emb_path, weights_bytes);
                // Only a margin is reserved here — the model and its kept
                // context are already built above, so they already show up
                // as used VRAM in the free-VRAM figure `--fit` reads next;
                // reserving them again would subtract the embedder's
                // footprint twice. See `AppState::reserved_embedding_bytes`
                // for the full rationale (the same reservation runs again on
                // every later generation-model swap).
                embedding_reserve_bytes = fit::EMBEDDING_COMPUTE_RESERVE_BYTES;
                launch_embedding = Some(api::EmbeddingSlot {
                    model_name: emb_name,
                    model: Arc::new(model),
                    is_reserved_companion: true,
                });
            }
            Err(e) => {
                eprintln!("Error loading embedding model: {e}");
                std::process::exit(1);
            }
        }

        // If reserving the embedder's space would leave the generation
        // model no VRAM headroom at all, proceed anyway: drop the
        // reservation and let the embedder fall back to the ordinary
        // evict-on-generation-load path, exactly as if this flag had not
        // been given, rather than refusing to start over a sizing decision
        // automatic sizing already makes gracefully everywhere else.
        if let Some((free, total)) = fit::vram_bytes() {
            let floor = (total as f64 * fit::MIN_FREE_TOTAL_RATIO) as u64;
            let usable_before_reserve = free.saturating_sub(floor);
            if embedding_reserve_bytes >= usable_before_reserve {
                println!(
                    "[EULLM] Warning: reserving {} MiB for the embedding model would leave the \
                     generation model no VRAM headroom. Proceeding without the reservation — \
                     the embedding model stays loaded but will be evicted like any on-demand \
                     one when the generation model is sized.",
                    embedding_reserve_bytes / (1024 * 1024)
                );
                if let Some(ref mut slot) = launch_embedding {
                    slot.is_reserved_companion = false;
                }
                embedding_reserve_bytes = 0;
            }
        }
    }

    // --decision-model: the same treatment for the /v1/systemone slot. What
    // it reserves is a request's whole context (KV cache at --decision-ctx
    // plus a compute buffer), which does not exist yet at launch and is
    // released before a generation model is sized later — see
    // `fit::decision_reserve_bytes`.
    let mut launch_decision: Option<api::DecisionSlot> = None;
    let mut decision_reserve_bytes: u64 = 0;
    if let Some(ref arg) = decision_model {
        let slot = load_launch_decision(
            arg,
            store,
            resolved_threads,
            decision_ctx,
            flash_attn,
            backend.clone(),
        );
        decision_reserve_bytes = slot.reserve_bytes;
        launch_decision = Some(slot);
        // Same fallback as the embedding companion above, counting both.
        if let Some((free, total)) = fit::vram_bytes() {
            let floor = (total as f64 * fit::MIN_FREE_TOTAL_RATIO) as u64;
            let usable_before_reserve = free.saturating_sub(floor);
            if embedding_reserve_bytes.saturating_add(decision_reserve_bytes)
                >= usable_before_reserve
            {
                println!(
                    "[EULLM] Warning: reserving {} MiB for the decision model would leave the \
                     generation model no VRAM headroom. Proceeding without the reservation — \
                     the decision model stays loaded but will be evicted like any on-demand \
                     one when the generation model is sized.",
                    decision_reserve_bytes / (1024 * 1024)
                );
                if let Some(ref mut slot) = launch_decision {
                    slot.is_reserved_companion = false;
                }
                decision_reserve_bytes = 0;
            }
        }
    }
    // Everything a companion model needs kept free when the generation
    // model is sized below.
    let companion_reserve_bytes = embedding_reserve_bytes.saturating_add(decision_reserve_bytes);

    // Canonical, addressable name shown in the banner and the API model slot —
    // the same string the user types into `eullm run` and sees in `eullm list`
    // and the picker. For a catalog/store model that's the id; for a direct
    // .gguf path it's the file stem (the only sensible name); for a URL it's
    // the derived id. This deliberately does NOT use the GGUF file stem for
    // store models, so `gemma-4-12b` stays `gemma-4-12b` everywhere instead of
    // surfacing as `gemma-4-12b-it-Q4_K_M`.
    let canonical_name: String = if is_url(model) {
        url_to_model_id(model).0
    } else if let Some(hf) = registry::parse_hf_ref(model) {
        hf_ref_to_model_id(&hf)
    } else {
        let p = PathBuf::from(model);
        if p.exists() && p.extension().is_some_and(|e| e == "gguf") {
            p.file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| model.to_string())
        } else {
            model.strip_prefix("eullm/").unwrap_or(model).to_string()
        }
    };

    // Set when a bare HuggingFace repo ref resolves to a quant already on
    // disk, whose store id differs from the bare-repo id (see below), and
    // when the pull found the file already stored under another id that it
    // could not link (`models::pull::reuse_stored_weights`).
    let mut return_hf_id: Option<String> = None;
    // Try to resolve as a local GGUF file or downloaded model
    let gguf_path = if is_url(model) {
        // Direct URL: pull into the store (if not already there), then load
        // by the derived id.
        let (id, _) = url_to_model_id(model);
        if store.gguf_path(&id).is_none() {
            println!("Model not found locally. Pulling from URL...");
            cmd_pull_url(store, model).await;
        }
        store.gguf_path(&id)
    } else if let Some(hf) = registry::parse_hf_ref(model) {
        // HuggingFace shorthand: pull into the store (if not already there),
        // then load by the derived id.
        let id = hf_ref_to_model_id(&hf);
        if store.gguf_path(&id).is_none() {
            // A ref with no `:quant` asks for "this repo", and quants of it
            // may already be on disk under their own ids. Downloading
            // another copy of a file the user already has is the wrong
            // default (#345): use what is there, and say which one, since
            // the choice belongs to the user when several exist.
            let local = if hf.quant.is_none() {
                local_quants_of_repo(store, &hf)
            } else {
                Vec::new()
            };
            match local.as_slice() {
                [] => {
                    println!("Model not found locally. Pulling from HuggingFace...");
                    let stored = cmd_pull_hf(store, &hf).await;
                    if stored != id {
                        return_hf_id = Some(stored);
                    }
                }
                [only] => {
                    println!("Using the copy already downloaded: {only}");
                    return_hf_id = Some(only.clone());
                }
                many => {
                    eprintln!(
                        "Error: {} quants of this repo are already downloaded. Name the one to run:",
                        many.len()
                    );
                    for id in many {
                        eprintln!("  eullm run {id}");
                    }
                    std::process::exit(1);
                }
            }
        }
        store.gguf_path(return_hf_id.as_deref().unwrap_or(&id))
    } else if let Some(path) = resolve_model_path(model, store) {
        Some(path)
    } else {
        // Catalog model — try to pull if not available, then load GGUF
        // from wherever the pull put it: the id asked for, or another id that
        // already had the same file (see `cmd_pull`).
        let mut stored_as: Option<String> = None;
        if !store.exists(model) {
            if catalog::find_model(model).is_some() {
                println!("Model not found locally. Pulling...");
                stored_as = Some(cmd_pull(store, model).await);
            } else {
                eprintln!("Error: model '{model}' not found.");
                eprintln!();
                eprintln!("Usage:");
                eprintln!("  eullm run ./path/to/model.gguf         # Run a local GGUF file");
                eprintln!("  eullm run https://host/model.gguf      # Run any GGUF by URL");
                eprintln!("  eullm run hf.co/owner/repo[:quant]     # Run from HuggingFace");
                eprintln!("  eullm run legal-it-7b                  # Run a catalog model");
                std::process::exit(1);
            }
        }
        store.gguf_path(stored_as.as_deref().unwrap_or(model))
    };

    // The batch size actually used, which is not always the one asked for: a
    // multimodal model forces the sequential engine below. This has to outlive
    // the block, because both the startup banner and `ServeConfig` are built
    // outside it — and until it did, a multimodal model printed
    // "Multimodal model — falling back to sequential mode (batch_size=0)" and
    // then a banner saying "continuous batching", while the API server was told
    // it had a scheduler it did not have.
    let mut batch_size = batch_size;

    if let Some(gguf_path) = gguf_path {
        model_name = canonical_name.clone();
        launch_gguf_path = Some(gguf_path.clone());

        println!("Loading GGUF: {}", gguf_path.display());

        // Look up an mmproj projector for this model (if any was pulled
        // alongside the GGUF). On text-only builds the value is read but
        // ignored at InferenceConfig level; on multimodal builds it is
        // what enables `generate_multimodal`.
        // Order of precedence, most explicit first: what the user named, what
        // the store recorded when the model was pulled, and finally a
        // projector sitting next to the weights — the layout of every
        // HuggingFace vision repo, and the case that used to be unreachable.
        //
        // Resolved here, ahead of sizing, and not where it used to be: the
        // projector loads with the model every time, and sizing that has not
        // heard of it hands its VRAM to text layers, which the context probe
        // then finds missing.
        let mmproj_for_config = mmproj
            .clone()
            .or_else(|| store.mmproj_path(&model_name))
            .or_else(|| store.mmproj_path(model))
            .or_else(|| crate::models::store::mmproj_beside(&gguf_path));
        let mmproj_bytes = fit::mmproj_footprint_bytes(mmproj_for_config.as_deref());
        let mut mmproj_placement = fit::MmprojPlacement::from_flag(mmproj_offload);

        // --fit: auto-size the GPU offload to free VRAM before loading. Opt-in;
        // headless-safe (never prompts unless both stdin and stdout are TTYs).
        //
        // MoE auto-sizing (roadmap 0.7-E) decides FIRST: a MoE decision
        // always resolves to a loadable configuration (expert offload, in
        // the worst case combined with a reduced layer split down to
        // fully-CPU), so there is no "doesn't fit, continue anyway?" left to
        // ask — and asking it with the dense whole-layer numbers, as an
        // earlier ordering did, describes a split that MoE sizing is about
        // to override anyway. Only when the model is not MoE (or the user
        // already chose --cpu-moe/--n-cpu-moe themselves — explicit intent
        // beats a guess) does the dense run_fit flow, with its prompt and
        // its --fit-strict handling, take over.
        if fit {
            // What the user asked for with --gpu-layers, kept as an upper
            // bound: sizing may lower it, never raise it.
            let ceiling = gpu_layers;
            let kv_bpe_k = inference::cache_type_bytes_per_elem(&cache_type_k);
            let kv_bpe_v = inference::cache_type_bytes_per_elem(&cache_type_v);
            if mmproj_offload.is_none() {
                mmproj_placement = fit::decide_mmproj_placement(
                    &gguf_path,
                    mmproj_bytes,
                    ctx_size,
                    kv_bpe_k,
                    kv_bpe_v,
                    companion_reserve_bytes.saturating_add(fit::ubatch_reserve_bytes(n_ubatch)),
                );
            }
            // The MTP head's context, built once the model has loaded (see
            // `fit::mtp_reserve_bytes`): only where it will draft, on the
            // scheduler's one slot, which a model with a projector never gets.
            let mtp_reserve = if mtp > 0 && batch_size == 1 && mmproj_for_config.is_none() {
                match mtp_model.as_deref() {
                    // The head's own file: loaded whole onto the GPU.
                    Some(file) => fit::mtp_file_reserve_bytes(
                        std::fs::metadata(file).map_or(0, |m| m.len()),
                        fit::read_gguf_info(&gguf_path).as_ref(),
                        ctx_size,
                        kv_bpe_k,
                        kv_bpe_v,
                        n_ubatch,
                    ),
                    None => fit::mtp_reserve_bytes(
                        fit::read_gguf_info(&gguf_path).as_ref(),
                        ctx_size,
                        kv_bpe_k,
                        kv_bpe_v,
                        n_ubatch,
                    ),
                }
            } else {
                0
            };
            // Everything already spoken for before the text model is sized:
            // reserved embedding and decision companions, the projector
            // unless it is going to RAM, and the MTP head's context.
            let sizing_reserve = companion_reserve_bytes
                .saturating_add(mmproj_placement.reserve(mmproj_bytes))
                .saturating_add(fit::ubatch_reserve_bytes(n_ubatch))
                .saturating_add(mtp_reserve);
            // An expert cache, asked for and with room for one, places the
            // experts itself: all in RAM, the VRAM they leave to the cache.
            // Otherwise the usual MoE sizing below decides, as before.
            let cache = moe_cache.and_then(|request| match fit::moe_cache_support() {
                Ok(()) => fit::run_moe_cache(
                    &gguf_path,
                    ctx_size,
                    kv_bpe_k,
                    kv_bpe_v,
                    sizing_reserve,
                    request,
                    cpu_moe,
                    n_cpu_moe,
                    n_ubatch_flag.is_none(),
                    // The cache runs on one CUDA GPU, which is all the
                    // prefetch needs besides pinned experts.
                    fit::MoePrefetch {
                        slots: moe_prefetch,
                        no_mmap,
                        keep_mapped: mmap,
                        ram_total: fit::system_ram_bytes(),
                    },
                ),
                Err(why) => {
                    println!("[EULLM] --moe-cache: {why}; running without the cache.");
                    None
                }
            });
            if let Some(cache) = cache {
                println!(
                    "[EULLM] MoE model: expert tensors in CPU RAM, {} of VRAM caching the \
                     ones it uses most{}{}.",
                    fit::gib(cache.bytes),
                    cache
                        .asked_bytes
                        .map(|asked| format!(
                            " (--moe-cache asked for {}, that is what was left)",
                            fit::gib(asked)
                        ))
                        .unwrap_or_default(),
                    fit::prefetch_room(cache.prefetch_bytes)
                );
                cpu_moe = cache.cpu_moe;
                n_cpu_moe = cache.n_cpu_moe;
                gpu_layers = -1;
                moe_cache_bytes = cache.bytes;
                if let Some(larger) = cache.n_ubatch {
                    n_ubatch = larger;
                    n_batch = inference::batch_for_ubatch(n_batch, larger);
                    println!(
                        "[EULLM] --moe-cache: reading prompts {larger} tokens at a time \
                         (--n-ubatch), so that the experts in RAM are copied to the GPU once \
                         per {larger} prompt tokens instead of {}.",
                        inference::DEFAULT_N_UBATCH
                    );
                }
                let (read_in, why) = fit::plan_read_into_memory(
                    no_mmap,
                    mmap,
                    cache.host_bytes,
                    fit::system_ram_bytes(),
                );
                if let Some(why) = why {
                    println!("[EULLM] {why}.");
                }
                no_mmap = read_in;
            } else {
                let moe_decision = if !cpu_moe && n_cpu_moe == 0 {
                    fit::run_moe_fit(&gguf_path, ctx_size, kv_bpe_k, kv_bpe_v, sizing_reserve)
                } else {
                    fit::MoeFitDecision::NotMoe
                };
                match moe_decision {
                    fit::MoeFitDecision::Proceed { n_cpu_moe: computed } if computed > 0 => {
                        println!(
                            "[EULLM] MoE model: keeping expert tensors on CPU RAM for the \
                             first {computed} layers so the rest fits in VRAM."
                        );
                        n_cpu_moe = computed;
                        gpu_layers = -1;
                    }
                    fit::MoeFitDecision::ProceedCpuMoeAndPartial { gpu_layers: gl } => {
                        println!(
                            "[EULLM] MoE model: even with every expert tensor on CPU RAM, the \
                             rest doesn't fit fully — offloading a reduced layer split too."
                        );
                        cpu_moe = true;
                        gpu_layers = gl;
                    }
                    // Dense model, a MoE that already fits fully as-is
                    // (computed == 0), or an unreadable layout.
                    // Only a `--fit` the user typed may stop and ask; automatic
                    // sizing applies the split and logs it, because a default
                    // that interrupts every launch of a model too big for the
                    // card is its own kind of failure.
                    _ => match if fit_explicit {
                        fit::run_fit(
                            &gguf_path,
                            gpu_layers,
                            ctx_size,
                            fit_strict,
                            kv_bpe_k,
                            kv_bpe_v,
                            sizing_reserve,
                        )
                    } else {
                        fit::run_fit_headless(
                            &gguf_path,
                            gpu_layers,
                            ctx_size,
                            fit_strict,
                            kv_bpe_k,
                            kv_bpe_v,
                            sizing_reserve,
                        )
                    } {
                        fit::FitOutcome::Proceed(n) => gpu_layers = n,
                        fit::FitOutcome::Abort => {
                            // Clean return: don't load, don't bind a port. If we
                            // were invoked from the picker flow, the user lands
                            // back there.
                            return;
                        }
                    },
                }
            }

            let capped = fit::apply_gpu_layers_ceiling(gpu_layers, ceiling);
            if capped != gpu_layers {
                println!(
                    "[EULLM] --gpu-layers {ceiling}: offloading {capped} layers instead of the                      {} that would fit. Drop the flag to use the whole card, or --no-fit to                      force a count past the estimate.",
                    if gpu_layers < 0 {
                        "all".to_string()
                    } else {
                        gpu_layers.to_string()
                    }
                );
                gpu_layers = capped;
            }
        } else if let Some(request) = moe_cache {
            // Without sizing, a size given in MiB is used as given and the
            // experts go where the user's own flags put them.
            match (request, fit::moe_cache_support()) {
                (fit::MoeCache::Mib(mib), Ok(())) => moe_cache_bytes = u64::from(mib) << 20,
                (fit::MoeCache::Auto, Ok(())) => println!(
                    "[EULLM] --moe-cache auto needs --fit to size the cache; running without \
                     it. Give a size in MiB to use one with --no-fit."
                ),
                (_, Err(why)) => {
                    println!("[EULLM] --moe-cache: {why}; running without the cache.")
                }
            }
        }

        // Multimodal MVP: --image requires the `multimodal` feature build.
        // Refuse the flag upfront on text-only builds with an actionable
        // error so the user is not left guessing why the file was ignored.
        #[cfg(not(feature = "multimodal"))]
        if image.is_some() {
            eprintln!(
                "Error: --image requires a multimodal engine build. \
                 Rebuild with --features multimodal, or use the beta binary."
            );
            std::process::exit(2);
        }

        if let Some(ref p) = mmproj_for_config {
            println!("Found mmproj: {}", p.display());
            println!("  {}", mmproj_placement.describe());
        }
        // Only an explicit `--mmproj` becomes the server's fallback for
        // later swaps. A projector discovered for THIS model belongs to
        // this model: handing it to the next one produced
        // "mismatch between text model (n_embd = 2048) and mmproj
        // (n_embd = 2560)" and a failed load, after launching a vision
        // model and switching to anything else from the chat UI. Every
        // model that has its own projector still finds it in load_generation_model,
        // by store entry or by the file sitting beside its weights.
        api_mmproj = mmproj.clone();

        // A model that goes to the GPUs whole and is more than half the
        // memory is read in: mapped, it does not load (`fit::read_in_whole_on_gpu`).
        if !no_mmap {
            let whole_on_gpu = fit::vram_bytes().is_some_and(|(_, total)| total > 0)
                && fit::every_layer_on_gpu(
                    gpu_layers,
                    fit::read_gguf_info(&gguf_path).map(|info| info.n_layers),
                    cpu_moe,
                    n_cpu_moe,
                    moe_cache_bytes,
                );
            if let Some(why) = fit::read_in_whole_on_gpu(
                fit::model_file_bytes(&gguf_path),
                whole_on_gpu,
                readahead::memory_for_cache(),
                mmap,
            ) {
                println!("[EULLM] {why}.");
                no_mmap = true;
            }
        }

        // The prefetch's slots, where experts are kept in RAM and pinned, on
        // the one CUDA GPU it runs on (the expert cache's own condition).
        moe_prefetch_slots = if fit::moe_cache_support().is_ok() {
            fit::prefetch_slots(moe_prefetch, no_mmap, cpu_moe, n_cpu_moe)
        } else {
            0
        };
        if moe_prefetch_slots > 0 {
            println!(
                "[EULLM] --moe-prefetch: the experts in RAM of a long prompt are copied to the \
                 GPU ahead of their layer, into {moe_prefetch_slots} slots of VRAM \
                 (--moe-prefetch 0 turns it off)."
            );
        }
        let config = InferenceConfig {
            model_path: gguf_path,
            gpu_layers,
            context_size: ctx_size,
            threads: resolved_threads,
            flash_attn,
            n_batch,
            n_ubatch,
            cache_type_k,
            cache_type_v,
            mmproj_path: mmproj_for_config.clone(),
            mmproj_on_gpu: mmproj_placement.on_gpu(),
            cpu_moe,
            n_cpu_moe,
            rs_seq,
            mtp,
            mtp_p_min,
            mtp_model: mtp_model.clone(),
            moe_cache_bytes,
            no_mmap,
            moe_prefetch_slots,
            load_threads,
            kv_unified,
        };

        // The continuous-batching scheduler is text-only; multimodal models
        // must be served by the sequential `InferenceEngine` so that
        // `/api/chat` requests carrying `images` reach `generate_multimodal`.
        // Force batch_size=0 when an mmproj is present (vision is single-user
        // interactive — losing batching here is not a practical regression).
        if mmproj_for_config.is_some() {
            if batch_size > 0 {
                println!("Multimodal model — falling back to sequential mode (batch_size=0).");
            }
            batch_size = 0;
        }
        // …for THIS model. The server keeps the batch size the user asked
        // for, so a later swap to a text-only model gets the scheduler back:
        // `load_generation_model` re-applies the same sequential fallback for whatever
        // model actually carries a projector. Passing the zeroed value on
        // pinned every subsequent model to sequential mode — the same shape
        // of bug as handing the launch model's projector to its successors.

        if batch_size > 0 {
            // ── Continuous batching mode ────────────────────────────
            let sched_config = SchedulerConfig {
                max_batch_size: batch_size,
                queue_capacity: batch_size * 8,
                ctx_checkpoints,
                checkpoint_min_step,
                debug_logit_check: rust_debug,
            };
            let sched = BatchScheduler::new(config, sched_config);
            match sched.start(backend.clone()) {
                Ok((handle, model_info)) => {
                    n_ctx_train = model_info.n_ctx_train;
                    kv_k_mib = model_info.kv_k_mib;
                    kv_v_mib = model_info.kv_v_mib;
                    scheduler = Some(handle);
                    println!("Model loaded (continuous batching, max_batch_size={batch_size}).");
                }
                Err(e) => {
                    eprintln!("Error starting scheduler: {e}");
                    std::process::exit(1);
                }
            }
        } else {
            // ── Sequential mode ────────────────────────────────────
            match InferenceEngine::load(config, backend.clone()) {
                Ok(eng) => {
                    let info = eng.ready_info();
                    n_ctx_train = info.n_ctx_train;
                    kv_k_mib = info.kv_k_mib;
                    kv_v_mib = info.kv_v_mib;
                    // May be smaller than what was passed in: `load()` shrinks
                    // it automatically when the requested size does not fit
                    // (see `InferenceEngine::probe_and_shrink_context`). The
                    // banner has to show what actually loaded, not what was
                    // asked for — otherwise it states a KV cost that belongs
                    // to a different context size than the one printed next
                    // to it.
                    ctx_size = eng.context_size();
                    engine = Some(Arc::new(eng));
                    println!("Model loaded (sequential mode).");
                }
                Err(e) => {
                    eprintln!("Error loading model: {e}");
                    std::process::exit(1);
                }
            }
        }
    } else {
        model_name = canonical_name.clone();
        launch_gguf_path = None;
        eprintln!("Warning: no GGUF file available for this model.");
        eprintln!("  The model may not have been published yet.");
        eprintln!("  API will start but inference requests will return 503.");
        eprintln!("  To test inference, use a local GGUF file:");
        eprintln!("    eullm run ./path/to/model.gguf");
        eprintln!();
    }

    let short = model_name.strip_prefix("eullm/").unwrap_or(&model_name);

    println!();
    println!(
        "eullm ready.  [v{} {}]",
        env!("CARGO_PKG_VERSION"),
        env!("EULLM_GIT_HASH")
    );
    println!("  API (EULLM):   http://localhost:{port}/api");
    println!("  API (OpenAI):  http://localhost:{port}/v1");
    if let Some(p) = ui_port {
        println!("  Chat UI:       http://localhost:{p}/");
    }
    if engine.is_some() || scheduler.is_some() {
        crate::banner::ModelBanner {
            model_name: short.to_string(),
            gpu_layers,
            cpu_moe,
            n_cpu_moe,
            rs_seq,
            mtp,
            mtp_p_min,
            mtp_model: mtp_model.clone(),
            moe_cache_bytes,
            no_mmap,
            moe_prefetch_slots,
            ctx_checkpoints,
            checkpoint_min_step,
            batch_size,
            ctx_size,
            n_ctx_train,
            flash_attn,
            cache_type_k,
            cache_type_v,
            kv_k_mib,
            kv_v_mib,
            web,
            threads: resolved_threads,
            n_batch,
            n_ubatch,
            rust_debug,
        }
        .print();
    } else {
        println!("  Model:         {short}");
    }
    println!();

    // ── Multimodal one-shot probe ─────────────────────────────────────────
    // When --image was given we don't open an API or REPL; instead we run
    // a single multimodal generation and exit. MVP scope: vision/audio
    // only via the sequential engine path (Phase 1 of the mtmd plan).
    if multimodal_oneshot {
        #[cfg(feature = "multimodal")]
        {
            let image_path = image.expect("multimodal_oneshot implies image is Some");
            let eng = match engine.as_ref() {
                Some(e) => e.clone(),
                None => {
                    eprintln!(
                        "Error: multimodal one-shot needs the sequential engine but none is loaded."
                    );
                    std::process::exit(1);
                }
            };
            run_multimodal_oneshot(eng, image_path).await;
            return;
        }
        #[cfg(not(feature = "multimodal"))]
        unreachable!(
            "multimodal_oneshot==true is gated on --image, which is refused on text-only builds"
        );
    }

    // Take the REPL's backend before both halves move into api::serve.
    //
    // The REPL used to accept only a `SchedulerHandle`, and a sequentially
    // loaded model has none: every multimodal model, and anything run with
    // --batch-size 0. So `--cli` and `--no-ui` on such a model printed "Type a
    // message to chat", found no scheduler, fell out of the branch and ended
    // `main` — killing the API server that had just been spawned. Now the REPL
    // takes either backend and streams through the same channel, so the
    // terminal works for exactly the set of models the API works for.
    //
    // Prefer the scheduler when both exist: it is what serves the API, and two
    // decode loops over one model would contend for the same KV cache.
    let repl_backend = match (scheduler.clone(), engine.clone()) {
        (Some(s), _) => Some(ChatBackend::Batched(s)),
        (None, Some(e)) => Some(ChatBackend::Sequential(e)),
        (None, None) => None,
    };
    let is_tty = std::io::IsTerminal::is_terminal(&std::io::stdin());
    let repl_possible = repl_backend.is_some() && is_tty && !open_chat;

    // Banner must match what we're actually about to do. If the browser chat
    // is going to take over, telling the user to "Type a message" in this
    // terminal is a lie.
    if repl_possible {
        println!("Type a message to chat. /bye or Ctrl+D to quit, Ctrl+C to discard the line.\n");
    } else {
        // Asking for the terminal and not getting it must never be silent —
        // that is how the --cli bug stayed invisible. Only two things can stop
        // it now, and neither is about the model.
        if !open_chat && !repl_possible {
            if repl_backend.is_none() {
                println!("No model is loaded, so there is nothing to chat with in the terminal.");
            } else if !is_tty {
                println!(
                    "Standard input is not a terminal, so the terminal chat cannot run.\n\
                     Use the API on the port below, or run this from an interactive shell."
                );
            }
        }
        println!("Press Ctrl+C to stop.\n");
    }

    // Start the API server in the background.
    let api_model_name = model_name.clone();
    // The name/path pair the API may always resolve, even with
    // EULLM_ALLOW_MODEL_PATHS off: `/api/tags` advertises this name, so a
    // client echoing it back must not be refused. See
    // `api::AppState::launch_model`.
    let api_launch_model = launch_gguf_path.map(|p| (model_name.clone(), p));
    let api_store = ModelStore::default_store().expect("model store");
    tokio::spawn(async move {
        if let Err(e) = api::serve(api::ServeConfig {
            port,
            // `run` resolves the projector itself and hands the loaded engine
            // over; this is only the fallback for a later swap.
            mmproj: api_mmproj,
            // The flag, not `mmproj_placement`: that was worked out for the
            // launch model, and each swap decides again for its own.
            mmproj_offload,
            model_name: Some(api_model_name),
            engine,
            scheduler,
            gpu_layers: flag_gpu_layers,
            fit,
            fit_strict,
            ctx_size,
            threads: resolved_threads,
            flash_attn,
            n_batch: n_batch_flag,
            n_ubatch: n_ubatch_flag,
            cache_type_k,
            cache_type_v,
            batch_size: launch_batch_size,
            cpu_moe: flag_cpu_moe,
            n_cpu_moe: flag_n_cpu_moe,
            rs_seq,
            mtp,
            mtp_p_min,
            mtp_model,
            moe_cache,
            no_mmap: no_mmap_flag,
            mmap,
            moe_prefetch,
            load_threads,
            kv_unified,
            ctx_checkpoints,
            checkpoint_min_step,
            rust_debug,
            web_enabled: web,
            store: api_store,
            ui_port,
            launch_model: api_launch_model,
            keep_alive,
            launch_embedding,
            launch_decision,
            decision_ctx,
            residency,
            // What the launch model itself got, for `/api/ps` only.
            launch_gpu_layers: Some(gpu_layers),
            backend,
        })
        .await
        {
            eprintln!("Server error: {e}");
            std::process::exit(1);
        }
    });

    // Give the API server a moment to bind.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Auto-open the browser chat (default). Suppressed by --cli / --no-chat
    // (open_chat=false) or --no-ui (ui_port=None).
    if open_chat && let Some(p) = ui_port {
        let url = format!("http://localhost:{p}/");
        match open_browser(&url) {
            Ok(()) => println!(
                "Opening chat in your browser: {url}\n  (use --cli to stay in the terminal)\n"
            ),
            Err(_) => println!("Open the chat in your browser: {url}\n"),
        }
    }

    // The terminal REPL is the CLI counterpart to the browser chat: at most
    // one should be active at a time. If we opened the browser (default), the
    // user is chatting there — the REPL would just compete for the same model
    // on the same line discipline. Only drop into the REPL when the browser
    // was suppressed (--cli / --no-chat) or unavailable (--no-ui).
    if let Some(backend) = repl_backend.filter(|_| repl_possible) {
        interactive_chat(backend, &model_name, ctx_size, batch_size, web).await;
    } else {
        // No REPL — wait for shutdown signal.
        // The API server handles graceful shutdown internally via SIGTERM/SIGINT.
        tokio::signal::ctrl_c().await.ok();
        tracing::info!("Shutting down...");
    }
}

#[allow(clippy::too_many_arguments)]
async fn cmd_serve(
    port: u16,
    replace: bool,
    ui_port: Option<u16>,
    batch_size: usize,
    gpu_layers: i32,
    fit: bool,
    fit_strict: bool,
    ctx_size: u32,
    threads: Option<u32>,
    flash_attn: bool,
    n_batch: u32,
    n_ubatch: Option<u32>,
    cache_type_k: inference::KvCacheType,
    cache_type_v: inference::KvCacheType,
    web: bool,
    cpu_moe: bool,
    n_cpu_moe: u32,
    rs_seq: u32,
    mtp: u32,
    mtp_p_min: f32,
    mtp_model: Option<PathBuf>,
    moe_cache: Option<fit::MoeCache>,
    no_mmap: bool,
    mmap: bool,
    moe_prefetch: u32,
    load_threads: readahead::LoadThreads,
    kv_unified: bool,
    ctx_checkpoints: usize,
    checkpoint_min_step: u32,
    rust_debug: bool,
    mmproj: Option<PathBuf>,
    mmproj_offload: Option<bool>,
    keep_alive: Option<std::time::Duration>,
    embedding_model: Option<String>,
    decision_model: Option<String>,
    decision_ctx: u32,
    residency: api::ResidencyConfig,
) {
    let batch_size = validate_launch_batch_size(batch_size).unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        std::process::exit(1);
    });
    ensure_port_available(port, replace).await;
    if let Some(p) = ui_port {
        ensure_port_available(p, replace).await;
    }

    let threads = threads.unwrap_or_else(inference::default_thread_count);
    let store = ModelStore::default_store().expect("model store");

    // The one `LlamaBackend` this process will ever create — shared by the
    // embedding model below and by every generation model this server loads
    // later via `load_generation_model`. See `inference::init_shared_backend`.
    let backend = inference::init_shared_backend().unwrap_or_else(|e| {
        eprintln!("Error initializing llama.cpp backend: {e}");
        std::process::exit(1);
    });

    // --embedding-model: load the companion embedding model now. There is no
    // generation model loaded yet to size against, so the reservation only
    // starts to matter once one is loaded later via a request's "model"
    // field, through `AppState::reserved_embedding_bytes` inside
    // `load_generation_model`. See the flag's doc comment on `RuntimeOpts`.
    let launch_embedding = embedding_model.map(|emb_arg| {
        let emb_path = resolve_model_path(&emb_arg, &store).unwrap_or_else(|| {
            eprintln!("Error: embedding model '{emb_arg}' not found.");
            std::process::exit(1);
        });
        let model = inference::embedding::EmbeddingModel::load(
            &emb_path,
            threads,
            inference::embedding::DEFAULT_EMBEDDING_CTX,
            backend.clone(),
        )
        .unwrap_or_else(|e| {
            eprintln!("Error loading embedding model: {e}");
            std::process::exit(1);
        });
        let weights_bytes = std::fs::metadata(&emb_path).map(|m| m.len()).unwrap_or(0);
        let emb_name = launch_companion_name(&emb_arg, &emb_path);
        keep_launch_embedding_context(&model, &emb_path, weights_bytes);
        api::EmbeddingSlot {
            model_name: emb_name,
            model: Arc::new(model),
            is_reserved_companion: true,
        }
    });
    // --decision-model: same as above; `load_generation_model` protects its reserve
    // through `AppState::reserved_decision_bytes` once a generation model
    // is loaded.
    let launch_decision = decision_model.map(|arg| {
        load_launch_decision(
            &arg,
            &store,
            threads,
            decision_ctx,
            flash_attn,
            backend.clone(),
        )
    });

    println!("eullm ready (no model loaded — send a request with a \"model\" field to load one).");
    println!("  API (EULLM):   http://localhost:{port}/api");
    println!("  API (OpenAI):  http://localhost:{port}/v1");
    if let Some(p) = ui_port {
        println!("  Chat UI:       http://localhost:{p}/");
    }
    if rust_debug {
        println!("  Rust debug:    enabled (NaN/Inf logit check active — extra per-token cost)");
    }
    println!("\nPress Ctrl+C to stop.\n");

    if let Err(e) = api::serve(api::ServeConfig {
        port,
        mmproj,
        mmproj_offload,
        model_name: None,
        engine: None,
        scheduler: None,
        gpu_layers,
        fit,
        fit_strict,
        ctx_size,
        threads,
        flash_attn,
        n_batch,
        n_ubatch,
        cache_type_k,
        cache_type_v,
        batch_size,
        cpu_moe,
        n_cpu_moe,
        rs_seq,
        mtp,
        mtp_p_min,
        mtp_model,
        moe_cache,
        no_mmap,
        mmap,
        moe_prefetch,
        load_threads,
        kv_unified,
        ctx_checkpoints,
        checkpoint_min_step,
        rust_debug,
        web_enabled: web,
        store,
        ui_port,
        // Headless serve starts with an empty slot: there is no launch model to
        // grandfather in, so every name goes through the normal resolution.
        launch_model: None,
        keep_alive,
        launch_embedding,
        launch_decision,
        decision_ctx,
        residency,
        launch_gpu_layers: None,
        backend,
    })
    .await
    {
        eprintln!("Server error: {e}");
        std::process::exit(1);
    }
}

/// Load `--decision-model` at launch as a reserved companion — see the
/// flag's doc comment on `RuntimeOpts`. Exits when the model cannot be found
/// or loaded: it was asked for by name, and starting without it would only
/// move the error to the first `/v1/systemone` request.
fn load_launch_decision(
    arg: &str,
    store: &ModelStore,
    threads: u32,
    decision_ctx: u32,
    flash_attn: bool,
    backend: Arc<llama_cpp_2::llama_backend::LlamaBackend>,
) -> api::DecisionSlot {
    let path = resolve_model_path(arg, store).unwrap_or_else(|| {
        eprintln!("Error: decision model '{arg}' not found.");
        std::process::exit(1);
    });
    let model =
        inference::decision::DecisionModel::load(&path, threads, decision_ctx, flash_attn, backend)
            .unwrap_or_else(|e| {
                eprintln!("Error loading decision model: {e}");
                std::process::exit(1);
            });
    let reserve_bytes = fit::decision_reserve_bytes(&path, decision_ctx);
    let model_name = launch_companion_name(arg, &path);
    println!(
        "Decision model loaded: {} (up to {decision_ctx} tokens per request, {} MiB kept free \
         for a request's context)",
        path.display(),
        reserve_bytes / (1024 * 1024)
    );
    api::DecisionSlot {
        model_name,
        model: Arc::new(model),
        is_reserved_companion: true,
        reserve_bytes,
    }
}

/// `eullm unload` — free generation models' VRAM on a running `eullm
/// serve`/`eullm run` server, without restarting the process: every one, or
/// with `--model` that one only.
///
/// Thin CLI wrapper around `POST /api/unload`. The server keeps running; a
/// later request with a `model` field (or another `eullm run <model>`) loads
/// a model back in.
async fn cmd_unload(port: u16, model: Option<&str>) {
    let url = format!("http://127.0.0.1:{port}/api/unload");
    let client = match reqwest::Client::builder().build() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error building HTTP client: {e}");
            std::process::exit(1);
        }
    };
    let request = match model {
        Some(name) => client
            .post(&url)
            .json(&serde_json::json!({ "model": name })),
        None => client.post(&url),
    };
    let response = match request.send().await {
        Ok(r) => r,
        Err(e) => {
            eprintln!(
                "Error: could not reach eullm server at {url}: {e}\n  \
                 Is it running? (`eullm serve` or `eullm run <model>`)"
            );
            std::process::exit(1);
        }
    };
    if !response.status().is_success() {
        eprintln!("Error: server returned HTTP {}", response.status());
        std::process::exit(1);
    }
    match response.json::<serde_json::Value>().await {
        Ok(body) => {
            for line in unload_report(&body, model) {
                println!("{line}");
            }
        }
        Err(e) => eprintln!("Error reading response: {e}"),
    }
}

/// What `eullm unload` prints for the server's answer: every model the
/// server unloaded, or that there was none. `unloaded_all` lists them; a
/// server from before it did names one, in `unloaded`.
fn unload_report(body: &serde_json::Value, asked_for: Option<&str>) -> Vec<String> {
    let mut names: Vec<&str> = body
        .get("unloaded_all")
        .and_then(|v| v.as_array())
        .map(|all| all.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();
    if names.is_empty()
        && let Some(name) = body.get("unloaded").and_then(|v| v.as_str())
    {
        names.push(name);
    }
    if names.is_empty() {
        return vec![match asked_for {
            Some(name) => format!("'{name}' was not loaded."),
            None => "No model was loaded.".to_string(),
        }];
    }
    let mut lines: Vec<String> = names
        .iter()
        .map(|name| format!("Unloaded '{name}'."))
        .collect();
    lines.push("VRAM freed.".to_string());
    lines
}

// ── Import from Ollama ────────────────────────────────────────────────────
//
// Ollama stores downloaded models as content-addressed blobs under
// `~/.ollama/models/`.  The on-disk layout is:
//
//   ~/.ollama/models/
//   ├── manifests/registry.ollama.ai/library/{model}/{tag}   ← JSON manifest
//   └── blobs/sha256-{hex}                                    ← raw files
//
// Each manifest lists "layers" with an OCI-style mediaType.  The layer
// with `application/vnd.ollama.image.model` is the GGUF weights file.
//
// **Licensing note:** Ollama itself does not add any additional license or
// copyright on top of the original model weights.  The GGUF blob is the
// same file distributed by the upstream model author (e.g. on HuggingFace).
// Copying it into the EULLM store is no different from copying a local file
// you already possess.  The license of the model itself still applies — for
// example Apache 2.0 for Qwen 3, MIT for DeepSeek, Gemma terms for Gemma,
// etc.  Always verify the upstream license before redistribution.
//
// **What this command does:**
//
// 1. Reads the Ollama manifest at
//    `~/.ollama/models/manifests/registry.ollama.ai/library/{name}/{tag}`
// 2. Locates the model layer (`application/vnd.ollama.image.model`)
// 3. Resolves the blob path (`~/.ollama/models/blobs/sha256-{hash}`)
// 4. Copies the blob into `~/.eullm/models/{name}/{name}.gguf`
// 5. Writes a EULLM `manifest.json` so the model appears in `eullm list`
//
// After import, the model can be used with `eullm run {name}`, enabling
// bit-identical benchmarks between EULLM Engine and Ollama.

/// Ollama manifest layer entry (OCI-style).
#[derive(serde::Deserialize)]
struct OllamaLayer {
    /// OCI media type — `application/vnd.ollama.image.model` for the GGUF weights.
    #[serde(rename = "mediaType")]
    media_type: String,
    /// Content-addressed digest, e.g. `sha256:abc123...`.
    digest: String,
    /// Layer size in bytes.
    size: u64,
}

/// Top-level Ollama manifest (simplified — we only need `layers`).
#[derive(serde::Deserialize)]
struct OllamaManifest {
    layers: Vec<OllamaLayer>,
}

/// Whether `digest` is exactly `sha256:` followed by 64 lowercase hex chars —
/// the only shape ever produced by Ollama's own manifests. Rejects anything
/// else before it becomes a filesystem path component.
fn is_valid_sha256_digest(digest: &str) -> bool {
    let Some(hex) = digest.strip_prefix("sha256:") else {
        return false;
    };
    hex.len() == 64
        && hex
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

#[cfg(test)]
mod digest_validation_tests {
    use super::is_valid_sha256_digest;

    #[test]
    fn accepts_a_real_shaped_digest() {
        assert!(is_valid_sha256_digest(&format!(
            "sha256:{}",
            "a".repeat(64)
        )));
    }

    #[test]
    fn rejects_path_traversal_attempts() {
        assert!(!is_valid_sha256_digest("sha256:../../../../etc/passwd"));
    }

    #[test]
    fn rejects_wrong_length_and_missing_prefix() {
        assert!(!is_valid_sha256_digest("sha256:abcd"));
        assert!(!is_valid_sha256_digest(
            "c62ccde5630c20c8a9cc0548233e78dc9414540c62d4d5b3f1a5a89e4b6b6c0"
        ));
    }

    #[test]
    fn rejects_uppercase_hex() {
        assert!(!is_valid_sha256_digest(&format!(
            "sha256:{}",
            "A".repeat(64)
        )));
    }
}

/// Import a model from a local Ollama installation into the EULLM store.
///
/// This copies the GGUF blob so that EULLM and Ollama can be benchmarked
/// against the exact same model weights.  The copy is always a full
/// physical copy (no symlinks) to remain independent of Ollama's storage.
///
/// # Arguments
///
/// * `store` — EULLM local model store (`~/.eullm/models/`)
/// * `model` — Ollama model specifier, e.g. `"llama3.2"` or `"qwen3:14b"`
/// * `ollama_dir` — Optional override for the Ollama data directory
///   (defaults to `~/.ollama`)
fn cmd_import_ollama(store: &ModelStore, model: &str, ollama_dir: Option<&str>) {
    // Resolve Ollama data directory
    let ollama_root = if let Some(dir) = ollama_dir {
        PathBuf::from(dir)
    } else {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_else(|_| std::env::temp_dir().to_string_lossy().into_owned());
        PathBuf::from(home).join(".ollama")
    };

    if !ollama_root.exists() {
        eprintln!(
            "Error: Ollama directory not found: {}",
            ollama_root.display()
        );
        eprintln!("  Is Ollama installed? Try: ollama --version");
        eprintln!(
            "  Or specify a custom path: eullm import-ollama {model} --ollama-dir /path/to/ollama"
        );
        std::process::exit(1);
    }

    // Parse model name and tag (e.g., "llama3.2:8b" → name="llama3.2", tag="8b")
    let (model_name, model_tag) = if let Some(pos) = model.find(':') {
        (&model[..pos], &model[pos + 1..])
    } else {
        (model, "latest")
    };

    // Find the Ollama manifest file
    // Ollama stores manifests at: manifests/registry.ollama.ai/library/{name}/{tag}
    let manifest_path = ollama_root
        .join("models")
        .join("manifests")
        .join("registry.ollama.ai")
        .join("library")
        .join(model_name)
        .join(model_tag);

    if !manifest_path.exists() {
        eprintln!("Error: Ollama model '{model}' not found.");
        eprintln!("  Looked in: {}", manifest_path.display());
        eprintln!();

        // Try to list available models
        let library_dir = ollama_root
            .join("models")
            .join("manifests")
            .join("registry.ollama.ai")
            .join("library");
        if library_dir.is_dir() {
            eprintln!("Available Ollama models:");
            if let Ok(entries) = std::fs::read_dir(&library_dir) {
                for entry in entries.flatten() {
                    if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                        let name = entry.file_name();
                        // List tags
                        if let Ok(tags) = std::fs::read_dir(entry.path()) {
                            for tag in tags.flatten() {
                                let tag_name = tag.file_name();
                                println!(
                                    "  {}:{}",
                                    name.to_string_lossy(),
                                    tag_name.to_string_lossy()
                                );
                            }
                        }
                    }
                }
            }
        } else {
            eprintln!("No Ollama models found. Pull one first: ollama pull {model}");
        }
        std::process::exit(1);
    }

    // Parse the Ollama manifest JSON
    let manifest_data = match std::fs::read_to_string(&manifest_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error reading Ollama manifest: {e}");
            std::process::exit(1);
        }
    };

    let manifest: OllamaManifest = match serde_json::from_str(&manifest_data) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error parsing Ollama manifest: {e}");
            std::process::exit(1);
        }
    };

    // Find the model layer (the GGUF blob)
    let model_layer = manifest
        .layers
        .iter()
        .find(|l| l.media_type == "application/vnd.ollama.image.model");

    let model_layer = match model_layer {
        Some(l) => l,
        None => {
            eprintln!("Error: no model layer found in Ollama manifest for '{model}'.");
            eprintln!("  This may not be a standard Ollama model.");
            std::process::exit(1);
        }
    };

    if !is_valid_sha256_digest(&model_layer.digest) {
        eprintln!(
            "Error: malformed digest in Ollama manifest for '{model}': {}",
            model_layer.digest
        );
        std::process::exit(1);
    }

    // The blob is stored at: blobs/{digest} (with ":" replaced by "-")
    let blob_filename = model_layer.digest.replace(':', "-");
    let blob_path = ollama_root
        .join("models")
        .join("blobs")
        .join(&blob_filename);

    if !blob_path.exists() {
        eprintln!("Error: Ollama blob not found: {}", blob_path.display());
        eprintln!("  The model may be partially downloaded. Try: ollama pull {model}");
        std::process::exit(1);
    }

    // Determine EULLM model name
    let eullm_name = if model_tag == "latest" {
        model_name.to_string()
    } else {
        format!("{model_name}-{model_tag}")
    };

    // Check if already imported
    if let Some(existing) = store.gguf_path(&eullm_name) {
        println!("Model '{}' is already imported.", eullm_name);
        println!("  GGUF: {}", existing.display());
        println!("\nRun with: eullm run {eullm_name}");
        return;
    }

    let blob_size = model_layer.size;
    println!("Importing Ollama model '{model}' → eullm/{eullm_name}");
    println!("  Source: {}", blob_path.display());
    println!("  Size:   {}", format_bytes(blob_size));
    println!("  Copying GGUF blob...");

    // Create destination directory and copy
    let dest_dir = store.model_path(&eullm_name);
    if let Err(e) = std::fs::create_dir_all(&dest_dir) {
        eprintln!("Error creating directory: {e}");
        std::process::exit(1);
    }

    let gguf_filename = format!("{eullm_name}.gguf");
    let dest_path = dest_dir.join(&gguf_filename);

    // Try patched copy first — fixes Ollama GGUF metadata quirks
    // (e.g. qwen35.rope.dimension_sections with 3 elements instead of 4).
    let patched = match gguf_patch::patch_gguf_if_needed(&blob_path, &dest_path) {
        Ok(true) => {
            println!(
                "  Patched GGUF metadata during copy (fixed array lengths for llama.cpp compatibility)."
            );
            true
        }
        Ok(false) => false,
        Err(e) => {
            tracing::warn!("GGUF patch check failed ({e}), falling back to plain copy");
            false
        }
    };

    // If no patching was needed (or patching failed), do a normal copy.
    if !patched {
        match copy_with_progress(&blob_path, &dest_path, blob_size) {
            Ok(()) => {}
            Err(e) => {
                eprintln!("\nError copying model: {e}");
                // Clean up partial copy
                let _ = std::fs::remove_file(&dest_path);
                std::process::exit(1);
            }
        }
    }

    eprintln!(); // newline after progress

    // Write EULLM manifest
    let manifest = models::store::ModelManifest {
        id: eullm_name.clone(),
        name: format!("eullm/{eullm_name}"),
        description: format!("Imported from Ollama: {model}"),
        languages: vec![],
        base: model_name.to_string(),
        vram_gb: estimate_vram(blob_size),
        size_bytes: blob_size,
        license: "See original model".into(),
        digest: model_layer.digest.clone(),
        pulled_at: chrono::Utc::now().to_rfc3339(),
        status: "ready".into(),
        gguf_file: Some(gguf_filename),
        mmproj_file: None,
        hf_repo: None,
        hf_filename: None,
    };

    let manifest_json = serde_json::to_string_pretty(&manifest).unwrap();
    let manifest_path = dest_dir.join("manifest.json");
    if let Err(e) = std::fs::write(&manifest_path, manifest_json) {
        eprintln!("Warning: model copied but manifest write failed: {e}");
    }

    println!("  Done. Model imported successfully.");
    println!();
    println!("Run with: eullm run {eullm_name}");
}

/// Copy a file from `src` to `dst` with a progress indicator on stderr.
///
/// Uses 8 MB buffered I/O for throughput.  Progress is printed every 50 MB
/// as a carriage-return line (`\r`) so it updates in place.
fn copy_with_progress(
    src: &std::path::Path,
    dst: &std::path::Path,
    total: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::{Read, Write};

    let mut reader = std::io::BufReader::with_capacity(8 * 1024 * 1024, std::fs::File::open(src)?);
    let mut writer =
        std::io::BufWriter::with_capacity(8 * 1024 * 1024, std::fs::File::create(dst)?);

    let mut copied: u64 = 0;
    let mut buf = vec![0u8; 8 * 1024 * 1024]; // 8MB buffer
    let mut last_report: u64 = 0;

    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        writer.write_all(&buf[..n])?;
        copied += n as u64;

        // Report every 50MB
        if copied - last_report > 50_000_000 || copied >= total {
            last_report = copied;
            let pct = if total > 0 {
                (copied as f64 / total as f64 * 100.0) as u32
            } else {
                0
            };
            eprint!(
                "\r  {}/{} ({}%)",
                format_bytes(copied),
                format_bytes(total),
                pct
            );
            let _ = std::io::stderr().flush();
        }
    }

    writer.flush()?;
    Ok(())
}

/// Rough VRAM estimate from GGUF file size.
///
/// For Q4_K_M quantized models the file size is a reasonable proxy for
/// runtime memory usage.  We add ~500 MB for KV cache and runtime overhead.
fn estimate_vram(size_bytes: u64) -> u32 {
    let gb = size_bytes as f64 / 1_000_000_000.0;
    (gb + 0.5).ceil() as u32
}

#[allow(clippy::too_many_arguments)]
fn cmd_forge(
    source: &str,
    profile: Option<&str>,
    identity: Option<&str>,
    lang: Option<&str>,
    output: Option<&str>,
    target_vram: Option<u16>,
    estimate_only: bool,
    skip_pruning: bool,
    skip_distillation: bool,
    skip_quantization: bool,
    skip_identity: bool,
) {
    let mut args = vec!["forge".to_string(), source.to_string()];

    if let Some(p) = profile {
        args.push("--profile".into());
        args.push(p.into());
    }
    if let Some(i) = identity {
        args.push("--identity".into());
        args.push(i.into());
    }
    if let Some(l) = lang {
        args.push("--lang".into());
        args.push(l.into());
    }
    if let Some(o) = output {
        args.push("--output".into());
        args.push(o.into());
    }
    if let Some(v) = target_vram {
        args.push("--target-vram".into());
        args.push(v.to_string());
    }
    if estimate_only {
        args.push("--estimate-only".into());
    }
    if skip_pruning {
        args.push("--skip-pruning".into());
    }
    if skip_distillation {
        args.push("--skip-distillation".into());
    }
    if skip_quantization {
        args.push("--skip-quantization".into());
    }
    if skip_identity {
        args.push("--skip-identity".into());
    }

    println!("eullm forge — delegating to eullm-forge pipeline...\n");

    let status = std::process::Command::new("eullm-forge")
        .args(&args)
        .status();

    match status {
        Ok(s) if s.success() => {}
        Ok(s) => {
            std::process::exit(s.code().unwrap_or(1));
        }
        Err(_) => {
            let py_status = std::process::Command::new("python3")
                .arg("-m")
                .arg("eullm_forge.cli")
                .args(&args)
                .status();

            match py_status {
                Ok(s) if s.success() => {}
                Ok(s) => {
                    std::process::exit(s.code().unwrap_or(1));
                }
                Err(_) => {
                    eprintln!("Error: eullm-forge is not installed.");
                    eprintln!();
                    eprintln!("Install it with:");
                    eprintln!("  pip install eullm-forge");
                    eprintln!();
                    eprintln!("Or from source:");
                    eprintln!("  cd forge && pip install -e '.[dev]'");
                    std::process::exit(1);
                }
            }
        }
    }
}

// ── Interactive chat REPL ─────────────────────────────────────────────────────

/// A single message in the conversation history.
struct ChatMessage {
    role: &'static str,
    content: String,
}

/// Where the terminal chat gets its tokens.
///
/// The REPL used to take a `SchedulerHandle` and nothing else, so a model that
/// loads sequentially — every multimodal one, and anything with
/// `--batch-size 0` — had no terminal chat at all. Both sources emit
/// `StreamEvent`, so the loop is identical either way and only the submission
/// differs.
enum ChatBackend {
    Batched(inference::SchedulerHandle),
    Sequential(std::sync::Arc<inference::InferenceEngine>),
}

impl ChatBackend {
    fn submit(
        &self,
        request: inference::GenerateRequest,
    ) -> tokio::sync::mpsc::Receiver<inference::StreamEvent> {
        match self {
            Self::Batched(s) => s.submit(request),
            Self::Sequential(e) => {
                crate::api::routes::sequential_to_channel(std::sync::Arc::clone(e), request)
            }
        }
    }
}

/// Build the prompt (and matching stop sequences) for one `--cli` turn.
///
/// Mirrors `api::routes::build_chat_prompt` exactly, on purpose: the web/API
/// path and this terminal one are two doors onto the same loaded model, and
/// they must decide the same way or the same conversation answers
/// differently depending only on which door was used to ask — which is
/// exactly what happened before this existed (`--cli` never got the dynamic
/// GGUF template `build_chat_prompt` added for the web/API path). Tries the
/// model's own embedded chat template first — on both backends, the batched
/// one via the scheduler's weak model reference (see `SharedModel`); falls
/// back to the hardcoded per-family `template` otherwise, exactly as
/// `--cli` always has. The third element is the response prefix, as there.
fn build_cli_prompt(
    backend: &ChatBackend,
    template: &chat_template::ChatTemplate,
    pairs: &[(&str, &str)],
    think_arg: bool,
) -> (String, Vec<String>, String) {
    let dynamic = match backend {
        ChatBackend::Sequential(engine) => engine.apply_jinja_chat_template(pairs, think_arg),
        ChatBackend::Batched(scheduler) => scheduler.apply_jinja_chat_template(pairs, think_arg),
    };
    if let Some(dynamic) = dynamic {
        return (dynamic.prompt, Vec::new(), dynamic.preopened);
    }
    (
        template.build_prompt(pairs, think_arg),
        template.stop_sequences(),
        String::new(),
    )
}

async fn interactive_chat(
    backend: ChatBackend,
    model_name: &str,
    ctx_size: u32,
    batch_size: usize,
    web_enabled: bool,
) {
    // Effective per-slot context — with continuous batching the total ctx
    // is divided among slots; injected web content must fit in one slot.
    let effective_ctx = if batch_size > 1 {
        ctx_size / batch_size as u32
    } else {
        ctx_size
    };
    // Resolved once, outside the loop: the policy cannot change while the
    // session runs, and reading the environment per fetch would make the log
    // line below a claim about a different policy than the one enforced.
    let web_policy = crate::tools::guard::WebPolicy::from_env();
    if web_enabled {
        eprintln!("[web] enabled — fetchable: {}", web_policy.describe());
    }
    use std::io::Write;

    let short = model_name.strip_prefix("eullm/").unwrap_or(model_name);

    let mut temperature: f32 = 0.8;
    // u32::MAX = unlimited: the request already clamps to the remaining
    // context budget, which is the real bound. A fixed default cap truncated
    // reasoning models mid-think (Qwen3.6 spent ~2000 tokens thinking about a
    // hard question and hit the old 2048 default before answering at all);
    // /maxtokens <n> still sets a cap, /maxtokens 0 returns to unlimited.
    let mut max_reply_tokens: u32 = u32::MAX;
    // Sticky reasoning toggle. ON by default (reasoning models need it). When
    // OFF we append the ` /no_think` soft-switch to each user turn AND, for
    // every model except the DeepSeek-R1 family, also force an empty
    // `<think></think>` block in the template — the mechanism the API's
    // `"think": false` param already uses. R1-style models are always-
    // reasoning and never learned to see a pre-closed empty think block as
    // anything but malformed input, so they keep relying on the soft-switch
    // text alone.
    let mut think_mode = true;
    let is_r1_family = {
        let lower = model_name.to_lowercase();
        lower.contains("deepseek-r1") || lower.contains("deepseek_r1")
    };

    let mut history: Vec<ChatMessage> = vec![ChatMessage {
        role: "system",
        content: "You are a helpful assistant.".into(),
    }];

    let mut reader = LineReader::new();

    loop {
        // Read user input (supports multi-line with trailing \). rustyline
        // prints the prompt itself, so the fallback path does it too rather
        // than the caller — otherwise a redrawn line would duplicate it.
        let mut input = String::new();
        let mut prompt = ">>> ";
        loop {
            match reader.read(prompt) {
                Line::Eof => {
                    println!();
                    return;
                }
                Line::Interrupted => {
                    // Drop the whole entry, including any continuation lines
                    // already accepted, and start over at the main prompt.
                    input.clear();
                    prompt = ">>> ";
                }
                Line::Text(line) => {
                    let trimmed = line.trim_end_matches('\n').trim_end_matches('\r');
                    if let Some(stripped) = trimmed.strip_suffix('\\') {
                        input.push_str(stripped);
                        input.push('\n');
                        prompt = "... ";
                        continue;
                    }
                    input.push_str(trimmed);
                    break;
                }
            }
        }

        let mut input = input.trim().to_string();
        if input.is_empty() {
            continue;
        }
        // Commands are recalled too: /temp 0.2 is exactly the kind of line
        // someone types, adjusts, and types again.
        reader.remember(&input);

        // Commands
        if input == "/bye" || input == "/exit" || input == "/quit" || input == "/q" {
            println!("Bye!");
            return;
        } else if input == "/clear" {
            history.truncate(1);
            println!("Chat history cleared.\n");
            continue;
        } else if input == "/help" {
            println!("Commands:");
            println!("  /bye, /q          Exit the chat (Ctrl+D does the same)");
            println!("  /clear            Clear conversation history");
            println!(
                "  /think            Enable reasoning (current: {})",
                if think_mode { "on" } else { "off" }
            );
            println!("  /no_think         Disable reasoning (sticky until /think)");
            println!("  /temp <0.0–2.0>   Set temperature (current: {temperature:.1})");
            let max_reply_display = if max_reply_tokens == u32::MAX {
                "unlimited".to_string()
            } else {
                max_reply_tokens.to_string()
            };
            println!("  /maxtokens <n>    Cap reply tokens, 0 = unlimited (current: {max_reply_display})");
            println!("  /system <text>    Replace system prompt");
            println!("  /help             Show this help\n");
            continue;
        } else if input == "/think" {
            think_mode = true;
            println!("Reasoning ON.\n");
            continue;
        } else if input == "/no_think" {
            think_mode = false;
            println!("Reasoning OFF (sticky — re-enable with /think).\n");
            continue;
        } else if let Some(rest) = input.strip_prefix("/no_think ") {
            // Inline form: disable reasoning AND send this message.
            think_mode = false;
            input = rest.trim().to_string();
        } else if let Some(rest) = input.strip_prefix("/think ") {
            think_mode = true;
            input = rest.trim().to_string();
        } else if let Some(val) = input.strip_prefix("/temp ") {
            match val.trim().parse::<f32>() {
                Ok(t) if (0.0..=2.0).contains(&t) => {
                    temperature = t;
                    println!("Temperature set to {temperature:.2}\n");
                }
                _ => eprintln!("Usage: /temp <0.0–2.0>\n"),
            }
            continue;
        } else if let Some(val) = input.strip_prefix("/maxtokens ") {
            match val.trim().parse::<u32>() {
                Ok(0) => {
                    max_reply_tokens = u32::MAX;
                    println!("Max reply tokens set to unlimited (context window is the cap)\n");
                }
                Ok(n) => {
                    max_reply_tokens = n;
                    println!("Max reply tokens set to {max_reply_tokens}\n");
                }
                _ => eprintln!("Usage: /maxtokens <n>  (0 = unlimited)\n"),
            }
            continue;
        } else if let Some(sys) = input.strip_prefix("/system ") {
            if let Some(first) = history.first_mut() {
                first.content = sys.trim().to_string();
                println!("System prompt updated.\n");
            }
            continue;
        } else if input.starts_with('/') && !input[1..].starts_with('/') {
            // An unrecognised slash command is a typo, not a message.
            //
            // Reported from a real session: someone typed `/q` to leave, it
            // fell through to here as ordinary text, and the model spent a
            // minute earnestly answering it. Silently sending a mistyped
            // command to a 4 tok/s model is the worst of both outcomes — the
            // user waits for something they did not ask for, and nothing on
            // screen says why.
            //
            // `//` is the escape hatch for a message that really does start
            // with a slash: it is stripped and the rest is sent as typed.
            let cmd = input.split_whitespace().next().unwrap_or(&input);
            eprintln!("Unknown command: {cmd}");
            eprintln!("  /help lists the commands. To send this as a message, start it with // instead.\n");
            continue;
        }

        // A message the user deliberately started with a slash: drop the
        // escaping first one and send the rest.
        if let Some(rest) = input.strip_prefix("//") {
            input = rest.to_string();
        }

        // Add user message to permanent history. When reasoning is toggled
        // off, append the ` /no_think` soft-switch the models actually honour.
        let user_content = if think_mode {
            input.clone()
        } else {
            format!("{input} /no_think")
        };
        history.push(ChatMessage {
            role: "user",
            content: user_content,
        });

        // Whether to let build_prompt open the assistant turn normally
        // (true) or force the empty `<think></think>` suppression (false).
        // Only suppress via the template when reasoning is actually off AND
        // the model isn't DeepSeek-R1-family (see think_mode's doc comment).
        let think_arg = think_mode || is_r1_family;

        // Build prompt using the model-appropriate chat template.
        // If web browsing is enabled, fetch URLs and inject content into a
        // TEMPORARY message list — web content is NOT stored in history so it
        // doesn't accumulate across turns and bloat the context.
        let template = crate::chat_template::ChatTemplate::detect(model_name);
        let (prompt, stop_sequences, response_prefix) = if web_enabled {
            let urls = crate::tools::extract_urls(&input);
            if !urls.is_empty() {
                let existing_chars: usize = history.iter().map(|m| m.content.len()).sum();
                let mut injected = Vec::new();
                for url in &urls {
                    match crate::tools::fetch_for_context(
                        url,
                        effective_ctx,
                        existing_chars,
                        &input,
                        &web_policy,
                    )
                    .await
                    {
                        Ok((content, truncated)) => {
                            let note = if truncated {
                                " [truncated to fit context]"
                            } else {
                                ""
                            };
                            injected.push(format!("[Web content from {url}{note}]\n\n{content}"));
                            eprintln!(
                                "[web] fetched {} ({} chars{})",
                                url,
                                content.len(),
                                if truncated { ", truncated" } else { "" }
                            );
                        }
                        Err(e) => {
                            injected.push(format!("[Failed to fetch {url}: {e}]"));
                            eprintln!("[web] fetch failed for {url}: {e}");
                        }
                    }
                }
                if !injected.is_empty() {
                    // Build a temporary message list with the web content injected
                    // just before the current user turn — not stored in history.
                    let web_msg = ChatMessage {
                        role: "system",
                        content: injected.join("\n\n---\n\n"),
                    };
                    let insert_at = history.len() - 1; // before last (user) msg
                    let mut tmp: Vec<&ChatMessage> = history[..insert_at].iter().collect();
                    tmp.push(&web_msg);
                    tmp.push(history.last().unwrap());
                    let pairs: Vec<(&str, &str)> =
                        tmp.iter().map(|m| (m.role, m.content.as_str())).collect();
                    build_cli_prompt(&backend, &template, &pairs, think_arg)
                } else {
                    let pairs: Vec<(&str, &str)> = history
                        .iter()
                        .map(|m| (m.role, m.content.as_str()))
                        .collect();
                    build_cli_prompt(&backend, &template, &pairs, think_arg)
                }
            } else {
                let pairs: Vec<(&str, &str)> = history
                    .iter()
                    .map(|m| (m.role, m.content.as_str()))
                    .collect();
                build_cli_prompt(&backend, &template, &pairs, think_arg)
            }
        } else {
            let pairs: Vec<(&str, &str)> = history
                .iter()
                .map(|m| (m.role, m.content.as_str()))
                .collect();
            build_cli_prompt(&backend, &template, &pairs, think_arg)
        };

        // Rough token estimate: ~4 chars per token. This is the real bound —
        // whatever room is left in the context after the prompt — with no
        // extra cap layered on top. A hardcoded `.min(2048)` used to sit here,
        // silently contradicting the "unlimited by default" comment on
        // `max_reply_tokens` above: with `--ctx-size 4096` and a 76-token
        // prompt, ~4000 tokens of real room were available, but every reply
        // still got cut at 2048 and reported as `truncated — out of context`
        // — a context exhaustion that never actually happened. `max_tokens`
        // is combined with `max_reply_tokens` below, so `/maxtokens <n>`
        // still works as an explicit cap; only the silent unconditional one
        // is gone.
        let estimated_prompt_tokens = prompt.len() as u32 / 4;
        let max_tokens = ctx_size.saturating_sub(estimated_prompt_tokens);

        if max_tokens < 32 {
            eprintln!("Warning: conversation too long for context window. Use /clear to reset.\n");
            history.pop();
            continue;
        }

        let request = inference::GenerateRequest {
            prompt,
            max_tokens: max_tokens.min(max_reply_tokens),
            temperature,
            stop_sequences,
            // Starts the answer with the reasoning block the template
            // opened, so the history below can strip the block whole.
            response_prefix,
            ..Default::default()
        };

        // Submit to whichever backend is loaded and stream tokens.
        let mut rx = backend.submit(request);
        let mut response_text = String::new();
        let mut stats_line = String::new();

        while let Some(event) = rx.recv().await {
            match event {
                inference::StreamEvent::Token(piece) => {
                    print!("{piece}");
                    let _ = std::io::stdout().flush();
                    response_text.push_str(&piece);
                }
                inference::StreamEvent::Done {
                    tokens_generated,
                    tokens_prompt,
                    duration_ms,
                    stop_reason,
                    ..
                } => {
                    // Strip any trailing stop sequence that was printed as part of the stream.
                    // Use trim_end() before matching: some models append \n after the
                    // stop token (e.g. Gemma's <end_of_turn>\n), which would break
                    // an exact ends_with() check.
                    let trimmed = response_text.trim_end();
                    for stop in template.stop_sequences() {
                        if trimmed.ends_with(&stop) {
                            // Erase the stop token (+ any trailing whitespace) from
                            // the terminal using backspaces.
                            let suffix_len = response_text.len() - trimmed.len() + stop.len();
                            let erase_chars = response_text[response_text.len() - suffix_len..]
                                .chars()
                                .count();
                            let erase = "\x08 \x08".repeat(erase_chars);
                            print!("{erase}");
                            let _ = std::io::stdout().flush();
                            response_text.truncate(trimmed.len() - stop.len());
                            break;
                        }
                    }
                    let tps = if duration_ms > 0 {
                        tokens_generated as f64 / (duration_ms as f64 / 1000.0)
                    } else {
                        0.0
                    };
                    // Say it in the terminal too: an answer that stops because
                    // the slot ran out of context looks exactly like a finished
                    // one otherwise.
                    let truncated = match stop_reason {
                        inference::StopReason::Length => ", truncated — out of context",
                        inference::StopReason::Stop => "",
                    };
                    stats_line = format!(
                        "\n\n[{short}: {tokens_generated} tokens, {tokens_prompt} prompt, {:.1} tok/s{}]\n",
                        tps, truncated
                    );
                    break;
                }
                inference::StreamEvent::Error(e) => {
                    eprintln!("\nError: {e}\n");
                    break;
                }
            }
        }

        if !stats_line.is_empty() {
            print!("{stats_line}");
        }

        // Add assistant response to history.
        //
        // When thinking is on, the reasoning block is dropped before storing:
        // it existed to produce this answer, not to be re-read on every later
        // turn. Keeping it is what made a terminal conversation hit
        // `truncated — out of context` several exchanges before the same
        // conversation did in the web UI, which has always stripped it
        // (`ui/app.js`, `stripThink`) — a few hundred reasoning tokens per turn
        // add up far faster than the answers do.
        //
        // This does cost some prefix KV reuse: the reconstructed history no
        // longer matches the tokens actually resident in the cache, so reuse
        // now ends at the last user turn instead of covering the whole
        // conversation. The trade is clearly worth it — re-decoding one
        // stripped answer is a short prefill, while the reasoning it replaces
        // would otherwise occupy the context permanently, on every turn.
        //
        // When this turn suppressed thinking (think_arg == false) nothing is
        // stripped, and the model's decoded `think_suppression_prefix()` is
        // prepended instead: stored history must include it, or every later
        // turn reconstructs text that no longer matches this turn's KV cache
        // (see `ChatTemplate::build_prompt`'s doc comment).
        if !response_text.is_empty() {
            let content = if think_arg {
                crate::chat_template::strip_reasoning_blocks(&response_text)
            } else {
                format!("{}{response_text}", template.think_suppression_prefix())
            };
            history.push(ChatMessage {
                role: "assistant",
                content,
            });
        }
    }
}

/// Spawn a new copy of this process without --daemon, then exit.
///
/// We cannot use `fork()` because the tokio runtime has already created
/// threads — threads don't survive fork, causing an immediate segfault.
/// Instead, we re-exec the same binary with `--daemon` stripped from args
/// and the child's stdout/stderr redirected to a log file.
/// How long to wait before declaring a spawned daemon healthy.
///
/// Long enough for the failures that happen at startup — a bound port, an
/// unreadable model, malformed `EULLM_API_KEYS` — to have already killed the
/// child, short enough not to be felt when everything is fine. Model loading
/// continues well past this; we are only ruling out an immediate death.
const DAEMON_STARTUP_GRACE: std::time::Duration = std::time::Duration::from_millis(1200);

/// The value of `--flag VALUE` or `--flag=VALUE` in a raw argv, if present.
///
/// Used by the `--daemon` branch, which has to read a couple of flags before
/// clap parses anything.
fn arg_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    let eq_prefix = format!("{flag}=");
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .or_else(|| {
            args.iter()
                .find_map(|a| a.strip_prefix(&eq_prefix))
                .filter(|v| !v.is_empty())
        })
}

/// Where the background daemon writes its output.
///
/// Precedence, and the reason for each step:
///
/// 1. `--logfile` — an explicit choice wins over everything.
/// 2. `--pidfile` set, `--logfile` not — keep the log beside the PID file,
///    which is what the engine did for every release up to 0.6.81 and what a
///    deployment that already redirects the PID file expects.
/// 3. Neither set — `~/.eullm/logs/eullm.log`, next to the model store and
///    the audit trail. The old default landed in `/tmp` because it was derived
///    from the PID file, and `/tmp` is small on many systems and is cleared on
///    reboot or under space pressure: the log disappears precisely when it is
///    the only account of why the daemon died (#354). A PID file is genuinely
///    worthless after a reboot, so that one stays in `/tmp`.
/// 4. No home directory at all (a service account with `HOME` unset) — the
///    system temp directory, because a daemon that cannot write a log should
///    still start.
fn resolve_daemon_log_path(
    logfile: Option<&str>,
    pidfile: Option<&str>,
    home: Option<&str>,
) -> PathBuf {
    if let Some(explicit) = logfile.map(str::trim).filter(|p| !p.is_empty()) {
        return PathBuf::from(explicit);
    }

    if let Some(pid) = pidfile.map(str::trim).filter(|p| !p.is_empty()) {
        // Only the extension changes; a path that does not end in `.pid` just
        // gains `.log`, rather than having `.pid` substituted anywhere inside it.
        return PathBuf::from(match pid.strip_suffix(".pid") {
            Some(stem) => format!("{stem}.log"),
            None => format!("{pid}.log"),
        });
    }

    match home.map(str::trim).filter(|h| !h.is_empty()) {
        Some(h) => PathBuf::from(h)
            .join(".eullm")
            .join("logs")
            .join("eullm.log"),
        None => std::env::temp_dir().join("eullm.log"),
    }
}

fn daemonize(pidfile: &str, log_path: &std::path::Path) {
    use std::io::Write;

    let exe = match std::env::current_exe() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Error: cannot determine executable path: {e}");
            std::process::exit(1);
        }
    };

    // Rebuild args without --daemon and --pidfile.
    let args: Vec<String> = std::env::args()
        .skip(1) // skip argv[0]
        .filter(|a| a != "--daemon")
        .collect();

    // Filter out --pidfile and its value.
    let mut filtered_args = Vec::new();
    let mut skip_next = false;
    for arg in &args {
        if skip_next {
            skip_next = false;
            continue;
        }
        if arg == "--pidfile" {
            skip_next = true;
            continue;
        }
        if arg.starts_with("--pidfile=") {
            continue;
        }
        filtered_args.push(arg.clone());
    }

    // The default destination (`~/.eullm/logs/`) does not exist on a first
    // run, and neither does a directory an operator named on the command line.
    if let Some(parent) = log_path.parent()
        && !parent.as_os_str().is_empty()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        eprintln!(
            "Error: cannot create log directory {}: {e}",
            parent.display()
        );
        std::process::exit(1);
    }

    let log_file = match std::fs::File::create(log_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error: cannot create log file {}: {e}", log_path.display());
            std::process::exit(1);
        }
    };
    let log_err = match log_file.try_clone() {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error: cannot clone log file handle: {e}");
            std::process::exit(1);
        }
    };

    let child = std::process::Command::new(&exe)
        .args(&filtered_args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::from(log_file))
        .stderr(std::process::Stdio::from(log_err))
        .spawn();

    match child {
        Ok(mut child) => {
            let pid = child.id();

            // Do not claim success until the child has survived long enough to
            // have failed. It used to print "daemon started (PID N)" the instant
            // spawn() returned, so a child that died immediately — the port
            // already in use is the common one — still produced a success
            // message, a PID file pointing at a dead process, and an exit code
            // of 0. A caller's teardown then tried to kill a PID that no longer
            // existed while the *previous* server kept answering, so subsequent
            // requests silently went to a server started with different flags.
            // That happened in the wild and quietly invalidated a tester's
            // results across five machines.
            let deadline = std::time::Instant::now() + DAEMON_STARTUP_GRACE;
            while std::time::Instant::now() < deadline {
                match child.try_wait() {
                    Ok(Some(status)) => {
                        eprintln!("Error: the daemon exited immediately ({status}).");
                        // The child's own diagnostics went to the log file, so
                        // show them rather than making the operator go and look.
                        if let Ok(log) = std::fs::read_to_string(log_path) {
                            let tail: Vec<&str> = log.lines().rev().take(10).collect();
                            for line in tail.into_iter().rev() {
                                eprintln!("  {line}");
                            }
                        }
                        eprintln!("  Full log: {}", log_path.display());
                        // No PID file: a stale one is worse than none, because a
                        // stop script will believe it.
                        let _ = std::fs::remove_file(pidfile);
                        std::process::exit(1);
                    }
                    // Still running — good, that is what we want.
                    Ok(None) => std::thread::sleep(std::time::Duration::from_millis(100)),
                    Err(e) => {
                        eprintln!("Error: cannot check on the daemon process: {e}");
                        std::process::exit(1);
                    }
                }
            }

            // Write PID file. A missing or half-written pidfile is worse than
            // none — a stop script trusts it — so fail instead of claiming
            // success, mirroring the log-file handling above.
            //
            // Failing here cannot just exit, the way it can above. This is
            // spawn(), not fork(): the child is a separate process that has
            // already survived DAEMON_STARTUP_GRACE and is serving on the
            // port. Exiting alone would leave it running with nothing on disk
            // pointing at it, and the PID is only printed below — so the
            // operator would be told neither that it is up nor what to kill.
            // Stop it instead, which is the same rule the early-exit path
            // above follows: never leave a state a stop script would misread.
            let pidfile_err = match std::fs::File::create(pidfile) {
                Ok(mut f) => write!(f, "{pid}").err().map(|e| e.to_string()),
                Err(e) => Some(e.to_string()),
            };
            if let Some(e) = pidfile_err {
                eprintln!("Error: cannot write pidfile {pidfile}: {e}");
                eprintln!("  Stopping the daemon (PID {pid}): it would hold the port untracked.");
                let _ = child.kill();
                // Reap it before returning the shell, so a retry does not race
                // a dying process for the port.
                let _ = child.wait();
                // File::create may have succeeded and left an empty file.
                let _ = std::fs::remove_file(pidfile);
                std::process::exit(1);
            }
            println!("eullm daemon started (PID {pid}).");
            println!("  PID file: {pidfile}");
            println!("  Log file: {}", log_path.display());
            println!("  Stop with: kill {pid}");
            std::process::exit(0);
        }
        Err(e) => {
            eprintln!("Error: failed to start daemon: {e}");
            std::process::exit(1);
        }
    }
}

/// Install a signal handler for SIGABRT that prints diagnostic info.
///
/// llama.cpp uses `GGML_ASSERT` which calls `abort()` on failure, producing
/// a core dump with no useful message. This handler prints actionable
/// suggestions before re-raising the signal for the default handler.
#[cfg(unix)]
fn install_abort_handler() {
    unsafe {
        libc::signal(
            libc::SIGABRT,
            abort_handler as *const () as libc::sighandler_t,
        );
    }
}

#[cfg(unix)]
extern "C" fn abort_handler(_sig: libc::c_int) {
    // Only use async-signal-safe operations (write to stderr).
    let msg = b"\n\
==========================================================\n\
EULLM ENGINE CRASHED (SIGABRT)\n\
==========================================================\n\
llama.cpp hit a fatal assertion (GGML_ASSERT).\n\
\n\
Common causes and fixes:\n\
  1. Flash attention not supported by this model/quantization:\n\
     -> Re-run with: eullm run <model> --no-flash-attn\n\
\n\
  2. Out of GPU memory (VRAM):\n\
     -> Reduce batch size: eullm run <model> --batch-size 1\n\
     -> Reduce context:    eullm run <model> --ctx-size 2048\n\
     -> Use CPU only:      eullm run <model> --gpu-layers 0\n\
\n\
  3. Incompatible GGUF file or quantization:\n\
     -> Try a different quantization (Q4_K_M recommended)\n\
\n\
Run with RUST_LOG=debug for more context before the crash.\n\
==========================================================\n";
    unsafe {
        libc::write(2, msg.as_ptr() as *const libc::c_void, msg.len());
        // Re-raise SIGABRT with default handler for core dump.
        libc::signal(libc::SIGABRT, libc::SIG_DFL);
        libc::raise(libc::SIGABRT);
    }
}

#[cfg(not(unix))]
fn install_abort_handler() {
    // No-op on non-Unix platforms.
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1}GB", bytes as f64 / 1_000_000_000.0)
    } else if bytes >= 1_000_000 {
        format!("{:.1}MB", bytes as f64 / 1_000_000.0)
    } else {
        format!("{bytes}B")
    }
}

/// One-shot multimodal probe: load the file at `image_path`, read a prompt
/// from stdin (or use a default), wrap it in the model's own chat template
/// with the mtmd media marker, run a single multimodal generation, stream
/// tokens to stdout, exit.
///
/// This is the MVP entry point for the mtmd integration — deliberately tiny.
/// API/UI multimodal surface is intentionally out of scope here.
#[cfg(feature = "multimodal")]
async fn run_multimodal_oneshot(engine: Arc<InferenceEngine>, image_path: PathBuf) {
    use llama_cpp_2::mtmd::mtmd_default_marker;
    use std::io::Read;
    use tokio::sync::mpsc;

    // 1. Load the media bytes.
    let media_bytes = match std::fs::read(&image_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Error reading {}: {e}", image_path.display());
            std::process::exit(1);
        }
    };
    eprintln!(
        "Media loaded: {} ({} bytes)",
        image_path.display(),
        media_bytes.len()
    );

    // 2. Read the user prompt from stdin (if piped) or fall back to a default.
    let mut user_prompt = String::new();
    if !std::io::IsTerminal::is_terminal(&std::io::stdin()) {
        // Non-TTY: a prompt was piped in. Read it all.
        let _ = std::io::stdin().read_to_string(&mut user_prompt);
    }
    let user_prompt = user_prompt.trim();
    let user_prompt = if user_prompt.is_empty() {
        "Describe this image briefly."
    } else {
        user_prompt
    };

    // 3. Template the turn with the media marker inside the user content,
    //    using the model's own GGUF-embedded Jinja template — the same choice
    //    the API path makes in `multimodal_chat_prompt`. Hardcoding Gemma here
    //    was correct only while Gemma 4 was the sole multimodal model we
    //    shipped; a Qwen VL fed `<start_of_turn>` answers badly and says
    //    nothing about why.
    let marker = mtmd_default_marker();
    let marked = format!("{marker}\n{user_prompt}");
    let pairs: [(&str, &str); 1] = [("user", marked.as_str())];
    let (templated, stop_sequences, response_prefix) =
        match engine.apply_jinja_chat_template(&pairs, true) {
            // The model's own template ends generation on its EOG token, so
            // there is no stop sequence to add on top.
            Some(dynamic) => (dynamic.prompt, Vec::new(), dynamic.preopened),
            None => (
                format!("<start_of_turn>user\n{marked}<end_of_turn>\n<start_of_turn>model\n"),
                vec!["<end_of_turn>".to_string()],
                String::new(),
            ),
        };

    // 4. Build the request and stream the answer to stdout.
    let request = inference::GenerateRequest {
        prompt: templated,
        max_tokens: 512,
        temperature: 0.7,
        raw: true, // already templated, no extra BOS / formatting
        stop_sequences,
        response_prefix,
        ..Default::default()
    };

    let (tx, mut rx) = mpsc::channel(64);
    let eng_for_task = engine.clone();
    let request_for_task = request.clone();
    let media_for_task = vec![media_bytes];
    let join = tokio::task::spawn_blocking(move || {
        // One turn, one attachment, and it is what the prompt asks about.
        eng_for_task.generate_multimodal(&request_for_task, &media_for_task, 1, tx);
    });

    use std::io::Write;
    // Not `stdout().lock()`: the log lines go to stdout too, and the
    // generation thread writes some (`Multimodal stream: …`, or the batch
    // being raised for a large image) before its first token. Held across
    // this loop, the lock left that thread waiting on it and this loop
    // waiting on a token: every `run --image` hung after the prompt, with
    // nothing printed. Locking per write lets both through.
    let mut stdout = std::io::stdout();
    while let Some(ev) = rx.recv().await {
        match ev {
            inference::StreamEvent::Token(t) => {
                let _ = stdout.write_all(t.as_bytes());
                let _ = stdout.flush();
            }
            inference::StreamEvent::Done {
                tokens_generated,
                tokens_prompt,
                duration_ms,
                // The one-shot multimodal probe prints no stop reason.
                stop_reason: _,
                stats,
            } => {
                let _ = writeln!(stdout);
                let _ = writeln!(
                    stdout,
                    "[done — {tokens_generated} tokens, prompt {tokens_prompt} read in {} ms, \
                     {duration_ms} ms]",
                    stats.prompt_time.as_millis()
                );
            }
            inference::StreamEvent::Error(e) => {
                let _ = writeln!(stdout, "\n[error] {e}");
                let _ = join.await;
                std::process::exit(1);
            }
        }
    }
    let _ = join.await;
}

#[cfg(test)]
mod cli_default_parity_tests {
    use super::*;
    use clap::Parser;

    /// The shared runtime flags as `run` or `serve` parsed them.
    ///
    /// Both subcommands now flatten the same `RuntimeOpts`, so this cannot see
    /// two different field sets any more. The tests below are kept regardless:
    /// they assert the *values* a user gets, which is what the divergences they
    /// were written for actually broke, and they would still catch someone
    /// pulling a flag back out of the shared struct.
    fn runtime_opts(argv: &[&str]) -> RuntimeOpts {
        match Cli::parse_from(argv).command.expect("subcommand") {
            Commands::Run { opts, .. } | Commands::Serve { opts, .. } => opts,
            other => panic!(
                "unexpected subcommand: {:?}",
                std::mem::discriminant(&other)
            ),
        }
    }

    /// The KV cache defaults of `run` and `serve`.
    fn kv_defaults(argv: &[&str]) -> (String, String) {
        let o = runtime_opts(argv);
        (o.cache_type_k, o.cache_type_v)
    }

    /// Automatic sizing as both subcommand arms resolve it.
    fn sizing_on(argv: &[&str]) -> bool {
        !runtime_opts(argv).no_fit
    }

    /// Sizing is on by default (0.6.80): the alternative default is an
    /// out-of-memory error at load on any model bigger than the free VRAM,
    /// which is what a user hit while a model swap was choosing `all`.
    #[test]
    fn sizing_is_on_unless_told_otherwise() {
        assert!(sizing_on(&["eullm", "run", "m.gguf"]));
        assert!(sizing_on(&["eullm", "serve"]));
        assert!(sizing_on(&["eullm", "run", "m.gguf", "--fit"]));
    }

    /// An explicit `--gpu-layers` is a ceiling, not an off switch: sizing
    /// keeps running so the count cannot exceed what fits. Turning the
    /// guardrail off exactly when the user steers would hand back the
    /// out-of-memory failure this default exists to prevent — a count
    /// chosen for one model says nothing about the next one loaded.
    #[test]
    fn an_explicit_gpu_layers_keeps_sizing_on() {
        assert!(sizing_on(&["eullm", "run", "m.gguf", "--gpu-layers", "20"]));
        assert!(sizing_on(&["eullm", "serve", "--gpu-layers", "40"]));
        assert_eq!(
            runtime_opts(&["eullm", "run", "m.gguf", "--gpu-layers", "20"]).gpu_layers,
            Some(20)
        );
    }

    /// The ceiling itself: it lowers a computed offload, never raises one,
    /// and a negative value on either side means "no bound" (`-1` = all).
    #[test]
    fn the_gpu_layers_ceiling_only_ever_lowers() {
        use crate::fit::apply_gpu_layers_ceiling as cap;
        // Sizing says 43 fit; the user asked for at most 20.
        assert_eq!(cap(43, 20), 20);
        // The user asked for 60, only 43 fit: the estimate wins.
        assert_eq!(cap(43, 60), 43);
        // No ceiling set: whatever was computed, including "all".
        assert_eq!(cap(43, -1), 43);
        assert_eq!(cap(-1, -1), -1);
        // Everything fits, but the user wants only 20 on the card.
        assert_eq!(cap(-1, 20), 20);
        // CPU-only is a legitimate ceiling.
        assert_eq!(cap(43, 0), 0);
    }

    /// `--mtp-p-min` takes a probability, and 0 — llama.cpp's default,
    /// which drafts the full `--mtp` every step — unless asked.
    #[test]
    fn the_mtp_threshold_is_a_probability() {
        assert_eq!(runtime_opts(&["eullm", "serve"]).mtp_p_min, 0.0);
        let asked = runtime_opts(&["eullm", "serve", "--mtp", "3", "--mtp-p-min", "0.5"]);
        assert_eq!((asked.mtp, asked.mtp_p_min), (3, 0.5));
        for refused in ["1.5", "-0.1", "half"] {
            assert!(Cli::try_parse_from(["eullm", "serve", "--mtp-p-min", refused]).is_err());
        }
    }

    /// `--mtp-model` is unset unless asked, and exists on both subcommands.
    #[test]
    fn the_mtp_head_file_is_unset_unless_asked() {
        assert_eq!(runtime_opts(&["eullm", "serve"]).mtp_model, None);
        let asked = runtime_opts(&["eullm", "serve", "--mtp", "2", "--mtp-model", "head.gguf"]);
        assert_eq!(asked.mtp_model, Some(PathBuf::from("head.gguf")));
        let run = runtime_opts(&["eullm", "run", "m.gguf", "--mtp-model", "head.gguf"]);
        assert_eq!(run.mtp_model, Some(PathBuf::from("head.gguf")));
    }

    #[test]
    fn the_expert_cache_is_off_unless_asked_and_takes_auto_or_mib() {
        assert_eq!(runtime_opts(&["eullm", "serve"]).moe_cache, None);
        assert_eq!(
            runtime_opts(&["eullm", "serve", "--moe-cache", "auto"]).moe_cache,
            Some(fit::MoeCache::Auto)
        );
        assert_eq!(
            runtime_opts(&["eullm", "run", "x", "--moe-cache", "6000"]).moe_cache,
            Some(fit::MoeCache::Mib(6000))
        );
        assert!(Cli::try_parse_from(["eullm", "serve", "--moe-cache", "0"]).is_err());
    }

    /// `--moe-prefetch` is on with four slots unless told otherwise, on both
    /// subcommands; 0 turns it off, and one slot, or more than eight, is
    /// refused rather than quietly changed.
    #[test]
    fn the_prefetch_takes_four_slots_unless_told_otherwise() {
        assert_eq!(runtime_opts(&["eullm", "serve"]).moe_prefetch, 4);
        assert_eq!(runtime_opts(&["eullm", "run", "x"]).moe_prefetch, 4);
        assert_eq!(
            runtime_opts(&["eullm", "serve", "--moe-prefetch", "0"]).moe_prefetch,
            0
        );
        assert_eq!(
            runtime_opts(&["eullm", "run", "x", "--moe-prefetch", "8"]).moe_prefetch,
            8
        );
        for refused in ["1", "9", "-1", "four"] {
            assert!(
                Cli::try_parse_from(["eullm", "serve", "--moe-prefetch", refused]).is_err(),
                "accepted --moe-prefetch {refused}"
            );
        }
    }

    /// `--no-mmap` and `--mmap` are off unless asked, on both subcommands,
    /// and cannot be asked together.
    #[test]
    fn the_model_file_is_mapped_unless_no_mmap_is_asked() {
        let neither = runtime_opts(&["eullm", "serve"]);
        assert!(!neither.no_mmap && !neither.mmap);
        assert!(runtime_opts(&["eullm", "serve", "--no-mmap"]).no_mmap);
        assert!(runtime_opts(&["eullm", "run", "x", "--no-mmap"]).no_mmap);
        assert!(runtime_opts(&["eullm", "serve", "--mmap"]).mmap);
        assert!(Cli::try_parse_from(["eullm", "serve", "--mmap", "--no-mmap"]).is_err());
    }

    #[test]
    fn the_model_is_read_ahead_only_when_asked() {
        use readahead::LoadThreads;
        assert_eq!(
            runtime_opts(&["eullm", "serve"]).load_threads,
            LoadThreads::Fixed(0)
        );
        assert_eq!(
            runtime_opts(&["eullm", "run", "x", "--load-threads", "auto"]).load_threads,
            LoadThreads::Auto
        );
        assert_eq!(
            runtime_opts(&["eullm", "serve", "--load-threads", "0"]).load_threads,
            LoadThreads::Fixed(0)
        );
        assert_eq!(
            runtime_opts(&["eullm", "run", "x", "--load-threads", "16"]).load_threads,
            LoadThreads::Fixed(16)
        );
        assert!(Cli::try_parse_from(["eullm", "serve", "--load-threads", "many"]).is_err());
    }

    /// `--n-ubatch` is unset unless asked, on both subcommands: llama.cpp's
    /// 512 then, or an expert cache's choice (`fit::MOE_CACHE_N_UBATCH`). A
    /// micro-batch above `--n-batch` raises the batch to hold it, and a value
    /// outside 32..=16384 is refused when the command line is read.
    #[test]
    fn the_micro_batch_defaults_to_llama_cpps_and_raises_the_batch() {
        assert_eq!(runtime_opts(&["eullm", "run", "m.gguf"]).n_ubatch, None);
        assert_eq!(runtime_opts(&["eullm", "serve"]).n_ubatch, None);
        let asked = runtime_opts(&["eullm", "serve", "--n-ubatch", "4096"]);
        assert_eq!(asked.n_ubatch, Some(4096));
        assert_eq!(launch_n_batch(asked.n_batch, 4096), 4096);
        let larger = runtime_opts(&[
            "eullm",
            "run",
            "m.gguf",
            "--n-batch",
            "8192",
            "--n-ubatch",
            "4096",
        ]);
        assert_eq!(larger.n_ubatch, Some(4096));
        assert_eq!(launch_n_batch(larger.n_batch, 4096), 8192);
        for refused in ["16", "32768"] {
            assert!(Cli::try_parse_from(["eullm", "serve", "--n-ubatch", refused]).is_err());
        }
    }

    #[test]
    fn no_fit_turns_sizing_off_on_its_own() {
        assert!(!sizing_on(&["eullm", "run", "m.gguf", "--no-fit"]));
        assert!(!sizing_on(&["eullm", "serve", "--no-fit"]));
    }

    /// Neither flag leaves the projector to sizing; either one decides, on
    /// both subcommands, and asking for both is refused rather than resolved
    /// by whichever clap happened to read last.
    #[test]
    fn the_projector_flags_force_a_placement_and_exclude_each_other() {
        for sub in [&["eullm", "run", "m.gguf"][..], &["eullm", "serve"][..]] {
            let o = runtime_opts(sub);
            assert!(
                !o.mmproj_offload && !o.no_mmproj_offload,
                "unset by default"
            );

            let on = runtime_opts(&[sub, &["--mmproj-offload"]].concat());
            assert!(on.mmproj_offload && !on.no_mmproj_offload);

            let off = runtime_opts(&[sub, &["--no-mmproj-offload"]].concat());
            assert!(off.no_mmproj_offload && !off.mmproj_offload);

            let both =
                Cli::try_parse_from([sub, &["--mmproj-offload", "--no-mmproj-offload"]].concat());
            assert!(both.is_err(), "both flags at once must be refused");
        }
    }

    #[test]
    fn eullm_unload_names_every_model_the_server_unloaded() {
        let report = |body: serde_json::Value, model| unload_report(&body, model);
        assert_eq!(
            report(
                serde_json::json!({ "unloaded": "a", "unloaded_all": ["a", "b"] }),
                None
            ),
            ["Unloaded 'a'.", "Unloaded 'b'.", "VRAM freed."]
        );
        // A server from before `unloaded_all`.
        assert_eq!(
            report(serde_json::json!({ "unloaded": "a" }), None),
            ["Unloaded 'a'.", "VRAM freed."]
        );
        assert_eq!(
            report(
                serde_json::json!({ "unloaded": null, "unloaded_all": [] }),
                None
            ),
            ["No model was loaded."]
        );
        assert_eq!(
            report(serde_json::json!({ "unloaded": null }), Some("qwen3-8b")),
            ["'qwen3-8b' was not loaded."]
        );
        let parsed = Cli::parse_from(["eullm", "unload", "--model", "qwen3-8b"]);
        assert!(matches!(
            parsed.command,
            Some(Commands::Unload { model: Some(ref m), .. }) if m == "qwen3-8b"
        ));
    }

    /// One generation model at a time unless asked: a second one only ever
    /// gets what the first left, so a default above 1 would divide the card
    /// for whoever loads second. Sixteen at most, and never none.
    #[test]
    fn max_loaded_models_defaults_to_one_on_both_commands() {
        for sub in [&["eullm", "run", "m.gguf"][..], &["eullm", "serve"][..]] {
            assert_eq!(runtime_opts(sub).max_loaded_models, 1);
            let four = runtime_opts(&[sub, &["--max-loaded-models", "4"]].concat());
            assert_eq!(four.max_loaded_models, 4);
            for refused in ["0", "17", "-1", "two"] {
                let parsed = Cli::try_parse_from([sub, &["--max-loaded-models", refused]].concat());
                assert!(
                    parsed.is_err(),
                    "--max-loaded-models {refused} must be refused"
                );
            }
        }
    }

    /// `--auto-model` repeats, keeping its order and the text after `=`
    /// whole; `--auto-timeout-ms` is 1000 unless given, from 10 to 60000.
    #[test]
    fn auto_model_repeats_and_its_timeout_is_bounded() {
        for sub in [&["eullm", "run", "m.gguf"][..], &["eullm", "serve"][..]] {
            let plain = runtime_opts(sub);
            assert!(plain.auto_model.is_empty());
            assert_eq!(plain.auto_timeout_ms, 1000);
            let routed = runtime_opts(
                &[
                    sub,
                    &[
                        "--auto-model",
                        "qwen3-4b=Short requests, simple facts",
                        "--auto-model",
                        "qwen3-8b",
                        "--auto-timeout-ms",
                        "250",
                    ],
                ]
                .concat(),
            );
            assert_eq!(
                routed.auto_model,
                ["qwen3-4b=Short requests, simple facts", "qwen3-8b"]
            );
            assert_eq!(routed.auto_timeout_ms, 250);
            for refused in ["9", "60001", "-1", "soon"] {
                let parsed = Cli::try_parse_from([sub, &["--auto-timeout-ms", refused]].concat());
                assert!(
                    parsed.is_err(),
                    "--auto-timeout-ms {refused} must be refused"
                );
            }
        }
    }

    /// `--default-model` is on both commands, and unset unless given.
    #[test]
    fn default_model_is_a_shared_flag_unset_by_default() {
        for sub in [&["eullm", "run", "m.gguf"][..], &["eullm", "serve"][..]] {
            assert_eq!(runtime_opts(sub).default_model, None);
            let named = runtime_opts(&[sub, &["--default-model", "qwen3-8b"]].concat());
            assert_eq!(named.default_model.as_deref(), Some("qwen3-8b"));
        }
    }

    /// `--gpu-layers` unset must reach the engine as the old default, so
    /// disabling sizing changes what decides the split, not the split.
    #[test]
    fn unset_gpu_layers_still_means_all() {
        let o = runtime_opts(&["eullm", "run", "m.gguf"]);
        assert_eq!(o.gpu_layers.unwrap_or(-1), -1);
    }

    #[test]
    fn run_and_serve_share_every_runtime_flag() {
        // One assertion covering all twenty shared defaults at once. Since
        // both subcommands flatten the same struct this can no longer fail by
        // drift — which is the point of H3-H — but it fails loudly if someone
        // pulls a flag back out into one variant, and it is the shortest
        // statement of the property the mandatory parity rule in `CLAUDE.md`
        // is trying to express.
        assert_eq!(
            runtime_opts(&["eullm", "run", "some-model"]),
            runtime_opts(&["eullm", "serve"]),
            "run and serve must agree on every shared runtime flag"
        );
    }

    /// Where a daemon's output lands (#354). The old default derived the log
    /// path from the PID file, so it went to `/tmp` — small on many systems,
    /// and cleared on reboot or under space pressure, which is exactly when
    /// the log is the only record of what the daemon was doing.
    #[test]
    fn the_daemon_log_defaults_under_the_eullm_home() {
        assert_eq!(
            resolve_daemon_log_path(None, None, Some("/home/u")),
            PathBuf::from("/home/u/.eullm/logs/eullm.log")
        );
    }

    /// An explicit `--logfile` beats everything, including a `--pidfile`
    /// pointing somewhere else.
    #[test]
    fn an_explicit_logfile_wins() {
        assert_eq!(
            resolve_daemon_log_path(
                Some("/var/log/eullm.log"),
                Some("/run/x.pid"),
                Some("/home/u")
            ),
            PathBuf::from("/var/log/eullm.log")
        );
    }

    /// The pre-0.6.81 convention, kept for anyone who already redirects the
    /// PID file: the log stays next to it. Only the extension is replaced —
    /// `.pid` occurring earlier in the path is left alone.
    #[test]
    fn a_custom_pidfile_still_carries_the_log_with_it() {
        assert_eq!(
            resolve_daemon_log_path(None, Some("/var/run/eullm/eullm.pid"), Some("/home/u")),
            PathBuf::from("/var/run/eullm/eullm.log")
        );
        assert_eq!(
            resolve_daemon_log_path(None, Some("/srv/a.pid.d/server"), None),
            PathBuf::from("/srv/a.pid.d/server.log")
        );
    }

    /// A service account with no home directory still gets a daemon: the log
    /// falls back to the temp directory rather than the start failing.
    #[test]
    fn no_home_falls_back_to_the_temp_directory() {
        assert_eq!(
            resolve_daemon_log_path(None, None, None),
            std::env::temp_dir().join("eullm.log")
        );
        // An empty HOME is the same as no HOME, not a path relative to `/`.
        assert_eq!(
            resolve_daemon_log_path(None, None, Some("  ")),
            std::env::temp_dir().join("eullm.log")
        );
    }

    /// Both spellings of a flag value, since the `--daemon` branch reads argv
    /// itself — clap has not run at that point and cannot.
    #[test]
    fn raw_argv_reads_both_flag_spellings() {
        let argv: Vec<String> = [
            "eullm",
            "serve",
            "--logfile",
            "/a/b.log",
            "--pidfile=/c/d.pid",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert_eq!(arg_value(&argv, "--logfile"), Some("/a/b.log"));
        assert_eq!(arg_value(&argv, "--pidfile"), Some("/c/d.pid"));
        assert_eq!(arg_value(&argv, "--ctx-size"), None);
    }

    /// `--logfile` is a shared runtime flag, so it exists on both commands.
    #[test]
    fn logfile_is_available_to_run_and_serve() {
        assert_eq!(runtime_opts(&["eullm", "run", "m.gguf"]).logfile, None);
        assert_eq!(
            runtime_opts(&["eullm", "serve", "--logfile", "/var/log/eullm.log"]).logfile,
            Some("/var/log/eullm.log".to_string())
        );
        assert_eq!(
            runtime_opts(&["eullm", "run", "m.gguf", "--logfile", "/var/log/eullm.log"]).logfile,
            Some("/var/log/eullm.log".to_string())
        );
    }

    #[test]
    fn a_shared_flag_parses_the_same_on_both_commands() {
        // Defaults agreeing is not the same as the flags existing on both.
        // `--cache-type-k`, `--cache-type-v` and `--gpu-layers` were once
        // accepted by `run` and rejected by `serve`, so a `serve` deployment
        // had no way to override them at all.
        let flags = ["--gpu-layers", "20", "--ctx-size", "8192", "--cpu-moe"];
        let mut run_argv = vec!["eullm", "run", "some-model"];
        run_argv.extend_from_slice(&flags);
        let mut serve_argv = vec!["eullm", "serve"];
        serve_argv.extend_from_slice(&flags);
        assert_eq!(runtime_opts(&run_argv), runtime_opts(&serve_argv));
    }

    #[test]
    fn run_and_serve_default_to_the_same_kv_cache_types() {
        // Until v0.6.36 `serve` defaulted to q8_0 keys and q4_0 values while
        // `run` defaulted to f16/f16. The same model therefore produced
        // different output quality depending on which command started it, with
        // nothing in the output saying so — and a four-bit value cache is
        // aggressive enough for Qwen3 that external testing saw degraded
        // generations from it (issue #140).
        //
        // Quantizing the KV cache stays a supported and genuinely useful trade
        // at long context. It just has to be the operator's choice rather than
        // a side effect of the command name.
        let run = kv_defaults(&["eullm", "run", "some-model"]);
        let serve = kv_defaults(&["eullm", "serve"]);
        assert_eq!(
            run, serve,
            "run and serve must agree on the default KV cache types"
        );
        assert_eq!(run, ("f16".to_string(), "f16".to_string()));
    }

    #[test]
    fn an_explicit_kv_cache_type_still_overrides_the_default() {
        // The point of the change above is the *default*, not the capability.
        let serve = kv_defaults(&[
            "eullm",
            "serve",
            "--cache-type-k",
            "q8_0",
            "--cache-type-v",
            "q4_0",
        ]);
        assert_eq!(serve, ("q8_0".to_string(), "q4_0".to_string()));
    }

    #[test]
    fn run_and_serve_agree_on_context_size_too() {
        // Same class of divergence, checked while we are here: a default that
        // differs between the two commands is invisible to whoever hits it.
        let run = runtime_opts(&["eullm", "run", "some-model"]).ctx_size;
        let serve = runtime_opts(&["eullm", "serve"]).ctx_size;
        assert_eq!(
            run, serve,
            "run and serve must agree on the default context"
        );
    }
}

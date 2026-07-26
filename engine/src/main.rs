mod api;
mod audit;
mod chat_template;
mod fit;
mod gguf_patch;
mod inference;
mod models;
mod picker;
mod registry;
mod tools;
mod ui;

use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use models::{ModelStore, catalog};

use crate::inference::{BatchScheduler, InferenceConfig, InferenceEngine, SchedulerConfig};

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
#[cfg(feature = "cuda")]
const VERSION_STRING: &str = concat!(env!("CARGO_PKG_VERSION"), " (CUDA)");
#[cfg(feature = "metal")]
const VERSION_STRING: &str = concat!(env!("CARGO_PKG_VERSION"), " (Metal)");
#[cfg(feature = "rocm")]
const VERSION_STRING: &str = concat!(env!("CARGO_PKG_VERSION"), " (ROCm)");
#[cfg(feature = "vulkan")]
const VERSION_STRING: &str = concat!(env!("CARGO_PKG_VERSION"), " (Vulkan)");
#[cfg(not(any(
    feature = "cuda",
    feature = "metal",
    feature = "rocm",
    feature = "vulkan"
)))]
const VERSION_STRING: &str = concat!(env!("CARGO_PKG_VERSION"), " (CPU)");

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

        /// Port for the API server
        #[arg(short, long, default_value_t = 11434)]
        port: u16,

        /// Replace existing service on the port
        #[arg(long)]
        replace: bool,

        /// Number of GPU layers to offload (-1 = all, 0 = CPU only)
        #[arg(long, default_value_t = -1, allow_hyphen_values = true)]
        gpu_layers: i32,

        /// Auto-fit GPU layers to available VRAM (CUDA builds only). Probes
        /// free VRAM and the model's layer count, then offloads as many layers
        /// as fit. Opt-in: without it, --gpu-layers is used as-is. If VRAM
        /// can't be probed it falls back to --gpu-layers.
        #[arg(long)]
        fit: bool,

        /// With --fit, refuse to load (instead of offloading a partial split
        /// or falling back) when the model does not fully fit on the GPU.
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

        /// Number of CPU threads (default: all available)
        #[arg(short, long)]
        threads: Option<u32>,

        /// Maximum concurrent requests served by the continuous-batching scheduler.
        ///
        /// `--ctx-size` is split evenly across these slots (so per-sequence context
        /// = ctx_size / batch_size). Default 1 in interactive `run` mode means each
        /// chat gets the full context; raise to 4–16 if you're using this engine as
        /// a backend for multiple simultaneous users.
        #[arg(long, default_value_t = 1)]
        batch_size: usize,

        /// Disable flash attention (enabled by default for faster inference)
        #[arg(long)]
        no_flash_attn: bool,

        /// Prompt processing batch size (tokens per eval during prefill)
        #[arg(long, default_value_t = 2048)]
        n_batch: u32,

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

        /// (Multimodal MVP, --features multimodal builds only.) Path to an
        /// image or audio file to send together with the first prompt.
        /// Triggers the multimodal inference path (mtmd) which requires the
        /// model's mmproj projector to be available; for catalog models it
        /// is auto-downloaded during `pull`. The HTTP API also routes media
        /// (web chat / `/api/chat` `images`); this flag is the CLI one-shot.
        #[arg(long, value_name = "PATH")]
        image: Option<PathBuf>,

        /// Enable extra internal diagnostics for the Rust engine layer. Off
        /// by default (zero added per-token cost, matches upstream
        /// llama.cpp). Today this enables a NaN/Inf scan of every generated
        /// token's logits before sampling — added to help diagnose garbage
        /// output (issue #140) — at the cost of one extra linear scan over
        /// the vocab per token. Not a general log-level flag; use RUST_LOG
        /// for that.
        #[arg(long)]
        rust_debug: bool,
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
        /// Port for the API server
        #[arg(short, long, default_value_t = 11434)]
        port: u16,

        /// Replace existing service on the port
        #[arg(long)]
        replace: bool,

        /// Enable continuous batching with N max concurrent requests (0 = sequential)
        #[arg(long, default_value_t = 8)]
        batch_size: usize,

        /// Number of GPU layers to offload (-1 = all, 0 = CPU only). Applied
        /// to every model this server loads or swaps to. See `eullm run
        /// --help` for the full rationale.
        #[arg(long, default_value_t = -1, allow_hyphen_values = true)]
        gpu_layers: i32,

        /// Context window size. Applied to every model this server loads or
        /// swaps to (a request's `ctx_size` field overrides it for that
        /// swap). See `eullm run --help` for the full rationale.
        #[arg(short, long, default_value_t = 4096)]
        ctx_size: u32,

        /// Number of CPU threads (default: all available). Applied to every
        /// model this server loads or swaps to.
        #[arg(short, long)]
        threads: Option<u32>,

        /// Disable flash attention (enabled by default). Applied to every
        /// model this server loads or swaps to.
        #[arg(long)]
        no_flash_attn: bool,

        /// Prompt processing batch size (tokens per eval during prefill).
        /// Applied to every model this server loads or swaps to.
        #[arg(long, default_value_t = 2048)]
        n_batch: u32,

        /// KV cache type for keys. Options: f16 (default, best quality and
        /// GPU compat), f32, q8_0, q4_0, q4_1, q5_0, q5_1. Applied to every
        /// model this server loads or swaps to.
        ///
        /// Changed to f16 in v0.6.36. `serve` used to default to q8_0 keys
        /// and q4_0 values while `run` defaulted to f16/f16 — so the same
        /// model gave different output quality depending on which command
        /// started it, silently, and nothing in the output said so. A
        /// four-bit value cache is aggressive for Qwen3 in particular, and
        /// external testing on issue #140 saw degraded generations with
        /// exactly that setting. Quantizing the KV cache is a real and useful
        /// trade at long context, but it has to be a choice the operator
        /// makes, not one a command name makes for them.
        #[arg(long, default_value = "f16")]
        cache_type_k: String,

        /// KV cache type for values. Options: f16 (default, best quality and
        /// GPU compat), f32, q8_0, q4_0, q4_1, q5_0, q5_1. Applied to every
        /// model this server loads or swaps to. See `--cache-type-k` for why
        /// this defaults to f16 since v0.6.36.
        #[arg(long, default_value = "f16")]
        cache_type_v: String,

        /// Enable transparent web browsing: URLs in user messages are
        /// fetched and their content is injected into the prompt before
        /// inference. Applied to every model this server loads or swaps to.
        #[arg(long)]
        web: bool,

        /// For MoE models: keep expert tensors on CPU RAM, attention +
        /// embeddings + KV cache on GPU. Applied to every model this server
        /// loads or swaps to. See `eullm run --help` for the full rationale.
        #[arg(long)]
        cpu_moe: bool,

        /// For MoE models: keep expert tensors on CPU RAM for only the
        /// first N transformer layers, leaving the rest on GPU. Applied to
        /// every model this server loads or swaps to. Mutually exclusive
        /// with --cpu-moe. See `eullm run --help` for the full rationale.
        #[arg(long, default_value_t = 0)]
        n_cpu_moe: u32,

        /// Recurrent-state rollback window for hybrid/recurrent
        /// architectures. 0 (default) strongly recommended — this is a
        /// speculative-decoding rollback primitive upstream, not a
        /// conversation-caching one. Applied to every model this server
        /// loads or swaps to. See `eullm run --help` for the full rationale.
        #[arg(long, default_value_t = 0)]
        rs_seq: u32,

        /// Max full-sequence-state checkpoints kept for prompt-prefix
        /// restore. Applied to every model this server loads or swaps to.
        /// See `eullm run --help` for the full rationale.
        #[arg(long, default_value_t = 0)]
        ctx_checkpoints: usize,

        /// Minimum new tokens since the closest existing checkpoint before
        /// taking another one. See `eullm run --help` for the full rationale.
        #[arg(long, default_value_t = 8192)]
        checkpoint_min_step: u32,

        /// Enable the embedded chat UI (off by default for headless serve).
        /// Pass --ui to also expose the chat at http://localhost:<ui-port>/.
        #[arg(long)]
        ui: bool,

        /// Port for the embedded chat UI when --ui is set. Default 11435.
        #[arg(long, default_value_t = 11435)]
        ui_port: u16,

        /// Run as a background daemon (writes PID to --pidfile)
        #[arg(long)]
        daemon: bool,

        /// PID file path (used with --daemon)
        #[arg(long, default_value = DEFAULT_PIDFILE)]
        pidfile: String,

        /// Enable extra internal diagnostics for the Rust engine layer.
        /// Applied to every model this server loads or swaps to. Off by
        /// default (zero added per-token cost, matches upstream llama.cpp).
        /// See `eullm run --help` for the full rationale.
        #[arg(long)]
        rust_debug: bool,
    },
    /// Unload the currently loaded model from a running eullm server,
    /// freeing its VRAM — without restarting the server.
    ///
    /// A later request with a `model` field (or another `eullm run
    /// <model>`) loads a model back in. Useful for temporarily handing GPU
    /// memory to another process — e.g. an embedding model needed during
    /// RAG document ingestion — then reloading the LLM once it's done.
    ///
    /// Examples:
    ///   eullm unload
    ///   eullm unload --port 11500
    Unload {
        /// Port of the running eullm API server
        #[arg(short, long, default_value_t = 11434)]
        port: u16,
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
            // Find --pidfile value.
            let pidfile = args
                .iter()
                .position(|a| a == "--pidfile")
                .and_then(|i| args.get(i + 1))
                .map(|s| s.as_str())
                .or_else(|| {
                    args.iter()
                        .find(|a| a.starts_with("--pidfile="))
                        .map(|a| a.strip_prefix("--pidfile=").unwrap())
                })
                .unwrap_or(DEFAULT_PIDFILE);
            daemonize(pidfile);
            // daemonize exits the parent — child continues below without --daemon.
        }
    }

    tracing_subscriber::fmt()
        .with_env_filter(
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
            Some(picker::Picked::Local(path)) => Commands::Run {
                model: Some(path.to_string_lossy().into_owned()),
                port: 11434,
                replace: false,
                gpu_layers: -1,
                fit: true,
                fit_strict: false,
                cpu_moe: false,
                n_cpu_moe: 0,
                rs_seq: 0,
                ctx_checkpoints: 0,
                checkpoint_min_step: 8192,
                ctx_size: 4096,
                threads: None,
                batch_size: 1,
                no_flash_attn: false,
                n_batch: 2048,
                cache_type_k: "f16".into(),
                cache_type_v: "f16".into(),
                web: false,
                no_ui: false,
                cli: false,
                ui_port: 11435,
                daemon: false,
                pidfile: DEFAULT_PIDFILE.into(),
                image: None,
                rust_debug: false,
            },
            Some(picker::Picked::Catalog(entry)) => Commands::Run {
                model: Some(entry.id.clone()),
                port: 11434,
                replace: false,
                gpu_layers: -1,
                fit: true,
                fit_strict: false,
                cpu_moe: false,
                n_cpu_moe: 0,
                rs_seq: 0,
                ctx_checkpoints: 0,
                checkpoint_min_step: 8192,
                ctx_size: 4096,
                threads: None,
                batch_size: 1,
                no_flash_attn: false,
                n_batch: 2048,
                cache_type_k: "f16".into(),
                cache_type_v: "f16".into(),
                web: false,
                no_ui: false,
                cli: false,
                ui_port: 11435,
                daemon: false,
                pidfile: DEFAULT_PIDFILE.into(),
                image: None,
                rust_debug: false,
            },
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
            port,
            replace,
            gpu_layers,
            fit,
            fit_strict,
            cpu_moe,
            n_cpu_moe,
            rs_seq,
            ctx_checkpoints,
            checkpoint_min_step,
            ctx_size,
            threads,
            batch_size,
            no_flash_attn,
            n_batch,
            cache_type_k,
            cache_type_v,
            web,
            no_ui,
            cli,
            ui_port,
            daemon,
            pidfile,
            image,
            rust_debug,
        } => {
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
            let _ = (daemon, pidfile);
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
            // same correction also applies inside `swap_model` so it can't
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
                fit_strict,
                cpu_moe,
                n_cpu_moe,
                rs_seq,
                ctx_checkpoints,
                checkpoint_min_step,
                ctx_size,
                threads,
                batch_size,
                !no_flash_attn,
                n_batch,
                ctk,
                ctv,
                web,
                ui_port_opt,
                open_chat,
                image,
                rust_debug,
            )
            .await;
        }
        Commands::List => cmd_list(&store),
        Commands::Show { model } => cmd_show(&store, &model),
        Commands::Rm { model, force } => cmd_rm(&store, &model, force),
        Commands::Serve {
            port,
            replace,
            batch_size,
            gpu_layers,
            ctx_size,
            threads,
            no_flash_attn,
            n_batch,
            cache_type_k,
            cache_type_v,
            web,
            cpu_moe,
            n_cpu_moe,
            rs_seq,
            ctx_checkpoints,
            checkpoint_min_step,
            ui,
            ui_port,
            daemon,
            pidfile,
            rust_debug,
        } => {
            let _ = (daemon, pidfile);
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
                ctx_size,
                threads,
                !no_flash_attn,
                n_batch,
                ctk,
                ctv,
                web,
                cpu_moe,
                n_cpu_moe,
                rs_seq,
                ctx_checkpoints,
                checkpoint_min_step,
                rust_debug,
            )
            .await;
        }
        Commands::Unload { port } => cmd_unload(port).await,
        Commands::ImportOllama { model, ollama_dir } => {
            cmd_import_ollama(&store, &model, ollama_dir.as_deref())
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
}

/// `eullm pull` entry point: if `model` is `None`, open the picker filtered
/// to catalog selections; otherwise just call `cmd_pull` synchronously.
async fn cmd_pull_maybe(store: &ModelStore, model: Option<&str>) {
    if let Some(name) = model {
        cmd_pull(store, name).await;
        return;
    }
    match picker::pick(store).await {
        Some(picker::Picked::Catalog(entry)) => cmd_pull(store, &entry.id).await,
        Some(picker::Picked::Local(p)) => {
            println!("That model is already local: {}", p.display());
        }
        Some(picker::Picked::Url(url)) => cmd_pull_url(store, &url).await,
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

/// Derive a filesystem-safe model id from a HuggingFace ref. Uses the repo
/// name (last path segment), lowercased and sanitized like `url_to_model_id`,
/// with the quant appended when one was requested so different quants of the
/// same repo coexist:  `hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M` → `qwen3-8b-gguf-q4_k_m`.
fn hf_ref_to_model_id(hf: &registry::HfRef) -> String {
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

/// Pull a GGUF from a HuggingFace repo shorthand, outside the catalog.
///
/// Resolves the repo's `.gguf` siblings via the HF API, picks one (by the
/// requested `:quant` or a sensible default), downloads it into the store
/// under an id derived from the repo + quant, and writes an external manifest.
/// On ambiguity (multiple matches or sharded multi-file gguf) it prints the
/// available filenames and exits, asking the user to re-run with `:<quant>`.
async fn cmd_pull_hf(store: &ModelStore, hf: &registry::HfRef) {
    let id = hf_ref_to_model_id(hf);

    if let Some(gguf) = store.gguf_path(&id) {
        println!("Model '{id}' is already downloaded.");
        println!("  GGUF: {}", gguf.display());
        println!("\nRun with: eullm run {id}");
        return;
    }

    println!("Resolving HuggingFace repo: {}", hf.repo);
    let filename = match registry::resolve_hf_gguf(hf).await {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Could not resolve a GGUF to download: {e}");
            std::process::exit(1);
        }
    };

    println!("Pulling from HuggingFace: {} ({})", hf.repo, filename);
    println!("  Storing as: {id}");
    println!("  (off-catalog model — no license/VRAM metadata available)");
    println!();

    let model_dir = store.model_path(&id);
    let gguf_dest = model_dir.join(&filename);

    let result = {
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
        download_from_huggingface(&hf.repo, &filename, &gguf_dest, None, Some(progress)).await
    };
    eprintln!();

    match result {
        Ok(()) => {
            let size = std::fs::metadata(&gguf_dest).map(|m| m.len()).unwrap_or(0);
            match store.write_external_manifest(&id, &filename, &hf.original, size) {
                Ok(_) => {
                    println!("  Done. Model ready.");
                    println!("\nRun with: eullm run {id}");
                }
                Err(e) => eprintln!("Warning: download succeeded but manifest write failed: {e}"),
            }
        }
        Err(e) => {
            eprintln!("Download failed: {e}");
            let _ = store.delete(&id);
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
async fn cmd_pull_url(store: &ModelStore, url: &str) {
    let (id, filename) = url_to_model_id(url);

    if let Some(gguf) = store.gguf_path(&id) {
        println!("Model '{id}' is already downloaded.");
        println!("  GGUF: {}", gguf.display());
        println!("\nRun with: eullm run {id}");
        return;
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
            match store.write_external_manifest(&id, &filename, url, size) {
                Ok(_) => {
                    println!("  Done. Model ready.");
                    println!("\nRun with: eullm run {id}");
                }
                Err(e) => eprintln!("Warning: download succeeded but manifest write failed: {e}"),
            }
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

async fn cmd_pull(store: &ModelStore, model: &str) {
    if is_url(model) {
        cmd_pull_url(store, model).await;
        return;
    }

    if let Some(hf) = registry::parse_hf_ref(model) {
        cmd_pull_hf(store, &hf).await;
        return;
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
                let gguf_name = gguf.file_name().and_then(|s| s.to_str());
                if let Err(e) =
                    store.write_manifest(entry, "ready", gguf_name, Some(&mmproj_filename))
                {
                    eprintln!("  Warning: mmproj downloaded but manifest update failed: {e}");
                }
            }
        }

        println!("\nRun with: eullm run {model}");
        return;
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

        match store.write_manifest(entry, "metadata_only", None, None) {
            Ok(path) => println!("  Manifest saved to {}", path.display()),
            Err(e) => {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    // Download GGUF from HuggingFace
    let short_name = entry.id.as_str();
    let model_dir = store.model_path(&entry.id);
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
            ) {
                Ok(_) => {
                    println!("  Done. Model ready.");
                    println!("\nRun with: eullm run {}", short_name);
                }
                Err(e) => {
                    eprintln!("Warning: download succeeded but manifest write failed: {e}");
                }
            }
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

fn cmd_list(store: &ModelStore) {
    match store.list() {
        Ok(models) if models.is_empty() => {
            println!("No models installed.");
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
                println!(
                    "{:<24} {:>8} {:>4}GB  {:<16} {}",
                    id, size, m.vram_gb, m.status, desc
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
        Ok(Some(freed)) => println!(
            "Removed '{}' ({} freed).",
            manifest.name,
            format_bytes(freed)
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
    cmd.spawn().map(|_| ())
}

#[allow(clippy::too_many_arguments)]
async fn cmd_run(
    store: &ModelStore,
    model: &str,
    port: u16,
    replace: bool,
    gpu_layers: i32,
    fit: bool,
    fit_strict: bool,
    cpu_moe: bool,
    n_cpu_moe: u32,
    rs_seq: u32,
    ctx_checkpoints: usize,
    checkpoint_min_step: u32,
    ctx_size: u32,
    threads: Option<u32>,
    batch_size: usize,
    flash_attn: bool,
    n_batch: u32,
    cache_type_k: inference::KvCacheType,
    cache_type_v: inference::KvCacheType,
    web: bool,
    ui_port: Option<u16>,
    open_chat: bool,
    image: Option<PathBuf>,
    rust_debug: bool,
) {
    // `--image` is a one-shot multimodal probe: load, send the bytes + prompt,
    // print the output, exit. Forces sequential mode (the scheduler does not
    // yet route media) and skips port binding because we won't serve an API.
    let multimodal_oneshot = image.is_some();
    let batch_size = if multimodal_oneshot { 0 } else { batch_size };
    let ui_port = if multimodal_oneshot { None } else { ui_port };

    // `--fit` may override this below once the GGUF file is resolved; until
    // then it is exactly the user-provided `--gpu-layers`.
    let mut gpu_layers = gpu_layers;

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
    let mut engine: Option<Arc<InferenceEngine>> = None;
    let mut scheduler: Option<inference::SchedulerHandle> = None;
    let mut kv_k_mib: f64 = 0.0;
    let mut kv_v_mib: f64 = 0.0;

    let resolved_threads = threads.unwrap_or_else(inference::default_thread_count);

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
            println!("Model not found locally. Pulling from HuggingFace...");
            cmd_pull_hf(store, &hf).await;
        }
        store.gguf_path(&id)
    } else if let Some(path) = resolve_model_path(model, store) {
        Some(path)
    } else {
        // Catalog model — try to pull if not available, then load GGUF
        if !store.exists(model) {
            if catalog::find_model(model).is_some() {
                println!("Model not found locally. Pulling...");
                cmd_pull(store, model).await;
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
        store.gguf_path(model)
    };

    if let Some(gguf_path) = gguf_path {
        model_name = canonical_name.clone();
        launch_gguf_path = Some(gguf_path.clone());

        println!("Loading GGUF: {}", gguf_path.display());

        // --fit: auto-size the GPU offload to free VRAM before loading. Opt-in;
        // headless-safe (never prompts unless both stdin and stdout are TTYs).
        if fit {
            let kv_bpe_k = inference::cache_type_bytes_per_elem(&cache_type_k);
            let kv_bpe_v = inference::cache_type_bytes_per_elem(&cache_type_v);
            match fit::run_fit(
                &gguf_path, gpu_layers, ctx_size, fit_strict, kv_bpe_k, kv_bpe_v,
            ) {
                fit::FitOutcome::Proceed(n) => gpu_layers = n,
                fit::FitOutcome::Abort => {
                    // Clean return: don't load, don't bind a port. If we were
                    // invoked from the picker flow, the user lands back there.
                    return;
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

        // Look up an mmproj projector for this model (if any was pulled
        // alongside the GGUF). On text-only builds the value is read but
        // ignored at InferenceConfig level; on multimodal builds it is
        // what enables `generate_multimodal`.
        let mmproj_for_config = store
            .mmproj_path(&model_name)
            .or_else(|| store.mmproj_path(model)); // also try the user-typed id
        if let Some(ref p) = mmproj_for_config {
            println!("Found mmproj: {}", p.display());
        }

        let config = InferenceConfig {
            model_path: gguf_path,
            gpu_layers,
            context_size: ctx_size,
            threads: resolved_threads,
            flash_attn,
            n_batch,
            cache_type_k,
            cache_type_v,
            mmproj_path: mmproj_for_config.clone(),
            cpu_moe,
            n_cpu_moe,
            rs_seq,
        };

        // The continuous-batching scheduler is text-only; multimodal models
        // must be served by the sequential `InferenceEngine` so that
        // `/api/chat` requests carrying `images` reach `generate_multimodal`.
        // Force batch_size=0 when an mmproj is present (vision is single-user
        // interactive — losing batching here is not a practical regression).
        let batch_size = if mmproj_for_config.is_some() {
            if batch_size > 0 {
                println!("Multimodal model — falling back to sequential mode (batch_size=0).");
            }
            0
        } else {
            batch_size
        };

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
            match sched.start() {
                Ok((handle, model_info)) => {
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
            match InferenceEngine::load(config) {
                Ok(eng) => {
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
    let mode = if scheduler.is_some() {
        format!("continuous batching (max {batch_size} concurrent)")
    } else {
        "sequential".to_string()
    };

    println!();
    println!("eullm ready.  [v{}]", env!("CARGO_PKG_VERSION"));
    println!("  API (EULLM):   http://localhost:{port}/api");
    println!("  API (OpenAI):  http://localhost:{port}/v1");
    if let Some(p) = ui_port {
        println!("  Chat UI:       http://localhost:{p}/");
    }
    println!("  Model:         {short}");
    if engine.is_some() || scheduler.is_some() {
        let gpu_backend = if cfg!(feature = "cuda") {
            "CUDA"
        } else if cfg!(feature = "rocm") {
            "ROCm"
        } else if cfg!(feature = "vulkan") {
            "Vulkan"
        } else if cfg!(feature = "metal") {
            "Metal"
        } else {
            "none (CPU only!)"
        };
        println!("  GPU backend:   {gpu_backend}");
        println!("  CPU features:  {}", inference::cpu_features_summary());
        if rust_debug {
            println!(
                "  Rust debug:    enabled (NaN/Inf logit check active — extra per-token cost)"
            );
        }
        println!(
            "  GPU layers:    {}",
            if gpu_layers < 0 {
                "all".to_string()
            } else {
                gpu_layers.to_string()
            }
        );
        if cpu_moe {
            println!("  CPU MoE:       enabled (expert tensors on CPU RAM)");
        } else if n_cpu_moe > 0 {
            println!("  CPU MoE:       first {n_cpu_moe} layers (expert tensors on CPU RAM)");
        }
        if rs_seq > 0 {
            println!(
                "  RS rollback:   {rs_seq} (recurrent-state window for hybrid/SSM architectures)"
            );
        }
        if ctx_checkpoints > 0 {
            println!(
                "  Checkpoints:   {ctx_checkpoints} max, every {checkpoint_min_step}+ new tokens (prompt-prefix restore)"
            );
        }
        if batch_size > 0 {
            let per_seq = ctx_size / batch_size as u32;
            println!(
                "  Context:       {ctx_size} total ({per_seq} per sequence × {batch_size} slots)"
            );
            // The continuous-batching scheduler splits ctx_size evenly across
            // slots, so a single conversation that builds up history can only
            // use ctx_size / batch_size tokens before hitting "does not fit".
            // Warn early when the per-sequence window is small enough to
            // surprise interactive REPL users.
            if batch_size > 1 && per_seq < 8192 {
                println!(
                    "  ⚠ per-sequence context is only {per_seq} tokens — long histories will fail."
                );
                let one_slot = ctx_size;
                let target_per_slot = 32768u32;
                let target_total = target_per_slot.saturating_mul(batch_size as u32);
                println!("    For single-chat use:   --batch-size 1   (full {one_slot} tokens)");
                println!(
                    "    For 32k per slot:      --ctx-size {target_total}   (= 32768 × {batch_size} slots)"
                );
            }
        } else {
            println!("  Context:       {ctx_size}");
        }
        println!(
            "  Flash attn:    {} (auto-detect)",
            if flash_attn { "enabled" } else { "disabled" }
        );
        let k_name = inference::cache_type_display(&cache_type_k);
        let v_name = inference::cache_type_display(&cache_type_v);
        println!("  KV cache:      K={k_name} V={v_name}");
        if kv_k_mib > 0.0 || kv_v_mib > 0.0 {
            println!(
                "  KV memory:     K={:.0} MiB, V={:.0} MiB",
                kv_k_mib, kv_v_mib
            );
        }
        if web {
            println!("  Web browsing:  enabled (URLs in messages are fetched and injected)");
        }
        println!("  Threads:       {resolved_threads}");
        println!("  Batch (prefill): {n_batch}");
        println!("  Mode:          {mode}");
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

    // Clone the scheduler handle for the interactive REPL before moving into api::serve.
    let repl_scheduler = scheduler.clone();
    let has_backend = engine.is_some() || scheduler.is_some();
    let is_tty = std::io::IsTerminal::is_terminal(&std::io::stdin());

    // Banner must match what we're actually about to do (see REPL launch
    // condition below: `has_backend && is_tty && !open_chat`). If the browser
    // chat is going to take over, telling the user to "Type a message" in
    // this terminal is a lie.
    if has_backend && is_tty && !open_chat {
        println!("Type a message to chat, /bye to quit.\n");
    } else {
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
            model_name: Some(api_model_name),
            engine,
            scheduler,
            gpu_layers,
            ctx_size,
            threads: resolved_threads,
            flash_attn,
            n_batch,
            cache_type_k,
            cache_type_v,
            batch_size,
            cpu_moe,
            n_cpu_moe,
            rs_seq,
            ctx_checkpoints,
            checkpoint_min_step,
            rust_debug,
            web_enabled: web,
            store: api_store,
            ui_port,
            launch_model: api_launch_model,
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
    if has_backend && is_tty && !open_chat {
        if let Some(sched) = repl_scheduler {
            interactive_chat(sched, &model_name, ctx_size, batch_size, web).await;
        }
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
    ctx_size: u32,
    threads: Option<u32>,
    flash_attn: bool,
    n_batch: u32,
    cache_type_k: inference::KvCacheType,
    cache_type_v: inference::KvCacheType,
    web: bool,
    cpu_moe: bool,
    n_cpu_moe: u32,
    rs_seq: u32,
    ctx_checkpoints: usize,
    checkpoint_min_step: u32,
    rust_debug: bool,
) {
    ensure_port_available(port, replace).await;
    if let Some(p) = ui_port {
        ensure_port_available(p, replace).await;
    }

    let threads = threads.unwrap_or_else(inference::default_thread_count);

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

    let store = ModelStore::default_store().expect("model store");

    if let Err(e) = api::serve(api::ServeConfig {
        port,
        model_name: None,
        engine: None,
        scheduler: None,
        gpu_layers,
        ctx_size,
        threads,
        flash_attn,
        n_batch,
        cache_type_k,
        cache_type_v,
        batch_size,
        cpu_moe,
        n_cpu_moe,
        rs_seq,
        ctx_checkpoints,
        checkpoint_min_step,
        rust_debug,
        web_enabled: web,
        store,
        ui_port,
        // Headless serve starts with an empty slot: there is no launch model to
        // grandfather in, so every name goes through the normal resolution.
        launch_model: None,
    })
    .await
    {
        eprintln!("Server error: {e}");
        std::process::exit(1);
    }
}

/// `eullm unload` — free the currently loaded model's VRAM on a running
/// `eullm serve`/`eullm run` server, without restarting the process.
///
/// Thin CLI wrapper around `POST /api/unload`. The server keeps running
/// with an empty model slot; a later request with a `model` field (or
/// another `eullm run <model>`) loads a model back in.
async fn cmd_unload(port: u16) {
    let url = format!("http://127.0.0.1:{port}/api/unload");
    let client = match reqwest::Client::builder().build() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error building HTTP client: {e}");
            std::process::exit(1);
        }
    };
    let response = match client.post(&url).send().await {
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
        Ok(body) => match body.get("unloaded").and_then(|v| v.as_str()) {
            Some(name) => println!("Unloaded '{name}'. VRAM freed."),
            None => println!("No model was loaded."),
        },
        Err(e) => eprintln!("Error reading response: {e}"),
    }
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

async fn interactive_chat(
    scheduler: inference::SchedulerHandle,
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
    use std::io::{BufRead, Write};

    let short = model_name.strip_prefix("eullm/").unwrap_or(model_name);

    let mut temperature: f32 = 0.8;
    let mut max_reply_tokens: u32 = 2048;
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

    let stdin = std::io::stdin();
    let mut reader = stdin.lock();

    loop {
        // Print prompt
        print!(">>> ");
        if std::io::stdout().flush().is_err() {
            break;
        }

        // Read user input (supports multi-line with trailing \)
        let mut input = String::new();
        loop {
            let mut line = String::new();
            match reader.read_line(&mut line) {
                Ok(0) => {
                    // EOF (Ctrl+D)
                    println!();
                    return;
                }
                Ok(_) => {
                    let trimmed = line.trim_end_matches('\n').trim_end_matches('\r');
                    if let Some(stripped) = trimmed.strip_suffix('\\') {
                        input.push_str(stripped);
                        input.push('\n');
                        print!("... ");
                        let _ = std::io::stdout().flush();
                        continue;
                    }
                    input.push_str(trimmed);
                    break;
                }
                Err(_) => return,
            }
        }

        let mut input = input.trim().to_string();
        if input.is_empty() {
            continue;
        }

        // Commands
        if input == "/bye" || input == "/exit" || input == "/quit" {
            println!("Bye!");
            return;
        } else if input == "/clear" {
            history.truncate(1);
            println!("Chat history cleared.\n");
            continue;
        } else if input == "/help" {
            println!("Commands:");
            println!("  /bye              Exit the chat");
            println!("  /clear            Clear conversation history");
            println!(
                "  /think            Enable reasoning (current: {})",
                if think_mode { "on" } else { "off" }
            );
            println!("  /no_think         Disable reasoning (sticky until /think)");
            println!("  /temp <0.0–2.0>   Set temperature (current: {temperature:.1})");
            println!("  /maxtokens <n>    Set max reply tokens (current: {max_reply_tokens})");
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
                Ok(n) if n > 0 => {
                    max_reply_tokens = n;
                    println!("Max reply tokens set to {max_reply_tokens}\n");
                }
                _ => eprintln!("Usage: /maxtokens <n>\n"),
            }
            continue;
        } else if let Some(sys) = input.strip_prefix("/system ") {
            if let Some(first) = history.first_mut() {
                first.content = sys.trim().to_string();
                println!("System prompt updated.\n");
            }
            continue;
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
        let prompt = if web_enabled {
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
                    template.build_prompt(&pairs, think_arg)
                } else {
                    let pairs: Vec<(&str, &str)> = history
                        .iter()
                        .map(|m| (m.role, m.content.as_str()))
                        .collect();
                    template.build_prompt(&pairs, think_arg)
                }
            } else {
                let pairs: Vec<(&str, &str)> = history
                    .iter()
                    .map(|m| (m.role, m.content.as_str()))
                    .collect();
                template.build_prompt(&pairs, think_arg)
            }
        } else {
            let pairs: Vec<(&str, &str)> = history
                .iter()
                .map(|m| (m.role, m.content.as_str()))
                .collect();
            template.build_prompt(&pairs, think_arg)
        };

        // Rough token estimate: ~4 chars per token. Leave room for the response.
        let estimated_prompt_tokens = prompt.len() as u32 / 4;
        let max_tokens = ctx_size.saturating_sub(estimated_prompt_tokens).min(2048);

        if max_tokens < 32 {
            eprintln!("Warning: conversation too long for context window. Use /clear to reset.\n");
            history.pop();
            continue;
        }

        let request = inference::GenerateRequest {
            prompt,
            max_tokens: max_tokens.min(max_reply_tokens),
            temperature,
            stop_sequences: template.stop_sequences(),
            ..Default::default()
        };

        // Submit to scheduler and stream tokens.
        let mut rx = scheduler.submit(request);
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

        // Add assistant response to history. When this turn suppressed
        // thinking (think_arg == false), the model actually decoded
        // `think_suppression_prefix()` right before this response — stored
        // history must include it too, or every later turn that includes
        // this one reconstructs text that no longer matches what's really
        // resident in this turn's KV cache, breaking prefix-based KV reuse
        // from here on (see `ChatTemplate::build_prompt`'s doc comment).
        if !response_text.is_empty() {
            let content = if think_arg {
                response_text
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

fn daemonize(pidfile: &str) {
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

    // Determine log file path next to PID file.
    let log_path = pidfile.replace(".pid", ".log");

    let log_file = match std::fs::File::create(&log_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error: cannot create log file {log_path}: {e}");
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
                        if let Ok(log) = std::fs::read_to_string(&log_path) {
                            let tail: Vec<&str> = log.lines().rev().take(10).collect();
                            for line in tail.into_iter().rev() {
                                eprintln!("  {line}");
                            }
                        }
                        eprintln!("  Full log: {log_path}");
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

            // Write PID file.
            if let Ok(mut f) = std::fs::File::create(pidfile) {
                let _ = write!(f, "{pid}");
            }
            println!("eullm daemon started (PID {pid}).");
            println!("  PID file: {pidfile}");
            println!("  Log file: {log_path}");
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
/// from stdin (or use a default), wrap it in the Gemma chat template with
/// the mtmd media marker, run a single multimodal generation, stream tokens
/// to stdout, exit.
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

    // 3. Wrap in the Gemma chat template with the media marker placed inside
    //    the user turn. Future work: detect the template from the model name
    //    instead of hardcoding Gemma — but our only multimodal catalog entry
    //    today is Gemma 4 12B, so this is correct for the MVP.
    let marker = mtmd_default_marker();
    let templated = format!(
        "<start_of_turn>user\n{marker}\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"
    );

    // 4. Build the request and stream the answer to stdout.
    let request = inference::GenerateRequest {
        prompt: templated,
        max_tokens: 512,
        temperature: 0.7,
        raw: true, // template is hand-built, no extra BOS / formatting
        stop_sequences: vec!["<end_of_turn>".to_string()],
        ..Default::default()
    };

    let (tx, mut rx) = mpsc::channel(64);
    let eng_for_task = engine.clone();
    let request_for_task = request.clone();
    let media_for_task = vec![media_bytes];
    let join = tokio::task::spawn_blocking(move || {
        eng_for_task.generate_multimodal(&request_for_task, &media_for_task, tx);
    });

    use std::io::Write;
    let mut stdout = std::io::stdout().lock();
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
            } => {
                let _ = writeln!(stdout);
                let _ = writeln!(
                    stdout,
                    "[done — {tokens_generated} tokens, prompt {tokens_prompt}, {duration_ms} ms]"
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

    /// The KV cache defaults of `run` and `serve`.
    fn kv_defaults(argv: &[&str]) -> (String, String) {
        match Cli::parse_from(argv).command.expect("subcommand") {
            Commands::Run {
                cache_type_k,
                cache_type_v,
                ..
            }
            | Commands::Serve {
                cache_type_k,
                cache_type_v,
                ..
            } => (cache_type_k, cache_type_v),
            other => panic!(
                "unexpected subcommand: {:?}",
                std::mem::discriminant(&other)
            ),
        }
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
        let run = match Cli::parse_from(["eullm", "run", "some-model"])
            .command
            .expect("subcommand")
        {
            Commands::Run { ctx_size, .. } => ctx_size,
            _ => unreachable!(),
        };
        let serve = match Cli::parse_from(["eullm", "serve"])
            .command
            .expect("subcommand")
        {
            Commands::Serve { ctx_size, .. } => ctx_size,
            _ => unreachable!(),
        };
        assert_eq!(
            run, serve,
            "run and serve must agree on the default context"
        );
    }
}

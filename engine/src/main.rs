mod api;
mod audit;
mod chat_template;
mod gguf_patch;
mod inference;
mod models;
mod registry;

use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use models::{catalog, ModelStore};

use crate::inference::{BatchScheduler, InferenceConfig, InferenceEngine, SchedulerConfig};

#[derive(Parser)]
#[command(name = "eullm")]
#[command(about = "eullm — sovereign LLM runtime for Europe")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Pull a model from the EU registry
    Pull {
        /// Model name (e.g., general-eu-14b)
        model: String,
    },
    /// Run a model locally (starts API server)
    Run {
        /// Model name or path to a local GGUF file
        model: String,

        /// Port for the API server
        #[arg(short, long, default_value_t = 11434)]
        port: u16,

        /// Replace existing service on the port
        #[arg(long)]
        replace: bool,

        /// Number of GPU layers to offload (-1 = all, 0 = CPU only)
        #[arg(long, default_value_t = -1)]
        gpu_layers: i32,

        /// Context window size
        #[arg(short, long, default_value_t = 4096)]
        ctx_size: u32,

        /// Number of CPU threads (default: all available)
        #[arg(short, long)]
        threads: Option<u32>,

        /// Enable continuous batching with N max concurrent requests (0 = sequential)
        #[arg(long, default_value_t = 8)]
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

        /// Run as a background daemon (writes PID to --pidfile)
        #[arg(long)]
        daemon: bool,

        /// PID file path (used with --daemon)
        #[arg(long, default_value = "/tmp/eullm.pid")]
        pidfile: String,
    },
    /// List locally available models
    List,
    /// Show model information
    Show {
        /// Model name
        model: String,
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

        /// Run as a background daemon (writes PID to --pidfile)
        #[arg(long)]
        daemon: bool,

        /// PID file path (used with --daemon)
        #[arg(long, default_value = "/tmp/eullm.pid")]
        pidfile: String,
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
            let pidfile = args.iter()
                .position(|a| a == "--pidfile")
                .and_then(|i| args.get(i + 1))
                .map(|s| s.as_str())
                .or_else(|| {
                    args.iter()
                        .find(|a| a.starts_with("--pidfile="))
                        .map(|a| a.strip_prefix("--pidfile=").unwrap())
                })
                .unwrap_or("/tmp/eullm.pid");
            daemonize(pidfile);
            // daemonize exits the parent — child continues below without --daemon.
        }
    }

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "eullm_engine=info".into()),
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

    match cli.command {
        Commands::Pull { model } => cmd_pull(&store, &model),
        Commands::Run {
            model,
            port,
            replace,
            gpu_layers,
            ctx_size,
            threads,
            batch_size,
            no_flash_attn,
            n_batch,
            cache_type_k,
            cache_type_v,
            daemon,
            pidfile,
        } => {
            // --daemon is handled at the top of main() before tokio starts.
            let _ = (daemon, pidfile);
            let ctk = inference::parse_cache_type(&cache_type_k).unwrap_or_else(|e| {
                eprintln!("Error: {e}");
                std::process::exit(1);
            });
            let ctv = inference::parse_cache_type(&cache_type_v).unwrap_or_else(|e| {
                eprintln!("Error: {e}");
                std::process::exit(1);
            });
            // Mixed TurboQuant types (e.g. K=tq4_0, V=tq3_0) are valid:
            // K needs more precision (attention scores), V tolerates more compression.
            // Warn the user, but allow it — fallback handles failures gracefully.
            if let (inference::KvCacheType::Unknown(k_id), inference::KvCacheType::Unknown(v_id)) = (&ctk, &ctv)
                && k_id != v_id
            {
                eprintln!("Note: mixed TurboQuant KV cache (K={cache_type_k}, V={cache_type_v}).");
                eprintln!("  Advanced config — if OOM, will fallback to uniform type then F16.");
            }
            cmd_run(&store, &model, port, replace, gpu_layers, ctx_size, threads, batch_size, !no_flash_attn, n_batch, ctk, ctv).await;
        }
        Commands::List => cmd_list(&store),
        Commands::Show { model } => cmd_show(&store, &model),
        Commands::Serve { port, replace, batch_size: _, daemon, pidfile } => {
            let _ = (daemon, pidfile);
            cmd_serve(port, replace).await;
        }
        Commands::ImportOllama { model, ollama_dir } => cmd_import_ollama(&store, &model, ollama_dir.as_deref()),
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

fn cmd_pull(store: &ModelStore, model: &str) {
    let entry = match catalog::find_model(model) {
        Some(e) => e,
        None => {
            eprintln!("Error: model '{model}' not found in EU catalog.");
            eprintln!("Run `eullm list` to see available models.");
            std::process::exit(1);
        }
    };

    // Check if already downloaded with GGUF
    if let Some(gguf) = store.gguf_path(&entry.name) {
        println!("Model '{}' is already downloaded.", entry.name);
        println!("  GGUF: {}", gguf.display());
        println!("\nRun with: eullm run {model}");
        return;
    }

    println!("Pulling {} ...", entry.name);
    println!(
        "  {} | {} | ~{}GB VRAM | {}",
        entry.description, entry.base, entry.vram_gb, entry.license
    );

    if entry.hf_repo.is_empty() {
        println!("  Warning: no download source configured for this model.");
        println!("  Writing manifest only (no GGUF file).");

        match store.write_manifest(entry, "metadata_only", None) {
            Ok(path) => println!("  Manifest saved to {}", path.display()),
            Err(e) => {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    // Download GGUF from HuggingFace
    let short_name = entry.name.strip_prefix("eullm/").unwrap_or(&entry.name);
    let model_dir = store.model_path(&entry.name);
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

    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(async {
        use crate::registry::{download_from_huggingface, format_progress};
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::sync::Arc;

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

        download_from_huggingface(&hf_repo, &hf_filename, &gguf_dest, Some(progress)).await
    });

    eprintln!(); // newline after progress

    match result {
        Ok(()) => {
            // Write manifest with GGUF file reference
            match store.write_manifest(&entry_clone, "ready", Some(&entry_clone.hf_filename)) {
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

            // Still write manifest so we don't re-attempt
            let _ = store.write_manifest(&entry_clone, "download_failed", None);
            std::process::exit(1);
        }
    }
}

fn cmd_list(store: &ModelStore) {
    match store.list() {
        Ok(models) if models.is_empty() => {
            println!("No models installed.");
            println!("\nAvailable models in EU catalog:");
            for entry in catalog::EU_CATALOG.iter() {
                let short = entry.name.strip_prefix("eullm/").unwrap_or(&entry.name);
                println!(
                    "  {:<25} {:>3}GB  {}",
                    short, entry.vram_gb, entry.description
                );
            }
            println!("\nPull with: eullm pull <model-name>");
            println!("Or run a local GGUF: eullm run ./path/to/model.gguf");
        }
        Ok(models) => {
            println!("{:<30} {:>8} {:>6} STATUS", "NAME", "SIZE", "VRAM");
            for m in &models {
                let size = format_bytes(m.size_bytes);
                let short = m.name.strip_prefix("eullm/").unwrap_or(&m.name);
                println!(
                    "{:<30} {:>8} {:>4}GB  {}",
                    short, size, m.vram_gb, m.status
                );
            }
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
            println!("Model:       {}", manifest.name);
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
                println!("Base:        {}", entry.base);
                println!("Languages:   {}", entry.languages.join(", "));
                println!("VRAM:        {}GB", entry.vram_gb);
                println!("Size:        {}", format_bytes(entry.size_bytes));
                println!("License:     {}", entry.license);
                println!("\nPull with: eullm pull {model}");
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
            eprintln!("  2. Use a different port:  eullm serve --port {}", port + 1);
            std::process::exit(1);
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
    ctx_size: u32,
    threads: Option<u32>,
    batch_size: usize,
    flash_attn: bool,
    n_batch: u32,
    cache_type_k: inference::KvCacheType,
    cache_type_v: inference::KvCacheType,
) {
    ensure_port_available(port, replace).await;

    let model_name: String;
    let mut engine: Option<Arc<InferenceEngine>> = None;
    let mut scheduler: Option<inference::SchedulerHandle> = None;
    let mut kv_k_mib: f64 = 0.0;
    let mut kv_v_mib: f64 = 0.0;

    let resolved_threads = threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(4)
    });

    // Try to resolve as a local GGUF file or downloaded model
    let gguf_path = if let Some(path) = resolve_model_path(model, store) {
        Some(path)
    } else {
        // Catalog model — try to pull if not available, then load GGUF
        if !store.exists(model) {
            if catalog::find_model(model).is_some() {
                println!("Model not found locally. Pulling...");
                cmd_pull(store, model);
            } else {
                eprintln!("Error: model '{model}' not found.");
                eprintln!();
                eprintln!("Usage:");
                eprintln!("  eullm run ./path/to/model.gguf    # Run a local GGUF file");
                eprintln!("  eullm run legal-it-7b              # Run a catalog model");
                std::process::exit(1);
            }
        }
        store.gguf_path(model)
    };

    if let Some(gguf_path) = gguf_path {
        model_name = gguf_path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| model.to_string());

        println!("Loading GGUF: {}", gguf_path.display());

        let config = InferenceConfig {
            model_path: gguf_path,
            gpu_layers,
            context_size: ctx_size,
            threads: resolved_threads,
            flash_attn,
            n_batch,
            cache_type_k,
            cache_type_v,
        };

        if batch_size > 0 {
            // ── Continuous batching mode ────────────────────────────
            let sched_config = SchedulerConfig {
                max_batch_size: batch_size,
                queue_capacity: batch_size * 8,
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
        model_name = model.to_string();
        eprintln!("Warning: no GGUF file available for this model.");
        eprintln!("  The model may not have been published yet.");
        eprintln!("  API will start but inference requests will return 503.");
        eprintln!("  To test inference, use a local GGUF file:");
        eprintln!("    eullm run ./path/to/model.gguf");
        eprintln!();
    }

    let short = model_name
        .strip_prefix("eullm/")
        .unwrap_or(&model_name);
    let mode = if scheduler.is_some() {
        format!("continuous batching (max {batch_size} concurrent)")
    } else {
        "sequential".to_string()
    };

    println!();
    println!("eullm ready.  [v{}]", env!("CARGO_PKG_VERSION"));
    println!("  API (EULLM):   http://localhost:{port}/api");
    println!("  API (OpenAI):  http://localhost:{port}/v1");
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
        println!("  GPU layers:    {}", if gpu_layers < 0 { "all".to_string() } else { gpu_layers.to_string() });
        if batch_size > 0 {
            let per_seq = ctx_size / batch_size as u32;
            println!("  Context:       {ctx_size} total ({per_seq} per sequence × {batch_size} slots)");
        } else {
            println!("  Context:       {ctx_size}");
        }
        println!("  Flash attn:    {} (auto-detect)", if flash_attn { "enabled" } else { "disabled" });
        let k_name = inference::cache_type_display(&cache_type_k);
        let v_name = inference::cache_type_display(&cache_type_v);
        println!("  KV cache:      K={k_name} V={v_name}");
        // Show TurboQuant status if any cache type is TQ
        let is_tq = matches!(cache_type_k, inference::KvCacheType::Unknown(41..=43))
            || matches!(cache_type_v, inference::KvCacheType::Unknown(41..=43));
        if kv_k_mib > 0.0 || kv_v_mib > 0.0 {
            println!("  KV memory:     K={:.0} MiB, V={:.0} MiB", kv_k_mib, kv_v_mib);
        }
        if is_tq {
            println!("  TurboQuant:    active (experimental)");
        }
        println!("  Threads:       {resolved_threads}");
        println!("  Batch (prefill): {n_batch}");
        println!("  Mode:          {mode}");
    }
    println!();

    // Clone the scheduler handle for the interactive REPL before moving into api::serve.
    let repl_scheduler = scheduler.clone();
    let has_backend = engine.is_some() || scheduler.is_some();
    let is_tty = std::io::IsTerminal::is_terminal(&std::io::stdin());

    if has_backend && is_tty {
        println!("Type a message to chat, /bye to quit.\n");
    } else {
        println!("Press Ctrl+C to stop.\n");
    }

    // Start the API server in the background.
    let api_model_name = model_name.clone();
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
            store: api_store,
        })
        .await
        {
            eprintln!("Server error: {e}");
            std::process::exit(1);
        }
    });

    // Give the API server a moment to bind.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    if has_backend && is_tty {
        if let Some(sched) = repl_scheduler {
            interactive_chat(sched, &model_name, ctx_size).await;
        }
    } else {
        // No REPL — wait for shutdown signal.
        // The API server handles graceful shutdown internally via SIGTERM/SIGINT.
        tokio::signal::ctrl_c().await.ok();
        tracing::info!("Shutting down...");
    }
}

async fn cmd_serve(port: u16, replace: bool) {
    ensure_port_available(port, replace).await;

    let threads = std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(4);

    println!("eullm ready (no model loaded — send a request with a \"model\" field to load one).");
    println!("  API (EULLM):   http://localhost:{port}/api");
    println!("  API (OpenAI):  http://localhost:{port}/v1");
    println!("\nPress Ctrl+C to stop.\n");

    let store = ModelStore::default_store().expect("model store");

    if let Err(e) = api::serve(api::ServeConfig {
        port,
        model_name: None,
        engine: None,
        scheduler: None,
        gpu_layers: -1,
        ctx_size: 4096,
        threads,
        flash_attn: true,
        n_batch: 2048,
        cache_type_k: inference::KvCacheType::Q8_0,
        cache_type_v: inference::KvCacheType::Q4_0,
        batch_size: 8,
        store,
    })
    .await
    {
        eprintln!("Server error: {e}");
        std::process::exit(1);
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
        let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
        PathBuf::from(home).join(".ollama")
    };

    if !ollama_root.exists() {
        eprintln!("Error: Ollama directory not found: {}", ollama_root.display());
        eprintln!("  Is Ollama installed? Try: ollama --version");
        eprintln!("  Or specify a custom path: eullm import-ollama {model} --ollama-dir /path/to/ollama");
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
                                println!("  {}:{}", name.to_string_lossy(), tag_name.to_string_lossy());
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
            println!("  Patched GGUF metadata during copy (fixed array lengths for llama.cpp compatibility).");
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
    let mut writer = std::io::BufWriter::with_capacity(8 * 1024 * 1024, std::fs::File::create(dst)?);

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
) {
    use std::io::{BufRead, Write};

    let short = model_name
        .strip_prefix("eullm/")
        .unwrap_or(model_name);

    let mut temperature: f32 = 0.8;
    let mut max_reply_tokens: u32 = 2048;

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

        let input = input.trim().to_string();
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
            println!("  /temp <0.0–2.0>   Set temperature (current: {temperature:.1})");
            println!("  /maxtokens <n>    Set max reply tokens (current: {max_reply_tokens})");
            println!("  /system <text>    Replace system prompt");
            println!("  /help             Show this help\n");
            continue;
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

        // Add user message to history.
        history.push(ChatMessage {
            role: "user",
            content: input,
        });

        // Build prompt using the model-appropriate chat template.
        let template = crate::chat_template::ChatTemplate::detect(model_name);
        let pairs: Vec<(&str, &str)> = history.iter()
            .map(|m| (m.role, m.content.as_str()))
            .collect();
        let prompt = template.build_prompt(&pairs, true);

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
                } => {
                    // Strip any trailing stop sequence that was printed as part of the stream.
                    for stop in template.stop_sequences() {
                        if response_text.ends_with(&stop) {
                            // Erase the stop token from the terminal using backspaces.
                            let erase = "\x08 \x08".repeat(stop.chars().count());
                            print!("{erase}");
                            let _ = std::io::stdout().flush();
                            response_text.truncate(response_text.len() - stop.len());
                            break;
                        }
                    }
                    let tps = if duration_ms > 0 {
                        tokens_generated as f64 / (duration_ms as f64 / 1000.0)
                    } else {
                        0.0
                    };
                    stats_line = format!(
                        "\n\n[{short}: {tokens_generated} tokens, {tokens_prompt} prompt, {:.1} tok/s]\n",
                        tps
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
        if !response_text.is_empty() {
            history.push(ChatMessage {
                role: "assistant",
                content: response_text,
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
        Ok(child) => {
            let pid = child.id();
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
        libc::signal(libc::SIGABRT, abort_handler as *const () as libc::sighandler_t);
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

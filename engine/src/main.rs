mod api;
mod audit;
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
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "eullm_engine=info".into()),
        )
        .init();

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
        } => cmd_run(&store, &model, port, replace, gpu_layers, ctx_size, threads, batch_size).await,
        Commands::List => cmd_list(&store),
        Commands::Show { model } => cmd_show(&store, &model),
        Commands::Serve { port, replace, batch_size: _ } => cmd_serve(port, replace).await,
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
    if let Ok(resp) = reqwest::get(&url).await {
        if let Ok(body) = resp.text().await {
            if body.contains("version") {
                if body.contains("eullm") {
                    return Some("eullm (already running)".into());
                }
                return Some(format!("another service (response: {body})"));
            }
        }
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
) {
    ensure_port_available(port, replace).await;

    let model_name: String;
    let mut engine: Option<Arc<InferenceEngine>> = None;
    let mut scheduler: Option<inference::SchedulerHandle> = None;

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
        };

        if batch_size > 0 {
            // ── Continuous batching mode ────────────────────────────
            let sched_config = SchedulerConfig {
                max_batch_size: batch_size,
                queue_capacity: batch_size * 8,
            };
            let sched = BatchScheduler::new(config, sched_config);
            match sched.start() {
                Ok(handle) => {
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
    println!("eullm ready.");
    println!("  API (EULLM):   http://localhost:{port}/api");
    println!("  API (OpenAI):  http://localhost:{port}/v1");
    println!("  Model:         {short}");
    if engine.is_some() || scheduler.is_some() {
        println!("  GPU layers:    {}", if gpu_layers < 0 { "all".to_string() } else { gpu_layers.to_string() });
        println!("  Context:       {ctx_size}");
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
    tokio::spawn(async move {
        if let Err(e) = api::serve(port, Some(api_model_name), engine, scheduler).await {
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
        // No REPL — just wait forever (API server runs in background task).
        std::future::pending::<()>().await;
    }
}

async fn cmd_serve(port: u16, replace: bool) {
    ensure_port_available(port, replace).await;

    println!("eullm ready (no model loaded).");
    println!("  API (EULLM):   http://localhost:{port}/api");
    println!("  API (OpenAI):  http://localhost:{port}/v1");
    println!("\nPress Ctrl+C to stop.\n");

    if let Err(e) = api::serve(port, None, None, None).await {
        eprintln!("Server error: {e}");
        std::process::exit(1);
    }
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

/// Build a ChatML prompt from conversation history.
///
/// Format:
/// ```text
/// <|im_start|>system
/// You are a helpful assistant.<|im_end|>
/// <|im_start|>user
/// Hello<|im_end|>
/// <|im_start|>assistant
/// ```
fn build_chatml_prompt(history: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in history {
        prompt.push_str("<|im_start|>");
        prompt.push_str(msg.role);
        prompt.push('\n');
        prompt.push_str(&msg.content);
        prompt.push_str("<|im_end|>\n");
    }
    // Open the assistant turn for the model to complete.
    prompt.push_str("<|im_start|>assistant\n");
    prompt
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
        match input.as_str() {
            "/bye" | "/exit" | "/quit" => {
                println!("Bye!");
                return;
            }
            "/clear" => {
                history.truncate(1); // keep system message
                println!("Chat history cleared.\n");
                continue;
            }
            "/help" => {
                println!("Commands:");
                println!("  /bye     Exit the chat");
                println!("  /clear   Clear conversation history");
                println!("  /help    Show this help\n");
                continue;
            }
            _ => {}
        }

        // Add user message to history.
        history.push(ChatMessage {
            role: "user",
            content: input,
        });

        // Build prompt and estimate token budget.
        let prompt = build_chatml_prompt(&history);

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
            max_tokens,
            temperature: 0.7,
            stop_sequences: vec!["<|im_end|>".into()],
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

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1}GB", bytes as f64 / 1_000_000_000.0)
    } else if bytes >= 1_000_000 {
        format!("{:.1}MB", bytes as f64 / 1_000_000.0)
    } else {
        format!("{bytes}B")
    }
}

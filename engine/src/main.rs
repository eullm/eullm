mod api;
mod audit;
mod inference;
mod models;
mod registry;

use clap::{Parser, Subcommand};
use models::{catalog, ModelStore};

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
        /// Model name (e.g., general-eu-14b)
        model: String,

        /// Port for the API server
        #[arg(short, long, default_value_t = 11434)]
        port: u16,

        /// Replace existing service on the port (e.g., stop Ollama)
        #[arg(long)]
        replace: bool,
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

        /// Replace existing service on the port (e.g., stop Ollama)
        #[arg(long)]
        replace: bool,
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
        } => cmd_run(&store, &model, port, replace).await,
        Commands::List => cmd_list(&store),
        Commands::Show { model } => cmd_show(&store, &model),
        Commands::Serve { port, replace } => cmd_serve(port, replace).await,
    }
}

fn cmd_pull(store: &ModelStore, model: &str) {
    let entry = match catalog::find_model(model) {
        Some(e) => e,
        None => {
            eprintln!("Error: model '{model}' not found in EU catalog.");
            eprintln!("Run `eullm list --remote` to see available models.");
            std::process::exit(1);
        }
    };

    if store.exists(&entry.name) {
        println!("Model '{}' is already pulled.", entry.name);
        return;
    }

    println!("Pulling {} from EU registry...", entry.name);
    println!(
        "  {} | {} | ~{}GB VRAM | {}",
        entry.description,
        entry.base,
        entry.vram_gb,
        entry.license
    );

    // Simulate download progress
    println!("  Downloading manifest...");

    match store.pull(entry) {
        Ok(path) => {
            println!("  Done. Model saved to {}", path.display());
            println!("\nRun with: eullm run {model}");
        }
        Err(e) => {
            eprintln!("Error pulling model: {e}");
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
        }
        Ok(models) => {
            println!("{:<30} {:>8} {:>6} {}", "NAME", "SIZE", "VRAM", "STATUS");
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

/// Check what service is running on a given port.
///
/// Returns a description of the detected service, or `None` if the port is free.
async fn detect_port_service(port: u16) -> Option<String> {
    use tokio::net::TcpStream;

    // Try to connect to the port
    let addr = format!("127.0.0.1:{port}");
    if TcpStream::connect(&addr).await.is_err() {
        return None; // Port is free
    }

    // Port is in use — try to identify the service
    // Check if it's Ollama by hitting /api/version
    let url = format!("http://127.0.0.1:{port}/api/version");
    if let Ok(resp) = reqwest::get(&url).await {
        if let Ok(body) = resp.text().await {
            if body.contains("version") {
                // Check if it's us or Ollama
                if body.contains("eullm") {
                    return Some("eullm (already running)".into());
                }
                return Some(format!("Ollama (response: {body})"));
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
            // Try to stop Ollama if that's what's running
            if service.contains("Ollama") {
                eprintln!("Hint: run 'systemctl stop ollama' or 'ollama stop' first.");
            }
            eprintln!("Error: --replace is not yet implemented. Stop the service manually.");
            std::process::exit(1);
        } else {
            eprintln!("Error: port {port} is already in use by {service}.");
            eprintln!();
            if service.contains("Ollama") {
                eprintln!("Ollama is running on the default port. Options:");
                eprintln!("  1. Stop Ollama:   systemctl stop ollama");
                eprintln!("  2. Use a different port:  eullm serve --port 11435");
            } else {
                eprintln!("Options:");
                eprintln!("  1. Stop the existing service on port {port}");
                eprintln!("  2. Use a different port:  eullm serve --port {}", port + 1);
            }
            std::process::exit(1);
        }
    }
}

async fn cmd_run(store: &ModelStore, model: &str, port: u16, replace: bool) {
    // Check port availability before doing anything
    ensure_port_available(port, replace).await;

    // Auto-pull if not available
    if !store.exists(model) {
        if let Some(entry) = catalog::find_model(model) {
            println!("Model not found locally. Pulling...");
            if let Err(e) = store.pull(entry) {
                eprintln!("Error pulling model: {e}");
                std::process::exit(1);
            }
        } else {
            eprintln!("Error: model '{model}' not found in EU catalog.");
            std::process::exit(1);
        }
    }

    let short = model.strip_prefix("eullm/").unwrap_or(model);
    println!("Loading model {short}...");
    println!("eullm ready.");
    println!("  API (EULLM):   http://localhost:{port}/api");
    println!("  API (OpenAI):  http://localhost:{port}/v1");
    println!("  Model:         {short}");
    println!("\nPress Ctrl+C to stop.\n");

    if let Err(e) = api::serve(port, Some(model.to_string())).await {
        eprintln!("Server error: {e}");
        std::process::exit(1);
    }
}

async fn cmd_serve(port: u16, replace: bool) {
    // Check port availability
    ensure_port_available(port, replace).await;

    println!("eullm ready (no model loaded).");
    println!("  API (EULLM):   http://localhost:{port}/api");
    println!("  API (OpenAI):  http://localhost:{port}/v1");
    println!("\nPress Ctrl+C to stop.\n");

    if let Err(e) = api::serve(port, None).await {
        eprintln!("Server error: {e}");
        std::process::exit(1);
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

mod api;
mod audit;
mod inference;
mod registry;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "eullm")]
#[command(about = "EULLM Engine — sovereign LLM runtime for Europe")]
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
    /// Run a model locally
    Run {
        /// Model name (e.g., general-eu-14b)
        model: String,

        /// Port for the API server
        #[arg(short, long, default_value_t = 11435)]
        port: u16,
    },
    /// List locally available models
    List,
    /// Show model information
    Show {
        /// Model name
        model: String,
    },
    /// Start the API server
    Serve {
        /// Port for the API server
        #[arg(short, long, default_value_t = 11435)]
        port: u16,
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

    match cli.command {
        Commands::Pull { model } => {
            tracing::info!("Pulling model: {model}");
            println!("Pulling {model} from EU registry...");
            // TODO: implement registry pull
        }
        Commands::Run { model, port } => {
            tracing::info!("Running model: {model} on port {port}");
            println!("Starting {model} on port {port}...");
            // TODO: load model and start API server
        }
        Commands::List => {
            println!("Local models:");
            // TODO: list local models
        }
        Commands::Show { model } => {
            println!("Model: {model}");
            // TODO: show model info
        }
        Commands::Serve { port } => {
            tracing::info!("Starting API server on port {port}");
            if let Err(e) = api::serve(port).await {
                tracing::error!("Server error: {e}");
                std::process::exit(1);
            }
        }
    }
}

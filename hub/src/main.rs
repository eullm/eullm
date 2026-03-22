//! EULLM Hub — EU-hosted model registry API.
//!
//! Serves model metadata, model cards, AI Act compliance cards,
//! and GGUF model files from local storage or S3-compatible backends
//! on European infrastructure (Hetzner DE, OVH FR).

use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{header, StatusCode};
use axum::response::IntoResponse;
use axum::{routing::get, Json, Router};
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tokio_util::io::ReaderStream;

/// Hub configuration and shared state.
#[derive(Clone)]
struct HubState {
    /// Root directory for GGUF model files.
    /// Layout: {storage_root}/{model-short-name}/{model-short-name}.gguf
    storage_root: PathBuf,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "eullm_hub=info".into()),
        )
        .init();

    // Storage root from env or default
    let storage_root = std::env::var("EULLM_HUB_STORAGE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
            PathBuf::from(home).join(".eullm").join("hub").join("models")
        });

    std::fs::create_dir_all(&storage_root)?;
    tracing::info!("Model storage: {}", storage_root.display());

    let state = Arc::new(HubState { storage_root });

    let app = Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/models/{name}", get(get_model))
        .route("/v1/models/{name}/card", get(model_card))
        .route("/v1/models/{name}/compliance", get(compliance_card))
        .route("/v1/models/{name}/download", get(download_model))
        .route("/health", get(health))
        .with_state(state);

    let port = std::env::var("EULLM_HUB_PORT")
        .ok()
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(8080);
    let addr = format!("0.0.0.0:{port}");
    tracing::info!("EULLM Hub listening on {addr}");

    let listener = TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

// -- Model catalog --

/// Static catalog of EULLM models.
/// In production this would be backed by a database.
fn catalog() -> Vec<Value> {
    vec![
        model_entry("legal-it-7b", "Italian legal domain — civil code, GDPR, Cassazione rulings", &["it", "en"], "legal", "qwen3", 6, "Qwen/Qwen3-14B", 4_500_000_000),
        model_entry("medical-de-7b", "German medical — clinical guidelines, medical documentation", &["de", "en"], "medical", "qwen3", 6, "Qwen/Qwen3-14B", 4_500_000_000),
        model_entry("finance-fr-7b", "French finance — AMF regulations, BCE directives, banking", &["fr", "en"], "finance", "qwen3", 6, "Qwen/Qwen3-14B", 4_500_000_000),
        model_entry("general-eu-7b", "General purpose multilingual", &["en", "it", "de", "fr", "es", "pt", "nl"], "general", "qwen3", 6, "Qwen/Qwen3-14B", 4_500_000_000),
        model_entry("general-eu-14b", "General purpose multilingual (larger)", &["en", "it", "de", "fr", "es", "pt", "nl"], "general", "qwen3", 10, "Qwen/Qwen3-30B-A3B", 8_500_000_000),
        model_entry("legal-it-14b", "Italian legal domain (larger)", &["it", "en"], "legal", "qwen3", 10, "Qwen/Qwen3-30B-A3B", 8_200_000_000),
        model_entry("code-eu-14b", "Multilingual coding model", &["en", "it", "de", "fr", "es"], "code", "deepseek", 10, "deepseek-ai/DeepSeek-V3", 8_500_000_000),
    ]
}

fn find_in_catalog(name: &str) -> Option<Value> {
    let full_name = if name.starts_with("eullm/") {
        name.to_string()
    } else {
        format!("eullm/{name}")
    };
    catalog().into_iter().find(|m| m["name"] == full_name)
}

// -- Handlers --

async fn list_models() -> Json<Value> {
    Json(json!({ "models": catalog() }))
}

async fn get_model(Path(name): Path<String>) -> Result<Json<Value>, StatusCode> {
    find_in_catalog(&name)
        .map(Json)
        .ok_or(StatusCode::NOT_FOUND)
}

async fn model_card(Path(name): Path<String>) -> Json<Value> {
    Json(json!({
        "model": format!("eullm/{name}"),
        "card_version": "1.0",
        "summary": {
            "description": format!("EULLM verticalizzato model: {name}"),
            "intended_use": "Domain-specific AI assistance for European businesses",
            "out_of_scope": "Medical diagnosis, legal advice (informational use only)",
            "architecture": "Transformer (decoder-only)",
            "base_model": "Qwen3-14B (Apache 2.0)",
            "compression_pipeline": "Structural pruning → Knowledge distillation → Quantization (Q4_K_M) → Identity LoRA",
            "format": "GGUF",
        },
        "training": {
            "methodology": "NVIDIA Minitron-style pruning + distillation + identity LoRA",
            "data_sources": "Publicly available domain-specific corpora (see compliance card)",
            "data_governance": "All training data sourced from public domain or openly licensed sources",
            "compute": "EU cloud infrastructure (Hetzner DE)",
            "carbon_footprint": "Estimated via ML CO2 Impact calculator",
        },
        "evaluation": {
            "benchmarks": "Domain-specific benchmarks + general EU language benchmarks",
            "known_limitations": [
                "May hallucinate legal/medical/financial information",
                "Not a substitute for professional advice",
                "Performance degrades on languages not in training set"
            ],
        },
        "license": "Apache-2.0",
        "contact": "dev@eullm.eu"
    }))
}

async fn compliance_card(Path(name): Path<String>) -> Json<Value> {
    Json(json!({
        "model": format!("eullm/{name}"),
        "regulation": "EU AI Act — Regulation (EU) 2024/1689",
        "card_version": "1.0",
        "risk_classification": {
            "category": "General Purpose AI (GPAI)",
            "systemic_risk": false,
            "high_risk_use": "Depends on deployment context — deployer responsibility",
        },
        "transparency": {
            "model_card_available": true,
            "training_data_documented": true,
            "intended_purpose_stated": true,
            "limitations_disclosed": true,
            "ai_generated_content_disclosure": "Model outputs should be clearly marked as AI-generated by the deployer",
        },
        "data_governance": {
            "gdpr_compliant": true,
            "training_data_origin": "EU/public domain sources",
            "personal_data": "No personal data in training set",
            "data_retention": "Training data not stored in model weights",
            "right_to_erasure": "Not applicable — no personal data",
        },
        "technical_documentation": {
            "architecture": "Transformer decoder-only, pruned + distilled from Qwen3-14B",
            "compression_method": "NVIDIA Minitron approach: structural pruning + knowledge distillation",
            "quantization": "Q4_K_M (4-bit, K-quants mixed)",
            "inference_requirements": "CPU with 8GB RAM or GPU with 6GB VRAM",
            "audit_trail": "Built into EULLM Engine — logs every inference request",
        },
        "human_oversight": {
            "mechanism": "EULLM Engine audit trail provides full inference logging",
            "deployer_responsibility": "Deployer must implement appropriate oversight per their risk classification",
        },
        "infrastructure": {
            "training_location": "EU (Hetzner, Nuremberg DE)",
            "registry_location": "EU (Hetzner DE, OVH FR)",
            "data_residency": "All data stays within EU borders",
            "telemetry": "Zero telemetry to non-EU servers",
        },
        "contact": {
            "provider": "EULLM / I3K Technologies",
            "email": "compliance@eullm.eu",
            "address": "Milan, Italy"
        }
    }))
}

/// Serve a GGUF model file for download.
///
/// Looks for the file at: `{storage_root}/{name}/{name}.gguf`
/// Returns 404 if the model hasn't been uploaded to this Hub instance.
async fn download_model(
    State(state): State<Arc<HubState>>,
    Path(name): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<Value>)> {
    let short_name = name.strip_prefix("eullm/").unwrap_or(&name);

    // Look for GGUF file in storage
    let model_dir = state.storage_root.join(short_name);
    let gguf_path = find_gguf_in_dir(&model_dir);

    let gguf_path = gguf_path.ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(json!({
                "error": format!("Model '{name}' not available for download on this Hub instance"),
                "hint": "Upload the GGUF file to the Hub storage directory, or use HuggingFace directly"
            })),
        )
    })?;

    let file_name = gguf_path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| format!("{short_name}.gguf"));

    let file = tokio::fs::File::open(&gguf_path).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("Failed to read model file: {e}") })),
        )
    })?;

    let metadata = file.metadata().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("Failed to read file metadata: {e}") })),
        )
    })?;

    let stream = ReaderStream::new(file);
    let body = Body::from_stream(stream);

    let headers = [
        (header::CONTENT_TYPE, "application/octet-stream".to_string()),
        (
            header::CONTENT_DISPOSITION,
            format!("attachment; filename=\"{file_name}\""),
        ),
        (header::CONTENT_LENGTH, metadata.len().to_string()),
    ];

    Ok((headers, body))
}

async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

// -- Helpers --

/// Find the first .gguf file in a directory.
fn find_gguf_in_dir(dir: &std::path::Path) -> Option<PathBuf> {
    if !dir.is_dir() {
        return None;
    }

    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|ext| ext == "gguf")
        })
        .collect();

    // Sort by name to be deterministic
    entries.sort_by_key(|e| e.file_name());
    entries.first().map(|e| e.path())
}

#[allow(clippy::too_many_arguments)]
fn model_entry(
    name: &str,
    description: &str,
    languages: &[&str],
    domain: &str,
    base: &str,
    vram_gb: u32,
    source_model: &str,
    size_bytes: u64,
) -> Value {
    json!({
        "name": format!("eullm/{name}"),
        "description": description,
        "languages": languages,
        "domain": domain,
        "base": base,
        "vram_gb": vram_gb,
        "size_bytes": size_bytes,
        "source_model": source_model,
        "license": "Apache-2.0",
        "format": "gguf",
        "quantization": "Q4_K_M",
        "model_card": format!("/v1/models/{name}/card"),
        "compliance_card": format!("/v1/models/{name}/compliance"),
        "download": format!("/v1/models/{name}/download"),
    })
}

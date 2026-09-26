//! EULLM Hub — EU-hosted model registry API.
//!
//! Serves model metadata, model cards, AI Act compliance cards,
//! and GGUF model files from local storage or S3-compatible backends
//! on European infrastructure (Hetzner DE, OVH FR).

use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{StatusCode, header};
use axum::response::IntoResponse;
use axum::{Json, Router, routing::get};
use serde_json::{Value, json};
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

    // Storage root from env or default. A blank value counts as unset, the
    // same rule the engine's audit trail uses: an empty `EULLM_HUB_STORAGE`
    // (a bare `EULLM_HUB_STORAGE:` line in compose) would otherwise become
    // `PathBuf("")`, and every download would 500 on unresolvable storage
    // while listing kept working.
    let storage_root = resolve_storage_root(
        std::env::var("EULLM_HUB_STORAGE").ok().as_deref(),
        &std::env::var("HOME").unwrap_or_else(|_| "/tmp".into()),
    );

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

    // 3000 matches `EXPOSE 3000` in hub/Dockerfile and the `3000:3000` mapping
    // in docker-compose.yml. It used to default to 8080 while the image
    // advertised 3000, so `docker compose up hub` started a container nothing
    // could reach.
    let port = std::env::var("EULLM_HUB_PORT")
        .ok()
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(3000);
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
        model_entry(
            "legal-it-7b",
            "Italian legal domain — civil code, GDPR, Cassazione rulings",
            &["it", "en"],
            "legal",
            "qwen3",
            6,
            "Qwen/Qwen3-14B",
            4_500_000_000,
        ),
        model_entry(
            "medical-de-7b",
            "German medical — clinical guidelines, medical documentation",
            &["de", "en"],
            "medical",
            "qwen3",
            6,
            "Qwen/Qwen3-14B",
            4_500_000_000,
        ),
        model_entry(
            "finance-fr-7b",
            "French finance — AMF regulations, BCE directives, banking",
            &["fr", "en"],
            "finance",
            "qwen3",
            6,
            "Qwen/Qwen3-14B",
            4_500_000_000,
        ),
        model_entry(
            "general-eu-7b",
            "General purpose multilingual",
            &["en", "it", "de", "fr", "es", "pt", "nl"],
            "general",
            "qwen3",
            6,
            "Qwen/Qwen3-14B",
            4_500_000_000,
        ),
        model_entry(
            "general-eu-14b",
            "General purpose multilingual (larger)",
            &["en", "it", "de", "fr", "es", "pt", "nl"],
            "general",
            "qwen3",
            10,
            "Qwen/Qwen3-30B-A3B",
            8_500_000_000,
        ),
        model_entry(
            "legal-it-14b",
            "Italian legal domain (larger)",
            &["it", "en"],
            "legal",
            "qwen3",
            10,
            "Qwen/Qwen3-30B-A3B",
            8_200_000_000,
        ),
        model_entry(
            "code-eu-14b",
            "Multilingual coding model",
            &["en", "it", "de", "fr", "es"],
            "code",
            "deepseek",
            10,
            "deepseek-ai/DeepSeek-V3",
            8_500_000_000,
        ),
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

/// Reject a name that isn't in the catalog.
///
/// The card endpoints below used to answer 200 for *any* name, so
/// `GET /v1/models/anything-at-all/compliance` returned a fully affirmative AI
/// Act compliance card — `"gdpr_compliant": true`, "no personal data in
/// training set" — for a model that does not exist and was never assessed. For
/// a project whose value proposition is documented compliance, an attestation
/// generated on demand for an arbitrary string is worse than no endpoint at
/// all: it is the one output here that someone might reasonably rely on.
fn require_known_model(name: &str) -> Result<Value, (StatusCode, Json<Value>)> {
    find_in_catalog(name).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(json!({
                "error": format!("model '{name}' is not in this Hub's catalog"),
                "hint": "GET /v1/models lists the models this instance knows about"
            })),
        )
    })
}

async fn model_card(Path(name): Path<String>) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    require_known_model(&name)?;
    Ok(Json(json!({
        "model": format!("eullm/{name}"),
        "card_version": "1.0",
        "summary": {
            "description": format!("EULLM verticalizzato model: {name}"),
            "intended_use": "Domain-specific AI assistance for European businesses",
            "out_of_scope": "Medical diagnosis, legal advice (informational use only)",
            "architecture": "Transformer (decoder-only)",
            "base_model": "Qwen3-14B (Apache 2.0)",
            "compression_pipeline": "Structural pruning → Knowledge distillation → Identity LoRA (merged) → GGUF Q4_K_M",
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
    })))
}

async fn compliance_card(
    Path(name): Path<String>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    require_known_model(&name)?;
    Ok(Json(json!({
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
    })))
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

    if !is_valid_model_slug(short_name) {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "invalid model name" })),
        ));
    }

    // Look for GGUF file in storage
    let model_dir = state.storage_root.join(short_name);

    // Belt-and-suspenders on top of the allowlist above: canonicalize and
    // verify the resolved path is still inside storage_root before reading
    // anything from it.
    let canonical_root = state.storage_root.canonicalize().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("Failed to resolve storage root: {e}") })),
        )
    })?;
    match model_dir.canonicalize() {
        Ok(canonical_dir) if canonical_dir.starts_with(&canonical_root) => {}
        _ => {
            return Err((
                StatusCode::NOT_FOUND,
                Json(json!({
                    "error": format!("Model '{name}' not available for download on this Hub instance"),
                    "hint": "Upload the GGUF file to the Hub storage directory, or use HuggingFace directly"
                })),
            ));
        }
    }

    let mut ggufs = list_ggufs_sorted(&model_dir);

    // A sharded model must not serve its first shard as the whole model:
    // the client would treat a fragment as complete. Refuse instead.
    if ggufs.len() > 1 {
        return Err((
            StatusCode::CONFLICT,
            Json(json!({
                "error": format!(
                    "Model '{name}' is sharded ({} files); this endpoint serves single-file models only",
                    ggufs.len()
                ),
                "hint": "Download the shards from HuggingFace directly or upload a merged single GGUF"
            })),
        ));
    }

    let gguf_path = ggufs.pop().ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(json!({
                "error": format!("Model '{name}' not available for download on this Hub instance"),
                "hint": "Upload the GGUF file to the Hub storage directory, or use HuggingFace directly"
            })),
        )
    })?;

    // The directory check above does not cover the file itself:
    // `find_gguf_in_dir` matches on the entry name's extension, so a symlink
    // like `model/x.gguf -> /etc/passwd` passes the directory check and would
    // be served. Resolve the file and re-verify containment before opening.
    // Same 404 as above, so missing and rejected are indistinguishable.
    //
    // Keep the resolved path and open *that*. Checking one path and then
    // opening another leaves a window in which the link can be repointed
    // between the two syscalls, which is the hole this check exists to close.
    let canonical_path = match gguf_path.canonicalize() {
        Ok(canonical_file) if canonical_file.starts_with(&canonical_root) => canonical_file,
        _ => {
            return Err((
                StatusCode::NOT_FOUND,
                Json(json!({
                    "error": format!("Model '{name}' not available for download on this Hub instance"),
                    "hint": "Upload the GGUF file to the Hub storage directory, or use HuggingFace directly"
                })),
            ));
        }
    };

    // Named from `gguf_path`, not from the resolved path: the client should
    // get the name the operator published under `storage_root`, so a
    // legitimate in-storage symlink keeps serving under the name it was given
    // rather than leaking its target's.
    let raw_name = gguf_path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| format!("{short_name}.gguf"));
    let file_name = sanitize_download_filename(&raw_name, short_name);

    let file = tokio::fs::File::open(&canonical_path).await.map_err(|e| {
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

/// Resolve the model storage root: an explicitly set, non-blank
/// `EULLM_HUB_STORAGE` wins, anything else falls back to the default.
///
/// Pure so the precedence rule is testable without mutating process
/// environment variables (which would race every other test in the binary) —
/// the same split the engine's audit trail uses for `EULLM_AUDIT_DIR`.
fn resolve_storage_root(env: Option<&str>, home: &str) -> PathBuf {
    match env.map(str::trim).filter(|d| !d.is_empty()) {
        Some(dir) => PathBuf::from(dir),
        None => PathBuf::from(home)
            .join(".eullm")
            .join("hub")
            .join("models"),
    }
}

/// Whether `slug` is safe to join onto `storage_root` as a single path
/// component: lowercase alphanumerics, `.`, `_`, `-` only, starting with an
/// alphanumeric. Rejects `/`, `\`, `..`, and anything else that could step
/// outside storage_root once joined — in particular, rejects a raw `..` on
/// its own even though it technically matches a naive per-char allowlist,
/// since a segment that's entirely `.` is never a legitimate model name.
fn is_valid_model_slug(slug: &str) -> bool {
    let is_lower_alnum = |c: char| c.is_ascii_lowercase() || c.is_ascii_digit();
    let mut chars = slug.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    is_lower_alnum(first)
        && chars.all(|c| is_lower_alnum(c) || matches!(c, '.' | '_' | '-'))
        && !slug.contains("..")
}

/// Makes a filename from storage safe to interpolate into a quoted
/// `Content-Disposition` header value.
///
/// The name comes from the filesystem, where `"`, `\` and CR/LF are all legal
/// and all survive `to_string_lossy`. The two cases differ:
///
/// - `"` and `\` are valid header-value bytes, so they reach the client and
///   break out of the quoted string in `Content-Disposition`. This is the
///   injection the function exists to stop.
/// - CR/LF and the other ASCII controls cannot split the response: axum builds
///   the header through `TryInto<HeaderValue>`, which rejects bytes below
///   `0x20`, and the conversion error is returned as a 500. Mapping them to
///   `_` turns a download that fails into one that works.
///
/// Quotes, backslashes and ASCII controls become `_`; everything else,
/// including non-ASCII names, passes through unchanged.
fn sanitize_download_filename(raw: &str, short_name: &str) -> String {
    let clean: String = raw
        .chars()
        .map(|c| {
            if c == '"' || c == '\\' || c.is_control() {
                '_'
            } else {
                c
            }
        })
        .collect();
    if clean.is_empty() {
        format!("{short_name}.gguf")
    } else {
        clean
    }
}

/// All .gguf files in a directory, sorted by name for determinism.
fn list_ggufs_sorted(dir: &std::path::Path) -> Vec<PathBuf> {
    if !dir.is_dir() {
        return Vec::new();
    }

    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .ok()
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter(|e| e.path().extension().is_some_and(|ext| ext == "gguf"))
                .collect()
        })
        .unwrap_or_default();

    // Sort by name to be deterministic
    entries.sort_by_key(|e| e.file_name());
    entries.into_iter().map(|e| e.path()).collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_valid_model_slug_accepts_real_names() {
        assert!(is_valid_model_slug("legal-it-7b"));
        assert!(is_valid_model_slug("medical-de-7b"));
        assert!(is_valid_model_slug("finance-fr-7b"));
        assert!(is_valid_model_slug("a"));
        assert!(is_valid_model_slug("model.v2"));
    }

    #[test]
    fn is_valid_model_slug_rejects_traversal() {
        assert!(!is_valid_model_slug("../../../../etc/passwd"));
        assert!(!is_valid_model_slug("..%2F..%2F..%2Fetc"));
        assert!(!is_valid_model_slug("foo/../bar"));
        assert!(!is_valid_model_slug(".."));
        assert!(!is_valid_model_slug("foo/bar"));
        assert!(!is_valid_model_slug("foo\\bar"));
    }

    #[test]
    fn is_valid_model_slug_rejects_absolute_paths_and_bad_chars() {
        assert!(!is_valid_model_slug("/etc/shadow"));
        assert!(!is_valid_model_slug(""));
        assert!(!is_valid_model_slug("-leading-dash"));
        assert!(!is_valid_model_slug("UPPER-case"));
        assert!(!is_valid_model_slug("has space"));
    }

    #[test]
    fn blank_storage_env_falls_back_to_the_default() {
        let home = "/home/tester";
        let default = PathBuf::from(home)
            .join(".eullm")
            .join("hub")
            .join("models");
        // Unset, empty, and whitespace-only all mean "not configured" — the
        // same rule the engine's audit trail uses. An empty value must never
        // become PathBuf(""), which unresolvable storage 500s every download.
        assert_eq!(resolve_storage_root(None, home), default);
        assert_eq!(resolve_storage_root(Some(""), home), default);
        assert_eq!(resolve_storage_root(Some("   "), home), default);
        assert_eq!(
            resolve_storage_root(Some("/data/models"), home),
            PathBuf::from("/data/models")
        );
    }

    fn scratch_model_dir(name: &str, files: &[&str]) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("eullm-hub-shard-test-{}", uuid::Uuid::new_v4()));
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        for f in files {
            std::fs::write(dir.join(f), b"dummy").unwrap();
        }
        root
    }

    #[test]
    fn sharded_model_dir_lists_every_shard() {
        let root = scratch_model_dir(
            "testmodel",
            &[
                "testmodel-00001-of-00002.gguf",
                "testmodel-00002-of-00002.gguf",
                "notes.txt",
            ],
        );
        let listed = list_ggufs_sorted(&root.join("testmodel"));
        assert_eq!(listed.len(), 2);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn missing_model_dir_lists_nothing() {
        let absent =
            std::env::temp_dir().join(format!("eullm-hub-absent-{}", uuid::Uuid::new_v4()));
        assert!(list_ggufs_sorted(&absent).is_empty());
    }

    #[tokio::test]
    async fn sharded_model_is_refused_not_partially_served() {
        use axum::extract::{Path, State};
        let root = scratch_model_dir(
            "testmodel",
            &[
                "testmodel-00001-of-00002.gguf",
                "testmodel-00002-of-00002.gguf",
            ],
        );
        let state = Arc::new(HubState {
            storage_root: root.clone(),
        });
        let err = download_model(State(state), Path("testmodel".to_string()))
            .await
            .err()
            .expect("a sharded model must be refused, not served");
        assert_eq!(err.0, StatusCode::CONFLICT);
        std::fs::remove_dir_all(&root).ok();
    }

    #[tokio::test]
    async fn single_file_model_still_serves() {
        use axum::extract::{Path, State};
        let root = scratch_model_dir("testmodel", &["testmodel.gguf"]);
        let state = Arc::new(HubState {
            storage_root: root.clone(),
        });
        assert!(
            download_model(State(state), Path("testmodel".to_string()))
                .await
                .is_ok()
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn plain_and_unicode_filenames_pass_through() {
        assert_eq!(
            sanitize_download_filename("model.gguf", "model"),
            "model.gguf"
        );
        // Every character here is ASCII; the non-ASCII cases are below.
        assert_eq!(
            sanitize_download_filename("Modello 2026.gguf", "model"),
            "Modello 2026.gguf"
        );
        // Accents, CJK and an emoji: all multi-byte, none of them a control
        // character, so all must survive untouched.
        assert_eq!(
            sanitize_download_filename("modèllo-perità.gguf", "model"),
            "modèllo-perità.gguf"
        );
        assert_eq!(
            sanitize_download_filename("日本語モデル.gguf", "model"),
            "日本語モデル.gguf"
        );
        assert_eq!(
            sanitize_download_filename("modello-🇪🇺.gguf", "model"),
            "modello-🇪🇺.gguf"
        );
    }

    /// The property the sanitizer actually owes the caller: whatever comes out
    /// of it can be interpolated into `Content-Disposition` and still build a
    /// `HeaderValue`. Without this the function is only tested against the
    /// characters someone thought to list.
    #[test]
    fn sanitized_names_always_build_a_header_value() {
        for raw in [
            "model.gguf",
            "modèllo-perità.gguf",
            "日本語モデル.gguf",
            "evil\".gguf",
            "a\\b.gguf",
            "a\r\nX-Evil: 1.gguf",
            "\u{7f}del.gguf",
            "",
        ] {
            let clean = sanitize_download_filename(raw, "model");
            // `TryFrom<String>` is the conversion axum itself performs on the
            // `[(HeaderName, String); N]` this handler returns, so testing any
            // other one would be testing the wrong thing.
            let header = format!("attachment; filename=\"{clean}\"");
            assert!(
                axum::http::HeaderValue::try_from(header).is_ok(),
                "sanitized name did not produce a valid header value: {raw:?} -> {clean:?}"
            );
        }
    }

    #[test]
    fn quotes_backslashes_and_crlf_become_underscores() {
        assert_eq!(
            sanitize_download_filename("evil\".gguf", "model"),
            "evil_.gguf"
        );
        assert_eq!(sanitize_download_filename("a\\b.gguf", "model"), "a_b.gguf");
        assert_eq!(
            sanitize_download_filename("a\r\nX-Evil-1.gguf", "model"),
            "a__X-Evil-1.gguf"
        );
    }

    #[test]
    fn empty_sanitized_name_falls_back_to_slug() {
        assert_eq!(sanitize_download_filename("", "model"), "model.gguf");
    }
}

#[cfg(test)]
mod card_scope_tests {
    use super::*;

    #[test]
    fn cards_are_only_issued_for_catalogued_models() {
        // Accepts the real catalog entries, with or without the eullm/ prefix.
        assert!(require_known_model("legal-it-7b").is_ok());
        assert!(require_known_model("eullm/legal-it-7b").is_ok());
        assert!(require_known_model("medical-de-7b").is_ok());
    }

    #[test]
    fn an_unknown_name_gets_404_not_an_affirmative_compliance_card() {
        for name in [
            "anything-at-all",
            "not-a-model",
            "legal-it-70b",
            "",
            "../etc/passwd",
        ] {
            let err = require_known_model(name)
                .err()
                .unwrap_or_else(|| panic!("{name:?} must not be issued a card"));
            assert_eq!(err.0, StatusCode::NOT_FOUND, "for {name:?}");
        }
    }
}

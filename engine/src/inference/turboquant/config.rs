//! TurboQuant runtime configuration and KV cache type integration.
//!
//! Bridges TurboQuant types with the engine's `KvCacheType` system.
//! When the llama.cpp backend supports TQ types natively, this module
//! maps them directly.  Otherwise, it returns a fallback type (F16)
//! and logs a warning — unless strict mode is enabled, in which case
//! it returns an error.

use super::types::{TurboquantType, parse_turboquant_type};

/// Result of resolving a TurboQuant cache type for the current backend.
#[derive(Debug)]
pub enum ResolvedCacheType {
    /// The backend supports this TQ type natively.
    Native(TurboquantType),
    /// The backend does not support TQ; fell back to this standard type.
    Fallback {
        requested: TurboquantType,
        fallback: &'static str,
    },
    /// Strict mode: the backend does not support TQ and fallback is disabled.
    Unsupported {
        requested: TurboquantType,
        reason: String,
    },
}

/// Check whether the string is a TurboQuant cache type.
pub fn is_turboquant_type(s: &str) -> bool {
    parse_turboquant_type(s).is_some()
}

/// Attempt to resolve a TurboQuant cache type for the current backend.
///
/// Returns `None` if `s` is not a TurboQuant type (i.e., it's a standard
/// type like "f16" or "q8_0" and should be handled by the normal path).
///
/// When `strict` is true, returns `Unsupported` instead of `Fallback`
/// if the backend does not support TQ — use this for benchmarks where
/// silent fallback would produce misleading results.
pub fn resolve_turboquant_cache_type(s: &str, strict: bool) -> Option<ResolvedCacheType> {
    let tq_type = parse_turboquant_type(s)?;

    if backend_supports_turboquant() {
        tracing::info!(
            "TurboQuant {tq_type} — native backend support detected (turboquant_native)"
        );
        Some(ResolvedCacheType::Native(tq_type))
    } else if strict {
        tracing::error!(
            "TurboQuant {tq_type} requested in strict mode but backend does not support it. \
             Build with --features turboquant_native against the AmesianX/TurboQuant fork."
        );
        Some(ResolvedCacheType::Unsupported {
            requested: tq_type,
            reason: format!(
                "TurboQuant {tq_type} is not supported by the current llama.cpp backend. \
                 Run scripts/setup-turboquant.sh, uncomment [patch.crates-io] in Cargo.toml, \
                 and rebuild with --features turboquant_native."
            ),
        })
    } else {
        tracing::warn!(
            "TurboQuant {tq_type} requested but backend lacks native support — \
             falling back to F16 KV cache. Build with --features turboquant_native \
             for real TQ compression."
        );
        Some(ResolvedCacheType::Fallback {
            requested: tq_type,
            fallback: "f16",
        })
    }
}

/// Log the TurboQuant status at engine startup.
pub fn log_turboquant_status() {
    if backend_supports_turboquant() {
        tracing::info!(
            "TurboQuant: ACTIVE (AmesianX v1.4.1+, tbq3_0/tbq4_0/tbqp3_0/tbqp4_0 available)"
        );
    } else {
        tracing::info!(
            "TurboQuant: STANDBY (module loaded, backend lacks native support — \
             tbq3_0/tbq4_0 will fall back to F16)"
        );
    }
}

/// Detect whether the linked llama.cpp backend supports TurboQuant.
///
/// Returns true when built with `--features turboquant_native`, which
/// implies the vendored llama.cpp has TQ type definitions (GGML type
/// IDs 41-44).  This is set automatically when using the AmesianX fork
/// via `scripts/setup-turboquant.sh`.
fn backend_supports_turboquant() -> bool {
    cfg!(feature = "turboquant_native")
}

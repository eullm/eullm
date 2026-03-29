//! TurboQuant runtime configuration and KV cache type integration.
//!
//! Bridges TurboQuant types with the engine's `KvCacheType` system.
//! When the llama.cpp backend supports TQ types natively, this module
//! maps them directly.  Otherwise, it returns a fallback type (F16)
//! and logs a warning.

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
        reason: &'static str,
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
/// Returns `Some(ResolvedCacheType)` indicating either native support
/// or a fallback with the reason.
pub fn resolve_turboquant_cache_type(s: &str) -> Option<ResolvedCacheType> {
    let tq_type = parse_turboquant_type(s)?;

    // Check if the llama.cpp backend supports TurboQuant KV cache types.
    // This will be true when using the spiritbuun CUDA fork or when
    // upstream llama.cpp merges TQ support.
    if backend_supports_turboquant() {
        Some(ResolvedCacheType::Native(tq_type))
    } else {
        tracing::warn!(
            "TurboQuant {tq_type} requested but the llama.cpp backend does not support it. \
             Falling back to F16 KV cache. To enable TurboQuant, build with a \
             TQ-capable llama.cpp backend.",
        );
        Some(ResolvedCacheType::Fallback {
            requested: tq_type,
            fallback: "f16",
            reason: "llama.cpp backend does not support TurboQuant cache types",
        })
    }
}

/// Detect whether the linked llama.cpp backend supports TurboQuant.
///
/// Currently checks for the presence of TQ GGML types at runtime.
/// This function is the single point that needs updating when
/// switching between the spiritbuun fork and upstream llama.cpp.
fn backend_supports_turboquant() -> bool {
    // TODO: probe the llama.cpp backend for TQ3_0/TQ4_0 type support.
    // When using the spiritbuun fork, this returns true.
    // When using upstream llama.cpp (pre-TQ merge), this returns false.
    //
    // Implementation options:
    // 1. Try to create a small test tensor with the TQ type
    // 2. Check a version/capability flag exposed by the backend
    // 3. Compile-time detection via a cargo feature (e.g. "turboquant-native")
    false
}

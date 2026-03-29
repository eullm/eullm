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
/// Checks at compile time whether the vendored llama.cpp includes
/// TurboQuant type definitions.  When built against the spiritbuun
/// fork (via scripts/setup-turboquant.sh + [patch.crates-io]),
/// this returns true.  With upstream llama.cpp, it returns false.
///
/// This is the single point that needs updating when switching
/// between backends.  All other code uses this function to decide
/// whether to use TQ types or fall back.
fn backend_supports_turboquant() -> bool {
    // The vendored llama.cpp defines GGML_TYPE_TURBO3_0 = 41 when
    // TurboQuant is available.  We probe this via the KvCacheType
    // enum exposed by llama-cpp-2.  If the Unknown(41) variant
    // round-trips correctly through the FFI, the backend has TQ.
    //
    // For now, we use a compile-time check: the setup script sets
    // a cargo cfg flag that we detect here.
    cfg!(feature = "turboquant_native")
}

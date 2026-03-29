//! TurboQuant KV cache type definitions.
//!
//! Defines the TQ3_0 and TQ4_0 cache types and their mapping to/from
//! llama.cpp's `KvCacheType` enum.  When the llama.cpp backend supports
//! TurboQuant natively (via fork or upstream merge), these types map
//! directly to GGML quant types.  Otherwise, the engine falls back to F16.

use std::fmt;

/// TurboQuant cache type variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurboquantType {
    /// 3-bit TurboQuant: ~5x compression, ~99.5% attention cosine similarity.
    /// Block layout: 4-byte norm + packed 3-bit indices per 128-element vector.
    TQ3_0,
    /// 4-bit TurboQuant: ~3.8x compression, ~99.8% attention cosine similarity.
    /// Block layout: 4-byte norm + packed 4-bit indices per 128-element vector.
    TQ4_0,
}

impl TurboquantType {
    /// Bits per value (excluding overhead).
    pub fn bits(&self) -> u8 {
        match self {
            Self::TQ3_0 => 3,
            Self::TQ4_0 => 4,
        }
    }

    /// Approximate compression ratio vs FP16.
    pub fn compression_ratio(&self) -> f32 {
        16.0 / self.bits() as f32
    }
}

impl fmt::Display for TurboquantType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TQ3_0 => write!(f, "tq3_0"),
            Self::TQ4_0 => write!(f, "tq4_0"),
        }
    }
}

/// Parse a TurboQuant type from a CLI/API string.
pub fn parse_turboquant_type(s: &str) -> Option<TurboquantType> {
    match s.to_lowercase().as_str() {
        "tq3_0" | "tq3" | "turbo3" => Some(TurboquantType::TQ3_0),
        "tq4_0" | "tq4" | "turbo4" => Some(TurboquantType::TQ4_0),
        _ => None,
    }
}

/// GGML type IDs used by the spiritbuun/llama-cpp-turboquant-cuda fork.
///
/// These IDs are added to the `ggml_type` enum in the fork's `ggml.h`.
/// When upstream llama.cpp merges TurboQuant support, the IDs may change
/// — update these constants accordingly.
pub mod ggml_ids {
    /// `GGML_TYPE_TURBO3_0` = 41 in the spiritbuun fork.
    pub const TURBO3_0: u32 = 41;
    /// `GGML_TYPE_TURBO4_0` = 42 in the spiritbuun fork.
    pub const TURBO4_0: u32 = 42;
    /// `GGML_TYPE_TURBO2_0` = 43 in the spiritbuun fork (not yet exposed).
    pub const TURBO2_0: u32 = 43;
}

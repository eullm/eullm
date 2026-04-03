//! TurboQuant KV cache type definitions.
//!
//! Defines the TBQ3_0 / TBQ4_0 (MSE-only) and TBQP3_0 / TBQP4_0 (MSE + QJL
//! sign correction) cache types and their mapping to/from llama.cpp's
//! `KvCacheType` enum.  Backend: AmesianX/TurboQuant v1.4.1+.

use std::fmt;

/// TurboQuant cache type variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurboquantType {
    /// 3-bit TurboQuant (MSE-only): ~5x compression.
    /// FWHT rotation + Lloyd-Max quantization, blck_size=256.
    TQ3_0,
    /// 4-bit TurboQuant (MSE-only): ~3.8x compression.
    /// FWHT rotation + Lloyd-Max quantization, blck_size=256.
    TQ4_0,
    /// 3-bit TurboQuant with QJL sign correction (higher quality, same compression).
    /// Requires independent sign patterns for WHT and SRHT.
    TQP3_0,
    /// 4-bit TurboQuant with QJL sign correction (highest quality, ~3.8x compression).
    TQP4_0,
}

impl TurboquantType {
    /// Bits per value (excluding overhead).
    pub fn bits(&self) -> u8 {
        match self {
            Self::TQ3_0 | Self::TQP3_0 => 3,
            Self::TQ4_0 | Self::TQP4_0 => 4,
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
            Self::TQ3_0  => write!(f, "tbq3_0"),
            Self::TQ4_0  => write!(f, "tbq4_0"),
            Self::TQP3_0 => write!(f, "tbqp3_0"),
            Self::TQP4_0 => write!(f, "tbqp4_0"),
        }
    }
}

/// Parse a TurboQuant type from a CLI/API string.
/// Accepts legacy spiritbuun names (tq3_0, turbo3) and AmesianX names (tbq3_0, tbqp3_0).
pub fn parse_turboquant_type(s: &str) -> Option<TurboquantType> {
    match s.to_lowercase().as_str() {
        "tq3_0" | "tq3" | "turbo3" | "tbq3_0"   => Some(TurboquantType::TQ3_0),
        "tq4_0" | "tq4" | "turbo4" | "tbq4_0"   => Some(TurboquantType::TQ4_0),
        "tbqp3_0" | "tqp3_0" | "tqp3"            => Some(TurboquantType::TQP3_0),
        "tbqp4_0" | "tqp4_0" | "tqp4"            => Some(TurboquantType::TQP4_0),
        _ => None,
    }
}

/// GGML type IDs from AmesianX/TurboQuant v1.4.1 (ggml/include/ggml.h).
///
/// IDs 41–44 are stable across spiritbuun and AmesianX forks.
/// When upstream llama.cpp merges TurboQuant, these IDs may be confirmed
/// or renumbered — update these constants accordingly.
pub mod ggml_ids {
    /// `GGML_TYPE_TBQ3_0` = 41 — TurboQuant 3-bit MSE-only.
    pub const TBQ3_0: u32 = 41;
    /// `GGML_TYPE_TBQ4_0` = 42 — TurboQuant 4-bit MSE-only.
    pub const TBQ4_0: u32 = 42;
    /// `GGML_TYPE_TBQP3_0` = 43 — TurboQuant 3-bit with QJL correction.
    pub const TBQP3_0: u32 = 43;
    /// `GGML_TYPE_TBQP4_0` = 44 — TurboQuant 4-bit with QJL correction.
    pub const TBQP4_0: u32 = 44;
}

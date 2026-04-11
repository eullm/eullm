//! TurboQuant KV cache type definitions.
//!
//! AmesianX/TurboQuant v1.5.3 auto-detects head_dim at the llama-cli level,
//! but when using llama-cpp-2 Rust bindings we bypass that layer and must
//! pass the correct GGML type ID directly.
//!
//! Suffix convention:
//!   _0 = blck_size=256  (head_dim=256, QJL correction)
//!   _1 = blck_size=128  (head_dim=128, Direct Sign correction) ← most models
//!   _2 = blck_size=64   (head_dim=64,  V-only; K falls back to q8_0)
//!   _4 = blck_size=576  (head_dim=576, GLM-4.7-Flash)
//!
//! Common models by head_dim:
//!   head_dim=128: Qwen3, Llama, Mistral, Falcon → use tbq3_1 / tbqp3_1
//!   head_dim=256: Qwen2.5-72B, some large models → use tbq3_0 / tbqp3_0
//!   head_dim=64:  Phi-3-mini, some small models  → use tbq3_2 / tbqp3_2

use std::fmt;

/// TurboQuant cache type variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurboquantType {
    // ── head_dim=256 (blck_size=256, QJL correction) ──────────────────────
    /// 3-bit MSE-only, head_dim=256.
    TQ3_0,
    /// 4-bit MSE-only, head_dim=256.
    TQ4_0,
    /// 3-bit with QJL correction, head_dim=256.
    TQP3_0,
    /// 4-bit with QJL correction, head_dim=256.
    TQP4_0,

    // ── head_dim=128 (blck_size=128, Direct Sign correction) ───────────────
    // Most common: Qwen3, Llama, Mistral, Falcon
    /// 3-bit MSE-only, head_dim=128.
    TQ3_1,
    /// 4-bit MSE-only, head_dim=128.
    TQ4_1,
    /// 3-bit with Direct Sign correction, head_dim=128.
    TQP3_1,
    /// 4-bit with Direct Sign correction, head_dim=128.
    TQP4_1,

    // ── head_dim=64 (blck_size=64; K auto-falls back to q8_0) ─────────────
    /// 3-bit MSE-only, head_dim=64 (V only; K → q8_0).
    TQ3_2,
    /// 4-bit MSE-only, head_dim=64 (V only; K → q8_0).
    TQ4_2,
}

impl TurboquantType {
    /// Bits per value (excluding overhead).
    pub fn bits(&self) -> u8 {
        match self {
            Self::TQ3_0 | Self::TQP3_0 |
            Self::TQ3_1 | Self::TQP3_1 |
            Self::TQ3_2 => 3,
            Self::TQ4_0 | Self::TQP4_0 |
            Self::TQ4_1 | Self::TQP4_1 |
            Self::TQ4_2 => 4,
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
            Self::TQ3_1  => write!(f, "tbq3_1"),
            Self::TQ4_1  => write!(f, "tbq4_1"),
            Self::TQP3_1 => write!(f, "tbqp3_1"),
            Self::TQP4_1 => write!(f, "tbqp4_1"),
            Self::TQ3_2  => write!(f, "tbq3_2"),
            Self::TQ4_2  => write!(f, "tbq4_2"),
        }
    }
}

/// Parse a TurboQuant type from a CLI/API string.
///
/// AmesianX auto-detects head_dim at the llama-cli layer; via Rust bindings
/// the suffix must be correct for the model. Most models (Qwen3, Llama,
/// Mistral) have head_dim=128 → use tbq4_1 / tbqp4_1.
pub fn parse_turboquant_type(s: &str) -> Option<TurboquantType> {
    match s.to_lowercase().as_str() {
        // head_dim=256 (_0)
        "tbq3_0"  | "tq3_0" | "tq3" | "turbo3"  => Some(TurboquantType::TQ3_0),
        "tbq4_0"  | "tq4_0" | "tq4" | "turbo4"  => Some(TurboquantType::TQ4_0),
        "tbqp3_0" | "tqp3_0"                     => Some(TurboquantType::TQP3_0),
        "tbqp4_0" | "tqp4_0"                     => Some(TurboquantType::TQP4_0),
        // head_dim=128 (_1) — Qwen3, Llama, Mistral, Falcon
        "tbq3_1"  | "tq3_1"                      => Some(TurboquantType::TQ3_1),
        "tbq4_1"  | "tq4_1"                      => Some(TurboquantType::TQ4_1),
        "tbqp3_1" | "tqp3_1"                     => Some(TurboquantType::TQP3_1),
        "tbqp4_1" | "tqp4_1"                     => Some(TurboquantType::TQP4_1),
        // head_dim=64 (_2) — Phi-3-mini
        "tbq3_2"  | "tq3_2"                      => Some(TurboquantType::TQ3_2),
        "tbq4_2"  | "tq4_2"                      => Some(TurboquantType::TQ4_2),
        _ => None,
    }
}

/// GGML type IDs from AmesianX/TurboQuant v1.5.3 (ggml/include/ggml.h).
pub mod ggml_ids {
    // head_dim=256 (_0)
    pub const TBQ3_0:  u32 = 41;
    pub const TBQ4_0:  u32 = 42;
    pub const TBQP3_0: u32 = 43;
    pub const TBQP4_0: u32 = 44;
    // head_dim=128 (_1)
    pub const TBQ3_1:  u32 = 45;
    pub const TBQ4_1:  u32 = 46;
    pub const TBQP3_1: u32 = 47;
    pub const TBQP4_1: u32 = 48;
    // head_dim=64 (_2)
    pub const TBQ3_2:  u32 = 49;
    pub const TBQ4_2:  u32 = 50;
}

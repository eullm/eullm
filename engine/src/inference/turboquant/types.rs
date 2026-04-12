//! TurboQuant KV cache type definitions.
//!
//! AmesianX/TurboQuant v1.5.3 auto-detects head_dim at the llama-cli level,
//! but when using llama-cpp-2 Rust bindings we bypass that layer and must
//! pass the correct GGML type ID directly.
//!
//! Suffix convention (v1.5.3):
//!   _0 = blck_size=256  (head_dim=256)
//!   _1 = blck_size=128  (head_dim=128) ← most models (Qwen3, Llama, Mistral)
//!   _2 = blck_size=64   (head_dim=64,  single WHT — legacy)
//!   _3 = blck_size=64   (head_dim=64,  double WHT per-head — v1.5.3 fix)
//!   _4 = blck_size=576  (head_dim=576, GLM-4.7-Flash)
//!
//! Common models by head_dim:
//!   head_dim=128: Qwen3, Llama, Mistral, Falcon → use tbqp3_1 / tbq3_1
//!   head_dim=256: Qwen2.5-72B, some large models → use tbqp3_0 / tbq3_0
//!   head_dim=64:  Phi-3-mini (v1.5.3+: use _3 for multi-turn) → tbqp3_3 / tbq3_3
//!   head_dim=576: GLM-4.7-Flash → use tbqp3_4 / tbq3_4

use std::fmt;

/// TurboQuant cache type variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurboquantType {
    // ── head_dim=256 (_0) ─────────────────────────────────────────────────
    /// 3-bit MSE-only, head_dim=256.
    TQ3_0,
    /// 4-bit MSE-only, head_dim=256.
    TQ4_0,
    /// 3-bit with QJL correction, head_dim=256.
    TQP3_0,
    /// 4-bit with QJL correction, head_dim=256.
    TQP4_0,

    // ── head_dim=128 (_1) — Qwen3, Llama, Mistral, Falcon ────────────────
    /// 3-bit MSE-only, head_dim=128.
    TQ3_1,
    /// 4-bit MSE-only, head_dim=128.
    TQ4_1,
    /// 3-bit with Direct Sign correction, head_dim=128.
    TQP3_1,
    /// 4-bit with Direct Sign correction, head_dim=128.
    TQP4_1,

    // ── head_dim=64 (_2) — single WHT, legacy ─────────────────────────────
    /// 3-bit MSE-only, head_dim=64, single WHT.
    TQ3_2,
    /// 4-bit MSE-only, head_dim=64, single WHT.
    TQ4_2,
    /// 3-bit with correction, head_dim=64, single WHT.
    TQP3_2,
    /// 4-bit with correction, head_dim=64, single WHT.
    TQP4_2,

    // ── head_dim=64 (_3) — double WHT per-head, v1.5.3 fix ───────────────
    /// 3-bit MSE-only, head_dim=64, double WHT per-head.
    TQ3_3,
    /// 4-bit MSE-only, head_dim=64, double WHT per-head.
    TQ4_3,
    /// 3-bit with correction + QJL, head_dim=64, double WHT (recommended D=64).
    TQP3_3,
    /// 4-bit with correction + QJL, head_dim=64, double WHT (recommended D=64).
    TQP4_3,

    // ── head_dim=576 (_4) — GLM-4.7-Flash ────────────────────────────────
    /// 3-bit MSE-only, head_dim=576.
    TQ3_4,
    /// 4-bit MSE-only, head_dim=576.
    TQ4_4,
    /// 3-bit with correction, head_dim=576.
    TQP3_4,
    /// 4-bit with correction, head_dim=576.
    TQP4_4,
}

impl TurboquantType {
    /// Bits per value (excluding overhead).
    pub fn bits(&self) -> u8 {
        match self {
            Self::TQ3_0 | Self::TQP3_0 |
            Self::TQ3_1 | Self::TQP3_1 |
            Self::TQ3_2 | Self::TQP3_2 |
            Self::TQ3_3 | Self::TQP3_3 |
            Self::TQ3_4 | Self::TQP3_4 => 3,
            Self::TQ4_0 | Self::TQP4_0 |
            Self::TQ4_1 | Self::TQP4_1 |
            Self::TQ4_2 | Self::TQP4_2 |
            Self::TQ4_3 | Self::TQP4_3 |
            Self::TQ4_4 | Self::TQP4_4 => 4,
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
            Self::TQP3_2 => write!(f, "tbqp3_2"),
            Self::TQP4_2 => write!(f, "tbqp4_2"),
            Self::TQ3_3  => write!(f, "tbq3_3"),
            Self::TQ4_3  => write!(f, "tbq4_3"),
            Self::TQP3_3 => write!(f, "tbqp3_3"),
            Self::TQP4_3 => write!(f, "tbqp4_3"),
            Self::TQ3_4  => write!(f, "tbq3_4"),
            Self::TQ4_4  => write!(f, "tbq4_4"),
            Self::TQP3_4 => write!(f, "tbqp3_4"),
            Self::TQP4_4 => write!(f, "tbqp4_4"),
        }
    }
}

/// Parse a TurboQuant type from a CLI/API string.
///
/// AmesianX auto-detects head_dim at the llama-cli layer; via Rust bindings
/// the suffix must be correct for the model. Most models (Qwen3, Llama,
/// Mistral) have head_dim=128 → use tbqp3_1 / tbq3_1.
///
/// Bare aliases without suffix (tbqp3, tbq3, tbqp4, tbq4) default to _1
/// (head_dim=128) following the AmesianX recommended config:
///   --cache-type-k tbqp3 --cache-type-v tbq3 --flash-attn on
pub fn parse_turboquant_type(s: &str) -> Option<TurboquantType> {
    match s.to_lowercase().as_str() {
        // head_dim=256 (_0)
        "tbq3_0"  | "tq3_0" | "tq3" | "turbo3"  => Some(TurboquantType::TQ3_0),
        "tbq4_0"  | "tq4_0" | "tq4" | "turbo4"  => Some(TurboquantType::TQ4_0),
        "tbqp3_0" | "tqp3_0"                     => Some(TurboquantType::TQP3_0),
        "tbqp4_0" | "tqp4_0"                     => Some(TurboquantType::TQP4_0),
        // head_dim=128 (_1) — Qwen3, Llama, Mistral, Falcon
        // Bare aliases (no suffix) map here — AmesianX recommended defaults.
        "tbq3_1"  | "tq3_1"  | "tbq3"            => Some(TurboquantType::TQ3_1),
        "tbq4_1"  | "tq4_1"  | "tbq4"            => Some(TurboquantType::TQ4_1),
        "tbqp3_1" | "tqp3_1" | "tbqp3"           => Some(TurboquantType::TQP3_1),
        "tbqp4_1" | "tqp4_1" | "tbqp4"           => Some(TurboquantType::TQP4_1),
        // head_dim=64, single WHT (_2) — legacy
        "tbq3_2"  | "tq3_2"                      => Some(TurboquantType::TQ3_2),
        "tbq4_2"  | "tq4_2"                      => Some(TurboquantType::TQ4_2),
        "tbqp3_2" | "tqp3_2"                     => Some(TurboquantType::TQP3_2),
        "tbqp4_2" | "tqp4_2"                     => Some(TurboquantType::TQP4_2),
        // head_dim=64, double WHT per-head (_3) — v1.5.3 fix, recommended for D=64
        "tbq3_3"  | "tq3_3"                      => Some(TurboquantType::TQ3_3),
        "tbq4_3"  | "tq4_3"                      => Some(TurboquantType::TQ4_3),
        "tbqp3_3" | "tqp3_3"                     => Some(TurboquantType::TQP3_3),
        "tbqp4_3" | "tqp4_3"                     => Some(TurboquantType::TQP4_3),
        // head_dim=576 (_4) — GLM-4.7-Flash
        "tbq3_4"  | "tq3_4"                      => Some(TurboquantType::TQ3_4),
        "tbq4_4"  | "tq4_4"                      => Some(TurboquantType::TQ4_4),
        "tbqp3_4" | "tqp3_4"                     => Some(TurboquantType::TQP3_4),
        "tbqp4_4" | "tqp4_4"                     => Some(TurboquantType::TQP4_4),
        _ => None,
    }
}

/// GGML type IDs from AmesianX/TurboQuant v1.5.3 (ggml/include/ggml.h).
///
/// v1.5.3 inserted GGML_TYPE_Q1_0=41 before the TBQ block, shifting all
/// TBQ IDs by +1 versus v1.5.2 (TBQ3_0 was 41, now 42).
pub mod ggml_ids {
    // head_dim=256 (_0)
    pub const TBQ3_0:   u32 = 42;
    pub const TBQ4_0:   u32 = 43;
    pub const TBQP3_0:  u32 = 44;
    pub const TBQP4_0:  u32 = 45;
    // head_dim=128 (_1)
    pub const TBQ3_1:   u32 = 46;
    pub const TBQ4_1:   u32 = 47;
    pub const TBQP3_1:  u32 = 48;
    pub const TBQP4_1:  u32 = 49;
    // head_dim=64, single WHT (_2)
    pub const TBQ3_2:   u32 = 50;
    pub const TBQ4_2:   u32 = 51;
    pub const TBQP3_2:  u32 = 52;
    pub const TBQP4_2:  u32 = 53;
    // head_dim=64, double WHT per-head (_3) — v1.5.3 fix
    pub const TBQ3_3:   u32 = 54;
    pub const TBQ4_3:   u32 = 55;
    pub const TBQP3_3:  u32 = 56;
    pub const TBQP4_3:  u32 = 57;
    // head_dim=576, GLM-4.7-Flash (_4)
    pub const TBQ3_4:   u32 = 58;
    pub const TBQ4_4:   u32 = 59;
    pub const TBQP3_4:  u32 = 60;
    pub const TBQP4_4:  u32 = 61;
}

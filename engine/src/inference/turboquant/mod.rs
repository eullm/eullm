//! TurboQuant — Extreme KV cache compression for inference.
//!
//! Implements Google's TurboQuant algorithm (Zandieh et al., ICLR 2026) for
//! compressing the key-value cache at inference time.  This is orthogonal to
//! model weight quantization (GPTQ/AWQ/GGML Q4_K_M) — both can be applied
//! simultaneously for compounding memory savings.
//!
//! ## Algorithm
//!
//! 1. **WHT rotation**: Apply a Walsh-Hadamard Transform to each KV vector,
//!    making coordinates follow a predictable near-Gaussian distribution.
//! 2. **Lloyd-Max quantization**: Quantize each rotated coordinate using
//!    precomputed optimal scalar codebooks (3-4 bits per value).
//!
//! QJL residual correction (Algorithm 2) behaviour depends on head_dim:
//! for D≥128 it is omitted (increases variance, hurts attention quality);
//! for D=64 it is re-enabled in the AmesianX fork (v1.5.3) — critical for
//! multi-turn accuracy (verified stable across 9+ turns on GPT-OSS 120B).
//!
//! ## Feature gate
//!
//! This module is only compiled when `--features turboquant` is enabled.
//! Without the feature flag, zero code from this module enters the binary.
//!
//! ## Status: Experimental
//!
//! TurboQuant KV cache types (`tq3_0`, `tq4_0`) are experimental.  The engine
//! falls back to F16 automatically if the llama.cpp backend does not support
//! TurboQuant cache types.

pub mod types;
pub mod config;
pub mod codebook;

pub use config::log_turboquant_status;

#[cfg(test)]
mod tests;

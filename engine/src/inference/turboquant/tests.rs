//! Unit tests for TurboQuant module.

use super::codebook::{
    CODEBOOK_3BIT, CODEBOOK_4BIT,
    boundaries_3bit, boundaries_4bit,
    quantize_scalar_3bit, quantize_scalar_4bit,
};
use super::config::{is_turboquant_type, resolve_turboquant_cache_type, ResolvedCacheType};
use super::types::{TurboquantType, parse_turboquant_type};

#[test]
fn parse_tq_types() {
    // head_dim=256 (_0)
    assert_eq!(parse_turboquant_type("tbq3_0"),  Some(TurboquantType::TQ3_0));
    assert_eq!(parse_turboquant_type("tbq4_0"),  Some(TurboquantType::TQ4_0));
    assert_eq!(parse_turboquant_type("tbqp3_0"), Some(TurboquantType::TQP3_0));
    assert_eq!(parse_turboquant_type("tbqp4_0"), Some(TurboquantType::TQP4_0));
    // head_dim=128 (_1) — Qwen3, Llama, Mistral
    assert_eq!(parse_turboquant_type("tbq3_1"),  Some(TurboquantType::TQ3_1));
    assert_eq!(parse_turboquant_type("tbq4_1"),  Some(TurboquantType::TQ4_1));
    assert_eq!(parse_turboquant_type("tbqp3_1"), Some(TurboquantType::TQP3_1));
    assert_eq!(parse_turboquant_type("tbqp4_1"), Some(TurboquantType::TQP4_1));
    // head_dim=64 (_2)
    assert_eq!(parse_turboquant_type("tbq3_2"),  Some(TurboquantType::TQ3_2));
    assert_eq!(parse_turboquant_type("tbq4_2"),  Some(TurboquantType::TQ4_2));
    // Legacy aliases
    assert_eq!(parse_turboquant_type("tq3_0"),  Some(TurboquantType::TQ3_0));
    assert_eq!(parse_turboquant_type("tq4_0"),  Some(TurboquantType::TQ4_0));
    assert_eq!(parse_turboquant_type("turbo3"), Some(TurboquantType::TQ3_0));
    assert_eq!(parse_turboquant_type("turbo4"), Some(TurboquantType::TQ4_0));
    // AmesianX recommended config bare aliases (default → _1, head_dim=128)
    assert_eq!(parse_turboquant_type("tbq3"),   Some(TurboquantType::TQ3_1));
    assert_eq!(parse_turboquant_type("tbq4"),   Some(TurboquantType::TQ4_1));
    assert_eq!(parse_turboquant_type("tbqp3"),  Some(TurboquantType::TQP3_1));
    assert_eq!(parse_turboquant_type("tbqp4"),  Some(TurboquantType::TQP4_1));
    assert_eq!(parse_turboquant_type("f16"),  None);
    assert_eq!(parse_turboquant_type("q8_0"), None);
}

#[test]
fn is_tq_type() {
    assert!(is_turboquant_type("tbq3_0"));
    assert!(is_turboquant_type("tbq4_0"));
    assert!(is_turboquant_type("tbqp3_0"));
    assert!(is_turboquant_type("tbqp4_0"));
    assert!(is_turboquant_type("tbq3_1"));
    assert!(is_turboquant_type("tbq4_1"));
    assert!(is_turboquant_type("tbqp3_1"));
    assert!(is_turboquant_type("tbqp4_1"));
    assert!(is_turboquant_type("tq3_0"));   // legacy alias
    assert!(is_turboquant_type("tq4_0"));   // legacy alias
    assert!(is_turboquant_type("tbq3"));    // AmesianX recommended bare alias → _1
    assert!(is_turboquant_type("tbq4"));
    assert!(is_turboquant_type("tbqp3"));
    assert!(is_turboquant_type("tbqp4"));
    assert!(!is_turboquant_type("f16"));
    assert!(!is_turboquant_type("q4_0"));
}

#[test]
fn compression_ratios() {
    assert!((TurboquantType::TQ3_0.compression_ratio() - 5.33).abs() < 0.1);
    assert!((TurboquantType::TQ4_0.compression_ratio() - 4.0).abs() < 0.1);
    assert!((TurboquantType::TQ3_1.compression_ratio() - 5.33).abs() < 0.1);
    assert!((TurboquantType::TQ4_1.compression_ratio() - 4.0).abs() < 0.1);
}

#[test]
fn codebook_3bit_is_sorted() {
    for i in 1..CODEBOOK_3BIT.len() {
        assert!(
            CODEBOOK_3BIT[i] > CODEBOOK_3BIT[i - 1],
            "3-bit codebook not sorted at index {i}"
        );
    }
}

#[test]
fn codebook_4bit_is_sorted() {
    for i in 1..CODEBOOK_4BIT.len() {
        assert!(
            CODEBOOK_4BIT[i] > CODEBOOK_4BIT[i - 1],
            "4-bit codebook not sorted at index {i}"
        );
    }
}

#[test]
fn boundaries_between_centroids() {
    let b = boundaries_3bit();
    let c = &CODEBOOK_3BIT;
    for i in 0..b.len() {
        assert!(b[i] > c[i], "boundary {i} not between centroids");
        assert!(b[i] < c[i + 1], "boundary {i} not between centroids");
    }
}

#[test]
fn boundaries_4bit_between_centroids() {
    let b = boundaries_4bit();
    let c = &CODEBOOK_4BIT;
    for i in 0..b.len() {
        assert!(b[i] > c[i], "boundary {i} not between centroids");
        assert!(b[i] < c[i + 1], "boundary {i} not between centroids");
    }
}

#[test]
fn resolve_without_native_backend() {
    let result = resolve_turboquant_cache_type("tq3_0", false);
    assert!(result.is_some());
    match result.unwrap() {
        #[cfg(not(feature = "turboquant_native"))]
        ResolvedCacheType::Fallback { requested, fallback, .. } => {
            assert_eq!(requested, TurboquantType::TQ3_0);
            assert_eq!(fallback, "f16");
        }
        #[cfg(feature = "turboquant_native")]
        ResolvedCacheType::Native(tq) => {
            assert_eq!(tq, TurboquantType::TQ3_0);
        }
        other => panic!("Unexpected resolution: {other:?}"),
    }
}

#[test]
fn resolve_strict_without_native_backend() {
    let result = resolve_turboquant_cache_type("tq4_0", true);
    assert!(result.is_some());
    match result.unwrap() {
        #[cfg(not(feature = "turboquant_native"))]
        ResolvedCacheType::Unsupported { requested, reason } => {
            assert_eq!(requested, TurboquantType::TQ4_0);
            assert!(reason.contains("not supported"));
        }
        #[cfg(feature = "turboquant_native")]
        ResolvedCacheType::Native(tq) => {
            // Strict mode is irrelevant when backend supports TQ — it resolves natively.
            assert_eq!(tq, TurboquantType::TQ4_0);
        }
        other => panic!("Unexpected resolution: {other:?}"),
    }
}

#[test]
fn resolve_ignores_standard_types() {
    // Standard types should return None (handled by normal path).
    assert!(resolve_turboquant_cache_type("f16", false).is_none());
    assert!(resolve_turboquant_cache_type("q8_0", false).is_none());
    assert!(resolve_turboquant_cache_type("q4_0", true).is_none());
}

// ── Accuracy / SQNR tests ─────────────────────────────────────────────────────
//
// These tests verify the Rust-side Lloyd-Max codebook achieves expected
// quantization accuracy.  They run fully in-process (no GPU required) and
// catch codebook regressions of the kind reported as KV cache degradation
// (AmesianX/TurboQuant v1.5.3 upgrade, April 2026).
//
// Test distribution: uniform grid over [-3.0, 3.0] (601 points, step 0.01).
// Signal power ≈ 3.0 (mean x² for uniform [-3,3]).
// Expected SQNR: ≥11 dB for 3-bit, ≥17 dB for 4-bit (conservative floor).

/// Compute mean-squared error and SQNR (dB) for a quantizer over a test grid.
fn sqnr_db(quantize: impl Fn(f32) -> f32) -> (f32, f32) {
    const N: usize = 601;
    let mut signal_power = 0.0f32;
    let mut noise_power = 0.0f32;
    for i in 0..N {
        let x = -3.0 + (i as f32) * (6.0 / (N as f32 - 1.0));
        let q = quantize(x);
        signal_power += x * x;
        noise_power += (x - q) * (x - q);
    }
    signal_power /= N as f32;
    noise_power /= N as f32;
    let mse = noise_power;
    let sqnr = 10.0 * (signal_power / noise_power).log10();
    (mse, sqnr)
}

#[test]
fn sqnr_3bit_meets_spec() {
    let (mse, sqnr) = sqnr_db(quantize_scalar_3bit);
    assert!(
        sqnr >= 11.0,
        "3-bit SQNR {sqnr:.2} dB below 11 dB floor (MSE={mse:.4}). \
         Codebook regression? Check AmesianX/TurboQuant version."
    );
}

#[test]
fn sqnr_4bit_meets_spec() {
    let (mse, sqnr) = sqnr_db(quantize_scalar_4bit);
    assert!(
        sqnr >= 17.0,
        "4-bit SQNR {sqnr:.2} dB below 17 dB floor (MSE={mse:.4}). \
         Codebook regression? Check AmesianX/TurboQuant version."
    );
}

#[test]
fn mse_3bit_bounded() {
    let (mse, _) = sqnr_db(quantize_scalar_3bit);
    // 3-bit Lloyd-Max on uniform [-3,3]: expected ~0.144, must stay under 0.20.
    // (Actual 0.1438 on this grid; bound gives headroom for f32 precision.)
    assert!(
        mse < 0.20,
        "3-bit MSE {mse:.4} exceeds 0.20 — possible codebook corruption."
    );
}

#[test]
fn mse_4bit_bounded() {
    let (mse, _) = sqnr_db(quantize_scalar_4bit);
    // 4-bit Lloyd-Max on uniform [-3,3]: MSE must stay well under 0.05.
    assert!(
        mse < 0.05,
        "4-bit MSE {mse:.4} exceeds 0.05 — possible codebook corruption."
    );
}

#[test]
fn codebook_3bit_near_symmetry() {
    // For i in 0..3, centroids[i] ≈ -centroids[6-i] (roughly symmetric around 0).
    // The 8th entry (index 7) is the upper-tail clamp and has no mirror.
    for i in 0..3usize {
        let lo = CODEBOOK_3BIT[i];
        let hi = CODEBOOK_3BIT[6 - i];
        let diff = (lo + hi).abs();
        assert!(
            diff < 1e-3,
            "3-bit codebook not symmetric at index {i}: {lo} + {hi} = {diff}"
        );
    }
}

#[test]
fn codebook_4bit_near_symmetry() {
    // For i in 0..7, centroids[i] ≈ -centroids[14-i].
    // Index 15 is the upper-tail clamp.
    for i in 0..7usize {
        let lo = CODEBOOK_4BIT[i];
        let hi = CODEBOOK_4BIT[14 - i];
        let diff = (lo + hi).abs();
        assert!(
            diff < 1e-3,
            "4-bit codebook not symmetric at index {i}: {lo} + {hi} = {diff}"
        );
    }
}

#[test]
fn all_centroids_reachable_3bit() {
    // Every centroid must be the nearest reconstruction for at least one input.
    let mut seen = [false; 8];
    for i in 0..=1000 {
        let x = -3.0 + (i as f32) * 0.006;
        let q = quantize_scalar_3bit(x);
        if let Some(idx) = CODEBOOK_3BIT.iter().position(|&c| (c - q).abs() < 1e-6) {
            seen[idx] = true;
        }
    }
    for (i, &reachable) in seen.iter().enumerate() {
        assert!(reachable, "3-bit centroid index {i} ({}) is unreachable", CODEBOOK_3BIT[i]);
    }
}

#[test]
fn all_centroids_reachable_4bit() {
    let mut seen = [false; 16];
    for i in 0..=1000 {
        let x = -3.0 + (i as f32) * 0.006;
        let q = quantize_scalar_4bit(x);
        if let Some(idx) = CODEBOOK_4BIT.iter().position(|&c| (c - q).abs() < 1e-6) {
            seen[idx] = true;
        }
    }
    for (i, &reachable) in seen.iter().enumerate() {
        assert!(reachable, "4-bit centroid index {i} ({}) is unreachable", CODEBOOK_4BIT[i]);
    }
}

#[test]
fn quantize_is_idempotent_3bit() {
    // Quantizing an already-quantized value (a centroid) must return itself.
    for &c in &CODEBOOK_3BIT {
        let q = quantize_scalar_3bit(c);
        assert!(
            (q - c).abs() < 1e-6,
            "3-bit centroid {c} not idempotent: quantize({c}) = {q}"
        );
    }
}

#[test]
fn quantize_is_idempotent_4bit() {
    for &c in &CODEBOOK_4BIT {
        let q = quantize_scalar_4bit(c);
        assert!(
            (q - c).abs() < 1e-6,
            "4-bit centroid {c} not idempotent: quantize({c}) = {q}"
        );
    }
}

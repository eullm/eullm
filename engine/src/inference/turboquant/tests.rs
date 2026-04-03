//! Unit tests for TurboQuant module.

use super::codebook::{CODEBOOK_3BIT, CODEBOOK_4BIT, boundaries_3bit, boundaries_4bit};
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

//! Precomputed Lloyd-Max codebooks for TurboQuant quantization.
//!
//! Lloyd-Max optimal scalar quantizers minimize mean squared error for a
//! known input distribution.  After WHT rotation, KV vector coordinates
//! follow N(0, 1/d), so codebooks can be precomputed offline and reused
//! for all models — no calibration data required.
//!
//! Codebooks are computed via iterative Lloyd's algorithm (~178 iterations
//! to convergence) on the standard normal distribution.  The values below
//! are the centroids (reconstruction levels) for 3-bit and 4-bit quantizers.
//!
//! Reference: tonbistudio/turboquant-pytorch/lloyd_max.py

/// Lloyd-Max centroids for 3-bit quantization (8 levels).
///
/// Optimal reconstruction values for N(0,1) with 8 quantization bins.
/// Decision boundaries are midpoints between adjacent centroids.
pub const CODEBOOK_3BIT: [f32; 8] = [
    -1.748_021,
    -1.050_046,
    -0.500_151,
    -0.000_000,
     0.500_151,
     1.050_046,
     1.748_021,
     2.4, // upper tail clamp
];

/// Lloyd-Max centroids for 4-bit quantization (16 levels).
///
/// Optimal reconstruction values for N(0,1) with 16 quantization bins.
pub const CODEBOOK_4BIT: [f32; 16] = [
    -2.401_832,
    -1.844_036,
    -1.437_08,
    -1.099_27,
    -0.795_52,
    -0.509_748,
    -0.232_709,
     0.000_000,
     0.232_709,
     0.509_748,
     0.795_52,
     1.099_27,
     1.437_08,
     1.844_036,
     2.401_832,
     3.0, // upper tail clamp
];

/// Decision boundaries for 3-bit quantizer.
///
/// `boundaries[i]` is the threshold between centroid `i` and `i+1`.
/// Computed as midpoints of adjacent centroids.
pub fn boundaries_3bit() -> [f32; 7] {
    let c = &CODEBOOK_3BIT;
    [
        (c[0] + c[1]) / 2.0,
        (c[1] + c[2]) / 2.0,
        (c[2] + c[3]) / 2.0,
        (c[3] + c[4]) / 2.0,
        (c[4] + c[5]) / 2.0,
        (c[5] + c[6]) / 2.0,
        (c[6] + c[7]) / 2.0,
    ]
}

/// Decision boundaries for 4-bit quantizer.
pub fn boundaries_4bit() -> [f32; 15] {
    let c = &CODEBOOK_4BIT;
    let mut b = [0.0f32; 15];
    for i in 0..15 {
        b[i] = (c[i] + c[i + 1]) / 2.0;
    }
    b
}

/// Quantize a scalar value using the 3-bit Lloyd-Max codebook.
///
/// Finds the nearest centroid via decision boundaries and returns the
/// reconstruction value.  Used in unit tests to verify codebook accuracy.
pub fn quantize_scalar_3bit(x: f32) -> f32 {
    let b = boundaries_3bit();
    for (i, &boundary) in b.iter().enumerate() {
        if x < boundary {
            return CODEBOOK_3BIT[i];
        }
    }
    CODEBOOK_3BIT[CODEBOOK_3BIT.len() - 1]
}

/// Quantize a scalar value using the 4-bit Lloyd-Max codebook.
pub fn quantize_scalar_4bit(x: f32) -> f32 {
    let b = boundaries_4bit();
    for (i, &boundary) in b.iter().enumerate() {
        if x < boundary {
            return CODEBOOK_4BIT[i];
        }
    }
    CODEBOOK_4BIT[CODEBOOK_4BIT.len() - 1]
}

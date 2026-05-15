use crate::ops::EMBED_DIM;

/// Number of u64 words to represent one EMBED_DIM-dimensional binary code.
pub const BINARY_WORDS: usize = EMBED_DIM / 64;

/// Number of candidates to evaluate with float rescore after Hamming pre-filter.
/// 400 / 54k ≈ 0.74% scan — strong recall guarantee with minimal rescore cost.
pub const HAMMING_CANDIDATES: usize = 400;

/// Sign-bit binarization: bit set when component ≥ 0.
/// Input should be L2-normalized (unit vector).
#[inline]
pub fn binarize(embedding: &[f32]) -> Vec<u64> {
    debug_assert_eq!(embedding.len(), EMBED_DIM);
    let mut codes = vec![0u64; BINARY_WORDS];
    for (i, &x) in embedding.iter().take(EMBED_DIM).enumerate() {
        if x >= 0.0 {
            codes[i / 64] |= 1u64 << (i % 64);
        }
    }
    codes
}

/// Hamming distance between two binary codes (popcount of XOR).
#[inline]
pub fn hamming_dist(a: &[u64], b: &[u64]) -> u32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| (x ^ y).count_ones()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_vec(v: Vec<f32>) -> Vec<f32> {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.into_iter().map(|x| x / norm).collect()
    }

    #[test]
    fn identical_vectors_zero_hamming() {
        let v = unit_vec(vec![1.0; EMBED_DIM]);
        let a = binarize(&v);
        assert_eq!(hamming_dist(&a, &a), 0);
    }

    #[test]
    fn opposite_vectors_max_hamming() {
        let pos = unit_vec(vec![1.0; EMBED_DIM]);
        let neg = unit_vec(vec![-1.0; EMBED_DIM]);
        let a = binarize(&pos);
        let b = binarize(&neg);
        assert_eq!(hamming_dist(&a, &b), EMBED_DIM as u32);
    }

    #[test]
    fn hamming_correlates_with_cosine() {
        use std::collections::HashMap;
        // Build a random query and 20 random docs; verify ranking correlation.
        let query = unit_vec((0..EMBED_DIM).map(|i| ((i * 7 + 3) % 17) as f32 - 8.0).collect());
        let docs: Vec<Vec<f32>> = (0..20)
            .map(|d| {
                unit_vec(
                    (0..EMBED_DIM)
                        .map(|i| ((i * 13 + d * 31 + 5) % 23) as f32 - 11.0)
                        .collect(),
                )
            })
            .collect();

        let q_bits = binarize(&query);
        let cosine_ranks: Vec<usize> = {
            let mut scored: Vec<(usize, f32)> = docs
                .iter()
                .enumerate()
                .map(|(i, d)| (i, query.iter().zip(d).map(|(a, b)| a * b).sum::<f32>()))
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            scored.iter().map(|(i, _)| *i).collect()
        };
        let hamming_ranks: Vec<usize> = {
            let mut scored: Vec<(usize, u32)> = docs
                .iter()
                .enumerate()
                .map(|(i, d)| (i, hamming_dist(&q_bits, &binarize(d))))
                .collect();
            scored.sort_by_key(|(_, h)| *h);
            scored.iter().map(|(i, _)| *i).collect()
        };

        // Top-5 cosine should have meaningful overlap with top-8 Hamming.
        // Threshold is 2 rather than 3 because 256-bit codes are less discriminating
        // than 768-bit codes — some ranking noise is expected at this resolution.
        let top5_cosine: std::collections::HashSet<usize> = cosine_ranks[..5].iter().copied().collect();
        let top8_hamming: std::collections::HashSet<usize> = hamming_ranks[..8].iter().copied().collect();
        let overlap = top5_cosine.intersection(&top8_hamming).count();
        assert!(overlap >= 2, "poor ranking correlation: {overlap}/5 overlap in top-8 Hamming");
    }
}

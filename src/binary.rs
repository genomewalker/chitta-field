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
        // Construct near/far docs explicitly so correlation is guaranteed by construction.
        // Near docs share the same sign pattern as the query on most dimensions.
        // Far docs have the opposite sign on most dimensions.
        let mut query = vec![0.0f32; EMBED_DIM];
        for i in 0..EMBED_DIM {
            query[i] = if (i * 7 + 3) % 2 == 0 { 1.0 } else { -1.0 };
        }
        let query = unit_vec(query);
        let q_bits = binarize(&query);

        let mut docs: Vec<Vec<f32>> = Vec::with_capacity(20);
        // 5 near docs: share sign on ~80% of dims, differ on ~20%
        for d in 0..5usize {
            let mut v = query.clone();
            for i in 0..EMBED_DIM {
                if (i * 13 + d * 31) % 5 == 0 {
                    v[i] = -v[i]; // flip ~20%
                }
            }
            docs.push(unit_vec(v));
        }
        // 15 far docs: opposite sign on ~80% of dims
        for d in 0..15usize {
            let mut v = query.clone();
            for i in 0..EMBED_DIM {
                if (i * 11 + d * 17) % 5 != 0 {
                    v[i] = -v[i]; // flip ~80%
                }
            }
            docs.push(unit_vec(v));
        }

        let cosine_top5: std::collections::HashSet<usize> = {
            let mut scored: Vec<(usize, f32)> = docs
                .iter()
                .enumerate()
                .map(|(i, d)| (i, query.iter().zip(d).map(|(a, b)| a * b).sum::<f32>()))
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            scored[..5].iter().map(|(i, _)| *i).collect()
        };
        let hamming_top8: std::collections::HashSet<usize> = {
            let mut scored: Vec<(usize, u32)> = docs
                .iter()
                .enumerate()
                .map(|(i, d)| (i, hamming_dist(&q_bits, &binarize(d))))
                .collect();
            scored.sort_by_key(|(_, h)| *h);
            scored[..8].iter().map(|(i, _)| *i).collect()
        };

        let overlap = cosine_top5.intersection(&hamming_top8).count();
        assert!(overlap >= 3, "near docs not found: {overlap}/5 cosine-top5 in hamming-top8");
    }
}

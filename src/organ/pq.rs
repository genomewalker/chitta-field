use serde::{Deserialize, Serialize};

pub const N_SUBVECTORS: usize = 32;
pub const DIM_PER_SUB: usize = 24; // 768 / 32 = 24
pub const N_CENTROIDS: usize = 256;
pub const PQ_BYTES: usize = N_SUBVECTORS; // 32 bytes per quantized residual

/// Product quantizer for 768-dim residual vectors.
/// Trained once on accumulated residuals; then used for fast approximate storage.
#[derive(Serialize, Deserialize, Clone)]
pub struct ProductQuantizer {
    /// [N_SUBVECTORS][N_CENTROIDS][DIM_PER_SUB]
    pub codebooks: Vec<Vec<Vec<f32>>>,
}

impl ProductQuantizer {
    /// Train a ProductQuantizer from a slice of 768-dim residual vectors.
    /// Requires at least 256 residuals. Runs Lloyd's k-means for n_iter iterations.
    pub fn train(residuals: &[Vec<f32>], n_iter: usize) -> Result<Self, String> {
        if residuals.len() < N_CENTROIDS {
            return Err(format!(
                "need at least {} residuals to train PQ, got {}",
                N_CENTROIDS,
                residuals.len()
            ));
        }

        let mut codebooks = Vec::with_capacity(N_SUBVECTORS);

        for sub in 0..N_SUBVECTORS {
            let start = sub * DIM_PER_SUB;
            let end = start + DIM_PER_SUB;

            // Extract subvectors for this subspace
            let subvecs: Vec<Vec<f32>> = residuals
                .iter()
                .map(|r| r[start..end].to_vec())
                .collect();

            let centroids = kmeans(&subvecs, N_CENTROIDS, n_iter);
            codebooks.push(centroids);
        }

        Ok(ProductQuantizer { codebooks })
    }

    /// Quantize a 768-dim residual vector into PQ_BYTES (32 bytes).
    /// Each byte is the centroid index for the corresponding subvector.
    pub fn quantize(&self, residual: &[f32]) -> [u8; PQ_BYTES] {
        let mut codes = [0u8; PQ_BYTES];
        for sub in 0..N_SUBVECTORS {
            let start = sub * DIM_PER_SUB;
            let end = start + DIM_PER_SUB;
            let subvec = &residual[start..end];
            let centroids = &self.codebooks[sub];

            let best = centroids
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let da = l2sq(subvec, a);
                    let db = l2sq(subvec, b);
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);

            codes[sub] = best as u8;
        }
        codes
    }

    /// Reconstruct an approximate 768-dim residual from PQ codes.
    pub fn reconstruct(&self, codes: &[u8; PQ_BYTES]) -> Vec<f32> {
        let mut out = vec![0.0f32; N_SUBVECTORS * DIM_PER_SUB];
        for sub in 0..N_SUBVECTORS {
            let start = sub * DIM_PER_SUB;
            let centroid_idx = codes[sub] as usize;
            let centroid = &self.codebooks[sub][centroid_idx];
            out[start..start + DIM_PER_SUB].copy_from_slice(centroid);
        }
        out
    }
}

/// Lloyd's k-means over a slice of equal-length vectors.
/// Initializes by evenly striding through data, runs n_iter iterations.
fn kmeans(data: &[Vec<f32>], k: usize, n_iter: usize) -> Vec<Vec<f32>> {
    let n = data.len();
    let dim = data[0].len();
    let k = k.min(n);

    // Initialize centroids by striding through data
    let stride = n / k;
    let mut centroids: Vec<Vec<f32>> = (0..k)
        .map(|i| data[i * stride].clone())
        .collect();

    for _ in 0..n_iter {
        // Assignment step: assign each point to nearest centroid
        let assignments: Vec<usize> = data
            .iter()
            .map(|point| {
                centroids
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        let da = l2sq(point, a);
                        let db = l2sq(point, b);
                        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect();

        // Update step: recompute centroids as mean of assigned points
        let mut sums: Vec<Vec<f32>> = vec![vec![0.0f32; dim]; k];
        let mut counts: Vec<usize> = vec![0usize; k];

        for (point, &cluster) in data.iter().zip(assignments.iter()) {
            for (s, &v) in sums[cluster].iter_mut().zip(point.iter()) {
                *s += v;
            }
            counts[cluster] += 1;
        }

        for (i, centroid) in centroids.iter_mut().enumerate() {
            if counts[i] > 0 {
                let inv = 1.0 / counts[i] as f32;
                for (c, s) in centroid.iter_mut().zip(sums[i].iter()) {
                    *c = s * inv;
                }
            }
            // If a centroid has no assigned points, keep it unchanged
        }
    }

    centroids
}

/// Sum of squared differences between two equal-length slices.
fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

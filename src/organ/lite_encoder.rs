use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::organ::cortex::{SparseCode, N_LEFT, N_RIGHT, K_ACTIVE};

pub const VOCAB_SIZE: usize = 16_384;
pub const MIN_WORD_LEN: usize = 3;
pub const MIN_WORD_FREQ: u32 = 2;

/// Bag-of-words → product-key latent space.
/// Learns word→(left_centroid, right_centroid) associations from
/// existing (content, sparse_code) pairs.
#[derive(Debug, Serialize, Deserialize)]
pub struct LiteEncoder {
    /// word → vocab index (top VOCAB_SIZE by frequency)
    pub vocab: HashMap<String, usize>,
    /// word_left_weights[vocab_id][left_centroid_idx] = association weight
    pub word_left: Vec<Vec<f32>>,   // VOCAB_SIZE × N_LEFT
    /// word_right_weights[vocab_id][right_centroid_idx] = association weight
    pub word_right: Vec<Vec<f32>>, // VOCAB_SIZE × N_RIGHT
    /// idf[vocab_id] = log(N / df(w))
    pub idf: Vec<f32>,
    /// Number of training examples seen
    pub training_examples: usize,
}

impl LiteEncoder {
    /// Train from a slice of (content, sparse_code) pairs.
    pub fn train(examples: &[(String, SparseCode)]) -> Self {
        let n = examples.len();

        // Step 1: Count word frequencies across all content strings
        let mut word_freq: HashMap<String, u32> = HashMap::new();
        for (content, _) in examples {
            for word in Self::tokenize(content) {
                *word_freq.entry(word).or_insert(0) += 1;
            }
        }

        // Build vocab: top VOCAB_SIZE words by frequency, filtered by MIN_WORD_FREQ
        let mut freq_pairs: Vec<(String, u32)> = word_freq
            .into_iter()
            .filter(|(_, freq)| *freq >= MIN_WORD_FREQ)
            .collect();
        freq_pairs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        freq_pairs.truncate(VOCAB_SIZE);

        let vocab: HashMap<String, usize> = freq_pairs
            .iter()
            .enumerate()
            .map(|(i, (w, _))| (w.clone(), i))
            .collect();

        // Step 2: Compute IDF — for each vocab word, count docs containing it
        let mut df: Vec<u64> = vec![0u64; vocab.len()];
        for (content, _) in examples {
            let words: std::collections::HashSet<String> = Self::tokenize(content).into_iter().collect();
            for word in &words {
                if let Some(&vid) = vocab.get(word) {
                    df[vid] += 1;
                }
            }
        }

        let idf: Vec<f32> = df.iter()
            .map(|&d| ((n as f32 + 1.0) / (d as f32 + 1.0)).ln() + 1.0)
            .collect();

        // Step 3: Initialize weight matrices
        let vocab_size = vocab.len();
        let mut word_left: Vec<Vec<f32>> = vec![vec![0.0f32; N_LEFT]; vocab_size];
        let mut word_right: Vec<Vec<f32>> = vec![vec![0.0f32; N_RIGHT]; vocab_size];

        // Step 4: Accumulate word→centroid associations
        for (content, sparse_code) in examples {
            if sparse_code.is_empty() {
                continue;
            }

            let words = Self::tokenize(content);
            if words.is_empty() {
                continue;
            }

            // TF: count occurrences per word
            let total_words = words.len() as f32;
            let mut tf_counts: HashMap<String, u32> = HashMap::new();
            for word in &words {
                *tf_counts.entry(word.clone()).or_insert(0) += 1;
            }

            // Extract left/right centroid activations from sparse_code
            let mut left_act = [0.0f32; N_LEFT];
            let mut right_act = [0.0f32; N_RIGHT];
            for (&atom_idx, &weight) in sparse_code.feature_ids.iter().zip(sparse_code.activations.iter()) {
                let left_idx = atom_idx as usize / N_RIGHT;
                let right_idx = atom_idx as usize % N_RIGHT;
                left_act[left_idx] += weight;
                right_act[right_idx] += weight;
            }

            // Normalize left_act and right_act to sum=1
            let left_sum: f32 = left_act.iter().sum();
            let right_sum: f32 = right_act.iter().sum();
            if left_sum > 1e-9 {
                for v in left_act.iter_mut() { *v /= left_sum; }
            }
            if right_sum > 1e-9 {
                for v in right_act.iter_mut() { *v /= right_sum; }
            }

            // Accumulate word weights
            for (word, count) in &tf_counts {
                let vid = match vocab.get(word) {
                    Some(&v) => v,
                    None => continue,
                };
                let tf = *count as f32 / total_words;
                let tfidf = tf * idf[vid];

                let wl = &mut word_left[vid];
                for (li, &la) in left_act.iter().enumerate() {
                    wl[li] += tfidf * la;
                }

                let wr = &mut word_right[vid];
                for (ri, &ra) in right_act.iter().enumerate() {
                    wr[ri] += tfidf * ra;
                }
            }
        }

        // Step 5: L2-normalize each row
        for row in word_left.iter_mut() {
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-9 {
                for x in row.iter_mut() { *x /= norm; }
            }
        }
        for row in word_right.iter_mut() {
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-9 {
                for x in row.iter_mut() { *x /= norm; }
            }
        }

        Self {
            vocab,
            word_left,
            word_right,
            idf,
            training_examples: n,
        }
    }

    /// Encode text into a SparseCode (same format as SPAF encoder output).
    pub fn encode(&self, text: &str) -> Option<SparseCode> {
        let words = Self::tokenize(text);
        if words.is_empty() {
            return None;
        }

        // TF computation
        let total_words = words.len() as f32;
        let mut tf_counts: HashMap<String, u32> = HashMap::new();
        for word in &words {
            *tf_counts.entry(word.clone()).or_insert(0) += 1;
        }

        // Accumulate left_score and right_score
        let mut left_score = [0.0f32; N_LEFT];
        let mut right_score = [0.0f32; N_RIGHT];

        for (word, count) in &tf_counts {
            let vid = match self.vocab.get(word) {
                Some(&v) => v,
                None => continue,
            };
            let tf = *count as f32 / total_words;
            let tfidf = tf * self.idf[vid];

            let wl = &self.word_left[vid];
            for (li, &wv) in wl.iter().enumerate() {
                left_score[li] += tfidf * wv;
            }

            let wr = &self.word_right[vid];
            for (ri, &wv) in wr.iter().enumerate() {
                right_score[ri] += tfidf * wv;
            }
        }

        // Top-8 left indices by left_score
        const TOP_HALF: usize = 8;
        let mut left_indexed: Vec<(f32, usize)> = left_score.iter().enumerate()
            .map(|(i, &s)| (s, i))
            .collect();
        left_indexed.select_nth_unstable_by(TOP_HALF, |a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let top_left = &left_indexed[..TOP_HALF];

        // Top-8 right indices by right_score
        let mut right_indexed: Vec<(f32, usize)> = right_score.iter().enumerate()
            .map(|(i, &s)| (s, i))
            .collect();
        right_indexed.select_nth_unstable_by(TOP_HALF, |a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let top_right = &right_indexed[..TOP_HALF];

        // Build TOP_HALF² = 64 (left, right) pairs by score product
        let mut candidates: Vec<(f32, u32)> = Vec::with_capacity(TOP_HALF * TOP_HALF);
        for &(ls, li) in top_left {
            for &(rs, ri) in top_right {
                let atom_idx = (li * N_RIGHT + ri) as u32;
                candidates.push((ls * rs, atom_idx));
            }
        }

        let k = K_ACTIVE.min(candidates.len());
        if k == 0 {
            return None;
        }

        candidates.select_nth_unstable_by(k, |a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(k);

        // Filter non-positive scores
        candidates.retain(|(s, _)| *s > 0.0);
        if candidates.is_empty() {
            return None;
        }

        // Sort by atom_idx ascending
        candidates.sort_by(|a, b| a.1.cmp(&b.1));

        // Normalize weights to sum=1
        let weight_sum: f32 = candidates.iter().map(|(s, _)| *s).sum();
        if weight_sum < 1e-9 {
            return None;
        }

        let feature_ids: Vec<u32> = candidates.iter().map(|(_, id)| *id).collect();
        let activations: Vec<f32> = candidates.iter().map(|(s, _)| s / weight_sum).collect();

        Some(SparseCode { feature_ids, activations })
    }

    /// Tokenize text: lowercase, split on non-alpha, filter short words
    pub fn tokenize(text: &str) -> Vec<String> {
        let lower = text.to_lowercase();
        lower
            .split(|c: char| !c.is_alphabetic())
            .filter(|w| w.len() >= MIN_WORD_LEN)
            .map(|w| w.to_string())
            .collect()
    }

    /// Save to bytes (bincode)
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }

    /// Load from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        let enc = bincode::deserialize(data)?;
        Ok(enc)
    }
}

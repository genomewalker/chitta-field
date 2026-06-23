/// Domain Reliability Learner — Proof of Expertise (PoE) per realm/domain.
///
/// Tracks how often memories from each realm/domain have been corrected.
/// A correction (a memory stored with kind="correction" pointing at a source realm)
/// decrements reliability; a successful recall that is NOT subsequently corrected
/// (reinforcement signal) increments reliability.
///
/// Reliability score ∈ [0.5, 1.0] multiplies recall scores at query time so
/// memories from unreliable domains are naturally ranked lower.
///
/// ## Update protocol
/// - `record_correction(realm)` — called when a correction is stored that
///   targets memories from `realm`. Signals the domain produced an error.
/// - `record_success(realm)` — called when a memory from `realm` is recalled
///   and positively reinforced (e.g. user accepted its suggestion).
///
/// ## Retrieval
/// - `reliability(realm)` → f32 in [0.5, 1.0]
/// - Multiply the recall score of a candidate by its domain's reliability
///   before ranking.
use super::bandit::BetaPrior;
use std::collections::HashMap;

/// Minimum reliability floor — even a maximally corrected domain still
/// contributes at 50% weight rather than being silenced entirely.
const RELIABILITY_FLOOR: f64 = 0.5;

/// Reliability ceiling — perfect track record = 1.0 weight (no boost).
const RELIABILITY_CEIL: f64 = 1.0;

/// Default reliability for unknown domains (optimistic prior).
const DEFAULT_RELIABILITY: f64 = 0.85;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DomainReliability {
    /// Per-realm Beta prior: alpha = evidence of accuracy, beta = evidence of error.
    domains: HashMap<String, BetaPrior>,
}

impl DomainReliability {
    pub fn new() -> Self {
        Self {
            domains: HashMap::new(),
        }
    }

    /// Record a correction targeting this realm — penalises reliability.
    pub fn record_correction(&mut self, realm: &str) {
        self.domains
            .entry(realm.to_string())
            .or_insert_with(BetaPrior::new)
            .update(0.0); // failure signal
    }

    /// Record a positive reinforcement for this realm — boosts reliability.
    pub fn record_success(&mut self, realm: &str) {
        self.domains
            .entry(realm.to_string())
            .or_insert_with(BetaPrior::new)
            .update(1.0); // success signal
    }

    /// Record a partial positive reinforcement for this realm. Recurrence
    /// (same content observed again) is real but weak evidence of accuracy,
    /// so it counts as a fractional success rather than a full trial.
    pub fn record_partial_success(&mut self, realm: &str, weight: f64) {
        self.domains
            .entry(realm.to_string())
            .or_insert_with(BetaPrior::new)
            .update(weight.clamp(0.0, 1.0) as f32);
    }

    /// Reliability score for `realm` in [FLOOR, CEIL].
    /// Unknown realms return DEFAULT_RELIABILITY.
    pub fn reliability(&self, realm: &str) -> f32 {
        let raw = self
            .domains
            .get(realm)
            .map(|p| p.mean())
            .unwrap_or(DEFAULT_RELIABILITY);
        // Map Beta mean [0,1] → [FLOOR, CEIL]
        let scaled = RELIABILITY_FLOOR + raw * (RELIABILITY_CEIL - RELIABILITY_FLOOR);
        scaled.clamp(RELIABILITY_FLOOR, RELIABILITY_CEIL) as f32
    }

    /// All tracked realms and their current reliability scores, sorted descending.
    pub fn scores(&self) -> Vec<(String, f32)> {
        let mut v: Vec<(String, f32)> = self
            .domains
            .iter()
            .map(|(realm, _)| (realm.clone(), self.reliability(realm)))
            .collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        v
    }

    /// Thompson-sample a per-realm recall slot cap in [1, k].
    ///
    /// Draws from the realm's Beta posterior (exploration noise built in) and
    /// maps the sample through the [FLOOR, CEIL] reliability window onto [1, k].
    /// A high-reliability realm samples near `k` (more recall slots); an
    /// unreliable or unknown realm samples near 1 (anti-flooding floor).
    ///
    /// `seed` decorrelates the draw across queries and realms — pass a
    /// time-derived value mixed with the realm name from the call site.
    pub fn sample_arm(&self, realm: &str, k: usize, seed: u64) -> usize {
        let s = self
            .domains
            .get(realm)
            .map(|p| p.sample(seed))
            .unwrap_or(DEFAULT_RELIABILITY);
        let frac = ((s - RELIABILITY_FLOOR) / (RELIABILITY_CEIL - RELIABILITY_FLOOR)).clamp(0.0, 1.0);
        let cap = (frac * k as f64).round() as usize;
        cap.max(1)
    }

    /// Number of correction events recorded for `realm`.
    pub fn correction_count(&self, realm: &str) -> u32 {
        self.domains
            .get(realm)
            .map(|p| (p.beta - 1.0).max(0.0).round() as u32)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unknown_realm_returns_default() {
        let dr = DomainReliability::new();
        let r = dr.reliability("brahman");
        // Unknown realm uses BetaPrior::new() mean = 0.5, scaled to [FLOOR, CEIL]:
        // 0.5 + 0.5 * 0.5 = 0.75. Verify it's in a reasonable range.
        let expected = (RELIABILITY_FLOOR + DEFAULT_RELIABILITY * (RELIABILITY_CEIL - RELIABILITY_FLOOR)) as f32;
        assert!((r - expected).abs() < 0.01, "expected {expected}, got {r}");
    }

    #[test]
    fn test_corrections_lower_reliability() {
        let mut dr = DomainReliability::new();
        for _ in 0..10 {
            dr.record_correction("project:foo");
        }
        let r = dr.reliability("project:foo");
        assert!(r < DEFAULT_RELIABILITY as f32, "corrections should lower reliability, got {}", r);
    }

    #[test]
    fn test_successes_raise_reliability() {
        let mut dr = DomainReliability::new();
        // Start with a few corrections then lots of successes
        dr.record_correction("project:bar");
        for _ in 0..20 {
            dr.record_success("project:bar");
        }
        let r = dr.reliability("project:bar");
        assert!(r > 0.85, "successes should recover reliability, got {}", r);
    }

    #[test]
    fn test_floor_respected() {
        let mut dr = DomainReliability::new();
        for _ in 0..1000 {
            dr.record_correction("bad_domain");
        }
        let r = dr.reliability("bad_domain");
        assert!(r >= RELIABILITY_FLOOR as f32, "must not go below floor, got {}", r);
    }
}

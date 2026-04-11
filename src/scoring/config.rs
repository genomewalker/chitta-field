use serde::{Deserialize, Serialize};
use std::path::Path;

/// Runtime-configurable scoring parameters.
///
/// Loaded from `scoring.json` in the data directory. If absent, defaults
/// are used. Every constant that was previously hardcoded in the scoring
/// pipeline now lives here — tunable without recompilation.
///
/// The config file is a developmental scaffold: as the cognitive system
/// matures, parameters can shift (e.g. increasing surprise weight as the
/// memory store grows, or reducing frustration sensitivity as corrections
/// accumulate). This is the "neuroplasticity" layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ScoringConfig {
    // ── ACT-R base-level activation ─────────────────────────────────────
    /// Power-law decay exponent (d in B_i = ln(Σ t_j^(-d))). Default 0.5.
    pub actr_decay_exponent: f32,
    /// Sigmoid temperature for activation → [0,1] mapping.
    pub actr_sigmoid_tau: f32,
    /// Sigmoid midpoint (activation value that maps to 0.5).
    pub actr_sigmoid_threshold: f32,
    /// Floor of the ACT-R factor (minimum contribution even for zero activation).
    pub actr_floor: f32,
    /// Range above floor (factor = floor + range * activation).
    pub actr_range: f32,

    // ── Prediction error (FEP) ──────────────────────────────────────────
    /// Surprise boost weight: factor = 1 + weight * surprise.
    pub surprise_weight: f32,

    // ── Flashbulb / arousal ─────────────────────────────────────────────
    /// Arousal boost weight: factor = 1 + weight * arousal.
    pub arousal_weight: f32,

    // ── Mood-congruent recall ───────────────────────────────────────────
    /// Maximum mood-congruence boost.
    pub mood_weight: f32,
    /// Minimum query arousal to activate mood matching.
    pub mood_min_arousal: f32,

    // ── Frustration-escalation ──────────────────────────────────────────
    /// Query valence threshold below which frustration activates.
    pub frustration_valence_threshold: f32,
    /// Query arousal threshold above which frustration activates.
    pub frustration_arousal_threshold: f32,
    /// Frustration boost weight for corrections.
    pub frustration_correction_weight: f32,
    /// Frustration boost weight for preferences.
    pub frustration_preference_weight: f32,

    // ── Status multipliers ──────────────────────────────────────────────
    pub status_verified: f32,
    pub status_active: f32,
    pub status_observed: f32,
    pub status_proposed: f32,

    // ── Kind multipliers ────────────────────────────────────────────────
    pub kind_correction: f32,
    pub kind_wisdom: f32,
    pub kind_insight: f32,
    pub kind_episode: f32,
    pub kind_default: f32,

    // ── Epistemic multipliers ───────────────────────────────────────────
    pub epistemic_user_stated: f32,
    pub epistemic_tool_derived: f32,
    pub epistemic_model_inferred: f32,
    pub epistemic_autonomous: f32,

    // ── Strength decay ──────────────────────────────────────────────────
    /// Floor of strength factor: factor = floor + range * effective_strength.
    pub strength_floor: f32,
    /// Range of strength factor above floor.
    pub strength_range: f32,
    /// Default per-day exponential decay rate for new memories.
    pub default_decay_rate: f32,

    // ── Deduplication ───────────────────────────────────────────────────
    /// Cosine similarity threshold for semantic dedup at ingestion.
    pub dedup_cosine_threshold: f32,
    /// Cosine similarity upper bound (above = exact duplicate, handled by chunk hash).
    pub dedup_cosine_upper: f32,

    // ── Association spreading ───────────────────────────────────────────
    /// Per-hop decay factor for spreading activation.
    pub assoc_hop_decay: f32,

    // ── Interference density (Price of Meaning) ────────────────────────
    /// Penalty weight for local competitor crowding.
    /// Factor = 1 / (1 + interference_penalty * competitive_weight).
    pub interference_penalty: f32,

    // ── Spacing boost (Geometry of Forgetting) ─────────────────────────
    /// Floor of spacing factor (minimum contribution).
    pub spacing_floor: f32,
    /// Range above floor: factor = floor + range * spacing_quality.
    pub spacing_range: f32,

    // ── Lure detection (post-pipeline) ─────────────────────────────────
    /// Lure risk threshold above which candidates may be suppressed.
    pub lure_risk_threshold: f32,
    /// Maximum number of lure-flagged candidates to suppress per query.
    pub lure_max_suppressed: usize,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            // ACT-R
            actr_decay_exponent: 0.5,
            actr_sigmoid_tau: 1.5,
            actr_sigmoid_threshold: -1.0,
            actr_floor: 0.3,
            actr_range: 0.7,

            // Surprise
            surprise_weight: 0.25,

            // Arousal
            arousal_weight: 0.15,

            // Mood congruence
            mood_weight: 0.2,
            mood_min_arousal: 0.1,

            // Frustration
            frustration_valence_threshold: -0.3,
            frustration_arousal_threshold: 0.4,
            frustration_correction_weight: 1.5,
            frustration_preference_weight: 1.0,

            // Status
            status_verified: 1.15,
            status_active: 1.0,
            status_observed: 0.85,
            status_proposed: 0.65,

            // Kind
            kind_correction: 1.3,
            kind_wisdom: 1.1,
            kind_insight: 1.05,
            kind_episode: 0.7,
            kind_default: 1.0,

            // Epistemic
            epistemic_user_stated: 1.0,
            epistemic_tool_derived: 0.95,
            epistemic_model_inferred: 0.85,
            epistemic_autonomous: 0.75,

            // Strength
            strength_floor: 0.5,
            strength_range: 0.5,
            default_decay_rate: 0.001,

            // Dedup
            dedup_cosine_threshold: 0.88,
            dedup_cosine_upper: 0.9999,

            // Association
            assoc_hop_decay: 0.55,

            // Interference (Price of Meaning)
            interference_penalty: 0.3,

            // Spacing (Geometry of Forgetting)
            spacing_floor: 0.85,
            spacing_range: 0.15,

            // Lure detection
            lure_risk_threshold: 0.7,
            lure_max_suppressed: 2,
        }
    }
}

impl ScoringConfig {
    /// Load config from `scoring.json` in the given directory.
    /// Falls back to defaults if the file doesn't exist or is malformed.
    pub fn load(data_dir: &Path) -> Self {
        let path = data_dir.join("scoring.json");
        if !path.exists() {
            log::info!("No scoring.json found, using default scoring config");
            return Self::default();
        }
        match std::fs::read_to_string(&path) {
            Ok(contents) => match serde_json::from_str(&contents) {
                Ok(config) => {
                    log::info!("Loaded scoring config from {}", path.display());
                    config
                }
                Err(e) => {
                    log::warn!("Malformed scoring.json: {}, using defaults", e);
                    Self::default()
                }
            },
            Err(e) => {
                log::warn!("Cannot read scoring.json: {}, using defaults", e);
                Self::default()
            }
        }
    }

    /// Save current config to `scoring.json` for inspection/editing.
    pub fn save(&self, data_dir: &Path) -> std::io::Result<()> {
        let path = data_dir.join("scoring.json");
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }
}

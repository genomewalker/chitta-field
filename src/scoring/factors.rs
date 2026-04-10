use super::{
    epistemic_multiplier, kind_multiplier, status_multiplier,
    RecallMode, ScoreDecomposition, ScoringContext, ScoringFactor,
};
use super::config::ScoringConfig;

// ── Relevance (semantic cosine or BM25) ─────────────────────────────────────

pub struct RelevanceFactor;

impl ScoringFactor for RelevanceFactor {
    fn name(&self) -> &'static str { "relevance" }

    fn compute(
        &self,
        ctx: &ScoringContext,
        _config: &ScoringConfig,
        decomp: &mut ScoreDecomposition,
    ) -> Option<f32> {
        let weight = match ctx.recall_mode {
            RecallMode::Semantic => ((ctx.relevance_score + 1.0) / 2.0).max(0.0),
            RecallMode::Keyword => ctx.relevance_score,
        };
        decomp.semantic_weight = weight;
        Some(weight)
    }
}

// ── ACT-R base-level activation (Anderson & Schooler 1991) ──────────────────

pub struct ACTRActivationFactor;

impl ScoringFactor for ACTRActivationFactor {
    fn name(&self) -> &'static str { "actr_activation" }

    fn compute(
        &self,
        ctx: &ScoringContext,
        config: &ScoringConfig,
        decomp: &mut ScoreDecomposition,
    ) -> Option<f32> {
        let activation = ctx.state.actr_base_level_activation(ctx.now_ms);
        decomp.actr_activation = activation;
        Some(config.actr_floor + config.actr_range * activation)
    }
}

// ── Confidence ──────────────────────────────────────────────────────────────

pub struct ConfidenceFactor;

impl ScoringFactor for ConfidenceFactor {
    fn name(&self) -> &'static str { "confidence" }

    fn compute(
        &self,
        ctx: &ScoringContext,
        _config: &ScoringConfig,
        _decomp: &mut ScoreDecomposition,
    ) -> Option<f32> {
        Some(ctx.state.confidence)
    }
}

// ── Prediction error / surprise (FEP §2.3) ─────────────────────────────────

pub struct SurpriseFactor;

impl ScoringFactor for SurpriseFactor {
    fn name(&self) -> &'static str { "surprise" }

    fn compute(
        &self,
        ctx: &ScoringContext,
        config: &ScoringConfig,
        decomp: &mut ScoreDecomposition,
    ) -> Option<f32> {
        let boost = 1.0 + config.surprise_weight * ctx.state.surprise;
        decomp.surprise_boost = boost;
        Some(boost)
    }
}

// ── Flashbulb / arousal ─────────────────────────────────────────────────────

pub struct ArousalFactor;

impl ScoringFactor for ArousalFactor {
    fn name(&self) -> &'static str { "arousal" }

    fn compute(
        &self,
        ctx: &ScoringContext,
        config: &ScoringConfig,
        decomp: &mut ScoreDecomposition,
    ) -> Option<f32> {
        let boost = 1.0 + config.arousal_weight * ctx.state.affect_arousal;
        decomp.arousal_boost = boost;
        Some(boost)
    }
}

// ── Mood-congruent recall (Bower 1981) ──────────────────────────────────────

pub struct MoodCongruenceFactor;

impl ScoringFactor for MoodCongruenceFactor {
    fn name(&self) -> &'static str { "mood_congruence" }

    fn compute(
        &self,
        ctx: &ScoringContext,
        config: &ScoringConfig,
        decomp: &mut ScoreDecomposition,
    ) -> Option<f32> {
        let boost = match (ctx.query_valence, ctx.query_arousal) {
            (Some(qv), Some(qa)) if qa > config.mood_min_arousal => {
                let valence_match = 1.0 - (qv - ctx.state.affect_valence).abs();
                let intensity = qa.max(ctx.state.affect_arousal);
                1.0 + config.mood_weight * valence_match.max(0.0) * intensity
            }
            _ => 1.0,
        };
        decomp.mood_congruence = boost;
        Some(boost)
    }
}

// ── Frustration-escalation correction boost ─────────────────────────────────

pub struct FrustrationEscalationFactor;

impl ScoringFactor for FrustrationEscalationFactor {
    fn name(&self) -> &'static str { "frustration" }

    fn compute(
        &self,
        ctx: &ScoringContext,
        config: &ScoringConfig,
        decomp: &mut ScoreDecomposition,
    ) -> Option<f32> {
        let boost = match (ctx.query_valence, ctx.query_arousal) {
            (Some(qv), Some(qa))
                if qv < config.frustration_valence_threshold
                    && qa > config.frustration_arousal_threshold =>
            {
                let frustration = (-qv - config.frustration_valence_threshold.abs()).min(0.7)
                    * (qa - config.frustration_arousal_threshold).min(0.6);
                match ctx.kind {
                    "correction" => 1.0 + config.frustration_correction_weight * frustration,
                    "preference" => 1.0 + config.frustration_preference_weight * frustration,
                    _ => 1.0,
                }
            }
            _ => 1.0,
        };
        decomp.frustration_boost = boost;
        Some(boost)
    }
}

// ── Status gate/multiplier ──────────────────────────────────────────────────

pub struct StatusFactor;

impl ScoringFactor for StatusFactor {
    fn name(&self) -> &'static str { "status" }

    fn compute(
        &self,
        ctx: &ScoringContext,
        config: &ScoringConfig,
        decomp: &mut ScoreDecomposition,
    ) -> Option<f32> {
        let mul = status_multiplier(&ctx.state.status, config)?;
        decomp.status_mul = mul;
        Some(mul)
    }
}

// ── Epistemic status multiplier ─────────────────────────────────────────────

pub struct EpistemicFactor;

impl ScoringFactor for EpistemicFactor {
    fn name(&self) -> &'static str { "epistemic" }

    fn compute(
        &self,
        ctx: &ScoringContext,
        config: &ScoringConfig,
        decomp: &mut ScoreDecomposition,
    ) -> Option<f32> {
        let mul = epistemic_multiplier(&ctx.state.epistemic_status, config);
        decomp.epistemic_mul = mul;
        Some(mul)
    }
}

// ── Kind tier multiplier ────────────────────────────────────────────────────

pub struct KindFactor;

impl ScoringFactor for KindFactor {
    fn name(&self) -> &'static str { "kind" }

    fn compute(
        &self,
        ctx: &ScoringContext,
        config: &ScoringConfig,
        _decomp: &mut ScoreDecomposition,
    ) -> Option<f32> {
        Some(kind_multiplier(ctx.kind, config))
    }
}

// ── Realm reliability (Product of Experts) ──────────────────────────────────

pub struct RealmReliabilityFactor;

impl ScoringFactor for RealmReliabilityFactor {
    fn name(&self) -> &'static str { "realm_reliability" }

    fn compute(
        &self,
        ctx: &ScoringContext,
        _config: &ScoringConfig,
        _decomp: &mut ScoreDecomposition,
    ) -> Option<f32> {
        Some(ctx.realm_reliability)
    }
}

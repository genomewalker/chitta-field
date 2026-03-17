use crate::ids::MemoryId;
use std::collections::HashMap;

#[derive(Debug, Clone)]
struct AccessStats {
    /// Exponentially weighted mean inter-access interval (ms).
    ewma_interval_ms: f64,
    last_accessed_ms: i64,
    access_count: u32,
}

pub struct PlasticityLearner {
    stats: HashMap<MemoryId, AccessStats>,
    /// EMA decay factor (0.3 = recent accesses weighted more).
    alpha: f64,
}

impl PlasticityLearner {
    pub fn new() -> Self {
        Self {
            stats: HashMap::new(),
            alpha: 0.3,
        }
    }

    /// Record an access to a memory. Returns the new recommended decay rate.
    /// Frequently accessed memories get lower decay rates (they matter).
    /// Never-accessed memories keep the default.
    pub fn record_access(&mut self, memory_id: MemoryId, now_ms: i64) -> f32 {
        let entry = self.stats.entry(memory_id).or_insert(AccessStats {
            ewma_interval_ms: 86_400_000.0, // default: 1 day
            last_accessed_ms: now_ms,
            access_count: 0,
        });

        if entry.access_count > 0 {
            let interval = (now_ms - entry.last_accessed_ms).max(1) as f64;
            entry.ewma_interval_ms =
                self.alpha * interval + (1.0 - self.alpha) * entry.ewma_interval_ms;
        }
        entry.last_accessed_ms = now_ms;
        entry.access_count += 1;

        self.recommended_decay_rate(memory_id)
    }

    /// Get recommended decay rate for a memory based on its access history.
    /// Base: 0.001/day. Faster for rarely accessed, slower for frequently accessed.
    /// Range: [0.0001, 0.01] per day.
    pub fn recommended_decay_rate(&self, memory_id: MemoryId) -> f32 {
        let ewma = self
            .stats
            .get(&memory_id)
            .map(|s| s.ewma_interval_ms)
            .unwrap_or(86_400_000.0);

        // ewma_interval_ms: short = frequent access = lower decay
        // 1 hour = 3_600_000 ms → decay 0.0001 (slow decay)
        // 7 days = 604_800_000 ms → decay 0.01 (fast decay)
        let days = ewma / 86_400_000.0;
        let rate = (0.0001_f64 * days.powf(0.5)).clamp(0.0001, 0.01);
        rate as f32
    }

    pub fn memory_count(&self) -> usize {
        self.stats.len()
    }
}

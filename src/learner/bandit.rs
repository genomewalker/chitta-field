/// Beta distribution prior for binary outcomes (success/failure).
/// Used by RouteLearner for multi-armed bandit over retrieval routes.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BetaPrior {
    pub alpha: f64, // successes + 1
    pub beta: f64,  // failures + 1
}

impl BetaPrior {
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
        }
    }

    /// Update with outcome (reward in [0,1]).
    pub fn update(&mut self, reward: f32) {
        let r = reward.clamp(0.0, 1.0) as f64;
        self.alpha += r;
        self.beta += 1.0 - r;
    }

    /// Expected value (mean of Beta distribution).
    pub fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Approximate Thompson sample using normal approximation.
    /// Returns a sample from Beta(alpha, beta) using Wilson score approximation.
    /// Pure Rust, no external RNG crate needed.
    pub fn sample(&self, seed: u64) -> f64 {
        // Use simple pseudo-random normal via Box-Muller with seed
        let mean = self.mean();
        let var = (self.alpha * self.beta)
            / ((self.alpha + self.beta).powi(2) * (self.alpha + self.beta + 1.0));
        let std = var.sqrt();
        // Box-Muller with LCG seed
        let u1 = lcg_uniform(seed);
        let u2 = lcg_uniform(
            seed.wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407),
        );
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        (mean + std * z).clamp(0.0, 1.0)
    }
}

fn lcg_uniform(seed: u64) -> f64 {
    let x = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (x >> 11) as f64 / (1u64 << 53) as f64
}

/// Gaussian prior for continuous parameters.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GaussianPrior {
    pub mu: f64,
    pub sigma: f64,
    pub n: u64,
}

impl GaussianPrior {
    pub fn new(mu: f64, sigma: f64) -> Self {
        Self { mu, sigma, n: 0 }
    }

    /// Online update with a new observation (Welford's algorithm for sigma).
    pub fn update(&mut self, value: f64) {
        self.n += 1;
        let delta = value - self.mu;
        self.mu += delta / self.n as f64;
        // Bayesian shrinkage toward prior: blend with prior sigma
        let new_sigma = self.sigma * 0.95 + (value - self.mu).abs() * 0.05;
        self.sigma = new_sigma.max(0.01);
    }

    pub fn mean(&self) -> f64 {
        self.mu
    }
}

//! What varies between agents.
//!
//! Structure and parameters both, because the interesting axis of the archive — `dim(g)` —
//! is a property of the ansatz rather than of the trained angles.

use overtone_rl::{Policy, Scaling, SpectralControlAnsatz};
use rand::Rng;

/// One agent, entirely.
#[derive(Clone, Debug, PartialEq)]
pub struct Genome {
    pub qubits: usize,
    pub layers: usize,
    pub softmax: bool,
    pub trainable_lambda: bool,
    pub entangle: bool,
    /// Circuit parameters.
    ///
    /// **Invariant: the length is determined by the four fields above.** Changing any of them
    /// without redrawing this leaves a genome that will panic deep inside the policy rather
    /// than here, which is exactly how the first version of the descriptor test failed. Use
    /// [`Genome::structured`] to build one, and [`Genome::is_well_formed`] to check one.
    pub params: Vec<f64>,
}

impl Genome {
    /// Build the policy this genome describes.
    pub fn policy(&self) -> Policy {
        let scaling = if self.trainable_lambda {
            Scaling::Trainable
        } else {
            Scaling::Pinned
        };
        let ansatz = SpectralControlAnsatz::build(self.qubits, self.layers, scaling, self.entangle);
        if self.softmax {
            Policy::softmax(ansatz, 1.0)
        } else {
            Policy::raw(ansatz)
        }
    }

    /// A genome with a chosen structure and freshly drawn angles of the right length.
    pub fn structured<R: Rng + ?Sized>(
        qubits: usize,
        layers: usize,
        softmax: bool,
        trainable_lambda: bool,
        entangle: bool,
        rng: &mut R,
    ) -> Genome {
        let mut g = Genome {
            qubits,
            layers,
            softmax,
            trainable_lambda,
            entangle,
            params: Vec::new(),
        };
        g.reseed_params(rng);
        g
    }

    /// Whether the parameter vector matches the structure.
    pub fn is_well_formed(&self) -> bool {
        self.params.len() == self.policy().num_params()
    }

    /// A random agent. The structure is drawn uniformly and the angles from a spread wide
    /// enough to reach every part of the circle, because a QD archive that starts near zero
    /// spends its whole budget crawling away from it.
    pub fn random<R: Rng + ?Sized>(rng: &mut R) -> Genome {
        let mut g = Genome {
            qubits: rng.gen_range(2..=5),
            layers: rng.gen_range(1..=4),
            softmax: rng.gen_bool(0.5),
            trainable_lambda: rng.gen_bool(0.5),
            entangle: rng.gen_bool(0.7),
            params: Vec::new(),
        };
        g.reseed_params(rng);
        g
    }

    fn reseed_params<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        let policy = self.policy();
        let mut params = vec![0.0; policy.num_params()];
        for p in params.iter_mut().take(policy.ansatz.num_variational()) {
            *p = rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI);
        }
        for i in policy.ansatz.lambda_range() {
            params[i] = rng.gen_range(0.2..6.0);
        }
        let w = policy.weight_range();
        if !w.is_empty() {
            params[w.start] = -1.0;
            params[w.start + 1] = 1.0;
        }
        self.params = params;
    }

    /// Mutate. Mostly angles; occasionally structure.
    ///
    /// A structural mutation changes the parameter count, so the angles are redrawn rather
    /// than truncated — carrying angles across a change of circuit would map them onto gates
    /// they were never fitted for, which is worse than starting fresh and much harder to
    /// reason about.
    pub fn mutate<R: Rng + ?Sized>(&self, rng: &mut R, sigma: f64) -> Genome {
        let mut g = self.clone();
        if rng.gen_bool(0.15) {
            match rng.gen_range(0..4) {
                0 => g.qubits = rng.gen_range(2..=5),
                1 => g.layers = rng.gen_range(1..=4),
                2 => g.entangle = !g.entangle,
                _ => g.softmax = !g.softmax,
            }
            g.reseed_params(rng);
            return g;
        }
        let policy = g.policy();
        let lambda = policy.ansatz.lambda_range();
        let weights = policy.weight_range();
        for (i, p) in g.params.iter_mut().enumerate() {
            if weights.contains(&i) {
                continue;
            }
            *p += rng.gen_range(-sigma..sigma);
            if lambda.contains(&i) {
                *p = p.clamp(0.05, 8.0);
            }
        }
        g
    }
}

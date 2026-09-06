//! RAW-PQC and SOFTMAX-PQC policies (Part I 6.3; Jerbi et al., NeurIPS 2021).
//!
//! Both are two-action policies reading a single observable, `Z` on qubit 0.
//!
//! **RAW-PQC** is the Born rule directly: `pi(1|s) = (1 + <Z_0>_s)/2`. Since `<Z_0>_s` is a
//! trig polynomial of degree at most the encoding ceiling, the policy is *strictly*
//! band-limited. This is the scientifically clean one.
//!
//! **SOFTMAX-PQC** is `pi(a|s) = softmax_a(beta <O_a>_s)` with `<O_a> = w_a <Z_0>`, for
//! trainable weights `w` and inverse temperature `beta`. Empirically stronger. Note what
//! this does spectrally: the *logit* stays band-limited, but the probability is a sigmoid
//! of a band-limited function, and a sigmoid of a trig polynomial has energy at harmonics
//! the circuit cannot reach. The leakage lives in `pi(a|s)`, not in the logit -- which is
//! why the spectral instrument of Phase 3 must transform the probability. Transforming the
//! logit, as Part I 6.5's wording suggests, would show no leakage at all and quietly
//! falsify a true claim.

use overtone_sim::grad::adjoint;
use overtone_sim::Observable;

use crate::ansatz::Ansatz;

/// Which policy family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyKind {
    /// Born rule. Strictly band-limited.
    Raw,
    /// Softmax over weighted observables. Spectrally leaky.
    Softmax,
}

/// A two-action parameterised quantum policy.
#[derive(Clone, Debug)]
pub struct Policy {
    pub kind: PolicyKind,
    pub ansatz: Ansatz,
    /// Inverse temperature. Ignored by RAW-PQC.
    pub beta: f64,
    observable: Observable,
}

/// Probabilities are clamped this far from the boundary before a logarithm is taken.
///
/// RAW-PQC can reach `pi = 0` or `1` exactly -- the Born rule has no floor -- and the score
/// function `grad log pi` diverges there. Clamping bounds the gradient without changing the
/// policy that is actually sampled from.
const PROB_FLOOR: f64 = 1e-9;

impl Policy {
    pub fn new(kind: PolicyKind, ansatz: Ansatz, beta: f64) -> Self {
        Policy {
            kind,
            ansatz,
            beta,
            observable: Observable::z(0),
        }
    }

    pub fn raw(ansatz: Ansatz) -> Self {
        Policy::new(PolicyKind::Raw, ansatz, 1.0)
    }

    pub fn softmax(ansatz: Ansatz, beta: f64) -> Self {
        Policy::new(PolicyKind::Softmax, ansatz, beta)
    }

    /// Total parameter count: the circuit's, plus two output weights for SOFTMAX-PQC.
    pub fn num_params(&self) -> usize {
        self.ansatz.num_params() + self.num_weights()
    }

    fn num_weights(&self) -> usize {
        match self.kind {
            PolicyKind::Raw => 0,
            PolicyKind::Softmax => 2,
        }
    }

    /// Index range of the trainable output weights, empty for RAW-PQC.
    pub fn weight_range(&self) -> std::ops::Range<usize> {
        self.ansatz.num_params()..self.ansatz.num_params() + self.num_weights()
    }

    /// `<Z_0>` and its gradient with respect to the circuit parameters.
    fn observable_value_grad(&self, params: &[f64], s: &[f64]) -> (f64, Vec<f64>) {
        let circuit = self.ansatz.build(s);
        let circuit_params = &params[..self.ansatz.num_params()];
        let vg = adjoint::value_and_grad(&circuit, circuit_params, &self.observable);
        (vg.value, vg.grad)
    }

    /// `<Z_0>` alone.
    pub fn observable_value(&self, params: &[f64], s: &[f64]) -> f64 {
        let circuit = self.ansatz.build(s);
        self.observable
            .expectation(&circuit.run(&params[..self.ansatz.num_params()]))
    }

    /// `pi(1|s)`.
    pub fn prob_action1(&self, params: &[f64], s: &[f64]) -> f64 {
        let z = self.observable_value(params, s);
        self.prob_from_observable(params, z)
    }

    fn prob_from_observable(&self, params: &[f64], z: f64) -> f64 {
        match self.kind {
            // Born rule. Already in [0, 1] because |<Z>| <= 1.
            PolicyKind::Raw => (1.0 + z) * 0.5,
            PolicyKind::Softmax => {
                let w = self.weight_range();
                let (w0, w1) = (params[w.start], params[w.start + 1]);
                // Two actions collapse the softmax to a logistic in the logit difference.
                logistic(self.beta * (w1 - w0) * z)
            }
        }
    }

    /// `pi(a|s)` for either action.
    pub fn prob(&self, params: &[f64], s: &[f64], action: usize) -> f64 {
        let p1 = self.prob_action1(params, s);
        if action == 1 {
            p1
        } else {
            1.0 - p1
        }
    }

    /// Sample an action.
    pub fn sample<R: rand::Rng + ?Sized>(&self, params: &[f64], s: &[f64], rng: &mut R) -> usize {
        usize::from(rng.gen::<f64>() < self.prob_action1(params, s))
    }

    /// The score function `grad_theta log pi(a|s)`, over the full parameter vector.
    ///
    /// For RAW-PQC, `pi(1|s) = (1 + z)/2` gives `grad log pi(1|s) = grad z / (1 + z)` and
    /// `grad log pi(0|s) = -grad z / (1 - z)`.
    ///
    /// For SOFTMAX-PQC this is Lemma 1 of Jerbi et al.:
    /// `grad log pi(a|s) = beta (grad <O_a> - sum_a' pi(a'|s) grad <O_a'>)`.
    pub fn grad_log_prob(&self, params: &[f64], s: &[f64], action: usize) -> Vec<f64> {
        let (z, dz) = self.observable_value_grad(params, s);
        let mut grad = vec![0.0; self.num_params()];
        let n_circuit = self.ansatz.num_params();

        match self.kind {
            PolicyKind::Raw => {
                // d log pi / d theta = (dp/dtheta) / p, with p = (1 +- z)/2.
                let denom = if action == 1 { 1.0 + z } else { z - 1.0 };
                let scale = 1.0 / clamp_away_from_zero(denom);
                for i in 0..n_circuit {
                    grad[i] = scale * dz[i];
                }
            }
            PolicyKind::Softmax => {
                let w = self.weight_range();
                let (w0, w1) = (params[w.start], params[w.start + 1]);
                let p1 = logistic(self.beta * (w1 - w0) * z);
                let p = [1.0 - p1, p1];

                // <O_a> = w_a z, so d<O_a>/dtheta = w_a dz/dtheta.
                let w_a = if action == 1 { w1 } else { w0 };
                let w_mean = p[0] * w0 + p[1] * w1;
                let circuit_scale = self.beta * (w_a - w_mean);
                for i in 0..n_circuit {
                    grad[i] = circuit_scale * dz[i];
                }

                // d<O_a>/dw_b = delta_{ab} z, so the weight gradient is
                // beta z (delta_{ab} - pi(b|s)).
                for b in 0..2 {
                    let delta = f64::from(b == action);
                    grad[w.start + b] = self.beta * z * (delta - p[b]);
                }
            }
        }

        grad
    }

    /// Exact expected return on `SpectralControl-k`, by quadrature.
    pub fn expected_return(&self, params: &[f64], k: usize, nodes: usize) -> f64 {
        let env = crate::env::SpectralControl::new(k);
        env.expected_return(nodes, |s| self.prob_action1(params, &[s]))
    }

    /// The exact policy gradient of the expected return, by quadrature.
    ///
    /// For this bandit the return is differentiable in closed form:
    /// `J = E_s[cos(ks) (2 pi(1|s) - 1)]`, so `grad J = E_s[2 cos(ks) grad pi(1|s)]`.
    /// REINFORCE does not need this -- but having it means the sampled estimator can be
    /// checked against the quantity it is supposed to estimate, which is the RL analogue
    /// of the adjoint-versus-parameter-shift check in Phase 1.
    pub fn analytic_return_grad(&self, params: &[f64], k: usize, nodes: usize) -> Vec<f64> {
        let pi_c = std::f64::consts::PI;
        let mut acc = vec![0.0; self.num_params()];

        for i in 0..nodes {
            let s = -pi_c + 2.0 * pi_c * (i as f64 + 0.5) / nodes as f64;
            let cos_ks = ((k as f64) * s).cos();
            let dp1 = self.grad_prob_action1(params, &[s]);
            for (a, g) in acc.iter_mut().zip(&dp1) {
                *a += 2.0 * cos_ks * g;
            }
        }

        for a in &mut acc {
            *a /= nodes as f64;
        }
        acc
    }

    /// `grad_theta pi(1|s)`.
    pub fn grad_prob_action1(&self, params: &[f64], s: &[f64]) -> Vec<f64> {
        let (z, dz) = self.observable_value_grad(params, s);
        let mut grad = vec![0.0; self.num_params()];
        let n_circuit = self.ansatz.num_params();

        match self.kind {
            PolicyKind::Raw => {
                for i in 0..n_circuit {
                    grad[i] = 0.5 * dz[i];
                }
            }
            PolicyKind::Softmax => {
                let w = self.weight_range();
                let (w0, w1) = (params[w.start], params[w.start + 1]);
                let d = self.beta * (w1 - w0);
                let p1 = logistic(d * z);
                let sig = p1 * (1.0 - p1);
                for i in 0..n_circuit {
                    grad[i] = sig * d * dz[i];
                }
                grad[w.start] = sig * (-self.beta * z);
                grad[w.start + 1] = sig * (self.beta * z);
            }
        }

        grad
    }
}

#[inline]
fn logistic(x: f64) -> f64 {
    // Branch on the sign so neither exp() overflows for large |x|.
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[inline]
fn clamp_away_from_zero(x: f64) -> f64 {
    if x.abs() < PROB_FLOOR {
        if x < 0.0 {
            -PROB_FLOOR
        } else {
            PROB_FLOOR
        }
    } else {
        x
    }
}

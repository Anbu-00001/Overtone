//! Distributional RL, and the shot-budget dial (Part V 4).
//!
//! # The rhyme, and where the spec overstates it
//!
//! Part V 4's observation is exact: a quantum measurement never returns an expectation
//! value, it returns samples, and `<O>` is an average over shots. Distributional RL is the
//! value-side formalism that matches what the policy side was already doing.
//!
//! The spec then says: "sweep shots per estimate from 10 to 10,000 and watch the return
//! distribution sharpen." **It does not sharpen, and it should not.** Two different
//! distributions are in play and only one of them depends on the shot count:
//!
//! - The **return distribution** `Z^pi(s)` is aleatoric. On `SpectralControl-k` it is two
//!   atoms, `+cos(ks)` with probability `pi(1|s)` and `-cos(ks)` otherwise. Its spread is a
//!   property of the policy and the environment. More shots do not narrow it, and a
//!   distributional critic that reported otherwise would be wrong.
//! - The **estimate** of any summary of that distribution -- `J`, a quantile, the critic's
//!   own parameters -- is epistemic, and *that* sharpens as `1/sqrt(N)`.
//!
//! Both are measured in `examples/shot_dial.rs`. Conflating them is the trap the panel has
//! to avoid, because "more shots, tighter distribution" is exactly the intuition a reader
//! arrives with.
//!
//! # Where shots do change the physics
//!
//! The Born-rule policy is *exactly* implementable at one shot: `pi(1|s) = (1 + <Z>)/2`, so
//! a single measurement outcome already is a sample from `pi`. The action distribution is
//! unbiased at any shot count, and the panel should say so -- it is the only genuinely
//! free thing in NISQ.
//!
//! The **gradient** is not. `grad log pi(1|s) = grad z / (1 + z)` is nonlinear in the
//! measured `z`, so a shot-noisy `z` gives a biased score function by Jensen's inequality,
//! and the near-deterministic states where `z -> -1` are exactly where it blows up. That
//! bias falls as `1/N` while the noise falls as `1/sqrt(N)`, so it is invisible at large
//! budgets and dominant at small ones. This module measures it rather than asserting it.

use overtone_sim::measure::{Budget, DiagonalTerm};
use overtone_sim::Observable;
use rand::Rng;

use crate::env::SpectralControl;
use crate::policy::Policy;

/// A quantile representation of a return distribution: `N` locations at fixed cumulative
/// probabilities.
///
/// Dabney et al., *Distributional RL with Quantile Regression* (AAAI 2018). The locations
/// sit at the quantile **midpoints** `tau_i = (2i - 1) / 2N`, which is Lemma 2 of that
/// paper: those are the points that minimise the 1-Wasserstein distance to the true
/// distribution, and a naive `i/N` would be biased toward the upper tail.
#[derive(Clone, Debug)]
pub struct QuantileCritic {
    /// `theta_i`, ascending is not enforced -- quantile regression can cross, and pretending
    /// otherwise by sorting would hide a real failure mode.
    pub theta: Vec<f64>,
}

impl QuantileCritic {
    pub fn new(n: usize, initial: f64) -> QuantileCritic {
        assert!(n > 0, "a quantile critic needs at least one quantile");
        QuantileCritic {
            theta: vec![initial; n],
        }
    }

    pub fn len(&self) -> usize {
        self.theta.len()
    }

    pub fn is_empty(&self) -> bool {
        self.theta.is_empty()
    }

    /// `tau_hat_i = (2i - 1) / 2N`, one-based `i`.
    pub fn tau(&self, i: usize) -> f64 {
        (2 * i + 1) as f64 / (2 * self.theta.len()) as f64
    }

    /// The mean of the represented distribution, which is what an ordinary critic learns.
    pub fn mean(&self) -> f64 {
        self.theta.iter().sum::<f64>() / self.theta.len() as f64
    }

    /// One quantile-regression step against a sampled return.
    ///
    /// `rho^kappa_tau(u) = |tau - 1{u < 0}| L_kappa(u)`, Dabney et al. eq. 10, with the
    /// Huber loss of eq. 9. The gradient of `L_kappa` is `u` inside the quadratic region and
    /// `kappa sign(u)` outside, so the update below is exactly `-d rho / d theta`.
    ///
    /// # `kappa` is not scale-free, and the default is wrong here
    ///
    /// Dabney et al. write `rho^0_tau = rho_tau`: at `kappa = 0` the Huber loss *is* the
    /// plain quantile loss, whose subgradient is `sign(u)`. Reading eq. 9 literally at
    /// `kappa = 0` instead gives `0 sign(u) = 0` and no learning at all, so that case is
    /// handled explicitly below rather than by taking a limit.
    ///
    /// The opposite end matters more. `kappa = 1` is the published default, chosen against
    /// Atari returns in the hundreds. `SpectralControl` returns live in `[-1, 1]`, so every
    /// residual falls inside the quadratic region, the subgradient becomes `u` rather than
    /// `sign(u)`, and what is being fit is an **expectile**, not a quantile. Measured in
    /// `examples/shot_dial.rs`: at `kappa = 1` the 1-Wasserstein distance to a known
    /// two-atom distribution stalls near `0.33` no matter how many samples arrive, and it
    /// falls by more than an order of magnitude as `kappa` goes to zero. The right rule is
    /// `kappa` well below the spread of the returns, and the wrong one is a default.
    pub fn observe(&mut self, sample: f64, rate: f64, kappa: f64) {
        for i in 0..self.theta.len() {
            let u = sample - self.theta[i];
            let weight = (self.tau(i) - f64::from(u < 0.0)).abs();
            let huber_grad = if kappa <= 0.0 {
                u.signum()
            } else if u.abs() <= kappa {
                u
            } else {
                kappa * u.signum()
            };
            self.theta[i] += rate * weight * huber_grad;
        }
    }

    /// The 1-Wasserstein distance to a distribution given by its inverse CDF.
    ///
    /// The metric Dabney et al. prove the quantile projection minimises, so it is the
    /// honest way to score a learned critic against a known answer.
    pub fn wasserstein_to<F: Fn(f64) -> f64>(&self, inverse_cdf: F, grid: usize) -> f64 {
        let mut acc = 0.0;
        for g in 0..grid {
            let p = (g as f64 + 0.5) / grid as f64;
            let truth = inverse_cdf(p);
            let mine = self.quantile(p);
            acc += (truth - mine).abs();
        }
        acc / grid as f64
    }

    /// The critic's own inverse CDF: a step function through the `theta_i`.
    pub fn quantile(&self, p: f64) -> f64 {
        let n = self.theta.len();
        let idx = ((p * n as f64).floor() as usize).min(n - 1);
        self.theta[idx]
    }
}

/// The exact return distribution of a Born-rule policy on `SpectralControl-k` at one state.
///
/// Two atoms. Having it in closed form is what makes the critic checkable rather than
/// merely plausible: the quantile regression is scored against this, not against a second
/// approximation.
#[derive(Clone, Copy, Debug)]
pub struct TwoAtoms {
    /// Return when action 1 is taken.
    pub high: f64,
    /// Return when action 0 is taken.
    pub low: f64,
    /// Probability of the *high* atom.
    pub p_high: f64,
}

impl TwoAtoms {
    pub fn for_state(env: &SpectralControl, prob_action1: f64, s: f64) -> TwoAtoms {
        let c = (env.k as f64 * s).cos();
        TwoAtoms {
            high: c,
            low: -c,
            p_high: prob_action1,
        }
    }

    pub fn mean(&self) -> f64 {
        self.p_high * self.high + (1.0 - self.p_high) * self.low
    }

    pub fn variance(&self) -> f64 {
        let m = self.mean();
        self.p_high * (self.high - m).powi(2) + (1.0 - self.p_high) * (self.low - m).powi(2)
    }

    /// Inverse CDF. With `low` and `high` possibly out of order (`cos(ks) < 0`), the atoms
    /// are ordered here rather than assumed.
    pub fn inverse_cdf(&self, p: f64) -> f64 {
        let (small, large, p_small) = if self.low <= self.high {
            (self.low, self.high, 1.0 - self.p_high)
        } else {
            (self.high, self.low, self.p_high)
        };
        if p < p_small {
            small
        } else {
            large
        }
    }

    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        if rng.gen::<f64>() < self.p_high {
            self.high
        } else {
            self.low
        }
    }
}

/// A policy observable estimated under a shot budget.
///
/// `Budget::Exact` reproduces [`Policy::observable_value`] bit for bit; anything else is a
/// projective measurement of the same quantity.
pub fn measured_observable<R: Rng + ?Sized>(
    policy: &Policy,
    params: &[f64],
    s: &[f64],
    budget: Budget,
    rng: &mut R,
) -> f64 {
    let circuit = policy.ansatz.build(s);
    let psi = circuit.run(&params[..policy.ansatz.num_params()]);
    let term = DiagonalTerm::from_observable(&Observable::z(0))
        .expect("the RAW-PQC policy observable is Z_0");
    term.estimate(&psi, budget, rng)
}

/// The score function evaluated from a *measured* observable, with the clip a real device
/// forces on it.
///
/// `grad log pi(1|s) = grad z / (1 + z)` is unbounded as `z -> -1`, and a finite-shot `z`
/// reaches `-1` exactly whenever every shot comes back the same way. Some clip is therefore
/// mandatory, not a refinement: without it the estimator is not merely biased but infinite
/// with positive probability. `eps` is that clip, and its cost is measured in
/// [`score_bias`] rather than assumed small.
pub fn measured_score(z: f64, dz: &[f64], action: usize, eps: f64) -> Vec<f64> {
    let denominator = if action == 1 {
        (1.0 + z).max(eps)
    } else {
        -(1.0 - z).max(eps)
    };
    dz.iter().map(|g| g / denominator).collect()
}

/// Mean deviation of the shot-measured score function from the exact one, over `trials`.
///
/// Positive by Jensen for the `1/(1+z)` branch, and the quantity that tells a reader what a
/// shot budget actually costs them in training rather than in wall clock.
#[allow(clippy::too_many_arguments)]
pub fn score_bias<R: Rng + ?Sized>(
    policy: &Policy,
    params: &[f64],
    s: &[f64],
    action: usize,
    budget: Budget,
    eps: f64,
    trials: usize,
    rng: &mut R,
) -> f64 {
    let (z_exact, dz) = policy.observable_value_grad(params, s);
    let truth = measured_score(z_exact, &dz, action, eps);
    let mut acc = vec![0.0; dz.len()];
    for _ in 0..trials {
        let z = measured_observable(policy, params, s, budget, rng);
        for (a, v) in acc.iter_mut().zip(measured_score(z, &dz, action, eps)) {
            *a += v;
        }
    }
    acc.iter()
        .zip(&truth)
        .map(|(a, t)| (a / trials as f64 - t).abs())
        .fold(0.0, f64::max)
}

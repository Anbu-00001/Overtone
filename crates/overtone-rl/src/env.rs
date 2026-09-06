//! `SpectralControl-k` (Part I 7.1).
//!
//! A contextual bandit. Call it that; it is not dressed up as more than it is.
//!
//! ```text
//! state  s ~ Uniform[-pi, pi)
//! action a in {0, 1}
//! reward r(s, a) = (2a - 1) * cos(k s)
//! ```
//!
//! # Why this environment and not a cart
//!
//! The expected return decomposes as
//!
//! ```text
//! J = E_s[ cos(k s) * (2 pi(1|s) - 1) ].
//! ```
//!
//! For a RAW-PQC, `2 pi(1|s) - 1` is exactly `<Z_0>_s`, a trig polynomial whose degree is
//! the encoding's frequency ceiling. The expectation then picks out precisely that
//! polynomial's frequency-`k` Fourier coefficient and ignores everything else. So the
//! environment is a Fourier coefficient extractor wearing a reward function, and the
//! theorem of Part I 1 becomes a number you can measure.
//!
//! The consequence is unusually crisp for an ML benchmark: with `lambda` pinned and a
//! frequency ceiling below `k`, the agent scores **exactly zero**. Not "worse". Zero, for
//! any parameters, forever.

use rand::Rng;

/// The `SpectralControl-k` contextual bandit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpectralControl {
    /// The environment's frequency. The agent must reach it or score nothing.
    pub k: usize,
}

impl SpectralControl {
    pub fn new(k: usize) -> Self {
        assert!(
            k > 0,
            "SpectralControl-0 has a constant reward and is not a task"
        );
        SpectralControl { k }
    }

    /// Draw a state uniformly from `[-pi, pi)`.
    pub fn sample_state<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI)
    }

    /// `r(s, a) = (2a - 1) cos(k s)`.
    pub fn reward(&self, s: f64, action: usize) -> f64 {
        debug_assert!(action < 2, "SpectralControl has two actions");
        let sign = 2.0 * action as f64 - 1.0;
        sign * ((self.k as f64) * s).cos()
    }

    /// The optimal policy: take action 1 exactly where `cos(k s) > 0`.
    pub fn optimal_action(&self, s: f64) -> usize {
        usize::from(((self.k as f64) * s).cos() > 0.0)
    }

    /// The return of the optimal (unconstrained) policy, `E|cos(k s)| = 2/pi`.
    ///
    /// No band-limited policy attains this; the LP ceiling of [`crate::ceiling`] says how
    /// close a given frequency ceiling can get.
    pub fn optimal_return(&self) -> f64 {
        2.0 / std::f64::consts::PI
    }

    /// Exact expected return of a policy, by quadrature rather than by sampling.
    ///
    /// `prob_action1(s)` is the policy's probability of action 1 at `s`. The integrand is
    /// smooth and periodic, so the uniform midpoint rule converges spectrally: a few
    /// hundred nodes give machine precision, where a Monte Carlo estimate over the same
    /// budget would still be at the second decimal.
    ///
    /// This is what tests should assert on. An episodic average over 5000 samples has a
    /// standard error near `1/sqrt(5000) = 0.014`, so a threshold of 0.02 on the *sampled*
    /// return would be about 1.4 sigma and would flake. Measured this way the same
    /// quantity comes out at rounding.
    pub fn expected_return<F>(&self, nodes: usize, prob_action1: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let pi = std::f64::consts::PI;
        let mut acc = 0.0;
        for i in 0..nodes {
            let s = -pi + 2.0 * pi * (i as f64 + 0.5) / nodes as f64;
            let p1 = prob_action1(s);
            // E[r] at this state, marginalised over the policy's own action choice:
            //   p1 * cos(ks) + (1 - p1) * (-cos(ks)) = (2 p1 - 1) cos(ks).
            acc += (2.0 * p1 - 1.0) * ((self.k as f64) * s).cos();
        }
        acc / nodes as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimal_policy_attains_two_over_pi() {
        let env = SpectralControl::new(3);
        let j = env.expected_return(4096, |s| env.optimal_action(s) as f64);
        assert!(
            (j - env.optimal_return()).abs() < 1e-6,
            "optimal return {j} vs 2/pi {}",
            env.optimal_return()
        );
    }

    #[test]
    fn a_pure_cosine_policy_scores_one_half() {
        // pi(1|s) = (1 + cos(ks))/2 makes 2 pi(1|s) - 1 = cos(ks), whose frequency-k
        // coefficient is 1, giving J = 1/2. This is the L = k step of the staircase.
        let env = SpectralControl::new(3);
        let j = env.expected_return(4096, |s| (1.0 + (3.0 * s).cos()) / 2.0);
        assert!((j - 0.5).abs() < 1e-9, "J = {j}");
    }

    #[test]
    fn a_policy_at_the_wrong_frequency_scores_zero() {
        // The whole point: energy at any frequency other than k contributes nothing.
        let env = SpectralControl::new(3);
        for wrong in [1.0, 2.0, 4.0, 5.0] {
            let j = env.expected_return(4096, |s| (1.0 + (wrong * s).cos()) / 2.0);
            assert!(j.abs() < 1e-9, "frequency {wrong} scored {j}");
        }
    }

    #[test]
    fn a_blind_policy_scores_zero() {
        let env = SpectralControl::new(2);
        assert!(env.expected_return(2048, |_| 0.5).abs() < 1e-12);
    }
}

//! Advantage as a dial, not a reroll.
//!
//! Part IX 1.3 picks out the one D&D mechanic that is genuinely about reshaping a
//! distribution -- advantage, roll two dice and take the better -- and claims Overtone
//! already has a strictly richer version of it:
//!
//! > Grover iterations are advantage.
//!
//! Amplitude amplification rotates probability toward a marked outcome, and after `k`
//! iterations the success probability is `sin^2((2k+1) theta)` with `sin theta =
//! sqrt(M/N)`. That is a closed-form dial with a continuum of settings, where D&D's advantage
//! is a single fixed transform `p -> 1 - (1-p)^2`. [`dnd_equivalent_iterations`] puts a
//! number on the difference instead of leaving it as a claim.
//!
//! The half of the claim that matters more is the second one. Part VI-A T1's souffle problem
//! says too much advantage becomes disadvantage, at a computable step count:
//! [`optimal_iterations`] is where the rotation passes the target, and every iteration after
//! it *lowers* the success probability. A tabletop player cannot roll too many dice; an
//! Overtone player can over-rotate, and will.

use overtone_sim::StateVec;
use std::f64::consts::PI;

/// A Grover instance: `qubits` qubits and a set of marked basis states.
#[derive(Clone, Debug)]
pub struct Advantage {
    qubits: usize,
    marked: Vec<usize>,
}

impl Advantage {
    /// Mark `marked` out of `2^qubits` basis states. Panics if the marked set is empty, has
    /// duplicates, or names a state outside the register.
    pub fn new(qubits: usize, marked: Vec<usize>) -> Advantage {
        assert!(qubits > 0 && qubits <= 20, "qubits out of range");
        assert!(!marked.is_empty(), "nothing is marked");
        let dim = 1usize << qubits;
        assert!(
            marked.iter().all(|&m| m < dim),
            "a marked state is outside the register"
        );
        let mut sorted = marked.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), marked.len(), "the marked set has duplicates");
        Advantage { qubits, marked }
    }

    /// `2^qubits`.
    pub fn dim(&self) -> usize {
        1usize << self.qubits
    }

    /// The success probability before any amplification: `M / N`.
    pub fn base_probability(&self) -> f64 {
        self.marked.len() as f64 / self.dim() as f64
    }

    /// The rotation angle per half-iteration: `theta = asin(sqrt(M/N))`.
    pub fn theta(&self) -> f64 {
        self.base_probability().sqrt().asin()
    }

    /// The closed form Part IX 1.3 quotes: `sin^2((2k+1) theta)`.
    pub fn closed_form(&self, k: usize) -> f64 {
        let t = self.theta();
        (((2 * k + 1) as f64) * t).sin().powi(2)
    }

    /// The uniform superposition the algorithm starts from.
    pub fn uniform(&self) -> StateVec {
        let dim = self.dim();
        let a = 1.0 / (dim as f64).sqrt();
        StateVec::from_amplitudes(vec![a; dim], vec![0.0; dim])
    }

    /// Phase-flip the marked states.
    pub fn oracle(&self, psi: &mut StateVec) {
        for &m in &self.marked {
            psi.re[m] = -psi.re[m];
            psi.im[m] = -psi.im[m];
        }
    }

    /// Inversion about the mean: `2|s><s| - I`.
    pub fn diffuse(&self, psi: &mut StateVec) {
        let dim = self.dim() as f64;
        let mre = psi.re.iter().sum::<f64>() / dim;
        let mim = psi.im.iter().sum::<f64>() / dim;
        for i in 0..psi.re.len() {
            psi.re[i] = 2.0 * mre - psi.re[i];
            psi.im[i] = 2.0 * mim - psi.im[i];
        }
    }

    /// Run `k` amplification steps and return the resulting state.
    pub fn run(&self, k: usize) -> StateVec {
        let mut psi = self.uniform();
        for _ in 0..k {
            self.oracle(&mut psi);
            self.diffuse(&mut psi);
        }
        psi
    }

    /// The total probability sitting on marked states.
    pub fn success(&self, psi: &StateVec) -> f64 {
        self.marked
            .iter()
            .map(|&m| psi.re[m].powi(2) + psi.im[m].powi(2))
            .sum()
    }

    /// The measured success probability after `k` steps -- simulated, not the formula.
    pub fn measured(&self, k: usize) -> f64 {
        self.success(&self.run(k))
    }

    /// The number of iterations that lands closest to the target, `round(pi/(4 theta) - 1/2)`.
    /// Past this, the souffle falls.
    pub fn optimal_iterations(&self) -> usize {
        let raw = PI / (4.0 * self.theta()) - 0.5;
        raw.max(0.0).round() as usize
    }

    /// The whole dial: success probability at every setting from 0 to `k_max`.
    pub fn curve(&self, k_max: usize) -> Vec<f64> {
        (0..=k_max).map(|k| self.closed_form(k)).collect()
    }

    /// True once the dial has been turned past its peak, so that another turn costs
    /// probability. This is Part VI-A T1's souffle, stated as a predicate.
    pub fn over_rotated(&self, k: usize) -> bool {
        k > self.optimal_iterations() && self.closed_form(k) < self.closed_form(k - 1)
    }

    /// How many Grover iterations it takes to beat a D&D advantage roll from the same base
    /// probability, where advantage means `p -> 1 - (1-p)^2`.
    ///
    /// Returns `None` if no setting of the dial beats it, which happens only when the base
    /// probability is already so high that one rotation overshoots.
    pub fn dnd_equivalent_iterations(&self) -> Option<usize> {
        let p = self.base_probability();
        let target = 1.0 - (1.0 - p) * (1.0 - p);
        (1..=self.optimal_iterations()).find(|&k| self.closed_form(k) >= target)
    }
}

/// D&D advantage: roll twice, take the better. A single fixed reshaping of the distribution.
pub fn dnd_advantage(p: f64) -> f64 {
    1.0 - (1.0 - p) * (1.0 - p)
}

/// D&D disadvantage: roll twice, take the worse.
pub fn dnd_disadvantage(p: f64) -> f64 {
    p * p
}

//! Projective measurement with a finite number of shots.
//!
//! This lives in `overtone-sim` because it is a statement about a state vector and an
//! observable and about nothing else -- no policy, no optimiser, no environment. Two very
//! different consumers need it: the barren-plateau race of Part V 5.2, and the shot-budget
//! dial of Part V 4.
//!
//! # Why this module exists before the optimisers do
//!
//! Arrasmith et al. (*Quantum* **5**, 558) do not say gradient-free optimisers stall in a
//! barren plateau as a matter of arithmetic. They say the *cost function differences* are
//! exponentially suppressed, so that "the numbers of shots required in the optimization
//! grows exponentially with the number of qubits". The obstruction is **measurement
//! precision**, not the landscape's shape in exact arithmetic.
//!
//! That distinction has teeth. A demonstration written against `f64` expectation values
//! computed from a state vector gives every optimiser roughly sixteen digits of precision
//! for free, which is far more than the `2^(-1.03 n)` differences at any qubit count a
//! browser can simulate. Run the race that way and Nelder-Mead descends happily, and the
//! demo shows the opposite of the paper it cites. Measured here: see `examples/flatline.rs`.
//!
//! So a cost in this crate is evaluated through a [`Budget`], and the honest setting is
//! [`Budget::Shots`].
//!
//! # What is estimated, and exactly how
//!
//! For a single Pauli term diagonal in the computational basis -- a product of `Z` and `I`,
//! which covers both plateau observables -- one shot returns `+1` or `-1` with
//! `P(+1) = (1 + <O>) / 2`. So `k` successes in `N` shots is exactly binomial, and the
//! estimator is `coeff * (2k/N - 1)`. This is not an approximation of sampling: it *is*
//! sampling, with the multinomial over `2^n` bitstrings collapsed onto the one bit the
//! observable reads. Its variance is `coeff^2 (1 - <O>^2) / N`, and `tests/` checks that
//! against the empirical variance.
//!
//! A non-diagonal observable would need a basis rotation per term and is rejected rather
//! than silently mis-estimated.

use crate::observable::{Observable, Pauli};
use crate::state::StateVec;
use rand::Rng;

/// Shot counts up to this are drawn as an exact sum of Bernoullis.
///
/// Above it the sum is replaced by a Gaussian with the binomial's own mean and variance,
/// rounded back onto the integer lattice. That is an approximation and is named as one: the
/// Berry-Esseen bound puts the CDF error below `0.4 / sqrt(N)`, which at this threshold is
/// under `0.003` -- far smaller than the effects this crate measures, and the alternative is
/// a sweep that does not finish. `tests/` checks the two agree in mean and variance across
/// the boundary.
pub const EXACT_SAMPLING_LIMIT: usize = 65_536;

fn binomial<R: Rng + ?Sized>(n: usize, p: f64, rng: &mut R) -> usize {
    if n <= EXACT_SAMPLING_LIMIT {
        let mut k = 0usize;
        for _ in 0..n {
            if rng.gen_bool(p) {
                k += 1;
            }
        }
        return k;
    }
    let mean = n as f64 * p;
    let sd = (n as f64 * p * (1.0 - p)).sqrt();
    let u1: f64 = rng.gen_range(f64::MIN_POSITIVE..1.0);
    let u2: f64 = rng.gen_range(0.0..1.0);
    let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
    (mean + sd * z).round().clamp(0.0, n as f64) as usize
}

/// How an expectation value is obtained.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Budget {
    /// The state vector's own value, to machine precision. Not available on hardware, and
    /// not a fair setting for the flatline race -- it is offered so that the race can be
    /// run both ways and the difference reported.
    Exact,
    /// `n` projective measurements per expectation value.
    Shots(usize),
}

impl Budget {
    /// Shots consumed by one expectation value.
    ///
    /// `Exact` charges one unit rather than zero -- not because a state-vector read costs a
    /// shot, but because a meter that never advances is a loop that never ends. It means an
    /// `Exact` allowance is denominated in *evaluations*, so a control run is matched to a
    /// shot run on evaluation count, not on shots. Matching them on shots would be
    /// meaningless: the whole claim is that shots are the scarce resource.
    pub fn cost(&self) -> u64 {
        match self {
            Budget::Exact => 1,
            Budget::Shots(n) => *n as u64,
        }
    }
}

/// An observable that can be measured by sampling one bit of the computational basis.
#[derive(Clone, Debug)]
pub struct DiagonalTerm {
    coeff: f64,
    /// Bitmask of the qubits carrying a `Z`.
    mask: usize,
}

impl DiagonalTerm {
    /// Reject anything that is not a single `Z`/`I` product; there is no correct way to
    /// sample it from the computational basis alone.
    pub fn from_observable(o: &Observable) -> Option<DiagonalTerm> {
        if o.terms.len() != 1 {
            return None;
        }
        let t = &o.terms[0];
        let mut mask = 0usize;
        for &(q, p) in &t.factors {
            match p {
                Pauli::Z => mask |= 1 << q,
                Pauli::I => {}
                _ => return None,
            }
        }
        Some(DiagonalTerm {
            coeff: t.coeff,
            mask,
        })
    }

    /// `<O>` exactly, from the amplitudes.
    pub fn exact(&self, psi: &StateVec) -> f64 {
        let mut acc = 0.0;
        for i in 0..psi.dim() {
            let a = psi.amp(i);
            let p = a.re * a.re + a.im * a.im;
            acc += if (i & self.mask).count_ones() & 1 == 0 {
                p
            } else {
                -p
            };
        }
        self.coeff * acc
    }

    /// `<O>` from `shots` projective measurements.
    ///
    /// Drawn as a binomial on the parity bit rather than as a multinomial over `2^n`
    /// bitstrings: the observable cannot distinguish two strings of the same parity, so the
    /// two are the same distribution and this one costs `O(2^n + shots)` instead of
    /// `O(shots log 2^n)`.
    pub fn sample<R: Rng + ?Sized>(&self, psi: &StateVec, shots: usize, rng: &mut R) -> f64 {
        if shots == 0 {
            return 0.0;
        }
        let mean = self.exact(psi) / self.coeff;
        let p_plus = ((1.0 + mean) * 0.5).clamp(0.0, 1.0);
        let k = binomial(shots, p_plus, rng);
        self.coeff * (2.0 * k as f64 / shots as f64 - 1.0)
    }

    /// Estimate under a [`Budget`].
    pub fn estimate<R: Rng + ?Sized>(&self, psi: &StateVec, budget: Budget, rng: &mut R) -> f64 {
        match budget {
            Budget::Exact => self.exact(psi),
            Budget::Shots(n) => self.sample(psi, n, rng),
        }
    }

    /// The standard error of [`Self::sample`], `|coeff| sqrt((1 - <O>^2) / N)`.
    ///
    /// The noise floor an optimiser has to beat. Compare it against `2^(-1.03 n / 2)` --
    /// the scale of the cost differences it is trying to resolve -- to see where the race
    /// is decided.
    pub fn standard_error(&self, psi: &StateVec, shots: usize) -> f64 {
        if shots == 0 {
            return f64::INFINITY;
        }
        let mean = self.exact(psi) / self.coeff;
        self.coeff.abs() * ((1.0 - mean * mean).max(0.0) / shots as f64).sqrt()
    }

    pub fn coeff(&self) -> f64 {
        self.coeff
    }
}

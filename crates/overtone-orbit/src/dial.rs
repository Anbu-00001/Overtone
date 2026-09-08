//! M37: the dial chess does not have.
//!
//! Part VII 6, and the strongest claim in the series:
//!
//! ```text
//! low  dim(g), low  chi  ->  g-sim and MPS both work  ->  positions efficiently evaluable
//! high dim(g), high chi  ->  no efficient classical representation exists  ->  provably hard
//! ```
//!
//! Chess is a fixed point in complexity space. Orbit is a trajectory through it, and the
//! player holds the parameter.
//!
//! # Two independent representations, and the verdict needs both to fail
//!
//! The two axes are not the same claim measured twice. `g`-sim is efficient when `dim(g)` is
//! polynomial, whatever the entanglement; an MPS is efficient when the bond dimension is
//! small, whatever the algebra. A position is efficiently evaluable if **either** works, so
//! it is provably hard only when **both** fail -- and the verdict is written that way round
//! rather than as a single threshold on a single number.
//!
//! # What "provably" is doing in that sentence, and what it is not
//!
//! It means: these two specific classical representations, the ones this repository
//! implements and tests, do not have an efficient description of this state. It does **not**
//! mean no classical algorithm can evaluate the position -- that would be a complexity
//! separation, and nobody has one. The honest claim is bounded and it is the one made here.

use overtone_lie::Algebra;
use overtone_sim::StateVec;

/// Where a position sits on the dial.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Complexity {
    pub dim_g: usize,
    pub num_qubits: usize,
    /// Bond dimension across the half-chain cut, at the given truncation threshold.
    pub chi: usize,
    /// `4^n - 1`, the dimension of `su(2^n)`.
    pub full_algebra: usize,
    /// `2^(n/2)`, the largest bond dimension the cut can carry.
    pub full_chi: usize,
}

impl Complexity {
    /// `g`-sim is efficient when `dim(g)` stays polynomial in the qubit count.
    ///
    /// The cut is `n^4`, the same one the architecture search uses, and for the same reason:
    /// the classification of Wiersema et al. puts polynomial dynamical Lie algebras at `O(n)`
    /// to `O(n^2)` and exponential ones at a constant fraction of `4^n - 1`, so any cut in
    /// the gap separates them.
    pub fn gsim_works(&self) -> bool {
        self.dim_g <= self.num_qubits.pow(4)
    }

    /// An MPS is efficient when the bond dimension stays well below its maximum.
    ///
    /// The cut is half of the maximum: a state needing more than half the available bond
    /// dimension has no compression left worth having.
    pub fn mps_works(&self) -> bool {
        self.chi * 2 <= self.full_chi
    }

    /// Either representation suffices.
    pub fn efficiently_evaluable(&self) -> bool {
        self.gsim_works() || self.mps_works()
    }

    /// Neither does.
    pub fn provably_hard(&self) -> bool {
        !self.efficiently_evaluable()
    }

    /// The live verdict Part VII 6 puts on screen. Part VII 12: do not hide it.
    pub fn verdict(&self) -> String {
        match (self.gsim_works(), self.mps_works()) {
            (true, true) => "g-sim and MPS both work: efficiently evaluable".into(),
            (true, false) => "dim(g) is polynomial: g-sim evaluates this".into(),
            (false, true) => "low bond dimension: an MPS evaluates this".into(),
            (false, false) => {
                "neither g-sim nor MPS has an efficient description of this position".into()
            }
        }
    }
}

/// Measure a position on both axes.
pub fn measure(algebra: &Algebra, state: &StateVec, threshold: f64) -> Complexity {
    let n = state.num_qubits();
    Complexity {
        dim_g: algebra.dim(),
        num_qubits: n,
        chi: overtone_mps::exact_bond_dimension(state, threshold).max(1),
        full_algebra: 4usize.saturating_pow(n as u32).saturating_sub(1),
        full_chi: 1usize << (n / 2),
    }
}

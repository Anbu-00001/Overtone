//! Pauli observables.
//!
//! Everything measured in this project is a real linear combination of Pauli strings, so
//! that is the only observable type the simulator carries. Part I 6.3 needs `Z` on qubit
//! zero for the RAW-PQC Born rule; Part I 6.7 needs both a local `Z_0` and a global
//! `Z (x) Z (x) ... (x) Z` to show the barren-plateau contrast.

use crate::complex::{Mat2, C64};
use crate::state::StateVec;

/// A single-qubit Pauli factor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pauli {
    I,
    X,
    Y,
    Z,
}

impl Pauli {
    /// The 2x2 matrix, row-major.
    pub fn matrix(self) -> Mat2 {
        match self {
            Pauli::I => [C64::ONE, C64::ZERO, C64::ZERO, C64::ONE],
            Pauli::X => [C64::ZERO, C64::ONE, C64::ONE, C64::ZERO],
            Pauli::Y => [
                C64::ZERO,
                C64::new(0.0, -1.0),
                C64::new(0.0, 1.0),
                C64::ZERO,
            ],
            Pauli::Z => [C64::ONE, C64::ZERO, C64::ZERO, C64::new(-1.0, 0.0)],
        }
    }
}

/// A Pauli string with a real coefficient: `coeff * (x)_q P_q`.
#[derive(Clone, Debug, PartialEq)]
pub struct PauliTerm {
    pub coeff: f64,
    /// `(qubit, pauli)` pairs. Qubits absent from this list carry the identity.
    pub factors: Vec<(usize, Pauli)>,
}

impl PauliTerm {
    pub fn new(coeff: f64, factors: Vec<(usize, Pauli)>) -> Self {
        PauliTerm { coeff, factors }
    }

    /// Apply the string, without the coefficient, in place.
    fn apply_string(&self, st: &mut StateVec) {
        for &(q, p) in &self.factors {
            if p != Pauli::I {
                st.apply_1q(q, &p.matrix());
            }
        }
    }
}

/// A Hermitian observable: a real-weighted sum of Pauli strings.
#[derive(Clone, Debug, PartialEq)]
pub struct Observable {
    pub terms: Vec<PauliTerm>,
}

impl Observable {
    pub fn new(terms: Vec<PauliTerm>) -> Self {
        Observable { terms }
    }

    /// `Z` on a single qubit. The RAW-PQC policy observable.
    pub fn z(q: usize) -> Self {
        Observable::new(vec![PauliTerm::new(1.0, vec![(q, Pauli::Z)])])
    }

    /// `Z (x) Z (x) ... (x) Z` across all `n` qubits. The global observable whose gradient
    /// variance collapses exponentially (Part I 6.7).
    pub fn z_global(n: usize) -> Self {
        Observable::new(vec![PauliTerm::new(
            1.0,
            (0..n).map(|q| (q, Pauli::Z)).collect(),
        )])
    }

    /// Write `M|psi>` into `out`. `scratch` is reused across terms so that a repeated call
    /// in a gradient loop does not allocate.
    pub fn apply_into(&self, psi: &StateVec, out: &mut StateVec, scratch: &mut StateVec) {
        out.re.fill(0.0);
        out.im.fill(0.0);
        for term in &self.terms {
            scratch.copy_from(psi);
            term.apply_string(scratch);
            out.add_scaled(scratch, term.coeff);
        }
    }

    /// `M|psi>`, allocating.
    pub fn apply(&self, psi: &StateVec) -> StateVec {
        let mut out = psi.clone();
        let mut scratch = psi.clone();
        self.apply_into(psi, &mut out, &mut scratch);
        out
    }

    /// `<psi|M|psi>`.
    ///
    /// Real by construction: the coefficients are real and Pauli strings are Hermitian.
    pub fn expectation(&self, psi: &StateVec) -> f64 {
        let mut scratch = psi.clone();
        let mut acc = 0.0;
        for term in &self.terms {
            scratch.copy_from(psi);
            term.apply_string(&mut scratch);
            acc += term.coeff * psi.inner_re(&scratch);
        }
        acc
    }
}

//! The quantum Fisher information matrix, and the rank the algebra caps it at.
//!
//! Part III 2's third arrow: `rank(QFIM) <= dim(g)`. The Fisher matrix counts how many
//! directions in state space the circuit can actually move, and the dynamical Lie algebra
//! bounds that count from above, before any parameters are chosen.
//!
//! For a pure state,
//!
//! ```text
//! F_ij = 4 Re( <d_i psi | d_j psi> - <d_i psi | psi> <psi | d_j psi> )
//! ```
//!
//! the Fubini-Study metric in the standard normalisation of Stokes, Izaac, Killoran and
//! Carleo, *Quantum Natural Gradient*, Quantum 4, 269 (2020).

use overtone_sim::{Circuit, StateVec};

use crate::entropy::jacobi_eigenvalues;

/// `d|psi>/d theta_i` for every parameter, by inserting the gate derivative in place.
///
/// A parameter used by several gates contributes the sum of those insertions, each scaled
/// by `d angle / d parameter` -- which is how an encoding gate's `lambda * s` is handled.
pub fn state_derivatives(circuit: &Circuit, params: &[f64]) -> Vec<StateVec> {
    let n = circuit.num_qubits();
    let mut out =
        vec![StateVec::from_amplitudes(vec![0.0; 1 << n], vec![0.0; 1 << n]); params.len()];
    for (g, gate) in circuit.gates().iter().enumerate() {
        let Some((index, scale)) = gate.param_ref() else {
            continue;
        };
        let mut st = StateVec::zero(n);
        for (h, other) in circuit.gates().iter().enumerate() {
            if h == g {
                other.apply_derivative(&mut st, params);
            } else {
                other.apply(&mut st, params);
            }
        }
        out[index].add_scaled(&st, scale);
    }
    out
}

/// The quantum Fisher information matrix, row-major, `P x P`.
pub fn qfim(circuit: &Circuit, params: &[f64]) -> Vec<f64> {
    let p = params.len();
    let psi = circuit.run(params);
    let d = state_derivatives(circuit, params);
    let overlaps: Vec<overtone_sim::C64> = d.iter().map(|di| di.inner(&psi)).collect();
    let mut f = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..p {
            let a = d[i].inner(&d[j]);
            // <d_i|psi><psi|d_j> = overlaps[i] * conj(overlaps[j])
            let b_re = overlaps[i].re * overlaps[j].re + overlaps[i].im * overlaps[j].im;
            f[i * p + j] = 4.0 * (a.re - b_re);
        }
    }
    f
}

/// Numerical rank: eigenvalues above `tol` times the largest.
pub fn rank(matrix: &[f64], p: usize, tol: f64) -> usize {
    let mut a = matrix.to_vec();
    let vals = jacobi_eigenvalues(&mut a, p);
    let max = vals.iter().cloned().fold(0.0_f64, |m, v| m.max(v.abs()));
    if max <= 0.0 {
        return 0;
    }
    vals.iter().filter(|v| v.abs() > tol * max).count()
}

/// Eigenvalues of the QFIM, descending. The C4 panel of Part III 6 plots these: the rank
/// saturates and the null space shows up as a cliff.
pub fn qfim_spectrum(circuit: &Circuit, params: &[f64]) -> Vec<f64> {
    let p = params.len();
    let mut f = qfim(circuit, params);
    let mut vals = jacobi_eigenvalues(&mut f, p);
    vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
    vals
}

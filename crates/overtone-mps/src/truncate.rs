//! Bond spectra and truncation to a fixed bond dimension.
//!
//! Part III 5's instrument. A matrix product state of bond dimension `chi` is exactly a
//! state whose every bipartition has Schmidt rank at most `chi`, so truncating a state
//! vector to bond dimension `chi` is projecting each cut onto the top `chi` eigenvectors of
//! its reduced density matrix.
//!
//! One left-to-right sweep is used, not a variational optimum. That makes the result an
//! upper bound on the error an optimally compressed MPS would incur, so an effective `chi`
//! reported here is never smaller than the truth -- the direction that cannot flatter the
//! quantum agent.

use overtone_sim::StateVec;

use crate::linalg::hermitian_eigh;

/// The bipartition matrix at `cut`: rows index qubits `0..cut`, columns the rest.
fn bipartition(state: &StateVec, cut: usize) -> (Vec<f64>, Vec<f64>, usize, usize) {
    let n = state.num_qubits();
    assert!(cut >= 1 && cut < n);
    let rows = 1usize << cut;
    let cols = 1usize << (n - cut);
    let mut re = vec![0.0; rows * cols];
    let mut im = vec![0.0; rows * cols];
    for i in 0..state.dim() {
        let r = i & (rows - 1);
        let c = i >> cut;
        re[r * cols + c] = state.re[i];
        im[r * cols + c] = state.im[i];
    }
    (re, im, rows, cols)
}

/// Squared Schmidt coefficients at `cut`, descending. Their count above a threshold is the
/// Schmidt rank, and the bond dimension an exact MPS would need there.
pub fn bond_spectrum(state: &StateVec, cut: usize) -> Vec<f64> {
    let (re, im, rows, cols) = bipartition(state, cut);
    let (rho_re, rho_im, d) = reduced(&re, &im, rows, cols);
    let (vals, _, _) = hermitian_eigh(&rho_re, &rho_im, d);
    vals.into_iter().map(|v| v.max(0.0)).collect()
}

/// The reduced density matrix of whichever side is smaller.
fn reduced(re: &[f64], im: &[f64], rows: usize, cols: usize) -> (Vec<f64>, Vec<f64>, usize) {
    if rows <= cols {
        let d = rows;
        let mut rr = vec![0.0; d * d];
        let mut ri = vec![0.0; d * d];
        for a in 0..d {
            for b in 0..d {
                let (mut sr, mut si) = (0.0, 0.0);
                for c in 0..cols {
                    let (ar, ai) = (re[a * cols + c], im[a * cols + c]);
                    let (br, bi) = (re[b * cols + c], im[b * cols + c]);
                    sr += ar * br + ai * bi;
                    si += ai * br - ar * bi;
                }
                rr[a * d + b] = sr;
                ri[a * d + b] = si;
            }
        }
        (rr, ri, d)
    } else {
        let d = cols;
        let mut rr = vec![0.0; d * d];
        let mut ri = vec![0.0; d * d];
        for a in 0..d {
            for b in 0..d {
                let (mut sr, mut si) = (0.0, 0.0);
                for r in 0..rows {
                    let (ar, ai) = (re[r * cols + a], im[r * cols + a]);
                    let (br, bi) = (re[r * cols + b], im[r * cols + b]);
                    // conj(M[r,a]) M[r,b]
                    sr += ar * br + ai * bi;
                    si += ar * bi - ai * br;
                }
                rr[a * d + b] = sr;
                ri[a * d + b] = si;
            }
        }
        (rr, ri, d)
    }
}

/// Project one cut onto its leading `chi` Schmidt vectors, in place.
fn project_cut(state: &mut StateVec, cut: usize, chi: usize) {
    let n = state.num_qubits();
    let (re, im, rows, cols) = bipartition(state, cut);
    let (rho_re, rho_im, d) = reduced(&re, &im, rows, cols);
    if chi >= d {
        return;
    }
    let (_, vr, vi) = hermitian_eigh(&rho_re, &rho_im, d);

    let mut out_re = vec![0.0; rows * cols];
    let mut out_im = vec![0.0; rows * cols];
    if rows <= cols {
        // M' = sum_{i<chi} u_i (u_i^dagger M)
        for i in 0..chi {
            for c in 0..cols {
                // t = u_i^dagger M[:, c]
                let (mut tr, mut ti) = (0.0, 0.0);
                for r in 0..d {
                    let (ur, ui) = (vr[r * d + i], vi[r * d + i]);
                    let (mr, mi) = (re[r * cols + c], im[r * cols + c]);
                    tr += ur * mr + ui * mi;
                    ti += ur * mi - ui * mr;
                }
                for r in 0..d {
                    let (ur, ui) = (vr[r * d + i], vi[r * d + i]);
                    out_re[r * cols + c] += ur * tr - ui * ti;
                    out_im[r * cols + c] += ur * ti + ui * tr;
                }
            }
        }
    } else {
        // M' = sum_{i<chi} (M w_i) w_i^dagger
        for i in 0..chi {
            for r in 0..rows {
                let (mut tr, mut ti) = (0.0, 0.0);
                for c in 0..d {
                    let (wr, wi) = (vr[c * d + i], vi[c * d + i]);
                    let (mr, mi) = (re[r * cols + c], im[r * cols + c]);
                    tr += mr * wr - mi * wi;
                    ti += mr * wi + mi * wr;
                }
                for c in 0..d {
                    let (wr, wi) = (vr[c * d + i], vi[c * d + i]);
                    out_re[r * cols + c] += tr * wr + ti * wi;
                    out_im[r * cols + c] += ti * wr - tr * wi;
                }
            }
        }
    }

    for i in 0..state.dim() {
        let r = i & (rows - 1);
        let c = i >> cut;
        state.re[i] = out_re[r * cols + c];
        state.im[i] = out_im[r * cols + c];
    }
    let _ = n;
}

/// Compress a state to bond dimension `chi` at every cut, then renormalise.
///
/// Returns the compressed state and the fidelity `|<psi|psi_chi>|^2` against the original.
pub fn compress(state: &StateVec, chi: usize) -> (StateVec, f64) {
    let n = state.num_qubits();
    let mut out = state.clone();
    for cut in 1..n {
        project_cut(&mut out, cut, chi);
    }
    let norm = out.norm_sqr().sqrt();
    if norm > 1e-300 {
        out.scale(1.0 / norm);
    }
    let overlap = state.inner(&out);
    (out, overlap.re * overlap.re + overlap.im * overlap.im)
}

/// The exact bond dimension a state needs: the largest Schmidt rank over all cuts, counting
/// coefficients above `threshold` of the total weight.
pub fn exact_bond_dimension(state: &StateVec, threshold: f64) -> usize {
    let n = state.num_qubits();
    (1..n)
        .map(|cut| {
            bond_spectrum(state, cut)
                .into_iter()
                .filter(|v| *v > threshold)
                .count()
        })
        .max()
        .unwrap_or(1)
}

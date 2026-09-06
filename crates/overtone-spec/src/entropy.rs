//! Entanglement entropy across the half-chain bipartition (Part I 6.6).
//!
//! `S = -Tr(rho_A log rho_A)` where `rho_A` traces out the second half of the register.
//!
//! # Getting the eigenvalues without a linear-algebra dependency
//!
//! Reshaping the state vector gives a matrix `M` with `M[i_A][i_B] = psi[i_A + 2^a i_B]`,
//! and `rho_A = M M^dag` is Hermitian and positive semidefinite with unit trace. Its
//! eigenvalues are the Schmidt coefficients squared.
//!
//! A complex Hermitian eigensolver is more machinery than this needs. Instead, a Hermitian
//! `H = R + iI` (with `R` symmetric and `I` antisymmetric) embeds into the real symmetric
//! matrix
//!
//! ```text
//! T = [ R  -I ]
//!     [ I   R ]
//! ```
//!
//! whose spectrum is that of `H` with every eigenvalue doubled. So a plain cyclic Jacobi
//! sweep on a real symmetric matrix is sufficient, and Jacobi is chosen over anything
//! faster because it is unconditionally stable for symmetric matrices and short enough to
//! read.
//!
//! Cost is `O((2^a)^3)`. Keep `n <= 16`.

use overtone_sim::StateVec;

/// The result of a bipartition analysis.
#[derive(Clone, Debug, PartialEq)]
pub struct Bipartition {
    /// Eigenvalues of `rho_A`, descending. These sum to 1.
    pub spectrum: Vec<f64>,
    /// Von Neumann entropy in nats.
    pub entropy: f64,
    /// `Tr(rho_A^2)`. One for a product state, `1/d` for a maximally entangled one.
    pub purity: f64,
    /// Number of qubits in subsystem A.
    pub cut: usize,
}

/// Eigenvalues below this are treated as zero when forming `-sum p log p`.
///
/// Jacobi returns tiny negatives for eigenvalues that are mathematically zero, and
/// `log` of those is not merely inaccurate but `NaN`. The cutoff is well above the
/// rounding floor and well below any physically meaningful Schmidt weight.
const EIGENVALUE_FLOOR: f64 = 1e-14;

/// Analyse the bipartition that puts the lowest `cut` qubits in subsystem A.
pub fn bipartition(state: &StateVec, cut: usize) -> Bipartition {
    let n = state.num_qubits();
    assert!(
        cut > 0 && cut < n,
        "cut {cut} must split a {n}-qubit register"
    );

    let da = 1usize << cut;
    let db = 1usize << (n - cut);

    // rho_A[i][j] = sum_b M[i][b] conj(M[j][b]), with M[i][b] = psi[i + da*b].
    let mut re = vec![0.0; da * da];
    let mut im = vec![0.0; da * da];
    for i in 0..da {
        for j in 0..da {
            let (mut acc_re, mut acc_im) = (0.0, 0.0);
            for b in 0..db {
                let x = i + da * b;
                let y = j + da * b;
                // psi_x * conj(psi_y)
                acc_re += state.re[x] * state.re[y] + state.im[x] * state.im[y];
                acc_im += state.im[x] * state.re[y] - state.re[x] * state.im[y];
            }
            re[i * da + j] = acc_re;
            im[i * da + j] = acc_im;
        }
    }

    let spectrum = hermitian_eigenvalues(&re, &im, da);

    let entropy = spectrum
        .iter()
        .filter(|p| **p > EIGENVALUE_FLOOR)
        .map(|p| -p * p.ln())
        .sum();
    let purity = spectrum.iter().map(|p| p * p).sum();

    Bipartition {
        spectrum,
        entropy,
        purity,
        cut,
    }
}

/// Half-chain entropy. For odd `n` the larger half goes to B.
pub fn half_chain_entropy(state: &StateVec) -> f64 {
    bipartition(state, state.num_qubits() / 2).entropy
}

/// Eigenvalues of a Hermitian matrix given as separate real and imaginary planes,
/// descending.
///
/// Uses the real symmetric embedding described in the module docs, so each eigenvalue of
/// the Hermitian matrix appears twice in the embedded spectrum; every second one is kept.
pub fn hermitian_eigenvalues(re: &[f64], im: &[f64], d: usize) -> Vec<f64> {
    let m = 2 * d;
    let mut t = vec![0.0; m * m];
    for i in 0..d {
        for j in 0..d {
            let r = re[i * d + j];
            let s = im[i * d + j];
            t[i * m + j] = r; // R
            t[(i + d) * m + (j + d)] = r; // R
            t[i * m + (j + d)] = -s; // -I
            t[(i + d) * m + j] = s; // I
        }
    }

    let mut eigen = jacobi_eigenvalues(&mut t, m);
    eigen.sort_by(|a, b| b.partial_cmp(a).expect("no NaN eigenvalues"));
    // Every eigenvalue is doubled by the embedding.
    eigen.into_iter().step_by(2).collect()
}

/// Cyclic Jacobi eigenvalue iteration for a real symmetric matrix, row-major, size `n`.
///
/// Returns the eigenvalues, unsorted. The matrix is overwritten.
pub fn jacobi_eigenvalues(a: &mut [f64], n: usize) -> Vec<f64> {
    const MAX_SWEEPS: usize = 100;
    const TOL: f64 = 1e-15;

    for _ in 0..MAX_SWEEPS {
        // Frobenius norm of the strictly upper triangle.
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[p * n + q] * a[p * n + q];
            }
        }
        if off.sqrt() <= TOL {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() <= TOL {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];

                // The stable form: solve for tan of the rotation without cancellation.
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;

                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
            }
        }
    }

    (0..n).map(|i| a[i * n + i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use overtone_sim::random::{random_params, rng};
    use overtone_sim::{Angle, Circuit};
    use rand::Rng;

    const LN2: f64 = std::f64::consts::LN_2;

    #[test]
    fn a_product_state_has_zero_entropy() {
        let mut c = Circuit::new(4);
        for q in 0..4 {
            c.ry(q, Angle::param(q));
        }
        let st = c.run(&[0.3, 1.1, -0.7, 2.2]);
        let b = bipartition(&st, 2);
        assert!(b.entropy.abs() < 1e-12, "entropy {}", b.entropy);
        assert!((b.purity - 1.0).abs() < 1e-12, "purity {}", b.purity);
    }

    #[test]
    fn a_bell_state_has_entropy_ln_two() {
        let mut c = Circuit::new(2);
        c.h(0);
        c.cnot(0, 1);
        let st = c.run(&[]);
        let b = bipartition(&st, 1);
        assert!(
            (b.entropy - LN2).abs() < 1e-12,
            "entropy {} vs ln2 {LN2}",
            b.entropy
        );
        assert!((b.purity - 0.5).abs() < 1e-12, "purity {}", b.purity);
        assert!((b.spectrum[0] - 0.5).abs() < 1e-12);
        assert!((b.spectrum[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn a_ghz_state_has_entropy_ln_two_at_every_cut() {
        // GHZ is maximally entangled between any bipartition but has only two Schmidt
        // coefficients, so the entropy stays at ln 2 rather than growing with the cut.
        let n = 6;
        let mut c = Circuit::new(n);
        c.h(0);
        for q in 1..n {
            c.cnot(0, q);
        }
        let st = c.run(&[]);
        for cut in 1..n {
            let b = bipartition(&st, cut);
            assert!(
                (b.entropy - LN2).abs() < 1e-12,
                "cut {cut}: entropy {}",
                b.entropy
            );
        }
    }

    #[test]
    fn a_maximally_entangled_pair_of_blocks_saturates() {
        // Two Bell pairs across the cut give 2 ln 2, the maximum for a two-qubit subsystem.
        let mut c = Circuit::new(4);
        c.h(0);
        c.cnot(0, 2);
        c.h(1);
        c.cnot(1, 3);
        let st = c.run(&[]);
        let b = bipartition(&st, 2);
        assert!(
            (b.entropy - 2.0 * LN2).abs() < 1e-12,
            "entropy {} vs 2 ln2 {}",
            b.entropy,
            2.0 * LN2
        );
    }

    #[test]
    fn the_spectrum_is_a_probability_distribution() {
        for seed in 0..10u64 {
            let n = 6;
            let mut c = Circuit::new(n);
            let mut idx = 0;
            for _ in 0..3 {
                for q in 0..n {
                    c.ry(q, Angle::param(idx));
                    idx += 1;
                    c.rz(q, Angle::param(idx));
                    idx += 1;
                }
                for q in 0..n {
                    c.cz(q, (q + 1) % n);
                }
            }
            let st = c.run(&random_params(seed, idx));
            let b = bipartition(&st, 3);

            let total: f64 = b.spectrum.iter().sum();
            assert!((total - 1.0).abs() < 1e-10, "seed={seed}: trace {total}");
            for p in &b.spectrum {
                assert!(*p > -1e-12, "seed={seed}: negative eigenvalue {p}");
                assert!(*p < 1.0 + 1e-12, "seed={seed}: eigenvalue {p} exceeds 1");
            }
            // Entropy is bounded by the log of the smaller subsystem's dimension.
            assert!(b.entropy >= -1e-12 && b.entropy <= 3.0 * LN2 + 1e-10);
        }
    }

    #[test]
    fn entropy_from_either_side_of_the_cut_agrees() {
        // For a pure state S(rho_A) = S(rho_B), where B is the *complement* of A. Note this
        // is not the same as comparing bipartition(cut) with bipartition(n - cut): those
        // are both taken over the lowest qubits and describe different subsystems, which
        // only coincide for permutation-symmetric states.
        //
        // rho_B is built here from scratch, so this genuinely tests the reshaping index
        // arithmetic rather than running the same code path twice.
        let n = 6;
        let mut c = Circuit::new(n);
        let mut idx = 0;
        for _ in 0..4 {
            for q in 0..n {
                c.ry(q, Angle::param(idx));
                idx += 1;
            }
            for q in 0..n {
                c.cnot(q, (q + 1) % n);
            }
        }
        let st = c.run(&random_params(42, idx));

        for cut in 1..n {
            let da = 1usize << cut;
            let db = 1usize << (n - cut);

            // rho_B[b][b'] = sum_a psi[a + da*b] conj(psi[a + da*b'])
            let mut re = vec![0.0; db * db];
            let mut im = vec![0.0; db * db];
            for b in 0..db {
                for bp in 0..db {
                    let (mut ar, mut ai) = (0.0, 0.0);
                    for a in 0..da {
                        let x = a + da * b;
                        let y = a + da * bp;
                        ar += st.re[x] * st.re[y] + st.im[x] * st.im[y];
                        ai += st.im[x] * st.re[y] - st.re[x] * st.im[y];
                    }
                    re[b * db + bp] = ar;
                    im[b * db + bp] = ai;
                }
            }
            let spectrum_b = hermitian_eigenvalues(&re, &im, db);
            let entropy_b: f64 = spectrum_b
                .iter()
                .filter(|p| **p > EIGENVALUE_FLOOR)
                .map(|p| -p * p.ln())
                .sum();

            let a = bipartition(&st, cut).entropy;
            assert!(
                (a - entropy_b).abs() < 1e-10,
                "cut {cut}: {a} vs {entropy_b}"
            );
        }
    }

    #[test]
    fn jacobi_matches_a_known_symmetric_spectrum() {
        // Diagonal plus a rank-one perturbation with an eigenvalue that is easy to state:
        // the all-ones matrix of size n has spectrum {n, 0, 0, ...}.
        for n in [2usize, 4, 8, 16] {
            let mut a = vec![1.0; n * n];
            let mut eigen = jacobi_eigenvalues(&mut a, n);
            eigen.sort_by(|x, y| y.partial_cmp(x).unwrap());
            assert!((eigen[0] - n as f64).abs() < 1e-10, "n={n}: {}", eigen[0]);
            for e in eigen.iter().skip(1) {
                assert!(e.abs() < 1e-10, "n={n}: residual eigenvalue {e}");
            }
        }
    }

    #[test]
    fn hermitian_embedding_handles_a_genuinely_complex_matrix() {
        // The 2x2 Hermitian [[1, -i], [i, 1]] has eigenvalues 0 and 2. If the embedding's
        // sign convention were wrong this would still be symmetric, but wrong.
        let re = [1.0, 0.0, 0.0, 1.0];
        let im = [0.0, -1.0, 1.0, 0.0];
        let eigen = hermitian_eigenvalues(&re, &im, 2);
        assert_eq!(eigen.len(), 2);
        assert!((eigen[0] - 2.0).abs() < 1e-12, "{:?}", eigen);
        assert!(eigen[1].abs() < 1e-12, "{:?}", eigen);
    }

    #[test]
    fn entropy_matches_the_analytic_two_qubit_form() {
        // For a pure two-qubit state the Schmidt weights are (1 +- sqrt(1 - C^2))/2 with
        // C = 2|a d - b c| the concurrence. An independent formula, not a second run of the
        // same code path.
        let mut r = rng(9);
        for _ in 0..20 {
            let mut amps: Vec<(f64, f64)> = (0..4)
                .map(|_| (r.gen_range(-1.0..1.0), r.gen_range(-1.0..1.0)))
                .collect();
            let norm: f64 = amps.iter().map(|(x, y)| x * x + y * y).sum::<f64>().sqrt();
            for a in &mut amps {
                a.0 /= norm;
                a.1 /= norm;
            }
            let st = overtone_sim::StateVec::from_amplitudes(
                amps.iter().map(|a| a.0).collect(),
                amps.iter().map(|a| a.1).collect(),
            );

            // ad - bc for amplitudes indexed |00>, |01>, |10>, |11>.
            let (a, b, cc, d) = (amps[0], amps[1], amps[2], amps[3]);
            let det_re = a.0 * d.0 - a.1 * d.1 - (b.0 * cc.0 - b.1 * cc.1);
            let det_im = a.0 * d.1 + a.1 * d.0 - (b.0 * cc.1 + b.1 * cc.0);
            let concurrence = 2.0 * (det_re * det_re + det_im * det_im).sqrt();

            let root = (1.0 - concurrence * concurrence).max(0.0).sqrt();
            let (p0, p1) = ((1.0 + root) / 2.0, (1.0 - root) / 2.0);
            let expected: f64 = [p0, p1]
                .iter()
                .filter(|p| **p > 1e-14)
                .map(|p| -p * p.ln())
                .sum();

            let got = bipartition(&st, 1).entropy;
            assert!((got - expected).abs() < 1e-10, "{got} vs {expected}");
        }
    }
}

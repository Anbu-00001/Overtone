//! The prediction report: what the algebra says before a single gradient step.
//!
//! Part III 2's chain, implemented literally.
//!
//! ```text
//! generators -[ Lie closure ]-> dim(g) -> Var[loss], rank(QFIM) <= dim(g), M_c
//! ```
//!
//! The variance is Theorem 1 of Ragone, Bakalov, Sauvage, Kemper, Ortiz Marrero, Larocca
//! and Cerezo, *A Lie algebraic theory of barren plateaus for deep parameterized quantum
//! circuits*, Nat. Commun. 15, 7172 (2024):
//!
//! ```text
//! Var_theta[ loss ] = sum_j P_{g_j}(rho) P_{g_j}(O) / dim(g_j)
//! ```
//!
//! over the simple components of `g`, with the `g`-purity
//! `P_g(H) = sum_j |tr[B_j H]|^2` in an orthonormal basis of the complexified algebra.
//! The theorem assumes the circuit is deep enough to be a 2-design over `exp(g)`, and that
//! the observable or the state lies in the algebra. Both assumptions are checkable and
//! `Prediction` records when they fail rather than quietly reporting a number anyway.

use crate::closure::Algebra;
use crate::pauli::PauliString;

/// Decompose the algebra into blocks that commute elementwise.
///
/// Union-find over the non-commuting graph: isolated elements are central, and each
/// connected component is a subalgebra commuting with every other one.
///
/// **This is the Pauli-visible decomposition, which can be coarser than the abstract one.**
/// A direct sum whose summands are not each spanned by Pauli strings is invisible here: the
/// Heisenberg chain on an even number of sites is `su(2^(n-2))^(+4)` in the classification
/// of Wiersema et al., but its 60 Pauli strings at `n = 4` form a single connected block,
/// because a Pauli string there is a sum of pieces from several summands rather than living
/// in one. Reported blocks are therefore a lower bound on the number of simple ideals. The
/// variance in [`Prediction`] uses the blocks it can see, and the measurement in
/// `overtone-gsim/examples/theorem_one.rs` shows that staying within 1% for exactly this
/// case -- but the two are not the same object and the difference is not a rounding error.
pub fn components(algebra: &Algebra) -> Vec<Vec<usize>> {
    let d = algebra.dim();
    let basis = algebra.basis();
    let mut parent: Vec<usize> = (0..d).collect();
    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    for i in 0..d {
        for j in (i + 1)..d {
            if basis[i].anticommutes(&basis[j]) {
                let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                if a != b {
                    parent[a] = b;
                }
            }
        }
    }
    let mut groups: std::collections::HashMap<usize, Vec<usize>> = Default::default();
    for i in 0..d {
        let r = find(&mut parent, i);
        groups.entry(r).or_default().push(i);
    }
    let mut out: Vec<Vec<usize>> = groups.into_values().collect();
    out.sort_by_key(|g| (std::cmp::Reverse(g.len()), g[0]));
    out
}

/// Choose an observable Theorem 1 can speak about.
///
/// The theorem needs `O` inside the algebra. `Z_0` -- what Part I's Born rule reads -- is
/// often outside it: the transverse-field Ising algebra contains `X_q` and `Z_qZ_{q+1}` but
/// not `Z_q`. Rather than report "not applicable" and leave the panel blank, fall back to
/// the first generator, which is in the algebra by construction. The caller is expected to
/// say which observable was used, because swapping it silently would be worse than a blank.
pub fn pick_observable(algebra: &Algebra, preferred: &PauliString) -> PauliString {
    if algebra.contains(preferred) {
        *preferred
    } else {
        algebra.basis()[0]
    }
}

/// How `dim(g)` grows with the qubit count. Corollary IV.1 of Wiersema et al.: for 2-local
/// translation-invariant chains it is always one of these three.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scaling {
    /// `dim(g)` at most a small multiple of `n`.
    Linear,
    /// Polynomial in `n`. Trainable, and — Part III 4 — classically simulable.
    Polynomial,
    /// A constant fraction of `4^n`. Barren plateau.
    Exponential,
}

/// The algebraic prediction, made before training.
#[derive(Clone, Debug)]
pub struct Prediction {
    pub num_qubits: usize,
    pub dim_g: usize,
    pub truncated: bool,
    /// `4^n - 1`, the dimension of `su(2^n)`, for scale.
    pub dim_su: f64,
    pub scaling: Scaling,
    /// Sizes of the simple ideals, largest first, centre excluded.
    pub ideals: Vec<usize>,
    /// Number of central (mutually commuting, globally commuting) elements.
    pub centre: usize,
    /// Theorem 1's variance of the loss, when the theorem applies.
    pub loss_variance: Option<f64>,
    /// `rank(QFIM) <= dim(g)`: how many directions in state space the circuit can move.
    pub qfim_rank_bound: usize,
    /// Overparameterisation sets in by `M ~ dim(g)` parameters (Larocca et al. 2023).
    pub overparameterisation_bound: usize,
    /// Reasons the numbers above may not be what the reader assumes.
    pub caveats: Vec<String>,
    /// Set when the circuit contains gates that are not one-parameter subgroups of
    /// `exp(g)`. Theorem 1 then has no hypothesis to stand on and the variance above is a
    /// statement about a different circuit than the one the user has.
    pub leaves_the_group: bool,
}

impl Prediction {
    /// Build the report for an algebra measured with a single Pauli observable, starting
    /// from `|0...0>`.
    pub fn new(algebra: &Algebra, observable: &PauliString) -> Self {
        let n = algebra.num_qubits();
        let d = algebra.dim();
        let dim_su = 4f64.powi(n as i32) - 1.0;
        let comps = components(algebra);
        let centre = comps.iter().filter(|c| c.len() == 1).count();
        let ideals: Vec<usize> = comps.iter().map(|c| c.len()).filter(|&l| l > 1).collect();

        let mut caveats = Vec::new();
        if algebra.truncated() {
            caveats.push(format!(
                "closure abandoned at {d} elements; dim(g) is a lower bound, not the dimension"
            ));
        }

        // Theorem 1 needs the observable inside the algebra. Otherwise the loss is not a
        // function on exp(g) at all and the formula has nothing to say.
        let loss_variance = match algebra.index_of(observable) {
            None => {
                caveats.push(format!(
                    "observable {} is not in g; Theorem 1 does not apply",
                    observable.render(n)
                ));
                None
            }
            Some(oi) => {
                let comp = comps
                    .iter()
                    .find(|c| c.contains(&oi))
                    .expect("every index lies in a component");
                if comp.len() == 1 {
                    caveats.push(
                        "observable lies in the centre of g; the loss has zero variance and a \
                         nonzero mean"
                            .into(),
                    );
                    Some(0.0)
                } else {
                    // P_g(rho) P_g(O) / dim(g) with rho = |0><0| and O a single Pauli in
                    // the component: (Z-type strings in the component) / dim(component).
                    let z_type = comp.iter().filter(|&&i| algebra.basis()[i].x == 0).count() as f64;
                    Some(z_type / comp.len() as f64)
                }
            }
        };

        let scaling = if (d as f64) >= 0.25 * dim_su {
            Scaling::Exponential
        } else if d <= 4 * n + 4 {
            Scaling::Linear
        } else {
            Scaling::Polynomial
        };

        Prediction {
            num_qubits: n,
            dim_g: d,
            truncated: algebra.truncated(),
            dim_su,
            scaling,
            ideals,
            centre,
            loss_variance,
            qfim_rank_bound: d,
            overparameterisation_bound: d,
            caveats,
            leaves_the_group: false,
        }
    }

    /// Record that the circuit interleaves fixed entangling gates.
    ///
    /// This is the difference between an ansatz and its algebra, and it is load-bearing.
    /// A fixed CZ layer is a Clifford, not a one-parameter subgroup: the circuit's unitary
    /// is not in `exp(g)` for the `g` generated by its trainable rotations, and Theorem 1
    /// says nothing about it. Our own Part I ansatz is exactly this case -- `dim(g) = 3n`,
    /// polynomial, "trainable" -- while Part I 6.7 measures its global-observable gradient
    /// variance falling like `2^(-1.03 n)`. The algebra is not wrong; it is being asked a
    /// question about a circuit it does not describe. See Diaz, Garcia-Martin, Kazi,
    /// Larocca and Cerezo, *Showcasing a barren plateau theory beyond the dynamical Lie
    /// algebra*, arXiv:2310.11505.
    pub fn note_fixed_entanglers(&mut self, count: usize) {
        if count == 0 {
            return;
        }
        self.leaves_the_group = true;
        self.caveats.push(format!(
            "{count} fixed entangling gates are not one-parameter subgroups of exp(g), so \
             this circuit is not in exp(g) and Theorem 1 does not apply to it. dim(g) below \
             is the algebra of the trainable generators alone."
        ));
    }

    /// One line, the verdict. Part III 4: the caveat travels with the claim.
    pub fn verdict(&self) -> String {
        if self.leaves_the_group {
            return format!(
                "dim(g) = {} of {:.0} for the trainable generators, but this circuit's fixed \
                 entanglers put it outside exp(g). No trainability claim follows. Measure it.",
                self.dim_g, self.dim_su
            );
        }
        match self.scaling {
            Scaling::Exponential => format!(
                "dim(g) = {} of {:.0} — exponential: barren plateau, not trainable at scale",
                self.dim_g, self.dim_su
            ),
            Scaling::Polynomial | Scaling::Linear => format!(
                "dim(g) = {} of {:.0} — polynomial: trainable, and therefore also \
                 classically simulable by g-sim",
                self.dim_g, self.dim_su
            ),
        }
    }
}

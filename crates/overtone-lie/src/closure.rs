//! The Lie closure, and what it is a closure of.
//!
//! Part III 2: take the circuit's gate generators, close them under nested commutators
//! until the set stops growing, and the dimension of what you get predicts whether the
//! circuit will train — before a single gradient step.
//!
//! The algorithm is Algorithm 1 of Wiersema et al. (arXiv:2309.05690) run in rounds, so
//! that the growth curve `|S|` against round is a real quantity rather than an artifact of
//! iteration order. That curve is what Part III 6 C1 animates, and its plateau *is* the
//! closure.

use std::collections::HashMap;

use crate::pauli::PauliString;

/// A dynamical Lie algebra: a set of Pauli strings closed under commutation.
///
/// The dimension is the number of strings. That is exact rather than approximate: a
/// commutator of Pauli strings is a single Pauli string, so the span of a commutator-closed
/// set of distinct strings has them as a basis.
#[derive(Clone, Debug)]
pub struct Algebra {
    n: usize,
    basis: Vec<PauliString>,
    index: HashMap<PauliString, usize>,
    growth: Vec<usize>,
    parents: Vec<(u32, u32)>,
    truncated: bool,
}

impl Algebra {
    /// Qubit count the algebra was built on.
    pub fn num_qubits(&self) -> usize {
        self.n
    }

    /// `dim(g)`. The integer the whole of Part III hangs on.
    pub fn dim(&self) -> usize {
        self.basis.len()
    }

    pub fn basis(&self) -> &[PauliString] {
        &self.basis
    }

    /// `|S|` after each round of Algorithm 1, starting from the generating set.
    pub fn growth(&self) -> &[usize] {
        &self.growth
    }

    /// For each basis element, the pair whose commutator produced it.
    ///
    /// Generators point at themselves. This is the edge set the C1 animation of Part III 6
    /// draws: new strings appearing and connecting to the two they came from.
    pub fn parents(&self) -> &[(u32, u32)] {
        &self.parents
    }

    /// True when the closure hit its element limit and was abandoned. A truncated algebra
    /// is not an algebra; every consumer has to check this rather than read `dim`.
    pub fn truncated(&self) -> bool {
        self.truncated
    }

    pub fn index_of(&self, p: &PauliString) -> Option<usize> {
        self.index.get(p).copied()
    }

    pub fn contains(&self, p: &PauliString) -> bool {
        self.index.contains_key(p)
    }

    /// For a generator `g`, the involution it induces on the basis.
    ///
    /// Entry `alpha` is `None` if `g` commutes with basis element `alpha`, and otherwise
    /// `(gamma, sign)` with `[b_alpha, g] = 2i * sign * b_gamma`. This is the whole of the
    /// g-sim gate: a set of independent 2D rotations, `O(dim g)` rather than `O(dim(g)^2)`.
    pub fn pairing(&self, g: &PauliString) -> Vec<Option<(usize, i8)>> {
        self.basis
            .iter()
            .map(|b| {
                b.commutator(g).map(|(prod, sign)| {
                    let j = self
                        .index_of(&prod)
                        .expect("algebra not closed under commutation");
                    (j, sign)
                })
            })
            .collect()
    }

    /// The `g`-purity of the initial state `|0...0>`, as defined in Ragone et al. Theorem 1.
    ///
    /// `P_g(rho) = sum_j |tr[B_j rho]|^2` over an orthonormal basis of the complexified
    /// algebra. With `B_j = P_j / sqrt(2^n)` and `rho = |0><0|` that is the count of Z-type
    /// strings in the algebra divided by `2^n`.
    pub fn purity_of_zero_state(&self) -> f64 {
        let z_type = self.basis.iter().filter(|p| p.x == 0).count() as f64;
        z_type / (1u64 << self.n) as f64
    }

    /// The `g`-purity of a single Pauli observable that lies in the algebra: `2^n`.
    /// Returns `None` for an observable outside the algebra, where Theorem 1 does not apply.
    pub fn purity_of_pauli_observable(&self, o: &PauliString) -> Option<f64> {
        self.contains(o).then(|| (1u64 << self.n) as f64)
    }
}

/// Close a generating set under commutation.
///
/// `limit` caps the number of basis elements; the closure of a hardware-efficient ansatz is
/// `4^n - 1`, which is the point being made but not something a browser should discover by
/// exhausting memory.
pub fn closure(generators: &[PauliString], n: usize, limit: usize) -> Algebra {
    let mut basis: Vec<PauliString> = Vec::new();
    let mut index: HashMap<PauliString, usize> = HashMap::new();
    let mut parents: Vec<(u32, u32)> = Vec::new();
    for g in generators {
        if g.is_identity() {
            continue;
        }
        if let std::collections::hash_map::Entry::Vacant(e) = index.entry(*g) {
            let k = basis.len() as u32;
            e.insert(basis.len());
            basis.push(*g);
            parents.push((k, k));
        }
    }

    let mut growth = vec![basis.len()];
    let mut truncated = false;
    // Pairs (i, j) with i < j and j < done are all examined. New elements land past `done`
    // and are picked up by the next round.
    let mut done = 1usize;
    while done < basis.len() {
        let len = basis.len();
        'round: for j in done..len {
            for i in 0..j {
                if let Some((c, _)) = basis[i].commutator(&basis[j]) {
                    if let std::collections::hash_map::Entry::Vacant(e) = index.entry(c) {
                        e.insert(basis.len());
                        basis.push(c);
                        parents.push((i as u32, j as u32));
                        if basis.len() >= limit {
                            truncated = true;
                            break 'round;
                        }
                    }
                }
            }
        }
        done = len;
        growth.push(basis.len());
        if truncated {
            break;
        }
    }

    Algebra {
        n,
        basis,
        index,
        growth,
        parents,
        truncated,
    }
}

/// The closure with no practical limit. Only for tests and the CLI, where an exponential
/// algebra is a legitimate answer rather than a denial of service.
pub fn closure_unbounded(generators: &[PauliString], n: usize) -> Algebra {
    closure(generators, n, usize::MAX)
}

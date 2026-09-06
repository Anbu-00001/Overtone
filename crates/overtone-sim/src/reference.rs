//! A dense reference simulator, for differential testing only.
//!
//! This module shares **no code** with the strided kernels in [`crate::state`]. It builds
//! the full `2^n x 2^n` matrix of every gate by explicit Kronecker product and multiplies
//! it into a dense vector. That is exactly the thing the fast path exists to avoid, which
//! is what makes it a useful oracle: an indexing mistake in the pair loop, a transposed
//! matrix, or a wrong qubit-ordering convention shows up as a disagreement rather than as
//! two copies of the same bug.
//!
//! Cost is `O(4^n)` per gate. Keep `n <= 8` here. It is not for production use.

use crate::circuit::Circuit;
use crate::complex::{Mat2, C64};
use crate::gate::{h_matrix, rx_matrix, ry_matrix, rz_matrix, x_matrix, Gate};
use crate::observable::Observable;
use crate::state::StateVec;

/// A dense square matrix in row-major order.
#[derive(Clone, Debug)]
pub struct DenseMat {
    pub dim: usize,
    pub data: Vec<C64>,
}

impl DenseMat {
    pub fn identity(dim: usize) -> Self {
        let mut data = vec![C64::ZERO; dim * dim];
        for i in 0..dim {
            data[i * dim + i] = C64::ONE;
        }
        DenseMat { dim, data }
    }

    pub fn zeros(dim: usize) -> Self {
        DenseMat {
            dim,
            data: vec![C64::ZERO; dim * dim],
        }
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> C64 {
        self.data[r * self.dim + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: C64) {
        self.data[r * self.dim + c] = v;
    }

    /// Kronecker product `self (x) other`.
    pub fn kron(&self, other: &DenseMat) -> DenseMat {
        let dim = self.dim * other.dim;
        let mut out = DenseMat::zeros(dim);
        for ra in 0..self.dim {
            for ca in 0..self.dim {
                let a = self.get(ra, ca);
                if a == C64::ZERO {
                    continue;
                }
                for rb in 0..other.dim {
                    for cb in 0..other.dim {
                        let v = a * other.get(rb, cb);
                        out.set(ra * other.dim + rb, ca * other.dim + cb, v);
                    }
                }
            }
        }
        out
    }

    /// Matrix product `self * other`.
    pub fn matmul(&self, other: &DenseMat) -> DenseMat {
        assert_eq!(self.dim, other.dim);
        let mut out = DenseMat::zeros(self.dim);
        for r in 0..self.dim {
            for k in 0..self.dim {
                let a = self.get(r, k);
                if a == C64::ZERO {
                    continue;
                }
                for c in 0..self.dim {
                    let v = out.get(r, c) + a * other.get(k, c);
                    out.set(r, c, v);
                }
            }
        }
        out
    }

    /// Entrywise sum.
    pub fn add(&self, other: &DenseMat) -> DenseMat {
        assert_eq!(self.dim, other.dim);
        DenseMat {
            dim: self.dim,
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| *a + *b)
                .collect(),
        }
    }

    /// Apply to a dense state vector.
    pub fn apply(&self, v: &[C64]) -> Vec<C64> {
        assert_eq!(v.len(), self.dim);
        self.data
            .chunks(self.dim)
            .map(|row| {
                row.iter()
                    .zip(v)
                    .fold(C64::ZERO, |acc, (m, x)| acc + *m * *x)
            })
            .collect()
    }
}

fn mat2_dense(m: &Mat2) -> DenseMat {
    DenseMat {
        dim: 2,
        data: vec![m[0], m[1], m[2], m[3]],
    }
}

/// `|0><0|` (`which = 0`) or `|1><1|` (`which = 1`) as a 2x2 dense matrix.
fn projector(which: usize) -> DenseMat {
    let mut p = DenseMat::zeros(2);
    p.set(which, which, C64::ONE);
    p
}

/// Embed a 2x2 operator on qubit `q` of an `n`-qubit register.
///
/// With qubit `q` occupying bit `q` of the basis index (little-endian), the embedding is
/// `I_{2^(n-1-q)} (x) u (x) I_{2^q}`.
pub fn embed_1q(n: usize, q: usize, u: &DenseMat) -> DenseMat {
    assert!(q < n);
    let low = DenseMat::identity(1 << q);
    let high = DenseMat::identity(1 << (n - 1 - q));
    high.kron(u).kron(&low)
}

/// Embed a controlled 2x2 operator: `P0_c (x) I + P1_c (x) u_t`.
pub fn embed_controlled(n: usize, c: usize, t: usize, u: &DenseMat) -> DenseMat {
    assert_ne!(c, t);
    let p0 = embed_1q(n, c, &projector(0));
    let p1 = embed_1q(n, c, &projector(1));
    let ut = embed_1q(n, t, u);
    // The two embeddings act on different qubits and therefore commute.
    p0.add(&p1.matmul(&ut))
}

/// The full `2^n x 2^n` matrix of one gate.
pub fn gate_matrix(n: usize, gate: &Gate, params: &[f64]) -> DenseMat {
    match gate {
        Gate::Rx { q, angle } => embed_1q(n, *q, &mat2_dense(&rx_matrix(angle.value(params)))),
        Gate::Ry { q, angle } => embed_1q(n, *q, &mat2_dense(&ry_matrix(angle.value(params)))),
        Gate::Rz { q, angle } => embed_1q(n, *q, &mat2_dense(&rz_matrix(angle.value(params)))),
        Gate::H { q } => embed_1q(n, *q, &mat2_dense(&h_matrix())),
        Gate::U { q, m } => embed_1q(n, *q, &mat2_dense(m)),
        Gate::Cnot { control, target } => {
            embed_controlled(n, *control, *target, &mat2_dense(&x_matrix()))
        }
        Gate::Cz { a, b } => {
            // Diagonal: -1 exactly on the basis states where both qubits are set.
            let dim = 1usize << n;
            let mask = (1usize << a) | (1usize << b);
            let mut m = DenseMat::zeros(dim);
            for i in 0..dim {
                let v = if i & mask == mask { -1.0 } else { 1.0 };
                m.set(i, i, C64::new(v, 0.0));
            }
            m
        }
    }
}

/// Run a circuit densely, returning the amplitudes of the final state.
pub fn run(circuit: &Circuit, params: &[f64]) -> Vec<C64> {
    let n = circuit.num_qubits();
    let dim = 1usize << n;
    let mut v = vec![C64::ZERO; dim];
    v[0] = C64::ONE;
    for gate in circuit.gates() {
        v = gate_matrix(n, gate, params).apply(&v);
    }
    v
}

/// Run a circuit densely and return the result as a [`StateVec`] for comparison.
pub fn run_as_state(circuit: &Circuit, params: &[f64]) -> StateVec {
    let v = run(circuit, params);
    StateVec::from_amplitudes(
        v.iter().map(|c| c.re).collect(),
        v.iter().map(|c| c.im).collect(),
    )
}

/// The full matrix of an observable.
pub fn observable_matrix(n: usize, obs: &Observable) -> DenseMat {
    let dim = 1usize << n;
    let mut acc = DenseMat::zeros(dim);
    for term in &obs.terms {
        let mut m = DenseMat::identity(dim);
        for &(q, p) in &term.factors {
            m = m.matmul(&embed_1q(n, q, &mat2_dense(&p.matrix())));
        }
        for e in &mut m.data {
            *e = *e * term.coeff;
        }
        acc = acc.add(&m);
    }
    acc
}

/// `<psi|M|psi>` computed densely.
pub fn expectation(n: usize, obs: &Observable, v: &[C64]) -> f64 {
    let mv = observable_matrix(n, obs).apply(v);
    v.iter()
        .zip(&mv)
        .map(|(a, b)| a.conj() * *b)
        .fold(C64::ZERO, |acc, x| acc + x)
        .re
}

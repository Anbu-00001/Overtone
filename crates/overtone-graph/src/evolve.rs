//! The shared eigenbasis, and the one character that separates diffusion from interference.
//!
//! Part V 1.1: a classical random walk on a graph evolves under `e^(-Lt)` and a
//! continuous-time quantum walk under an imaginary exponent of the same operator. Both are
//! diagonal in the eigenbasis of the Laplacian, so one eigendecomposition serves both, and
//! the difference between them really is a single character.
//!
//! # One correction, and it matters for the panel
//!
//! Part V 1.3 writes the toggle as `e^(-Lt)` against `e^(-iAt)` — the Laplacian on one side
//! and the adjacency matrix on the other. Those are **different operators**, and the claim
//! that only one character changed is then not literally true. On a *regular* graph they
//! agree up to an unobservable global phase, because `L = dI - A` gives
//! `e^(-iLt) = e^(-idt) e^(iAt)`; on an irregular graph — which every maze is, since a
//! corridor end has degree one and a crossing has degree four — the degree term is a
//! position-dependent phase and the two evolutions differ in their probabilities, not just
//! in a phase.
//!
//! So this module evolves both under `L`. That keeps the panel's claim exact, and the
//! adjacency-matrix convention is offered separately for anyone comparing against the
//! continuous-time-quantum-walk literature, which usually writes `A`.

use crate::graph::Graph;

/// The eigendecomposition both walks are diagonal in.
#[derive(Clone, Debug)]
pub struct Eigenbasis {
    /// Ascending eigenvalues.
    pub values: Vec<f64>,
    /// Column `k` is eigenvector `k`; stored row-major, `vectors[i * n + k]`.
    pub vectors: Vec<f64>,
    pub n: usize,
}

impl Eigenbasis {
    /// Diagonalise the graph Laplacian.
    pub fn of_laplacian(graph: &Graph) -> Eigenbasis {
        Eigenbasis::of_matrix(&graph.laplacian(), graph.order())
    }

    /// Diagonalise the adjacency matrix, for comparison against the usual CTQW convention.
    pub fn of_adjacency(graph: &Graph) -> Eigenbasis {
        Eigenbasis::of_matrix(&graph.adjacency_matrix(), graph.order())
    }

    /// Diagonalise any symmetric matrix in this basis's layout.
    ///
    /// Public because the option-discovery module of Part V 6 needs the *normalized*
    /// Laplacian, which is a different diffusion model rather than a different graph, and
    /// building a second eigensolver for it would let the two drift apart.
    pub fn of_matrix(m: &[f64], n: usize) -> Eigenbasis {
        let mut a = m.to_vec();
        let mut vectors = vec![0.0; n * n];
        let values = overtone_mps::linalg::jacobi_eigh(&mut a, n, &mut vectors);
        // Ascending, so that mode zero is the slowest and the low-order modes are the ones a
        // proto-value function basis would keep.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| values[i].partial_cmp(&values[j]).unwrap());
        let sorted_values: Vec<f64> = order.iter().map(|&i| values[i]).collect();
        let mut sorted = vec![0.0; n * n];
        for (new, &old) in order.iter().enumerate() {
            for i in 0..n {
                sorted[i * n + new] = vectors[i * n + old];
            }
        }
        Eigenbasis {
            values: sorted_values,
            vectors: sorted,
            n,
        }
    }

    /// Eigenvector `k`, as a field over the vertices.
    pub fn mode(&self, k: usize) -> Vec<f64> {
        (0..self.n).map(|i| self.vectors[i * self.n + k]).collect()
    }

    /// Expand a vertex field in the eigenbasis.
    fn project(&self, field: &[f64]) -> Vec<f64> {
        (0..self.n)
            .map(|k| {
                (0..self.n)
                    .map(|i| self.vectors[i * self.n + k] * field[i])
                    .sum()
            })
            .collect()
    }

    /// `e^(-Lt) p0`: diffusion. The modes decay, fastest first.
    ///
    /// Total probability is conserved because the constant vector is the eigenvector of
    /// eigenvalue zero, so the mode carrying the total never decays.
    pub fn diffuse(&self, p0: &[f64], t: f64) -> Vec<f64> {
        let c = self.project(p0);
        let mut out = vec![0.0; self.n];
        for (k, &ck) in c.iter().enumerate() {
            let w = ck * (-self.values[k] * t).exp();
            if w == 0.0 {
                continue;
            }
            for (i, o) in out.iter_mut().enumerate() {
                *o += w * self.vectors[i * self.n + k];
            }
        }
        out
    }

    /// `e^(-iLt) psi0`: interference. The modes rotate rather than decay, so they come back
    /// into phase with each other and the amplitudes add and cancel.
    ///
    /// Returns `(re, im)`.
    pub fn interfere(&self, psi0: &[f64], t: f64) -> (Vec<f64>, Vec<f64>) {
        let c = self.project(psi0);
        let (mut re, mut im) = (vec![0.0; self.n], vec![0.0; self.n]);
        for (k, &ck) in c.iter().enumerate() {
            let (s, co) = (-self.values[k] * t).sin_cos();
            for i in 0..self.n {
                let v = ck * self.vectors[i * self.n + k];
                re[i] += v * co;
                im[i] += v * s;
            }
        }
        (re, im)
    }

    /// `|psi|^2` for the quantum walk, so the two evolutions can be compared as
    /// distributions over the same vertices.
    pub fn interference_probability(&self, psi0: &[f64], t: f64) -> Vec<f64> {
        let (re, im) = self.interfere(psi0, t);
        re.iter().zip(&im).map(|(a, b)| a * a + b * b).collect()
    }
}

/// Spread of a distribution over the graph, as the mean hop distance from the source.
///
/// The comparison Part V 1.3 is after: ballistic against diffusive, on one maze, with the
/// only difference being whether the exponent is real or imaginary.
///
/// The quantum field leads at short times and does **not** simply keep leading. It is
/// unitary on a finite graph, so it never settles: the modes come back into phase, the field
/// returns toward where it started, and the comparison inverts. On a 101-vertex maze the
/// quantum mean distance passes the classical one around `t = 4` and is behind it again by
/// `t = 8`. Diffusion, having only decaying modes, spreads monotonically to uniform. Saying
/// "the quantum walk spreads faster" without a time attached is therefore false half the
/// time.
pub fn mean_distance(distances: &[Option<usize>], p: &[f64]) -> f64 {
    let mut total = 0.0;
    let mut mass = 0.0;
    for (i, w) in p.iter().enumerate() {
        if let Some(d) = distances[i] {
            total += w * d as f64;
            mass += w;
        }
    }
    if mass <= 0.0 {
        0.0
    } else {
        total / mass
    }
}

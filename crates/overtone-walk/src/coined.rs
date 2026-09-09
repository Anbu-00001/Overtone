//! One coined walk, any graph, any degree — the generalised core Decisions 03 Q7.4 asks for.
//!
//! ```text
//! U = S C
//! C_u = 2|phi_u><phi_u| - I     Grover reflection at u
//! |phi_u> = (1/sqrt(d_u)) sum_{v ~ u} |u,v>
//! S|u,v> = |v,u>                flip-flop shift
//! ```
//!
//! Decisions 03 Q7.4 is explicit that this is **one** engine and not three: hypercubes, grids
//! and welded trees differ in the graph handed to it, not in the walk. The Szegedy module the
//! plan used to carry was deleted on the same ruling — with a Grover coin and a flip-flop
//! shift, two coined steps are one Szegedy step (Wong 2016; Portugal & Segawa 2017), so there
//! was never a second engine to write.
//!
//! # Two things this module does not let you get away with
//!
//! **Degree is per vertex.** The welded tree's roots have degree 2 and everything else has
//! degree 3, and Li, Li and Luo use a genuinely two-dimensional coin there. Padding the roots
//! to three ports gives a walk that runs, looks plausible, and is not the published one.
//!
//! **The coin is not always Grover.** [`VertexCoin::Dft`] is here because Krovi and Brun's
//! infinite hitting times are a *DFT-coin* phenomenon on the hypercube, not a Grover-coin one
//! — see [`crate::hitting`]. Part VI-A T2's cage has a concrete instance because of it.

use overtone_sim::C64;

/// The coin at a vertex, as a `d_u`-dimensional unitary on its ports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VertexCoin {
    /// `2|phi_u><phi_u| - I`, the reflection about the uniform neighbour superposition.
    ///
    /// Permutation-symmetric, so the order of a vertex's neighbours cannot change the walk.
    Grover,
    /// The discrete Fourier transform `F_{jk} = omega^{jk} / sqrt(d)`.
    ///
    /// Not permutation-symmetric: this coin *does* read the neighbour order, which is part of
    /// why it behaves so differently from Grover on the same graph.
    Dft,
}

/// A coined walk over the arcs of a graph.
///
/// State lives on **arcs**, not vertices: `arc(u, i)` is the directed edge from `u` to its
/// `i`-th neighbour, and the flip-flop shift is the involution that reverses it. On a regular
/// graph this is the same object as position-tensor-coin; on a graph of mixed degree it is the
/// only one of the two that still makes sense, which is why it is the representation here.
#[derive(Clone, Debug)]
pub struct Coined {
    adjacency: Vec<Vec<usize>>,
    /// Index of `arc(u, 0)`; `start[order]` is the arc count.
    start: Vec<usize>,
    /// `rev[a]` is the reversed arc. An involution.
    rev: Vec<usize>,
    coin: VertexCoin,
    amps: Vec<C64>,
    scratch: Vec<C64>,
}

impl Coined {
    /// Build the walk over a neighbour list.
    ///
    /// Panics if an edge is listed at one endpoint and not the other — a half-edge would make
    /// the shift non-unitary, and the failure would otherwise show up much later as amplitude
    /// quietly leaking out of the graph.
    pub fn new(adjacency: &[Vec<usize>], coin: VertexCoin) -> Self {
        let order = adjacency.len();
        let mut start = Vec::with_capacity(order + 1);
        let mut n = 0;
        for nbrs in adjacency {
            start.push(n);
            n += nbrs.len();
        }
        start.push(n);

        let mut rev = vec![usize::MAX; n];
        for (u, nbrs) in adjacency.iter().enumerate() {
            for (i, &v) in nbrs.iter().enumerate() {
                let j = adjacency[v]
                    .iter()
                    .position(|&w| w == u)
                    .unwrap_or_else(|| panic!("edge {u}-{v} is listed at {u} but not at {v}"));
                rev[start[u] + i] = start[v] + j;
            }
        }

        Coined {
            adjacency: adjacency.to_vec(),
            start,
            rev,
            coin,
            amps: vec![C64::ZERO; n],
            scratch: vec![C64::ZERO; n],
        }
    }

    /// Number of vertices.
    pub fn order(&self) -> usize {
        self.adjacency.len()
    }

    /// Number of arcs — twice the edge count, and the dimension of the state.
    pub fn arcs(&self) -> usize {
        self.amps.len()
    }

    /// Degree of `u`.
    pub fn degree(&self, u: usize) -> usize {
        self.adjacency[u].len()
    }

    /// Set the state to `|u, phi(u)>`: amplitude spread evenly over the arcs leaving `u`.
    pub fn set_uniform(&mut self, u: usize) {
        self.amps.iter_mut().for_each(|a| *a = C64::ZERO);
        let d = self.degree(u);
        let r = 1.0 / (d as f64).sqrt();
        for k in self.start[u]..self.start[u + 1] {
            self.amps[k] = C64::new(r, 0.0);
        }
    }

    /// Read the amplitude on one arc, for tests that need more than a probability.
    pub fn arc_amplitude(&self, u: usize, i: usize) -> C64 {
        self.amps[self.start[u] + i]
    }

    /// `<u, phi(u) | psi>`, the overlap with the uniform state at `u`.
    ///
    /// This is the quantity Li, Li and Luo's Theorem 4.1 is about: the target is
    /// `|t, phi(t)>`, not the bare vertex `t`, and on a degree-2 root the two differ.
    pub fn uniform_overlap(&self, u: usize) -> C64 {
        let r = 1.0 / (self.degree(u) as f64).sqrt();
        let mut s = C64::ZERO;
        for k in self.start[u]..self.start[u + 1] {
            s = s + self.amps[k];
        }
        C64::new(s.re * r, s.im * r)
    }

    /// Total probability on the arcs leaving `u` — the chance of finding the walker there.
    pub fn vertex_probability(&self, u: usize) -> f64 {
        (self.start[u]..self.start[u + 1])
            .map(|k| self.amps[k].norm_sqr())
            .sum()
    }

    /// Total probability still in the walk. One under a unitary step; less after a projection.
    pub fn norm(&self) -> f64 {
        self.amps.iter().map(|a| a.norm_sqr()).sum()
    }

    /// One step, `U = S C`.
    pub fn step(&mut self) {
        // Coin, block by block.
        for u in 0..self.adjacency.len() {
            let (lo, hi) = (self.start[u], self.start[u + 1]);
            let d = hi - lo;
            match self.coin {
                VertexCoin::Grover => {
                    let mut s = C64::ZERO;
                    for k in lo..hi {
                        s = s + self.amps[k];
                    }
                    let s = C64::new(2.0 * s.re / d as f64, 2.0 * s.im / d as f64);
                    for k in lo..hi {
                        self.scratch[k] = s - self.amps[k];
                    }
                }
                VertexCoin::Dft => {
                    let r = 1.0 / (d as f64).sqrt();
                    let w = 2.0 * std::f64::consts::PI / d as f64;
                    for j in 0..d {
                        let mut s = C64::ZERO;
                        for k in 0..d {
                            s = s + C64::expi(w * (j * k % d) as f64) * self.amps[lo + k];
                        }
                        self.scratch[lo + j] = C64::new(s.re * r, s.im * r);
                    }
                }
            }
        }
        // Flip-flop shift: amplitude on u->v becomes amplitude on v->u.
        for a in 0..self.amps.len() {
            self.amps[a] = self.scratch[self.rev[a]];
        }
    }

    /// Measure "is the walker at `u`". Returns the probability found there and **removes** it.
    ///
    /// The state is deliberately left un-normalised: the residual norm after many steps is the
    /// probability the walker has still not arrived, and re-normalising would destroy exactly
    /// the quantity [`crate::hitting::concurrent`] exists to measure.
    pub fn project_out(&mut self, u: usize) -> f64 {
        let mut p = 0.0;
        for k in self.start[u]..self.start[u + 1] {
            p += self.amps[k].norm_sqr();
            self.amps[k] = C64::ZERO;
        }
        p
    }

    /// Worst deviation from unitarity over one step, measured on the basis states.
    ///
    /// A walk that has stopped being unitary is the single easiest way to publish a hitting
    /// time that is not real, so it is cheap to check and it is checked in the tests.
    pub fn unitarity_error(&self) -> f64 {
        let n = self.arcs();
        let mut worst: f64 = 0.0;
        let mut probe = self.clone();
        let mut columns = Vec::with_capacity(n);
        for a in 0..n {
            probe.amps.iter_mut().for_each(|x| *x = C64::ZERO);
            probe.amps[a] = C64::ONE;
            probe.step();
            columns.push(probe.amps.clone());
        }
        for a in 0..n {
            for b in a..n {
                let mut s = C64::ZERO;
                for (ca, cb) in columns[a].iter().zip(columns[b].iter()) {
                    s = s + ca.conj() * *cb;
                }
                let want = if a == b { C64::ONE } else { C64::ZERO };
                worst = worst.max((s - want).re.hypot((s - want).im));
            }
        }
        worst
    }
}

//! Conserved quantities of the reachable orbit, and why they are the right certificate.
//!
//! # The setting
//!
//! Part VII 3 defines checkmate as a controllability condition. The formal basis is
//! d'Alessandro: for a driftless bilinear control system whose controls generate the
//! dynamical Lie algebra `g`, the reachable set from `psi` is exactly the orbit
//! `{ U psi : U in exp(g) }`. If `g = su(2^n)` the orbit is the whole sphere and nothing is
//! ever unreachable; if `g` is a proper subalgebra the orbit is a submanifold and states off
//! it are unreachable *in principle*.
//!
//! # Deciding orbit membership is not the same problem as computing dim(g)
//!
//! "Decidable from `g`, computable in milliseconds" is true at the ends and not obviously
//! true in between. Computing `dim(g)` is a bitset closure. Deciding whether one specific
//! state lies in the orbit of another under a proper subgroup is a different and harder
//! question, and this module does not pretend to answer it exactly.
//!
//! What it does instead has a theorem behind it. `exp(g)` is a **compact** connected group,
//! and for a compact group acting on a real vector space the ring of invariant polynomials
//! **separates orbits** -- and is finitely generated, by Hilbert and Nagata. So a complete
//! certificate exists in principle, as a finite list of polynomial invariants. This module
//! computes the two lowest-degree families:
//!
//! - **Degree one in `rho`: the commutant.** Every Pauli string `Q` that commutes with all
//!   of `g` satisfies `U† Q U = Q` for `U in exp(g)`, so `<Q>` is *exactly* conserved. This
//!   is the superselection sector, and it is the bishop of Part VII 2 made computable: a
//!   bishop cannot change square colour because a `Z2` invariant forbids it.
//! - **Degree two in `rho`: the `g`-purity of each simple ideal.** The adjoint action of
//!   `exp(g)` preserves each ideal and is orthogonal there in the trace inner product, so
//!   `P_j = sum_{alpha in g_j} <b_alpha>^2` is conserved. This is Ragone et al.'s `g`-purity,
//!   already computed elsewhere in this workspace for a different purpose.
//!
//! Truncating a separating family at degree two leaves a certificate that is **sound and
//! not complete**: differing invariants *prove* unreachability, matching invariants prove
//! nothing. For a win condition that is the correct direction -- a game must never declare a
//! checkmate that is not one. The predicate may miss checkmates, which makes games longer
//! rather than wrong, and the size of that gap is measured in `examples/checkmate.rs` rather
//! than assumed small.

use overtone_lie::{closure::Algebra, PauliString};
use overtone_sim::{Observable, PauliTerm, StateVec};

/// A basis of the commutant of `g` within the Pauli group, as Pauli strings.
///
/// A Pauli string `Q` commutes with `P` exactly when the symplectic form
/// `x_P . z_Q + z_P . x_Q` vanishes mod 2, so the commutant is the null space of a
/// `dim(g) x 2n` matrix over GF(2). Gaussian elimination there costs `O(dim(g) n^2)` and
/// returns the sector labels directly, which is far cheaper than testing `4^n` strings.
///
/// The identity is excluded: it commutes with everything and carries no information.
pub fn commutant_basis(algebra: &Algebra) -> Vec<PauliString> {
    let n = algebra.num_qubits();
    // Row for basis element P, in coordinates (x_Q | z_Q): the pairing with Q is
    // z_P . x_Q + x_P . z_Q, so the row is (z_P | x_P).
    let mut rows: Vec<u128> = algebra
        .basis()
        .iter()
        .map(|p| (p.z & mask(n)) | ((p.x & mask(n)) << n))
        .filter(|r| *r != 0)
        .collect();

    // Row-reduce over GF(2), recording which column is each row's pivot.
    let width = 2 * n;
    let mut pivot_of_row: Vec<usize> = Vec::new();
    let mut reduced: Vec<u128> = Vec::new();
    for col in 0..width {
        let bit = 1u128 << col;
        let Some(idx) = rows.iter().position(|r| r & bit != 0) else {
            continue;
        };
        let pivot_row = rows.swap_remove(idx);
        for r in rows.iter_mut() {
            if *r & bit != 0 {
                *r ^= pivot_row;
            }
        }
        for r in reduced.iter_mut() {
            if *r & bit != 0 {
                *r ^= pivot_row;
            }
        }
        reduced.push(pivot_row);
        pivot_of_row.push(col);
    }

    let pivots: std::collections::HashSet<usize> = pivot_of_row.iter().copied().collect();
    let free: Vec<usize> = (0..width).filter(|c| !pivots.contains(c)).collect();

    // One null-space vector per free column: set it to one, back-substitute the pivots.
    let mut out = Vec::new();
    for &f in &free {
        let mut v = 1u128 << f;
        for (row, &p) in reduced.iter().zip(&pivot_of_row) {
            if row & (1u128 << f) != 0 {
                v |= 1u128 << p;
            }
        }
        // v is in (x_Q | z_Q) coordinates.
        let q = PauliString::new(v & mask(n), (v >> n) & mask(n));
        if !q.is_identity() {
            out.push(q);
        }
    }
    out
}

fn mask(n: usize) -> u128 {
    if n >= 128 {
        u128::MAX
    } else {
        (1u128 << n) - 1
    }
}

/// The conserved quantities of one state under `exp(g)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Invariants {
    /// `<Q>` for each commutant generator, in the order [`commutant_basis`] returns them.
    /// Exactly conserved, and linear in the state.
    pub sector: Vec<f64>,
    /// `g`-purity of each simple ideal, in the order [`overtone_lie::predict::components`]
    /// returns them. Conserved because the adjoint action is orthogonal on each ideal.
    pub purity: Vec<f64>,
}

impl Invariants {
    /// Largest absolute disagreement between two invariant vectors.
    ///
    /// `f64::INFINITY` when the two came from different algebras, which is a programming
    /// error rather than a game state and is reported as maximally different rather than
    /// silently compared elementwise.
    pub fn distance(&self, other: &Invariants) -> f64 {
        if self.sector.len() != other.sector.len() || self.purity.len() != other.purity.len() {
            return f64::INFINITY;
        }
        let s = self
            .sector
            .iter()
            .zip(&other.sector)
            .map(|(a, b)| (a - b).abs());
        let p = self
            .purity
            .iter()
            .zip(&other.purity)
            .map(|(a, b)| (a - b).abs());
        s.chain(p).fold(0.0, f64::max)
    }
}

/// The invariants a state carries under `exp(g)`.
///
/// Precompute the commutant and the ideal decomposition with [`OrbitCertificate::new`] when
/// they are needed more than once; both are properties of the algebra alone and neither
/// depends on the state.
#[derive(Clone, Debug)]
pub struct OrbitCertificate {
    commutant: Vec<PauliString>,
    ideals: Vec<Vec<PauliString>>,
    num_qubits: usize,
    full: bool,
}

impl OrbitCertificate {
    pub fn new(algebra: &Algebra) -> OrbitCertificate {
        let basis = algebra.basis();
        let ideals = overtone_lie::predict::components(algebra)
            .into_iter()
            .map(|c| c.into_iter().map(|i| basis[i]).collect())
            .collect();
        let n = algebra.num_qubits();
        let full = algebra.dim() == 4usize.saturating_pow(n as u32).saturating_sub(1);
        OrbitCertificate {
            commutant: commutant_basis(algebra),
            ideals,
            num_qubits: n,
            full,
        }
    }

    /// True when `g = su(2^n)`: the group is transitive on the sphere, so nothing is ever
    /// unreachable and no certificate can ever fire. Exact, and the one case Part VII 3
    /// calls out by name.
    pub fn fully_controllable(&self) -> bool {
        self.full
    }

    pub fn commutant(&self) -> &[PauliString] {
        &self.commutant
    }

    pub fn ideals(&self) -> &[Vec<PauliString>] {
        &self.ideals
    }

    pub fn invariants(&self, psi: &StateVec) -> Invariants {
        let sector = self.commutant.iter().map(|q| expectation(q, psi)).collect();
        let scale = 1.0 / (1u64 << self.num_qubits) as f64;
        let purity = self
            .ideals
            .iter()
            .map(|ideal| {
                ideal
                    .iter()
                    .map(|p| {
                        let e = expectation(p, psi);
                        e * e
                    })
                    .sum::<f64>()
                    * scale
            })
            .collect();
        Invariants { sector, purity }
    }

    /// **Sound**: `true` proves `target` is not in the orbit of `source`. `false` proves
    /// nothing at all.
    ///
    /// `tolerance` is absolute and applies to quantities that are `O(1)` by construction --
    /// a Pauli expectation lies in `[-1, 1]` and a `g`-purity in `[0, 1]` -- so an absolute
    /// tolerance is at the scale of the quantity here, unlike the desirability of an LMDP.
    pub fn certainly_unreachable(
        &self,
        source: &StateVec,
        target: &StateVec,
        tolerance: f64,
    ) -> bool {
        if self.full {
            return false;
        }
        self.invariants(source).distance(&self.invariants(target)) > tolerance
    }
}

/// `<psi|P|psi>` for a Pauli string.
///
/// Routed through [`overtone_sim::Observable`] rather than hand-rolled. A Pauli string's
/// expectation looks like a one-line loop until the `Y` factors are counted, and the phase
/// `i^|x & z|` is precisely the kind of sign that hides comfortably in a loop that "obviously
/// works". The simulator's path is already checked against a dense Kronecker reference, so
/// this borrows that verification instead of asking for its own.
pub fn expectation(p: &PauliString, psi: &StateVec) -> f64 {
    let factors = p.factors(psi.num_qubits());
    if factors.is_empty() {
        return 1.0;
    }
    Observable::new(vec![PauliTerm::new(1.0, factors)]).expectation(psi)
}

/// The real dimension of the orbit through `psi`.
///
/// The orbit is the image of `exp(g)`, so its tangent space at `psi` is spanned by
/// `{ i P_alpha psi }` for `P_alpha` in the basis of `g`, taken modulo the global phase
/// direction `i psi` which changes no physical state. The dimension is the rank of the Gram
/// matrix of those vectors after projecting that direction out.
///
/// This is what makes the incompleteness of a low-degree certificate a *counting* argument
/// rather than an empirical observation. See [`separation_deficit`].
pub fn orbit_dimension(algebra: &Algebra, psi: &StateVec, tolerance: f64) -> usize {
    let dim = psi.dim();
    let basis = algebra.basis();
    // Each tangent vector as a real vector of length 2 * dim.
    let mut vectors: Vec<Vec<f64>> = Vec::with_capacity(basis.len() + 1);
    let phase: Vec<f64> = (0..dim)
        .flat_map(|i| {
            let a = psi.amp(i);
            [-a.im, a.re]
        })
        .collect();
    for p in basis {
        let factors = p.factors(psi.num_qubits());
        if factors.is_empty() {
            continue;
        }
        let applied = Observable::new(vec![PauliTerm::new(1.0, factors)]).apply(psi);
        let v: Vec<f64> = (0..dim)
            .flat_map(|i| {
                let a = applied.amp(i);
                // i * (P psi)
                [-a.im, a.re]
            })
            .collect();
        vectors.push(v);
    }
    // Project out the global phase, then take the rank by Gram-Schmidt.
    let np = norm(&phase);
    if np > 1e-12 {
        for v in vectors.iter_mut() {
            let d = dot(v, &phase) / (np * np);
            for (a, b) in v.iter_mut().zip(&phase) {
                *a -= d * b;
            }
        }
    }
    rank(&mut vectors, tolerance)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn rank(vectors: &mut [Vec<f64>], tolerance: f64) -> usize {
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for v in vectors.iter() {
        let mut w = v.clone();
        for b in &basis {
            let d = dot(&w, b);
            for (a, c) in w.iter_mut().zip(b) {
                *a -= d * c;
            }
        }
        let n = norm(&w);
        if n > tolerance {
            for a in w.iter_mut() {
                *a /= n;
            }
            basis.push(w);
        }
    }
    basis.len()
}

/// How many dimensions of the invariant level set the orbit fails to fill.
///
/// The pure states of `n` qubits form a real manifold of dimension `2 * 2^n - 2` once the
/// global phase and the norm are removed. Fixing the certificate's `k` invariants cuts that
/// to at least `2 * 2^n - 2 - k`. The orbit inside it has dimension at most `dim(g)`. So when
///
/// ```text
///     2 * 2^n - 2 - k  >  orbit dimension
/// ```
///
/// the level set of the invariants contains a **continuum** of distinct orbits, and no
/// certificate built from those `k` invariants can separate them. The certificate is then
/// incomplete as a matter of counting, before any experiment is run.
///
/// This is why measuring completeness on two independent Haar-random states is not a test:
/// two such states disagree on essentially every invariant, so the certificate fires
/// trivially and reports a completeness of one on a family that is provably not complete.
/// A real measurement has to construct pairs that *share* the invariants -- see
/// `examples/checkmate.rs`.
pub fn separation_deficit(
    algebra: &Algebra,
    certificate: &OrbitCertificate,
    psi: &StateVec,
) -> isize {
    let ambient = 2 * (psi.dim() as isize) - 2;
    let invariants = (certificate.commutant().len() + certificate.ideals().len()) as isize;
    let orbit = orbit_dimension(algebra, psi, 1e-9) as isize;
    ambient - invariants - orbit
}

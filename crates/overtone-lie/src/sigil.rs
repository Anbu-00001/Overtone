//! The sigil: an agent's mark, computed from its dynamical Lie algebra.
//!
//! Part IV 2.1. This is **not an identicon**. An identicon hashes a name into a shape, and
//! two different objects that happen to hash alike look alike for no reason. Every
//! coordinate below is a function of the algebra's own structure, so two agents with the
//! same mark have the same mark *because they are the same agent*, and an algebra that
//! explodes toward `4^n` is visibly denser than one held down by a symmetry.
//!
//! Part IV 7: if the mark ever stops being a deterministic function of the algebra, delete
//! it. A meaningless glyph is worse than no glyph.
//!
//! The encoding, and what each choice reads as:
//!
//! - **Radius is the Pauli weight.** Weight-one strings sit on the inner ring, weight-`n`
//!   on the outer. An algebra of local operators is a tight core; one that reaches every
//!   string fills the disc.
//! - **Angle is the support's centre of mass** on the register, so strings acting on the
//!   same qubits point the same way. A translation-invariant algebra therefore comes out
//!   evenly spaced, and a symmetry-broken one does not.
//! - **Segment colour is the X, Y, Z content** of the string, as three fractions. The
//!   Ising-like algebras come out cold; the ones that reach all of `su(2^n)` come out grey,
//!   because they contain everything in equal measure.
//! - **Ring count is the largest weight present**, which is the algebra's reach across the
//!   register in one integer.

use crate::closure::Algebra;
use crate::predict::components;
use overtone_sim::Pauli;

/// One basis element, placed.
#[derive(Clone, Copy, Debug)]
pub struct Spoke {
    /// Radians, from the support's centre of mass on the register.
    pub angle: f64,
    /// `0` at the centre, `1` at the rim: the Pauli weight over the largest weight present.
    pub radius: f64,
    /// Fractions of the string's non-identity factors that are X, Y and Z. They sum to one.
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// An agent's mark.
#[derive(Clone, Debug)]
pub struct Sigil {
    pub spokes: Vec<Spoke>,
    /// Largest Pauli weight in the algebra: the number of rings to draw.
    pub rings: usize,
    pub dim: usize,
    pub num_qubits: usize,
    /// Blocks that commute elementwise — the sigil's rotational symmetry order.
    pub blocks: usize,
}

impl Sigil {
    /// Compute the mark. Deterministic in the algebra alone: the spokes are sorted into a
    /// canonical order so that the sequence in which the closure happened to discover its
    /// elements cannot change the picture.
    pub fn of(algebra: &Algebra) -> Sigil {
        let n = algebra.num_qubits();
        let mut basis: Vec<_> = algebra.basis().to_vec();
        basis.sort_by_key(|p| (p.weight(), p.x, p.z));
        let max_weight = basis.iter().map(|p| p.weight()).max().unwrap_or(1).max(1);

        let spokes = basis
            .iter()
            .map(|p| {
                let factors = p.factors(n);
                let support: Vec<usize> = factors.iter().map(|(q, _)| *q).collect();
                let centre = if support.is_empty() {
                    0.0
                } else {
                    support.iter().sum::<usize>() as f64 / support.len() as f64
                };
                let counts = factors.iter().fold([0.0f64; 3], |mut acc, (_, pl)| {
                    match pl {
                        Pauli::X => acc[0] += 1.0,
                        Pauli::Y => acc[1] += 1.0,
                        Pauli::Z => acc[2] += 1.0,
                        Pauli::I => {}
                    }
                    acc
                });
                let total = (counts[0] + counts[1] + counts[2]).max(1.0);
                Spoke {
                    angle: std::f64::consts::TAU * centre / n.max(1) as f64,
                    radius: p.weight() as f64 / max_weight as f64,
                    x: counts[0] / total,
                    y: counts[1] / total,
                    z: counts[2] / total,
                }
            })
            .collect();

        Sigil {
            spokes,
            rings: max_weight as usize,
            dim: algebra.dim(),
            num_qubits: n,
            blocks: components(algebra).len(),
        }
    }

    /// Flattened as `[angle, radius, x, y, z]` per spoke, for the browser boundary.
    pub fn to_flat(&self) -> Vec<f64> {
        self.spokes
            .iter()
            .flat_map(|s| [s.angle, s.radius, s.x, s.y, s.z])
            .collect()
    }
}

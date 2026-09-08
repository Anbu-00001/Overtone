//! The Chinese Rings, and the hypercube Part III has been carrying without walking on it.
//!
//! Part IX 3 is the counterexample that reframes the project, so this module derives it
//! rather than asserting it. The Chinese Rings (baguenaudier, jiu lian huan) is a
//! disentanglement puzzle whose legal moves are:
//!
//! ```text
//! ring 0     may always be flipped;
//! ring k + 1 may be flipped iff ring k is on and rings 0 .. k are all off.
//! ```
//!
//! From those two lines alone, [`rings_graph`] builds the state graph and `tests/rings.rs`
//! *checks* the three claims Part IX makes about it: it is a path on `2^n` vertices, the Gray
//! code `G(i) = i XOR (i >> 1)` orders it, and the distance from all-on to all-off is
//! OEIS A000975. Nothing here takes the puzzle's difficulty on trust either -- the branching
//! factor is measured, and it is 2.
//!
//! > Difficulty does not come from the number of options. It comes from the difficulty of
//! > knowing which option is forward.
//!
//! # A correction to Part IX 3
//!
//! The spec calls the puzzle "roughly two thousand years old" and "older than algebra". That
//! is folklore. The attribution to the 2nd/3rd century general Zhuge Liang traces to Stewart
//! Culin relying on an unnamed informant; the earliest definitive references are Yang Shen's
//! *Sheng an ji* (early 16th century) and Pacioli's *De Viribus Quantitatis* (1509), with
//! Cardano's *De subtilitate* (1550) giving the puzzle its other name. Both of those postdate
//! al-Khwarizmi by seven centuries. The argument does not need the date -- a puzzle people
//! have been failing to solve since 1509 makes the point -- so [`PROVENANCE`] carries the
//! documented version and the docs use it.
//!
//! # The hypercube (Part IX 3.1)
//!
//! The rings state space is the `n`-bit hypercube traversed by a Gray code: each step flips
//! exactly one bit. Part III represents Pauli strings as bitsets under XOR, which is the same
//! object. [`pauli_hypercube`] builds it: cells are Pauli strings on `q` qubits, legal moves
//! flip one generator, and a maze whose geometry is the algebra itself falls out.

use crate::Graph;

/// What is actually documented about the puzzle's age, as against what is repeated about it.
pub const PROVENANCE: &str = "\
Earliest definitive references: Yang Shen, Sheng an ji (early 16th c.); Luca Pacioli, \
De Viribus Quantitatis (1509); Girolamo Cardano, De subtilitate (1550), whence \"Cardan's \
rings\". The 2nd/3rd century attribution to Zhuge Liang comes from Stewart Culin via an \
unnamed informant and is not documented.";

/// The reflected binary Gray code: `G(i) = i XOR (i >> 1)`.
pub fn gray(i: usize) -> usize {
    i ^ (i >> 1)
}

/// The inverse Gray code: the index `i` for which `gray(i) == g`.
pub fn ungray(g: usize) -> usize {
    let mut i = g;
    let mut shift = 1;
    while (g >> shift) != 0 {
        i ^= g >> shift;
        shift += 1;
    }
    i
}

/// Is flipping ring `k` legal in state `s`? Bit `k` set means ring `k` is on.
///
/// Ring 0 is always free; ring `k + 1` needs ring `k` on and every ring below it off.
pub fn is_legal(s: usize, k: usize) -> bool {
    if k == 0 {
        return true;
    }
    let below = s & ((1 << (k - 1)) - 1);
    (s >> (k - 1)) & 1 == 1 && below == 0
}

/// Every state reachable from `s` in one legal move.
pub fn moves(s: usize, n: usize) -> Vec<usize> {
    (0..n)
        .filter(|&k| is_legal(s, k))
        .map(|k| s ^ (1 << k))
        .collect()
}

/// The state graph of the `n`-ring puzzle, built from [`moves`] alone.
pub fn rings_graph(n: usize) -> Graph {
    let size = 1usize << n;
    let mut edges = Vec::new();
    for s in 0..size {
        for t in moves(s, n) {
            if s < t {
                edges.push((s, t));
            }
        }
    }
    Graph::from_edges(size, &edges)
}

/// The minimum number of moves to take all `n` rings off, by the closed form:
/// `(2^(n+1) - 2)/3` for even `n`, `(2^(n+1) - 1)/3` for odd. This is OEIS A000975.
pub fn solution_length(n: usize) -> u64 {
    let p = 1u64 << (n + 1);
    if n % 2 == 0 {
        (p - 2) / 3
    } else {
        (p - 1) / 3
    }
}

/// The first `n` terms of A000975, from the closed form.
pub fn a000975(n: usize) -> Vec<u64> {
    (1..=n).map(solution_length).collect()
}

/// The `2q`-bit hypercube of Pauli strings on `q` qubits: vertices are symplectic bitsets,
/// edges flip one generator. Part IX 3.1's substrate, and the same object as the rings state
/// space with the path restriction lifted.
///
/// Order is `4^q`, so `q` past 8 or so is not a maze anyone will walk.
pub fn pauli_hypercube(q: usize) -> Graph {
    let bits = 2 * q;
    let size = 1usize << bits;
    let mut edges = Vec::new();
    for s in 0..size {
        for k in 0..bits {
            let t = s ^ (1 << k);
            if s < t {
                edges.push((s, t));
            }
        }
    }
    Graph::from_edges(size, &edges)
}

/// A Gray-code Hamiltonian path through a `bits`-dimensional hypercube: consecutive entries
/// differ in exactly one bit, and every vertex appears once.
pub fn gray_path(bits: usize) -> Vec<usize> {
    (0..(1usize << bits)).map(gray).collect()
}

/// The average degree of a graph -- the branching factor, measured rather than claimed.
pub fn branching_factor(g: &Graph) -> f64 {
    if g.order() == 0 {
        return 0.0;
    }
    (0..g.order()).map(|i| g.degree(i) as f64).sum::<f64>() / g.order() as f64
}

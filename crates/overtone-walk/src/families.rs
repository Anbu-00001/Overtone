//! The graph families Phase 5 walks on, and where their degree stops being uniform.
//!
//! Part II 6 wants glued trees; Decisions 03 Q7.4 reorders that so the *hypercube* comes
//! first, because Kempe's exponential hitting-time gap has been peer-reviewed since 2005 and
//! the welded-tree result is from 2024. Validate the engine against the settled result before
//! taking on the recent one.
//!
//! Both families are built here as plain neighbour lists rather than as
//! `overtone_graph::Graph`, which keeps this crate a leaf of the dependency graph. A `Graph`
//! converts in one line: `(0..g.order()).map(|i| g.neighbours(i).to_vec()).collect()`.
//!
//! # The trap the welded tree sets
//!
//! Decisions 03 Q7.2 flags it in advance and it is worth repeating where the graph is built:
//! **the welded tree is 3-regular except at the two roots, which have degree 2.** The paper
//! handles that with a two-dimensional coin at the roots, not by padding them to three ports.
//! [`Family::degrees`] reports the distribution so a walk can assert on it, and
//! [`crate::coined::Coined`] takes the degree per vertex rather than a global one.

use rand::{Rng, SeedableRng};

/// A named graph with a source and a target vertex, as a neighbour list.
#[derive(Clone, Debug)]
pub struct Family {
    /// `adjacency[u]` is the neighbours of `u`, in the order the coin sees them.
    pub adjacency: Vec<Vec<usize>>,
    /// Where the walk starts.
    pub source: usize,
    /// The vertex a hitting time is measured to.
    pub target: usize,
    /// For error messages and panels.
    pub name: String,
}

impl Family {
    /// Number of vertices.
    pub fn order(&self) -> usize {
        self.adjacency.len()
    }

    /// Number of undirected edges. Each is counted once.
    pub fn edges(&self) -> usize {
        self.adjacency.iter().map(|a| a.len()).sum::<usize>() / 2
    }

    /// Degrees of the source and the target. `(2, 2)` on the welded tree, `(n, n)` on the cube.
    pub fn degree_of_source_and_target(&self) -> (usize, usize) {
        (
            self.adjacency[self.source].len(),
            self.adjacency[self.target].len(),
        )
    }

    /// Degrees, ascending, with multiplicity — the honest form of "is it regular".
    pub fn degrees(&self) -> Vec<(usize, usize)> {
        let mut counts: Vec<(usize, usize)> = Vec::new();
        for a in &self.adjacency {
            match counts.iter_mut().find(|(d, _)| *d == a.len()) {
                Some((_, c)) => *c += 1,
                None => counts.push((a.len(), 1)),
            }
        }
        counts.sort_unstable();
        counts
    }

    /// Every edge appears in both endpoints' lists, no self-loops, no repeats.
    ///
    /// A welded tree built from a bad cycle is still a graph — it just is not *the* graph, and
    /// the reduced model would then disagree with the full walk for a reason that has nothing
    /// to do with the reduction. Check the graph before blaming the walk.
    pub fn is_consistent(&self) -> bool {
        for (u, nbrs) in self.adjacency.iter().enumerate() {
            for (i, &v) in nbrs.iter().enumerate() {
                if v == u || v >= self.adjacency.len() {
                    return false;
                }
                if nbrs[..i].contains(&v) {
                    return false;
                }
                if !self.adjacency[v].contains(&u) {
                    return false;
                }
            }
        }
        true
    }
}

/// The `n`-dimensional hypercube: `2^n` vertices, vertex `u` adjacent to `u XOR (1 << i)`.
///
/// Source is `0`, target is the antipode `2^n - 1` — the corner-to-corner problem Kempe
/// solves. The neighbour order is bit order, which matters only for coins that are not
/// permutation-symmetric (the DFT coin is one; see [`crate::coined::Face`]).
pub fn hypercube(n: usize) -> Family {
    assert!(
        (1..=20).contains(&n),
        "hypercube dimension {n} out of range"
    );
    let order = 1usize << n;
    let adjacency = (0..order)
        .map(|u| (0..n).map(|i| u ^ (1 << i)).collect())
        .collect();
    Family {
        adjacency,
        source: 0,
        target: order - 1,
        name: format!("hypercube n={n}"),
    }
}

/// The welded tree `G_n`: two depth-`n` binary trees, leaves joined by one random cycle.
///
/// `2^(n+2) - 2` vertices. The left root is the source, the right root the target, and both
/// have **degree 2** — every other vertex has degree 3.
///
/// The cycle alternates left leaf, right leaf, left leaf, ... so each leaf is entered twice
/// and the welding is a single cycle of length `2^(n+1)` by construction rather than by
/// rejection sampling. Both leaf orders are shuffled from `seed`; Li, Li and Luo's Lemma 3.1
/// holds *regardless* of the naming and of which cycle came out, which is exactly what
/// `tests/hitting.rs` checks by varying the seed.
pub fn welded_tree(n: usize, seed: u64) -> Family {
    assert!((1..=10).contains(&n), "welded tree depth {n} out of range");
    let per_tree = (1usize << (n + 1)) - 1;
    let order = 2 * per_tree;
    let mut adjacency = vec![Vec::new(); order];

    // Two heap-ordered trees. Level k, position j is at (2^k - 1) + j, the right tree
    // offset by a whole tree.
    fn link(adjacency: &mut [Vec<usize>], a: usize, b: usize) {
        adjacency[a].push(b);
        adjacency[b].push(a);
    }
    for base in [0usize, per_tree] {
        for k in 0..n {
            for j in 0..(1usize << k) {
                let parent = base + (1 << k) - 1 + j;
                let left = base + (1 << (k + 1)) - 1 + 2 * j;
                link(&mut adjacency, parent, left);
                link(&mut adjacency, parent, left + 1);
            }
        }
    }

    let leaves = 1usize << n;
    let first_leaf = (1usize << n) - 1;
    let mut left: Vec<usize> = (0..leaves).map(|j| first_leaf + j).collect();
    let mut right: Vec<usize> = (0..leaves).map(|j| per_tree + first_leaf + j).collect();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    shuffle(&mut left, &mut rng);
    shuffle(&mut right, &mut rng);

    // left[0] - right[0] - left[1] - right[1] - ... - right[m-1] - left[0].
    for i in 0..leaves {
        link(&mut adjacency, left[i], right[i]);
        link(&mut adjacency, right[i], left[(i + 1) % leaves]);
    }

    Family {
        adjacency,
        source: 0,
        target: per_tree,
        name: format!("welded tree n={n} seed={seed}"),
    }
}

fn shuffle<R: Rng>(v: &mut [usize], rng: &mut R) {
    for i in (1..v.len()).rev() {
        v.swap(i, rng.gen_range(0..=i));
    }
}

/// Expected steps for the *classical* walk to go corner to corner on the `n`-cube.
///
/// The classical chain lumps to Hamming weight, and the resulting birth-death chain is
/// tridiagonal, so this is an exact solve rather than a simulation, on
/// `h[x] = 1 + (x/n) h[x-1] + ((n-x)/n) h[x+1]` with `h[n] = 0`. It is the `2^n` the quantum
/// walk is measured against, and it belongs next to the graph rather than in a comment.
pub fn classical_hitting(n: usize) -> f64 {
    assert!(n >= 1);
    let nf = n as f64;
    // Thomas algorithm on rows x = 0 .. n-1 of  -a_x h[x-1] + h[x] - c_x h[x+1] = 1.
    let mut c = vec![0.0f64; n];
    let mut d = vec![0.0f64; n];
    for x in 0..n {
        let a = x as f64 / nf;
        let cx = (nf - x as f64) / nf;
        let denom = 1.0 - a * if x == 0 { 0.0 } else { c[x - 1] };
        c[x] = if x + 1 < n { cx / denom } else { 0.0 };
        d[x] = (1.0 + a * if x == 0 { 0.0 } else { d[x - 1] }) / denom;
    }
    let mut h = vec![0.0f64; n];
    for x in (0..n).rev() {
        h[x] = d[x] + c[x] * if x + 1 < n { h[x + 1] } else { 0.0 };
    }
    h[0]
}

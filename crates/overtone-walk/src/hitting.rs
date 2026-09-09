//! Three things the literature calls "hitting time", and the one Part II conflated.
//!
//! Decisions 03 Q7.3 is a correction to Part II 6, and it is worth stating where the code
//! lives rather than only in the phase notes:
//!
//! | notion | definition | where it holds |
//! |---|---|---|
//! | one-shot | `\|<v\|U^T\|u>\|^2 >= p`, no measurement during the walk | welded trees, hypercube |
//! | concurrent | measure every step, first detection time | Krovi and Brun's setting |
//! | search | uniform start, marked subset, modified coin | grids, Johnson graphs |
//!
//! **Marked-vertex search is not entrance-to-exit traversal.** Part II 6 quotes Szegedy's
//! square-root-of-hitting-time result and then applies it to maze traversal; those are
//! different problems and the square root does not carry across. Only the first two notions
//! are implemented here, because only they are what Phase 5's two families are about.
//!
//! # The cage, made concrete
//!
//! Part VI-A T2 warns about configurations where interference means the walker never arrives.
//! Krovi and Brun exhibit one on the hypercube: with the **DFT coin**, the evolution operator
//! is so degenerate that eigenvectors exist with no amplitude at the target at all, and an
//! initial state overlapping them has a strictly positive probability of never being found —
//! an infinite concurrent hitting time, which no classical walk on a connected graph can have.
//!
//! The coin is the part that matters and the part that is easy to drop: **the same graph, the
//! same initial state, and the Grover coin instead, has a polynomial hitting time.** The cage
//! is a property of the walk, not of the maze. [`residual`] measures it without diagonalising
//! anything — run the measured walk and watch what is left.

use crate::coined::Coined;

/// The unmeasured walk's probability of being at `target` after each of `steps` steps.
///
/// This is one-shot hitting: nothing is measured along the way, so the number at step `t` is
/// what a walk stopped at exactly `t` would give, and the series does **not** sum to anything.
pub fn one_shot(walk: &mut Coined, target: usize, steps: usize) -> Vec<f64> {
    (0..steps)
        .map(|_| {
            walk.step();
            walk.vertex_probability(target)
        })
        .collect()
}

/// One-shot hitting measured against `|target, phi(target)>` rather than the bare vertex.
///
/// Li, Li and Luo's Theorem 4.1 is about this quantity: the target state is the *uniform
/// superposition of the target's arcs*, not the target vertex. On the welded tree's degree-2
/// roots the two differ, and the reduced model tracks this one — so this is the function a
/// reduced-versus-full comparison has to use.
pub fn one_shot_uniform(walk: &mut Coined, target: usize, steps: usize) -> Vec<f64> {
    (0..steps)
        .map(|_| {
            walk.step();
            walk.uniform_overlap(target).norm_sqr()
        })
        .collect()
}

/// What a measured walk finds, step by step.
#[derive(Clone, Debug)]
pub struct Concurrent {
    /// Probability of the *first* detection landing on each step.
    pub detected: Vec<f64>,
    /// Total probability of having arrived by the last step.
    pub total: f64,
    /// Probability still in the walk. Zero for a walk that always arrives.
    pub residual: f64,
}

/// Run the measured walk: step, look at `target`, remove what was found, repeat.
///
/// The state is never renormalised. That is the whole point — the leftover norm is the
/// probability of not having arrived, and it is the thing an infinite hitting time is made of.
pub fn concurrent(walk: &mut Coined, target: usize, steps: usize) -> Concurrent {
    let mut detected = Vec::with_capacity(steps);
    for _ in 0..steps {
        walk.step();
        detected.push(walk.project_out(target));
    }
    let total = detected.iter().sum();
    Concurrent {
        residual: walk.norm(),
        detected,
        total,
    }
}

/// The probability that the measured walk has still not arrived after `steps` steps.
///
/// Shorthand for `concurrent(..).residual`, because that is the number the cage is about.
pub fn residual(walk: &mut Coined, target: usize, steps: usize) -> f64 {
    concurrent(walk, target, steps).residual
}

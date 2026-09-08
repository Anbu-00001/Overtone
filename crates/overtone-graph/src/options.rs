//! Eigenoptions (Part V 6, M26): options built from the same Laplacian spectrum M22 computes.
//!
//! Machado, Bellemare and Bowling, *A Laplacian Framework for Option Discovery in
//! Reinforcement Learning* (ICML 2017), definitions 3.1 and 3.2, taken literally:
//!
//! - **Eigenpurpose.** An intrinsic reward from a proto-value function `e`,
//!   `r_e(s, s') = e^T(phi(s') - phi(s))`, which in the tabular case is `e[s'] - e[s]`.
//! - **Eigenbehavior.** The policy optimal for that reward,
//!   `chi_e(s) = argmax_a q*_e(s, a)`, obtained here by value iteration.
//! - **Termination.** The action set is augmented with a `terminate` action of value zero,
//!   so the option terminates wherever `q(s, a) <= 0` for every real action, and its
//!   initiation set is every state where some action has `q(s, a) > 0`.
//! - Both signs of each eigenvector are used, so `k` eigenvectors give `2k` options.
//!
//! # The paper uses two different Laplacians, and on a maze they disagree
//!
//! Section 2.3 says PVFs are "commonly defined by the combinatorial graph Laplacian
//! `L = D - A`" and then: "Different diffusion models can be used to generate PVFs, such as
//! the normalized graph Laplacian `L = D^(-1/2)(D - A)D^(-1/2)`, **which we use in this
//! paper**." Section 5's sample-based algorithm instead recovers the *combinatorial* one:
//! Theorem 5.1 shows the incidence matrix satisfies `T^T T = 2(D - W)`.
//!
//! On a regular graph that is a distinction without a difference -- `D = dI` makes the two
//! proportional, with identical eigenvectors. A maze is never regular: a dead end has degree
//! one and a crossing four. So the two sections of one paper generate **different options**
//! on the graph this project actually has, and neither is wrong; they are different
//! diffusion models. Measured in `examples/eigenoptions.rs` and asserted in `tests/`:
//! the two option sets differ on a WFC maze and agree exactly on a cycle.
//!
//! [`Laplacian`] therefore makes the choice explicit at every call site. A default here
//! would silently pick one of the paper's two answers.

use crate::evolve::Eigenbasis;
use crate::graph::Graph;

/// Which diffusion model the proto-value functions come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Laplacian {
    /// `L = D - A`. What Machado et al.'s section 5 recovers, and what M22 already computes.
    Combinatorial,
    /// `L = D^(-1/2)(D - A) D^(-1/2)`. What Machado et al. say they use in section 2.3.
    Normalized,
}

impl Laplacian {
    pub fn matrix(&self, g: &Graph) -> Vec<f64> {
        let n = g.order();
        let mut l = g.laplacian();
        if *self == Laplacian::Normalized {
            let d: Vec<f64> = (0..n)
                .map(|i| {
                    let deg = g.degree(i) as f64;
                    if deg > 0.0 {
                        1.0 / deg.sqrt()
                    } else {
                        0.0
                    }
                })
                .collect();
            for i in 0..n {
                for j in 0..n {
                    l[i * n + j] *= d[i] * d[j];
                }
            }
        }
        l
    }

    pub fn eigenbasis(&self, g: &Graph) -> Eigenbasis {
        Eigenbasis::of_matrix(&self.matrix(g), g.order())
    }
}

/// One option: where it may start, what it does, and where it stops.
#[derive(Clone, Debug)]
pub struct Eigenoption {
    /// Index of the proto-value function it came from.
    pub mode: usize,
    /// `true` when built from `-e` rather than `+e`.
    pub negated: bool,
    /// `q*(s, a)` for the best real action at each state.
    pub q_best: Vec<f64>,
    /// The chosen neighbour at each state, or `None` where the option terminates.
    pub policy: Vec<Option<usize>>,
}

impl Eigenoption {
    /// States where some action has positive value -- Machado et al.'s initiation set.
    pub fn initiation(&self) -> Vec<usize> {
        (0..self.policy.len())
            .filter(|&i| self.policy[i].is_some())
            .collect()
    }

    /// States where every action has value at most zero. Theorem 3.1 guarantees this is
    /// nonempty for `gamma < 1` on a finite graph, which `tests/` checks rather than trusts.
    pub fn termination(&self) -> Vec<usize> {
        (0..self.policy.len())
            .filter(|&i| self.policy[i].is_none())
            .collect()
    }

    /// Follow the option from `start` until it terminates or `limit` steps pass.
    pub fn rollout(&self, start: usize, limit: usize) -> Vec<usize> {
        let mut path = vec![start];
        let mut at = start;
        for _ in 0..limit {
            match self.policy[at] {
                Some(next) => {
                    at = next;
                    path.push(at);
                }
                None => break,
            }
        }
        path
    }
}

/// Build the eigenoption for one signed proto-value function.
///
/// Value iteration on `M_e = <S, A + {terminate}, r_e, p, gamma>`. The transitions are
/// deterministic -- an agent on a maze graph choosing a neighbour arrives there -- so
/// `q(s, a) = e[a] - e[s] + gamma v(a)` and the sweep is exact rather than sampled.
pub fn eigenoption(
    graph: &Graph,
    basis: &Eigenbasis,
    mode: usize,
    negated: bool,
    gamma: f64,
    sweeps: usize,
) -> Eigenoption {
    let n = graph.order();
    let sign = if negated { -1.0 } else { 1.0 };
    let e: Vec<f64> = basis.mode(mode).iter().map(|v| sign * v).collect();

    // The terminate action has value exactly zero, so "is this action worth taking" is a
    // comparison against zero -- and a comparison against an exact zero needs a tolerance at
    // the scale of the quantity being compared. The eigenvector entries are of order
    // `1/sqrt(n)`, so anything this far below them is the eigensolver's residual, not a
    // reward. Without it the *constant* mode, whose eigenpurpose is identically zero, gets an
    // initiation set of 94 states on a 101-vertex maze built entirely out of 1e-17.
    let scale = e.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let tol = 1e-9 * scale.max(f64::MIN_POSITIVE);

    // v(s) = max(0, max_a [ e[a] - e[s] + gamma v(a) ]); the zero is the terminate action.
    let mut v = vec![0.0; n];
    for _ in 0..sweeps {
        let mut delta: f64 = 0.0;
        for s in 0..n {
            let best = graph
                .neighbours(s)
                .iter()
                .map(|&a| e[a] - e[s] + gamma * v[a])
                .fold(f64::NEG_INFINITY, f64::max);
            let updated = if best > tol { best } else { 0.0 };
            delta = delta.max((updated - v[s]).abs());
            v[s] = updated;
        }
        if delta < 1e-12 {
            break;
        }
    }

    let mut q_best = vec![0.0; n];
    let mut policy = vec![None; n];
    for s in 0..n {
        let mut best = f64::NEG_INFINITY;
        let mut arg = None;
        for &a in graph.neighbours(s) {
            let q = e[a] - e[s] + gamma * v[a];
            if q > best {
                best = q;
                arg = Some(a);
            }
        }
        q_best[s] = if best.is_finite() { best } else { 0.0 };
        // q(s, terminate) = 0, so a real action is only taken when it beats zero.
        policy[s] = if best > tol { arg } else { None };
    }

    Eigenoption {
        mode,
        negated,
        q_best,
        policy,
    }
}

/// The first `count` proto-value functions, both signs, as `2 * count` options.
///
/// Mode zero is skipped: for the combinatorial Laplacian it is constant, so its eigenpurpose
/// is identically zero and the option terminates everywhere. Keeping it would put a
/// guaranteed no-op in the option set and quietly halve the useful ones.
pub fn eigenoptions(graph: &Graph, which: Laplacian, count: usize, gamma: f64) -> Vec<Eigenoption> {
    let basis = which.eigenbasis(graph);
    let mut out = Vec::new();
    for mode in 1..=count.min(graph.order().saturating_sub(1)) {
        for negated in [false, true] {
            out.push(eigenoption(graph, &basis, mode, negated, gamma, 10_000));
        }
    }
    out
}

/// How far two option sets disagree: the fraction of states at which the chosen action
/// differs, maximised over the pairing of modes.
///
/// Compared mode by mode and sign by sign, because the two diffusion models order their
/// spectra the same way even when the vectors differ.
pub fn policy_disagreement(a: &[Eigenoption], b: &[Eigenoption]) -> f64 {
    assert_eq!(a.len(), b.len(), "option sets must be the same size");
    let mut worst: f64 = 0.0;
    for (x, y) in a.iter().zip(b) {
        let n = x.policy.len();
        let differing = (0..n).filter(|&i| x.policy[i] != y.policy[i]).count();
        worst = worst.max(differing as f64 / n as f64);
    }
    worst
}

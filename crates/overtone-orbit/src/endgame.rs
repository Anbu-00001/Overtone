//! M36: the endgame tablebase, which is an eigenvector.
//!
//! Part VII 4 gets chess's phase structure free from Axiom II's arrow. Openings are coherent
//! and classically hard; middlegames are partially decohered and searched; the endgame is
//! **fully decohered**, which makes it a classical MDP on the maze graph -- and Part V 2's
//! linearly-solvable MDP turns that into a largest-eigenvalue problem.
//!
//! > Chess endgame tablebases cost decades of compute and terabytes. This one is a
//! > Perron-Frobenius eigenvector computed live, the moment coherence runs out.
//!
//! # The transition is physical, not conventional
//!
//! Coherence only decreases (rule 5). When it reaches zero the amplitude field has no phase
//! information left to interfere with, so the position *is* a distribution over cells and the
//! optimal policy is the LMDP's. Nothing declares the endgame: it arrives.
//!
//! # What is reused rather than rebuilt
//!
//! The solver is `overtone_graph::Lmdp`, unchanged, including the correction that made it
//! work: solve in **log space**, never in `z`. On a maze of any interesting diameter the
//! desirability spans `e^(-rho * D)`, so a tolerance on `z` is a tolerance at the wrong scale
//! and the answer comes back confidently wrong. That is Phase 8's finding and it is why M36
//! was built after M23 rather than beside it.

use overtone_graph::{Graph, Lmdp};

use crate::game::Game;

/// The exact endgame solution: a value per cell, and the optimal step from each.
#[derive(Clone, Debug)]
pub struct Tablebase {
    /// `v(i)`, the exact cost-to-go in hops. Solved in log space.
    pub value: Vec<f64>,
    /// The optimal successor cell from each cell, or `None` at a goal or an isolated cell.
    pub policy: Vec<Option<usize>>,
    pub rho: f64,
}

impl Tablebase {
    /// Solve the decohered position exactly.
    ///
    /// # `recommended_rho` is the wrong function here, and using it was a real bug
    ///
    /// [`Lmdp::recommended_rho`] exists to keep `exp(-rho * D)` above the `f64` floor, which
    /// is a constraint of the **`z`-space** solve. It clamps `rho` to at most 60. Phase 8's
    /// whole finding was that iterating the value function with a stable log-sum-exp removes
    /// that floor, so in log space the clamp buys nothing and costs accuracy: the error falls
    /// as `1/rho`, and at `rho = 60` on a diameter-36 maze it is still around `0.5` -- enough
    /// to round a true distance of 14 to 15, which is exactly what the first version of this
    /// function did.
    ///
    /// Reusing a safety cap across the representation change silently reintroduced the very
    /// bug that the representation change had removed. So `rho` is chosen here for accuracy
    /// alone. The error accumulated along a path of `D` hops is about `D ln(deg) / rho`, so
    /// `rho = 40 D` leaves it near `0.03` at any degree a maze produces -- an order of
    /// magnitude inside the half-hop that rounding needs.
    pub fn solve(graph: &Graph, goals: &[usize]) -> Tablebase {
        let rho = (40.0 * graph.diameter() as f64).max(100.0);
        let value = Lmdp::new(graph, goals.to_vec(), rho).shortest_paths_stable(20_000, 1e-12);
        let policy = (0..graph.order())
            .map(|i| {
                graph
                    .neighbours(i)
                    .iter()
                    .copied()
                    .min_by(|&a, &b| value[a].partial_cmp(&value[b]).unwrap())
                    .filter(|&best| value[best] < value[i] - 1e-9)
            })
            .collect();
        Tablebase { value, policy, rho }
    }

    /// Follow the tablebase from `start` until a goal or `limit` steps.
    pub fn line(&self, start: usize, limit: usize) -> Vec<usize> {
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

/// Whether the position has reached the endgame: rule 5's arrow has run out.
pub fn is_endgame(game: &Game) -> bool {
    game.players.iter().all(|p| p.coherence == 0)
}

/// Brute-force optimal play on the same graph, by breadth-first search.
///
/// The oracle the eigensolve is checked against. BFS on an unweighted graph is exact rather
/// than approximate, which is the whole reason it is the oracle and not a second estimate.
pub fn brute_force(graph: &Graph, goals: &[usize]) -> Vec<Option<usize>> {
    graph.bfs_distances(goals)
}
